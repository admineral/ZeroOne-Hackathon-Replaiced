import { useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, LossPoint, RunInfo } from "../api";
import { useSSE } from "../hooks";
import { JobStatusBadge } from "./JobStatusBadge";

interface Props {
  run: RunInfo;
  /** Bump to (re)start the live stream, e.g. right after submitting a job. */
  liveKey: number;
  live: boolean;
  /** When set, show this archived run (read-only) instead of live data. */
  inspectJobId?: string;
  /** Fresh run just submitted: clear and wait for the new run's data. */
  awaitFresh?: boolean;
  /** Job id backing the live / latest-snapshot view (for the "showing" label). */
  liveJobId?: string;
  /** Called when the streamed run reaches a terminal state. */
  onDone?: () => void;
}

type MetricKey = "loss" | "perplexity" | "accuracy";

interface MetricConfig {
  key: MetricKey;
  label: string;
  trainKey: string;
  valKey: string;
  domain: [number | "auto", number | "auto"];
  format: (v: number) => string;
}

const METRICS: Record<MetricKey, MetricConfig> = {
  loss: {
    key: "loss",
    label: "Loss",
    trainKey: "train_loss",
    valKey: "val_loss",
    domain: ["auto", "auto"],
    format: (v) => v.toFixed(4),
  },
  perplexity: {
    key: "perplexity",
    label: "Perplexity",
    trainKey: "train_ppl",
    valKey: "val_ppl",
    domain: ["auto", "auto"],
    format: (v) => v.toFixed(2),
  },
  accuracy: {
    key: "accuracy",
    label: "Accuracy",
    trainKey: "train_acc",
    valKey: "val_acc",
    domain: [0, 1],
    format: (v) => `${(v * 100).toFixed(1)}%`,
  },
};

export function LossChart({
  run,
  liveKey,
  live,
  inspectJobId,
  awaitFresh,
  liveJobId,
  onDone,
}: Props) {
  const [points, setPoints] = useState<LossPoint[]>([]);
  const [state, setState] = useState<string | null>(null);
  const [metric, setMetric] = useState<MetricKey>("loss");
  const [archivedHit, setArchivedHit] = useState(false);
  const seen = useRef<Map<number, LossPoint>>(new Map());

  const mergePoint = (point: LossPoint) => {
    seen.current.set(point.epoch, point);
    setPoints(
      Array.from(seen.current.values()).sort((a, b) => a.epoch - b.epoch)
    );
  };

  // Initial snapshot so the chart isn't empty before the stream catches up.
  // When inspecting a past job, this loads that job's archived metrics instead.
  // For a freshly-submitted run we skip the snapshot so the previous run's
  // curve doesn't flash — we wait for the new run's first epoch.
  useEffect(() => {
    let cancelled = false;
    seen.current = new Map();
    setPoints([]);
    setArchivedHit(false);
    if (awaitFresh && !inspectJobId) {
      return () => {
        cancelled = true;
      };
    }
    // Nothing to show (e.g. history was just cleared): stay blank instead of
    // re-loading the last run's curve from the canonical log on disk.
    if (!inspectJobId && !live && !liveJobId) {
      return () => {
        cancelled = true;
      };
    }
    api
      .getLossSnapshot(run.key, inspectJobId)
      .then((res) => {
        if (cancelled) return;
        for (const p of res.rows) seen.current.set(p.epoch, p);
        setPoints(res.rows);
        setArchivedHit(!!res.archived);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [run.key, liveKey, inspectJobId, awaitFresh, live, liveJobId]);

  const resetPoints = () => {
    seen.current = new Map();
    setPoints([]);
  };

  // No live stream while inspecting an archived run. On a fresh run we tail
  // from connect time so the previous run's rows aren't replayed.
  const streamUrl =
    live && !inspectJobId
      ? `/api/loss/stream?run=${run.key}${awaitFresh ? "&tail=1" : ""}`
      : null;
  const status = useSSE(
    streamUrl,
    {
      epoch: (data) => mergePoint(data as LossPoint),
      reset: () => resetPoints(),
      status: (data) => setState((data as { state: string | null }).state),
      done: (data) => {
        setState((data as { state: string | null }).state);
        onDone?.();
      },
    },
    liveKey
  );

  const hasAccuracy = points.some((p) => p.val_acc != null);
  const activeMetric = metric === "accuracy" && !hasAccuracy ? "loss" : metric;
  const cfg = METRICS[activeMetric];

  const chartData = useMemo<LossPoint[]>(
    () =>
      points.map((p) => ({
        ...p,
        train_ppl: p.train_loss != null ? Math.exp(p.train_loss) : undefined,
        val_ppl: p.val_loss != null ? Math.exp(p.val_loss) : undefined,
      })),
    [points]
  );

  const last = chartData[chartData.length - 1];
  const lastTrain = last?.[cfg.trainKey];
  const lastVal = last?.[cfg.valKey];
  const lastSec = last?.sec;

  // A human-readable "what am I looking at" descriptor for the header.
  let viewKind: "inspect" | "live" | "snapshot";
  let viewLabel: string;
  if (inspectJobId) {
    viewKind = "inspect";
    viewLabel = archivedHit
      ? `inspecting archived run · job ${inspectJobId}`
      : `inspecting job ${inspectJobId} · latest snapshot`;
  } else if (live) {
    viewKind = "live";
    viewLabel = awaitFresh
      ? liveJobId
        ? `live · new run · job ${liveJobId}`
        : "live · waiting for new run"
      : liveJobId
        ? `live · job ${liveJobId}`
        : "live";
  } else {
    viewKind = "snapshot";
    viewLabel = liveJobId
      ? `latest run · job ${liveJobId}`
      : "latest run snapshot";
  }

  return (
    <div className="card">
      <div className="card-head">
        <h3>
          Training · {run.label}
          <span className={`view-tag ${viewKind}`} style={{ marginLeft: 8 }}>
            {viewLabel}
          </span>
        </h3>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          {live && !inspectJobId && status === "open" && (
            <span className="conn">
              <span className="spinner" /> live
            </span>
          )}
          <div className="log-tabs">
            {(Object.keys(METRICS) as MetricKey[]).map((m) => (
              <button
                key={m}
                className={`log-tab ${activeMetric === m ? "active" : ""}`}
                disabled={m === "accuracy" && !hasAccuracy}
                onClick={() => setMetric(m)}
              >
                {METRICS[m].label}
              </button>
            ))}
          </div>
          <JobStatusBadge state={state} />
        </div>
      </div>

      {points.length === 0 ? (
        <div className="empty">
          {awaitFresh
            ? "Waiting for the new run's first epoch…"
            : "No data yet. Start a training run to watch metrics stream in live."}
        </div>
      ) : (
        <>
          <div className="stat-row" style={{ marginBottom: 14 }}>
            <div className="stat">
              <div className="label">Epoch</div>
              <div className="value accent">{last?.epoch ?? "-"}</div>
            </div>
            <div className="stat">
              <div className="label">Train {cfg.label.toLowerCase()}</div>
              <div className="value">
                {lastTrain != null ? cfg.format(lastTrain as number) : "-"}
              </div>
            </div>
            <div className="stat">
              <div className="label">Val {cfg.label.toLowerCase()}</div>
              <div className="value good">
                {lastVal != null ? cfg.format(lastVal as number) : "-"}
              </div>
            </div>
            <div className="stat">
              <div className="label">Epoch time</div>
              <div className="value">{lastSec != null ? `${lastSec.toFixed(1)}s` : "-"}</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: -8 }}>
              <CartesianGrid stroke="#1e2636" strokeDasharray="3 3" />
              <XAxis
                dataKey="epoch"
                stroke="#5f6b80"
                tick={{ fontSize: 11 }}
                tickLine={false}
              />
              <YAxis
                stroke="#5f6b80"
                tick={{ fontSize: 11 }}
                tickLine={false}
                width={52}
                domain={cfg.domain}
                tickFormatter={(v: number) =>
                  activeMetric === "accuracy" ? `${Math.round(v * 100)}%` : `${v}`
                }
              />
              <Tooltip
                contentStyle={{
                  background: "#121723",
                  border: "1px solid #232b3d",
                  borderRadius: 10,
                  fontSize: 12,
                }}
                labelStyle={{ color: "#97a3b8" }}
                formatter={(value: number, name: string) => [cfg.format(value), name]}
              />
              <Line
                type="monotone"
                dataKey={cfg.trainKey}
                name="train"
                stroke="#5b8cff"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey={cfg.valKey}
                name="val"
                stroke="#36d399"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}
