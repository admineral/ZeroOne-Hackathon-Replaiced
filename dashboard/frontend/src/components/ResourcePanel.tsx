import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, GpuTimelinePoint, GpuTimelineSummary, JobRecord } from "../api";

const TERMINAL = new Set([
  "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
  "NODE_FAIL", "DEADLINE", "BOOT_FAIL", "PREEMPTED", "REVOKED",
]);
function isActiveStatus(status?: string): boolean {
  if (!status) return false;
  return !TERMINAL.has(status.split(" ")[0].toUpperCase());
}

function fmtMB(mb: number | null | undefined): string {
  if (mb == null) return "-";
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
  return `${mb.toFixed(0)} MB`;
}

function fmtTime(epochSeconds?: number | null): string {
  if (!epochSeconds) return "-";
  const d = new Date(epochSeconds * 1000);
  if (Number.isNaN(d.getTime())) return "-";
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

interface Props {
  /** The job whose resources to show (selected in history, or latest run). */
  job?: JobRecord;
  /** Whether `job` is the user-selected one (vs. an auto-picked latest). */
  selected?: boolean;
}

export function ResourcePanel({ job, selected }: Props) {
  const res = job?.resources;
  const ts = res?.train_stats;
  const pct = ts?.gpu_peak_pct ?? null;
  const tl = res?.gpu_timeline;
  const ds = job?.dataset;
  const params = job?.params ?? {};

  const [timeline, setTimeline] = useState<GpuTimelinePoint[]>([]);
  const [liveSummary, setLiveSummary] = useState<GpuTimelineSummary | null>(null);

  const jobId = job?.job_id ?? null;
  const runKey = job?.run_key ?? null;
  const active = isActiveStatus(job?.status);
  useEffect(() => {
    let cancelled = false;
    setTimeline([]);
    setLiveSummary(null);
    // Only transformer training writes a GPU sample log.
    if (!jobId || runKey !== "transformer") return;
    const load = () =>
      api
        .getGpuTimeline(runKey, jobId)
        .then((d) => {
          if (cancelled) return;
          setTimeline(d.rows ?? []);
          setLiveSummary(d.summary ?? null);
        })
        .catch(() => {
          if (!cancelled) setTimeline([]);
        });
    load();
    // While the job is still running, keep refreshing the live GPU samples.
    const timer = active ? window.setInterval(load, 5000) : undefined;
    return () => {
      cancelled = true;
      if (timer) window.clearInterval(timer);
    };
  }, [jobId, runKey, active]);

  const hasFamilies = ds?.families && Object.keys(ds.families).length > 0;
  // Prefer the post-run sacct-merged summary; fall back to the live one.
  const gpuSummary = tl ?? liveSummary ?? undefined;

  const err = job?.error;
  const failed =
    !!job?.status &&
    ["FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "DEADLINE", "BOOT_FAIL", "PREEMPTED", "REVOKED"].includes(
      job.status.split(" ")[0].toUpperCase()
    );

  return (
    <>
      {/* Module 1 — live GPU utilisation & memory chart */}
      <div className="card">
        <div className="card-head">
          <h3>GPU utilisation &amp; memory</h3>
          <span className="hint">
            {job?.job_id
              ? `${selected ? "" : "latest · "}job ${job.job_id}`
              : "no run selected"}
          </span>
        </div>
        {timeline.length > 1 ? (
          <>
            <div className="hint" style={{ marginBottom: 6 }}>
              {active ? "live" : "over the run"}
              {gpuSummary?.samples ? ` · ${gpuSummary.samples} samples (~2s)` : ""}
            </div>
            {gpuSummary && (
              <div className="hint" style={{ marginBottom: 6 }}>
                util avg {gpuSummary.avg_util ?? "-"}% / peak {gpuSummary.max_util ?? "-"}% · power
                avg {gpuSummary.avg_power_w ?? "-"} W · mem peak {gpuSummary.max_mem_gb ?? "-"} GB
              </div>
            )}
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={timeline} margin={{ top: 4, right: 8, left: -8, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3344" />
                <XAxis
                  dataKey="t"
                  tick={{ fontSize: 10, fill: "#7a8699" }}
                  tickFormatter={(t) => `${t}s`}
                />
                <YAxis
                  yAxisId="pct"
                  domain={[0, 100]}
                  tick={{ fontSize: 10, fill: "#7a8699" }}
                  width={32}
                />
                <YAxis
                  yAxisId="gb"
                  orientation="right"
                  tick={{ fontSize: 10, fill: "#7a8699" }}
                  width={32}
                />
                <Tooltip
                  contentStyle={{
                    background: "#141a24",
                    border: "1px solid #2a3344",
                    fontSize: 12,
                  }}
                />
                <Line
                  yAxisId="pct"
                  type="monotone"
                  dataKey="util"
                  name="GPU util %"
                  stroke="#4ea1ff"
                  dot={false}
                  strokeWidth={2}
                />
                <Line
                  yAxisId="gb"
                  type="monotone"
                  dataKey="mem_gb"
                  name="Mem GB"
                  stroke="#ffb454"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </>
        ) : (
          <div className="empty">
            {!job || !job.job_id
              ? "Select a run in the history to see its GPU timeline."
              : active
                ? "Waiting for GPU samples… transformer runs log nvidia-smi every ~2s."
                : job.run_key === "transformer"
                  ? "No GPU samples were recorded for this run."
                  : "GPU sampling is only recorded for transformer training runs."}
          </div>
        )}
      </div>

      {/* Module 2 — Slurm accounting & run config */}
      <div className="card">
        <div className="card-head">
          <h3>Run resources</h3>
          <span className="hint">
            {job?.job_id
              ? `${selected ? "" : "latest · "}job ${job.job_id}`
              : "no run selected"}
          </span>
        </div>

        {!job || !job.job_id ? (
          <div className="empty">Click a job in the history to see its resource usage.</div>
        ) : (
          <>
          {failed && (
            <div className="err-banner">
              <div className="err-detail-head">
                <strong>
                  Run failed · {(err?.state || job.status).split(" ")[0].toUpperCase()}
                </strong>
                {err?.exit_code && <span className="hint">exit code {err.exit_code}</span>}
                {err?.source && (
                  <span className="hint">
                    from {err.source === "err" ? "stderr" : "stdout"}
                  </span>
                )}
              </div>
              {err?.message ? (
                <pre className="err-log">{err.message}</pre>
              ) : (
                <div className="hint">
                  {err
                    ? "No log output was captured for this failure."
                    : "Capturing the error log… refresh in a moment."}
                </div>
              )}
            </div>
          )}

          {!res || Object.keys(res).length === 0 ? (
            <div className="empty">
              {active
                ? "GPU usage is shown live above. Host RAM peak, elapsed and exit code come from Slurm accounting once the job finishes."
                : "Resource usage is captured from Slurm accounting once the job finishes. Nothing recorded yet for this run."}
            </div>
          ) : (
            <>
              <div className="stat-row" style={{ marginBottom: 12 }}>
            <div className="stat">
              <div className="label">Elapsed</div>
              <div className="value accent">{res.elapsed ?? "-"}</div>
            </div>
            <div className="stat">
              <div className="label">Avg GPU util</div>
              <div className="value">
                {tl?.avg_util != null ? `${tl.avg_util}%` : "-"}
                {tl?.max_util != null && (
                  <span className="hint"> · peak {tl.max_util}%</span>
                )}
              </div>
            </div>
            <div className="stat">
              <div className="label">Avg GPU power</div>
              <div className="value">
                {tl?.avg_power_w != null ? `${tl.avg_power_w} W` : "-"}
              </div>
            </div>
            <div className="stat">
              <div className="label">Host RAM peak</div>
              <div className="value">{fmtMB(res.max_rss_mb)}</div>
            </div>
          </div>

          {ts && ts.gpu_total_gb != null && (
            <div style={{ marginBottom: 12 }}>
              <div className="hint" style={{ marginBottom: 4 }}>
                GPU memory peak — {ts.gpu_name ?? "GPU"}:{" "}
                <b>{ts.gpu_peak_reserved_gb}</b> / {ts.gpu_total_gb} GB
                {pct != null ? ` (${pct}%)` : ""}
              </div>
              <div className="mem-bar">
                <div
                  className={`mem-fill ${pct != null && pct > 85 ? "hot" : ""}`}
                  style={{ width: `${Math.min(100, pct ?? 0)}%` }}
                />
              </div>
              <div className="hint" style={{ marginTop: 4 }}>
                Allocated peak {ts.gpu_peak_alloc_gb} GB · reserved is what counts toward the
                limit. Headroom: {(ts.gpu_total_gb - (ts.gpu_peak_reserved_gb ?? 0)).toFixed(1)} GB.
              </div>
            </div>
          )}

          <table className="tbl">
            <tbody>
              {ts?.params_millions != null && (
                <tr>
                  <td className="hint">Model params</td>
                  <td>{ts.params_millions} M</td>
                </tr>
              )}
              {ts?.examples && (
                <tr>
                  <td className="hint">Split (train/val/test)</td>
                  <td className="mono">
                    {(() => {
                      const tr = ts.examples.train ?? 0;
                      const va = ts.examples.val ?? 0;
                      const te = ts.examples.test ?? 0;
                      const tot = tr + va + te;
                      const pct = (n: number) =>
                        tot ? `${Math.round((100 * n) / tot)}%` : "?";
                      return (
                        <>
                          {tr.toLocaleString()} / {va.toLocaleString()} /{" "}
                          {te.toLocaleString()}
                          <span className="hint">
                            {" "}
                            ({pct(tr)} / {pct(va)} / {pct(te)})
                          </span>
                        </>
                      );
                    })()}
                  </td>
                </tr>
              )}
              {ts?.batch_size != null && (
                <tr>
                  <td className="hint">Batch size · seq len</td>
                  <td className="mono">
                    {ts.batch_size} · {ts.max_seq_len ?? "?"}
                  </td>
                </tr>
              )}
              {ts?.best_epoch != null && ts.best_epoch > 0 && (
                <tr>
                  <td className="hint">Best epoch (saved)</td>
                  <td className="mono">
                    {ts.best_epoch}
                    {ts.best_val_loss != null && (
                      <span className="hint"> · val_loss {ts.best_val_loss}</span>
                    )}
                  </td>
                </tr>
              )}
              {ts?.total_train_sec != null && (
                <tr>
                  <td className="hint">Pure train time</td>
                  <td>{ts.total_train_sec}s</td>
                </tr>
              )}
              {res.req_mem && (
                <tr>
                  <td className="hint">Requested mem</td>
                  <td className="mono">{res.req_mem}</td>
                </tr>
              )}
              {res.alloc_tres && (
                <tr>
                  <td className="hint">Alloc TRES</td>
                  <td className="mono" style={{ fontSize: 11 }}>{res.alloc_tres}</td>
                </tr>
              )}
              {res.exit_code && (
                <tr>
                  <td className="hint">Exit code</td>
                  <td className="mono">{res.exit_code}</td>
                </tr>
              )}
            </tbody>
          </table>
            </>
          )}

          {/* Run config — params/dataset are known at submit; train_stats fills in after. */}
          {(Object.keys(params).length > 0 || ds?.total_sequences != null || ts) && (
            <>
          <div className="hint" style={{ margin: "14px 0 4px" }}>
            Run config (reproducibility)
          </div>
          <table className="tbl">
            <tbody>
              {ts?.precision && (
                <tr>
                  <td className="hint">Precision</td>
                  <td className="mono">{ts.precision}</td>
                </tr>
              )}
              {ts?.family_dropout != null && (
                <tr>
                  <td className="hint">Family-token dropout</td>
                  <td className="mono">{Math.round(ts.family_dropout * 100)}%</td>
                </tr>
              )}
              {ts?.ddp != null && (
                <tr>
                  <td className="hint">Multi-GPU (DDP)</td>
                  <td className="mono">
                    {ts.ddp ? `on · ${ts.world_size ?? "?"} GPUs` : "off · 1 GPU"}
                  </td>
                </tr>
              )}
              {params.keep_checkpoint ? (
                <tr>
                  <td className="hint">Checkpoint</td>
                  <td className="mono" style={{ fontSize: 11 }}>
                    archived → runs/{job?.job_id}/
                  </td>
                </tr>
              ) : null}
              {Object.keys(params).length > 0 && (
                <tr>
                  <td className="hint">Hyperparameters</td>
                  <td className="mono" style={{ fontSize: 11 }}>
                    {Object.entries(params)
                      .map(([k, v]) => `${k}=${v}`)
                      .join(", ")}
                  </td>
                </tr>
              )}
              {ds?.total_sequences != null && (
                <tr>
                  <td className="hint">Dataset</td>
                  <td className="mono">
                    {ds.total_sequences.toLocaleString()} seqs
                    {ds.count_param != null ? ` · ${ds.count_param}/family` : ""}
                    {ds.seed != null ? ` · seed ${ds.seed}` : ""}
                    {ds.generated_on ? ` · ${ds.generated_on}` : ""}
                  </td>
                </tr>
              )}
              {hasFamilies && (
                <tr>
                  <td className="hint">Per family</td>
                  <td className="mono" style={{ fontSize: 11 }}>
                    {Object.entries(ds!.families!)
                      .map(([fam, n]) => `${fam}: ${n ?? "?"}`)
                      .join(" · ")}
                  </td>
                </tr>
              )}
              {ds?.generated_at != null && (
                <tr>
                  <td className="hint">Dataset generated</td>
                  <td>{fmtTime(ds.generated_at)}</td>
                </tr>
              )}
            </tbody>
          </table>
            </>
          )}
          </>
        )}
      </div>
    </>
  );
}
