import { useEffect, useState } from "react";
import { api, ResultsPayload } from "../api";
import { ruleLabel } from "../ruleLabels";

/** One selectable evaluation run for the results dropdown. */
export interface EvalResultOption {
  jobId: string;
  label: string;
}

interface Props {
  run: string;
  refreshKey: number;
  /** When set, load this past job's archived results instead of the latest. */
  jobId?: string;
  /** Training job whose archived checkpoint this evaluation scored, if any. */
  sourceJobId?: string;
  /**
   * Whether there is an evaluation to show. When false (e.g. history was just
   * cleared) we stay blank instead of re-loading the last eval summary on disk.
   */
  active?: boolean;
  /** Selectable evaluation runs (newest first) for the results dropdown. */
  options?: EvalResultOption[];
  /** Explicitly-selected eval job id ("" / undefined = latest). Drives the dropdown. */
  selectedJobId?: string;
  /** Called when the user picks a different evaluation from the dropdown. */
  onSelectJob?: (jobId: string | undefined) => void;
}

function pct(rate: string | number | undefined): string {
  if (rate === undefined || rate === "") return "-";
  const value = Number(rate);
  if (Number.isNaN(value)) return String(rate);
  return `${(value * 100).toFixed(1)}%`;
}

function num(value: string | number | undefined, digits = 2): string {
  if (value === undefined || value === "") return "-";
  const v = Number(value);
  if (Number.isNaN(v)) return String(value);
  return v.toFixed(digits);
}

function qualityClass(rate: number | null): string {
  if (rate === null) return "";
  if (rate >= 0.6) return "good";
  if (rate >= 0.3) return "warn";
  return "bad";
}

export function ResultsPanel({
  run,
  refreshKey,
  jobId,
  sourceJobId,
  active = true,
  options,
  selectedJobId,
  onSelectJob,
}: Props) {
  const [data, setData] = useState<ResultsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    // Nothing to show (no inspected/latest eval job): clear and stay blank.
    if (!active && !jobId) {
      setData(null);
      setError(null);
      setLoading(false);
      return () => {
        cancelled = true;
      };
    }
    setLoading(true);
    api
      .getResults(run, jobId)
      .then((res) => {
        if (!cancelled) {
          setData(res);
          setError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [run, refreshKey, jobId, active]);

  const summary = data?.summary ?? [];
  const modelRows = summary.filter((r) => r.source === "model_generated");
  const hasQuality = summary.some((r) => r.quality_rate !== undefined);

  const split = (data?.split ?? null) as {
    train?: number;
    validation?: number;
    test?: number;
    train_ratio?: number;
    val_ratio?: number;
    test_ratio?: number;
  } | null;
  const ratioPct = (r: number | undefined): string =>
    r === undefined || r === null || Number.isNaN(Number(r))
      ? "?"
      : `${Math.round(Number(r) * 100)}%`;
  const hasSplit =
    !!split && (split.train != null || split.validation != null || split.test != null);
  const testRatio =
    split?.test_ratio ??
    (split && split.train_ratio != null && split.val_ratio != null
      ? 1 - Number(split.train_ratio) - Number(split.val_ratio)
      : undefined);

  // Headline metrics use the best-quality model row so a single degenerate
  // fraction can't hide behind a good one.
  const bestRow =
    modelRows.length > 0
      ? modelRows.reduce((best, r) =>
          (Number(r.quality_rate) || 0) > (Number(best.quality_rate) || 0) ? r : best
        )
      : null;
  const bestValid =
    modelRows.length > 0
      ? Math.max(...modelRows.map((r) => Number(r.valid_rate) || 0))
      : null;
  const bestQuality = bestRow ? Number(bestRow.quality_rate) || 0 : null;

  return (
    <div className="card">
      <div className="card-head">
        <h3>Rule-aware results</h3>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {options && options.length > 0 && onSelectJob && (
            <select
              className="inline-input"
              value={selectedJobId ?? ""}
              onChange={(e) => onSelectJob(e.target.value || undefined)}
              title="Choose which evaluation run's results to display"
            >
              <option value="">Latest evaluation</option>
              {options.map((o) => (
                <option key={o.jobId} value={o.jobId}>
                  {o.label}
                </option>
              ))}
            </select>
          )}
          <span className="hint">
            {(() => {
              const shownJobId = jobId ?? data?.job_id ?? undefined;
              const isArchived = data?.archived ?? !!jobId;
              const base = shownJobId
                ? `${run} · ${isArchived ? "archived" : "latest"} job ${shownJobId}`
                : `${run} · latest`;
              return sourceJobId
                ? `${base} · checkpoint from training job ${sourceJobId}`
                : base;
            })()}
          </span>
        </div>
      </div>

      {loading && !data && <div className="empty">Loading results…</div>}
      {error && <div className="empty">No results yet ({error})</div>}

      {hasSplit && (
        <div
          className="hint"
          style={{
            margin: "0 0 12px",
            display: "flex",
            gap: 14,
            flexWrap: "wrap",
            alignItems: "center",
          }}
        >
          <span>
            Data split:{" "}
            <strong style={{ color: "#e6e9ef" }}>train {split?.train ?? "?"}</strong> ·{" "}
            <strong style={{ color: "#e6e9ef" }}>val {split?.validation ?? "?"}</strong> ·{" "}
            <strong style={{ color: "#e6e9ef" }}>test {split?.test ?? "?"}</strong>
          </span>
          {split?.train_ratio != null && (
            <span>
              ({ratioPct(split.train_ratio)} / {ratioPct(split.val_ratio)} /{" "}
              {ratioPct(testRatio)})
            </span>
          )}
          <span>Scored on {split?.validation ?? "?"} held-out val sequences.</span>
        </div>
      )}

      {summary.length > 0 && (
        <>
          <div className="stat-row" style={{ marginBottom: 6 }}>
            <div className="stat">
              <div className="label">Quality rate (non-gameable)</div>
              <div className={`value ${qualityClass(bestQuality)}`}>
                {hasQuality && bestQuality !== null ? pct(bestQuality) : "-"}
              </div>
            </div>
            <div className="stat">
              <div className="label">Valid rate (gameable)</div>
              <div className="value">
                {bestValid !== null ? pct(bestValid) : "-"}
              </div>
            </div>
            <div className="stat">
              <div className="label">Mean next-step acc</div>
              <div className="value accent">
                {bestRow ? pct(bestRow.mean_suffix_acc) : "-"}
              </div>
            </div>
            <div className="stat">
              <div className="label">Finished (EOS)</div>
              <div className="value">{bestRow ? pct(bestRow.eos_rate) : "-"}</div>
            </div>
          </div>
          <p className="hint" style={{ margin: "0 0 14px" }}>
            Quality rate only counts completions that are rule-valid <em>and</em> actually
            finish, have a realistic length, don't loop, and reproduce ≥50% of the true
            continuation — so trivial output can't score high.
          </p>

          <table className="tbl">
            <thead>
              <tr>
                <th>Source</th>
                <th>Cut</th>
                <th>Seqs</th>
                <th>Valid</th>
                <th>Quality</th>
                <th>Acc</th>
                <th>Len×</th>
                <th>EOS</th>
                <th>Jacc</th>
              </tr>
            </thead>
            <tbody>
              {summary.map((row, idx) => (
                <tr key={idx}>
                  <td className="mono">{row.source}</td>
                  <td>{row.completion_fraction}</td>
                  <td>{row.sequences}</td>
                  <td style={{ color: "#9aa4b2" }}>{pct(row.valid_rate)}</td>
                  <td style={{ color: "#36d399", fontWeight: 600 }}>
                    {pct(row.quality_rate)}
                  </td>
                  <td>{pct(row.mean_suffix_acc)}</td>
                  <td>{num(row.mean_len_ratio)}</td>
                  <td>{pct(row.eos_rate)}</td>
                  <td>{num(row.mean_jaccard)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {data?.rule_counts && data.rule_counts.length > 0 && (
        <div style={{ marginTop: 18 }}>
          <div className="card-head">
            <h3 style={{ fontSize: 12 }}>Which rules the model broke</h3>
          </div>
          <table className="tbl">
            <thead>
              <tr>
                <th>Rule</th>
                <th>What it catches</th>
                <th>Count</th>
              </tr>
            </thead>
            <tbody>
              {[...data.rule_counts]
                .sort((a, b) => (Number(b.count) || 0) - (Number(a.count) || 0))
                .map((row, idx) => (
                  <tr key={idx}>
                    <td className="mono">{row.rule}</td>
                    <td style={{ color: "#9aa4b2" }}>{ruleLabel(row.rule)}</td>
                    <td style={{ fontWeight: 600 }}>{row.count}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}

      {!loading && summary.length === 0 && !error && (
        <div className="empty">
          Run an evaluation to populate valid-rate, quality-rate and violation metrics.
        </div>
      )}
    </div>
  );
}
