import { useEffect, useState } from "react";
import {
  api,
  CheckpointInfo,
  JobRecord,
  RemoteListing,
  SubmissionStatus,
} from "../api";
import { JobStatusBadge } from "./JobStatusBadge";

// Map the known prediction files to their organizer task for a friendly label.
const TASK_BY_FILE: Record<string, string> = {
  "predictions_nextstep.csv": "Task 1 next-step",
  "predictions_completion.csv": "Task 2 completion",
  "predictions_anomaly.csv": "Task 3 anomaly",
};

function fmtBytes(n: number | null | undefined): string {
  if (n == null) return "-";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function fmtWhen(epoch: number | null | undefined): string {
  if (!epoch) return "-";
  const d = new Date(epoch * 1000);
  if (Number.isNaN(d.getTime())) return "-";
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

function isJobActive(status: string): boolean {
  const base = (status || "").split(" ")[0].toUpperCase();
  if (base === "SUBMITTED") return true;
  return [
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "RESIZING",
    "PENDING",
    "REQUEUED",
    "SUSPENDED",
  ].includes(base);
}

interface Props {
  refreshKey: number;
  // run_key === "submission" jobs, newest first (from App's job poll).
  submissionJobs: JobRecord[];
  onChanged?: () => void;
}

export function SubmissionCard({ refreshKey, submissionJobs, onChanged }: Props) {
  const [data, setData] = useState<SubmissionStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [remote, setRemote] = useState<RemoteListing | null>(null);
  const [remoteError, setRemoteError] = useState<string | null>(null);
  const [remoteLoading, setRemoteLoading] = useState(false);
  const [submitting, setSubmitting] = useState<string | null>(null);
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[] | null>(null);
  const [checkpointsLoading, setCheckpointsLoading] = useState(false);
  const [checkpointsError, setCheckpointsError] = useState<string | null>(null);
  const [selectedSource, setSelectedSource] = useState<string>("");
  const [removing, setRemoving] = useState<string | null>(null);

  const activeJob = submissionJobs[0];
  const jobRunning = !!activeJob && isJobActive(activeJob.status);

  const load = () => {
    setLoading(true);
    api
      .getSubmission()
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  };

  const loadRemote = () => {
    setRemoteLoading(true);
    setRemoteError(null);
    api
      .getSubmissionRemote()
      .then((d) => {
        setRemote(d);
        setRemoteError(null);
      })
      .catch((err) => {
        setRemote(null);
        setRemoteError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setRemoteLoading(false));
  };

  const loadCheckpoints = () => {
    setCheckpointsLoading(true);
    setCheckpointsError(null);
    api
      .getCheckpoints()
      .then((res) => {
        setCheckpoints(res.checkpoints);
        setSelectedSource((prev) => {
          if (prev && res.checkpoints.some((c) => c.source === prev)) return prev;
          const def = res.checkpoints.find((c) => c.is_current) ?? res.checkpoints[0];
          return def ? def.source : "";
        });
      })
      .catch((err) => {
        setCheckpoints(null);
        setCheckpointsError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setCheckpointsLoading(false));
  };

  useEffect(load, [refreshKey]);
  useEffect(loadRemote, [refreshKey]);
  useEffect(loadCheckpoints, [refreshKey]);

  // Refresh status + the Leonardo listing once the latest job finishes and its
  // results have been pulled back from Leonardo.
  useEffect(() => {
    if (activeJob && !isJobActive(activeJob.status)) {
      load();
      loadRemote();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeJob?.job_id, activeJob?.status, activeJob?.submission_fetched]);

  const run = (tasks: string[]) => {
    setSubmitting(tasks.join(","));
    api
      .runSubmission({ source: selectedSource || undefined, tasks })
      .then(() => {
        setError(null);
        onChanged?.();
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setSubmitting(null));
  };

  const remove = (source: string) => {
    setRemoving(source);
    api
      .removeCheckpoint(source)
      .then((res) => {
        setCheckpoints(res.checkpoints);
        setError(null);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setRemoving(null));
  };

  const selectedLabel =
    checkpoints?.find((c) => c.source === selectedSource)?.label ?? "—";

  const busy = !!submitting || jobRunning;

  return (
    <div className="card">
      <div className="card-head">
        <div>
          <h3>Official submission</h3>
          <span className="hint">runs on Leonardo · results land in participant_files/submission</span>
        </div>
        <div className="btn-row">
          <button className="btn sm ghost" onClick={load} disabled={loading}>
            {loading ? "…" : "Refresh"}
          </button>
          <button
            className="btn sm ghost"
            onClick={() => run(["anomaly"])}
            disabled={busy}
            title="Task 3 only (rule-based, no model needed)"
          >
            {submitting === "anomaly" ? "Submitting…" : "Anomaly only"}
          </button>
          <button
            className="btn sm primary"
            onClick={() => run(["all"])}
            disabled={busy}
            title="Run all three tasks on Leonardo with the selected checkpoint"
          >
            {submitting === "all" ? "Submitting…" : "Run on Leonardo"}
          </button>
        </div>
      </div>

      {error && <div className="empty">Error: {error}</div>}

      {activeJob && (
        <div
          className={`validate-banner ${jobRunning ? "" : "good"}`}
          style={{ marginBottom: 12, display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}
        >
          <JobStatusBadge state={activeJob.status} />
          <span className="mono">{activeJob.job_id ?? "-"}</span>
          <span className="hint">
            {jobRunning
              ? "Running on Leonardo — result CSVs download automatically when it finishes."
              : activeJob.submission_fetched
                ? "Finished — result CSVs pulled back below."
                : "Finished — fetching result CSVs…"}
          </span>
        </div>
      )}

      {!error && !data && (
        <div className="empty">
          {loading ? "Checking official submission files…" : "No submission status loaded yet."}
        </div>
      )}

      {!error && data && (
        <>
          <div className="stat-row" style={{ marginBottom: 12 }}>
            <div className="stat">
              <div className="label">Official valid input</div>
              <div className="value">{data.inputs.valid.rows?.toLocaleString() ?? "-"}</div>
            </div>
            <div className="stat">
              <div className="label">Official anomaly input</div>
              <div className="value">{data.inputs.anomaly.rows?.toLocaleString() ?? "-"}</div>
            </div>
            <div className="stat">
              <div className="label">Predicted invalid</div>
              <div className="value accent">
                {data.anomaly_invalid != null ? data.anomaly_invalid.toLocaleString() : "-"}
              </div>
            </div>
            <div className="stat">
              <div className="label">Checkpoint</div>
              <div className="value" style={{ fontSize: 13 }}>{selectedLabel}</div>
            </div>
          </div>

          {/* Remote checkpoint picker */}
          <div className="section" style={{ marginBottom: 12 }}>
            <div className="section-title" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span>Checkpoints on Leonardo</span>
              <button className="btn sm ghost" onClick={loadCheckpoints} disabled={checkpointsLoading}>
                {checkpointsLoading ? "Scanning…" : "Rescan"}
              </button>
            </div>
            {checkpointsError ? (
              <div className="empty">Checkpoint list unavailable: {checkpointsError}</div>
            ) : checkpointsLoading && !checkpoints ? (
              <div className="empty">Scanning Leonardo for checkpoints…</div>
            ) : checkpoints && checkpoints.length === 0 ? (
              <div className="empty">
                No trained checkpoint on Leonardo yet — train the transformer first. (Anomaly
                only still runs without a model.)
              </div>
            ) : (
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Use</th>
                    <th>Checkpoint</th>
                    <th>Size</th>
                    <th>Trained</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {checkpoints?.map((c) => (
                    <tr key={c.source} className={selectedSource === c.source ? "row-selected" : ""}>
                      <td>
                        <input
                          type="radio"
                          name="submission-checkpoint"
                          checked={selectedSource === c.source}
                          onChange={() => setSelectedSource(c.source)}
                          disabled={busy}
                        />
                      </td>
                      <td>
                        {c.label}
                        {c.is_current && <span className="hint"> · canonical</span>}
                      </td>
                      <td>{fmtBytes(c.bytes)}</td>
                      <td className="hint">{fmtWhen(c.mtime)}</td>
                      <td onClick={(e) => e.stopPropagation()}>
                        {c.is_current ? (
                          <span style={{ color: "#5f6b80" }}>-</span>
                        ) : (
                          <button
                            className="btn sm danger"
                            onClick={() => remove(c.source)}
                            disabled={removing === c.source || busy}
                            title="Delete this archived checkpoint folder on Leonardo"
                          >
                            {removing === c.source ? "Removing…" : "Remove"}
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {/* Prediction CSVs on Leonardo — the authoritative submission output */}
          <div className="section">
            <div
              className="section-title"
              style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}
            >
              <span>Prediction CSVs on Leonardo</span>
              <button className="btn sm ghost" onClick={loadRemote} disabled={remoteLoading}>
                {remoteLoading ? "Listing…" : "Rescan"}
              </button>
            </div>
            {remote?.dir && (
              <div className="hint mono" style={{ marginBottom: 8 }}>
                {remote.dir}
              </div>
            )}
            {remoteError ? (
              <div className="empty">Leonardo listing unavailable: {remoteError}</div>
            ) : remoteLoading && !remote ? (
              <div className="empty">Listing the Leonardo submission folder…</div>
            ) : remote && !remote.exists ? (
              <div className="empty">
                No submission folder on Leonardo yet — run a submission to create it.
              </div>
            ) : remote && remote.entries.length === 0 ? (
              <div className="empty">
                Folder is empty on Leonardo — run a submission to populate it.
              </div>
            ) : remote ? (
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Task</th>
                    <th>File</th>
                    <th>Rows</th>
                    <th>Size</th>
                    <th>Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {remote.entries.map((entry) => (
                    <tr key={entry.name}>
                      <td>{TASK_BY_FILE[entry.name] ?? (entry.is_dir ? "folder" : "—")}</td>
                      <td className="mono">{entry.name}</td>
                      <td>{entry.rows != null ? entry.rows.toLocaleString() : "-"}</td>
                      <td>{entry.is_dir ? "-" : fmtBytes(entry.bytes)}</td>
                      <td className="hint">{fmtWhen(entry.mtime)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="empty">Not listed yet.</div>
            )}
            <div className="info-note" style={{ marginTop: 12 }}>
              <i className="info-note-icon">i</i>
              <span>
                <strong>Hinweis:</strong> the job runs <span className="mono">make_submission.py</span>{" "}
                on Leonardo with the selected checkpoint and writes these CSVs to{" "}
                <span className="mono">{remote?.dir ?? "outputs/transformer/submission"}</span> on the
                cluster. They are mirrored to <span className="mono">{data.output_dir}</span> on this
                machine — hand that folder's CSVs to the organizers as your official submission.
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
