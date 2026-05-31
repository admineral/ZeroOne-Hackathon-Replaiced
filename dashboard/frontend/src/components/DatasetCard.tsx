import { useState } from "react";
import { DatasetSummary } from "../api";

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

interface Props {
  datasets: DatasetSummary[];
  loading: boolean;
  error: string | null;
  /** The dataset id train/eval will run against ("" = script default). */
  selectedId: string;
  onSelect: (id: string) => void;
  onRefresh: () => void;
  onDelete: (id: string) => Promise<void>;
  /** Submit a one-time packing (memmap) job for this dataset. */
  onPreprocess: (id: string) => Promise<void>;
  /** Dataset ids with a preprocess (packing) job currently in the queue. */
  activePackingIds?: string[];
}

export function DatasetCard({
  datasets,
  loading,
  error,
  selectedId,
  onSelect,
  onRefresh,
  onDelete,
  onPreprocess,
  activePackingIds = [],
}: Props) {
  // Which row is in the "confirm delete?" state, and which is mid-delete.
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  // Row whose packing job is being submitted (sbatch returns quickly).
  const [packingId, setPackingId] = useState<string | null>(null);

  const remove = async (id: string) => {
    setDeletingId(id);
    try {
      await onDelete(id);
    } finally {
      setDeletingId(null);
      setConfirmId(null);
    }
  };

  const pack = async (id: string) => {
    setPackingId(id);
    try {
      await onPreprocess(id);
    } finally {
      setPackingId(null);
    }
  };

  return (
    <div className="card">
      <div className="card-head">
        <h3>Dataset collection on Leonardo</h3>
        <button className="btn sm ghost" onClick={onRefresh} disabled={loading}>
          {loading ? "…" : "Refresh"}
        </button>
      </div>

      {error && <div className="empty">Error: {error}</div>}

      {!error && datasets.length === 0 && (
        <div className="empty">
          {loading
            ? "Scanning Leonardo for datasets…"
            : "No datasets yet. Use \u201CGenerate on Leonardo\u201D above to create one."}
        </div>
      )}

      {!error && datasets.length > 0 && (
        <>
          <table className="tbl">
            <thead>
              <tr>
                <th></th>
                <th>Dataset</th>
                <th>Families</th>
                <th>Count/family</th>
                <th>Seed</th>
                <th>Sequences</th>
                <th>Size</th>
                <th>Created</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((d) => {
                const active = d.id === selectedId;
                const confirming = confirmId === d.id;
                const deleting = deletingId === d.id;
                // A packing job for this dataset is live in the Slurm queue.
                const packing = activePackingIds.includes(d.id);
                return (
                  <tr
                    key={d.id}
                    className={active ? "row-active" : undefined}
                    style={active ? { background: "rgba(99,102,241,0.10)" } : undefined}
                  >
                    <td>
                      <input
                        type="radio"
                        name="active-dataset"
                        checked={active}
                        disabled={!d.ready}
                        onChange={() => onSelect(d.id)}
                        title={
                          d.ready
                            ? "Use this dataset for Train / Evaluate"
                            : "Still generating — not selectable yet"
                        }
                      />
                    </td>
                    <td className="mono" title={d.id}>
                      {d.id}
                      {d.legacy && (
                        <span className="hint" style={{ marginLeft: 6 }}>
                          (default)
                        </span>
                      )}
                      {!d.ready && (
                        <span className="hint" style={{ marginLeft: 6 }}>
                          (generating…)
                        </span>
                      )}
                      {d.packed && (
                        <span
                          title="A packed memmap blob exists — train/eval run on the full dataset with near-zero RAM."
                          style={{
                            marginLeft: 6,
                            padding: "1px 6px",
                            borderRadius: 6,
                            fontSize: 11,
                            fontWeight: 600,
                            color: "#065f46",
                            background: "rgba(16,185,129,0.18)",
                          }}
                        >
                          packed
                        </span>
                      )}
                    </td>
                    <td style={{ textTransform: "uppercase" }}>
                      {d.families.length ? d.families.join(", ") : "-"}
                    </td>
                    <td>{d.count_param != null ? d.count_param.toLocaleString() : "?"}</td>
                    <td>{d.seed ?? "?"}</td>
                    <td>
                      {d.total_sequences != null ? d.total_sequences.toLocaleString() : "?"}
                    </td>
                    <td>{fmtBytes(d.bytes)}</td>
                    <td className="hint">{fmtWhen(d.generated_at)}</td>
                    <td style={{ whiteSpace: "nowrap" }}>
                      {confirming ? (
                        <>
                          <button
                            className="btn sm danger"
                            onClick={() => remove(d.id)}
                            disabled={deleting}
                          >
                            {deleting ? "Deleting…" : "Confirm"}
                          </button>
                          <button
                            className="btn sm ghost"
                            onClick={() => setConfirmId(null)}
                            disabled={deleting}
                            style={{ marginLeft: 6 }}
                          >
                            Cancel
                          </button>
                        </>
                      ) : (
                        <>
                          <button
                            className="btn sm ghost"
                            onClick={() => pack(d.id)}
                            disabled={!d.ready || !!packingId || !!deletingId || packing}
                            title={
                              packing
                                ? "A packing job for this dataset is currently running on Leonardo"
                                : d.packed
                                  ? "Re-pack this dataset into a memmap blob (overwrites packed/)"
                                  : "Pack this dataset into a memmap blob so train/eval scale to the full set"
                            }
                          >
                            {packingId === d.id
                              ? "Submitting…"
                              : packing
                                ? "Packing…"
                                : d.packed
                                  ? "Re-pack"
                                  : "Preprocess"}
                          </button>
                          <button
                            className="btn sm ghost"
                            onClick={() => setConfirmId(d.id)}
                            disabled={!!deletingId || !!packingId}
                            title={
                              d.legacy
                                ? "Remove the default dataset's files (variant CSVs + packed blob) from Leonardo. The pipeline code in training_data/ is kept."
                                : "Delete this dataset from Leonardo"
                            }
                            style={{ marginLeft: 6 }}
                          >
                            Delete
                          </button>
                        </>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
