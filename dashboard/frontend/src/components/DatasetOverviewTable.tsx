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
  selectedId: string;
  onSelect: (id: string) => void;
  onRefresh: () => void;
  onOpenDataset?: () => void;
}

export function DatasetOverviewTable({
  datasets,
  loading,
  error,
  selectedId,
  onSelect,
  onRefresh,
  onOpenDataset,
}: Props) {
  return (
    <div className="card">
      <div className="card-head">
        <h3>Datasets on Leonardo</h3>
        <span style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {onOpenDataset && (
            <button className="btn sm ghost" type="button" onClick={onOpenDataset}>
              Manage
            </button>
          )}
          <button className="btn sm ghost" type="button" onClick={onRefresh} disabled={loading}>
            {loading ? "…" : "Refresh"}
          </button>
        </span>
      </div>

      {error && <div className="empty">Error: {error}</div>}

      {!error && datasets.length === 0 && (
        <div className="empty">
          {loading
            ? "Scanning Leonardo for datasets…"
            : "No datasets yet. Open Dataset to generate one."}
        </div>
      )}

      {!error && datasets.length > 0 && (
        <div className="tbl-scroll-3">
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
              </tr>
            </thead>
            <tbody>
              {datasets.map((d) => {
                const active = d.id === selectedId;
                return (
                  <tr
                    key={d.id}
                    className={active ? "row-active" : undefined}
                    style={active ? { background: "rgba(99,102,241,0.10)" } : undefined}
                  >
                    <td>
                      <input
                        type="radio"
                        name="overview-dataset"
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
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
