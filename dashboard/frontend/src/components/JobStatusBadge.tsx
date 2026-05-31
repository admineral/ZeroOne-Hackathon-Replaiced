interface Props {
  state: string | null | undefined;
  label?: string;
}

function classify(state: string | null | undefined): string {
  if (!state) return "";
  const base = state.split(" ")[0].toUpperCase();
  if (["RUNNING", "CONFIGURING", "COMPLETING", "RESIZING"].includes(base))
    return "running";
  if (["PENDING", "REQUEUED", "SUSPENDED"].includes(base)) return "pending";
  if (base === "COMPLETED") return "completed";
  if (["FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"].includes(base))
    return "failed";
  return "";
}

export function JobStatusBadge({ state, label }: Props) {
  const cls = classify(state);
  const text = label ?? state ?? "unknown";
  return (
    <span className={`badge ${cls}`}>
      <span className="led" />
      {text}
    </span>
  );
}
