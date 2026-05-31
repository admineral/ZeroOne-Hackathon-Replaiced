import { useState } from "react";
import { api, ValidateResult } from "../api";
import { ruleLabel } from "../ruleLabels";

const EXAMPLE = `DEVELOP PHOTORESIST
OXIDE ETCH
CLEAN AFTER OXIDE ETCH
DEPOSIT BACKSIDE METAL`;

export function ValidateBox() {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const [result, setResult] = useState<ValidateResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.validateSequence(text);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <button
        className="card-head collapse-head"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        type="button"
      >
        <h3>
          <span className={`chevron ${open ? "open" : ""}`}>▶</span> Validate a
          sequence
        </h3>
        <span className="hint">
          {open ? "10 process-logic rules" : "click to expand · check any recipe"}
        </span>
      </button>

      {!open ? null : (
        <>
          <p className="hint" style={{ marginTop: 0 }}>
            Paste steps (one per line, or comma/semicolon separated) to get the exact
            violation list back. Step names are matched case-insensitively against the
            recipe vocabulary.
          </p>

          <textarea
        className="seq-input"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={EXAMPLE}
        rows={6}
        spellCheck={false}
      />

      <div className="row" style={{ marginTop: 10, gap: 8 }}>
        <button className="btn primary" onClick={run} disabled={loading || !text.trim()}>
          {loading ? "Checking…" : "Validate"}
        </button>
        <button
          className="btn"
          onClick={() => setText(EXAMPLE)}
          disabled={loading}
          type="button"
        >
          Load example
        </button>
        {result && (
          <span className="hint" style={{ alignSelf: "center" }}>
            {result.steps} step{result.steps === 1 ? "" : "s"} checked
          </span>
        )}
      </div>

      {error && <div className="empty">Error: {error}</div>}

      {result && !error && (
        <div style={{ marginTop: 14 }}>
          {result.valid ? (
            <div className="validate-banner ok">
              No rule violations.
              <span className="hint" style={{ display: "block", fontWeight: 400 }}>
                Passing these 10 rules is necessary but not sufficient — short or
                incomplete recipes can still pass.
              </span>
            </div>
          ) : (
            <div className="validate-banner bad">
              {result.violations.length} violation
              {result.violations.length === 1 ? "" : "s"} found
            </div>
          )}

          {result.violations.length > 0 && (
            <table className="tbl" style={{ marginTop: 12 }}>
              <thead>
                <tr>
                  <th>Step</th>
                  <th>Rule</th>
                  <th>Why</th>
                </tr>
              </thead>
              <tbody>
                {result.violations.map((v, idx) => (
                  <tr key={idx}>
                    <td>
                      <span className="mono">#{v.step_index}</span>{" "}
                      <span className="hint">{v.step_name}</span>
                    </td>
                    <td className="mono">{v.rule}</td>
                    <td style={{ color: "#9aa4b2" }}>{ruleLabel(v.rule)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
