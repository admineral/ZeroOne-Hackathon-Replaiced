import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { DefaultChatTransport, UIMessage } from "ai";
import { useChat } from "@ai-sdk/react";
import {
  api,
  ExperimentAnalysisCard,
  ExperimentAnalysisResponse,
  RunParams,
} from "../api";

interface Props {
  currentControls: RunParams & {
    train_run?: string;
    eval_run?: string;
    generation_time_limit?: string;
  };
  trainRun: string;
  evalRun: string;
  selectedJobId?: string | null;
  /** Runs the coach can target; selecting one focuses its advice. */
  runOptions?: { jobId: string; label: string }[];
  onSelectRun?: (jobId: string | null) => void;
  autoTriggerKey?: string;
  onApplyParams: (params: RunParams, runKey?: string) => void;
  onApproveAction: (proposal: ExperimentAnalysisCard["actionProposal"]) => Promise<void>;
}

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function compactValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : String(value);
  if (typeof value === "boolean") return value ? "on" : "off";
  return String(value);
}

function paramsFromSingle(card: ExperimentAnalysisCard): RunParams {
  const field = card.singleChangeSuggestion.field as keyof RunParams;
  return { [field]: card.singleChangeSuggestion.to } as RunParams;
}

function messageText(message: UIMessage): string {
  return (message.parts ?? [])
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("");
}

export function AiCoachPanel({
  currentControls,
  trainRun,
  evalRun,
  selectedJobId,
  runOptions,
  onSelectRun,
  autoTriggerKey,
  onApplyParams,
  onApproveAction,
}: Props) {
  const [analysis, setAnalysis] = useState<ExperimentAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [approvalBusy, setApprovalBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const lastAuto = useRef<string | undefined>();
  const chatTransport = useMemo(
    () => new DefaultChatTransport<UIMessage>({ api: "/api/ai/chat" }),
    []
  );
  const chat = useChat({
    transport: chatTransport,
    onError: (err) => setError(err.message),
  });

  const analyze = async (
    prompt = "Analyze the latest dashboard state and suggest the next experiment.",
    forceRefresh = false
  ) => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.analyzeExperiment({
        prompt,
        currentControls,
        selectedJobId,
        trainRun,
        evalRun,
        forceRefresh,
      });
      setAnalysis(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!autoTriggerKey || autoTriggerKey === lastAuto.current) return;
    lastAuto.current = autoTriggerKey;
    void analyze("A watched job just reached a terminal state. Analyze the outcome and suggest the next step.");
  }, [autoTriggerKey]);

  const submitChat = async (event: FormEvent) => {
    event.preventDefault();
    const text = question.trim();
    if (!text || chat.status === "streaming" || chat.status === "submitted") return;
    setQuestion("");
    setError(null);
    try {
      await chat.sendMessage({ text });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const card = analysis?.card;

  return (
    <div className="card ai-coach">
      <div className="card-head">
        <div>
          <h3>AI Experiment Coach</h3>
          <span className="hint">
            typed tools only · {analysis ? analysis.model : "ready"}
            {analysis?.cached ? " · cached" : ""}
            {analysis?.warning ? ` · ${analysis.warning}` : ""}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {runOptions && runOptions.length > 0 && (
            <select
              className="inline-input coach-run-select"
              value={selectedJobId ?? ""}
              onChange={(e) => onSelectRun?.(e.target.value || null)}
              title="Pick which run the coach gives advice on"
            >
              <option value="">Latest state (auto)</option>
              {runOptions.map((opt) => (
                <option key={opt.jobId} value={opt.jobId}>
                  {opt.label}
                </option>
              ))}
            </select>
          )}
          <button className="btn sm" disabled={loading} onClick={() => analyze(undefined, true)}>
            {loading ? "Analyzing…" : "Analyze"}
          </button>
        </div>
      </div>

      {error && <div className="coach-error">{error}</div>}

      {card ? (
        <div className="coach-layout">
          <div className={`coach-verdict ${card.verdict}`}>
            <div>
              <span className="coach-kicker">{card.verdict}</span>
              <strong>{card.currentBottleneck}</strong>
            </div>
            <span className="hint">confidence {pct(card.confidence)}</span>
          </div>

          <p className="coach-summary">{card.englishSummary}</p>

          <div className="coach-list">
            {card.diagnosis.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>

          <div className="coach-suggestion">
            <div className="coach-suggestion-head">
              <div>
                <strong>{card.fullSuggestion.title}</strong>
                <span className="hint">full config · {card.fullSuggestion.runKey}</span>
              </div>
              <button
                className="btn sm primary"
                onClick={() => onApplyParams(card.fullSuggestion.params, card.fullSuggestion.runKey)}
              >
                Apply settings
              </button>
            </div>
            <div className="param-grid">
              {Object.entries(card.fullSuggestion.params).map(([key, value]) => (
                <span key={key} className="param-pill" title={card.fullSuggestion.reasonBySetting[key]}>
                  <span>{key}</span>
                  <strong>{compactValue(value)}</strong>
                </span>
              ))}
            </div>
          </div>

          <div className="coach-suggestion subtle">
            <div className="coach-suggestion-head">
              <div>
                <strong>Single-change check</strong>
                <span className="hint">
                  {card.singleChangeSuggestion.field}: {compactValue(card.singleChangeSuggestion.from)} →{" "}
                  {compactValue(card.singleChangeSuggestion.to)}
                </span>
              </div>
              <button className="btn sm" onClick={() => onApplyParams(paramsFromSingle(card))}>
                Apply one change
              </button>
            </div>
            <p className="hint">{card.singleChangeSuggestion.reason}</p>
            <p className="hint">{card.singleChangeSuggestion.expectedObservableDifference}</p>
          </div>

          {card.ablationPlan.length > 0 && (
            <div className="coach-ablation">
              {card.ablationPlan.map((item) => (
                <button key={item.label} className="ablation-item" onClick={() => onApplyParams(item.params)}>
                  <strong>{item.label}</strong>
                  <span>{item.reason}</span>
                </button>
              ))}
            </div>
          )}

          {card.riskNotes.length > 0 && (
            <div className="coach-list warn">
              {card.riskNotes.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
          )}

          <div className="coach-action">
            <div>
              <strong>{card.actionProposal.approvalText}</strong>
              <span className="hint">{card.actionProposal.expectedCostRisk}</span>
            </div>
            <button
              className="btn sm primary"
              disabled={approvalBusy || !card.actionProposal.requiresApproval}
              onClick={async () => {
                setApprovalBusy(true);
                try {
                  await onApproveAction(card.actionProposal);
                } finally {
                  setApprovalBusy(false);
                }
              }}
            >
              {approvalBusy ? "Working…" : card.actionProposal.requiresApproval ? "Approve" : "No action"}
            </button>
          </div>
        </div>
      ) : (
        <div className="empty">Run an analysis after data generation, training, or evaluation.</div>
      )}

      <form className="coach-chat" onSubmit={submitChat}>
        <div className="coach-chat-log">
          {chat.messages.length === 0 ? (
            <span className="hint">Ask for a comparison, an ablation, or why a metric moved.</span>
          ) : (
            chat.messages.map((msg) => (
              <div key={msg.id} className={`chat-msg ${msg.role}`}>
                {messageText(msg) || (msg.role === "assistant" && chat.status === "streaming" ? "Thinking…" : "")}
              </div>
            ))
          )}
        </div>
        <span className="input-with-actions">
          <input
            className="inline-input grow"
            value={question}
            placeholder="Ask the coach…"
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button
            className="btn sm"
            disabled={chat.status === "streaming" || chat.status === "submitted" || !question.trim()}
          >
            Send
          </button>
        </span>
      </form>
    </div>
  );
}
