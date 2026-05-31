import { Fragment } from "react";

export type StepStatus = "idle" | "running" | "done" | "error";

export interface Step {
  id: string;
  title: string;
  sub: string;
  status: StepStatus;
  busy?: boolean;
  disabled?: boolean;
  onRun: () => void;
  /** Overrides the default "Run" label for action steps. */
  actionLabel?: string;
  /** Opens a page on the right — the whole card is clickable (no Run button). */
  nav?: boolean;
  /** Highlight this step as the currently open page. */
  active?: boolean;
}

interface Props {
  steps: Step[];
  overview?: {
    active: boolean;
    onClick: () => void;
    /** Render the overview/home button inline after the step with this id
     *  instead of at the top of the rail. */
    afterStepId?: string;
  };
}

function dotContent(status: StepStatus, displayNumber: number, busy?: boolean): string {
  if (busy) return "";
  if (status === "done") return "\u2713";
  if (status === "error") return "!";
  return String(displayNumber);
}

function StepBody({ step }: { step: Step }) {
  return (
    <div className="step-body">
      <div className="step-title">{step.title}</div>
      {step.sub ? <div className="step-sub">{step.sub}</div> : null}
    </div>
  );
}

function OverviewButton({
  overview,
  lineBelow,
  number,
}: {
  overview: NonNullable<Props["overview"]>;
  lineBelow?: boolean;
  number?: number;
}) {
  return (
    <button
      type="button"
      className={`rail-step nav${overview.active ? " active" : ""}`}
      onClick={overview.onClick}
    >
      {lineBelow && <span className="rail-line" />}
      <div className="dot">{number != null ? String(number) : ""}</div>
      <div className="step-body">
        <div className="step-title">{"Train"}</div>
      </div>
    </button>
  );
}

export function PipelineRail({ steps, overview }: Props) {
  const overviewInline = overview?.afterStepId != null;
  // Index of the step the inline overview button is rendered after, so the
  // rail can number every item continuously (steps below it shift up by one).
  const overviewAfterIdx =
    overview && overviewInline
      ? steps.findIndex((s) => s.id === overview.afterStepId)
      : -1;
  return (
    <div className="rail">
      {overview && !overviewInline && (
        <>
          <OverviewButton overview={overview} />
          <div className="rail-divider" role="separator" />
        </>
      )}

      {steps.map((step, idx) => {
        const lineBelow = idx < steps.length - 1;
        const displayNumber =
          overviewAfterIdx >= 0 && idx > overviewAfterIdx ? idx + 2 : idx + 1;
        const dot = (
          <div className="dot">
            {step.busy ? (
              <span className="spinner" />
            ) : (
              dotContent(step.status, displayNumber)
            )}
          </div>
        );

        const stepNode = step.nav ? (
          <button
            type="button"
            className={`rail-step nav ${step.status}${step.active ? " active" : ""}`}
            onClick={step.onRun}
          >
            {lineBelow && <span className="rail-line" />}
            {dot}
            <StepBody step={step} />
          </button>
        ) : (
          <div className={`rail-step ${step.status}`}>
            {lineBelow && <span className="rail-line" />}
            {dot}
            <div className="step-body">
              <div className="step-title">{step.title}</div>
              {step.sub ? <div className="step-sub">{step.sub}</div> : null}
              <button
                className="btn sm ghost"
                style={{ marginTop: 8 }}
                onClick={step.onRun}
                disabled={step.busy || step.disabled}
              >
                {step.busy ? "Running…" : step.actionLabel ?? "Run"}
              </button>
            </div>
          </div>
        );

        const showOverviewAfter =
          overview && overviewInline && overview.afterStepId === step.id;

        return (
          <Fragment key={step.id}>
            {stepNode}
            {showOverviewAfter && (
              <OverviewButton
                overview={overview}
                lineBelow={lineBelow}
                number={overviewAfterIdx + 2}
              />
            )}
          </Fragment>
        );
      })}
    </div>
  );
}
