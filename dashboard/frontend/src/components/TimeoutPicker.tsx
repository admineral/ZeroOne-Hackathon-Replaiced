const MIN_HOURS = 1;
const MAX_HOURS = 10;

function parseToMinutes(value: string): number {
  const m = /^(\d{1,2}):(\d{2})(?::(\d{2}))?$/.exec(value.trim());
  if (!m) return 2 * 60;
  return parseInt(m[1], 10) * 60 + parseInt(m[2], 10);
}

function formatFromHours(hours: number): string {
  return `${String(hours).padStart(2, "0")}:00:00`;
}

function currentHours(value: string): number {
  return Math.floor(parseToMinutes(value) / 60);
}

function stepHours(value: string, deltaHours: number): string {
  const next = Math.max(MIN_HOURS, Math.min(MAX_HOURS, currentHours(value) + deltaHours));
  return formatFromHours(next);
}

interface Props {
  value: string;
  onChange: (value: string) => void;
}

export function TimeoutPicker({ value, onChange }: Props) {
  const hours = currentHours(value);

  return (
    <div className="timeout-picker">
      <button
        type="button"
        className="btn sm ghost stepper-btn"
        aria-label="Decrease by 1 hour"
        disabled={hours <= MIN_HOURS}
        onClick={() => onChange(stepHours(value, -1))}
      >
        −
      </button>
      <span className="timeout-value">
        {hours} {hours === 1 ? "hour" : "hours"}
      </span>
      <button
        type="button"
        className="btn sm ghost stepper-btn"
        aria-label="Increase by 1 hour"
        disabled={hours >= MAX_HOURS}
        onClick={() => onChange(stepHours(value, 1))}
      >
        +
      </button>
    </div>
  );
}

export const DEFAULT_GEN_TIMEOUT = "02:00:00";
