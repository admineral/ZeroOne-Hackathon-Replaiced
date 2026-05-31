const MIN = 10;
const MAX = 10_000_000;

function clamp(n: number): number {
  if (!Number.isFinite(n)) return MIN;
  return Math.max(MIN, Math.min(MAX, Math.round(n)));
}

function stepFor(value: number): number {
  if (value < 100) return 10;
  if (value < 10_000) return 100;
  return 1000;
}

interface Props {
  value: number;
  onChange: (value: number) => void;
}

export function VariantsPicker({ value, onChange }: Props) {
  const step = stepFor(value);
  const safe = clamp(value);

  return (
    <div className="variants-picker">
      <button
        type="button"
        className="btn sm ghost stepper-btn"
        aria-label={`Decrease by ${step}`}
        disabled={safe <= MIN}
        onClick={() => onChange(clamp(safe - step))}
      >
        −
      </button>
      <input
        className="variants-input"
        type="number"
        value={safe}
        min={MIN}
        max={MAX}
        aria-label="Variants per family"
        onChange={(e) => onChange(clamp(Number(e.target.value)))}
      />
      <button
        type="button"
        className="btn sm ghost stepper-btn"
        aria-label={`Increase by ${step}`}
        disabled={safe >= MAX}
        onClick={() => onChange(clamp(safe + step))}
      >
        +
      </button>
    </div>
  );
}
