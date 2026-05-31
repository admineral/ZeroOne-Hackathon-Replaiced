import {
  useCallback,
  useEffect,
  useRef,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";

const MIN_SEG = 1;
const MAX_PCT = 100;

type Boundary = "train" | "val";

interface Props {
  /** Train percentage (1..98). */
  train: number;
  /** Validation percentage (1..98). Test is derived as 100 - train - val. */
  val: number;
  onChange: (train: number, val: number) => void;
}

/**
 * Draggable three-segment data-split slider: blue = train, orange =
 * validation, green = test. Two handles set the train|val and val|test
 * boundaries; every segment is kept at >= 1%.
 */
export function SplitSlider({ train, val, onChange }: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef<Boundary | null>(null);
  // Latest values so the window listeners never read a stale closure.
  const valuesRef = useRef({ train, val });
  valuesRef.current = { train, val };

  const test = Math.max(MIN_SEG, MAX_PCT - train - val);
  const b1 = train; // train | val boundary
  const b2 = train + val; // val | test boundary

  // Move a boundary to an absolute percentage position, clamping so each
  // segment stays at least MIN_SEG wide.
  const setBoundary = useCallback(
    (which: Boundary, pos: number) => {
      const { train: t, val: v } = valuesRef.current;
      const clamped = Math.round(Math.min(MAX_PCT, Math.max(0, pos)));
      if (which === "train") {
        const boundary = t + v; // keep val|test fixed; train and val trade
        const next = Math.min(Math.max(clamped, MIN_SEG), boundary - MIN_SEG);
        onChange(next, boundary - next);
      } else {
        const next = Math.min(Math.max(clamped, t + MIN_SEG), MAX_PCT - MIN_SEG);
        onChange(t, next - t);
      }
    },
    [onChange]
  );

  const posFromClientX = useCallback((clientX: number) => {
    const el = trackRef.current;
    if (!el) return 0;
    const rect = el.getBoundingClientRect();
    return ((clientX - rect.left) / rect.width) * MAX_PCT;
  }, []);

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      if (!draggingRef.current) return;
      e.preventDefault();
      setBoundary(draggingRef.current, posFromClientX(e.clientX));
    };
    const onUp = () => {
      draggingRef.current = null;
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [setBoundary, posFromClientX]);

  const startDrag = (which: Boundary) => (e: ReactPointerEvent) => {
    e.preventDefault();
    (e.target as HTMLElement).focus();
    draggingRef.current = which;
  };

  const onKey = (which: Boundary) => (e: ReactKeyboardEvent) => {
    const step = e.shiftKey ? 5 : 1;
    const current = which === "train" ? b1 : b2;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
      e.preventDefault();
      setBoundary(which, current - step);
    } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
      e.preventDefault();
      setBoundary(which, current + step);
    }
  };

  return (
    <div className="split-slider-wrap">
      <div className="split-track" ref={trackRef}>
        <div className="split-seg seg-train" style={{ width: `${train}%` }}>
          <span className="split-seg-label">{train}%</span>
        </div>
        <div className="split-seg seg-val" style={{ width: `${val}%` }}>
          <span className="split-seg-label">{val}%</span>
        </div>
        <div className="split-seg seg-test" style={{ width: `${test}%` }}>
          <span className="split-seg-label">{test}%</span>
        </div>

        <div
          className="split-handle"
          style={{ left: `${b1}%` }}
          role="slider"
          tabIndex={0}
          aria-label="Train / validation boundary"
          aria-valuemin={MIN_SEG}
          aria-valuemax={b2 - MIN_SEG}
          aria-valuenow={b1}
          onPointerDown={startDrag("train")}
          onKeyDown={onKey("train")}
        />
        <div
          className="split-handle"
          style={{ left: `${b2}%` }}
          role="slider"
          tabIndex={0}
          aria-label="Validation / test boundary"
          aria-valuemin={b1 + MIN_SEG}
          aria-valuemax={MAX_PCT - MIN_SEG}
          aria-valuenow={b2}
          onPointerDown={startDrag("val")}
          onKeyDown={onKey("val")}
        />
      </div>
    </div>
  );
}
