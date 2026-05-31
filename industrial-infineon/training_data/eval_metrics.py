#!/usr/bin/env python3
"""Pure-Python quality metrics for generated process sequences.

These complement the 10-rule validator so the evaluation is harder to game.
Rule-validity alone is necessary but not sufficient: a model can score a perfect
valid-rate by emitting trivial output (immediate EOS, looping a safe step,
truncated recipes). The metrics here catch exactly those degenerate cases by
comparing each generated completion against the real held-out continuation and
against the known structure of real recipes.

No torch / heavy deps so this module can be unit-tested and reused anywhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Calibrated against the real variant datasets:
#   lengths: MOSFET 117-134, IGBT 139-155, IC 107-122
#   max consecutive repeated step: 1 (real recipes never repeat a step in a row)
QUALITY_LEN_LOW = 0.8
QUALITY_LEN_HIGH = 1.25
QUALITY_MAX_REPEAT = 2  # real data is 1; allow one accidental repeat
QUALITY_MIN_ACC = 0.5


def max_consecutive_repeat(steps: list[str]) -> int:
    """Longest run of identical consecutive steps (degeneracy signal)."""
    best = 0
    run = 0
    prev: str | None = None
    for step in steps:
        run = run + 1 if step == prev else 1
        prev = step
        best = max(best, run)
    return best


def suffix_accuracy(generated_cont: list[str], real_cont: list[str]) -> float:
    """Position-wise token accuracy of the generated continuation vs the real
    one. Denominator is the real length, so short/truncated completions are
    penalized (missing positions count as misses)."""
    if not real_cont:
        return 1.0
    overlap = min(len(generated_cont), len(real_cont))
    matches = sum(1 for i in range(overlap) if generated_cont[i] == real_cont[i])
    return matches / len(real_cont)


def jaccard(a: list[str], b: list[str]) -> float:
    """Jaccard similarity of the step-type sets (vocabulary overlap)."""
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


@dataclass
class CompletionMetrics:
    valid: bool
    reached_eos: bool
    len_ratio: float
    suffix_acc: float
    jaccard: float
    max_repeat: int
    quality_ok: bool

    def to_dict(self) -> dict:
        return asdict(self)


def completion_metrics(
    real_steps: list[str],
    generated_steps: list[str],
    cut: int,
    valid: bool,
    reached_eos: bool,
) -> CompletionMetrics:
    """Compute all quality signals plus the composite quality_ok flag.

    A completion is only "quality_ok" when it is rule-valid AND actually finishes
    (EOS) AND has a plausible length AND does not loop AND reproduces a
    meaningful fraction of the true continuation. None of these can be satisfied
    by trivial/degenerate output.
    """
    real_cont = real_steps[cut:]
    gen_cont = generated_steps[cut:]
    len_ratio = len(generated_steps) / max(len(real_steps), 1)
    acc = suffix_accuracy(gen_cont, real_cont)
    jac = jaccard(gen_cont, real_cont)
    mrep = max_consecutive_repeat(generated_steps)
    quality_ok = bool(
        valid
        and reached_eos
        and QUALITY_LEN_LOW <= len_ratio <= QUALITY_LEN_HIGH
        and mrep <= QUALITY_MAX_REPEAT
        and acc >= QUALITY_MIN_ACC
    )
    return CompletionMetrics(
        valid=valid,
        reached_eos=reached_eos,
        len_ratio=len_ratio,
        suffix_acc=acc,
        jaccard=jac,
        max_repeat=mrep,
        quality_ok=quality_ok,
    )
