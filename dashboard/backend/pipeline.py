"""Registry of pipeline runs and their remote file locations.

Every action the dashboard can trigger maps to an existing Slurm script plus
the output files it writes. Keeping this in one place means the backend and the
frontend agree on where to poll for logs, loss curves and results.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    slurm: str
    out: str
    err: str
    log: str | None = None
    model: str | None = None
    vocab: str | None = None
    summary: str | None = None
    rule_counts: str | None = None
    split_summary: str | None = None
    gpu_timeline: str | None = None


RUNS: dict[str, RunSpec] = {
    "transformer": RunSpec(
        key="transformer",
        label="Transformer training",
        slurm="training_data/run_train_transformer.slurm",
        out="outputs/transformer/train_transformer.out",
        err="outputs/transformer/train_transformer.err",
        log="outputs/transformer/train_log.csv",
        model="outputs/transformer/transformer_model.pt",
        vocab="outputs/transformer/vocab.json",
        split_summary="outputs/transformer/split_summary.json",
        gpu_timeline="outputs/transformer/gpu_timeline.csv",
    ),
    "ngram": RunSpec(
        key="ngram",
        label="N-gram baseline",
        slurm="training_data/run_baseline_ngram.slurm",
        out="outputs/baseline_ngram.out",
        err="outputs/baseline_ngram.err",
    ),
    "eval_transformer": RunSpec(
        key="eval_transformer",
        label="Rule evaluation (Transformer)",
        slurm="training_data/run_evaluate_transformer_rules.slurm",
        out="outputs/transformer/rule_eval.out",
        err="outputs/transformer/rule_eval.err",
        summary="outputs/transformer/rule_eval_summary.csv",
        rule_counts="outputs/transformer/generated_rule_counts.csv",
        split_summary="outputs/transformer/split_summary.json",
    ),
    "generate_remote": RunSpec(
        key="generate_remote",
        label="Generate dataset (Leonardo)",
        slurm="training_data/run_generate_data.slurm",
        out="outputs/generate_data.out",
        err="outputs/generate_data.err",
    ),
    "preprocess": RunSpec(
        key="preprocess",
        label="Preprocess dataset (pack)",
        slurm="training_data/run_preprocess_dataset.slurm",
        out="outputs/preprocess.out",
        err="outputs/preprocess.err",
    ),
    "submission": RunSpec(
        key="submission",
        label="Generate submission (Leonardo)",
        slurm="training_data/run_make_submission.slurm",
        out="outputs/transformer/make_submission.out",
        err="outputs/transformer/make_submission.err",
    ),
}


def get_run(key: str) -> RunSpec:
    if key not in RUNS:
        raise KeyError(f"Unknown run '{key}'. Known: {', '.join(RUNS)}")
    return RUNS[key]
