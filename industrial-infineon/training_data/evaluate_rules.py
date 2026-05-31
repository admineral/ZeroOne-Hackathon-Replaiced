#!/usr/bin/env python3
"""Rule-aware evaluation for source and model-generated process sequences."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch

from eval_metrics import (
    QUALITY_LEN_HIGH,
    QUALITY_LEN_LOW,
    QUALITY_MAX_REPEAT,
    QUALITY_MIN_ACC,
    CompletionMetrics,
    completion_metrics,
)
from generate_sequences import validate_sequence
from sequence_data import (
    BOS_TOKEN,
    EOS_TOKEN,
    FAMILY_TOKENS,
    FAMILY_UNKNOWN_TOKEN,
    SequenceExample,
    decode_example,
    ensure_dataset_available,
    has_packed_dataset,
    load_examples,
    load_packed,
    load_vocab,
    packed_dir_for,
    split_examples,
    split_indices,
    strip_model_tokens,
)
from train_transformer import TinyCausalTransformer


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Load a Transformer checkpoint produced by the training script."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    metadata = checkpoint["metadata"]
    model_type = metadata["model_type"]

    if model_type == "transformer":
        model = TinyCausalTransformer(
            vocab_size=metadata["vocab_size"],
            pad_id=metadata["pad_id"],
            max_seq_len=metadata["max_seq_len"],
            d_model=metadata["d_model"],
            num_layers=metadata["num_layers"],
            num_heads=metadata["num_heads"],
            dropout=metadata["dropout"],
        )
    else:
        raise ValueError(f"Unsupported model_type in checkpoint: {model_type}")

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, metadata


def greedy_generate(
    model: torch.nn.Module,
    prefix_tokens: list[str],
    token_to_id: dict[str, int],
    id_to_token: list[str],
    device: torch.device,
    max_seq_len: int,
) -> tuple[list[str], bool]:
    """Generate tokens greedily until EOS or max_seq_len.

    Returns (tokens, reached_eos) so callers can tell a real completion apart
    from one that ran out of budget (a sign of a runaway / non-terminating
    generation)."""
    ids = [token_to_id[token] for token in prefix_tokens]
    eos_id = token_to_id[EOS_TOKEN]
    with torch.no_grad():
        while len(ids) < max_seq_len and ids[-1] != eos_id:
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(input_ids)
            next_id = int(torch.argmax(logits[0, -1]).item())
            ids.append(next_id)
            if next_id == eos_id:
                break
    reached_eos = bool(ids and ids[-1] == eos_id)
    return [id_to_token[idx] for idx in ids], reached_eos


def summarize_violations(sequences: list[list[str]]) -> tuple[int, Counter[str]]:
    """Return invalid sequence count and violation-rule counts."""
    invalid = 0
    rule_counts: Counter[str] = Counter()
    for steps in sequences:
        violations = validate_sequence(steps)
        if violations:
            invalid += 1
            rule_counts.update(v.rule for v in violations)
    return invalid, rule_counts


SUMMARY_FIELDS = [
    "source",
    "completion_fraction",
    "sequences",
    "invalid_sequences",
    "valid_rate",
    "quality_rate",
    "mean_suffix_acc",
    "mean_len_ratio",
    "eos_rate",
    "mean_jaccard",
]


def write_summary(
    output_path: Path,
    rows: list[dict[str, str | int | float]],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_metrics(
    source: str,
    fraction: str | float,
    metrics: list[CompletionMetrics],
) -> dict[str, str | int | float]:
    """Roll up per-completion metrics into one summary row."""
    n = len(metrics)
    invalid = sum(1 for m in metrics if not m.valid)
    quality = sum(1 for m in metrics if m.quality_ok)
    return {
        "source": source,
        "completion_fraction": fraction,
        "sequences": n,
        "invalid_sequences": invalid,
        "valid_rate": f"{1.0 - invalid / max(n, 1):.6f}",
        "quality_rate": f"{quality / max(n, 1):.6f}",
        "mean_suffix_acc": f"{_mean([m.suffix_acc for m in metrics]):.6f}",
        "mean_len_ratio": f"{_mean([m.len_ratio for m in metrics]):.6f}",
        "eos_rate": f"{_mean([1.0 if m.reached_eos else 0.0 for m in metrics]):.6f}",
        "mean_jaccard": f"{_mean([m.jaccard for m in metrics]):.6f}",
    }


def write_rule_counts(output_path: Path, source: str, rule_counts: Counter[str]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "rule", "count"])
        writer.writeheader()
        for rule, count in rule_counts.most_common():
            writer.writerow({"source": source, "rule": rule, "count": count})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("training_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/transformer/transformer_model.pt"))
    parser.add_argument("--vocab", type=Path, default=Path("outputs/transformer/vocab.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fallback train fraction if the checkpoint doesn't record one.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fallback val fraction if the checkpoint doesn't record one.",
    )
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="Fallback per-family sequence cap (0 = all) if the checkpoint "
        "didn't record one. The checkpoint's value takes precedence so the "
        "held-out split matches training.",
    )
    parser.add_argument("--completion-fractions", type=float, nargs="+", default=[0.6, 0.8])
    return parser.parse_args()


def resolve_split(args: argparse.Namespace) -> tuple[float, float, int, "int | None"]:
    """Prefer the checkpoint's recorded split ratios/seed/cap so evaluation uses
    the exact held-out set training reserved (no train/val leakage). Falls back
    to the CLI args for older checkpoints that didn't record them."""
    train_ratio, val_ratio, seed = args.train_ratio, args.val_ratio, args.seed
    max_sequences = int(args.max_sequences) or None
    if args.checkpoint.exists():
        try:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            meta = ckpt.get("metadata", {}) if isinstance(ckpt, dict) else {}
            train_ratio = float(meta.get("train_ratio", train_ratio))
            val_ratio = float(meta.get("val_ratio", val_ratio))
            seed = int(meta.get("seed", seed))
            # The cap controls dataset membership, so it must match training's
            # for the val split to line up. Only override when present.
            if "max_sequences" in meta:
                cap = meta.get("max_sequences")
                max_sequences = int(cap) if cap else None
        except Exception:
            pass
    return train_ratio, val_ratio, seed, max_sequences


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ratio, val_ratio, seed, max_sequences = resolve_split(args)

    # Fail fast with a clear message if the chosen --data-dir holds neither a
    # packed dataset nor the raw CSVs (otherwise the legacy path dies deep in
    # pathlib on e.g. training_data/MOSFET_variants.csv).
    ensure_dataset_available(args.data_dir)

    # Phase 2: if the dataset is packed, reproduce the exact index split training
    # used and decode ONLY the sampled held-out sequences (RAM ~0). Otherwise use
    # the legacy in-memory path.
    if has_packed_dataset(args.data_dir):
        packed = load_packed(packed_dir_for(args.data_dir))
        _train_idx, val_idx, _test_idx = split_indices(
            packed.num_sequences, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )
        n_train, n_val, n_test = len(_train_idx), len(val_idx), len(_test_idx)
        sampled = []
        for idx in val_idx[: args.max_examples]:
            idx = int(idx)
            steps = strip_model_tokens(decode_example(packed.ids_of(idx), packed.id_to_token))
            sampled.append(
                SequenceExample(
                    family=packed.family_of(idx), sequence_id=str(idx), steps=steps
                )
            )
        source = "packed memmap"
    else:
        examples = load_examples(args.data_dir, max_per_family=max_sequences)
        train_examples, val_examples, test_examples = split_examples(
            examples, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )
        n_train, n_val, n_test = len(train_examples), len(val_examples), len(test_examples)
        sampled = val_examples[: args.max_examples]
        source = "in-memory CSV"
    print(
        f"split [{source}]: train_ratio={train_ratio} val_ratio={val_ratio} seed={seed} "
        f"max_sequences={max_sequences} -> "
        f"train={n_train} val={n_val} test={n_test}"
    )
    # Record the split so the dashboard's results card always reflects the data
    # used for the checkpoint under evaluation (self-contained per eval run).
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(
            {
                "train": n_train,
                "validation": n_val,
                "test": n_test,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": round(1.0 - train_ratio - val_ratio, 6),
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows: list[dict[str, str | int | float]] = []
    source_sequences = [example.steps for example in sampled]
    source_invalid, source_rule_counts = summarize_violations(source_sequences)
    # Real held-out recipes are complete and terminate, so the quality signals
    # are trivially satisfied — this row is the sanity anchor (~1.0 everywhere).
    n_source = len(source_sequences)
    rows.append(
        {
            "source": "heldout_source",
            "completion_fraction": "full",
            "sequences": n_source,
            "invalid_sequences": source_invalid,
            "valid_rate": f"{1.0 - source_invalid / max(n_source, 1):.6f}",
            "quality_rate": f"{1.0 - source_invalid / max(n_source, 1):.6f}",
            "mean_suffix_acc": "1.000000",
            "mean_len_ratio": "1.000000",
            "eos_rate": "1.000000",
            "mean_jaccard": "1.000000",
        }
    )

    generated_rule_counts: Counter[str] = Counter()
    if args.checkpoint.exists() and args.vocab.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, metadata = load_checkpoint_model(args.checkpoint, device)
        token_to_id, id_to_token = load_vocab(args.vocab)
        max_seq_len = int(metadata["max_seq_len"])

        print(f"device: {device}")
        if device.type == "cuda":
            print(f"gpu: {torch.cuda.get_device_name(0)}")
        print(f"checkpoint: {args.checkpoint}")

        def run_pass(label, family_token_fn, rule_counter):
            """Evaluate the model over all fractions using the family token
            returned by ``family_token_fn(example)``. Pass a Counter to tally
            rule violations, or None to skip tallying (e.g. the unknown pass)."""
            pass_rows = []
            for fraction in args.completion_fractions:
                metrics: list[CompletionMetrics] = []
                skipped = 0
                for example in sampled:
                    cut = max(1, int(len(example.steps) * fraction))
                    prefix_tokens = [
                        BOS_TOKEN,
                        family_token_fn(example),
                        *example.steps[:cut],
                    ]
                    if any(token not in token_to_id for token in prefix_tokens):
                        skipped += 1
                        continue
                    generated_tokens, reached_eos = greedy_generate(
                        model=model,
                        prefix_tokens=prefix_tokens,
                        token_to_id=token_to_id,
                        id_to_token=id_to_token,
                        device=device,
                        max_seq_len=max_seq_len,
                    )
                    generated_steps = strip_model_tokens(generated_tokens)
                    violations = validate_sequence(generated_steps)
                    if violations and rule_counter is not None:
                        rule_counter.update(v.rule for v in violations)
                    metrics.append(
                        completion_metrics(
                            real_steps=example.steps,
                            generated_steps=generated_steps,
                            cut=cut,
                            valid=not violations,
                            reached_eos=reached_eos,
                        )
                    )

                row = aggregate_metrics(label, fraction, metrics)
                pass_rows.append(row)
                print(
                    f"[{label}] fraction={fraction}: generated={len(metrics)} "
                    f"skipped={skipped} valid_rate={row['valid_rate']} "
                    f"quality_rate={row['quality_rate']} "
                    f"mean_acc={row['mean_suffix_acc']} eos_rate={row['eos_rate']}"
                )
            return pass_rows

        # Pass 1: real family prefix (the conditioned model).
        rows += run_pass(
            "model_generated",
            lambda e: FAMILY_TOKENS[e.family],
            generated_rule_counts,
        )
        # Pass 2: unknown/no-prefix — proves the model still produces valid
        # recipes when the family is missing or an unseen product category.
        rows += run_pass(
            "model_generated_unknown",
            lambda e: FAMILY_UNKNOWN_TOKEN,
            None,
        )
    else:
        print(
            "No checkpoint/vocab found; wrote source-data validation only. "
            "Run train_transformer.py first for generated-sequence evaluation."
        )

    write_summary(args.output_dir / "rule_eval_summary.csv", rows)
    write_rule_counts(args.output_dir / "source_rule_counts.csv", "heldout_source", source_rule_counts)
    write_rule_counts(
        args.output_dir / "generated_rule_counts.csv",
        "model_generated",
        generated_rule_counts,
    )
    (args.output_dir / "quality_summary.json").write_text(
        json.dumps(
            {
                "thresholds": {
                    "len_ratio_low": QUALITY_LEN_LOW,
                    "len_ratio_high": QUALITY_LEN_HIGH,
                    "max_consecutive_repeat": QUALITY_MAX_REPEAT,
                    "min_suffix_accuracy": QUALITY_MIN_ACC,
                },
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

