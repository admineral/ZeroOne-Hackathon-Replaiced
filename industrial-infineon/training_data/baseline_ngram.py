#!/usr/bin/env python3
"""N-gram baseline for next-step prediction over process-step tokens."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from sequence_data import load_examples, split_examples, tokens_for_example


def build_ngram_counts(
    token_sequences: list[list[str]],
    n: int,
) -> tuple[dict[tuple[str, ...], Counter[str]], Counter[str]]:
    """Collect next-token counts for contexts up to length n."""
    counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    unigram_next: Counter[str] = Counter()
    for tokens in token_sequences:
        for idx in range(1, len(tokens)):
            target = tokens[idx]
            unigram_next[target] += 1
            max_context = min(n, idx)
            for size in range(1, max_context + 1):
                context = tuple(tokens[idx - size:idx])
                counts[context][target] += 1
    return counts, unigram_next


def ranked_predictions(
    prefix: list[str],
    n: int,
    counts: dict[tuple[str, ...], Counter[str]],
    unigram_next: Counter[str],
    k: int = 5,
) -> list[str]:
    """Return top-k predictions with backoff to shorter contexts."""
    max_context = min(n, len(prefix))
    for size in range(max_context, 0, -1):
        context = tuple(prefix[-size:])
        if context in counts:
            return [token for token, _ in counts[context].most_common(k)]
    return [token for token, _ in unigram_next.most_common(k)]


def evaluate_ngram(
    val_sequences: list[list[str]],
    n: int,
    counts: dict[tuple[str, ...], Counter[str]],
    unigram_next: Counter[str],
) -> dict[str, float]:
    """Evaluate top-k next-step metrics on validation sequences."""
    total = 0
    top1 = 0
    top3 = 0
    top5 = 0
    reciprocal_rank_sum = 0.0

    for tokens in val_sequences:
        for idx in range(1, len(tokens)):
            target = tokens[idx]
            preds = ranked_predictions(tokens[:idx], n, counts, unigram_next, k=5)
            total += 1
            if preds and preds[0] == target:
                top1 += 1
            if target in preds[:3]:
                top3 += 1
            if target in preds[:5]:
                top5 += 1
            if target in preds:
                reciprocal_rank_sum += 1.0 / (preds.index(target) + 1)

    return {
        "n": float(n),
        "examples": float(total),
        "top1_accuracy": top1 / total,
        "top3_accuracy": top3 / total,
        "top5_accuracy": top5 / total,
        "mrr": reciprocal_rank_sum / total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("training_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 5])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.data_dir)
    train_examples, val_examples, test_examples = split_examples(examples, seed=args.seed)
    train_sequences = [tokens_for_example(example) for example in train_examples]
    val_sequences = [tokens_for_example(example) for example in val_examples]

    rows: list[dict[str, float]] = []
    print(
        f"examples: train={len(train_examples)} val={len(val_examples)} "
        f"test={len(test_examples)}"
    )
    for n in args.orders:
        counts, unigram_next = build_ngram_counts(train_sequences, n=n)
        metrics = evaluate_ngram(val_sequences, n=n, counts=counts, unigram_next=unigram_next)
        rows.append(metrics)
        print(
            f"N={n}: top1={metrics['top1_accuracy']:.4f} "
            f"top3={metrics['top3_accuracy']:.4f} "
            f"top5={metrics['top5_accuracy']:.4f} mrr={metrics['mrr']:.4f}"
        )

    output_path = args.output_dir / "ngram_metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["n", "examples", "top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "n": int(row["n"]),
                    "examples": int(row["examples"]),
                    "top1_accuracy": f"{row['top1_accuracy']:.6f}",
                    "top3_accuracy": f"{row['top3_accuracy']:.6f}",
                    "top5_accuracy": f"{row['top5_accuracy']:.6f}",
                    "mrr": f"{row['mrr']:.6f}",
                }
            )


if __name__ == "__main__":
    main()

