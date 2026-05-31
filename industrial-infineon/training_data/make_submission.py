#!/usr/bin/env python3
"""Generate organizer submission CSVs from the official participant eval inputs.

Tasks:
  1. next-step: rank the 5 most likely next process steps.
  2. completion: generate the remaining process steps after the provided cut.
  3. anomaly: classify full sequences with the rule validator.

Task 3 is purely rule-based and does not need a checkpoint. Tasks 1/2 require a
trained GRU/Transformer checkpoint plus vocab produced by the training scripts.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from generate_sequences import validate_sequence
from sequence_data import (
    BOS_TOKEN,
    EOS_TOKEN,
    FAMILY_TOKENS,
    FAMILY_UNKNOWN_TOKEN,
    SPECIAL_TOKENS,
    load_vocab,
    strip_model_tokens,
)


DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE_DIR = DEFAULT_PROJECT_DIR.parent
DEFAULT_INPUT_DIR = DEFAULT_WORKSPACE_DIR / "participant_files"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "submission"
DEFAULT_CHECKPOINT = DEFAULT_PROJECT_DIR / "outputs" / "transformer" / "transformer_model.pt"
DEFAULT_VOCAB = DEFAULT_PROJECT_DIR / "outputs" / "transformer" / "vocab.json"


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - torch is optional for the anomaly task.
        return "cpu"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [
            {str(k).strip().lstrip("\ufeff"): (v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def split_steps(text: str) -> list[str]:
    return [step.strip() for step in text.split("|") if step.strip()]


def family_token(family: str, token_to_id: dict[str, int]) -> str:
    token = FAMILY_TOKENS.get(family.lower(), FAMILY_UNKNOWN_TOKEN)
    return token if token in token_to_id else FAMILY_UNKNOWN_TOKEN


def step_tokens(id_to_token: list[str]) -> set[str]:
    blocked = set(SPECIAL_TOKENS) | set(FAMILY_TOKENS.values()) | {FAMILY_UNKNOWN_TOKEN}
    return {token for token in id_to_token if token not in blocked}


def encode_prefix(
    family: str,
    steps: list[str],
    token_to_id: dict[str, int],
    allow_unknown_family: bool = True,
) -> list[int] | None:
    fam = family_token(family, token_to_id)
    tokens = [BOS_TOKEN, fam, *steps]
    if not allow_unknown_family and fam == FAMILY_UNKNOWN_TOKEN:
        return None
    if any(token not in token_to_id for token in tokens):
        return None
    return [token_to_id[token] for token in tokens]


def top_k_next_steps(
    model: torch.nn.Module,
    ids: list[int],
    id_to_token: list[str],
    allowed_steps: set[str],
    device: torch.device,
    k: int = 5,
) -> list[str]:
    import torch

    with torch.no_grad():
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(input_ids)[0, -1]
        ranked_ids = torch.argsort(logits, descending=True).tolist()

    out: list[str] = []
    for idx in ranked_ids:
        token = id_to_token[int(idx)]
        if token in allowed_steps and token not in out:
            out.append(token)
        if len(out) == k:
            break
    while len(out) < k:
        out.append("")
    return out


def greedy_completion(
    model: torch.nn.Module,
    ids: list[int],
    id_to_token: list[str],
    token_to_id: dict[str, int],
    device: torch.device,
    max_seq_len: int,
) -> list[str]:
    import torch

    eos_id = token_to_id[EOS_TOKEN]
    current = list(ids)
    with torch.no_grad():
        while len(current) < max_seq_len and current[-1] != eos_id:
            input_ids = torch.tensor([current], dtype=torch.long, device=device)
            logits = model(input_ids)
            next_id = int(torch.argmax(logits[0, -1]).item())
            current.append(next_id)
            if next_id == eos_id:
                break
    return strip_model_tokens([id_to_token[idx] for idx in current])


def write_next_and_completion(
    input_path: Path,
    output_dir: Path,
    checkpoint_path: Path,
    vocab_path: Path,
    device_name: str,
) -> None:
    if not checkpoint_path.exists() or not vocab_path.exists():
        print(
            "[WARN] Checkpoint/vocab missing; skipped Tasks 1/2. "
            f"checkpoint={checkpoint_path} vocab={vocab_path}",
            file=sys.stderr,
        )
        return

    import torch
    from evaluate_rules import load_checkpoint_model

    device = torch.device(device_name)
    model, metadata = load_checkpoint_model(checkpoint_path, device)
    token_to_id, id_to_token = load_vocab(vocab_path)
    allowed_steps = step_tokens(id_to_token)
    max_seq_len = int(metadata["max_seq_len"])

    rows = read_rows(input_path)
    next_path = output_dir / "predictions_nextstep.csv"
    completion_path = output_dir / "predictions_completion.csv"

    skipped = 0
    with next_path.open("w", newline="", encoding="utf-8") as nf, completion_path.open(
        "w", newline="", encoding="utf-8"
    ) as cf:
        next_writer = csv.DictWriter(
            nf,
            fieldnames=["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"],
        )
        comp_writer = csv.DictWriter(cf, fieldnames=["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        next_writer.writeheader()
        comp_writer.writeheader()

        for row in rows:
            example_id = row["EXAMPLE_ID"]
            partial_steps = split_steps(row["PARTIAL_SEQUENCE"])
            ids = encode_prefix(row["FAMILY"], partial_steps, token_to_id)
            if ids is None:
                skipped += 1
                ranks = [""] * 5
                continuation: list[str] = []
            else:
                ranks = top_k_next_steps(model, ids, id_to_token, allowed_steps, device)
                generated_steps = greedy_completion(
                    model, ids, id_to_token, token_to_id, device, max_seq_len
                )
                continuation = generated_steps[len(partial_steps) :]

            next_writer.writerow(
                {
                    "EXAMPLE_ID": example_id,
                    "RANK_1": ranks[0],
                    "RANK_2": ranks[1],
                    "RANK_3": ranks[2],
                    "RANK_4": ranks[3],
                    "RANK_5": ranks[4],
                }
            )
            comp_writer.writerow(
                {
                    "EXAMPLE_ID": example_id,
                    "PREDICTED_SEQUENCE": "|".join(continuation),
                }
            )

    print(f"Wrote {next_path}")
    print(f"Wrote {completion_path}")
    if skipped:
        print(f"[WARN] Skipped model inference for {skipped} rows with unknown tokens.")


def write_anomaly(input_path: Path, output_dir: Path) -> None:
    rows = read_rows(input_path)
    output_path = output_dir / "predictions_anomaly.csv"
    invalid = 0
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"],
        )
        writer.writeheader()
        for row in rows:
            violations = validate_sequence(split_steps(row["SEQUENCE"]))
            first_rule = violations[0].rule if violations else ""
            is_valid = 0 if violations else 1
            invalid += int(bool(violations))
            writer.writerow(
                {
                    "EXAMPLE_ID": row["EXAMPLE_ID"],
                    "IS_VALID": is_valid,
                    "SCORE": "0.00" if violations else "1.00",
                    "PREDICTED_RULE": first_rule,
                }
            )
    print(f"Wrote {output_path} ({invalid}/{len(rows)} predicted invalid)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["next-step", "completion", "anomaly", "all"],
        default=["all"],
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Torch device for Tasks 1/2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = {"next-step", "completion", "anomaly"} if "all" in args.tasks else set(args.tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    valid_path = args.input_dir / "eval_input_valid.csv"
    anomaly_path = args.input_dir / "eval_input_anomaly.csv"
    if not valid_path.exists():
        raise FileNotFoundError(valid_path)
    if not anomaly_path.exists():
        raise FileNotFoundError(anomaly_path)

    if {"next-step", "completion"} & tasks:
        write_next_and_completion(
            valid_path,
            args.output_dir,
            args.checkpoint,
            args.vocab,
            args.device,
        )
    if "anomaly" in tasks:
        write_anomaly(anomaly_path, args.output_dir)


if __name__ == "__main__":
    main()
