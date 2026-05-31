#!/usr/bin/env python3
"""
eval_metrics.py — Official scoring script for the process-sequence benchmark.

Supports three evaluation tasks:

    next-step   Score next-step prediction (Top-1 Acc, Top-3 Acc,
                            Top-5 Acc, MRR).
  completion  Score full sequence completion (Normalized Edit Distance,
                            Exact Match Rate, Token Accuracy, Block-level Accuracy).
    anomaly     Score detection of forbidden-pattern sequences (Accuracy,
                            Precision/Recall/F1, Confusion Matrix, AUC,
                            Rule Attribution Accuracy).

Usage
-----
# Next-step prediction
python eval_metrics.py \\
    --task next-step \\
    --ground-truth eval_set_valid.csv \\
    --predictions predictions_nextstep.csv

# Sequence completion
python eval_metrics.py \\
    --task completion \\
    --ground-truth eval_set_valid.csv \\
    --predictions predictions_completion.csv

# Anomaly detection (forbidden patterns)
python eval_metrics.py \\
    --task anomaly \\
    --ground-truth eval_set_forbidden.csv \\
    --predictions predictions_anomaly.csv \\
    [--valid-supplement eval_set_valid.csv]  # add valid examples as negatives

Prediction file formats
-----------------------
next-step:
    EXAMPLE_ID, RANK_1, RANK_2, RANK_3, RANK_4, RANK_5
    (RANK_1 is your top prediction; RANK_2..5 are the next best)
    Example: valid_0001, GATE OXIDE GROWTH, MEASURE GATE OXIDE THICKNESS, ...

completion:
    EXAMPLE_ID, PREDICTED_SEQUENCE
    (PREDICTED_SEQUENCE is pipe-separated predicted remaining steps)
    Example: valid_0001, GATE OXIDE GROWTH|MEASURE GATE OXIDE THICKNESS|...

anomaly:
    EXAMPLE_ID, IS_VALID, [SCORE], [PREDICTED_RULE]
    IS_VALID: 0 = invalid/forbidden, 1 = valid
    SCORE:    optional float [0,1] where higher = more likely valid (used for AUC)
    PREDICTED_RULE: optional; the rule ID the model thinks was violated
    Example: forbidden_0001, 0, 0.12, RULE_DEP_NO_CLEAN
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Token-level edit distance (Levenshtein, O(m*n) DP)
# ---------------------------------------------------------------------------

def _levenshtein(seq1: list, seq2: list) -> int:
    """Token-level Levenshtein distance (each element is one token)."""
    m, n = len(seq1), len(seq2)
    # dp[j] = edit distance between seq1[:i] and seq2[:j]
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev_row = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev_row[j - 1]
            else:
                dp[j] = 1 + min(prev_row[j], dp[j - 1], prev_row[j - 1])
    return dp[n]


def normalized_edit_distance(pred: list[str], ref: list[str]) -> float:
    """
    Token-level edit distance normalized by max(|pred|, |ref|).
    Returns 0.0 for two empty sequences; 1.0 for maximally different.
    """
    if not pred and not ref:
        return 0.0
    return _levenshtein(pred, ref) / max(len(pred), len(ref))


def token_accuracy(pred: list[str], ref: list[str]) -> float:
    """Fraction of positions (up to min length) where pred matches ref."""
    n = min(len(pred), len(ref))
    if n == 0:
        return 0.0
    return sum(p == r for p, r in zip(pred, ref)) / n


# ---------------------------------------------------------------------------
# ROC-AUC (no external dependencies)
# ---------------------------------------------------------------------------

def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0

def _roc_auc(labels: list[int], scores: list[float]) -> float:
    """
    Compute ROC-AUC via pairwise concordance.
    labels: 1 = positive (valid), 0 = negative (invalid).
    scores: higher score = more likely positive.
    """
    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]
    if not pos_scores or not neg_scores:
        return float("nan")
    concordant = sum(p > n for p in pos_scores for n in neg_scores)
    tied       = sum(p == n for p in pos_scores for n in neg_scores)
    total = len(pos_scores) * len(neg_scores)
    return (concordant + 0.5 * tied) / total


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def _major_block(step: str) -> str:
    """
    Map a process step to a coarse major process block.
    This is used for block-level completion scoring.
    """
    s = step.upper()
    if "LITHO" in s or s.startswith("SPIN COAT PHOTORESIST") or "MASK LEVEL" in s:
        return "LITHO"
    if "ETCH" in s or s.startswith("OPEN PAD WINDOW"):
        return "ETCH"
    if "IMPLANT" in s or "ANNEAL" in s or "DIFFUSION" in s:
        return "DOPING_THERMAL"
    if s.startswith("DEPOSIT") or "OXIDATION" in s or "GROWTH" in s:
        return "DEPOSITION"
    if s.startswith("CMP") or "PLANAR" in s:
        return "PLANARIZATION"
    if "VIA" in s:
        return "VIA"
    if "PASSIVATION" in s:
        return "PASSIVATION"
    if "BACKSIDE" in s or "GRIND" in s:
        return "BACKSIDE"
    if "TEST" in s or "MEASURE" in s or "INSPECT" in s or "ANALYSIS" in s:
        return "METROLOGY_TEST"
    if "LOT" in s or "RELEASE" in s or "SHIP" in s:
        return "LOGISTICS"
    return "OTHER"


def _block_signature(seq: list[str]) -> list[str]:
    """Collapse a token sequence to de-duplicated major-process blocks."""
    sig: list[str] = []
    prev: Optional[str] = None
    for step in seq:
        b = _major_block(step)
        if b != prev:
            sig.append(b)
            prev = b
    return sig


def block_level_accuracy(pred: list[str], ref: list[str]) -> float:
    """
    Accuracy over coarse major-process blocks.
    Compares collapsed block signatures position-wise up to min length.
    """
    return token_accuracy(_block_signature(pred), _block_signature(ref))


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _norm_key(h: str) -> str:
    return h.strip().lstrip("\ufeff").strip('"')


def _read_csv_norm(path: Path) -> list[dict]:
    """Read CSV with normalised (BOM/quote-stripped) column names."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        raw_fields = reader.fieldnames or []
        norm = {_norm_key(h): h for h in raw_fields}
        rows = []
        for row in reader:
            rows.append({_norm_key(k): v.strip().strip('"') for k, v in row.items()})
    return rows


# ---------------------------------------------------------------------------
# Task: next-step prediction
# ---------------------------------------------------------------------------

def _score_next_step(gt_path: Path, pred_path: Path) -> None:
    gt_rows   = _read_csv_norm(gt_path)
    pred_rows = _read_csv_norm(pred_path)

    # Ground truth index: EXAMPLE_ID → NEXT_STEP, FAMILY, COMPLETION_FRACTION
    gt = {}
    for r in gt_rows:
        gt[r["EXAMPLE_ID"]] = {
            "next_step":  r["NEXT_STEP"],
            "family":     r["FAMILY"],
            "fraction":   r["COMPLETION_FRACTION"],
        }

    # Predictions index: EXAMPLE_ID → [RANK_1, ..., RANK_5]
    pred = {}
    rank_cols = ["RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"]
    for r in pred_rows:
        eid = r.get("EXAMPLE_ID", "")
        ranks = [r.get(c, "").strip() for c in rank_cols if r.get(c, "").strip()]
        pred[eid] = ranks

    matched = [eid for eid in gt if eid in pred]
    unmatched = len(gt) - len(matched)

    if not matched:
        print("[ERROR] No matching EXAMPLE_IDs between ground truth and predictions.")
        sys.exit(1)

    top1_hits = defaultdict(list)   # (family, fraction) -> [bool, ...]
    top3_hits = defaultdict(list)
    top5_hits = defaultdict(list)
    mrr_vals  = defaultdict(list)

    for eid in matched:
        info      = gt[eid]
        truth     = info["next_step"]
        ranks     = pred[eid]
        key_fam   = info["family"]
        key_frac  = info["fraction"]

        hit1 = bool(ranks) and ranks[0] == truth
        hit3 = truth in ranks[:3]
        hit5 = truth in ranks
        if truth in ranks:
            rr = 1.0 / (ranks.index(truth) + 1)
        else:
            rr = 0.0

        top1_hits[("ALL", "ALL")].append(hit1)
        top3_hits[("ALL", "ALL")].append(hit3)
        top5_hits[("ALL", "ALL")].append(hit5)
        mrr_vals[("ALL", "ALL")].append(rr)
        top1_hits[(key_fam, "ALL")].append(hit1)
        top3_hits[(key_fam, "ALL")].append(hit3)
        top5_hits[(key_fam, "ALL")].append(hit5)
        mrr_vals[(key_fam, "ALL")].append(rr)
        top1_hits[("ALL", key_frac)].append(hit1)
        top3_hits[("ALL", key_frac)].append(hit3)
        top5_hits[("ALL", key_frac)].append(hit5)
        mrr_vals[("ALL", key_frac)].append(rr)

    def _acc(hits):
        return sum(hits) / len(hits) if hits else float("nan")

    def _mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    sep = "=" * 60
    print(f"\n{sep}")
    print("EVAL RESULTS — NEXT-STEP PREDICTION")
    print(sep)
    print(f"Evaluated : {len(matched)}/{len(gt)} examples"
          + (f"  ({unmatched} unmatched EXAMPLE_IDs)" if unmatched else ""))
    print()

    all_t1 = _acc(top1_hits[("ALL", "ALL")])
    all_t3 = _acc(top3_hits[("ALL", "ALL")])
    all_t5 = _acc(top5_hits[("ALL", "ALL")])
    all_mrr = _mean(mrr_vals[("ALL", "ALL")])
    print(f"Top-1 Accuracy : {all_t1:.4f}  ({all_t1*100:.1f}%)")
    print(f"Top-3 Accuracy : {all_t3:.4f}  ({all_t3*100:.1f}%)")
    print(f"Top-5 Accuracy : {all_t5:.4f}  ({all_t5*100:.1f}%)")
    print(f"MRR            : {all_mrr:.4f}")

    # Per-family breakdown
    families = sorted({gt[e]["family"] for e in matched})
    if len(families) > 1:
        print("\nBy family:")
        for fam in families:
            t1 = _acc(top1_hits[(fam, "ALL")])
            t3 = _acc(top3_hits[(fam, "ALL")])
            t5 = _acc(top5_hits[(fam, "ALL")])
            mrr = _mean(mrr_vals[(fam, "ALL")])
            print(f"  {fam:<8s}  Top-1: {t1:.4f}  Top-3: {t3:.4f}  Top-5: {t5:.4f}  MRR: {mrr:.4f}")

    # Per-fraction breakdown
    fractions = sorted({gt[e]["fraction"] for e in matched})
    if len(fractions) > 1:
        print("\nBy completion fraction:")
        for frac in fractions:
            t1 = _acc(top1_hits[("ALL", frac)])
            t3 = _acc(top3_hits[("ALL", frac)])
            t5 = _acc(top5_hits[("ALL", frac)])
            mrr = _mean(mrr_vals[("ALL", frac)])
            print(f"  {frac:<6s}  Top-1: {t1:.4f}  Top-3: {t3:.4f}  Top-5: {t5:.4f}  MRR: {mrr:.4f}")

    print()


# ---------------------------------------------------------------------------
# Task: sequence completion
# ---------------------------------------------------------------------------

def _score_completion(gt_path: Path, pred_path: Path) -> None:
    gt_rows   = _read_csv_norm(gt_path)
    pred_rows = _read_csv_norm(pred_path)

    # Ground truth: EXAMPLE_ID → {partial, full, family, fraction}
    gt = {}
    for r in gt_rows:
        partial  = r["PARTIAL_SEQUENCE"].split("|")
        full     = r["FULL_SEQUENCE"].split("|")
        gt[r["EXAMPLE_ID"]] = {
            "partial":  partial,
            "full":     full,
            "remaining": full[len(partial):],
            "family":   r["FAMILY"],
            "fraction": r["COMPLETION_FRACTION"],
        }

    # Predictions: EXAMPLE_ID → remaining steps (pipe-separated)
    pred = {}
    for r in pred_rows:
        eid = r.get("EXAMPLE_ID", "")
        seq_str = r.get("PREDICTED_SEQUENCE", "").strip()
        pred[eid] = [s for s in seq_str.split("|") if s]

    matched   = [eid for eid in gt if eid in pred]
    unmatched = len(gt) - len(matched)

    if not matched:
        print("[ERROR] No matching EXAMPLE_IDs between ground truth and predictions.")
        sys.exit(1)

    ned_all:    list[float] = []
    exact_all:  list[bool]  = []
    tacc_all:   list[float] = []
    block_all:  list[float] = []
    ned_by:     dict = defaultdict(list)
    exact_by:   dict = defaultdict(list)

    for eid in matched:
        info      = gt[eid]
        ref       = info["remaining"]
        predicted = pred[eid]
        fam       = info["family"]
        frac      = info["fraction"]

        ned   = normalized_edit_distance(predicted, ref)
        exact = predicted == ref
        tacc  = token_accuracy(predicted, ref)
        bacc  = block_level_accuracy(predicted, ref)

        ned_all.append(ned)
        exact_all.append(exact)
        tacc_all.append(tacc)
        block_all.append(bacc)
        ned_by[(fam, "ALL")].append(ned)
        ned_by[("ALL", frac)].append(ned)
        exact_by[(fam, "ALL")].append(exact)
        exact_by[("ALL", frac)].append(exact)

    def _mean(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    sep = "=" * 60
    print(f"\n{sep}")
    print("EVAL RESULTS — SEQUENCE COMPLETION")
    print(sep)
    print(f"Evaluated : {len(matched)}/{len(gt)} examples"
          + (f"  ({unmatched} unmatched EXAMPLE_IDs)" if unmatched else ""))
    print()
    print(f"Mean Normalized Edit Distance : {_mean(ned_all):.4f}  (lower is better)")
    print(f"Exact Match Rate              : {_mean(exact_all):.4f}  ({_mean(exact_all)*100:.1f}%)")
    print(f"Mean Token Accuracy           : {_mean(tacc_all):.4f}  ({_mean(tacc_all)*100:.1f}%)")
    print(f"Mean Block-level Accuracy     : {_mean(block_all):.4f}  ({_mean(block_all)*100:.1f}%)")

    families = sorted({gt[e]["family"] for e in matched})
    if len(families) > 1:
        print("\nBy family:")
        for fam in families:
            n  = _mean(ned_by[(fam, "ALL")])
            ex = _mean(exact_by[(fam, "ALL")])
            print(f"  {fam:<8s}  NED: {n:.4f}  Exact: {ex:.4f}")

    fractions = sorted({gt[e]["fraction"] for e in matched})
    if len(fractions) > 1:
        print("\nBy completion fraction:")
        for frac in fractions:
            n  = _mean(ned_by[("ALL", frac)])
            ex = _mean(exact_by[("ALL", frac)])
            print(f"  {frac:<6s}  NED: {n:.4f}  Exact: {ex:.4f}")

    print()


# ---------------------------------------------------------------------------
# Task: anomaly detection (forbidden patterns)
# ---------------------------------------------------------------------------

def _score_anomaly(
    gt_path: Path,
    pred_path: Path,
    valid_supplement: Optional[Path],
) -> None:
    """
    Score detection of invalid (forbidden-pattern) sequences.

    Ground truth labels:
      - All rows in eval_set_forbidden.csv are IS_VALID=0 (invalid).
      - Rows added via --valid-supplement are IS_VALID=1 (valid).

    Prediction columns: EXAMPLE_ID, IS_VALID, [SCORE], [PREDICTED_RULE]
    """
    gt: dict[str, dict] = {}

    # Load forbidden examples (label = 0)
    for r in _read_csv_norm(gt_path):
        gt[r["EXAMPLE_ID"]] = {
            "is_valid":       0,
            "violation_rule": r.get("VIOLATION_RULE", ""),
        }

    # Optionally load valid examples (label = 1) as positive class for AUC
    if valid_supplement and valid_supplement.exists():
        for r in _read_csv_norm(valid_supplement):
            eid = r["EXAMPLE_ID"]
            if eid not in gt:
                gt[eid] = {"is_valid": 1, "violation_rule": ""}
        print(f"  Supplemented with {sum(1 for v in gt.values() if v['is_valid']==1)} "
              f"valid examples from {valid_supplement.name}")

    pred_rows = _read_csv_norm(pred_path)
    pred: dict[str, dict] = {}
    for r in pred_rows:
        eid = r.get("EXAMPLE_ID", "")
        try:
            is_valid_raw = r.get("IS_VALID", "").strip()
            is_valid_pred = int(float(is_valid_raw))
        except (ValueError, TypeError):
            is_valid_pred = -1
        try:
            score = float(r.get("SCORE", "").strip())
        except (ValueError, TypeError):
            score = float(is_valid_pred) if is_valid_pred >= 0 else 0.5
        predicted_rule = r.get("PREDICTED_RULE", "").strip()
        pred[eid] = {
            "is_valid":       is_valid_pred,
            "score":          score,
            "predicted_rule": predicted_rule,
        }

    matched   = [eid for eid in gt if eid in pred]
    unmatched = len(gt) - len(matched)

    if not matched:
        print("[ERROR] No matching EXAMPLE_IDs between ground truth and predictions.")
        sys.exit(1)

    labels, scores, preds_bin = [], [], []
    rule_gt, rule_pred = [], []

    for eid in matched:
        g = gt[eid]
        p = pred[eid]
        labels.append(g["is_valid"])
        scores.append(p["score"])
        preds_bin.append(p["is_valid"])
        if g["violation_rule"]:
            rule_gt.append(g["violation_rule"])
            rule_pred.append(p.get("predicted_rule", ""))

    n_pos = sum(l == 1 for l in labels)
    n_neg = sum(l == 0 for l in labels)
    accuracy = sum(p == l for p, l in zip(preds_bin, labels)) / len(labels)
    auc = _roc_auc(labels, scores)

    # Invalid class metrics (invalid = 0) for anomaly detection.
    tp = sum((l == 0) and (p == 0) for l, p in zip(labels, preds_bin))
    tn = sum((l == 1) and (p == 1) for l, p in zip(labels, preds_bin))
    fp = sum((l == 1) and (p == 0) for l, p in zip(labels, preds_bin))
    fn = sum((l == 0) and (p != 0) for l, p in zip(labels, preds_bin))
    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)

    # Rule attribution: among invalid examples where model correctly says invalid,
    # what fraction gets the rule right?
    correct_detection = [
        (rg, rp)
        for eid, rg, rp in zip(matched, rule_gt, rule_pred)
        if gt[eid]["is_valid"] == 0 and pred[eid]["is_valid"] == 0
    ]
    if correct_detection:
        rule_attr = sum(rg == rp for rg, rp in correct_detection) / len(correct_detection)
    else:
        rule_attr = float("nan")

    sep = "=" * 60
    print(f"\n{sep}")
    print("EVAL RESULTS — ANOMALY DETECTION (FORBIDDEN PATTERNS)")
    print(sep)
    print(f"Evaluated       : {len(matched)}/{len(gt)} examples"
          + (f"  ({unmatched} unmatched)" if unmatched else ""))
    print(f"Ground-truth    : {n_neg} invalid  /  {n_pos} valid")
    print()
    print(f"Binary Accuracy : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"Precision (invalid class) : {precision:.4f}")
    print(f"Recall (invalid class)    : {recall:.4f}")
    print(f"F1 (invalid class)        : {f1:.4f}")
    print("Confusion Matrix (invalid as positive):")
    print(f"  TP={tp}  FP={fp}")
    print(f"  FN={fn}  TN={tn}")
    if not (n_pos == 0 or n_neg == 0):
        auc_str = f"{auc:.4f}" if auc == auc else "n/a (single class)"
        print(f"ROC-AUC         : {auc_str}  (requires --valid-supplement for meaningful value)")
    else:
        print("ROC-AUC         : n/a  (only one class present — use --valid-supplement)")

    if rule_pred:
        ra_str = f"{rule_attr:.4f}" if rule_attr == rule_attr else "n/a"
        print(f"Rule Attribution Accuracy : {ra_str}  "
              f"(among {len(correct_detection)} correctly-detected invalid sequences)")

    # Per-rule breakdown
    rule_correct: dict[str, list[bool]] = defaultdict(list)
    for eid in matched:
        if gt[eid]["is_valid"] == 0:
            rule = gt[eid]["violation_rule"]
            is_detected = pred[eid]["is_valid"] == 0
            rule_correct[rule].append(is_detected)

    if rule_correct:
        print("\nDetection rate by rule:")
        for rule in sorted(rule_correct):
            hits = rule_correct[rule]
            rate = sum(hits) / len(hits)
            bar = "#" * int(rate * 20) + "-" * (20 - int(rate * 20))
        for rule in sorted(rule_correct):
            hits = rule_correct[rule]
            rate = sum(hits) / len(hits)
            print(f"  {rule:<40s}  {rate:.2f}  ({sum(hits)}/{len(hits)})")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval_metrics.py",
        description="Official scoring script for the process-sequence benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--task",
        choices=["next-step", "completion", "anomaly"],
        required=True,
        help="Evaluation task.",
    )
    p.add_argument(
        "--ground-truth",
        required=True,
        metavar="CSV",
        help="Path to eval_set_valid.csv (next-step / completion) or "
             "eval_set_forbidden.csv (anomaly).",
    )
    p.add_argument(
        "--predictions",
        required=True,
        metavar="CSV",
        help="Path to your model predictions CSV.",
    )
    p.add_argument(
        "--valid-supplement",
        default=None,
        metavar="CSV",
        help="[anomaly only] Path to eval_set_valid.csv to add valid examples "
             "as negative class for ROC-AUC computation.",
    )
    return p


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    gt_path   = Path(args.ground_truth)
    pred_path = Path(args.predictions)

    for path, label in [(gt_path, "--ground-truth"), (pred_path, "--predictions")]:
        if not path.exists():
            print(f"[ERROR] {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    if args.task == "next-step":
        _score_next_step(gt_path, pred_path)
    elif args.task == "completion":
        _score_completion(gt_path, pred_path)
    elif args.task == "anomaly":
        supplement = Path(args.valid_supplement) if args.valid_supplement else None
        _score_anomaly(gt_path, pred_path, supplement)


if __name__ == "__main__":
    main()
