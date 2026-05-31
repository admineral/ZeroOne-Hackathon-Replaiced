#!/usr/bin/env python3
"""Generate every product-family variant CSV plus a dataset manifest in one run.

This is meant to run on a Leonardo compute node via Slurm so large datasets are
produced on the cluster without uploading CSVs from the laptop. Generation is
pure-Python rule-based sampling (see generation_rules.md) and uses no GPU.

Usage
-----
    python generate_all.py --count 10000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from generate_sequences import generate_dataset, write_csv

FAMILY_FILES = {
    "mosfet": "MOSFET_variants.csv",
    "igbt": "IGBT_variants.csv",
    "ic": "IC_variants.csv",
}


def _dump_simple_yaml(data: dict, indent: int = 0) -> str:
    """Minimal YAML serializer (avoids a PyYAML dependency)."""
    pad = "  " * indent
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(_dump_simple_yaml(value, indent + 1))
        else:
            lines.append(f"{pad}{key}: {value}")
    return "\n".join(line for line in lines if line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--families",
        type=str,
        default="mosfet+igbt+ic",
        help="Subset to generate, separated by '+' or ',' (e.g. 'mosfet+ic').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Where to write the CSVs and manifest (defaults to this script's dir).",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = [f.strip().lower() for f in args.families.replace(",", "+").split("+")]
    selected = [f for f in FAMILY_FILES if f in requested]
    if not selected:
        raise SystemExit(f"No valid families in --families={args.families!r}")

    # Make the dataset exactly the selected families: drop variant CSVs of the
    # families that were NOT selected so a subset run is clean.
    for family, fname in FAMILY_FILES.items():
        if family not in selected:
            stale = out_dir / fname
            if stale.exists():
                stale.unlink()
                print(f"removed unselected {fname}")

    families: dict[str, dict] = {}
    total_seqs = 0
    total_rows = 0
    for family in selected:
        fname = FAMILY_FILES[family]
        print(f"Generating {args.count} {family.upper()} sequences (seed={args.seed}) ...")
        sequences = generate_dataset(family, args.count, seed=args.seed, validate=True)
        path = out_dir / fname
        write_csv(path, sequences)
        rows = sum(len(s) for s in sequences)
        families[family] = {
            "file": fname,
            "sequences": len(sequences),
            "step_rows": rows,
            "bytes": path.stat().st_size,
        }
        total_seqs += len(sequences)
        total_rows += rows

    now = time.time()
    manifest = {
        "generated_at": now,
        "generated_at_iso": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
        "count_param": args.count,
        "seed": args.seed,
        "total_sequences": total_seqs,
        "total_step_rows": total_rows,
        "families": families,
        "generated_on": "leonardo",
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (out_dir / "dataset_manifest.yaml").write_text(
        "# Dataset manifest — generated on Leonardo by generate_all.py.\n"
        + _dump_simple_yaml(manifest),
        encoding="utf-8",
    )
    print(
        f"Done: {total_seqs} sequences / {total_rows} step rows across "
        f"{len(families)} families -> {out_dir}"
    )


if __name__ == "__main__":
    main()
