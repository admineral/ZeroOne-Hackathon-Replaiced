#!/usr/bin/env python3
"""Pack the long-format CSV dataset into a memmap-friendly binary blob.

Runs two streaming passes over ``<data_dir>/{MOSFET,IGBT,IC}_variants.csv``:

  pass 1: collect the closed step-token vocabulary
  pass 2: encode each sequence to ``[BOS, <FAMILY_x>, steps..., EOS]`` ids and
          append them to ``tokens.bin`` (uint16), recording the per-sequence
          start in ``offsets.npy`` (int64) and the family in ``families.npy``.

Output (in ``<out-dir>``, default ``<data-dir>/packed/``):

  vocab.json    token mappings, identical scheme to save_vocab
  tokens.bin    flat uint16 token ids for every sequence, concatenated
  offsets.npy   int64[N+1]; sequence i = tokens[offsets[i]:offsets[i+1]]
  families.npy  uint8[N]; index into meta["family_order"]
  meta.json     counts, vocab_size, seed, source, schema version

Streaming keeps RAM at ~O(vocab + N offsets), so it scales to the full ~30M
sequence set. Training/eval then ``np.memmap`` the blob with near-zero RAM.

The sequences are written UNtruncated so ``max_seq_len`` stays a runtime knob.
"""

from __future__ import annotations

import argparse
import json
import time
from array import array
from pathlib import Path

from sequence_data import (
    BOS_TOKEN,
    EOS_TOKEN,
    FAMILY_FILES,
    FAMILY_TOKENS,
    build_vocab_from_steps,
    iter_long_csv,
    save_vocab,
)

# uint16 caps the vocabulary at this many tokens; the step vocabulary is small
# and closed, so this is comfortable, but we assert it to fail loudly if not.
UINT16_MAX = 65535


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("training_data"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <data-dir>/packed).",
    )
    parser.add_argument(
        "--max-per-family",
        type=int,
        default=0,
        help="Cap sequences packed per family (0 = all). Mainly for quick tests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Recorded in meta.json; the split seed train/eval use by default.",
    )
    return parser.parse_args()


def main() -> None:
    import numpy as np

    args = parse_args()
    cap = int(args.max_per_family) or None
    out_dir: Path = args.out_dir or (args.data_dir / "packed")
    out_dir.mkdir(parents=True, exist_ok=True)
    family_order = list(FAMILY_FILES.keys())

    present = [fam for fam in family_order if (args.data_dir / FAMILY_FILES[fam]).exists()]
    if not present:
        raise SystemExit(
            f"No family CSVs found in {args.data_dir} "
            f"(looked for {sorted(FAMILY_FILES.values())})."
        )

    # ---- Pass 1: vocabulary -------------------------------------------------
    print("pass 1/2: scanning step vocabulary ...", flush=True)
    step_tokens: set[str] = set()
    pass1_counts: dict[str, int] = {}
    for fam in present:
        path = args.data_dir / FAMILY_FILES[fam]
        n = 0
        for _seq_id, steps in iter_long_csv(path, max_sequences=cap):
            step_tokens.update(steps)
            n += 1
            if n % 1_000_000 == 0:
                print(f"  {fam}: {n:,} sequences scanned (vocab {len(step_tokens):,})", flush=True)
        pass1_counts[fam] = n
        print(f"  {fam}: {n:,} sequences, vocab now {len(step_tokens):,}", flush=True)

    token_to_id, id_to_token = build_vocab_from_steps(step_tokens)
    vocab_size = len(id_to_token)
    if vocab_size > UINT16_MAX:
        raise SystemExit(
            f"vocab_size {vocab_size} exceeds uint16 range ({UINT16_MAX}); "
            "switch tokens.bin to uint32."
        )
    save_vocab(out_dir / "vocab.json", token_to_id, id_to_token)
    print(f"vocab_size={vocab_size}; wrote {out_dir / 'vocab.json'}", flush=True)

    bos_id = token_to_id[BOS_TOKEN]
    eos_id = token_to_id[EOS_TOKEN]
    family_ids = {fam: token_to_id[FAMILY_TOKENS[fam]] for fam in family_order}

    # ---- Pass 2: encode + pack ---------------------------------------------
    print("pass 2/2: encoding sequences -> tokens.bin ...", flush=True)
    # array('q') = int64, array('B') = uint8; both stay compact in RAM (~8 B and
    # ~1 B per sequence) so even 30M sequences cost only a few hundred MB here.
    offsets = array("q", [0])
    families = array("B")
    total_tokens = 0
    total_seqs = 0
    pass2_counts: dict[str, int] = {}

    tokens_path = out_dir / "tokens.bin"
    with tokens_path.open("wb") as fbin:
        for fam_idx, fam in enumerate(family_order):
            if fam not in present:
                continue
            path = args.data_dir / FAMILY_FILES[fam]
            fid = family_ids[fam]
            n = 0
            for _seq_id, steps in iter_long_csv(path, max_sequences=cap):
                ids = [bos_id, fid]
                ids.extend(token_to_id[step] for step in steps)
                ids.append(eos_id)
                np.asarray(ids, dtype=np.uint16).tofile(fbin)
                total_tokens += len(ids)
                offsets.append(total_tokens)
                families.append(fam_idx)
                total_seqs += 1
                n += 1
                if n % 1_000_000 == 0:
                    print(f"  {fam}: {n:,} sequences packed ({total_tokens:,} tokens)", flush=True)
            pass2_counts[fam] = n
            print(f"  {fam}: {n:,} sequences packed", flush=True)

    np.save(out_dir / "offsets.npy", np.frombuffer(offsets, dtype=np.int64))
    np.save(out_dir / "families.npy", np.frombuffer(families, dtype=np.uint8))

    meta = {
        "schema_version": 1,
        "num_sequences": total_seqs,
        "num_tokens": total_tokens,
        "vocab_size": vocab_size,
        "family_order": family_order,
        "family_counts": pass2_counts,
        "max_per_family": cap,
        "seed": args.seed,
        "tokens_dtype": "uint16",
        "offsets_dtype": "int64",
        "source": str(args.data_dir),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if pass1_counts != pass2_counts:
        print(
            f"WARNING: sequence counts differ between passes "
            f"(pass1={pass1_counts}, pass2={pass2_counts}); CSV may be changing."
        )
    print(
        f"done: {total_seqs:,} sequences, {total_tokens:,} tokens "
        f"(~{total_tokens * 2 / 1e9:.2f} GB tokens.bin) -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
