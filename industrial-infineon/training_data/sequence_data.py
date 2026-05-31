#!/usr/bin/env python3
"""Shared data utilities for semiconductor process sequence models."""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


FAMILY_FILES = {
    "mosfet": "MOSFET_variants.csv",
    "igbt": "IGBT_variants.csv",
    "ic": "IC_variants.csv",
}

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>"]
FAMILY_TOKENS = {
    "mosfet": "<FAMILY_MOSFET>",
    "igbt": "<FAMILY_IGBT>",
    "ic": "<FAMILY_IC>",
}

# Neutral family marker used when the product family is unknown / not provided.
# Training randomly substitutes this for the real family token so the model
# also works without a family prefix (e.g. an unseen product category).
FAMILY_UNKNOWN_TOKEN = "<FAMILY_UNKNOWN>"

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"


@dataclass(frozen=True)
class SequenceExample:
    """One process sequence with its product family."""

    family: str
    sequence_id: str
    steps: list[str]


def _resolve_long_csv_keys(fieldnames: list[str] | None, path: Path) -> tuple[str, str]:
    """Map the (BOM/quote/whitespace-tolerant) SEQUENCE_ID and STEP columns to
    their raw header names."""
    if not fieldnames:
        raise ValueError(f"{path} has no header")

    def normalize(name: str) -> str:
        return name.lstrip("\ufeff").strip().strip('"').strip()

    norm_to_raw = {normalize(name): name for name in fieldnames}
    if "SEQUENCE_ID" not in norm_to_raw or "STEP" not in norm_to_raw:
        raise ValueError(
            f"{path} must contain SEQUENCE_ID and STEP columns; "
            f"found {fieldnames}"
        )
    return norm_to_raw["SEQUENCE_ID"], norm_to_raw["STEP"]


def iter_long_csv(
    path: Path, max_sequences: int | None = None
) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(sequence_id, steps)`` one sequence at a time, streaming.

    A sequence's rows are contiguous in the file (generated sequentially), so
    this never holds more than a single sequence in memory -- suitable for
    packing the multi-GB CSVs without OOM. ``max_sequences`` stops early after
    that many sequences have been emitted.
    """
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        seq_key, step_key = _resolve_long_csv_keys(reader.fieldnames, path)
        cur_id: str | None = None
        cur_steps: list[str] = []
        emitted = 0
        for row in reader:
            seq_id = row[seq_key].strip()
            step = row[step_key].strip().strip('"')
            if not (seq_id and step):
                continue
            if seq_id != cur_id:
                if cur_id is not None:
                    yield cur_id, cur_steps
                    emitted += 1
                    if max_sequences is not None and emitted >= max_sequences:
                        return
                cur_id = seq_id
                cur_steps = [step]
            else:
                cur_steps.append(step)
        if cur_id is not None and (max_sequences is None or emitted < max_sequences):
            yield cur_id, cur_steps


def read_long_csv(path: Path, max_sequences: int | None = None) -> dict[str, list[str]]:
    """Read a long-format CSV with SEQUENCE_ID and STEP columns into a dict.

    ``max_sequences`` caps how many distinct sequences are read and stops
    streaming early once that many have been seen. The CSV groups a sequence's
    rows together (generated sequentially), so this reads only the first N
    sequences without loading the rest of the file -- essential for the huge
    multi-GB datasets that would otherwise OOM the node.
    """
    sequences: dict[str, list[str]] = {}
    for seq_id, steps in iter_long_csv(path, max_sequences=max_sequences):
        # Contiguous grouping means each id appears once; if a file ever
        # interleaved them, keep the legacy "append" merge semantics.
        if seq_id in sequences:
            sequences[seq_id].extend(steps)
        else:
            sequences[seq_id] = steps
    return sequences


def load_examples(
    data_dir: Path, max_per_family: int | None = None
) -> list[SequenceExample]:
    """Load MOSFET, IGBT, and IC generated variants.

    ``max_per_family`` caps the number of sequences read from each family file
    (streaming stops early), keeping RAM bounded for very large datasets. None
    or 0 loads everything (legacy behaviour).
    """
    cap = max_per_family if max_per_family else None
    data_dir = Path(data_dir)
    examples: list[SequenceExample] = []
    for family, filename in FAMILY_FILES.items():
        path = data_dir / filename
        if not path.exists():
            # Turn the deep pathlib FileNotFoundError into an actionable message
            # naming the data dir, the missing family file, and how to recover.
            raise FileNotFoundError(
                f"Dataset file not found: {path}. The '{family}' family CSV "
                f"({filename}) is missing from data dir '{data_dir}'. Generate/upload "
                f"this dataset on the cluster and run preprocess_dataset.py to pack it, "
                f"or point --data-dir at the dataset the checkpoint was trained on."
            )
        family_sequences = read_long_csv(path, max_sequences=cap)
        for seq_id, steps in family_sequences.items():
            examples.append(SequenceExample(family=family, sequence_id=seq_id, steps=steps))
    return examples


def tokens_for_example(
    example: SequenceExample, family_token: str | None = None
) -> list[str]:
    """Return model tokens for one example, including family/BOS/EOS.

    Pass ``family_token`` to override the family slot (e.g.
    ``FAMILY_UNKNOWN_TOKEN``) for prefix-free / unknown-category evaluation.
    """
    family = family_token if family_token is not None else FAMILY_TOKENS[example.family]
    return [BOS_TOKEN, family, *example.steps, EOS_TOKEN]


def build_vocab_from_steps(
    step_tokens: Iterable[str],
) -> tuple[dict[str, int], list[str]]:
    """Build stable token mappings from a set/iterable of step tokens.

    The ordering is fully deterministic (specials, then family tokens, then the
    sorted remaining step tokens), so building the vocab over the full dataset
    during preprocessing yields the exact same ids as the per-run build."""
    tokens = set(SPECIAL_TOKENS)
    tokens.update(FAMILY_TOKENS.values())
    tokens.add(FAMILY_UNKNOWN_TOKEN)
    tokens.update(step_tokens)

    ordered = SPECIAL_TOKENS + list(FAMILY_TOKENS.values()) + [FAMILY_UNKNOWN_TOKEN]
    ordered += sorted(tokens - set(ordered))
    token_to_id = {token: idx for idx, token in enumerate(ordered)}
    return token_to_id, ordered


def build_vocab(examples: list[SequenceExample]) -> tuple[dict[str, int], list[str]]:
    """Build stable token mappings from examples."""
    steps: set[str] = set()
    for example in examples:
        steps.update(example.steps)
    return build_vocab_from_steps(steps)


def encode_tokens(
    tokens: list[str],
    token_to_id: dict[str, int],
    max_seq_len: int,
) -> list[int]:
    """Encode tokens and truncate safely to max_seq_len."""
    ids = [token_to_id[token] for token in tokens]
    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]
        ids[-1] = token_to_id[EOS_TOKEN]
    return ids


def encode_examples(
    examples: list[SequenceExample],
    token_to_id: dict[str, int],
    max_seq_len: int,
) -> list[list[int]]:
    """Encode examples as token-id sequences."""
    return [
        encode_tokens(tokens_for_example(example), token_to_id, max_seq_len)
        for example in examples
    ]


def split_examples(
    examples: list[SequenceExample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[SequenceExample], list[SequenceExample], list[SequenceExample]]:
    """Deterministically split examples into train/val/test."""
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def save_vocab(path: Path, token_to_id: dict[str, int], id_to_token: list[str]) -> None:
    """Save token mappings as JSON."""
    payload = {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "special_tokens": SPECIAL_TOKENS,
        "family_tokens": FAMILY_TOKENS,
        "family_unknown_token": FAMILY_UNKNOWN_TOKEN,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_vocab(path: Path) -> tuple[dict[str, int], list[str]]:
    """Load token mappings saved by save_vocab."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["token_to_id"], payload["id_to_token"]


def strip_model_tokens(tokens: list[str]) -> list[str]:
    """Remove special/family tokens to recover process steps."""
    blocked = set(SPECIAL_TOKENS) | set(FAMILY_TOKENS.values()) | {FAMILY_UNKNOWN_TOKEN}
    return [token for token in tokens if token not in blocked]


# ---------------------------------------------------------------------------
# Packed (memmap) dataset -- Phase 2 scalable loading.
#
# ``preprocess_dataset.py`` writes a <data_dir>/packed/ directory with:
#   vocab.json    token mappings (same scheme as save_vocab)
#   tokens.bin    flat uint16 ids for every sequence, concatenated
#   offsets.npy   int64[N+1]; sequence i = tokens[offsets[i]:offsets[i+1]]
#   families.npy  uint8[N]; index into meta["family_order"]
#   meta.json     counts, vocab_size, seed, source
#
# The token blob is opened read-only via ``np.memmap`` so all DDP ranks share
# the OS page cache instead of each materializing the dataset in RAM.
# ---------------------------------------------------------------------------

PACKED_SUBDIR = "packed"


def packed_dir_for(data_dir: Path) -> Path:
    """Conventional location of the packed dataset for a dataset dir."""
    return Path(data_dir) / PACKED_SUBDIR


def has_packed_dataset(data_dir: Path) -> bool:
    """True if a complete packed dataset exists under ``data_dir``."""
    packed = packed_dir_for(data_dir)
    return all(
        (packed / name).exists()
        for name in ("meta.json", "vocab.json", "tokens.bin", "offsets.npy", "families.npy")
    )


def ensure_dataset_available(data_dir: Path) -> None:
    """Fail fast (before any expensive work) if ``data_dir`` has neither a packed
    dataset nor the raw family CSVs.

    Train/eval pick the packed memmap when present and otherwise stream the raw
    CSVs; if *both* are absent the legacy path crashes deep inside ``pathlib``
    with a bare ``FileNotFoundError`` (e.g. ``training_data/MOSFET_variants.csv``)
    that gives no hint about the actual problem. This surfaces one clear,
    actionable message instead."""
    data_dir = Path(data_dir)
    if has_packed_dataset(data_dir):
        return
    missing = [
        data_dir / filename
        for filename in FAMILY_FILES.values()
        if not (data_dir / filename).exists()
    ]
    if not missing:
        return
    raise FileNotFoundError(
        f"No dataset found in '{data_dir}'. Expected either a packed dataset at "
        f"'{packed_dir_for(data_dir)}/' or the raw family CSVs "
        f"({', '.join(FAMILY_FILES.values())}). Missing: "
        f"{', '.join(str(p) for p in missing)}. Generate/upload this dataset on the "
        f"cluster and run preprocess_dataset.py to pack it, or point --data-dir at the "
        f"dataset the checkpoint was trained on."
    )


@dataclass
class PackedDataset:
    """Memmap-backed view of a preprocessed dataset (Phase 2)."""

    tokens: "object"  # np.memmap[uint16]
    offsets: "object"  # np.ndarray[int64], length num_sequences + 1
    families: "object"  # np.ndarray[uint8], length num_sequences
    token_to_id: dict[str, int]
    id_to_token: list[str]
    family_order: list[str]
    meta: dict

    @property
    def num_sequences(self) -> int:
        return int(self.offsets.shape[0] - 1)

    def ids_of(self, idx: int) -> list[int]:
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        return self.tokens[start:end].tolist()

    def family_of(self, idx: int) -> str:
        return self.family_order[int(self.families[idx])]


def load_packed(packed_dir: Path) -> PackedDataset:
    """Open the packed dataset read-only (memmap; near-zero RAM)."""
    import numpy as np

    packed_dir = Path(packed_dir)
    meta = json.loads((packed_dir / "meta.json").read_text(encoding="utf-8"))
    token_to_id, id_to_token = load_vocab(packed_dir / "vocab.json")
    offsets = np.load(packed_dir / "offsets.npy", mmap_mode="r")
    families = np.load(packed_dir / "families.npy", mmap_mode="r")
    num_tokens = int(meta["num_tokens"])
    tokens = np.memmap(
        packed_dir / "tokens.bin", dtype=np.uint16, mode="r", shape=(num_tokens,)
    )
    family_order = list(meta.get("family_order", list(FAMILY_TOKENS.keys())))
    return PackedDataset(
        tokens=tokens,
        offsets=offsets,
        families=families,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        family_order=family_order,
        meta=meta,
    )


def split_indices(
    n: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple["object", "object", "object"]:
    """Deterministic train/val/test index split mirroring ``split_examples``.

    Returns numpy index arrays. Because train and eval call this with the same
    (n, ratios, seed) over the fixed packed ordering, they reproduce the exact
    same held-out set (no leakage). numpy's PCG64 stream is stable across
    versions, so the split is reproducible within one environment."""
    import numpy as np

    perm = np.random.default_rng(seed).permutation(int(n))
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return perm[:train_end], perm[train_end:val_end], perm[val_end:]


def decode_example(ids: Iterable[int], id_to_token: list[str]) -> list[str]:
    """Decode token ids back to token strings (includes special/family tokens).

    Pair with ``strip_model_tokens`` to recover just the process steps."""
    return [id_to_token[int(i)] for i in ids]


class PackedSequenceDataset:
    """Map-style dataset (``__len__``/``__getitem__``) over the token memmap.

    Holds an index subset into the packed store. ``__getitem__`` slices one
    sequence's ids, truncates to ``max_seq_len`` (mirroring ``encode_tokens``),
    and applies per-access family-token dropout exactly like ``SequenceDataset``.
    No torch import is needed -- PyTorch's DataLoader accepts any map-style
    object -- and the memmap is shared across ranks via the OS page cache.
    """

    def __init__(
        self,
        tokens: "object",
        offsets: "object",
        indices: "object",
        max_seq_len: int,
        eos_id: int,
        family_dropout: float = 0.0,
        unknown_id: int | None = None,
        family_pos: int = 1,
    ) -> None:
        self.tokens = tokens
        self.offsets = offsets
        self.indices = indices
        self.max_seq_len = max_seq_len
        self.eos_id = eos_id
        self.family_dropout = family_dropout
        self.unknown_id = unknown_id
        self.family_pos = family_pos

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> list[int]:
        idx = int(self.indices[i])
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        seq = self.tokens[start:end].tolist()
        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len]
            seq[-1] = self.eos_id
        if (
            self.family_dropout > 0.0
            and self.unknown_id is not None
            and len(seq) > self.family_pos
            and random.random() < self.family_dropout
        ):
            seq[self.family_pos] = self.unknown_id
        return seq

