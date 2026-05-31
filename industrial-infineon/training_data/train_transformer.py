#!/usr/bin/env python3
"""Train a tiny causal Transformer on process-step tokens."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from sequence_data import (
    EOS_TOKEN,
    FAMILY_UNKNOWN_TOKEN,
    PAD_TOKEN,
    PackedSequenceDataset,
    build_vocab,
    encode_examples,
    ensure_dataset_available,
    has_packed_dataset,
    load_examples,
    load_packed,
    packed_dir_for,
    save_vocab,
    split_examples,
    split_indices,
)


class SequenceDataset(Dataset):
    """Token-id sequences with optional per-epoch family-token dropout.

    Each encoded sequence is ``[BOS, <FAMILY_x>, ...steps, EOS]`` so the family
    token always sits at ``family_pos`` (index 1). When ``family_dropout`` > 0,
    that slot is replaced by ``<FAMILY_UNKNOWN>`` with that probability on every
    access, so the substitution is re-randomized each epoch. This teaches the
    model to also generate without a real family prefix.
    """

    def __init__(
        self,
        sequences: list[list[int]],
        family_dropout: float = 0.0,
        unknown_id: int | None = None,
        family_pos: int = 1,
    ) -> None:
        self.sequences = sequences
        self.family_dropout = family_dropout
        self.unknown_id = unknown_id
        self.family_pos = family_pos

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> list[int]:
        seq = self.sequences[idx]
        if (
            self.family_dropout > 0.0
            and self.unknown_id is not None
            and len(seq) > self.family_pos
            and random.random() < self.family_dropout
        ):
            seq = list(seq)
            seq[self.family_pos] = self.unknown_id
        return seq


def collate_batch(batch: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in batch)
    inputs = torch.full((len(batch), max_len - 1), pad_id, dtype=torch.long)
    targets = torch.full((len(batch), max_len - 1), pad_id, dtype=torch.long)
    for row, seq in enumerate(batch):
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        inputs[row, : x.numel()] = x
        targets[row, : y.numel()] = y
    return inputs, targets


class TinyCausalTransformer(nn.Module):
    """GPT-style causal Transformer for next-token prediction."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_seq_len: int = 176,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) * math.sqrt(self.token_embedding.embedding_dim)
        hidden = hidden + self.position_embedding(positions)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        padding_mask = input_ids.eq(self.pad_id)
        hidden = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        hidden = self.norm(hidden)
        return self.output(hidden)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed() -> tuple[bool, int, int, int]:
    """Initialise DDP from torchrun env vars.

    Returns ``(is_distributed, rank, world_size, local_rank)``. When launched
    with plain ``python`` (WORLD_SIZE unset/1) this is a no-op single-process run.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    distributed: bool = False,
) -> tuple[float, float]:
    """Run one epoch and return (token-averaged loss, next-token accuracy)."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    # bf16 autocast on A100 (no GradScaler needed — bf16 has fp32 dynamic range).
    use_amp = device.type == "cuda"

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        mask = targets != criterion.ignore_index
        token_count = int(mask.sum().item())
        preds = logits.argmax(dim=-1)
        total_correct += int(((preds == targets) & mask).sum().item())
        total_loss += loss.item() * token_count
        total_tokens += token_count

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

    # Under DDP each rank only saw a shard of the data; sum the running totals
    # across ranks so every rank reports identical, global metrics.
    if distributed:
        agg = torch.tensor(
            [total_loss, float(total_tokens), float(total_correct)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
        total_loss, total_tokens, total_correct = (
            agg[0].item(),
            agg[1].item(),
            agg[2].item(),
        )

    denom = max(total_tokens, 1)
    return total_loss / denom, total_correct / denom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("training_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/transformer"))
    parser.add_argument("--max-seq-len", type=int, default=176)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--family-dropout",
        type=float,
        default=0.30,
        help="Probability of replacing the family token with <FAMILY_UNKNOWN> "
        "per example per epoch, for prefix-free robustness.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay (L2 regularization).",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Cross-entropy label smoothing (0 disables).",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["none", "cosine"],
        default="none",
        help="LR schedule: constant ('none') or linear warmup + cosine decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Fraction of total steps used for LR warmup (cosine schedule only).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (couple to --cpus-per-task; speeds up "
        "loading large datasets).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of sequences used for training (test = remainder).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of sequences used for validation (test = remainder).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="Cap sequences read PER FAMILY (0 = all). Bounds RAM for very "
        "large datasets by streaming only the first N sequences per family.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    # Offset the seed per rank so dropout/shuffle differ across ranks while
    # staying reproducible.
    set_seed(args.seed + rank)
    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Let the A100 use TF32 for matmuls/cuDNN — large speedup, negligible
    # accuracy impact for this task. Paired with bf16 autocast in run_epoch.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_ratio = float(args.train_ratio)
    val_ratio = float(args.val_ratio)
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise SystemExit(
            f"Invalid split ratios: train={train_ratio}, val={val_ratio}. "
            "Need 0<train, 0<val, and train+val<1 (test gets the remainder)."
        )

    max_sequences = int(args.max_sequences) or None

    # Fail fast with a clear message if the dataset dir has neither a packed
    # blob nor the raw CSVs, instead of crashing deep in pathlib later.
    ensure_dataset_available(args.data_dir)

    # Phase 2: if the dataset has been packed into a memmap blob, every rank
    # shares it via the OS page cache (RAM ~0) instead of materializing the
    # whole dataset. Otherwise fall back to the legacy in-memory path.
    use_packed = has_packed_dataset(args.data_dir)
    if use_packed:
        if max_sequences and is_main:
            print(
                f"NOTE: packed dataset present at {packed_dir_for(args.data_dir)}; "
                f"ignoring --max-sequences={max_sequences} (packed uses the full set)."
            )
        packed = load_packed(packed_dir_for(args.data_dir))
        token_to_id, id_to_token = packed.token_to_id, packed.id_to_token
        pad_id = token_to_id[PAD_TOKEN]
        unknown_id = token_to_id[FAMILY_UNKNOWN_TOKEN]
        eos_id = token_to_id[EOS_TOKEN]
        num_sequences = packed.num_sequences
        train_idx, val_idx, test_idx = split_indices(
            num_sequences, train_ratio=train_ratio, val_ratio=val_ratio, seed=args.seed
        )
        n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)
        train_dataset = PackedSequenceDataset(
            packed.tokens,
            packed.offsets,
            train_idx,
            args.max_seq_len,
            eos_id,
            family_dropout=args.family_dropout,
            unknown_id=unknown_id,
        )
        val_dataset = PackedSequenceDataset(
            packed.tokens, packed.offsets, val_idx, args.max_seq_len, eos_id
        )
        # Recorded full-set size makes the checkpoint self-describing for eval.
        max_sequences = None
    else:
        examples = load_examples(args.data_dir, max_per_family=max_sequences)
        train_examples, val_examples, test_examples = split_examples(
            examples, train_ratio=train_ratio, val_ratio=val_ratio, seed=args.seed
        )
        token_to_id, id_to_token = build_vocab(train_examples)
        pad_id = token_to_id[PAD_TOKEN]
        unknown_id = token_to_id[FAMILY_UNKNOWN_TOKEN]
        num_sequences = len(examples)
        n_train, n_val, n_test = len(train_examples), len(val_examples), len(test_examples)
        train_ids = encode_examples(train_examples, token_to_id, args.max_seq_len)
        val_ids = encode_examples(val_examples, token_to_id, args.max_seq_len)
        train_dataset = SequenceDataset(
            train_ids,
            family_dropout=args.family_dropout,
            unknown_id=unknown_id,
        )
        val_dataset = SequenceDataset(val_ids)

    pin = torch.cuda.is_available()
    # Under DDP each rank spawns its own loader workers but the ranks share the
    # task's CPU allocation, so split the requested workers across ranks.
    per_rank_workers = max(0, args.num_workers // world_size) if distributed else args.num_workers
    loader_kw = {
        "num_workers": per_rank_workers,
        "pin_memory": pin,
        "persistent_workers": per_rank_workers > 0,
    }
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=lambda batch: collate_batch(batch, pad_id),
        **loader_kw,
    )
    # Validation runs the full set on every rank (only rank 0 logs it), so no
    # sampler is needed and metrics stay exact (no DistributedSampler padding).
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, pad_id),
        **loader_kw,
    )

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCausalTransformer(
        vocab_size=len(id_to_token),
        pad_id=pad_id,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id, label_smoothing=args.label_smoothing
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Optional linear-warmup + cosine-decay LR schedule (stepped per batch).
    scheduler = None
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = max(0, int(args.warmup_ratio * total_steps))
    if args.lr_schedule == "cosine":

        def lr_lambda(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if is_main:
        print(f"device: {device}")
        if distributed:
            print(f"ddp: world_size={world_size} gpus, per_rank_workers={per_rank_workers}")
        print(
            f"optim: AdamW wd={args.weight_decay} label_smoothing={args.label_smoothing} "
            f"schedule={args.lr_schedule} warmup_steps={warmup_steps}/{total_steps} "
            f"num_workers={args.num_workers}"
        )
        if device.type == "cuda":
            print(f"gpu: {torch.cuda.get_device_name(local_rank)}")
        src = "packed memmap" if use_packed else "in-memory CSV"
        print(f"data source: {src} ({num_sequences} sequences)")
        print(f"examples: train={n_train} val={n_val} test={n_test}")
        print(f"vocab_size: {len(id_to_token)}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    param_count = sum(p.numel() for p in model.parameters())
    train_start = time.time()

    log_path = args.output_dir / "train_log.csv"
    fieldnames = [
        "epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr", "sec",
        "gpu_alloc_gb", "gpu_reserved_gb",
    ]
    # Only rank 0 owns the canonical log file; other ranks train silently.
    log_file = log_path.open("w", newline="", encoding="utf-8") if is_main else None
    writer = csv.DictWriter(log_file, fieldnames=fieldnames) if is_main else None
    if writer is not None:
        writer.writeheader()

    # Persist vocab + build checkpoint metadata up front so we can save the best
    # model *during* training (rank 0 only). This guarantees a usable checkpoint
    # exists even if the job is killed (walltime) before the last epoch: we keep
    # the best-by-val-loss and overwrite it whenever it improves.
    base_metadata = {
        "model_type": "transformer",
        # Stamp the Slurm job id so the dashboard can verify that a checkpoint /
        # stats file really belongs to a given job (cancelled/killed runs may
        # leave a previous run's canonical files behind).
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "vocab_size": len(id_to_token),
        "pad_id": pad_id,
        "max_seq_len": args.max_seq_len,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "family_dropout": args.family_dropout,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "lr_schedule": args.lr_schedule,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        # Per-family sequence cap used for this run (None/0 = full dataset). Eval
        # reads this back so it rebuilds the identical held-out split.
        "max_sequences": max_sequences,
        # Phase 2: whether this run used the packed memmap dataset, plus the
        # total sequence count, so eval reproduces the exact index split.
        "packed": use_packed,
        "num_sequences": num_sequences,
        "ddp": distributed,
        "world_size": world_size,
    }
    core_model = model.module if distributed else model
    checkpoint_path = args.output_dir / "transformer_model.pt"
    stats_path = args.output_dir / "train_stats.json"

    def save_best_checkpoint(best_epoch: int, best_val_loss: float, *, interrupted: bool = False) -> None:
        meta = {
            **base_metadata,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss) if best_epoch > 0 else None,
        }
        if interrupted:
            meta["interrupted"] = True
        # Atomic write so a kill mid-save never leaves a half-written checkpoint.
        tmp = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {"model_state": core_model.state_dict(), "metadata": meta},
            tmp,
        )
        tmp.replace(checkpoint_path)

    if is_main:
        save_vocab(args.output_dir / "vocab.json", token_to_id, id_to_token)
        # Drop stale artifacts from a previous run. We only overwrite the
        # checkpoint again after validation, so without this evaluate/archive
        # can pick up the old model (wrong architecture / wrong weights).
        for stale in (checkpoint_path, stats_path):
            if stale.exists():
                stale.unlink()

    best_val_loss = float("inf")
    best_epoch = 0
    ckpt_state = {"best_epoch": 0, "best_val_loss": float("inf")}

    if is_main:

        def _flush_checkpoint(interrupted: bool = False) -> None:
            be = ckpt_state["best_epoch"]
            bv = ckpt_state["best_val_loss"]
            if be > 0:
                save_best_checkpoint(be, bv, interrupted=interrupted)
                print(
                    f"  ↳ flushed best checkpoint (best_epoch={be}, "
                    f"val_loss={bv:.6f})",
                    flush=True,
                )
            elif interrupted:
                # Killed before the first validation — still persist current
                # weights so the run isn't a total loss (marked interrupted).
                save_best_checkpoint(0, float("inf"), interrupted=True)
                print(
                    "  ↳ saved interrupted checkpoint (no validation yet)",
                    flush=True,
                )

        def _handle_signal(signum: int, _frame) -> None:
            print(f"\nreceived signal {signum}, flushing checkpoint…", flush=True)
            _flush_checkpoint(interrupted=True)
            sys.exit(128 + (signum if signum < 128 else 0))

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        start = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scheduler,
            distributed=distributed,
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]
        if not is_main:
            continue
        # Per-epoch GPU memory snapshot (reserved is the high-water cache,
        # so it roughly tracks the peak so far without resetting global
        # peak stats captured at the end of training).
        if device.type == "cuda":
            gpu_alloc_gb = f"{torch.cuda.memory_allocated(device) / 1e9:.3f}"
            gpu_reserved_gb = f"{torch.cuda.memory_reserved(device) / 1e9:.3f}"
        else:
            gpu_alloc_gb = ""
            gpu_reserved_gb = ""
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_acc": f"{val_acc:.6f}",
                "lr": f"{lr:.6g}",
                "sec": f"{elapsed:.2f}",
                "gpu_alloc_gb": gpu_alloc_gb,
                "gpu_reserved_gb": gpu_reserved_gb,
            }
        )
        log_file.flush()
        print(
            f"epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} ({elapsed:.1f}s)"
        )
        # Keep the best-by-val-loss checkpoint, overwriting it on improvement so
        # the canonical model is always the best epoch so far (not just the last).
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            ckpt_state["best_epoch"] = best_epoch
            ckpt_state["best_val_loss"] = best_val_loss
            save_best_checkpoint(best_epoch, best_val_loss)
            print(
                f"  ↳ saved best checkpoint (best_epoch={best_epoch}, "
                f"val_loss={best_val_loss:.6f})"
            )
    if log_file is not None:
        log_file.close()

    # Non-main ranks have nothing more to persist; sync then exit cleanly.
    if not is_main:
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    # Best checkpoint + vocab were saved during the loop (or flushed on SIGTERM).
    if best_epoch > 0:
        print(f"best checkpoint: epoch {best_epoch} (val_loss={best_val_loss:.6f})")
    else:
        print("warning: no validation checkpoint saved (0 epochs completed?)", flush=True)
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(
            {
                "train": n_train,
                "validation": n_val,
                "test": n_test,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": round(1.0 - train_ratio - val_ratio, 6),
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Resource footprint of this run so the dashboard can show how close we got
    # to the GPU memory limit and how the run scaled.
    total_train_sec = round(time.time() - train_start, 1)
    stats: dict = {
        "device": device.type,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "params": int(param_count),
        "params_millions": round(param_count / 1e6, 3),
        "examples": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
        },
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "vocab_size": len(id_to_token),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "family_dropout": args.family_dropout,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "lr_schedule": args.lr_schedule,
        "warmup_ratio": args.warmup_ratio,
        "num_workers": args.num_workers,
        "ddp": distributed,
        "world_size": world_size,
        "total_train_sec": total_train_sec,
        "precision": "bf16+tf32" if device.type == "cuda" else "fp32",
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6) if best_epoch > 0 else None,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        total_gb = props.total_memory / 1e9
        peak_alloc_gb = torch.cuda.max_memory_allocated(device) / 1e9
        peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1e9
        stats.update(
            {
                "gpu_name": torch.cuda.get_device_name(local_rank),
                "gpu_total_gb": round(total_gb, 2),
                "gpu_peak_alloc_gb": round(peak_alloc_gb, 3),
                "gpu_peak_reserved_gb": round(peak_reserved_gb, 3),
                "gpu_peak_pct": round(100.0 * peak_reserved_gb / total_gb, 1)
                if total_gb
                else None,
            }
        )
    (args.output_dir / "train_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    if device.type == "cuda":
        print(
            f"gpu_mem: peak_reserved={stats['gpu_peak_reserved_gb']}GB / "
            f"total={stats['gpu_total_gb']}GB ({stats['gpu_peak_pct']}%) | "
            f"params={stats['params_millions']}M | train_time={total_train_sec}s"
        )
    print(f"train_stats: {json.dumps(stats)}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

