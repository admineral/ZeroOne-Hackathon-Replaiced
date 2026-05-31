"""Leonardo Pipeline Dashboard backend.

A local-only FastAPI service that orchestrates the existing Leonardo Slurm
pipeline over SSH. It never exposes the cluster password to the browser; the
frontend only talks to these REST + SSE endpoints on 127.0.0.1.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import posixpath
import re
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config import get_settings
from jobs import JobRecord, get_store
from leonardo import get_client
from pipeline import RUNS, get_run
from slurm import (
    SACCT_RESOURCE_FMT,
    parse_sacct_job,
    parse_sacct_resources,
    parse_sbatch_job_id,
    parse_squeue,
)

app = FastAPI(title="Leonardo Pipeline Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Real Slurm terminal states. Anything else (our "submitted" sentinel,
# PENDING/RUNNING/CONFIGURING, etc.) is still in-flight and should be refreshed.
TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "DEADLINE",
    "BOOT_FAIL",
    "PREEMPTED",
    "REVOKED",
}

# Code pushed to Leonardo by the Setup page's upload button: the pipeline scripts
# plus the tiny canonical reference sequences (synthetic_*.csv). Datasets are NOT
# uploaded -- the variant CSVs are *generated* on the cluster (generate_all.py
# writes them into datasets/<id>/), so they're an output of generation, not an
# input. Pushing them here would only re-seed the default training_data/ dataset.
UPLOAD_GLOBS = [
    "*.py",
    "*.slurm",
    "synthetic*.csv",
]

DATASET_FILES = {
    "mosfet": "MOSFET_variants.csv",
    "igbt": "IGBT_variants.csv",
    "ic": "IC_variants.csv",
}

# Versioned dataset collection on Leonardo: each remote generation writes into
# datasets/<id>/ instead of overwriting training_data/. Train/eval/baseline
# point their --data-dir at the chosen dataset.
DATASETS_DIR = "datasets"
DATASET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# The pre-existing single dataset lives here (the scripts' --data-dir default).
# It's listed in the collection as a non-deletable legacy entry for back-compat.
LEGACY_DATASET_ID = "training_data"


def _data_dir_for(dataset: str | None) -> str | None:
    """Map a selected dataset id to a --data-dir path (relative to the remote
    workdir). None/empty -> use the script default (training_data). The legacy
    id resolves to training_data so existing data keeps working unchanged."""
    if not dataset:
        return None
    if dataset == LEGACY_DATASET_ID:
        return LEGACY_DATASET_ID
    return f"{DATASETS_DIR}/{dataset}"


def _dataset_id(families: list[str], count: int, seed: int, when: "datetime | None" = None) -> str:
    """Auto-name a dataset folder from its generation config + timestamp,
    e.g. mosfet-igbt-ic_n10000_s42_20260530-1745."""
    chosen = {f.lower() for f in families}
    fams = "-".join(f for f in DATASET_FILES if f in chosen) or "none"
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{fams}_n{int(count)}_s{int(seed)}_{stamp}"


def _valid_dataset_id(dataset_id: str) -> bool:
    return bool(DATASET_ID_RE.match(dataset_id)) and "/" not in dataset_id and ".." not in dataset_id


# Terminal states that mean the run did not succeed. CANCELLED is excluded:
# it is user-initiated (scancel) and not an error worth surfacing.
FAILURE_STATES = {
    "FAILED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "DEADLINE",
    "BOOT_FAIL",
    "PREEMPTED",
    "REVOKED",
}


def is_terminal_state(state: str | None) -> bool:
    if not state:
        return False
    # e.g. "CANCELLED by 12345" -> "CANCELLED"
    base = state.split()[0].upper()
    return base in TERMINAL_STATES


def is_failure_state(state: str | None) -> bool:
    if not state:
        return False
    return state.split()[0].upper() in FAILURE_STATES


def _log_tail(text: str | None, lines: int = 40, max_chars: int = 4000) -> str:
    """Last few lines of a log, trimmed — enough to show the actual error."""
    if not text:
        return ""
    tail = "\n".join(text.splitlines()[-lines:]).strip()
    if len(tail) > max_chars:
        tail = "…" + tail[-max_chars:]
    return tail


def parse_loss_csv(text: str | None) -> list[dict]:
    """Parse train_log.csv generically: epoch as int, every other numeric
    column as float (so accuracy/lr/sec flow through automatically)."""
    if not text:
        return []
    rows: list[dict] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        if not row.get("epoch"):
            continue
        try:
            parsed: dict = {"epoch": int(row["epoch"])}
        except (ValueError, TypeError):
            continue
        for key, value in row.items():
            if key == "epoch" or key is None or value in (None, ""):
                continue
            try:
                parsed[key] = float(value)
            except (ValueError, TypeError):
                parsed[key] = value
        rows.append(parsed)
    return rows


def parse_csv_rows(text: str | None) -> list[dict]:
    if not text:
        return []
    return list(csv.DictReader(io.StringIO(text)))


def parse_gpu_timeline(text: str | None) -> list[dict]:
    """Parse an `nvidia-smi --format=csv,nounits` sample log into a series.

    Columns: timestamp, utilization.gpu, utilization.memory, memory.used,
    memory.total, power.draw. The x-axis is the sample index (~2s apart)."""
    if not text:
        return []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    rows: list[dict] = []
    for i, line in enumerate(lines[1:]):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue

        def num(idx: int) -> float | None:
            try:
                return float(parts[idx])
            except (ValueError, IndexError):
                return None

        used = num(3)
        total = num(4)
        rows.append(
            {
                "t": i * 2,
                "util": num(1),
                "util_mem": num(2),
                "mem_gb": round(used / 1024, 3) if used is not None else None,
                "mem_total_gb": round(total / 1024, 3) if total is not None else None,
                "power": num(5),
            }
        )
    return rows


def summarize_gpu_timeline(rows: list[dict]) -> dict:
    """Average/peak rollup of a GPU sample series for the resource panel."""
    if not rows:
        return {}

    def stats(key: str) -> tuple[float | None, float | None]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return (None, None)
        return (round(sum(vals) / len(vals), 1), round(max(vals), 1))

    avg_util, max_util = stats("util")
    avg_mem, max_mem = stats("mem_gb")
    avg_power, max_power = stats("power")
    return {
        "samples": len(rows),
        "avg_util": avg_util,
        "max_util": max_util,
        "avg_mem_gb": avg_mem,
        "max_mem_gb": max_mem,
        "avg_power_w": avg_power,
        "max_power_w": max_power,
    }


def _csv_counts(path: Path) -> tuple[int, int]:
    """Return (distinct SEQUENCE_ID count, data-row count) for a variants CSV."""
    if not path.exists():
        return (0, 0)
    seq_ids: set[str] = set()
    rows = 0
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = {(h or "").lstrip("\ufeff").strip().strip('"'): h for h in (reader.fieldnames or [])}
        seq_key = fields.get("SEQUENCE_ID")
        for row in reader:
            rows += 1
            if seq_key:
                seq_ids.add((row.get(seq_key) or "").strip())
    return (len(seq_ids) if seq_key else rows, rows)


def _dump_simple_yaml(data: dict) -> str:
    """Tiny YAML serializer for the flat-ish dataset manifest (no PyYAML dep)."""
    lines: list[str] = []

    def emit(obj, indent: int) -> None:
        pad = "  " * indent
        for key, val in obj.items():
            if isinstance(val, dict):
                lines.append(f"{pad}{key}:")
                emit(val, indent + 1)
            else:
                if isinstance(val, str):
                    val_str = val
                else:
                    val_str = json.dumps(val)
                lines.append(f"{pad}{key}: {val_str}")

    emit(data, 0)
    return "\n".join(lines) + "\n"


def build_dataset_manifest(script_dir: Path, count: int, seed: int) -> dict:
    """Count what was just generated locally and write JSON + YAML manifests."""
    now = time.time()
    families: dict[str, dict] = {}
    total_seqs = 0
    total_rows = 0
    for fam, fname in DATASET_FILES.items():
        path = script_dir / fname
        seqs, rows = _csv_counts(path)
        if not path.exists():
            continue
        families[fam] = {
            "file": fname,
            "sequences": seqs,
            "step_rows": rows,
            "bytes": path.stat().st_size,
        }
        total_seqs += seqs
        total_rows += rows
    manifest = {
        "generated_at": now,
        "generated_at_iso": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
        "count_param": count,
        "seed": seed,
        "total_sequences": total_seqs,
        "total_step_rows": total_rows,
        "families": families,
    }
    (script_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (script_dir / "dataset_manifest.yaml").write_text(
        "# Dataset manifest — generated by the Leonardo dashboard.\n"
        "# Describes the variant CSVs uploaded to the cluster.\n"
        + _dump_simple_yaml(manifest),
        encoding="utf-8",
    )
    return manifest


_validator_fn = None


def get_validator():
    """Import validate_sequence from the local training_data dir (no torch)."""
    global _validator_fn
    if _validator_fn is None:
        td = str(get_settings().local_project_dir / "training_data")
        if td not in sys.path:
            sys.path.insert(0, td)
        from generate_sequences import validate_sequence  # type: ignore

        _validator_fn = validate_sequence
    return _validator_fn


def parse_steps_text(text: str) -> list[str]:
    """Split a pasted block into step strings (newline / comma / semicolon)."""
    raw = text.replace(",", "\n").replace(";", "\n")
    steps = []
    for line in raw.splitlines():
        cleaned = line.strip().strip('"').strip()
        if cleaned:
            steps.append(cleaned.upper())
    return steps


def archived_path(canonical: str, job_id: str) -> str:
    """Per-job archive location for a canonical output file, e.g.
    outputs/transformer/train_log.csv -> outputs/transformer/runs/<job>/train_log.csv."""
    return posixpath.join(
        posixpath.dirname(canonical), "runs", job_id, posixpath.basename(canonical)
    )


def _train_stats_path(spec) -> str | None:
    """train_stats.json sits next to the run's train_log.csv."""
    if not spec.log:
        return None
    return posixpath.join(posixpath.dirname(spec.log), "train_stats.json")


def _archivable_files(spec) -> list[str]:
    """Canonical output files worth preserving per run (skip the large model)."""
    candidates = [
        spec.log,
        spec.split_summary,
        spec.summary,
        spec.rule_counts,
        spec.out,
        spec.err,
        spec.gpu_timeline,
        _train_stats_path(spec),
    ]
    return [f for f in candidates if f]


def _resolve_run_path(rec, canonical: str) -> str:
    """A run's canonical output file, or its per-job archive if a newer run for
    the same run_key now owns the canonical path."""
    latest = get_store().latest_for_run(rec.run_key)
    if latest is not None and latest.job_id != rec.job_id:
        return archived_path(canonical, rec.job_id)
    return canonical


async def _capture_error(client, rec, spec, resources: dict) -> dict:
    """Build a failure summary for a non-successful run: Slurm state + exit code
    plus the tail of the job's stderr (falling back to stdout)."""
    state = (resources.get("state") or rec.status or "").split()[0].upper()
    message = ""
    source = None
    if spec is not None:
        for attr in ("err", "out"):
            path = getattr(spec, attr, None)
            if not path:
                continue
            text = await asyncio.to_thread(
                client.read_remote_text, _resolve_run_path(rec, path), 262_144
            )
            tail = _log_tail(text)
            if tail:
                message = tail
                source = attr
                break
    return {
        "state": state,
        "exit_code": resources.get("exit_code"),
        "message": message,
        "source": source,
    }


async def _store_resources(client, rec, sacct_stdout: str) -> None:
    """Parse sacct output for a finished job and persist its resource usage,
    merging GPU/training stats (train_stats.json) when available. For failed
    runs it also captures the stderr tail so the UI can show what went wrong."""
    resources = parse_sacct_resources(sacct_stdout)
    if not resources or not resources.get("state"):
        return
    try:
        spec = get_run(rec.run_key)
    except KeyError:
        spec = None
    if spec is not None and spec.log:
        stats_path = _train_stats_path(spec)
        if stats_path:
            latest = get_store().latest_for_run(rec.run_key)
            path = stats_path
            if latest is not None and latest.job_id != rec.job_id:
                path = archived_path(stats_path, rec.job_id)
            text = await asyncio.to_thread(client.read_remote_text, path)
            if text:
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    stats_job = parsed.get("job_id")
                    # Only trust a stats file that identifies itself as this job.
                    # Cancelled/killed runs never write their own train_stats.json,
                    # so the canonical file can belong to a different run — never
                    # show another run's numbers. Legacy files without a job_id
                    # are accepted only when this job still owns the canonical path.
                    if stats_job is not None:
                        if str(stats_job) == str(rec.job_id):
                            resources["train_stats"] = parsed
                    elif not (latest is not None and latest.job_id != rec.job_id):
                        resources["train_stats"] = parsed
    # Roll up the nvidia-smi sample log into avg/peak util/mem/power.
    if spec is not None and spec.gpu_timeline:
        tl_path = spec.gpu_timeline
        latest = get_store().latest_for_run(rec.run_key)
        if latest is not None and latest.job_id != rec.job_id:
            tl_path = archived_path(spec.gpu_timeline, rec.job_id)
        tl_text = await asyncio.to_thread(client.read_remote_text, tl_path)
        summary = summarize_gpu_timeline(parse_gpu_timeline(tl_text))
        if summary:
            resources["gpu_timeline"] = summary
    get_store().set_resources(rec.job_id, resources)
    if is_failure_state(resources.get("state")):
        error = await _capture_error(client, rec, spec, resources)
        get_store().set_error(rec.job_id, error)


async def capture_resources(client, job_id: str) -> None:
    """Fetch sacct for one job and persist resources if it's finished."""
    rec = get_store().get(job_id)
    if rec is None or not rec.job_id or rec.resources:
        return
    if not is_terminal_state(rec.status):
        return
    res = await asyncio.to_thread(
        client.run, f"sacct -j {rec.job_id} -P --format={SACCT_RESOURCE_FMT}", 30
    )
    await _store_resources(client, rec, res.stdout)


def _manifest_summary(raw: dict) -> dict:
    """Condense a dataset manifest into the fields worth snapshotting per job."""
    families = raw.get("families") or {}
    return {
        "count_param": raw.get("count_param"),
        "seed": raw.get("seed"),
        "total_sequences": raw.get("total_sequences"),
        "total_step_rows": raw.get("total_step_rows"),
        "generated_at": raw.get("generated_at"),
        "generated_on": raw.get("generated_on", "local"),
        "families": {
            fam: (info or {}).get("sequences") for fam, info in families.items()
        },
    }


SUBMISSION_INPUT_FILES = {
    "valid": "eval_input_valid.csv",
    "anomaly": "eval_input_anomaly.csv",
}

SUBMISSION_OUTPUT_FILES = {
    "next_step": "predictions_nextstep.csv",
    "completion": "predictions_completion.csv",
    "anomaly": "predictions_anomaly.csv",
}

SUBMISSION_TASKS = {"all", "next-step", "completion", "anomaly"}

# Remote (on-Leonardo, relative to REMOTE_WORKDIR) locations used by the
# submission Slurm job. The job writes the prediction CSVs into
# REMOTE_SUBMISSION_OUTPUT_DIR; the dashboard fetches them back when it finishes.
REMOTE_SUBMISSION_OUTPUT_DIR = "outputs/transformer/submission"
REMOTE_PARTICIPANT_DIR = "participant_files"


def _submission_dirs() -> tuple[Path, Path, Path, Path, Path]:
    settings = get_settings()
    participant_dir = settings.local_project_dir.parent / "participant_files"
    output_dir = participant_dir / "submission"
    checkpoint = settings.local_project_dir / "outputs" / "transformer" / "transformer_model.pt"
    vocab = settings.local_project_dir / "outputs" / "transformer" / "vocab.json"
    script = settings.local_project_dir / "training_data" / "make_submission.py"
    return participant_dir, output_dir, checkpoint, vocab, script


def _generic_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.DictReader(f))


def _file_info(path: Path) -> dict:
    exists = path.exists()
    stat = path.stat() if exists else None
    rows = _generic_csv_rows(path) if exists and path.suffix.lower() == ".csv" else None
    return {
        "name": path.name,
        "path": str(path),
        "exists": exists,
        "rows": rows,
        "bytes": stat.st_size if stat else None,
        "mtime": stat.st_mtime if stat else None,
    }


def _dir_listing(path: Path) -> dict:
    """List the immediate contents (files + subfolders) of a directory."""
    if not path.exists() or not path.is_dir():
        return {"exists": path.exists(), "entries": []}
    entries = []
    # Folders first, then files, each alphabetical (case-insensitive).
    for child in sorted(
        path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
    ):
        is_dir = child.is_dir()
        try:
            stat = child.stat()
        except OSError:
            continue
        rows = (
            _generic_csv_rows(child)
            if not is_dir and child.suffix.lower() == ".csv"
            else None
        )
        entries.append(
            {
                "name": child.name,
                "is_dir": is_dir,
                "bytes": None if is_dir else stat.st_size,
                "mtime": stat.st_mtime,
                "rows": rows,
            }
        )
    return {"exists": True, "entries": entries}


def _anomaly_invalid_count(path: Path) -> int | None:
    if not path.exists():
        return None
    invalid = 0
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("IS_VALID", "")).strip() == "0":
                invalid += 1
    return invalid


def build_submission_status() -> dict:
    participant_dir, output_dir, checkpoint, vocab, script = _submission_dirs()
    anomaly_path = output_dir / SUBMISSION_OUTPUT_FILES["anomaly"]
    input_files = {
        key: _file_info(participant_dir / name)
        for key, name in SUBMISSION_INPUT_FILES.items()
    }
    output_files = {
        key: _file_info(output_dir / name)
        for key, name in SUBMISSION_OUTPUT_FILES.items()
    }
    return {
        "participant_dir": str(participant_dir),
        "output_dir": str(output_dir),
        "script": _file_info(script),
        "checkpoint": _file_info(checkpoint),
        "vocab": _file_info(vocab),
        "inputs": input_files,
        "outputs": output_files,
        "output_listing": _dir_listing(output_dir),
        "anomaly_invalid": _anomaly_invalid_count(anomaly_path),
        "ready": {
            "anomaly": input_files["anomaly"]["exists"],
            "next_step": input_files["valid"]["exists"] and script.exists(),
            "completion": input_files["valid"]["exists"] and script.exists(),
        },
    }


async def _ensure_remote_eval_inputs(client) -> list[str]:
    """Upload the organizer eval inputs to Leonardo if they're not there yet.

    The submission job reads them from <workdir>/participant_files/. They are
    static organizer files, so we only upload the ones missing remotely.
    """
    participant_dir, *_ = _submission_dirs()
    uploaded: list[str] = []
    for name in SUBMISSION_INPUT_FILES.values():
        local = participant_dir / name
        if not local.exists():
            raise HTTPException(status_code=404, detail=f"Missing eval input: {local}")
        remote = f"{REMOTE_PARTICIPANT_DIR}/{name}"
        if not await asyncio.to_thread(client.remote_exists, remote):
            await asyncio.to_thread(client.sftp_put, local, remote)
            uploaded.append(name)
    return uploaded


async def maybe_fetch_submission(client, job_id: str) -> None:
    """When a submission job finishes successfully, pull its prediction CSVs
    back into the local participant_files/submission/ folder. Idempotent: the
    job is flagged ``submission_fetched`` once the CSVs are local."""
    store = get_store()
    rec = store.get(job_id)
    if rec is None or rec.run_key != "submission" or not rec.job_id:
        return
    if rec.submission_fetched or not is_terminal_state(rec.status):
        return
    if is_failure_state(rec.status):
        # Nothing trustworthy to fetch; stop polling this job.
        store.mark_submission_fetched(job_id)
        return
    _, output_dir, *_ = _submission_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name in SUBMISSION_OUTPUT_FILES.values():
            remote = f"{REMOTE_SUBMISSION_OUTPUT_DIR}/{name}"
            if await asyncio.to_thread(client.remote_exists, remote):
                await asyncio.to_thread(client.sftp_get, remote, output_dir / name)
    except Exception:  # noqa: BLE001 - leave unflagged so the next poll retries
        return
    store.mark_submission_fetched(job_id)


async def _dataset_snapshot(client) -> dict:
    """Snapshot the dataset currently on Leonardo (remote manifest), falling
    back to the local manifest if the remote one is missing."""
    try:
        text = await asyncio.to_thread(
            client.read_remote_text, "training_data/dataset_manifest.json"
        )
        if text:
            return _manifest_summary(json.loads(text))
    except Exception:  # noqa: BLE001
        pass
    local = get_settings().local_project_dir / "training_data" / "dataset_manifest.json"
    if local.exists():
        try:
            return _manifest_summary(json.loads(local.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


async def maybe_archive(client, job_id: str) -> None:
    """Copy a finished run's outputs into outputs/.../runs/<job_id>/ so the run
    stays inspectable after the next run overwrites the canonical files.

    Guarded so we never archive stale data: only the most recent job for a given
    run_key still owns the canonical files, so older jobs are flagged archived
    (nothing to capture) without copying."""
    store = get_store()
    rec = store.get(job_id)
    if rec is None or rec.archived or not rec.job_id:
        return
    if not is_terminal_state(rec.status):
        return
    latest = store.latest_for_run(rec.run_key)
    if latest is None or latest.job_id != rec.job_id:
        store.mark_archived(job_id)
        return
    try:
        spec = get_run(rec.run_key)
    except KeyError:
        return
    files = _archivable_files(spec)
    # Opt-in: preserve this run's weights + vocab so it can be re-evaluated or
    # deployed later (large, hence off by default for the canonical archive).
    if (rec.params or {}).get("keep_checkpoint"):
        for extra in (spec.model, spec.vocab):
            if extra and extra not in files:
                files.append(extra)
    if not files:
        store.mark_archived(job_id)
        return
    run_dir = posixpath.join(posixpath.dirname(spec.out), "runs", job_id)
    quoted = " ".join(f"'{f}'" for f in files)
    cmd = (
        f"mkdir -p '{run_dir}' && for f in {quoted}; do "
        f'[ -f "$f" ] && cp -f "$f" \'{run_dir}/\'; done; echo archived'
    )
    res = await asyncio.to_thread(client.run_in_workdir, cmd, 60)
    if res.ok:
        store.mark_archived(job_id)


# -- models -------------------------------------------------------------------
class ValidateRequest(BaseModel):
    steps: list[str] | None = None
    text: str | None = None


class RunRequest(BaseModel):
    epochs: int | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    max_seq_len: int | None = None
    # Transformer architecture knobs.
    d_model: int | None = None
    num_layers: int | None = None
    num_heads: int | None = None
    dropout: float | None = None
    # Probability of dropping the family token during training (prefix-free
    # robustness). None lets the training script use its default (0.30).
    family_dropout: float | None = None
    # Regularization / schedule knobs.
    weight_decay: float | None = None
    label_smoothing: float | None = None
    lr_schedule: str | None = None  # "none" | "cosine"
    warmup_ratio: float | None = None
    # DataLoader worker processes (faster loading of large datasets).
    num_workers: int | None = None
    # Train/val split fractions (test = remainder). Evaluation reads the
    # checkpoint's recorded ratios so the held-out set matches training.
    train_ratio: float | None = None
    val_ratio: float | None = None
    # Cap sequences read PER FAMILY (0/None = all). Bounds RAM on huge datasets
    # by streaming only the first N sequences per family. Recorded in the
    # checkpoint so eval rebuilds the identical split.
    max_sequences: int | None = None
    # Number of GPUs to request; mem/cpus auto-scale per Leonardo fair-share.
    gpus: int | None = None
    seed: int | None = None
    # Number of sequences per family for remote (on-Leonardo) generation.
    count: int | None = None
    # Families to generate remotely (subset of mosfet/igbt/ic).
    families: list[str] | None = None
    # Dataset folder id (datasets/<id>) to train/evaluate against. None falls
    # back to the script default (training_data). Only for train/eval/baseline.
    dataset: str | None = None
    # Multi-GPU training via DDP (only meaningful with gpus > 1).
    ddp: bool | None = None
    # Archive this run's checkpoint + vocab so it can be re-evaluated later.
    keep_checkpoint: bool | None = None
    # Re-evaluate a specific past run's archived checkpoint (training job id)
    # instead of the latest canonical model. Only valid for evaluation runs.
    source_job_id: str | None = None
    # Slurm wall-clock limit override, e.g. "01:30:00". Falls back to the
    # script's #SBATCH --time when omitted.
    time_limit: str | None = None


class SubmissionRunRequest(BaseModel):
    # Remote checkpoint path from /submission/checkpoints. None = canonical.
    source: str | None = None
    # Subset of {"next-step", "completion", "anomaly"} or ["all"].
    tasks: list[str] | None = None
    # Slurm wall-clock override, e.g. "00:30:00".
    time_limit: str | None = None


class CheckpointRemoveRequest(BaseModel):
    # Remote model path (an archived runs/<job_id>/ entry) to delete.
    source: str


PARAM_LIMITS = {
    "epochs": {"min": 1, "max": 500},
    "learning_rate": {"min": 0.00001, "max": 1.0},
    "batch_size": {"min": 1, "max": 512},
    "max_seq_len": {"min": 8, "max": 512},
    "d_model": {"min": 8, "max": 1024, "step": 8},
    "num_layers": {"min": 1, "max": 12},
    "num_heads": {"min": 1, "max": 16},
    "dropout": {"min": 0.0, "max": 0.9},
    "family_dropout": {"min": 0.0, "max": 0.9},
    "weight_decay": {"min": 0.0, "max": 0.5},
    "label_smoothing": {"min": 0.0, "max": 0.3},
    "warmup_ratio": {"min": 0.0, "max": 0.5},
    "num_workers": {"min": 0, "max": 32},
    "train_ratio": {"min": 0.01, "max": 0.98},
    "val_ratio": {"min": 0.01, "max": 0.98},
    # 0 = no cap (load everything); otherwise per-family sequence limit.
    "max_sequences": {"min": 0, "max": 30_000_000},
    "gpus": {"allowed": [1, 2, 4]},
    "count": {"min": 1},
    "seed": {"min": 0, "max": 2_147_483_647},
}
ALLOWED_LR_SCHEDULES = {"none", "cosine"}
ALLOWED_FAMILIES = set(DATASET_FILES)
TIME_LIMIT_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")


def _check_range(errors: list[str], name: str, value) -> None:
    if value is None:
        return
    limits = PARAM_LIMITS[name]
    if "allowed" in limits:
        if value not in limits["allowed"]:
            errors.append(f"{name} must be one of {limits['allowed']}.")
        return
    if "min" in limits and value < limits["min"]:
        errors.append(f"{name} must be >= {limits['min']}.")
    if "max" in limits and value > limits["max"]:
        errors.append(f"{name} must be <= {limits['max']}.")
    step = limits.get("step")
    if step and int(value) % int(step) != 0:
        errors.append(f"{name} must be a multiple of {step}.")


def validate_run_request(run_key: str, req: RunRequest) -> None:
    """Strict dashboard/API validation before any Slurm submission.

    The UI already constrains these knobs; this makes the callable API equally
    safe for the AI coach and for direct REST calls.
    """
    try:
        get_run(run_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    errors: list[str] = []
    is_training = run_key in TRAINING_RUNS
    is_transformer = run_key == "transformer"
    is_generation = run_key in GENERATION_RUNS

    if req.time_limit is not None:
        if not TIME_LIMIT_RE.match(req.time_limit):
            errors.append("time_limit must use HH:MM:SS.")
        else:
            hours, minutes, seconds = [int(p) for p in req.time_limit.split(":")]
            if minutes > 59 or seconds > 59:
                errors.append("time_limit minutes and seconds must be < 60.")
            if hours > 24:
                errors.append("time_limit must be 24 hours or less.")

    if req.lr_schedule is not None and req.lr_schedule not in ALLOWED_LR_SCHEDULES:
        errors.append(f"lr_schedule must be one of {sorted(ALLOWED_LR_SCHEDULES)}.")
    if req.families is not None:
        invalid = [f for f in req.families if f.lower() not in ALLOWED_FAMILIES]
        if invalid:
            errors.append(f"families contains unsupported values: {invalid}.")
        if len(req.families) == 0:
            errors.append("families must include at least one family.")

    for name in (
        "epochs",
        "learning_rate",
        "batch_size",
        "max_seq_len",
        "d_model",
        "num_layers",
        "num_heads",
        "dropout",
        "family_dropout",
        "weight_decay",
        "label_smoothing",
        "warmup_ratio",
        "num_workers",
        "train_ratio",
        "val_ratio",
        "max_sequences",
        "gpus",
        "count",
        "seed",
    ):
        _check_range(errors, name, getattr(req, name))

    if req.train_ratio is not None and req.val_ratio is not None:
        if req.train_ratio + req.val_ratio >= 0.99:
            errors.append("train_ratio + val_ratio must leave at least 1% for test.")

    if req.ddp and (req.gpus or 1) < 2:
        errors.append("ddp requires gpus >= 2.")

    if is_transformer and req.d_model and req.num_heads:
        if int(req.d_model) % int(req.num_heads) != 0:
            errors.append(
                f"d_model ({req.d_model}) must be divisible by num_heads ({req.num_heads})."
            )

    if not is_training:
        disallowed = [
            name
            for name in (
                "epochs",
                "learning_rate",
                "batch_size",
                "max_seq_len",
                "d_model",
                "num_layers",
                "num_heads",
                "dropout",
                "family_dropout",
                "weight_decay",
                "label_smoothing",
                "lr_schedule",
                "warmup_ratio",
                "num_workers",
                "gpus",
                "ddp",
                "keep_checkpoint",
            )
            if getattr(req, name) is not None
        ]
        if disallowed:
            errors.append(f"{run_key} does not accept training params: {disallowed}.")

    if not is_generation and (req.count is not None or req.families is not None):
        errors.append(f"{run_key} does not accept dataset generation params.")

    if req.dataset is not None:
        if is_generation:
            errors.append("dataset is not valid for generation runs (they create one).")
        elif not _valid_dataset_id(req.dataset):
            errors.append(f"Invalid dataset id: {req.dataset!r}.")

    if not run_key.startswith("eval_") and req.source_job_id:
        errors.append("source_job_id is only valid for evaluation runs.")

    if errors:
        raise HTTPException(status_code=400, detail=" ".join(errors))


TRAINING_RUNS = {"transformer"}
GENERATION_RUNS = {"generate_remote"}
# One-time CPU job that packs a dataset into <data_dir>/packed/ for memmap
# training/eval (Phase 2 scalable loading).
PREPROCESS_RUNS = {"preprocess"}


# -- meta ---------------------------------------------------------------------
@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
async def config() -> dict:
    settings = get_settings()
    return {
        "host": settings.host,
        "user": settings.user,
        "remote_workdir": settings.remote_workdir,
        "has_password": bool(settings.password),
        "runs": [
            {
                "key": spec.key,
                "label": spec.label,
                "slurm": spec.slurm,
                "has_loss": spec.log is not None,
                "has_summary": spec.summary is not None,
            }
            for spec in RUNS.values()
        ],
    }


@app.get("/api/runs")
async def runs() -> dict:
    """Expose AI-safe run metadata and parameter bounds.

    This is read-only configuration: it contains no credentials and no direct
    SSH/Slurm access.
    """
    return {
        "runs": [
            {
                "key": spec.key,
                "label": spec.label,
                "slurm": spec.slurm,
                "has_loss": spec.log is not None,
                "has_summary": spec.summary is not None,
                "has_model": spec.model is not None,
                "has_gpu_timeline": spec.gpu_timeline is not None,
                "kind": (
                    "training"
                    if spec.key in TRAINING_RUNS
                    else "generation"
                    if spec.key in GENERATION_RUNS
                    else "evaluation"
                    if spec.key.startswith("eval_")
                    else "other"
                ),
            }
            for spec in RUNS.values()
        ],
        "parameter_limits": PARAM_LIMITS,
        "allowed": {
            "families": sorted(ALLOWED_FAMILIES),
            "lr_schedule": sorted(ALLOWED_LR_SCHEDULES),
            "time_limit_format": "HH:MM:SS",
        },
    }


@app.get("/api/runs/{run_key}")
async def run_info(run_key: str) -> dict:
    try:
        spec = get_run(run_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "key": spec.key,
        "label": spec.label,
        "slurm": spec.slurm,
        "paths": {
            "out": spec.out,
            "err": spec.err,
            "log": spec.log,
            "summary": spec.summary,
            "rule_counts": spec.rule_counts,
            "split_summary": spec.split_summary,
            "gpu_timeline": spec.gpu_timeline,
        },
        "kind": (
            "training"
            if spec.key in TRAINING_RUNS
            else "generation"
            if spec.key in GENERATION_RUNS
            else "evaluation"
            if spec.key.startswith("eval_")
            else "other"
        ),
        "parameter_limits": PARAM_LIMITS,
    }


@app.post("/api/params/validate/{run_key}")
async def validate_params(run_key: str, req: RunRequest = RunRequest()) -> dict:
    validate_run_request(run_key, req)
    return {"ok": True, "run_key": run_key, "params": req.model_dump(exclude_none=True)}


# -- official participant submission -----------------------------------------
@app.get("/api/submission")
async def submission_status() -> dict:
    """Inspect the official eval inputs and generated submission CSVs."""
    return build_submission_status()


@app.post("/api/submission/run")
async def run_submission(req: SubmissionRunRequest = SubmissionRunRequest()) -> dict:
    """Generate the organizer prediction CSVs as a Leonardo Slurm job.

    Uploads the current training scripts (incl. make_submission.py) and the
    organizer eval inputs, then submits run_make_submission.slurm against the
    chosen remote checkpoint. The result CSVs are pulled back into
    participant_files/submission/ automatically when the job finishes
    (see maybe_fetch_submission), so nothing is generated on the laptop.
    """
    spec = get_run("submission")
    tasks = req.tasks or ["all"]
    invalid = [task for task in tasks if task not in SUBMISSION_TASKS]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported submission task(s): {invalid}.",
        )
    task_arg = "all" if "all" in tasks else " ".join(tasks)

    client = get_client()
    # Deploy the latest scripts (so the make_submission.py fix is on Leonardo)
    # and make sure the organizer eval inputs are present remotely.
    try:
        await _upload_training_files(client)
        await _ensure_remote_eval_inputs(client)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any paramiko/transfer error
        raise HTTPException(status_code=502, detail=f"Upload to Leonardo failed: {exc}")

    # Resolve and validate the chosen checkpoint against what's on Leonardo.
    try:
        entries = await _list_remote_checkpoints(client)
    except Exception as exc:  # noqa: BLE001 - surface any paramiko/auth error
        raise HTTPException(status_code=502, detail=f"SSH connection failed: {exc}")
    chosen: dict | None = None
    if req.source:
        chosen = next((e for e in entries if e["source"] == req.source), None)
        if chosen is None:
            raise HTTPException(
                status_code=400, detail=f"Unknown checkpoint source: {req.source}"
            )
    elif entries:
        chosen = entries[0]

    settings = get_settings()
    exports = [f"INPUT_DIR={REMOTE_PARTICIPANT_DIR}", f"TASKS={task_arg}"]
    if chosen:
        exports.append(f"CHECKPOINT={chosen['source']}")
        exports.append(f"VOCAB={chosen['vocab']}")
    exports.append(f"REMOTE_WORKDIR={settings.remote_workdir}")
    export_arg = f"--export=ALL,{','.join(exports)} "

    sbatch_flags = ""
    if settings.slurm_account:
        sbatch_flags += f"--account={shlex.quote(settings.slurm_account)} "
    if settings.slurm_reservation:
        sbatch_flags += f"--reservation={shlex.quote(settings.slurm_reservation)} "
    time_arg = f"--time={req.time_limit} " if req.time_limit else ""

    res = await asyncio.to_thread(
        client.run_in_workdir,
        f"sbatch {sbatch_flags}{time_arg}{export_arg}{spec.slurm}",
    )
    if not res.ok:
        raise HTTPException(
            status_code=500, detail=res.stderr.strip() or res.stdout.strip()
        )
    job_id = parse_sbatch_job_id(res.stdout)
    note = res.stdout.strip()
    if chosen:
        note += f" · {chosen['label']}"
    note += f" · tasks={task_arg}"
    record = JobRecord(
        run_key="submission",
        label=spec.label,
        job_id=job_id,
        slurm_script=spec.slurm,
        status="submitted",
        note=note,
        params={
            "tasks": tasks,
            "source": chosen["source"] if chosen else None,
            "checkpoint_label": chosen["label"] if chosen else None,
        },
    )
    get_store().add(record)
    return {
        "ok": True,
        "job_id": job_id,
        "run_key": "submission",
        "checkpoint": chosen,
        "raw": res.stdout.strip(),
        "status": build_submission_status(),
    }


def _checkpoint_label(model_path: str, current_path: str) -> tuple[str, bool]:
    """Human label for a remote checkpoint + whether it is the canonical one."""
    if model_path == current_path:
        return "current (outputs/transformer)", True
    parts = model_path.split("/")
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 1 < len(parts):
            return f"run {parts[idx + 1]}", False
    return model_path, False


async def _list_remote_checkpoints(client) -> list[dict]:
    """Enumerate usable transformer checkpoints on Leonardo.

    A checkpoint is usable for Tasks 1/2 only if a sibling vocab.json sits
    next to the transformer_model.pt, so we filter to those. Covers the
    canonical outputs/transformer/ file plus any archived outputs/transformer/
    runs/<job_id>/ snapshots.
    """
    spec = get_run("transformer")
    base = posixpath.dirname(spec.model)  # outputs/transformer
    cmd = (
        f"find {shlex.quote(base)} -maxdepth 3 -name transformer_model.pt -type f 2>/dev/null | "
        'while read -r f; do d=$(dirname "$f"); '
        'if [ -f "$d/vocab.json" ]; then hv=1; else hv=0; fi; '
        "s=$(stat -c '%s' \"$f\" 2>/dev/null||echo 0); "
        "m=$(stat -c '%Y' \"$f\" 2>/dev/null||echo 0); "
        'echo "$f|$s|$m|$hv"; done'
    )
    res = await asyncio.to_thread(client.run_in_workdir, cmd, 60)
    entries: list[dict] = []
    for line in (res.stdout or "").splitlines():
        line = line.strip()
        if line.count("|") != 3:
            continue
        path, size, mtime, has_vocab = line.split("|")
        if has_vocab != "1":
            continue
        label, is_current = _checkpoint_label(path, spec.model)
        entries.append(
            {
                "source": path,
                "vocab": posixpath.join(posixpath.dirname(path), "vocab.json"),
                "label": label,
                "is_current": is_current,
                "bytes": int(size) if size.isdigit() else None,
                "mtime": float(mtime) if mtime.isdigit() else None,
            }
        )
    # Canonical first, then newest archived runs.
    entries.sort(key=lambda e: (not e["is_current"], -(e["mtime"] or 0)))
    return entries


@app.get("/api/submission/checkpoints")
async def list_checkpoints() -> dict:
    """List transformer checkpoints (model + vocab) available on Leonardo."""
    client = get_client()
    try:
        entries = await _list_remote_checkpoints(client)
    except Exception as exc:  # noqa: BLE001 - surface any paramiko/auth error
        raise HTTPException(status_code=502, detail=f"SSH connection failed: {exc}")
    return {"checkpoints": entries}


async def _remote_submission_listing(client) -> dict:
    """List the prediction CSVs the submission job wrote on Leonardo
    (REMOTE_SUBMISSION_OUTPUT_DIR), with size/mtime and CSV row counts.

    This is the authoritative submission output; the local
    participant_files/submission/ folder is just a downloaded mirror.
    """
    d = REMOTE_SUBMISSION_OUTPUT_DIR
    cmd = (
        f"d={shlex.quote(d)}; "
        'if [ -d "$d" ]; then '
        'find "$d" -maxdepth 1 -mindepth 1 2>/dev/null | while read -r f; do '
        'n=$(basename "$f"); '
        'if [ -d "$f" ]; then '
        "m=$(stat -c '%Y' \"$f\" 2>/dev/null||echo 0); "
        'echo "$n|dir|0|$m|-1"; '
        'else '
        "s=$(stat -c '%s' \"$f\" 2>/dev/null||echo 0); "
        "m=$(stat -c '%Y' \"$f\" 2>/dev/null||echo 0); "
        'case "$n" in *.csv) lc=$(wc -l < "$f" 2>/dev/null||echo -1);; *) lc=-1;; esac; '
        'echo "$n|file|$s|$m|$lc"; '
        'fi; done; '
        'else echo __MISSING__; fi'
    )
    res = await asyncio.to_thread(client.run_in_workdir, cmd, 60)
    out = (res.stdout or "").strip()
    remote_dir = f"{get_settings().remote_workdir.rstrip('/')}/{d}"
    if out == "__MISSING__" or not out:
        return {"dir": remote_dir, "exists": out != "__MISSING__" and res.ok, "entries": []}
    entries: list[dict] = []
    for line in out.splitlines():
        line = line.strip()
        if line.count("|") != 4:
            continue
        name, kind, size, mtime, linecount = line.split("|")
        is_dir = kind == "dir"
        lc = int(linecount) if linecount.lstrip("-").isdigit() else -1
        # CSV row count excludes the header line.
        rows = lc - 1 if (not is_dir and lc > 0) else None
        entries.append(
            {
                "name": name,
                "is_dir": is_dir,
                "bytes": None if is_dir else (int(size) if size.isdigit() else None),
                "mtime": float(mtime) if mtime.isdigit() else None,
                "rows": rows,
            }
        )
    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
    return {"dir": remote_dir, "exists": True, "entries": entries}


@app.get("/api/submission/remote")
async def submission_remote() -> dict:
    """List the prediction CSVs on Leonardo — the authoritative submission."""
    client = get_client()
    try:
        return await _remote_submission_listing(client)
    except Exception as exc:  # noqa: BLE001 - surface any paramiko/auth error
        raise HTTPException(status_code=502, detail=f"SSH connection failed: {exc}")


@app.post("/api/submission/remove-checkpoint")
async def remove_checkpoint(req: CheckpointRemoveRequest) -> dict:
    """Delete an archived remote checkpoint folder (outputs/transformer/runs/<id>).

    Only archived ``runs/<job_id>/`` snapshots can be removed — the canonical
    ``outputs/transformer/transformer_model.pt`` is left alone since it is the
    latest training output the rest of the pipeline relies on.
    """
    client = get_client()
    try:
        entries = await _list_remote_checkpoints(client)
    except Exception as exc:  # noqa: BLE001 - surface any paramiko/auth error
        raise HTTPException(status_code=502, detail=f"SSH connection failed: {exc}")

    chosen = next((e for e in entries if e["source"] == req.source), None)
    if chosen is None:
        raise HTTPException(status_code=404, detail=f"Unknown checkpoint: {req.source}")
    if chosen["is_current"]:
        raise HTTPException(
            status_code=400,
            detail="Refusing to delete the canonical checkpoint (outputs/transformer).",
        )

    run_dir = posixpath.dirname(chosen["source"])
    if "/runs/" not in f"/{run_dir}/":
        raise HTTPException(
            status_code=400,
            detail=f"Refusing to delete non-archive path: {run_dir}",
        )
    res = await asyncio.to_thread(
        client.run_in_workdir, f"rm -rf {shlex.quote(run_dir)} && echo removed", 60
    )
    if not res.ok:
        raise HTTPException(
            status_code=502, detail=res.stderr.strip() or "Remote delete failed."
        )
    try:
        remaining = await _list_remote_checkpoints(client)
    except Exception:  # noqa: BLE001
        remaining = []
    return {"ok": True, "removed": chosen["source"], "checkpoints": remaining}


# -- ssh ----------------------------------------------------------------------
@app.post("/api/ssh/test")
async def ssh_test() -> dict:
    client = get_client()
    try:
        res = await asyncio.to_thread(client.run, "hostname && whoami", 30)
    except Exception as exc:  # noqa: BLE001 - surface any paramiko/auth error
        raise HTTPException(status_code=502, detail=f"SSH connection failed: {exc}")
    if not res.ok:
        raise HTTPException(status_code=502, detail=res.stderr or "SSH command failed")
    lines = res.stdout.strip().splitlines()
    return {
        "ok": True,
        "hostname": lines[0] if lines else "",
        "user": lines[1] if len(lines) > 1 else "",
        "raw": res.stdout.strip(),
    }


# -- upload -------------------------------------------------------------------
def _collect_training_files() -> list[Path]:
    settings = get_settings()
    script_dir = settings.local_project_dir / "training_data"
    if not script_dir.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {script_dir}")
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in UPLOAD_GLOBS:
        for path in sorted(script_dir.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                files.append(path)
    return files


async def _upload_training_files(client) -> list[dict]:
    """Push training_data/*.py + *.slurm to Leonardo (deploys code changes)."""
    uploaded: list[dict] = []
    for path in _collect_training_files():
        remote = f"training_data/{path.name}"
        try:
            await asyncio.to_thread(client.sftp_put, path, remote)
            uploaded.append({"file": path.name, "remote": remote, "ok": True})
        except Exception as exc:  # noqa: BLE001
            uploaded.append({"file": path.name, "ok": False, "error": str(exc)})
    return uploaded


@app.post("/api/upload")
async def upload() -> dict:
    client = get_client()
    uploaded = await _upload_training_files(client)
    return {"uploaded": uploaded, "count": len(uploaded)}


# -- dataset inspection -------------------------------------------------------
@app.get("/api/dataset")
async def dataset() -> dict:
    """Report what dataset currently lives on Leonardo: per-family row counts
    (from the uploaded manifest) plus real file size + upload time (mtime).

    Only stats remote file metadata on the login node — no compute."""
    client = get_client()

    # Uploaded manifest (matches the remote CSVs since it ships with them).
    text = await asyncio.to_thread(
        client.read_remote_text, "training_data/dataset_manifest.json"
    )
    manifest = None
    if text:
        try:
            manifest = json.loads(text)
        except json.JSONDecodeError:
            manifest = None

    # Stat each CSV for size + mtime (cheap metadata read, login-node safe).
    rel = [f"training_data/{fname}" for fname in DATASET_FILES.values()]
    quoted = " ".join(f"'{f}'" for f in rel)
    res = await asyncio.to_thread(
        client.run_in_workdir, f"stat -c '%n|%s|%Y' {quoted} 2>/dev/null", 30
    )
    remote: dict[str, dict] = {}
    last_upload: float | None = None
    for line in res.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        name, size, mtime = parts
        try:
            size_i, mtime_f = int(size), float(mtime)
        except ValueError:
            continue
        remote[posixpath.basename(name)] = {"bytes": size_i, "mtime": mtime_f}
        last_upload = mtime_f if last_upload is None else max(last_upload, mtime_f)

    man_fams = (manifest or {}).get("families", {})

    # Fallback for datasets uploaded before the manifest existed: derive counts
    # cheaply from the remote files (wc -l for step rows, last sequential
    # SEQUENCE_ID for the sequence count). Light I/O only, login-node safe.
    fallback: dict[str, dict] = {}
    need = [
        (fam, fname)
        for fam, fname in DATASET_FILES.items()
        if remote.get(fname) and not man_fams.get(fam)
    ]
    if need:
        cmds = []
        for _, fname in need:
            f = f"training_data/{fname}"
            cmds.append(
                f"printf '%s|%s|%s\\n' '{fname}' "
                f"\"$(wc -l < '{f}' 2>/dev/null)\" "
                f"\"$(tail -n1 '{f}' 2>/dev/null | cut -d, -f1)\""
            )
        fb = await asyncio.to_thread(client.run_in_workdir, "; ".join(cmds), 45)
        for line in fb.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            fname, lines_s, last_id = parts
            try:
                step_rows = max(0, int(lines_s) - 1)  # minus header
            except ValueError:
                step_rows = None
            seq_match = re.search(r"(\d+)", last_id or "")
            seqs = int(seq_match.group(1)) if seq_match else None
            fallback[fname] = {"sequences": seqs, "step_rows": step_rows}

    families: list[dict] = []
    for fam, fname in DATASET_FILES.items():
        fm = man_fams.get(fam)
        rm = remote.get(fname)
        if not fm and not rm:
            continue
        fb_fam = fallback.get(fname, {})
        families.append(
            {
                "family": fam,
                "file": fname,
                "sequences": fm.get("sequences") if fm else fb_fam.get("sequences"),
                "step_rows": fm.get("step_rows") if fm else fb_fam.get("step_rows"),
                "remote_bytes": rm["bytes"] if rm else None,
                "uploaded_at": rm["mtime"] if rm else None,
            }
        )

    # Compare with the latest locally-generated manifest to flag staleness.
    local_path = get_settings().local_project_dir / "training_data" / "dataset_manifest.json"
    local_manifest = None
    if local_path.exists():
        try:
            local_manifest = json.loads(local_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            local_manifest = None
    stale = False
    if local_manifest:
        stale = not manifest or local_manifest.get("generated_at") != manifest.get(
            "generated_at"
        )

    def _sum(field: str) -> int | None:
        vals = [f[field] for f in families if f.get(field) is not None]
        return sum(vals) if vals else None

    return {
        "present": bool(remote),
        "last_upload": last_upload,
        "seed": (manifest or {}).get("seed"),
        "count_param": (manifest or {}).get("count_param"),
        "generated_at": (manifest or {}).get("generated_at"),
        "total_sequences": (manifest or {}).get("total_sequences") or _sum("sequences"),
        "total_step_rows": (manifest or {}).get("total_step_rows") or _sum("step_rows"),
        "families": families,
        "stale": stale,
        "local_total_sequences": (local_manifest or {}).get("total_sequences"),
    }


@app.get("/api/dataset/preview")
async def dataset_preview(family: str = "mosfet", lines: int = 200) -> dict:
    """Preview the first N rows of a locally-generated variant CSV (the data
    that *will* be uploaded). Reads only the head, so it's cheap even at 10k."""
    fname = DATASET_FILES.get(family.lower())
    if not fname:
        raise HTTPException(status_code=404, detail=f"Unknown family '{family}'")
    path = get_settings().local_project_dir / "training_data" / fname
    n = max(1, min(int(lines), 2000))

    def _read_head() -> dict:
        if not path.exists():
            return {"exists": False, "rows": [], "sequences_shown": 0}
        rows: list[dict] = []
        seqs: set[str] = set()
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for i, row in enumerate(reader):
                if i >= n:
                    break
                if len(row) >= 2:
                    rows.append({"seq_id": row[0].strip(), "step": row[1].strip()})
                    seqs.add(row[0].strip())
        return {"exists": True, "rows": rows, "sequences_shown": len(seqs)}

    data = await asyncio.to_thread(_read_head)
    mtime = path.stat().st_mtime if path.exists() else None
    return {"family": family.lower(), "file": fname, "lines": n, "mtime": mtime, **data}


# -- dataset collection (versioned datasets/<id>/ on Leonardo) ----------------
def _dataset_summary(
    dataset_id: str,
    manifest: dict | None,
    disk_bytes: int | None = None,
    packed: bool = False,
) -> dict:
    fams = (manifest or {}).get("families", {}) or {}
    byte_vals = [
        f.get("bytes")
        for f in fams.values()
        if isinstance(f, dict) and f.get("bytes") is not None
    ]
    manifest_bytes = sum(byte_vals) if byte_vals else None
    # training_data/ is the legacy/default dataset: it always exists and is what
    # train/eval read when nothing is selected. Mark it so the UI shows it but
    # doesn't offer to delete it.
    legacy = dataset_id == LEGACY_DATASET_ID
    ready = manifest is not None or (legacy and disk_bytes is not None)
    return {
        "id": dataset_id,
        "legacy": legacy,
        "ready": ready,
        "families": sorted(fams.keys()),
        "family_detail": fams,
        "count_param": (manifest or {}).get("count_param"),
        "seed": (manifest or {}).get("seed"),
        "total_sequences": (manifest or {}).get("total_sequences"),
        "total_step_rows": (manifest or {}).get("total_step_rows"),
        "generated_at": (manifest or {}).get("generated_at"),
        # Authoritative on-disk size (du), falling back to the manifest's sum.
        "bytes": disk_bytes if disk_bytes is not None else manifest_bytes,
        # Phase 2: a packed/ memmap blob exists -> train/eval run on the full set
        # with near-zero RAM. Drives the "packed" badge + Preprocess action.
        "packed": packed,
    }


def _parse_dataset_listing(stdout: str) -> list[dict]:
    """Parse the `===DATASET===<id> / ===SIZE===<bytes> / <manifest> / ===END===`
    stream emitted by the remote enumeration command into dataset summaries."""
    datasets: list[dict] = []
    current_id: str | None = None
    size: int | None = None
    packed: bool = False
    has_data: bool = False
    buf: list[str] = []

    def flush() -> None:
        nonlocal current_id, size, packed, has_data, buf
        if current_id is None:
            return
        manifest = None
        raw = "\n".join(buf).strip()
        if raw:
            try:
                manifest = json.loads(raw)
            except json.JSONDecodeError:
                manifest = None
        summary = _dataset_summary(current_id, manifest, size, packed)
        # Hide the legacy training_data entry once its variant CSVs are gone: the
        # folder still exists (it holds the pipeline code) but is no longer a
        # usable dataset. Versioned datasets mid-generation are kept (not legacy).
        if not (summary["legacy"] and not has_data):
            datasets.append(summary)
        current_id, size, packed, has_data, buf = None, None, False, False, []

    for line in stdout.splitlines():
        if line.startswith("===DATASET==="):
            flush()
            current_id = line[len("===DATASET===") :].strip()
            size = None
            packed = False
            has_data = False
            buf = []
        elif line.startswith("===SIZE==="):
            val = line[len("===SIZE===") :].strip()
            size = int(val) if val.isdigit() else None
        elif line.startswith("===PACKED==="):
            packed = line[len("===PACKED===") :].strip() == "1"
        elif line.startswith("===HASDATA==="):
            has_data = line[len("===HASDATA===") :].strip() == "1"
        elif line.strip() == "===END===":
            flush()
        elif current_id is not None:
            buf.append(line)
    flush()
    return datasets


@app.get("/api/datasets")
async def list_datasets() -> dict:
    """List the dataset collection on Leonardo.

    One SSH round-trip enumerates every versioned dataset folder
    (datasets/<id>/) plus the legacy/default training_data/ folder, reads each
    folder's size (du) and cats its manifest. Folders without a manifest yet
    (generation still running) come back with ready=false.
    """
    client = get_client()
    # `datasets/*/` and `training_data/` are relative to the remote workdir.
    # du -sb gives the real on-disk size; cut -f1 strips the trailing path.
    cmd = (
        f'for d in {DATASETS_DIR}/*/ {LEGACY_DATASET_ID}/; do '
        '[ -d "$d" ] || continue; '
        'echo "===DATASET===$(basename "$d")"; '
        'echo "===SIZE===$(du -sb "$d" 2>/dev/null | cut -f1)"; '
        'echo "===PACKED===$([ -f "${d}packed/meta.json" ] && echo 1 || echo 0)"; '
        'echo "===HASDATA===$(ls "${d}"*_variants.csv >/dev/null 2>&1 && echo 1 || echo 0)"; '
        '[ -f "${d}dataset_manifest.json" ] && cat "${d}dataset_manifest.json"; '
        # The manifest is written without a trailing newline, so emit one before
        # the end marker — otherwise `}` and `===END===` glue onto one line.
        'echo; '
        'echo "===END==="; '
        "done"
    )
    try:
        res = await asyncio.to_thread(client.run_in_workdir, cmd, 120)
    except Exception as exc:  # noqa: BLE001 - surface any paramiko/auth error
        raise HTTPException(status_code=502, detail=f"SSH connection failed: {exc}")
    datasets = _parse_dataset_listing(res.stdout or "")
    # Newest first, but always keep the legacy training_data default on top so
    # it's the obvious choice when no versioned dataset has been generated yet.
    datasets.sort(
        key=lambda d: (d.get("legacy", False), d.get("generated_at") or 0),
        reverse=True,
    )
    return {"datasets": datasets}


@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str) -> dict:
    """Delete a dataset from Leonardo.

    Versioned datasets (datasets/<id>/) are removed folder and all. The legacy
    training_data/ folder is special: it also holds the pipeline code/scripts, so
    we only delete its dataset artifacts (the *_variants.csv, dataset_manifest.json
    and packed/ blob) -- never the folder itself. That drops it out of the dataset
    collection (it no longer has variant CSVs) while leaving the code intact.
    """
    if not _valid_dataset_id(dataset_id):
        raise HTTPException(status_code=400, detail=f"Invalid dataset id: {dataset_id!r}")
    client = get_client()

    if dataset_id == LEGACY_DATASET_ID:
        # Only the data artifacts -- keep training_data/ (scripts live there).
        targets = [f"{LEGACY_DATASET_ID}/{fname}" for fname in DATASET_FILES.values()]
        targets.append(f"{LEGACY_DATASET_ID}/dataset_manifest.json")
        targets.append(f"{LEGACY_DATASET_ID}/packed")
        quoted = " ".join(shlex.quote(t) for t in targets)
        try:
            rm = await asyncio.to_thread(client.run_in_workdir, f"rm -rf {quoted}", 60)
        except Exception as exc:  # noqa: BLE001 - surface any paramiko/auth error
            raise HTTPException(status_code=502, detail=f"SSH connection failed: {exc}")
        if not rm.ok:
            raise HTTPException(
                status_code=500, detail=rm.stderr.strip() or "Failed to delete dataset."
            )
        return {"ok": True, "deleted": dataset_id}

    remote = f"{DATASETS_DIR}/{dataset_id}"
    quoted = shlex.quote(remote)
    try:
        check = await asyncio.to_thread(
            client.run_in_workdir,
            f"test -d {quoted} && echo OK || echo MISSING",
            30,
        )
    except Exception as exc:  # noqa: BLE001 - surface any paramiko/auth error
        raise HTTPException(status_code=502, detail=f"SSH connection failed: {exc}")
    if "OK" not in (check.stdout or ""):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    rm = await asyncio.to_thread(client.run_in_workdir, f"rm -rf {quoted}", 60)
    if not rm.ok:
        raise HTTPException(
            status_code=500, detail=rm.stderr.strip() or "Failed to delete dataset."
        )
    return {"ok": True, "deleted": dataset_id}


# -- setup --------------------------------------------------------------------
@app.post("/api/setup")
async def setup() -> dict:
    # Runs a metadata-only probe on the login node: it never imports torch, so
    # no CUDA libraries are loaded (login-node safe). GPU availability is
    # verified on compute nodes when a training/eval Slurm job starts.
    client = get_client()
    cmd = (
        'export PATH="$HOME/.pixi/bin:$PATH" && '
        "pixi run python training_data/env_check.py"
    )
    res = await asyncio.to_thread(client.run_in_workdir, cmd, 120)

    info: dict = {}
    for line in res.stdout.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                info = json.loads(stripped)
                break
            except json.JSONDecodeError:
                continue

    return {
        "ok": res.ok and bool(info.get("found")),
        "torch_version": info.get("torch_version", ""),
        "cuda_build": info.get("cuda_build", ""),
        "stdout": res.stdout.strip(),
        "stderr": res.stderr.strip(),
    }


# -- run / submit -------------------------------------------------------------
@app.post("/api/run/{run_key}")
async def submit_run(run_key: str, req: RunRequest = RunRequest()) -> dict:
    try:
        spec = get_run(run_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    validate_run_request(run_key, req)

    # Pass hyperparameters to the training script via env vars, which the Slurm
    # script forwards as CLI flags (e.g. EPOCHS -> --epochs N). Ignored for
    # non-training runs.
    export_arg = ""
    time_arg = ""
    note_suffix = ""
    dataset_id: str | None = None
    if run_key in TRAINING_RUNS:
        # Guard the transformer head/dim divisibility constraint up front so we
        # fail fast with a clear message instead of a cryptic Slurm log error.
        if (
            run_key == "transformer"
            and req.d_model
            and req.num_heads
            and int(req.d_model) % int(req.num_heads) != 0
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"d_model ({req.d_model}) must be divisible by num_heads "
                    f"({req.num_heads})."
                ),
            )

        exports: list[str] = []
        train_data_dir = _data_dir_for(req.dataset)
        if train_data_dir:
            exports.append(f"DATA_DIR={train_data_dir}")
        if req.epochs:
            exports.append(f"EPOCHS={max(1, int(req.epochs))}")
        if req.learning_rate:
            exports.append(f"LEARNING_RATE={float(req.learning_rate)}")
        if req.batch_size:
            exports.append(f"BATCH_SIZE={max(1, int(req.batch_size))}")
        if req.max_seq_len:
            exports.append(f"MAX_SEQ_LEN={max(8, int(req.max_seq_len))}")
        if req.seed is not None:
            exports.append(f"SEED={int(req.seed)}")
        if run_key == "transformer":
            if req.d_model:
                exports.append(f"D_MODEL={max(8, int(req.d_model))}")
            if req.num_layers:
                exports.append(f"NUM_LAYERS={max(1, int(req.num_layers))}")
            if req.num_heads:
                exports.append(f"NUM_HEADS={max(1, int(req.num_heads))}")
            if req.dropout is not None:
                exports.append(f"DROPOUT={float(req.dropout)}")
            if req.family_dropout is not None:
                exports.append(f"FAMILY_DROPOUT={float(req.family_dropout)}")
        # Regularization / schedule (transformer accepts these via env).
        if req.weight_decay is not None:
            exports.append(f"WEIGHT_DECAY={float(req.weight_decay)}")
        if req.label_smoothing is not None:
            exports.append(f"LABEL_SMOOTHING={float(req.label_smoothing)}")
        if req.lr_schedule:
            exports.append(f"LR_SCHEDULE={req.lr_schedule}")
        if req.warmup_ratio is not None:
            exports.append(f"WARMUP_RATIO={float(req.warmup_ratio)}")
        if req.train_ratio is not None:
            exports.append(f"TRAIN_RATIO={float(req.train_ratio)}")
        if req.val_ratio is not None:
            exports.append(f"VAL_RATIO={float(req.val_ratio)}")
        if req.max_sequences:
            exports.append(f"MAX_SEQUENCES={max(0, int(req.max_sequences))}")

        # GPU count -> auto-scale mem/cpus per Leonardo fair-share
        # (120 GB and 8 CPUs per GPU). Passed as sbatch CLI flags so they
        # override the script's single-GPU #SBATCH defaults.
        gpus = max(1, int(req.gpus)) if req.gpus else 1
        if gpus > 1:
            time_arg += f"--gpus-per-task={gpus} --mem={120 * gpus}GB --cpus-per-task={8 * gpus} "
        # Couple DataLoader workers to the allocated CPUs unless overridden.
        num_workers = req.num_workers if req.num_workers is not None else 8 * gpus
        exports.append(f"NUM_WORKERS={max(0, int(num_workers))}")

        # Multi-GPU launch: export the GPU count so the Slurm script can pick
        # torchrun, and flip DDP on (only meaningful with >1 GPU).
        exports.append(f"GPUS={gpus}")
        if req.ddp and gpus > 1:
            exports.append("DDP=1")

        if exports:
            export_arg = f"--export=ALL,{','.join(exports)} "
            note_suffix = " (" + ", ".join(e.lower() for e in exports) + ")"
        if req.time_limit:
            time_arg = f"--time={req.time_limit} " + time_arg
    elif run_key in GENERATION_RUNS:
        # Remote (on-Leonardo) data generation: forward count/seed/families and
        # write into a fresh datasets/<id>/ folder so we keep a versioned
        # collection instead of overwriting the previous dataset.
        gen_count = max(1, int(req.count)) if req.count else 1000
        gen_seed = int(req.seed) if req.seed is not None else 42
        gen_fams = [f.lower() for f in (req.families or []) if f.lower() in DATASET_FILES]
        dataset_id = _dataset_id(gen_fams or list(DATASET_FILES), gen_count, gen_seed)
        exports = [
            f"COUNT={gen_count}",
            f"SEED={gen_seed}",
            f"OUTPUT_DIR={DATASETS_DIR}/{dataset_id}",
        ]
        if gen_fams:
            exports.append(f"FAMILIES={'+'.join(gen_fams)}")
        export_arg = f"--export=ALL,{','.join(exports)} "
        note_suffix = f" -> {DATASETS_DIR}/{dataset_id}"
        if req.time_limit:
            time_arg = f"--time={req.time_limit} "
    elif run_key in PREPROCESS_RUNS:
        # One-time packing job: forward which dataset to pack (DATA_DIR) and an
        # optional per-family cap (reuses max_sequences). The packed blob lands
        # in <data_dir>/packed/ and is picked up automatically by train/eval.
        exports = []
        pp_data_dir = _data_dir_for(req.dataset)
        if pp_data_dir:
            exports.append(f"DATA_DIR={pp_data_dir}")
        if req.max_sequences:
            exports.append(f"MAX_PER_FAMILY={max(0, int(req.max_sequences))}")
        if req.seed is not None:
            exports.append(f"SEED={int(req.seed)}")
        if exports:
            export_arg = f"--export=ALL,{','.join(exports)} "
            note_suffix = " (" + ", ".join(e.lower() for e in exports) + ")"
        if req.time_limit:
            time_arg = f"--time={req.time_limit} "
    else:
        # Evaluation / baseline runs: forward the split ratios as a fallback so
        # checkpoints that didn't record them still match the training split,
        # plus any walltime override. Unused env vars are harmless.
        exports = []
        # Resolve which dataset to evaluate against. An explicit choice wins;
        # otherwise a re-eval inherits the source training job's dataset (set in
        # the source block below) so the held-out split matches AND the data is
        # actually present -- the default training_data/ may be empty/cleared.
        eval_dataset = req.dataset
        if req.train_ratio is not None:
            exports.append(f"TRAIN_RATIO={float(req.train_ratio)}")
        if req.val_ratio is not None:
            exports.append(f"VAL_RATIO={float(req.val_ratio)}")
        if req.max_sequences:
            # Fallback only; the checkpoint's recorded cap takes precedence so the
            # held-out split matches training.
            exports.append(f"MAX_SEQUENCES={max(0, int(req.max_sequences))}")
        if req.source_job_id:
            # Re-evaluate one specific past run's archived checkpoint so old runs
            # stay reproducible/inspectable instead of always hitting the latest
            # canonical model.
            src = get_store().get(req.source_job_id)
            if src is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown source job {req.source_job_id}",
                )
            if not (src.params or {}).get("keep_checkpoint"):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Job {req.source_job_id} did not archive its checkpoint. "
                        "Enable 'Keep checkpoint' when training to re-evaluate it later."
                    ),
                )
            try:
                src_spec = get_run(src.run_key)
            except KeyError:
                src_spec = None
            if src_spec is None or not src_spec.model:
                raise HTTPException(
                    status_code=400,
                    detail=f"Job {req.source_job_id} has no checkpoint to evaluate.",
                )
            exports.append(f"CHECKPOINT={archived_path(src_spec.model, req.source_job_id)}")
            if src_spec.vocab:
                exports.append(f"VOCAB={archived_path(src_spec.vocab, req.source_job_id)}")
            # Inherit the dataset the checkpoint was trained on (unless the caller
            # picked one explicitly) so DATA_DIR points at data that exists and
            # reproduces the same held-out split as training.
            if not eval_dataset:
                src_dataset = (src.params or {}).get("dataset")
                if src_dataset:
                    eval_dataset = str(src_dataset)
        # Export DATA_DIR now that any source-derived dataset is resolved.
        eval_data_dir = _data_dir_for(eval_dataset)
        if eval_data_dir:
            exports.append(f"DATA_DIR={eval_data_dir}")
        if exports:
            export_arg = f"--export=ALL,{','.join(exports)} "
            note_suffix = " (" + ", ".join(e.lower() for e in exports) + ")"
        if req.source_job_id:
            note_suffix += f" · re-eval of job {req.source_job_id}"
        if req.time_limit:
            time_arg = f"--time={req.time_limit} "

    # Make .env the single source of truth for the Slurm account/reservation and
    # the remote workdir: pass account/reservation as sbatch flags (they override
    # the script's #SBATCH fallbacks) and export REMOTE_WORKDIR so the script cd's
    # into the configured directory instead of a hardcoded one.
    settings = get_settings()
    sbatch_flags = ""
    if settings.slurm_account:
        sbatch_flags += f"--account={shlex.quote(settings.slurm_account)} "
    if settings.slurm_reservation:
        sbatch_flags += f"--reservation={shlex.quote(settings.slurm_reservation)} "
    workdir_export = f"REMOTE_WORKDIR={settings.remote_workdir}"
    if export_arg:
        export_arg = export_arg.rstrip() + f",{workdir_export} "
    else:
        export_arg = f"--export=ALL,{workdir_export} "

    client = get_client()
    res = await asyncio.to_thread(
        client.run_in_workdir,
        f"sbatch {sbatch_flags}{time_arg}{export_arg}{spec.slurm}",
    )
    if not res.ok:
        raise HTTPException(
            status_code=500, detail=res.stderr.strip() or res.stdout.strip()
        )
    job_id = parse_sbatch_job_id(res.stdout)
    params = {k: v for k, v in req.model_dump().items() if v is not None}
    # Snapshot which dataset this run trained on, so the history row is
    # self-describing even after the data is regenerated later.
    dataset_snapshot: dict = {}
    if run_key in TRAINING_RUNS:
        dataset_snapshot = await _dataset_snapshot(client)
    record = JobRecord(
        run_key=run_key,
        label=spec.label,
        job_id=job_id,
        slurm_script=spec.slurm,
        status="submitted",
        note=res.stdout.strip() + note_suffix,
        params=params,
        dataset=dataset_snapshot,
    )
    get_store().add(record)
    return {
        "job_id": job_id,
        "run_key": run_key,
        "raw": res.stdout.strip(),
        "dataset_id": dataset_id,
    }


# -- queue / job status -------------------------------------------------------
@app.get("/api/queue")
async def queue() -> dict:
    client = get_client()
    res = await asyncio.to_thread(client.run, "squeue --me", 30)
    return {"rows": parse_squeue(res.stdout), "raw": res.stdout.strip()}


@app.get("/api/jobs")
async def jobs() -> dict:
    store = get_store()
    client = get_client()
    # Refresh still-active jobs from sacct so the history reflects completion
    # promptly, archive outputs, and capture resource usage the moment a job
    # finishes (retrying capture while accounting data is still settling).
    for rec in store.all():
        if not rec.job_id:
            continue
        terminal = is_terminal_state(rec.status)
        # A failed job whose stderr wasn't flushed yet keeps polling so we can
        # still grab its error tail on a later refresh.
        needs_error = is_failure_state(rec.status) and not rec.error
        # A finished submission job keeps polling until its result CSVs land.
        needs_fetch = (
            rec.run_key == "submission"
            and not rec.submission_fetched
            and not is_failure_state(rec.status)
        )
        if terminal and rec.resources and not needs_error and not needs_fetch:
            continue
        res = await asyncio.to_thread(
            client.run,
            f"sacct -j {rec.job_id} -P --format={SACCT_RESOURCE_FMT}",
            30,
        )
        info = parse_sacct_job(res.stdout)
        state = info.get("State") if info else None
        if state and not terminal:
            store.update_status(rec.job_id, state)
            await maybe_archive(client, rec.job_id)
        fresh = store.get(rec.job_id)
        if fresh and is_terminal_state(fresh.status):
            if not fresh.resources:
                await _store_resources(client, fresh, res.stdout)
            elif is_failure_state(fresh.status) and not fresh.error:
                spec = RUNS.get(fresh.run_key)
                error = await _capture_error(
                    client, fresh, spec, parse_sacct_resources(res.stdout)
                )
                store.set_error(fresh.job_id, error)
            await maybe_fetch_submission(client, fresh.job_id)
    return {"jobs": [rec.to_dict() for rec in store.all()]}


@app.delete("/api/jobs")
async def clear_jobs(runs: str | None = None) -> dict:
    """Wipe the local job-history store. Does not touch remote files or Slurm.

    Pass ``?runs=eval_transformer`` to only clear those run types
    (e.g. the Evaluate page's history) and leave other jobs intact.
    """
    run_keys = {r for r in runs.split(",") if r} if runs else None
    removed = get_store().clear(run_keys)
    return {"removed": removed}


@app.get("/api/jobs/{job_id}/status")
async def job_status(job_id: str) -> dict:
    client = get_client()
    res = await asyncio.to_thread(
        client.run,
        f"sacct -j {job_id} -P --format={SACCT_RESOURCE_FMT}",
        30,
    )
    info = parse_sacct_job(res.stdout)
    state = info.get("State") if info else None
    if state and get_store().get(job_id):
        get_store().update_status(job_id, state)
        await maybe_archive(client, job_id)
        fresh = get_store().get(job_id)
        if fresh and is_terminal_state(fresh.status) and not fresh.resources:
            await _store_resources(client, fresh, res.stdout)
        await maybe_fetch_submission(client, job_id)
    return {"job_id": job_id, "state": state, "info": info, "raw": res.stdout.strip()}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    """Abort a pending/running job with ``scancel``.

    ``scancel`` is a lightweight Slurm control command (like sbatch/squeue), so
    it is allowed on the login node. We then re-read the state so the UI flips
    to the terminal status promptly.
    """
    if not job_id.isdigit():
        raise HTTPException(status_code=400, detail=f"Not a Slurm job id: {job_id!r}")
    client = get_client()
    store = get_store()
    rec = store.get(job_id)
    if rec and is_terminal_state(rec.status):
        return {
            "job_id": job_id,
            "cancelled": False,
            "state": rec.status,
            "detail": "Job already finished.",
        }

    # A single SSH round-trip keeps the button responsive. We optimistically
    # mark the record CANCELLED; the regular /api/jobs poll then confirms/refines
    # the real terminal state from sacct within a few seconds.
    res = await asyncio.to_thread(client.run, f"scancel {job_id}", 30)
    stderr = (res.stderr or "").strip()
    if store.get(job_id) and not stderr:
        store.update_status(job_id, "CANCELLED", note="cancel requested")
    state = "CANCELLED" if not stderr else (rec.status if rec else "unknown")
    return {"job_id": job_id, "cancelled": not stderr, "state": state, "stderr": stderr}


@app.get("/api/gpu/timeline")
async def gpu_timeline(run: str = "transformer", job_id: str | None = None) -> dict:
    """Return the nvidia-smi util/mem/power sample series for a run (live or
    archived), plus an avg/peak summary."""
    client = get_client()
    try:
        spec = get_run(run)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    path = spec.gpu_timeline
    if not path:
        return {"run": run, "rows": [], "summary": {}, "job_id": job_id}
    read_path = path
    if job_id:
        latest = get_store().latest_for_run(run)
        if latest is None or latest.job_id != job_id:
            read_path = archived_path(path, job_id)
    text = await asyncio.to_thread(client.read_remote_text, read_path)
    rows = parse_gpu_timeline(text)
    return {
        "run": run,
        "rows": rows,
        "summary": summarize_gpu_timeline(rows),
        "job_id": job_id,
    }


# -- results ------------------------------------------------------------------
@app.get("/api/results")
async def results(run: str = "eval_transformer", job_id: str | None = None) -> dict:
    try:
        spec = get_run(run)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    client = get_client()
    used_archive = False

    async def read_for(canonical: str | None) -> str | None:
        nonlocal used_archive
        if not canonical:
            return None
        if job_id:
            atext = await asyncio.to_thread(
                client.read_remote_text, archived_path(canonical, job_id)
            )
            if atext:
                used_archive = True
                return atext
        return await asyncio.to_thread(client.read_remote_text, canonical)

    summary_rows = parse_csv_rows(await read_for(spec.summary))
    rule_rows = parse_csv_rows(await read_for(spec.rule_counts))
    split: dict | None = None
    split_text = await read_for(spec.split_summary)
    if split_text:
        try:
            split = json.loads(split_text)
        except json.JSONDecodeError:
            split = None

    return {
        "run": run,
        "summary": summary_rows,
        "rule_counts": rule_rows,
        "split": split,
        "job_id": job_id,
        "archived": used_archive,
    }


@app.post("/api/validate")
async def validate(req: ValidateRequest) -> dict:
    """Run the 10-rule process-logic validator on an arbitrary sequence.

    Runs locally (the validator is pure Python, no SSH or GPU needed) so the
    UI gets an instant violation list for any pasted recipe."""
    steps = req.steps
    if steps is None and req.text:
        steps = parse_steps_text(req.text)
    steps = [s.strip().upper() for s in (steps or []) if s.strip()]

    validate_sequence = get_validator()
    violations = validate_sequence(steps)
    return {
        "steps": len(steps),
        "valid": len(violations) == 0,
        "violations": [
            {
                "rule": v.rule,
                "step_index": v.step_index,
                "step_name": v.step_name,
                "description": v.description,
            }
            for v in violations
        ],
    }


async def _canonical_log_is_fresh(client, log_path: str, rec) -> bool:
    """Whether the shared canonical log (e.g. train_log.csv) belongs to ``rec``.

    Each run overwrites the canonical log, but only once it reaches its training
    loop — which can be long after the job starts (e.g. while loading a large
    dataset). Until then the file still holds the *previous* run's curve. Treat
    it as the current run's only if it was modified at/after the run was
    submitted (60s tolerance for host/cluster clock skew). This stops the
    dashboard from showing a finished run's epochs for a job that hasn't logged
    anything yet."""
    if rec is None:
        return True  # no run context to compare against; keep legacy behaviour
    mtime = await asyncio.to_thread(client.remote_mtime, log_path)
    if mtime is None:
        return False  # no log on disk yet -> nothing to show for this run
    return mtime >= (rec.submitted_at - 60)


@app.get("/api/loss/snapshot")
async def loss_snapshot(run: str = "transformer", job_id: str | None = None) -> dict:
    try:
        spec = get_run(run)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if not spec.log:
        return {"run": run, "rows": [], "archived": False, "job_id": job_id}
    client = get_client()
    latest = get_store().latest_for_run(run)
    if job_id:
        atext = await asyncio.to_thread(
            client.read_remote_text, archived_path(spec.log, job_id)
        )
        if atext:
            return {
                "run": run,
                "rows": parse_loss_csv(atext),
                "archived": True,
                "job_id": job_id,
            }
        # No per-job archive. The canonical log belongs to whichever run last
        # wrote it, so only return it for the latest job — otherwise we'd show a
        # different run's curve for the selected job.
        if latest is None or latest.job_id != job_id:
            return {"run": run, "rows": [], "archived": False, "job_id": job_id}
    # Canonical (latest/live) log: don't serve a previous run's curve for a run
    # that hasn't written its own log yet (e.g. still loading data).
    if not await _canonical_log_is_fresh(client, spec.log, latest):
        return {"run": run, "rows": [], "archived": False, "job_id": job_id}
    text = await asyncio.to_thread(client.read_remote_text, spec.log)
    return {"run": run, "rows": parse_loss_csv(text), "archived": False, "job_id": job_id}


# -- SSE: live loss -----------------------------------------------------------
@app.get("/api/loss/stream")
async def loss_stream(run: str = "transformer", tail: bool = False):
    try:
        spec = get_run(run)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if not spec.log:
        raise HTTPException(status_code=400, detail=f"Run '{run}' has no loss log")

    client = get_client()

    async def event_generator():
        sent = 0
        first = True
        idle_after_done = 0
        for _ in range(2400):  # ~120 min safety cap at 3s
            # Ignore a stale canonical log left by a previous run until the
            # current run actually overwrites it (mirrors /api/loss/snapshot).
            fresh = await _canonical_log_is_fresh(
                client, spec.log, get_store().latest_for_run(run)
            )
            text = await asyncio.to_thread(client.read_remote_text, spec.log) if fresh else ""
            rows = parse_loss_csv(text)
            # tail=true (fresh run): skip rows that already existed at connect
            # time so we only stream the new run's epochs.
            if first and tail:
                sent = len(rows)
            first = False
            # A new run truncates+rewrites train_log.csv, so the row count can
            # shrink. Detect that and tell the client to clear stale points.
            if len(rows) < sent:
                sent = 0
                yield {"event": "reset", "data": "{}"}
            if len(rows) > sent:
                for row in rows[sent:]:
                    yield {"event": "epoch", "data": json.dumps(row)}
                sent = len(rows)

            state = await _latest_state(client, run)
            yield {"event": "status", "data": json.dumps({"state": state, "epochs": sent})}

            if is_terminal_state(state):
                rec = get_store().latest_for_run(run)
                if rec and rec.job_id:
                    await maybe_archive(client, rec.job_id)
                    await capture_resources(client, rec.job_id)
                idle_after_done += 1
                if idle_after_done >= 2:
                    yield {"event": "done", "data": json.dumps({"state": state, "epochs": sent})}
                    break
            await asyncio.sleep(3)

    return EventSourceResponse(event_generator())


# -- SSE: live logs -----------------------------------------------------------
@app.get("/api/logs/stream")
async def logs_stream(run: str = "transformer", which: str = "out"):
    try:
        spec = get_run(run)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    path = spec.out if which == "out" else spec.err
    client = get_client()

    async def event_generator():
        sent_len = 0
        idle_after_done = 0
        for _ in range(2400):
            text = await asyncio.to_thread(client.read_remote_text, path) or ""
            # Slurm truncates the .out/.err file when a new job starts, so the
            # byte length can shrink. Detect that and clear the client's buffer.
            if len(text) < sent_len:
                sent_len = 0
                yield {"event": "reset", "data": "{}"}
            if len(text) > sent_len:
                chunk = text[sent_len:]
                sent_len = len(text)
                yield {"event": "log", "data": json.dumps({"chunk": chunk})}

            state = await _latest_state(client, run)
            if is_terminal_state(state):
                rec = get_store().latest_for_run(run)
                if rec and rec.job_id:
                    await maybe_archive(client, rec.job_id)
                    await capture_resources(client, rec.job_id)
                idle_after_done += 1
                if idle_after_done >= 2:
                    yield {"event": "done", "data": json.dumps({"state": state})}
                    break
            await asyncio.sleep(2)

    return EventSourceResponse(event_generator())


async def _latest_state(client, run_key: str) -> str | None:
    record = get_store().latest_for_run(run_key)
    if record is None or not record.job_id:
        return None
    res = await asyncio.to_thread(
        client.run, f"sacct -j {record.job_id} -P --format=JobID,State", 30
    )
    info = parse_sacct_job(res.stdout)
    state = info.get("State") if info else None
    if state:
        get_store().update_status(record.job_id, state)
    return state


@app.on_event("shutdown")
async def _shutdown() -> None:
    get_client().close()
