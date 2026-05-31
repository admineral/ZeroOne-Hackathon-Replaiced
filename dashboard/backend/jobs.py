"""In-memory + on-disk store of submitted Slurm jobs."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config import get_settings


@dataclass
class JobRecord:
    run_key: str
    label: str
    job_id: str | None = None
    slurm_script: str = ""
    status: str = "submitted"
    submitted_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    note: str = ""
    params: dict = field(default_factory=dict)
    archived: bool = False
    # For run_key == "submission": result CSVs were fetched back from Leonardo.
    submission_fetched: bool = False
    resources: dict = field(default_factory=dict)
    dataset: dict = field(default_factory=dict)
    # Populated when a job ends in a failure state: {state, exit_code, message, source}.
    error: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class JobStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._records: dict[str, JobRecord] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                for job_id, data in raw.items():
                    self._records[job_id] = JobRecord(**data)
            except (json.JSONDecodeError, TypeError):
                self._records = {}

    def _persist_locked(self) -> None:
        data = {jid: rec.to_dict() for jid, rec in self._records.items()}
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add(self, record: JobRecord) -> JobRecord:
        with self._lock:
            key = record.job_id or f"local-{int(record.submitted_at * 1000)}"
            self._records[key] = record
            self._persist_locked()
            return record

    def update_status(self, job_id: str, status: str, note: str = "") -> None:
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return
            rec.status = status
            rec.updated_at = time.time()
            if note:
                rec.note = note
            self._persist_locked()

    def mark_archived(self, job_id: str) -> None:
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return
            rec.archived = True
            self._persist_locked()

    def mark_submission_fetched(self, job_id: str) -> None:
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return
            rec.submission_fetched = True
            self._persist_locked()

    def set_resources(self, job_id: str, resources: dict) -> None:
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return
            rec.resources = resources
            self._persist_locked()

    def set_error(self, job_id: str, error: dict) -> None:
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                return
            rec.error = error
            self._persist_locked()

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._records.get(job_id)

    def latest_for_run(self, run_key: str) -> JobRecord | None:
        with self._lock:
            candidates = [r for r in self._records.values() if r.run_key == run_key]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.submitted_at)

    def all(self) -> list[JobRecord]:
        with self._lock:
            return sorted(
                self._records.values(), key=lambda r: r.submitted_at, reverse=True
            )

    def clear(self, run_keys: set[str] | None = None) -> int:
        """Remove stored jobs. With ``run_keys`` only those run types are
        dropped (e.g. just the evaluation history); otherwise wipe everything."""
        with self._lock:
            if run_keys is None:
                count = len(self._records)
                self._records = {}
            else:
                to_remove = [
                    jid for jid, rec in self._records.items() if rec.run_key in run_keys
                ]
                count = len(to_remove)
                for jid in to_remove:
                    del self._records[jid]
            self._persist_locked()
            return count


_store_singleton: JobStore | None = None
_store_lock = threading.Lock()


def get_store() -> JobStore:
    global _store_singleton
    with _store_lock:
        if _store_singleton is None:
            _store_singleton = JobStore(get_settings().jobs_file)
        return _store_singleton
