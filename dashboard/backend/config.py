"""Configuration for the Leonardo Pipeline Dashboard backend.

Secrets (the Leonardo password) are read from a local ``.env`` file that is
gitignored. Nothing here is ever sent to the browser.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = BACKEND_DIR.parent
REPO_ROOT = DASHBOARD_DIR.parent

load_dotenv(BACKEND_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    host: str
    user: str
    password: str
    port: int
    remote_workdir: str
    slurm_account: str
    slurm_reservation: str
    local_project_dir: Path
    jobs_file: Path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    local_project = os.getenv(
        "LOCAL_PROJECT_DIR", str(REPO_ROOT / "industrial-infineon")
    )
    return Settings(
        host=os.getenv("LEONARDO_HOST", "login01-ext.leonardo.cineca.it"),
        user=os.getenv("LEONARDO_USER", "a08trc0x"),
        password=os.getenv("LEONARDO_PASSWORD", ""),
        port=int(os.getenv("LEONARDO_PORT", "22")),
        remote_workdir=os.getenv(
            "REMOTE_WORKDIR", "/leonardo_work/EUHPC_D30_031/industrial-infineon"
        ),
        slurm_account=os.getenv("SLURM_ACCOUNT", "EUHPC_D30_031"),
        slurm_reservation=os.getenv("SLURM_RESERVATION", "s_tra_ncc"),
        local_project_dir=Path(local_project),
        jobs_file=BACKEND_DIR / "jobs.json",
    )
