#!/usr/bin/env python3
"""Lightweight environment probe for the dashboard's "Check environment" step.

Reports the installed PyTorch build by reading package **metadata** only — it
never ``import torch``, so no CUDA shared libraries are loaded. That makes it
safe to run on a Leonardo login node (short, CPU-only, no heavy compute).

The real "can torch see a GPU" check happens on a compute node: every training
and evaluation Slurm job prints the device and GPU name when it starts.
"""

from __future__ import annotations

import json
import pathlib
import re
from importlib import metadata


def torch_build() -> dict:
    try:
        version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        return {"found": False, "torch_version": "", "cuda_build": ""}

    cuda_build = ""
    # Read torch/version.py (a ~1 KB file) to recover the CUDA build string
    # without importing the package. Located via the dist RECORD, not by import.
    try:
        files = metadata.files("torch") or []
        version_py = next(
            (
                f
                for f in files
                if f.name == "version.py" and f.parent.name == "torch"
            ),
            None,
        )
        if version_py is not None:
            text = pathlib.Path(version_py.locate()).read_text(encoding="utf-8")
            match = re.search(r"^cuda\s*=\s*['\"]?([^'\"\n]+)", text, re.MULTILINE)
            if match:
                cuda_build = match.group(1).strip()
    except Exception:  # noqa: BLE001 - metadata layout varies; fall back below
        cuda_build = ""

    if cuda_build.lower() in {"", "none"}:
        # pip wheels encode CUDA in the version tag, e.g. "2.3.1+cu121".
        # Normalise "cu121" -> "12.1" (all but last digit = major, last = minor).
        tag = re.search(r"\+cu(\d+)", version)
        if tag:
            digits = tag.group(1)
            cuda_build = f"{digits[:-1]}.{digits[-1]}" if len(digits) >= 2 else digits
        else:
            cuda_build = "cpu"

    return {"found": True, "torch_version": version, "cuda_build": cuda_build}


if __name__ == "__main__":
    print(json.dumps(torch_build()))
