"""Parsers for Slurm command output (sbatch / squeue / sacct)."""

from __future__ import annotations

import re

_SBATCH_RE = re.compile(r"Submitted batch job (\d+)")


def parse_sbatch_job_id(text: str) -> str | None:
    match = _SBATCH_RE.search(text or "")
    return match.group(1) if match else None


def parse_squeue(text: str) -> list[dict[str, str]]:
    """Parse the default ``squeue --me`` table into row dicts."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].split()
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < len(header):
            continue
        # NODELIST(REASON) can contain spaces only inside parens; join the tail.
        row = dict(zip(header[:-1], parts[: len(header) - 1]))
        row[header[-1]] = " ".join(parts[len(header) - 1 :])
        rows.append(row)
    return rows


def parse_sacct_job(text: str) -> dict[str, str] | None:
    """Return the top-level job row from ``sacct -j <id> -P`` output as a dict.

    The first data line is the job itself; subsequent ``.batch`` / ``.extern``
    steps are ignored so callers see a single clean State/Elapsed/etc.
    """
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    header = [h.strip() for h in lines[0].split("|")]
    for line in lines[1:]:
        fields = [f.strip() for f in line.split("|")]
        job_id = fields[0] if fields else ""
        # Skip step rows like "12345.batch" / "12345.extern".
        if "." in job_id:
            continue
        return dict(zip(header, fields))
    # Fall back to the first row if only step rows exist.
    return dict(zip(header, [f.strip() for f in lines[1].split("|")]))


def _mem_to_mb(value: str | None) -> float | None:
    """Convert a Slurm memory string (e.g. '1234K', '512M', '2.5G') to MB."""
    if not value:
        return None
    v = value.strip()
    if not v or v in {"0", "-"}:
        return 0.0
    units = {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024}
    mult = 1.0
    if v[-1] in units:
        mult = units[v[-1]]
        v = v[:-1]
    try:
        return round(float(v) * mult, 1)
    except ValueError:
        return None


def _parse_tres(tres: str | None) -> dict:
    """Parse an AllocTRES/ReqTRES string like 'cpu=8,mem=64G,gres/gpu=1'."""
    out: dict = {}
    for part in (tres or "").split(","):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key == "cpu":
            try:
                out["cpus"] = int(val)
            except ValueError:
                pass
        elif key == "mem":
            out["alloc_mem_mb"] = _mem_to_mb(val)
        elif key in ("gres/gpu", "gpu"):
            try:
                out["gpus"] = int(val)
            except ValueError:
                pass
    return out


def parse_sacct_resources(text: str) -> dict:
    """Extract resource usage from ``sacct -j <id> -P`` rich output.

    Memory high-water marks (MaxRSS/MaxVMSize) live on the ``.batch`` step rows,
    while State/Elapsed/AllocTRES live on the main job row, so we read both.
    """
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}
    header = [h.strip() for h in lines[0].split("|")]
    rows = [dict(zip(header, [f.strip() for f in ln.split("|")])) for ln in lines[1:]]

    main = next((r for r in rows if "." not in (r.get("JobID") or "")), rows[0])
    max_rss = 0.0
    max_vm = 0.0
    for r in rows:
        rss = _mem_to_mb(r.get("MaxRSS"))
        vm = _mem_to_mb(r.get("MaxVMSize"))
        if rss is not None:
            max_rss = max(max_rss, rss)
        if vm is not None:
            max_vm = max(max_vm, vm)

    tres = _parse_tres(main.get("AllocTRES"))
    return {
        "state": main.get("State"),
        "elapsed": main.get("Elapsed"),
        "total_cpu": main.get("TotalCPU"),
        "req_mem": main.get("ReqMem"),
        "exit_code": main.get("ExitCode"),
        "max_rss_mb": max_rss or None,
        "max_vmsize_mb": max_vm or None,
        "alloc_tres": main.get("AllocTRES"),
        "cpus": tres.get("cpus"),
        "alloc_mem_mb": tres.get("alloc_mem_mb"),
        "gpus": tres.get("gpus"),
    }


SACCT_RESOURCE_FMT = (
    "JobID,JobName,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,"
    "ReqMem,AllocTRES,ExitCode,Start,End"
)
