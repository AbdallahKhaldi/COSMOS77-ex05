"""Capture the host hardware spec (D1) — CPU, RAM, GPU/VRAM, disk, platform.

Why: the whole HW5 argument is *memory-bound*. Before we can justify AirLLM +
quantization we must record the box we run on — crucially its VRAM, since a
14B model in FP16 needs ~29 GB that no free T4 (16 GB) can hold. The GPU probe
is guarded so this runs on the student's torch-free Mac AND on the Kaggle T4.
"""

from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path

import psutil

from cosmos77_ex05.constants import DEFAULT_ENCODING

#: Sentinel name used when no CUDA device (or no torch) is available.
_NO_GPU_NAME = "none / CPU-only"


def _gpu_info() -> dict[str, str | float | None]:
    """Return the CUDA GPU name + VRAM in GB, or a CPU-only sentinel.

    ``torch`` lives in the optional ``experiment`` extra and is absent off the
    T4, so the import is guarded: any ``ImportError`` or missing CUDA collapses
    to ``{"name": "none / CPU-only", "vram_gb": None}``.
    """
    try:
        import torch
    except ImportError:
        return {"name": _NO_GPU_NAME, "vram_gb": None}
    if not torch.cuda.is_available():
        return {"name": _NO_GPU_NAME, "vram_gb": None}
    props = torch.cuda.get_device_properties(0)
    return {"name": props.name, "vram_gb": round(props.total_memory / 1e9, 1)}


def _cpu_model() -> str:
    """Return a human CPU label, falling back to the machine arch."""
    return platform.processor() or platform.machine()


def capture_spec(out_path: Path | str) -> dict:
    """Capture the host hardware spec, write it as JSON, and return it.

    Args:
        out_path: Destination JSON path; parent directories are created.

    Returns:
        The spec dict (cpu, ram_gb, gpu, disk, platform).
    """
    usage = shutil.disk_usage("/")
    spec: dict = {
        "cpu": {
            "model": _cpu_model(),
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
        },
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "gpu": _gpu_info(),
        "disk": {
            "total_gb": round(usage.total / 1e9, 1),
            "free_gb": round(usage.free / 1e9, 1),
        },
        "platform": platform.platform(),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2), encoding=DEFAULT_ENCODING)
    return spec
