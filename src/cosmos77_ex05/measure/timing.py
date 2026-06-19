"""Low-level timing + resource probes for the measurement harness.

Every external touch lives here behind a named function so the harness stays a
pure orchestrator and tests patch exactly one symbol per probe. ``torch`` is an
``experiment``-only optional dependency, so the VRAM probes import it lazily
inside a ``try``/``except`` and degrade to a GPU-free no-op when CUDA (or torch
itself) is absent. This keeps the suite and the developer Mac running without a
GPU while still reading real peak VRAM on the Kaggle/Colab T4.
"""

from __future__ import annotations

import time

import psutil


def default_clock() -> float:
    """Return a monotonic wall-clock timestamp in seconds.

    Wraps :func:`time.perf_counter`, the high-resolution timer used for both the
    Prefill (TTFT) and Decode (TPOT) phase timings.
    """
    return time.perf_counter()


def read_rss_bytes() -> int:
    """Return the process resident set size (peak RAM proxy) in bytes.

    Reads ``psutil.Process().memory_info().rss``. AirLLM's ``mmap`` layer
    streaming shows up here as resident host memory.
    """
    return psutil.Process().memory_info().rss


def read_peak_vram_bytes() -> int:
    """Return peak allocated VRAM in bytes since the last reset.

    Reads ``torch.cuda.max_memory_allocated()`` via a guarded import. Returns
    ``0`` when torch is not installed or CUDA is unavailable (the Mac/CI path),
    so the caller never needs a GPU and never raises.
    """
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


def reset_peak_vram() -> None:
    """Reset CUDA's peak-memory counter before a timed run.

    Calls ``torch.cuda.reset_peak_memory_stats()`` via a guarded import so a
    stale ``max_memory_allocated`` never over-reports across scenarios. A no-op
    when torch is missing or CUDA is unavailable.
    """
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats()
