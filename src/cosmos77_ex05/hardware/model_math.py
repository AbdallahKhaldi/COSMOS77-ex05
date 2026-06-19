"""The param->memory math (D1): how many GB a model needs at each precision.

Why: decode is memory-bound, so whether a model *fits* is decided by
``params x bytes_per_param``. This module turns that lecture identity into a
verdict — e.g. Qwen2.5-14B is 29.4 GB in FP16 (OOM on a 16 GB T4) but only
7.4 GB in Q4 (fits). Quantization moves it from impossible to runnable.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from cosmos77_ex05.constants import BYTES_PER_PARAM


def model_memory(params: float, dtype: str) -> float:
    """Return the weight memory (decimal GB) for ``params`` at ``dtype``.

    Computes ``params * bytes_per_param / 1e9`` and rounds to 1 decimal with
    round-half-up so the lecture's worked truth holds exactly (Q4 of 14.7e9 =
    7.35 -> 7.4 GB, not 7.3 from Python's default banker's rounding).

    Args:
        params: Parameter count (e.g. ``14.7e9``).
        dtype: One of ``"fp16"``, ``"q8"``, ``"q4"`` (see ``BYTES_PER_PARAM``).

    Returns:
        The weight memory in decimal GB, rounded to one decimal place.

    Raises:
        KeyError: If ``dtype`` is not a known precision.
    """
    raw = Decimal(params * BYTES_PER_PARAM[dtype]) / Decimal("1e9")
    return float(raw.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def justify(model_id: str, params: float, vram_gb: float) -> dict:
    """Build the fits/OOM verdict for ``model_id`` against ``vram_gb`` of VRAM.

    Args:
        model_id: Hugging Face id, e.g. ``"Qwen/Qwen2.5-14B-Instruct"``.
        params: Parameter count.
        vram_gb: Available VRAM in GB (e.g. ``16`` for a free T4).

    Returns:
        A dict with per-precision ``memory_gb``, ``fits_fp16`` and a
        lecture-tied ``verdict`` string.
    """
    memory_gb = {dtype: model_memory(params, dtype) for dtype in BYTES_PER_PARAM}
    fits_fp16 = memory_gb["fp16"] <= vram_gb
    short = model_id.split("/")[-1]
    if fits_fp16:
        verdict = (
            f"{short} needs {memory_gb['fp16']} GB in FP16 <= {vram_gb} GB VRAM "
            f"-> fits without quantization (memory-bound, but within budget)."
        )
    else:
        verdict = (
            f"{short} needs {memory_gb['fp16']} GB in FP16 > {vram_gb} GB VRAM "
            f"-> OOM; Q4 ({memory_gb['q4']} GB) fits."
        )
    return {
        "model_id": model_id,
        "params": params,
        "vram_gb": vram_gb,
        "memory_gb": memory_gb,
        "fits_fp16": fits_fp16,
        "verdict": verdict,
    }
