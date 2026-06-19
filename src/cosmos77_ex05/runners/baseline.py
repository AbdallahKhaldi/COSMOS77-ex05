"""The naive FP16 baseline — load Qwen2.5-14B directly and capture the OOM (D2).

This is the control condition: a 14B model in FP16 needs ~29.4 GB but a T4 has
16 GB, so the direct ``device_map="cuda"`` load is expected to raise
``torch.cuda.OutOfMemoryError``. We catch it, record the requested-vs-available
VRAM and the param→memory math, and diagnose the bottleneck as MEMORY (not compute).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cosmos77_ex05.hardware.model_math import model_memory
from cosmos77_ex05.runners._common import (
    Loader,
    default_transformers_loader,
    is_oom,
    transformers_generate,
)


def run_fp16_baseline(
    model_id: str,
    prompt: str,
    params: float,
    vram_gb: float,
    n_out: int = 1,
    *,
    loader: Loader | None = None,
    generate: Callable[[Any, Any, str, int], str] | None = None,
) -> dict[str, Any]:
    """Attempt the naive FP16 load+generate; return the OOM record or the (rare) success.

    On OOM ``success=False`` with the memory math; any non-OOM error re-raises (a
    real bug should not masquerade as the expected control-condition failure).
    """
    load = loader or default_transformers_loader
    gen = generate or transformers_generate
    requested_gb = model_memory(params, "fp16")
    record: dict[str, Any] = {
        "success": False,
        "scenario": "fp16_baseline",
        "model_id": model_id,
        "dtype": "fp16",
        "requested_vram_gb": requested_gb,
        "available_vram_gb": vram_gb,
        "memory_gb": {dt: model_memory(params, dt) for dt in ("fp16", "q8", "q4")},
        "bottleneck": "memory (VRAM capacity)",
    }
    try:
        model, tokenizer = load(model_id)
        output = gen(model, tokenizer, prompt, n_out)
    except BaseException as exc:  # noqa: BLE001 - we re-raise anything that is not OOM
        if not is_oom(exc):
            raise
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)[:500]
        record["note"] = (
            f"FP16 needs {requested_gb} GB > {vram_gb} GB VRAM -> OOM. "
            "Bottleneck is memory capacity, not compute."
        )
        return record
    record["success"] = True
    record["output"] = output
    record["note"] = "Loaded without OOM (unexpected on a 16 GB T4)."
    return record
