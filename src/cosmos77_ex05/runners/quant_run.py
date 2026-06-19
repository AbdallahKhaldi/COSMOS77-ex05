"""Quantization sweep — run the same task at FP16 / 8bit / 4bit (D4).

Parametrises the AirLLM runner by bitsandbytes ``compression`` and records one
ledger entry per level with the full metric set plus a qualitative output-quality
note. The memory/accuracy trade-off: FP16→Q8 ~halves memory near-losslessly; Q4 is
aggressive (~7.4 GB for 14B) and is where the accuracy "red line" is watched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cosmos77_ex05.runners.airllm_run import run_airllm

#: How a level name maps to the AirLLM ``compression`` argument.
_COMPRESSION = {"none": None, "fp16": None, "8bit": "8bit", "4bit": "4bit"}


def _quality_note(output: str) -> str:
    """A neutral, honest qualitative note on the captured output (judged in the report)."""
    text = (output or "").strip()
    if not text:
        return "no output captured"
    words = len(text.split())
    return f"{words} words; captured verbatim for manual quality review (accuracy red line)"


def run_quant_sweep(
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    shards_path: str,
    levels: list[str],
    watts: float,
    *,
    runner: Callable[..., dict[str, Any]] = run_airllm,
) -> dict[str, dict[str, Any]]:
    """Run ``runner`` once per level in ``levels`` and return ``{level: metrics}``."""
    results: dict[str, dict[str, Any]] = {}
    for level in levels:
        compression = _COMPRESSION.get(level, level)
        metrics = runner(
            model_id, prompt, max_new_tokens, shards_path, watts, compression=compression
        )
        metrics["level"] = level
        metrics["quality_note"] = _quality_note(metrics.get("output", ""))
        results[level] = metrics
    return results
