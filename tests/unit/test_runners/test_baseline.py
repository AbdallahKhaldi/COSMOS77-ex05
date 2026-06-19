"""Tests for the FP16 baseline runner — OOM captured, non-OOM re-raised (mocked)."""

from __future__ import annotations

import pytest

from cosmos77_ex05.runners.baseline import run_fp16_baseline


class OutOfMemoryError(Exception):
    """Stands in for torch.cuda.OutOfMemoryError without importing torch."""


def test_oom_is_captured_with_memory_math():
    def loader(model_id):
        raise OutOfMemoryError("CUDA out of memory")

    record = run_fp16_baseline("Qwen/Qwen2.5-14B-Instruct", "hi", 14.7e9, 16.0, loader=loader)
    assert record["success"] is False
    assert record["error_type"] == "OutOfMemoryError"
    assert record["requested_vram_gb"] == 29.4
    assert record["available_vram_gb"] == 16.0
    assert record["memory_gb"]["q4"] == 7.4
    assert "memory" in record["bottleneck"]


def test_non_oom_error_reraises():
    def loader(model_id):
        raise ValueError("a real bug, not OOM")

    with pytest.raises(ValueError):
        run_fp16_baseline("m", "hi", 14.7e9, 16.0, loader=loader)


def test_unexpected_success_is_recorded():
    def loader(model_id):
        return ("model", "tok")

    def generate(model, tok, prompt, n):
        return "an answer"

    record = run_fp16_baseline("m", "hi", 1e9, 16.0, loader=loader, generate=generate)
    assert record["success"] is True
    assert record["output"] == "an answer"
