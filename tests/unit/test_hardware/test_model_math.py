"""Tests for the param->memory math (D1): model_memory + justify.

The worked truth from the lecture: 14.7e9 params x 2 / 1e9 = 29.4 GB in FP16
(> 16 GB T4 -> OOM), Q8 = 14.7 GB, Q4 = 7.4 GB (fits).
"""

from __future__ import annotations

import pytest

from cosmos77_ex05.hardware.model_math import justify, model_memory

_QWEN = "Qwen/Qwen2.5-14B-Instruct"
_PARAMS = 14.7e9


def test_model_memory_fp16():
    assert model_memory(_PARAMS, "fp16") == 29.4


def test_model_memory_q8():
    assert model_memory(_PARAMS, "q8") == 14.7


def test_model_memory_q4():
    assert model_memory(_PARAMS, "q4") == 7.4


def test_model_memory_unknown_dtype_raises():
    with pytest.raises(KeyError):
        model_memory(_PARAMS, "int3")


def test_justify_oom_on_t4():
    """14B in FP16 does not fit a 16 GB T4 and the verdict says OOM."""
    result = justify(_QWEN, _PARAMS, 16)

    assert result["model_id"] == _QWEN
    assert result["params"] == _PARAMS
    assert result["vram_gb"] == 16
    assert result["memory_gb"] == {"fp16": 29.4, "q8": 14.7, "q4": 7.4}
    assert result["fits_fp16"] is False
    assert "OOM" in result["verdict"]
    assert "7.4 GB" in result["verdict"]


def test_justify_fits_when_vram_is_large():
    """An 80 GB card fits FP16 -> fits_fp16 True and no OOM in the verdict."""
    result = justify(_QWEN, _PARAMS, 80)

    assert result["fits_fp16"] is True
    assert "fits without quantization" in result["verdict"]
    assert "OOM" not in result["verdict"]
