"""Shared fixture: a results/ dir holding the REAL-shaped measured ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_LEDGER = {
    "hardware": {
        "cpu": {"model": "x86_64", "cores_physical": 2, "cores_logical": 4},
        "ram_gb": 33.7,
        "gpu": {"name": "Tesla T4", "vram_gb": 15.6},
        "disk": {"total_gb": 8656.9, "free_gb": 1182.8},
        "platform": "Linux",
    },
    "fp16_baseline": {
        "scenario": "fp16_baseline",
        "success": False,
        "dtype": "fp16",
        "requested_vram_gb": 29.4,
        "available_vram_gb": 16.0,
        "memory_gb": {"fp16": 29.4, "q8": 14.7, "q4": 7.4},
        "error_type": "OutOfMemoryError",
        "bottleneck": "memory (VRAM capacity)",
    },
    "airllm_none": {
        "scenario": "airllm_none",
        "success": True,
        "compression": "none",
        "throughput_tok_s": 0.007022,
        "ttft_s": 142.336688,
        "tpot_ms": 142423.526723,
        "peak_vram_gb": 1.595391,
        "peak_ram_gb": 4.787581,
        "total_s": 2848.383695,
        "est_power_wh": 55.385239,
        "n_out": 20,
        "output": "The attention mechanism ...",
    },
    "airllm_8bit": {
        "scenario": "airllm_8bit",
        "success": True,
        "compression": "8bit",
        "level": "8bit",
        "throughput_tok_s": 0.013784,
        "ttft_s": 70.093653,
        "tpot_ms": 72679.040485,
        "peak_vram_gb": 2.350556,
        "peak_ram_gb": 4.804698,
        "total_s": 1450.995422,
        "est_power_wh": 28.2138,
        "n_out": 20,
        "quality_note": "25 words; captured verbatim",
    },
    "airllm_4bit": {
        "scenario": "airllm_4bit",
        "success": True,
        "compression": "4bit",
        "level": "4bit",
        "throughput_tok_s": 0.040964,
        "ttft_s": 45.652846,
        "tpot_ms": 23294.021335,
        "peak_vram_gb": 3.153576,
        "peak_ram_gb": 4.815229,
        "total_s": 488.239252,
        "est_power_wh": 9.493541,
        "n_out": 20,
        "quality_note": "25 words; captured verbatim",
    },
}


@pytest.fixture
def ledger_dir(tmp_path: Path) -> Path:
    """A results/ directory populated with the real-shaped measured ledger."""
    out = tmp_path / "results"
    out.mkdir()
    for scenario, metrics in _LEDGER.items():
        (out / f"{scenario}.json").write_text(json.dumps(metrics), encoding="utf-8")
    return out


@pytest.fixture
def ledger(ledger_dir: Path) -> dict:
    """The loaded ledger dict."""
    from cosmos77_ex05.analysis.loader import load_ledger

    return load_ledger(ledger_dir)
