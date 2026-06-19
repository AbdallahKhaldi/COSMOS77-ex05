"""Tests for the ANALYSIS Roofline layer (D9 advanced deliverable).

The Roofline model places each measured AirLLM scenario on the
arithmetic-intensity (FLOPs/byte) vs attainable-throughput (TFLOP/s) plane.
Every number is derived *from the ledger* (the single source of truth), never
hardcoded inside ``roofline.py``. The key finding under test: autoregressive
decode is memory-bound, and AirLLM's *effective* bandwidth lands at disk speed
(~0.2-0.3 GB/s), ~1000x below the T4's 320 GB/s HBM.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no GPU, no display, no network

from cosmos77_ex05.analysis import roofline
from cosmos77_ex05.analysis.loader import label

# Measured throughputs as committed in results/airllm_*.json (tok/s).
TOK_S_NONE = 0.007022
TOK_S_4BIT = 0.040964
# Qwen2.5-14B parameter count and decode FLOP factor (2 FLOPs per weight).
PARAMS = 14.7e9


def test_ridge_point_is_about_203() -> None:
    """The T4 ridge (peak FLOP/s / peak B/s) is ~203 FLOPs/byte."""
    assert 200.0 <= roofline.RIDGE <= 205.0


def test_4bit_arithmetic_intensity_is_four(ledger: dict) -> None:
    """4-bit moves 0.5 byte/param, so AI = 2 FLOPs / 0.5 byte = 4.0 FLOP/byte."""
    point = roofline.scenario_point("airllm_4bit", ledger["airllm_4bit"])
    assert point["arithmetic_intensity"] == 4.0


def test_4bit_is_memory_bound(ledger: dict) -> None:
    """AI (4) sits far left of the ridge (~203), so decode is memory-bound."""
    point = roofline.scenario_point("airllm_4bit", ledger["airllm_4bit"])
    assert point["bound"] == "memory-bound"
    assert point["arithmetic_intensity"] < roofline.RIDGE


def test_4bit_achieved_flops_from_ledger(ledger: dict) -> None:
    """achieved_flops = throughput * 2 * params, sourced from the ledger."""
    point = roofline.scenario_point("airllm_4bit", ledger["airllm_4bit"])
    expected = TOK_S_4BIT * 2 * PARAMS
    assert point["achieved_flops"] == expected
    assert point["achieved_tflops"] == expected / 1e12


def test_fp16_effective_bandwidth_is_disk_speed(ledger: dict) -> None:
    """FP16 effective BW ~0.206 GB/s — disk, not the 320 GB/s HBM wall."""
    point = roofline.scenario_point("airllm_none", ledger["airllm_none"])
    expected = PARAMS * 2 * TOK_S_NONE / 1e9
    assert point["effective_bandwidth_gbs"] == expected
    assert 0.20 <= point["effective_bandwidth_gbs"] <= 0.21
    assert point["effective_bandwidth_gbs"] < roofline.T4_BW_GBs / 1000


def test_scenario_point_label_is_pretty(ledger: dict) -> None:
    """The point carries the human label from the shared loader."""
    point = roofline.scenario_point("airllm_8bit", ledger["airllm_8bit"])
    assert point["label"] == label("airllm_8bit")


def test_roofline_points_covers_three_airllm_scenarios(ledger: dict) -> None:
    """Exactly the 3 successful AirLLM scenarios, baseline excluded."""
    points = roofline.roofline_points(ledger)
    assert [p["label"] for p in points] == [
        label("airllm_none"),
        label("airllm_8bit"),
        label("airllm_4bit"),
    ]
    assert all(p["bound"] == "memory-bound" for p in points)


def test_roofline_points_intensity_ordering(ledger: dict) -> None:
    """Heavier quantization moves fewer bytes/param -> higher AI."""
    points = {p["label"]: p for p in roofline.roofline_points(ledger)}
    ai_none = points[label("airllm_none")]["arithmetic_intensity"]
    ai_4bit = points[label("airllm_4bit")]["arithmetic_intensity"]
    assert ai_4bit > ai_none


def test_plot_roofline_writes_non_empty_png(ledger: dict, tmp_path: Path) -> None:
    """plot_roofline saves a non-empty PNG and returns its Path."""
    out = roofline.plot_roofline(ledger, out_dir=tmp_path)
    assert isinstance(out, Path)
    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 0


def test_plot_roofline_creates_missing_out_dir(ledger: dict, tmp_path: Path) -> None:
    """A missing output directory is created on the fly."""
    target = tmp_path / "figures" / "nested"
    out = roofline.plot_roofline(ledger, out_dir=target)
    assert out.parent == target
    assert out.exists()
