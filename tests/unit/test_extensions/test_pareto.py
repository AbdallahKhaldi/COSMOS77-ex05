"""Tests for the quantization quality-vs-speed-vs-VRAM Pareto extension (D10).

We never touch a real GPU or the network: the figure renders head-less via the
matplotlib ``Agg`` backend (selected inside the module under test), and the
ledger is the REAL measured AirLLM ledger written to a ``tmp_path`` results dir
and reloaded through the shared :func:`load_ledger`. The three real points are
all Pareto-non-dominated (lower precision buys throughput but, via bitsandbytes
dequantization buffers, costs more peak VRAM — a genuine speed-vs-VRAM trade-off
with quality flat: every output was coherent, so the accuracy red line holds).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmos77_ex05.analysis.loader import load_ledger
from cosmos77_ex05.extensions import pareto

#: The three successful AirLLM scenarios with their REAL measured numbers.
_AIRLLM = {
    "airllm_none": {"compression": "none", "throughput_tok_s": 0.007022, "peak_vram_gb": 1.595391},
    "airllm_8bit": {"compression": "8bit", "throughput_tok_s": 0.013784, "peak_vram_gb": 2.350556},
    "airllm_4bit": {"compression": "4bit", "throughput_tok_s": 0.040964, "peak_vram_gb": 3.153576},
}


def _write(results: Path, scenario: str, metrics: dict) -> None:
    """Write one ``results/<scenario>.json`` carrying the canonical scenario id."""
    payload = {"scenario": scenario, "success": True, **metrics}
    (results / f"{scenario}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def ledger_dir(tmp_path: Path) -> Path:
    """A ``results/`` dir holding the real-shaped three-scenario AirLLM ledger."""
    out = tmp_path / "results"
    out.mkdir()
    for scenario, metrics in _AIRLLM.items():
        _write(out, scenario, metrics)
    return out


@pytest.fixture
def ledger(ledger_dir: Path) -> dict:
    """The real AirLLM ledger loaded through the shared loader."""
    return load_ledger(ledger_dir)


def test_pareto_points_returns_three_real_points(ledger: dict) -> None:
    """One point per AirLLM scenario, carrying label/compression/metrics/flag."""
    points = pareto.pareto_points(ledger)
    assert len(points) == 3
    for point in points:
        assert set(point) == {
            "label",
            "compression",
            "throughput_tok_s",
            "peak_vram_gb",
            "dominated",
        }


def test_4bit_has_highest_throughput_and_vram(ledger: dict) -> None:
    """4-bit buys the most throughput but pays the most peak VRAM (dequant buffers)."""
    points = pareto.pareto_points(ledger)
    fastest = max(points, key=lambda p: p["throughput_tok_s"])
    hungriest = max(points, key=lambda p: p["peak_vram_gb"])
    assert fastest["compression"] == "4bit"
    assert hungriest["compression"] == "4bit"
    assert fastest["throughput_tok_s"] == pytest.approx(0.040964)
    assert fastest["peak_vram_gb"] == pytest.approx(3.153576)


def test_all_three_real_points_are_non_dominated(ledger: dict) -> None:
    """None of FP16/8-bit/4-bit beats another on BOTH lower VRAM and higher tok/s."""
    points = pareto.pareto_points(ledger)
    assert all(point["dominated"] is False for point in points)
    frontier = pareto.pareto_frontier(ledger)
    assert len(frontier) == 3
    vrams = [point["peak_vram_gb"] for point in frontier]
    assert vrams == sorted(vrams)


def test_strictly_worse_point_is_flagged_dominated(tmp_path: Path) -> None:
    """A synthetic point worse on BOTH axes is dominated by the better one."""
    results = tmp_path / "results"
    results.mkdir()
    # "good" wins on both axes; "bad" loses on both -> bad is dominated.
    _write(
        results,
        "airllm_4bit",
        {"compression": "4bit", "throughput_tok_s": 0.05, "peak_vram_gb": 1.0},
    )
    _write(
        results,
        "airllm_none",
        {"compression": "none", "throughput_tok_s": 0.01, "peak_vram_gb": 2.0},
    )
    points = pareto.pareto_points(load_ledger(results))
    by_comp = {p["compression"]: p for p in points}
    assert by_comp["none"]["dominated"] is True
    assert by_comp["4bit"]["dominated"] is False
    assert [p["compression"] for p in pareto.pareto_frontier(load_ledger(results))] == ["4bit"]


def test_plot_pareto_writes_non_empty_png(ledger: dict, tmp_path: Path) -> None:
    """``plot_pareto`` renders a real, non-empty PNG into ``tmp_path``."""
    out_dir = tmp_path / "figures"
    path = pareto.plot_pareto(ledger, out_dir)
    assert path.exists()
    assert path.name == "pareto.png"
    assert path.stat().st_size > 0
    assert out_dir.is_dir()
