"""Tests for the ANALYSIS tables layer (DataFrame + METRICS.md).

Every assertion checks values flow *from the ledger* (the single source of
truth), never hardcoded inside the module under test. Uses the shared
``ledger``/``ledger_dir`` fixtures from ``conftest.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from cosmos77_ex05.analysis.loader import label
from cosmos77_ex05.analysis.tables import build_dataframe, write_metrics_md

# The 4-bit throughput as committed in results/airllm_4bit.json (tok/s).
EXPECTED_4BIT_TOK_S = 0.040964


def test_build_dataframe_has_four_scenario_rows(ledger: dict) -> None:
    """One row per scenario, in canonical order, hardware excluded."""
    df = build_dataframe(ledger)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert list(df["scenario"]) == [
        label("fp16_baseline"),
        label("airllm_none"),
        label("airllm_8bit"),
        label("airllm_4bit"),
    ]


def test_4bit_throughput_comes_from_ledger(ledger: dict) -> None:
    """The 4-bit throughput cell equals the measured ledger value."""
    df = build_dataframe(ledger)
    row = df.loc[df["scenario"] == label("airllm_4bit")].iloc[0]
    assert row["throughput_tok_s"] == EXPECTED_4BIT_TOK_S
    assert row["throughput_tok_s"] == ledger["airllm_4bit"]["throughput_tok_s"]


def test_tpot_seconds_derived_from_ledger_ms(ledger: dict) -> None:
    """tpot_s column is tpot_ms / 1000 from the ledger."""
    df = build_dataframe(ledger)
    row = df.loc[df["scenario"] == label("airllm_8bit")].iloc[0]
    expected = round(ledger["airllm_8bit"]["tpot_ms"] / 1000, 3)
    assert row["tpot_s"] == expected


def test_fp16_baseline_marks_oom_and_missing_perf(ledger: dict) -> None:
    """The OOM control row is success=False with NaN throughput/VRAM peak."""
    df = build_dataframe(ledger)
    row = df.loc[df["scenario"] == label("fp16_baseline")].iloc[0]
    assert bool(row["success"]) is False
    assert math.isnan(row["throughput_tok_s"])
    assert math.isnan(row["peak_vram_gb"])
    assert row["requested_vram_gb"] == ledger["fp16_baseline"]["requested_vram_gb"]


def test_write_metrics_md_default_path(ledger: dict, tmp_path: Path) -> None:
    """Default path lives under reports/METRICS.md and parent dirs are made."""
    out = write_metrics_md(ledger, tmp_path / "reports" / "METRICS.md")
    assert out.exists()
    assert out.name == "METRICS.md"
    assert out.read_text(encoding="utf-8").strip()


def test_write_metrics_md_contents(ledger: dict, tmp_path: Path) -> None:
    """The report carries the GPU, a pipe table, all 4 labels, and a real number."""
    out = write_metrics_md(ledger, tmp_path / "METRICS.md")
    text = out.read_text(encoding="utf-8")
    assert "Tesla T4" in text
    assert "|" in text  # a Markdown table
    for scenario in ("fp16_baseline", "airllm_none", "airllm_8bit", "airllm_4bit"):
        assert label(scenario) in text
    assert str(EXPECTED_4BIT_TOK_S) in text


def test_write_metrics_md_returns_path(ledger: dict, tmp_path: Path) -> None:
    """Return value is a Path pointing at the written file."""
    out = write_metrics_md(ledger, tmp_path / "sub" / "M.md")
    assert isinstance(out, Path)
    assert out == tmp_path / "sub" / "M.md"
