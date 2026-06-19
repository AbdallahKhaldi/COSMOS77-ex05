"""Tests for the Phase-8 break-even plot (D7 acceptance figure).

The figure must render head-less (matplotlib ``Agg``) into ``tmp_path`` and
produce a real, non-empty PNG — no display, no GPU, no network. Every value
flows from the cost-model CORE + config; nothing is hard-coded here.
"""

from __future__ import annotations

from pathlib import Path

from cosmos77_ex05.economics import breakeven_plot


def test_plot_breakeven_writes_non_empty_png(config, tmp_path: Path) -> None:
    """``plot_breakeven`` saves a real PNG and returns its path."""
    out_dir = tmp_path / "figures"
    path = breakeven_plot.plot_breakeven(
        config,
        tokens_in=500,
        tokens_out=200,
        runtime_s=10.0,
        provider="google_gemini_flash",
        out_dir=out_dir,
    )
    assert path.exists()
    assert path.suffix == ".png"
    assert path.name == "breakeven.png"
    assert path.stat().st_size > 0


def test_plot_breakeven_creates_missing_out_dir(config, tmp_path: Path) -> None:
    """The plotter creates ``out_dir`` when it does not yet exist."""
    out_dir = tmp_path / "nested" / "figures"
    assert not out_dir.exists()
    path = breakeven_plot.plot_breakeven(
        config,
        tokens_in=500,
        tokens_out=200,
        runtime_s=10.0,
        provider="openai_gpt4o_mini",
        out_dir=out_dir,
    )
    assert out_dir.is_dir()
    assert path.stat().st_size > 0


def test_plot_breakeven_renders_when_no_crossover(config, tmp_path: Path) -> None:
    """A long runtime gives no break-even; the figure still renders (no crossover mark)."""
    path = breakeven_plot.plot_breakeven(
        config,
        tokens_in=500,
        tokens_out=200,
        runtime_s=100.0,
        provider="google_gemini_flash",
        out_dir=tmp_path / "figures",
    )
    assert path.stat().st_size > 0
