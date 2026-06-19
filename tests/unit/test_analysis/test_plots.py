"""Tests for the analysis plotting layer (Phase 7).

Every figure must be rendered head-less (matplotlib ``Agg`` backend) into
``tmp_path`` and produce a real, non-empty PNG. We never open a display, touch
the GPU, or hit the network — the figures are derived purely from the loaded
ledger fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cosmos77_ex05.analysis import plots


def _assert_png(path: Path) -> None:
    """A figure path must exist, be a ``.png``, and carry real bytes."""
    assert path.exists(), f"missing figure: {path}"
    assert path.suffix == ".png"
    assert path.stat().st_size > 0, f"empty figure: {path}"


@pytest.mark.parametrize(
    "func",
    [
        plots.plot_tokens_per_sec,
        plots.plot_peak_vram,
        plots.plot_ttft_vs_tpot,
        plots.plot_quant_tradeoff,
    ],
)
def test_each_plot_writes_non_empty_png(func, ledger: dict, tmp_path: Path) -> None:
    """Each ``plot_*`` renders a non-empty PNG and returns its path."""
    out_dir = tmp_path / "figures"
    path = func(ledger, out_dir)
    _assert_png(path)


def test_plots_create_missing_out_dir(ledger: dict, tmp_path: Path) -> None:
    """A plotter creates ``out_dir`` when it does not yet exist."""
    out_dir = tmp_path / "nested" / "figures"
    assert not out_dir.exists()
    path = plots.plot_tokens_per_sec(ledger, out_dir)
    assert out_dir.is_dir()
    _assert_png(path)


def test_generate_all_returns_four_non_empty_paths(ledger: dict, tmp_path: Path) -> None:
    """``generate_all`` returns four existing, non-empty figure paths."""
    out_dir = tmp_path / "figures"
    paths = plots.generate_all(ledger, out_dir)
    assert len(paths) == 4
    assert len({p.name for p in paths}) == 4
    for path in paths:
        _assert_png(path)
