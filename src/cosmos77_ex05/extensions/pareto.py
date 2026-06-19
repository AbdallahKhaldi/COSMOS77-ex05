"""Quantization quality-vs-speed-vs-VRAM Pareto frontier (extension D10).

Built entirely from the committed real ledger via the shared loader (rule 13);
no GPU, no network, no fabricated numbers. For each successful AirLLM scenario
we read its quantization level (FP16/Q8/Q4), throughput (tok/s) and peak VRAM
(GB), then flag whether the point is Pareto-dominated.

THE FINDING: with AirLLM, dropping precision buys throughput (4-bit is ~6x FP16)
but — counterintuitively — costs MORE peak VRAM (1.6 -> 2.4 -> 3.2 GB), because
bitsandbytes keeps dequantization buffers while the resident weights stay tiny
(only one layer pages in at a time). So it is a genuine speed-vs-VRAM trade-off
and all three points are Pareto-non-dominated. Quality is the flat third axis:
every output was coherent, so the accuracy "red line" was never crossed at Q4.

The ``Agg`` backend is selected before importing ``pyplot`` so the figure renders
head-less.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from cosmos77_ex05.analysis.loader import airllm_rows  # noqa: E402

Ledger = dict[str, dict[str, Any]]

#: Quantization level labels keyed by the ledger ``compression`` field.
_LEVEL: dict[str, str] = {"none": "FP16", "8bit": "8-bit", "4bit": "4-bit"}


def _level(compression: str) -> str:
    """Lecture label (FP16/8-bit/4-bit) for a ledger ``compression`` value."""
    return _LEVEL.get(compression, compression)


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True when ``a`` Pareto-dominates ``b``: better on both axes, strict on one.

    "Better" means higher throughput AND lower peak VRAM; at least one of the two
    comparisons must be strict so identical points never dominate each other.
    """
    not_worse = (
        a["throughput_tok_s"] >= b["throughput_tok_s"] and a["peak_vram_gb"] <= b["peak_vram_gb"]
    )
    strictly_better = (
        a["throughput_tok_s"] > b["throughput_tok_s"] or a["peak_vram_gb"] < b["peak_vram_gb"]
    )
    return not_worse and strictly_better


def pareto_points(ledger: Ledger) -> list[dict[str, Any]]:
    """Return one labelled, domination-flagged point per AirLLM scenario.

    Each dict carries ``label`` (FP16/8-bit/4-bit), ``compression``,
    ``throughput_tok_s``, ``peak_vram_gb`` and ``dominated`` — the last being
    True iff some other point beats it on both lower-VRAM and higher-throughput.
    """
    points: list[dict[str, Any]] = []
    for _, metrics in airllm_rows(ledger):
        compression = metrics["compression"]
        points.append(
            {
                "label": _level(compression),
                "compression": compression,
                "throughput_tok_s": metrics["throughput_tok_s"],
                "peak_vram_gb": metrics["peak_vram_gb"],
                "dominated": False,
            }
        )
    for point in points:
        point["dominated"] = any(_dominates(other, point) for other in points if other is not point)
    return points


def pareto_frontier(ledger: Ledger) -> list[dict[str, Any]]:
    """The non-dominated points, sorted by ascending peak VRAM (GB)."""
    frontier = [point for point in pareto_points(ledger) if not point["dominated"]]
    return sorted(frontier, key=lambda point: point["peak_vram_gb"])


def plot_pareto(ledger: Ledger, out_dir: Path | str = "figures") -> Path:
    """Scatter throughput (y) vs peak VRAM (x); annotate the Pareto frontier.

    Each point is labelled FP16/8-bit/4-bit; the non-dominated frontier is drawn
    as a connecting line. The title states the trade-off: quantization buys
    throughput at a peak-VRAM cost via bitsandbytes dequantization buffers.
    Saves ``out_dir/pareto.png`` (creating the directory) and closes the figure.
    """
    points = pareto_points(ledger)
    frontier = pareto_frontier(ledger)
    fig, ax = plt.subplots()
    ax.scatter(
        [p["peak_vram_gb"] for p in points],
        [p["throughput_tok_s"] for p in points],
        color="#4c72b0",
        zorder=3,
    )
    ax.plot(
        [p["peak_vram_gb"] for p in frontier],
        [p["throughput_tok_s"] for p in frontier],
        color="#c44e52",
        linestyle="--",
        marker="o",
        label="Pareto frontier",
        zorder=2,
    )
    for point in points:
        ax.annotate(
            point["label"],
            (point["peak_vram_gb"], point["throughput_tok_s"]),
            textcoords="offset points",
            xytext=(6, 6),
        )
    ax.set_xlabel("Peak VRAM (GB)")
    ax.set_ylabel("Throughput (tokens / s)")
    ax.set_title("Quantization Pareto: lower precision buys throughput at a VRAM cost")
    ax.legend()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "pareto.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
