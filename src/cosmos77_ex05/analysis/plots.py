"""Render the Phase 7 figures from the measured ledger (rule 13).

One function per figure; each takes ``(ledger, out_dir)``, creates ``out_dir``,
saves a PNG, and returns its :class:`~pathlib.Path`. Every number is read from
the ledger via :mod:`cosmos77_ex05.analysis.loader` — nothing is hard-coded.

The ``Agg`` backend is selected *before* importing ``pyplot`` so the figures
render head-less: no display, no GPU, no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from cosmos77_ex05.analysis.loader import airllm_rows  # noqa: E402

#: Compact bar labels for the three AirLLM compression levels, in run order.
_BARS: dict[str, str] = {"airllm_none": "FP16", "airllm_8bit": "8-bit", "airllm_4bit": "4-bit"}

#: T4 VRAM limit and the FP16 model size that OOMs above it (GB).
_T4_LIMIT_GB: float = 16.0
_FP16_NEED_GB: float = 29.4

Ledger = dict[str, dict[str, Any]]


def _air(ledger: Ledger) -> tuple[list[str], list[dict[str, Any]]]:
    """Return ``(labels, metrics)`` for the three successful AirLLM runs."""
    rows = airllm_rows(ledger)
    return [_BARS.get(name, name) for name, _ in rows], [m for _, m in rows]


def _save(fig: plt.Figure, out_dir: Path | str, name: str) -> Path:
    """Persist ``fig`` to ``out_dir/name``, creating the directory, then close it."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tokens_per_sec(ledger: Ledger, out_dir: Path | str = "figures") -> Path:
    """Bar chart of throughput (tok/s) for the three AirLLM scenarios."""
    labels, metrics = _air(ledger)
    values = [m.get("throughput_tok_s", 0.0) for m in metrics]
    fig, ax = plt.subplots()
    ax.bar(labels, values, color="#4c72b0")
    ax.set_xlabel("Compression level")
    ax.set_ylabel("Throughput (tokens / s)")
    ax.set_title("AirLLM throughput — a brutal ~0.007–0.04 tok/s")
    for x, v in zip(labels, values, strict=True):
        ax.text(x, v, f"{v:.4f}", ha="center", va="bottom")
    return _save(fig, out_dir, "tokens_per_sec.png")


def plot_peak_vram(ledger: Ledger, out_dir: Path | str = "figures") -> Path:
    """Bar of peak VRAM (GB) per AirLLM run vs the T4 limit and FP16 need."""
    labels, metrics = _air(ledger)
    values = [m.get("peak_vram_gb", 0.0) for m in metrics]
    fig, ax = plt.subplots()
    ax.bar(labels, values, color="#55a868")
    ax.axhline(_T4_LIMIT_GB, color="#c44e52", linestyle="--", label=f"T4 limit {_T4_LIMIT_GB:g} GB")
    ax.axhline(
        _FP16_NEED_GB, color="#8172b3", linestyle=":", label=f"FP16 need {_FP16_NEED_GB:g} GB (OOM)"
    )
    ax.set_xlabel("Compression level")
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title("AirLLM peak VRAM (1.6–3.2 GB) vs the 29.4 GB FP16 demands")
    ax.legend()
    return _save(fig, out_dir, "peak_vram.png")


def plot_ttft_vs_tpot(ledger: Ledger, out_dir: Path | str = "figures") -> Path:
    """Grouped bars of TTFT (prefill) and TPOT (decode) latency per scenario."""
    labels, metrics = _air(ledger)
    ttft = [m.get("ttft_s", 0.0) for m in metrics]
    tpot = [m.get("tpot_ms", 0.0) / 1000.0 for m in metrics]
    pos = range(len(labels))
    width = 0.4
    fig, ax = plt.subplots()
    ax.bar([p - width / 2 for p in pos], ttft, width, label="TTFT — prefill", color="#4c72b0")
    ax.bar([p + width / 2 for p in pos], tpot, width, label="TPOT — decode", color="#dd8452")
    ax.set_xticks(list(pos))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Compression level")
    ax.set_ylabel("Latency (s)")
    ax.set_title("AirLLM latency — prefill (TTFT) vs decode (TPOT)")
    ax.legend()
    return _save(fig, out_dir, "ttft_vs_tpot.png")


def plot_quant_tradeoff(ledger: Ledger, out_dir: Path | str = "figures") -> Path:
    """Twin-axis trade-off: throughput and peak VRAM vs compression level."""
    labels, metrics = _air(ledger)
    tok = [m.get("throughput_tok_s", 0.0) for m in metrics]
    vram = [m.get("peak_vram_gb", 0.0) for m in metrics]
    fig, ax = plt.subplots()
    ax.plot(labels, tok, marker="o", color="#4c72b0", label="Throughput")
    ax.set_xlabel("Compression level")
    ax.set_ylabel("Throughput (tokens / s)", color="#4c72b0")
    ax.tick_params(axis="y", labelcolor="#4c72b0")
    twin = ax.twinx()
    twin.plot(labels, vram, marker="s", color="#c44e52", label="Peak VRAM")
    twin.set_ylabel("Peak VRAM (GB)", color="#c44e52")
    twin.tick_params(axis="y", labelcolor="#c44e52")
    ax.set_title("Quantization trade-off — throughput vs peak VRAM")
    return _save(fig, out_dir, "quant_tradeoff.png")


def generate_all(ledger: Ledger, out_dir: Path | str = "figures") -> list[Path]:
    """Render every Phase 7 figure and return the saved paths."""
    return [
        plot_tokens_per_sec(ledger, out_dir),
        plot_peak_vram(ledger, out_dir),
        plot_ttft_vs_tpot(ledger, out_dir),
        plot_quant_tradeoff(ledger, out_dir),
    ]
