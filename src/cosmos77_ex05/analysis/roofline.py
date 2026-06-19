"""Roofline analysis of the AirLLM decode benchmark (D9, advanced deliverable).

The Roofline model bounds attainable throughput by two ceilings: the compute
peak (flat, FLOP/s) and the memory slope (``y = AI * bandwidth``). Their crossing
is the *ridge point*; a kernel left of it is **memory-bound**, right of it
**compute-bound**. Autoregressive **decode is GEMV** — each weight is read once
per token for ~2 FLOPs, so arithmetic intensity is ~1-2 FLOP/byte, far left of
the T4 ridge (~203). The headline finding: AirLLM's *effective* bandwidth is
~0.2-0.3 GB/s (disk mmap speed), ~1000x below the T4's 320 GB/s HBM — it trades
the HBM-memory wall for a **disk-bandwidth** wall. Every number flows from the
measured ledger (rule 13); only model/hardware physics are named constants here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend: no GPU, no display (rule 6/17)

import matplotlib.pyplot as plt

from cosmos77_ex05.analysis.loader import airllm_rows, label

#: T4 FP16 peak compute (TFLOP/s) and GDDR6 memory bandwidth (GB/s).
T4_PEAK_TFLOPS: float = 65.0
T4_BW_GBs: float = 320.0
#: Ridge point = peak FLOP/s / peak B/s (~203 FLOPs/byte) — the compute/memory crossover.
RIDGE: float = T4_PEAK_TFLOPS * 1e12 / (T4_BW_GBs * 1e9)

#: Qwen2.5-14B parameter count; decode does ~2 FLOPs per weight per token.
MODEL_PARAMS: float = 14.7e9
FLOPS_PER_PARAM: float = 2.0
#: Bytes read per parameter per token, keyed by the ledger ``compression`` field.
BYTES_PER_PARAM: dict[str, float] = {"none": 2.0, "8bit": 1.0, "4bit": 0.5}


def scenario_point(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Place one measured scenario on the Roofline plane.

    Derives arithmetic intensity, sustained FLOP/s, and *effective* bandwidth
    from the ledger's ``compression`` and ``throughput_tok_s`` — never a
    hardcoded measurement. ``bound`` compares AI against :data:`RIDGE`.
    """
    bytes_per_param = BYTES_PER_PARAM[metrics["compression"]]
    throughput = metrics["throughput_tok_s"]
    bytes_per_token = MODEL_PARAMS * bytes_per_param
    achieved_flops = throughput * FLOPS_PER_PARAM * MODEL_PARAMS
    arithmetic_intensity = FLOPS_PER_PARAM / bytes_per_param
    return {
        "label": label(name),
        "arithmetic_intensity": arithmetic_intensity,
        "achieved_flops": achieved_flops,
        "achieved_tflops": achieved_flops / 1e12,
        "effective_bandwidth_gbs": bytes_per_token * throughput / 1e9,
        "bound": "memory-bound" if arithmetic_intensity < RIDGE else "compute-bound",
    }


def roofline_points(ledger: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Roofline points for the three successfully-measured AirLLM scenarios."""
    return [scenario_point(name, metrics) for name, metrics in airllm_rows(ledger)]


def _draw_roofline(ax: Any, max_ai: float) -> None:
    """Draw the memory slope, the compute ceiling, and the ridge marker."""
    ridge_floor = min(RIDGE, max_ai)
    ax.plot(
        [ridge_floor / 100, ridge_floor],
        [ridge_floor / 100 * T4_BW_GBs * 1e9 / 1e12, ridge_floor * T4_BW_GBs * 1e9 / 1e12],
        color="tab:blue",
        label=f"HBM memory roof ({T4_BW_GBs:.0f} GB/s)",
    )
    ax.axhline(
        T4_PEAK_TFLOPS,
        color="tab:red",
        ls="--",
        label=f"Compute roof ({T4_PEAK_TFLOPS:.0f} TFLOP/s)",
    )
    ax.axvline(RIDGE, color="gray", ls=":", label=f"Ridge ~{RIDGE:.0f} FLOPs/byte")


def plot_roofline(ledger: dict[str, dict[str, Any]], out_dir: Path | str = "figures") -> Path:
    """Render the log-log Roofline to ``<out_dir>/roofline.png`` and return its path.

    Plots the HBM memory slope, the compute ceiling, the ridge, and each measured
    point (its AI vs achieved TFLOP/s). The points sit far below the memory roof
    because the *effective* bandwidth is disk, not HBM — annotated against the
    320 GB/s HBM line to make the disk-bound wall explicit.
    """
    points = roofline_points(ledger)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_roofline(ax, max(p["arithmetic_intensity"] for p in points))
    for point in points:
        eff_bw = point["effective_bandwidth_gbs"]
        ax.scatter(point["arithmetic_intensity"], point["achieved_tflops"], zorder=5)
        ax.annotate(
            f"{point['label']}\n{eff_bw:.2f} GB/s (disk)",
            (point["arithmetic_intensity"], point["achieved_tflops"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Arithmetic intensity (FLOPs/byte)",
        ylabel="Attainable (TFLOP/s)",
    )
    ax.set_title("AirLLM decode Roofline (T4): memory-bound, at disk bandwidth")
    ax.text(
        0.5,
        0.02,
        f"Effective BW ~{points[0]['effective_bandwidth_gbs']:.2f} GB/s vs {T4_BW_GBs:.0f} GB/s HBM "
        f"(~{T4_BW_GBs / points[0]['effective_bandwidth_gbs']:.0f}x below)",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path / "roofline.png", dpi=120)
    plt.close(fig)
    return out_path / "roofline.png"


__all__ = [
    "RIDGE",
    "T4_BW_GBs",
    "T4_PEAK_TFLOPS",
    "plot_roofline",
    "roofline_points",
    "scenario_point",
]
