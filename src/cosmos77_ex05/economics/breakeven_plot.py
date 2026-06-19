"""Phase-8 break-even plot (D7): cumulative cost vs monthly request volume.

Draws the four cumulative-cost curves from the cost-model CORE — On-Prem (starts
at CAPEX, slow electricity slope), API (from the origin, steeper), the caching-
discounted API, and Cloud-GPU — and marks the On-Prem<->API crossover. The
``Agg`` backend is selected *before* importing ``pyplot`` so the figure renders
head-less: no display, no GPU, no network. Every value comes from config + model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from cosmos77_ex05.economics.model import cumulative_costs  # noqa: E402
from cosmos77_ex05.economics.report import compute_scenario  # noqa: E402


def _volumes(break_even: float | None) -> list[float]:
    """A volume axis spanning ~2x the crossover (or a default when none exists)."""
    top = 2 * break_even if break_even else 2_000_000.0
    return [top * i / 50 for i in range(51)]


def plot_breakeven(
    config: Any,
    *,
    tokens_in: float,
    tokens_out: float,
    runtime_s: float,
    provider: str,
    out_dir: Path | str = "figures",
) -> Path:
    """Render the cumulative-cost break-even figure and return its PNG path.

    Plots On-Prem, API, caching-discounted API and Cloud-GPU cumulative cost
    (USD) over a request-volume axis, marks the crossover, creates ``out_dir``,
    and saves ``breakeven.png``. All curves come from :func:`compute_scenario`
    and :func:`cumulative_costs`; nothing is hard-coded.
    """
    s = compute_scenario(
        config,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        runtime_s=runtime_s,
        provider=provider,
    )
    volumes = _volumes(s["break_even_requests"])
    curves = cumulative_costs(
        volumes, s["capex"], s["api_per_req"], s["on_prem_opex_per_req"], s["cloud_per_req"]
    )
    cached = [n * s["cached_api_per_req"] for n in volumes]

    fig, ax = plt.subplots()
    ax.plot(volumes, curves["on_prem"], label="On-Prem (CAPEX + electricity)", color="#4c72b0")
    ax.plot(volumes, curves["api"], label=f"API `{provider}`", color="#dd8452")
    ax.plot(volumes, cached, label="API + prompt caching", color="#937860", linestyle="--")
    ax.plot(volumes, curves["cloud_gpu"], label="Cloud-GPU (T4)", color="#55a868", linestyle=":")
    _mark_crossover(ax, s)
    ax.set_xlabel("Cumulative requests")
    ax.set_ylabel("Cumulative cost (USD)")
    ax.set_title("On-Prem vs API break-even")
    ax.legend()
    return _save(fig, out_dir, "breakeven.png")


def _mark_crossover(ax: Any, scenario: dict[str, Any]) -> None:
    """Drop a vertical line at the On-Prem<->API break-even volume, if finite."""
    break_even = scenario["break_even_requests"]
    if break_even is None:
        return
    cost = break_even * scenario["api_per_req"]
    ax.axvline(break_even, color="#c44e52", linestyle="-.", linewidth=1)
    ax.annotate(
        f"break-even ~{break_even:,.0f}",
        xy=(break_even, cost),
        xytext=(break_even, cost * 1.1),
        ha="center",
        color="#c44e52",
    )


def _save(fig: Any, out_dir: Path | str, name: str) -> Path:
    """Persist ``fig`` to ``out_dir/name``, creating the directory, then close it."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
