"""Tabular ANALYSIS layer: the measured ledger as a DataFrame + a METRICS.md.

Turns ``results/*.json`` (the single source of truth) into (a) a one-row-per-
scenario :class:`pandas.DataFrame` and (b) a GitHub-flavoured Markdown report.
The ``fp16_baseline`` row is the out-of-memory control: it carries no
throughput / TTFT / peak-VRAM (the run OOM'd before generating a token), so its
performance cells are ``NaN`` while it still records the requested 29.4 GB that
overflowed the T4. Every number is read from the ledger — nothing is hardcoded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cosmos77_ex05.analysis.loader import label, ordered

#: Columns rendered for each scenario, in display order.
_COLUMNS: tuple[str, ...] = (
    "scenario",
    "success",
    "throughput_tok_s",
    "ttft_s",
    "tpot_s",
    "peak_vram_gb",
    "peak_ram_gb",
    "total_s",
    "est_power_wh",
    "requested_vram_gb",
)
#: Default destination for the Markdown report.
_DEFAULT_REPORT: Path = Path("reports/METRICS.md")


def _row(scenario: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Project one ledger entry onto the table columns (rounding floats)."""
    tpot_ms = metrics.get("tpot_ms")
    return {
        "scenario": label(scenario),
        "success": bool(metrics.get("success", False)),
        "throughput_tok_s": metrics.get("throughput_tok_s"),
        "ttft_s": _round(metrics.get("ttft_s"), 3),
        "tpot_s": _round(tpot_ms / 1000 if tpot_ms is not None else None, 3),
        "peak_vram_gb": _round(metrics.get("peak_vram_gb"), 3),
        "peak_ram_gb": _round(metrics.get("peak_ram_gb"), 3),
        "total_s": _round(metrics.get("total_s"), 1),
        "est_power_wh": _round(metrics.get("est_power_wh"), 3),
        "requested_vram_gb": metrics.get("requested_vram_gb"),
    }


def _round(value: float | None, ndigits: int) -> float | None:
    """Round a float, passing ``None`` through (so missing cells stay ``NaN``)."""
    return None if value is None else round(value, ndigits)


def build_dataframe(ledger: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Return one row per scenario (canonical order) as a :class:`pandas.DataFrame`.

    Performance columns (throughput / TTFT / TPOT / peak VRAM …) are ``NaN`` for
    the OOM baseline; ``requested_vram_gb`` records the 29.4 GB FP16 footprint
    that overflowed the GPU.
    """
    rows = [_row(name, metrics) for name, metrics in ordered(ledger)]
    return pd.DataFrame(rows, columns=list(_COLUMNS))


def _to_markdown(df: pd.DataFrame) -> str:
    """Render ``df`` as a GitHub pipe table (``to_markdown`` if tabulate is present)."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return _manual_markdown(df)


def _cell(value: Any) -> str:
    """Format one cell: blank for missing, plain string otherwise."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _manual_markdown(df: pd.DataFrame) -> str:
    """Build a Markdown table by hand (fallback when ``tabulate`` is unavailable)."""
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_cell(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def _gpu_line(ledger: dict[str, dict[str, Any]]) -> str:
    """One-line GPU/spec summary from ``ledger['hardware']``."""
    hw = ledger.get("hardware", {})
    gpu = hw.get("gpu", {})
    return (
        f"**Hardware:** {gpu.get('name', 'unknown GPU')} "
        f"({gpu.get('vram_gb', '?')} GB VRAM), {hw.get('ram_gb', '?')} GB host RAM, "
        f"platform `{hw.get('platform', 'unknown')}`."
    )


def _takeaways(ledger: dict[str, dict[str, Any]]) -> str:
    """Honest, ledger-derived takeaways (paging works; the disk-bandwidth price)."""
    none = ledger.get("airllm_none", {})
    four = ledger.get("airllm_4bit", {})
    base = ledger.get("fp16_baseline", {})
    speedup = four.get("throughput_tok_s", 0) / none.get("throughput_tok_s", 1)
    return (
        f"AirLLM runs the {base.get('requested_vram_gb', '?')} GB FP16 model at "
        f"{none.get('peak_vram_gb', '?'):.2f}–{four.get('peak_vram_gb', '?'):.2f} GB peak "
        f"VRAM by paging one layer at a time — quantization shrinks the per-layer page "
        f"further. The cost is throughput: TPOT runs "
        f"{four.get('tpot_ms', 0) / 1000:.0f}–{none.get('tpot_ms', 0) / 1000:.0f} s/token, "
        f"the disk-bandwidth price of streaming weights from storage. 4-bit is "
        f"~{speedup:.0f}x the FP16-AirLLM throughput, while the FP16 baseline OOMs "
        f"({base.get('requested_vram_gb', '?')} GB > {base.get('available_vram_gb', '?')} "
        f"GB available) and never reaches the compute stage."
    )


def write_metrics_md(
    ledger: dict[str, dict[str, Any]], out_path: Path | str = _DEFAULT_REPORT
) -> Path:
    """Write the Markdown metrics report and return its :class:`~pathlib.Path`.

    Emits a title, the GPU/spec line, the scenario comparison table, and 2–4
    sentences of takeaways. Parent directories are created as needed; every
    number is sourced from ``ledger``.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = _to_markdown(build_dataframe(ledger))
    body = "\n\n".join(
        [
            "# AirLLM measured metrics",
            _gpu_line(ledger),
            "## Scenario comparison",
            table,
            "## Takeaways",
            _takeaways(ledger),
        ]
    )
    out.write_text(body + "\n", encoding="utf-8")
    return out
