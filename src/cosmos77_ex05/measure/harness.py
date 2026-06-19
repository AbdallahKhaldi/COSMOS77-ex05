"""The measurement harness: the one seam every runner calls (D5).

``measure`` wraps any runner's ``generate_fn`` and emits the canonical metric
dict ``{ttft_s, tpot_ms, throughput_tok_s, peak_ram_gb, peak_vram_gb, total_s,
est_power_wh, n_out}``. It times two opaque invocations and derives the
lecture-anchored metrics:

- **TTFT / Prefill** (``ttft_s``): a single ``generate_fn(1)`` call — process
  the whole prompt in one GEMM pass, compute-bound.
- **TPOT / Decode** (``tpot_ms``): the steady-state per-token cost of the
  remaining ``n_out - 1`` tokens — each a GEMV re-reading all weights, so
  memory-bound.
- **throughput** (``throughput_tok_s``): the aggregate ``n_out / total_s`` rate.

The clock, RSS, VRAM and VRAM-reset probes default to ``timing.py`` but are
injectable so tests pass deterministic fakes (no real torch/psutil/clock).
"""

from __future__ import annotations

from collections.abc import Callable

from cosmos77_ex05.measure import timing

_SIG_FIGS = 6


def _round(value: float) -> float:
    """Round a metric to a sensible, table-friendly precision."""
    return round(value, _SIG_FIGS)


def measure(
    generate_fn: Callable[[int], object],
    n_out: int,
    watts: float,
    *,
    clock: Callable[[], float] = timing.default_clock,
    rss_fn: Callable[[], int] = timing.read_rss_bytes,
    vram_fn: Callable[[], int] = timing.read_peak_vram_bytes,
    reset_vram_fn: Callable[[], None] = timing.reset_peak_vram,
) -> dict:
    """Time a generation and return the canonical metric dict (D5).

    Args:
        generate_fn: Runner closure called with ``max_new_tokens``; its return
            value is ignored — the harness only **times** it.
        n_out: Target output-token budget for the full Decode run.
        watts: Board-power assumption for the coarse energy estimate.
        clock: Monotonic clock; defaults to ``timing.default_clock``.
        rss_fn: Resident-RAM probe; defaults to ``timing.read_rss_bytes``.
        vram_fn: Peak-VRAM probe; defaults to ``timing.read_peak_vram_bytes``.
        reset_vram_fn: Peak-VRAM reset; defaults to ``timing.reset_peak_vram``.

    Returns:
        Dict with keys ``ttft_s``, ``tpot_ms``, ``throughput_tok_s``,
        ``peak_ram_gb``, ``peak_vram_gb``, ``total_s``, ``est_power_wh`` and
        ``n_out``. ``tpot_ms`` is ``0.0`` when ``n_out <= 1`` (Decode undefined).
    """
    # Phase 1 — TTFT / Prefill: a single timed 1-token call.
    t0 = clock()
    generate_fn(1)
    t1 = clock()
    ttft_s = t1 - t0

    # Phase 2 — full Decode run: reset VRAM stats first, then time n_out tokens.
    reset_vram_fn()
    t2 = clock()
    generate_fn(n_out)
    t3 = clock()
    total_s = t3 - t2

    tpot_ms = (total_s - ttft_s) / (n_out - 1) * 1000 if n_out > 1 else 0.0
    throughput_tok_s = n_out / total_s if total_s > 0 else 0.0
    peak_ram_gb = rss_fn() / 1e9
    peak_vram_gb = vram_fn() / 1e9
    est_power_wh = watts * total_s / 3600

    return {
        "ttft_s": _round(ttft_s),
        "tpot_ms": _round(tpot_ms),
        "throughput_tok_s": _round(throughput_tok_s),
        "peak_ram_gb": _round(peak_ram_gb),
        "peak_vram_gb": _round(peak_vram_gb),
        "total_s": _round(total_s),
        "est_power_wh": _round(est_power_wh),
        "n_out": n_out,
    }
