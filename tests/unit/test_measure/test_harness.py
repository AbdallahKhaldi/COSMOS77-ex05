"""Tests for the measurement harness arithmetic (D5).

Everything is mocked and deterministic — a scripted clock, a recording
``generate_fn`` and constant RSS/VRAM probes — so the suite is GPU-free,
network-free and never depends on a real wall clock. We assert the exact
TTFT/TPOT/throughput/power arithmetic from the metric definitions.
"""

from __future__ import annotations

from cosmos77_ex05.measure import harness

# Scripted perf_counter timestamps:
#   ttft call spans 100.0 -> 100.5  => ttft_s = 0.5
#   full call spans 100.5 -> 110.5  => total_s = 10.0
_CLOCK_SCRIPT = [100.0, 100.5, 100.5, 110.5]
_WATTS = 70.0
_N_OUT = 20
_RSS_BYTES = 16_000_000_000  # 16 GB resident
_VRAM_BYTES = 31_138_512_896  # ~31 GB peak VRAM


class _FakeClock:
    """Returns a scripted increasing sequence of timestamps."""

    def __init__(self, script: list[float]) -> None:
        self._script = list(script)
        self._i = 0

    def __call__(self) -> float:
        value = self._script[self._i]
        self._i += 1
        return value


class _RecordingGenerate:
    """Records the ``max_new_tokens`` value of every call."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, max_new_tokens: int) -> str:
        self.calls.append(max_new_tokens)
        return "x" * max_new_tokens


def test_measure_exact_arithmetic():
    clock = _FakeClock(_CLOCK_SCRIPT)
    gen = _RecordingGenerate()
    reset_calls: list[int] = []

    result = harness.measure(
        gen,
        _N_OUT,
        _WATTS,
        clock=clock,
        rss_fn=lambda: _RSS_BYTES,
        vram_fn=lambda: _VRAM_BYTES,
        reset_vram_fn=lambda: reset_calls.append(1),
    )

    assert result["ttft_s"] == 0.5
    assert result["total_s"] == 10.0
    assert result["tpot_ms"] == round((10.0 - 0.5) / 19 * 1000, 6)
    assert result["tpot_ms"] == 500.0
    assert result["throughput_tok_s"] == round(20 / 10.0, 6)
    assert result["throughput_tok_s"] == 2.0
    assert result["peak_ram_gb"] == round(_RSS_BYTES / 1e9, 6)
    assert result["peak_vram_gb"] == round(_VRAM_BYTES / 1e9, 6)
    assert result["est_power_wh"] == round(_WATTS * 10.0 / 3600, 6)
    assert result["n_out"] == _N_OUT


def test_measure_calls_generate_with_one_then_n_out():
    gen = _RecordingGenerate()
    harness.measure(
        gen,
        _N_OUT,
        _WATTS,
        clock=_FakeClock(_CLOCK_SCRIPT),
        rss_fn=lambda: _RSS_BYTES,
        vram_fn=lambda: _VRAM_BYTES,
        reset_vram_fn=lambda: None,
    )
    assert gen.calls == [1, _N_OUT]


def test_measure_resets_vram_before_full_run():
    # reset must happen after the TTFT call and before the full run.
    gen = _RecordingGenerate()
    reset_calls: list[int] = []
    harness.measure(
        gen,
        _N_OUT,
        _WATTS,
        clock=_FakeClock(_CLOCK_SCRIPT),
        rss_fn=lambda: _RSS_BYTES,
        vram_fn=lambda: _VRAM_BYTES,
        reset_vram_fn=lambda: reset_calls.append(1),
    )
    assert reset_calls == [1]


def test_measure_n_out_one_has_zero_tpot():
    # n_out == 1 => Decode undefined => tpot_ms is 0.0, no ZeroDivisionError.
    gen = _RecordingGenerate()
    result = harness.measure(
        gen,
        1,
        _WATTS,
        clock=_FakeClock([0.0, 0.5, 0.5, 1.5]),
        rss_fn=lambda: _RSS_BYTES,
        vram_fn=lambda: 0,
        reset_vram_fn=lambda: None,
    )
    assert result["tpot_ms"] == 0.0
    assert result["n_out"] == 1
    assert gen.calls == [1, 1]


def test_measure_zero_total_guards_throughput():
    # A degenerate clock with zero elapsed full run must not divide by zero.
    result = harness.measure(
        _RecordingGenerate(),
        _N_OUT,
        _WATTS,
        clock=_FakeClock([0.0, 0.0, 0.0, 0.0]),
        rss_fn=lambda: _RSS_BYTES,
        vram_fn=lambda: 0,
        reset_vram_fn=lambda: None,
    )
    assert result["throughput_tok_s"] == 0.0


def test_measure_default_probes_are_gpu_free():
    # Using the real default probes (no injection) must work without a GPU.
    result = harness.measure(_RecordingGenerate(), 4, _WATTS)
    assert result["peak_vram_gb"] == 0.0
    assert result["peak_ram_gb"] > 0.0
    assert set(result) == {
        "ttft_s",
        "tpot_ms",
        "throughput_tok_s",
        "peak_ram_gb",
        "peak_vram_gb",
        "total_s",
        "est_power_wh",
        "n_out",
    }
