"""Tests for the low-level timing + resource probes.

The VRAM probes must stay GPU-free: on the developer Mac / CI box ``torch`` is
not installed (the ``experiment`` extra), so ``read_peak_vram_bytes`` must
return ``0`` without raising and ``reset_peak_vram`` must be a silent no-op.
"""

from __future__ import annotations

import sys
import types

import pytest

from cosmos77_ex05.measure import timing


def _fake_torch(*, cuda_available: bool, peak_bytes: int = 0) -> types.ModuleType:
    """Build a stand-in ``torch`` module with a ``cuda`` namespace.

    Lets the CUDA-present branches run without importing real torch (forbidden
    in this GPU-free suite). ``reset_peak_memory_stats`` records its calls.
    """
    module = types.ModuleType("torch")
    cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        max_memory_allocated=lambda: peak_bytes,
        reset_calls=[],
    )
    cuda.reset_peak_memory_stats = lambda: cuda.reset_calls.append(1)
    module.cuda = cuda
    return module


def test_default_clock_is_monotonic_float():
    first = timing.default_clock()
    second = timing.default_clock()
    assert isinstance(first, float)
    assert second >= first


def test_read_rss_bytes_positive_int():
    rss = timing.read_rss_bytes()
    assert isinstance(rss, int)
    assert rss > 0


def test_read_peak_vram_bytes_zero_without_torch():
    # The real Mac/CI path: torch is absent -> 0 bytes, no exception raised.
    assert timing.read_peak_vram_bytes() == 0


def test_reset_peak_vram_noop_without_torch():
    # Must not raise when torch/CUDA is unavailable.
    assert timing.reset_peak_vram() is None


@pytest.fixture
def fake_torch_cuda(request, monkeypatch):
    """Install a fake ``torch`` in ``sys.modules`` (never the real torch)."""
    fake = _fake_torch(**request.param)
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


@pytest.mark.parametrize(
    "fake_torch_cuda", [{"cuda_available": True, "peak_bytes": 31_138_512_896}], indirect=True
)
def test_read_peak_vram_bytes_reads_torch_when_cuda_present(fake_torch_cuda):
    assert timing.read_peak_vram_bytes() == 31_138_512_896


@pytest.mark.parametrize("fake_torch_cuda", [{"cuda_available": False}], indirect=True)
def test_read_peak_vram_bytes_zero_when_cuda_unavailable(fake_torch_cuda):
    # torch present but no CUDA device -> 0 bytes.
    assert timing.read_peak_vram_bytes() == 0


@pytest.mark.parametrize("fake_torch_cuda", [{"cuda_available": True}], indirect=True)
def test_reset_peak_vram_calls_torch_when_cuda_present(fake_torch_cuda):
    timing.reset_peak_vram()
    assert fake_torch_cuda.cuda.reset_calls == [1]


@pytest.mark.parametrize("fake_torch_cuda", [{"cuda_available": False}], indirect=True)
def test_reset_peak_vram_noop_when_cuda_unavailable(fake_torch_cuda):
    timing.reset_peak_vram()
    assert fake_torch_cuda.cuda.reset_calls == []
