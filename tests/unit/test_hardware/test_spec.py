"""Tests for hardware spec capture (D1).

GPU I/O is mocked per CLAUDE.md rule 6: a *fake* ``torch`` is injected so the
suite never imports the real (absent) torch and never needs a GPU. Two paths:
a fake CUDA T4 box and a torch-free CPU-only box (the student's Mac).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from cosmos77_ex05.constants import DEFAULT_ENCODING
from cosmos77_ex05.hardware import spec


def _fake_torch(*, available: bool, name: str = "Tesla T4", vram_bytes: int = 16_000_000_000):
    """Build a fake ``torch`` module exposing the CUDA surface we probe."""
    torch = types.ModuleType("torch")
    props = types.SimpleNamespace(name=name, total_memory=vram_bytes)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_properties=lambda _idx: props,
    )
    return torch


@pytest.fixture(autouse=True)
def _stub_psutil_shutil(monkeypatch):
    """Pin psutil/shutil so the assertions don't depend on real hardware."""
    monkeypatch.setattr(spec.psutil, "cpu_count", lambda logical=True: 8 if logical else 4)
    mem = types.SimpleNamespace(total=32_000_000_000)
    monkeypatch.setattr(spec.psutil, "virtual_memory", lambda: mem)
    usage = types.SimpleNamespace(total=500_000_000_000, free=200_000_000_000, used=0)
    monkeypatch.setattr(spec.shutil, "disk_usage", lambda _p: usage)
    monkeypatch.setattr(spec.platform, "processor", lambda: "Intel(R) Xeon(R)")
    monkeypatch.setattr(spec.platform, "platform", lambda: "Linux-fake")


def test_capture_spec_on_fake_cuda_box(monkeypatch, tmp_path):
    """A fake CUDA T4 is captured (name + VRAM) and the JSON file is written."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=True))
    out = tmp_path / "nested" / "spec.json"

    result = spec.capture_spec(out)

    assert result["gpu"] == {"name": "Tesla T4", "vram_gb": 16.0}
    assert result["cpu"] == {
        "model": "Intel(R) Xeon(R)",
        "cores_physical": 4,
        "cores_logical": 8,
    }
    assert result["ram_gb"] == 32.0
    assert result["disk"] == {"total_gb": 500.0, "free_gb": 200.0}
    assert result["platform"] == "Linux-fake"
    assert out.exists()
    assert json.loads(out.read_text(encoding=DEFAULT_ENCODING)) == result


def test_capture_spec_cpu_only_when_torch_missing(monkeypatch, tmp_path):
    """Import failure (the Mac path) -> the CPU-only GPU sentinel; file written."""
    real_import = __import__

    def _no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _no_torch)
    out = tmp_path / "spec.json"

    result = spec.capture_spec(out)

    assert result["gpu"] == {"name": "none / CPU-only", "vram_gb": None}
    assert out.exists()


def test_capture_spec_cpu_only_when_cuda_unavailable(monkeypatch, tmp_path):
    """torch present but ``cuda.is_available()`` False -> CPU-only sentinel."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False))

    result = spec.capture_spec(tmp_path / "spec.json")

    assert result["gpu"] == {"name": "none / CPU-only", "vram_gb": None}


def test_cpu_model_falls_back_to_machine(monkeypatch):
    """Empty ``platform.processor()`` falls back to the machine arch."""
    monkeypatch.setattr(spec.platform, "processor", lambda: "")
    monkeypatch.setattr(spec.platform, "machine", lambda: "arm64")
    assert spec._cpu_model() == "arm64"


def test_capture_spec_accepts_str_path(monkeypatch, tmp_path):
    """A string path (not only ``Path``) is accepted and written."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=True))
    out = tmp_path / "spec.json"
    spec.capture_spec(str(out))
    assert Path(out).exists()
