"""Tests for the SDK surface — capture_hardware delegates, stubs raise (rule 2)."""

from __future__ import annotations

import pytest

from cosmos77_ex05.sdk.sdk import SDK


def test_repo_root_is_config_parent(config, config_dir):
    assert SDK(config=config).repo_root == config_dir.parent


def test_ledger_starts_empty(config):
    assert SDK(config=config).ledger() == {}


def test_capture_hardware_delegates(config, monkeypatch):
    import cosmos77_ex05.hardware.model_math as mathmod
    import cosmos77_ex05.hardware.spec as specmod

    captured: dict = {}

    def fake_capture(out_path):
        captured["out"] = out_path
        return {
            "cpu": {"model": "M1"},
            "ram_gb": 16.0,
            "gpu": {"name": "none / CPU-only", "vram_gb": None},
        }

    def fake_justify(model_id, params, vram_gb):
        captured["justify"] = (model_id, params, vram_gb)
        return {
            "model_id": model_id,
            "memory_gb": {"fp16": 29.4},
            "fits_fp16": False,
            "verdict": "OOM",
        }

    monkeypatch.setattr(specmod, "capture_spec", fake_capture)
    monkeypatch.setattr(mathmod, "justify", fake_justify)
    out = SDK(config=config).capture_hardware()
    assert out["model_math"]["fits_fp16"] is False
    assert out["spec"]["ram_gb"] == 16.0
    # CPU-only -> the model math uses the configured target T4 VRAM (16 GB)
    assert captured["justify"][2] == 16.0
    assert captured["justify"][1] == pytest.approx(14.7e9)


@pytest.mark.parametrize(
    "method",
    ["run_baseline", "run_airllm", "run_quant_sweep", "measure", "analyze", "economics", "report"],
)
def test_unimplemented_stages_raise(config, method):
    with pytest.raises(NotImplementedError):
        getattr(SDK(config=config), method)()
