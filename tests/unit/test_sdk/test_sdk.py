"""Tests for the SDK surface — capture_hardware delegates, stubs raise (rule 2)."""

from __future__ import annotations

import pytest

from cosmos77_ex05.sdk.sdk import SDK


def test_repo_root_is_config_parent(config, config_dir):
    assert SDK(config=config).repo_root == config_dir.parent


def test_ledger_starts_empty(config):
    assert SDK(config=config).ledger() == {}


def test_results_dir_override_redirects_ledger(config, tmp_path):
    """The Kaggle notebook passes results_dir to write the ledger to the captured output."""
    out = tmp_path / "kaggle_working_results"
    sdk = SDK(config=config, results_dir=out)
    assert sdk.results_dir == out
    sdk.gatekeeper.record("airllm_none", {"throughput_tok_s": 2.0})
    assert (out / "airllm_none.json").exists()


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


@pytest.mark.parametrize("method", ["measure", "analyze", "economics", "report"])
def test_unimplemented_stages_raise(config, method):
    with pytest.raises(NotImplementedError):
        getattr(SDK(config=config), method)()


def test_run_baseline_records_ledger(config, monkeypatch):
    import cosmos77_ex05.runners.baseline as baseline

    def fake_baseline(model_id, prompt, params, vram_gb, *a, **k):
        return {"success": False, "requested_vram_gb": 29.4, "available_vram_gb": vram_gb}

    monkeypatch.setattr(baseline, "run_fp16_baseline", fake_baseline)
    sdk = SDK(config=config)
    out = sdk.run_baseline()
    assert out["success"] is False
    assert sdk.gatekeeper.read("fp16_baseline")["requested_vram_gb"] == 29.4


def test_run_airllm_records_ledger(config, monkeypatch):
    import cosmos77_ex05.runners.airllm_run as airllm_mod

    def fake_run(model_id, prompt, n, shards, watts, **k):
        return {"success": True, "throughput_tok_s": 2.0, "compression": "none"}

    monkeypatch.setattr(airllm_mod, "run_airllm", fake_run)
    sdk = SDK(config=config)
    out = sdk.run_airllm()
    assert out["throughput_tok_s"] == 2.0
    assert sdk.gatekeeper.read("airllm_none")["success"] is True


def test_run_quant_sweep_records_each_level(config, monkeypatch):
    import cosmos77_ex05.runners.quant_run as quant_mod

    def fake_sweep(model_id, prompt, n, shards, levels, watts, **k):
        return {lv: {"compression": lv, "throughput_tok_s": 3.0} for lv in levels}

    monkeypatch.setattr(quant_mod, "run_quant_sweep", fake_sweep)
    sdk = SDK(config=config)
    out = sdk.run_quant_sweep(["8bit", "4bit"])
    assert set(out) == {"8bit", "4bit"}
    assert sdk.gatekeeper.read("airllm_8bit")["compression"] == "8bit"
    assert sdk.gatekeeper.read("airllm_4bit")["throughput_tok_s"] == 3.0
