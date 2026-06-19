"""Tests for the JSON + .env config loader (rule 4)."""

from __future__ import annotations

import json

import pytest

from cosmos77_ex05.shared.config import Config


def test_get_dot_path(config):
    assert config.get("experiment.max_new_tokens") == 20
    assert config.get("experiment.model_id") == "Qwen/Qwen2.5-14B-Instruct"


def test_get_missing_returns_default(config):
    assert config.get("nope.key", default="x") == "x"


def test_get_missing_raises(config):
    with pytest.raises(KeyError):
        config.get("nope.key")


def test_section_accessors(config):
    assert config.experiment()["model_params_billions"] == 14.7
    assert config.hardware_assumptions()["on_prem_gpu_price_usd"] == 1600
    assert config.paths()["results_dir"] == "results"


def test_pricing_accessors(config):
    pricing = config.pricing()
    assert "providers" in pricing
    assert config.provider_prices("openai_gpt4o_mini")["input_per_1m"] == 0.15
    assert pricing["prompt_caching"]["cached_input_discount"] == 0.5


def test_provider_unknown_raises(config):
    with pytest.raises(KeyError):
        config.provider_prices("nope")


def test_version_dir_repr_from_path(config, config_dir):
    assert config.version == "1.00"
    assert config.config_dir == config_dir
    assert "1.00" in repr(config)
    assert Config.from_path(config_dir).version == "1.00"


def test_env_reads_environment(config, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_example")
    assert config.env("HF_TOKEN") == "hf_example"
    assert config.env("MISSING_ENV_VAR", "d") == "d"


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config(tmp_path)


def test_bad_version_raises(tmp_path):
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "setup.json").write_text(json.dumps({"version": "0.9"}), encoding="utf-8")
    (cfg / "pricing.json").write_text(json.dumps({"version": "1.00"}), encoding="utf-8")
    with pytest.raises(ValueError):
        Config(cfg)


def test_non_dict_json_raises(tmp_path):
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "setup.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    (cfg / "pricing.json").write_text(json.dumps({"version": "1.00"}), encoding="utf-8")
    with pytest.raises(ValueError):
        Config(cfg)
