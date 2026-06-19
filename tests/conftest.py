"""Shared pytest fixtures and deterministic-seed setup (CLAUDE.md rule 17)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

_SETUP = {
    "version": "1.00",
    "experiment": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "model_params_billions": 14.7,
        "target_vram_gb": 16,
        "platform": "kaggle_t4",
        "max_new_tokens": 20,
        "prompt": "Explain the attention mechanism in transformers.",
        "quant_levels": ["fp16_baseline", "airllm_none", "airllm_8bit", "airllm_4bit"],
        "layer_shards_saving_path": "/kaggle/working/shards",
        "max_seq_len": 128,
    },
    "hardware_assumptions": {
        "on_prem_gpu_price_usd": 1600,
        "hardware_life_years": 3,
        "electricity_usd_per_kwh": 0.15,
        "gpu_power_watts": 70,
    },
    "paths": {"results_dir": "results", "figures_dir": "figures", "reports_dir": "reports"},
}
_PRICING = {
    "version": "1.00",
    "providers": {
        "openai_gpt4o_mini": {"input_per_1m": 0.15, "output_per_1m": 0.60},
        "anthropic_claude_haiku": {"input_per_1m": 0.80, "output_per_1m": 4.00},
        "google_gemini_flash": {"input_per_1m": 0.075, "output_per_1m": 0.30},
    },
    "cloud_gpu": {"t4_usd_per_hour": 0.35, "a10g_usd_per_hour": 1.00},
    "prompt_caching": {"cached_input_discount": 0.5},
}


@pytest.fixture(autouse=True)
def _seed_random() -> None:
    """Seed `random` before every test so nothing flakes."""
    random.seed(1729)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A throwaway ``config/`` dir with valid setup.json + pricing.json."""
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "setup.json").write_text(json.dumps(_SETUP), encoding="utf-8")
    (cfg / "pricing.json").write_text(json.dumps(_PRICING), encoding="utf-8")
    return cfg


@pytest.fixture
def config(config_dir: Path):
    """A :class:`Config` loaded from the throwaway config dir."""
    from cosmos77_ex05.shared.config import Config

    return Config(config_dir)
