"""Tests for the download + AirLLM sharding wrapper (factory mocked)."""

from __future__ import annotations

from cosmos77_ex05.runners.download import download_and_shard


def test_download_and_shard_returns_manifest(tmp_path):
    def factory(model_id, shards_path, compression):
        # mimic AirLLM writing per-layer shards
        (tmp_path / "layer_0.safetensors").write_bytes(b"x" * 5000)
        (tmp_path / "layer_1.safetensors").write_bytes(b"y" * 5000)
        return "MODEL"

    manifest = download_and_shard("Qwen/Qwen2.5-14B-Instruct", str(tmp_path), model_factory=factory)
    assert manifest["model_id"] == "Qwen/Qwen2.5-14B-Instruct"
    assert manifest["n_shards"] == 2
    assert manifest["disk_used_gb"] == round(10000 / 1e9, 2)
