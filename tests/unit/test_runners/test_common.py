"""Tests for shared runner helpers — OOM detection + shard manifest (GPU-free)."""

from __future__ import annotations

from cosmos77_ex05.runners._common import is_oom, shard_manifest


class OutOfMemoryError(Exception):
    """Stands in for torch.cuda.OutOfMemoryError without importing torch."""


def test_is_oom_by_class_name():
    assert is_oom(OutOfMemoryError("boom")) is True


def test_is_oom_by_message():
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate 29 GB")) is True


def test_is_oom_false_for_unrelated():
    assert is_oom(ValueError("bad prompt")) is False


def test_shard_manifest_counts_and_sizes(tmp_path):
    (tmp_path / "layer_0.safetensors").write_bytes(b"x" * 1000)
    (tmp_path / "layer_1.safetensors").write_bytes(b"y" * 2000)
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    manifest = shard_manifest(tmp_path)
    assert manifest["n_shards"] == 2
    assert manifest["disk_used_gb"] == round(3000 / 1e9, 2)
    assert manifest["shards_path"] == str(tmp_path)


def test_shard_manifest_missing_dir_is_zero(tmp_path):
    manifest = shard_manifest(tmp_path / "nope")
    assert manifest["n_shards"] == 0
    assert manifest["disk_used_gb"] == 0.0
