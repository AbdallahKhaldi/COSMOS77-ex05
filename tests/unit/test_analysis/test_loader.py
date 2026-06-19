"""Tests for the ledger loader + ordering helpers."""

from __future__ import annotations

from cosmos77_ex05.analysis.loader import airllm_rows, label, load_ledger, ordered


def test_load_ledger_reads_all(ledger_dir):
    led = load_ledger(ledger_dir)
    assert set(led) == {"hardware", "fp16_baseline", "airllm_none", "airllm_8bit", "airllm_4bit"}


def test_ordered_is_canonical_and_excludes_hardware(ledger):
    names = [name for name, _ in ordered(ledger)]
    assert names == ["fp16_baseline", "airllm_none", "airllm_8bit", "airllm_4bit"]


def test_airllm_rows_excludes_baseline_and_hardware(ledger):
    names = [name for name, _ in airllm_rows(ledger)]
    assert names == ["airllm_none", "airllm_8bit", "airllm_4bit"]
    assert all(m["success"] for _, m in airllm_rows(ledger))


def test_label_pretty_names():
    assert label("airllm_4bit") == "AirLLM 4-bit"
    assert label("fp16_baseline") == "FP16 baseline (OOM)"
    assert label("unknown") == "unknown"
