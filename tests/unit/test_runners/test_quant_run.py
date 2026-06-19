"""Tests for the quantization sweep — one entry per level, quality note (mocked)."""

from __future__ import annotations

from cosmos77_ex05.runners.quant_run import _quality_note, run_quant_sweep


def test_sweep_records_each_level_with_quality_note():
    def fake_runner(model_id, prompt, n, shards, watts, *, compression=None):
        return {"compression": compression or "none", "output": "a b c", "throughput_tok_s": 2.0}

    results = run_quant_sweep("m", "p", 20, "/s", ["8bit", "4bit"], 70.0, runner=fake_runner)
    assert set(results) == {"8bit", "4bit"}
    assert results["8bit"]["level"] == "8bit"
    assert results["8bit"]["compression"] == "8bit"
    assert "words" in results["4bit"]["quality_note"]


def test_quality_note_handles_empty_and_text():
    assert _quality_note("") == "no output captured"
    assert "3 words" in _quality_note("one two three")
