"""Tests for the AirLLM runner — factory + generate mocked, output captured."""

from __future__ import annotations

from cosmos77_ex05.runners.airllm_run import run_airllm


def _fake_measure(generate_fn, n_out, watts, **kwargs):
    """Mimic the harness: time a 1-token call then the n_out call (capturing output)."""
    generate_fn(1)
    generate_fn(n_out)
    return {"ttft_s": 0.5, "tpot_ms": 500.0, "throughput_tok_s": 2.0, "n_out": n_out}


def test_run_airllm_captures_output_and_labels():
    calls = {}

    def factory(model_id, shards_path, compression):
        calls["factory"] = (model_id, shards_path, compression)
        return "MODEL"

    def generate(model, prompt, n, max_seq_len):
        return f"answer-{n}"

    metrics = run_airllm(
        "Qwen/Qwen2.5-14B-Instruct",
        "Explain attention.",
        20,
        "/kaggle/working/shards",
        70.0,
        model_factory=factory,
        generate=generate,
        measure_fn=_fake_measure,
    )
    assert metrics["success"] is True
    assert metrics["scenario"] == "airllm_none"
    assert metrics["compression"] == "none"
    assert metrics["output"] == "answer-20"  # the n_out call, not the 1-token TTFT call
    assert metrics["throughput_tok_s"] == 2.0
    assert calls["factory"][2] is None


def test_run_airllm_compression_labels_scenario():
    metrics = run_airllm(
        "m",
        "p",
        20,
        "/s",
        70.0,
        compression="4bit",
        model_factory=lambda *a: "M",
        generate=lambda *a: "out",
        measure_fn=_fake_measure,
    )
    assert metrics["scenario"] == "airllm_4bit"
    assert metrics["compression"] == "4bit"
