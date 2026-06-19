"""AirLLM runner — make the SAME model run, layer by layer (D3, the "layer = page").

AirLLM keeps only the active transformer layer in VRAM and streams the rest from
per-layer SafeTensors shards (``mmap``) — the OS virtual-memory/Paging analogy. The
29 GB model "fits" in 16 GB at the cost of latency (disk BW ~3-7 GB/s ≪ HBM ~320
GB/s), so we MEASURE it (TTFT/TPOT/throughput/peak mem) rather than chase speed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cosmos77_ex05.measure.harness import measure
from cosmos77_ex05.runners._common import airllm_generate, default_airllm_factory


def run_airllm(
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    shards_path: str,
    watts: float,
    *,
    compression: str | None = None,
    max_seq_len: int = 128,
    model_factory: Callable[[str, str, str | None], Any] | None = None,
    generate: Callable[[Any, str, int, int], str] | None = None,
    measure_fn: Callable[..., dict[str, Any]] = measure,
) -> dict[str, Any]:
    """Run AirLLM for the configured prompt and return the measured metrics + output.

    ``compression`` ∈ {None (FP16), '8bit', '4bit'} selects the bitsandbytes level;
    the same function powers the quantization sweep (D4).
    """
    factory = model_factory or default_airllm_factory
    gen = generate or airllm_generate
    model = factory(model_id, shards_path, compression)

    captured: dict[str, Any] = {}

    def generate_fn(n: int) -> str:
        text = gen(model, prompt, n, max_seq_len)
        captured["output"] = text  # the last (n_out) call is the recorded output
        return text

    metrics = measure_fn(generate_fn, max_new_tokens, watts)
    metrics["success"] = True
    metrics["scenario"] = f"airllm_{compression}" if compression else "airllm_none"
    metrics["compression"] = compression or "none"
    metrics["model_id"] = model_id
    metrics["prompt"] = prompt
    metrics["output"] = captured.get("output", "")
    metrics["mechanism"] = "layer-by-layer mmap (layer = page)"
    return metrics
