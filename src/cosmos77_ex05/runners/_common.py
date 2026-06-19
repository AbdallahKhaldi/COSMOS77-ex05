"""Shared helpers for the runners (CLAUDE.md rule 3 — no duplication).

All heavy imports (``torch``, ``transformers``, ``airllm``) are LAZY — imported
inside the default factory/loader functions, never at module load — so the GPU-free
test suite imports the runners without those packages installed. Tests inject fakes
for every default here. OOM is detected by exception identity, not a torch import.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

#: Type of a (model, tokenizer)-returning loader.
Loader = Callable[[str], tuple[Any, Any]]


def is_oom(exc: BaseException) -> bool:
    """True if ``exc`` is a CUDA out-of-memory error (torch-free detection)."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    if "OutOfMemoryError" in names:
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda" in text and "memory" in text


def default_transformers_loader(model_id: str) -> tuple[Any, Any]:  # pragma: no cover - GPU only
    """Load Qwen FP16 onto CUDA the naive way (the baseline that OOMs)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    )
    return model, tokenizer


def transformers_generate(
    model: Any, tokenizer: Any, prompt: str, n: int
) -> str:  # pragma: no cover - GPU only
    """Generate ``n`` tokens with a plain transformers model (the baseline path)."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=n)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def default_airllm_factory(
    model_id: str, shards_path: str, compression: str | None = None
) -> Any:  # pragma: no cover - GPU only
    """Build an AirLLM model (layer-by-layer; ``compression`` ∈ {None,'8bit','4bit'})."""
    from airllm import AutoModel

    # delete_original=False keeps the HF download so each scenario can re-shard
    # without re-downloading; the notebook clears the (smaller) shards dir between
    # scenarios to cap peak disk on Kaggle's single shared overlay filesystem.
    kwargs: dict[str, Any] = {
        "layer_shards_saving_path": shards_path,
        "delete_original": False,
        "profiling_mode": True,
    }
    if compression:
        kwargs["compression"] = compression
    return AutoModel.from_pretrained(model_id, **kwargs)


def airllm_generate(
    model: Any, prompt: str, n: int, max_seq_len: int = 128
) -> str:  # pragma: no cover - GPU only
    """Tokenize ``prompt`` and generate ``n`` tokens via AirLLM (the .cuda() placement)."""
    tokens = model.tokenizer(
        [prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
    )
    output = model.generate(
        tokens["input_ids"].cuda(),
        max_new_tokens=n,
        use_cache=True,
        return_dict_in_generate=True,
    )
    return model.tokenizer.decode(output.sequences[0])


def shard_manifest(shards_path: str | Path) -> dict[str, Any]:
    """Summarise the per-layer SafeTensors shards written under ``shards_path``."""
    root = Path(shards_path)
    shards = sorted(root.rglob("*.safetensors")) if root.exists() else []
    total_bytes = sum(p.stat().st_size for p in shards)
    return {
        "shards_path": str(shards_path),
        "n_shards": len(shards),
        "disk_used_gb": round(total_bytes / 1e9, 2),
    }
