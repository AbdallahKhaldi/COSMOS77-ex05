"""Model download + AirLLM sharding wrapper (Phase 3).

AirLLM's first ``from_pretrained`` downloads the model and splits it into per-layer
SafeTensors shards under ``layer_shards_saving_path`` (with ``delete_original=True``
to free the monolithic download). We return the resulting shard manifest + disk
usage so the notebook can report how much the sharding cost.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cosmos77_ex05.runners._common import default_airllm_factory, shard_manifest


def download_and_shard(
    model_id: str,
    shards_path: str,
    *,
    model_factory: Callable[[str, str, str | None], Any] | None = None,
) -> dict[str, Any]:
    """Trigger the AirLLM download+shard and return ``{model_id, shards_path, n_shards, disk_used_gb}``."""
    factory = model_factory or default_airllm_factory
    factory(model_id, shards_path, None)
    manifest = shard_manifest(shards_path)
    manifest["model_id"] = model_id
    return manifest
