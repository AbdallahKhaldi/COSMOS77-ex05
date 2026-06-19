"""JSON + .env config loader for COSMOS77-ex05 (CLAUDE.md rule 4).

Every module reads its tunables through :class:`Config`, so the model id, quant
levels, max tokens, prices, hardware assumptions, and paths are never hardcoded.
``setup.json`` + ``pricing.json`` are version-checked at load; ``.env`` supplies
the optional ``HF_TOKEN``. A future migration to YAML/pydantic touches only this file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from cosmos77_ex05.shared.version import validate_config_version

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
_SENTINEL: Any = object()


class Config:
    """Loads ``setup.json`` + ``pricing.json`` and exposes dot-path access."""

    def __init__(self, config_dir: Path | str | None = None) -> None:
        self._config_dir = Path(config_dir) if config_dir is not None else _DEFAULT_CONFIG_DIR
        self._setup = self._load_json("setup.json")
        self._pricing = self._load_json("pricing.json")
        validate_config_version(str(self._setup.get("version", "")))
        load_dotenv(self._config_dir.parent / ".env", override=False)

    @classmethod
    def from_path(cls, path: Path | str) -> Config:
        """Construct from an explicit ``config/`` directory."""
        return cls(path)

    def _load_json(self, filename: str) -> dict[str, Any]:
        path = self._config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing required config file: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object at the top level")
        return data

    def get(self, dot_path: str, default: Any = _SENTINEL) -> Any:
        """Return the value at ``dot_path`` (e.g. ``experiment.max_new_tokens``)."""
        node: Any = self._setup
        for part in dot_path.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                if default is _SENTINEL:
                    raise KeyError(dot_path)
                return default
        return node

    def env(self, key: str, default: str | None = None) -> str | None:
        """Read an environment variable (after ``.env`` has been loaded)."""
        return os.environ.get(key, default)

    def experiment(self) -> dict[str, Any]:
        """The ``experiment`` section (model_id, quant_levels, max_new_tokens, prompt...)."""
        return dict(self.get("experiment", default={}))

    def hardware_assumptions(self) -> dict[str, Any]:
        """The ``hardware_assumptions`` section (CAPEX, life, kWh price, watts)."""
        return dict(self.get("hardware_assumptions", default={}))

    def paths(self) -> dict[str, str]:
        """The ``paths`` section (results_dir, figures_dir, reports_dir)."""
        return dict(self.get("paths", default={}))

    def pricing(self) -> dict[str, Any]:
        """The parsed ``pricing.json`` payload (providers, cloud_gpu, prompt_caching)."""
        return dict(self._pricing)

    def provider_prices(self, name: str) -> dict[str, Any]:
        """Return one API provider's ``{input_per_1m, output_per_1m}`` from pricing.json."""
        providers = self._pricing.get("providers", {})
        if name not in providers:
            raise KeyError(f"unknown provider {name!r}; known: {sorted(providers)}")
        return dict(providers[name])

    @property
    def version(self) -> str:
        """The ``setup.json`` version string (e.g. ``"1.00"``)."""
        return str(self._setup.get("version", ""))

    @property
    def config_dir(self) -> Path:
        """The directory the loader was pointed at."""
        return self._config_dir

    def __repr__(self) -> str:
        return f"Config(version={self.version!r}, dir={self._config_dir})"
