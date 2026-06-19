"""Gatekeeper — the measurement LEDGER (CLAUDE.md rule 13).

Repurposed from HW4's token meter: every measured number flows through here into
``results/<scenario>.json`` — the SINGLE SOURCE OF TRUTH for every table and
graph. ``record`` writes/merges a scenario's metrics; ``ledger`` aggregates all
``results/*.json`` into one dict. Nothing is "true" unless it is in the ledger,
and nothing is ever fabricated. ``scrub`` redacts secrets before anything is logged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_SECRET_RE = re.compile(
    r"(hf_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9_\-]{6,}"
    r"|gh[pousr]_[A-Za-z0-9]{16,}|Bearer\s+[A-Za-z0-9._\-]+)"
)


class Gatekeeper:
    """Append-style measurement ledger over ``results/<scenario>.json``."""

    def __init__(self, results_dir: Path | str = "results") -> None:
        self.results_dir = Path(results_dir)

    def record(self, scenario: str, metrics: dict[str, Any]) -> Path:
        """Write/merge ``metrics`` into ``results/<scenario>.json`` and return its path."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = self.results_dir / f"{scenario}.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data["scenario"] = scenario
        data.update(metrics)
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def read(self, scenario: str) -> dict[str, Any]:
        """Return one scenario's recorded metrics (``{}`` if not yet measured)."""
        path = self.results_dir / f"{scenario}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def ledger(self) -> dict[str, dict[str, Any]]:
        """Aggregate every ``results/*.json`` into ``{scenario: metrics}``."""
        out: dict[str, dict[str, Any]] = {}
        if not self.results_dir.exists():
            return out
        for path in sorted(self.results_dir.glob("*.json")):
            try:
                out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
        return out

    @staticmethod
    def scrub(text: str) -> str:
        """Redact anything resembling an HF token / API key before logging."""
        return _SECRET_RE.sub("[REDACTED]", text)
