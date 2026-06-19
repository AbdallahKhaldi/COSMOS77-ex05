"""Load + order the measurement ledger for the analysis layer (rule 13).

Single place that reads ``results/*.json`` (via the Gatekeeper) and orders the
scenarios, so ``tables``/``plots``/``roofline`` never re-implement loading
(rule 3). ``fp16_baseline`` is the OOM control (``success=False``, no throughput);
the three ``airllm_*`` rows carry the full metric set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cosmos77_ex05.constants import SCENARIOS
from cosmos77_ex05.shared.gatekeeper import Gatekeeper

#: Human labels for the four scenarios, in canonical order.
SCENARIO_LABELS: dict[str, str] = {
    "fp16_baseline": "FP16 baseline (OOM)",
    "airllm_none": "AirLLM FP16",
    "airllm_8bit": "AirLLM 8-bit",
    "airllm_4bit": "AirLLM 4-bit",
}


def load_ledger(results_dir: Path | str = "results") -> dict[str, dict[str, Any]]:
    """Return ``{scenario: metrics}`` for every committed ``results/<scenario>.json``."""
    return Gatekeeper(results_dir).ledger()


def ordered(ledger: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Scenarios present in ``ledger``, in canonical ``SCENARIOS`` order."""
    return [(name, ledger[name]) for name in SCENARIOS if name in ledger]


def airllm_rows(ledger: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """The successfully-measured AirLLM scenarios (exclude the OOM baseline)."""
    return [
        (name, metrics)
        for name, metrics in ordered(ledger)
        if name != "fp16_baseline" and metrics.get("success") and not metrics.get("skipped")
    ]


def label(scenario: str) -> str:
    """Pretty label for a scenario id."""
    return SCENARIO_LABELS.get(scenario, scenario)
