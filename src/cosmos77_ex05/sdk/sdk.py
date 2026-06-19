"""The single business-logic entry point (CLAUDE.md rule 2).

The CLI, the Kaggle notebook, and the tests all go through ``class SDK`` — one
audited surface, one method per pipeline stage. Each stage lands in its phase (a
``NotImplementedError`` until then). The measurement ledger (Gatekeeper) is
created once over ``results/`` so every table and graph rests on ONE ledger.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cosmos77_ex05.shared.config import Config
from cosmos77_ex05.shared.gatekeeper import Gatekeeper


class SDK:
    """All business logic for the AirLLM benchmarking + analysis pipeline."""

    def __init__(self, config: Config | None = None, gatekeeper: Gatekeeper | None = None) -> None:
        self.config = config or Config()
        results_dir = self.repo_root / self.config.paths().get("results_dir", "results")
        self.gatekeeper = gatekeeper or Gatekeeper(results_dir)

    @property
    def repo_root(self) -> Path:
        """The repository root (the parent of the ``config/`` directory)."""
        return self.config.config_dir.parent

    def capture_hardware(self) -> dict[str, Any]:
        """Capture the machine spec → results/hardware.json + the param→memory math (D1)."""
        from cosmos77_ex05.hardware.model_math import justify
        from cosmos77_ex05.hardware.spec import capture_spec

        results_dir = self.repo_root / self.config.paths().get("results_dir", "results")
        spec = capture_spec(results_dir / "hardware.json")
        experiment = self.config.experiment()
        params = float(experiment.get("model_params_billions", 14.7)) * 1e9
        vram = spec.get("gpu", {}).get("vram_gb") or float(experiment.get("target_vram_gb", 16))
        math = justify(experiment.get("model_id", "Qwen/Qwen2.5-14B-Instruct"), params, vram)
        return {"spec": spec, "model_math": math}

    def run_baseline(self) -> Any:
        """Run the naive FP16 direct load (expected OOM) (Phase 4)."""
        raise NotImplementedError("run_baseline lands in Phase 4")

    def run_airllm(self) -> Any:
        """Run the same model via AirLLM, layer-by-layer (Phase 5)."""
        raise NotImplementedError("run_airllm lands in Phase 5")

    def run_quant_sweep(self) -> Any:
        """Run the quantization sweep FP16/Q8/Q4 (Phase 6)."""
        raise NotImplementedError("run_quant_sweep lands in Phase 6")

    def measure(self) -> Any:
        """The measurement harness — TTFT/TPOT/throughput/peak mem (Phase 7)."""
        raise NotImplementedError("measure lands in Phase 7")

    def analyze(self) -> Any:
        """Build the comparison tables + graphs + Roofline (Phase 7)."""
        raise NotImplementedError("analyze lands in Phase 7")

    def economics(self) -> Any:
        """The On-Prem vs API vs Cloud-GPU break-even report (Phase 8)."""
        raise NotImplementedError("economics lands in Phase 8")

    def report(self) -> Any:
        """Assemble the README technical report (Phase 10)."""
        raise NotImplementedError("report lands in Phase 10")

    def ledger(self) -> dict[str, Any]:
        """Return the aggregated measurement ledger (all results/*.json)."""
        return self.gatekeeper.ledger()
