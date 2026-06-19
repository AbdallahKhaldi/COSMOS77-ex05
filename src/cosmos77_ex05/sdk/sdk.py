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

    def __init__(
        self,
        config: Config | None = None,
        gatekeeper: Gatekeeper | None = None,
        results_dir: Path | str | None = None,
    ) -> None:
        self.config = config or Config()
        # results_dir override lets the Kaggle notebook write the ledger DIRECTLY to
        # the captured output dir (/kaggle/working/results) so every record survives
        # even if a later stage crashes.
        if results_dir is not None:
            self.results_dir = Path(results_dir)
        else:
            self.results_dir = self.repo_root / self.config.paths().get("results_dir", "results")
        self.gatekeeper = gatekeeper or Gatekeeper(self.results_dir)

    @property
    def repo_root(self) -> Path:
        """The repository root (the parent of the ``config/`` directory)."""
        return self.config.config_dir.parent

    def capture_hardware(self) -> dict[str, Any]:
        """Capture the machine spec → results/hardware.json + the param→memory math (D1)."""
        from cosmos77_ex05.hardware.model_math import justify
        from cosmos77_ex05.hardware.spec import capture_spec

        spec = capture_spec(self.results_dir / "hardware.json")
        experiment = self.config.experiment()
        params = float(experiment.get("model_params_billions", 14.7)) * 1e9
        vram = spec.get("gpu", {}).get("vram_gb") or float(experiment.get("target_vram_gb", 16))
        math = justify(experiment.get("model_id", "Qwen/Qwen2.5-14B-Instruct"), params, vram)
        return {"spec": spec, "model_math": math}

    def run_baseline(self) -> dict[str, Any]:
        """Run the naive FP16 direct load (expected OOM) and record fp16_baseline (D2)."""
        from cosmos77_ex05.runners.baseline import run_fp16_baseline

        exp = self.config.experiment()
        params = float(exp.get("model_params_billions", 14.7)) * 1e9
        vram = float(exp.get("target_vram_gb", 16))
        metrics = run_fp16_baseline(exp["model_id"], exp["prompt"], params, vram)
        self.gatekeeper.record("fp16_baseline", metrics)
        return metrics

    def run_airllm(self) -> dict[str, Any]:
        """Run the same model via AirLLM, layer-by-layer, and record airllm_none (D3)."""
        from cosmos77_ex05.runners.airllm_run import run_airllm

        exp = self.config.experiment()
        metrics = run_airllm(
            exp["model_id"],
            exp["prompt"],
            int(exp["max_new_tokens"]),
            exp["layer_shards_saving_path"],
            self._watts(),
            max_seq_len=int(exp.get("max_seq_len", 128)),
        )
        self.gatekeeper.record("airllm_none", metrics)
        return metrics

    def run_quant_sweep(self, levels: list[str] | None = None) -> dict[str, Any]:
        """Run the quantization sweep (default 8bit + 4bit) and record each (D4)."""
        from cosmos77_ex05.runners.quant_run import run_quant_sweep

        exp = self.config.experiment()
        results = run_quant_sweep(
            exp["model_id"],
            exp["prompt"],
            int(exp["max_new_tokens"]),
            exp["layer_shards_saving_path"],
            levels or ["8bit", "4bit"],
            self._watts(),
        )
        for level, metrics in results.items():
            self.gatekeeper.record(f"airllm_{level}", metrics)
        return results

    def _watts(self) -> float:
        """The configured GPU power draw used for the energy estimate."""
        return float(self.config.hardware_assumptions().get("gpu_power_watts", 70))

    def measure(self) -> Any:
        """The measurement harness runs inside the runners on the GPU (notebook), not here."""
        raise NotImplementedError("measurement happens in the runners on the T4, not on the Mac")

    def analyze(self) -> dict[str, Any]:
        """Build the comparison tables + graphs + Roofline from the ledger (D6/D9)."""
        from cosmos77_ex05.analysis import plots, roofline, tables

        ledger = self.gatekeeper.ledger()
        reports_dir = self.repo_root / self.config.paths().get("reports_dir", "reports")
        figures_dir = self.repo_root / self.config.paths().get("figures_dir", "figures")
        metrics_md = tables.write_metrics_md(ledger, reports_dir / "METRICS.md")
        figs = list(plots.generate_all(ledger, figures_dir))
        figs.append(roofline.plot_roofline(ledger, figures_dir))
        return {"metrics_md": metrics_md, "figures": figs, "scenarios": list(ledger)}

    def economics(self) -> Any:
        """The On-Prem vs API vs Cloud-GPU break-even report (Phase 8)."""
        raise NotImplementedError("economics lands in Phase 8")

    def report(self) -> Any:
        """Assemble the README technical report (Phase 10)."""
        raise NotImplementedError("report lands in Phase 10")

    def ledger(self) -> dict[str, Any]:
        """Return the aggregated measurement ledger (all results/*.json)."""
        return self.gatekeeper.ledger()
