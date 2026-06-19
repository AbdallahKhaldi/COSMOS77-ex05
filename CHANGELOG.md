# Changelog

All notable changes to COSMOS77-ex05 are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses a single
course-mandated version line starting at **1.00** (CLAUDE.md rule 10).

## [1.00] — 2026-06-20

**HW5 complete.** Ran `Qwen/Qwen2.5-14B-Instruct` (~29 GB FP16) on a free Kaggle
**Tesla T4**: the naive FP16 load **OOMs** (D2); **AirLLM** runs the same model in
**1.6–3.2 GB** of VRAM via layer-by-layer paging (D3) at ~0.007–0.04 tok/s; the
**bitsandbytes quant sweep** (8-bit, 4-bit) makes 4-bit ~6× faster than FP16 (D4). Every
number flows from the committed measurement ledger (`results/*.json`) — measured on the
T4, never fabricated. Deliverables: hardware capture + param→memory math (D1); the
measurement harness (TTFT/TPOT/throughput/peak VRAM+RAM, D5); tables + 7 figures incl.
the **Roofline** (D6/D9); the On-Prem-vs-API **break-even** economic analysis (D7);
concept linking to Lecture 08 (D8); a quantization speed-vs-VRAM **Pareto** extension
(D10); the deep **report-as-README** (D11); reproducible from a fresh clone (D12/D13);
the §4 research questions answered (D14); an honesty/acceptance audit (D15). 147 tests,
100% coverage, GPU-free CI; built with Claude Code across phases 0–12.

### Added (Phase 0 — repo bootstrap)
- Repository scaffold: `src/cosmos77_ex05/` package skeleton (constants + CLI
  entry point + empty `sdk/`, `shared/`, `hardware/`, `runners/`, `measure/`,
  `analysis/`, `economics/` subpackages), `tests/`, `docs/`, `config/`, and the
  deliverable directories `experiments/`, `results/`, `reports/`, `figures/`,
  `data/`.
- Tooling ported from `COSMOS77-ex04`: `pyproject.toml` (project `cosmos77-ex05`
  v1.00, Python `>=3.11,<3.12`, light analysis deps + an `experiment` optional
  extra for the GPU-only `airllm`/`torch`/`bitsandbytes` stack, dev group with
  `kaggle`, ruff/coverage-85/pytest config), `.pre-commit-config.yaml`,
  `.github/workflows/ci.yml`, `scripts/check_line_cap.py`,
  `scripts/generate_cover_pdf.py` (retargeted to ex05 / exercise 5).
- Configuration: `config/setup.json` (Qwen2.5-14B on a Kaggle T4, the 4 scenarios,
  hardware assumptions for the economics), `config/pricing.json` (API token prices
  + cloud-GPU $/hr + prompt-caching discount), `config/logging_config.json`.
  `.env.example` (HF_TOKEN optional; no secrets).
- Governance: `CLAUDE.md` (the 17 binding rules), `CONTRIBUTING.md`, `LICENSE`
  (MIT 2026, both authors), `README.md` (placeholder — becomes the report in
  Phase 10).

[1.00]: https://github.com/AbdallahKhaldi/COSMOS77-ex05/releases/tag/v1.00
