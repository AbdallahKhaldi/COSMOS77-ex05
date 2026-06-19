# Changelog

All notable changes to COSMOS77-ex05 are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses a single
course-mandated version line starting at **1.00** (CLAUDE.md rule 10).

## [1.00] — 2026-06-19

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
