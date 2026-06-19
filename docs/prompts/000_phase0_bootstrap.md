# Prompt log 000 — Phase 0: Repo bootstrap

**Phase:** 0 — Repo bootstrap + tooling (reuse HW1–HW4)
**Authors:** Abdallah Khaldi, Tasneem Natour
**Date:** 2026-06-19

## Prompt issued

> Senior engineering pair, deliver HW5 ("Running a Massive LLM Locally: AirLLM,
> Quantization & Performance Benchmarking") for Orchestration of AI Agents
> (203.3763). Authority: `../CLAUDE_CODE_PLAYBOOK.md` (§1 the 17 rules, §1.5
> acceptance D1–D15, §16 the `CLAUDE.md` body), then `./CLAUDE.md`. Phase 0 goal:
> repo skeleton, tooling, `CLAUDE.md`, git init + remote + first push — **no
> business logic**. Reuse the proven tooling from `~/COSMOS77/HW4/COSMOS77-ex04/`.
> The grade is the ANALYSIS, not the model.

## What was done

- **Environment audit.** Confirmed `uv` 0.11.9, `gh` 2.86 (authed as
  `AbdallahKhaldi`), `git` present. Kaggle creds `~/.kaggle/kaggle.json` present
  (username `abdallahkhaldi07`); the `kaggle` CLI is added as a dev dep (installed
  by `uv sync`). Qwen2.5-14B is **ungated** → no HF token required for the run.
- **Repo created:** `https://github.com/AbdallahKhaldi/COSMOS77-ex05` (public),
  as a subdirectory of `~/COSMOS77/HW5/` so `CLAUDE.md` resolves the playbook at
  `../CLAUDE_CODE_PLAYBOOK.md`.
- **Scaffold** (ported + adapted from `COSMOS77-ex04`, package renamed
  `cosmos77_ex05`):
  - `pyproject.toml` — project `cosmos77-ex05` v1.00, Python `>=3.11,<3.12`; light
    analysis deps (`matplotlib`, `pandas`, `numpy`, `psutil`, `python-dotenv`,
    `pydantic>=2.6`, `rich`, `pyyaml`); an **`experiment` optional extra** holding
    the GPU-only heavy stack (`airllm`, `transformers`, `torch`, `accelerate`,
    `bitsandbytes`, `huggingface-hub`) so **CI never needs a GPU**; dev group with
    `kaggle`; `cosmos77-airllm` console script; ruff/coverage(85)/pytest config.
  - `.gitignore` (ignores `data/*`, `shards/`, `kaggle_out/`, `.ipynb_checkpoints/`,
    `.env`, `kaggle.json`; **keeps** `results/*.json`, `figures/*.png`,
    `experiments/*.ipynb`), `.env.example` (HF_TOKEN optional; no secrets),
    `.python-version`.
  - `config/setup.json` (Qwen2.5-14B on a Kaggle T4, the 4 scenarios
    fp16_baseline/airllm_none/airllm_8bit/airllm_4bit, `max_new_tokens=20`,
    hardware assumptions for the economics), `config/pricing.json` (OpenAI/
    Anthropic/Google token prices + cloud-GPU $/hr + a prompt-caching discount),
    `config/logging_config.json`.
  - `CLAUDE.md` (the 17 rules, §16 verbatim), `README.md` (placeholder + CI badge),
    `LICENSE` (MIT 2026), `CHANGELOG.md` (`[1.00]`), `CONTRIBUTING.md` (the two-
    environment note: our uv code vs the Kaggle/Colab experiment env).
  - `scripts/check_line_cap.py` (ported), `scripts/generate_cover_pdf.py` (ported,
    retargeted to ex05 + exercise 5).
  - `src/cosmos77_ex05/` skeleton (`__init__` v1.00, `constants.py` with
    `SCENARIOS` + `BYTES_PER_PARAM` + `PIPELINE_STAGES`, `cli/main.py` dispatcher,
    and empty `sdk/shared/hardware/runners/measure/analysis/economics` packages),
    `tests/` with the Phase-0 constants smoke test.
  - `.pre-commit-config.yaml` and `.github/workflows/ci.yml` (ruff, format,
    line-cap, pytest with the 85% coverage gate; live GPU/model/Kaggle tests
    excluded via the `live` marker — CI is CPU-only).

## Verification

```bash
uv sync
uv run ruff check .            # zero
uv run ruff format --check .   # clean
uv run python scripts/check_line_cap.py   # 0 offenders
uv run pytest -m "not live"    # green, coverage >= 85%
uv run cosmos77-airllm --version   # cosmos77-airllm 1.00
```

## Notes / decisions

- **Two environments.** Our repo code is `uv`-only and CPU-only (CI mocks all
  model/GPU/HF I/O). The heavy experiment (`airllm`/`torch`/`bitsandbytes`) runs
  on the Kaggle T4 via the notebook (Phase 3) — kept in the optional extra.
- **No business logic** in Phase 0: shared modules + the runners/measure/analysis/
  economics land in Phase 2 onward.
- **Kaggle automation** (`../KAGGLE_AUTOMATION.md`) drives Phases 3–6 via the API;
  `~/.kaggle/kaggle.json` stays outside the repo and is never committed.
- **Dual authorship.** Commits alternate between both partners (Phase-11 audit).
