# Prompt log 002 — Phase 2: Shared infra + hardware capture + ledger + economics model

**Phase:** 2 — The deterministic, GPU-free core (shared + hardware + ledger + economics)
**Authors:** Abdallah Khaldi, Tasneem Natour
**Date:** 2026-06-19

## Prompt issued

> Phase 2 goal: port the shared layer and build the deterministic, GPU-free parts —
> hardware capture + the measurement ledger + the economic model. All TDD; no
> model/GPU in the suite (`../CLAUDE_CODE_PLAYBOOK.md` §4). Port
> `shared/{version,config,logging_setup,gatekeeper}` from HW4; **repurpose the
> gatekeeper as the measurement LEDGER** (`record(scenario, metrics)` →
> `results/<scenario>.json`; `ledger()` aggregates). `hardware/spec.py` +
> `model_math.py`; `economics/model.py` + `caching.py`; SDK stubs.

## What was done

- **`shared/`** — `version.py` (config-version guard), `config.py` (dot-path loader
  over `setup.json` + `pricing.json` with `experiment()`/`hardware_assumptions()`/
  `paths()`/`pricing()`/`provider_prices()` accessors), `logging_setup.py`, and
  **`gatekeeper.py` repurposed as the measurement ledger**: `record(scenario,
  metrics)` merges into `results/<scenario>.json`, `read()`/`ledger()` aggregate,
  `scrub()` redacts HF tokens. Nothing is "true" unless it is in the ledger.
- **`hardware/spec.py`** (built by a subagent) — `capture_spec(out_path)` captures
  CPU/cores/RAM/GPU+VRAM/disk via `platform`/`psutil`/`shutil` with a **guarded
  `import torch`** (CPU-only sentinel when torch/CUDA is absent — works on the Mac
  AND captures the real T4 on Kaggle) → `results/hardware.json`.
- **`hardware/model_math.py`** — `model_memory(params, dtype)` + `justify(...)`: the
  param→memory math (`14.7e9 × 2 / 1e9 = 29.4 GB` FP16; Q4 = 7.4 GB via
  ROUND_HALF_UP) + the OOM verdict (D1).
- **`economics/model.py` + `caching.py`** (built by a subagent) — `break_even`
  (On-Prem CAPEX+OPEX vs API tokens×price vs Cloud-GPU $/hr; crossover; missing
  assumption RAISES, no silent default) + `apply_caching_discount` (caching lowers
  the API price ⇒ moves break-even to a HIGHER volume).
- **`sdk/sdk.py`** — `capture_hardware()` (real, delegates to `hardware/`) + the
  later stages stubbed; `ledger()`; CLI `hardware` command wired. `config/setup.json`
  gains `model_params_billions` + `target_vram_gb`.

## Verification

```bash
uv run pytest -m "not live"   # 70 passed, coverage 100%
uv run cosmos77-airllm hardware
#   CPU: arm (8 physical cores)
#   RAM: 17.2 GB   GPU: none / CPU-only
#   model: Qwen2.5-14B -> FP16 29.4 GB
#   verdict: needs 29.4 GB > 16.0 GB VRAM -> OOM; Q4 (7.4 GB) fits.
```

## Notes / decisions

- **GPU-free suite.** `torch` is NOT installed (it is the optional `experiment`
  extra); `hardware/spec.py` guards the import and the tests inject a fake torch —
  the suite proves the CPU-only path WITHOUT torch present.
- **The committed `results/hardware.json` here is THIS Mac** (CPU-only) — the Phase
  2 local capture. The Kaggle run (Phases 3–6) captures the real **T4** spec and
  that hardware.json is fetched + committed there. The model-choice math (29.4 GB >
  16 GB T4 ⇒ OOM) holds regardless of the local machine (target VRAM is config-driven).
- The economics `report.py`/`breakeven_plot.py` are deliberately deferred to Phase 8;
  Phase 2 ships only the tested cost-model core.
