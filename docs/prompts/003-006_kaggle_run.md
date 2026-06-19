# Prompt log 003–006 — Phases 3–6: the automated Kaggle run (Hard Stop 1)

**Phases:** 3 (env + download + sharding) · 4 (FP16 baseline OOM) · 5 (AirLLM) ·
6 (quantization sweep) — combined into ONE self-contained Kaggle notebook driven via
the Kaggle API, per `../KAGGLE_AUTOMATION.md`.
**Authors:** Abdallah Khaldi, Tasneem Natour
**Date:** 2026-06-19

## Prompt issued

> Phases 3–6 (AUTOMATED via the Kaggle API). Build ONE self-contained Kaggle notebook
> that runs the entire experiment headless on a free T4 (download Qwen2.5-14B → FP16
> OOM baseline → AirLLM → 8bit → 4bit, writing all results to /kaggle/working/results),
> create `kernel-metadata.json`, push with `uv run kaggle kernels push -p experiments/`,
> then STOP and report the kernel URL. All unit tests still mock everything — the suite
> never needs a GPU.

## What was built (the tested src the notebook imports)

- **`measure/`** — `harness.measure(generate_fn, n_out, watts, ...)` (injectable
  clock/RSS/VRAM) returning `{ttft_s, tpot_ms, throughput_tok_s, peak_ram_gb,
  peak_vram_gb, total_s, est_power_wh, n_out}`; TTFT = a 1-token timed call (Prefill),
  TPOT = `(total−ttft)/(n−1)` (Decode). `timing.py` guards the torch VRAM probe.
- **`runners/`** — `_common.py` (lazy `torch`/`transformers`/`airllm` imports;
  torch-free OOM detection; the AirLLM factory + generate; shard manifest);
  `baseline.py` (naive FP16 load → captures `OutOfMemoryError` + the 29.4 GB > 16 GB
  math, bottleneck = memory); `airllm_run.py` (AirLLM layer-by-layer, measured);
  `quant_run.py` (the FP16/8bit/4bit sweep + a quality note); `download.py` (the
  download+shard wrapper).
- **SDK** — `run_baseline()`/`run_airllm()`/`run_quant_sweep()` wired, each recording
  to the measurement ledger; a `results_dir` override so the notebook writes the
  ledger straight to the captured Kaggle output.
- **`experiments/airllm_benchmark.ipynb`** — installs `airllm`/`bitsandbytes`,
  pip-installs our package from the public repo, captures the T4 spec, then runs
  baseline → AirLLM → 8bit → 4bit via the SDK, printing the OOM, the AirLLM output,
  and a summary table (the rendered cells are the "screenshots").
- **`experiments/kernel-metadata.json`** (`abdallahkhaldi07/cosmos77-ex05-airllm-benchmark`,
  GPU + internet on) and **`experiments/SETUP.md`** (reproduction + disk guidance).

All model/GPU/HF I/O is mocked in the suite (≈110 tests, 100% coverage; `torch`,
`airllm`, `bitsandbytes` are NOT installed locally/CI).

## Adversarial pre-flight verification (a 6-reviewer workflow)

Because the run is unattended and expensive, a 6-dimension adversarial workflow
audited the bundle before the push. **Clean:** AirLLM API (checked against the real
`lyogavin/airllm` v2 README), the FP16 OOM capture, the package/config resolution on
Kaggle, the ledger contract. **Fixed three blocking findings:**
1. *Shared overlay disk* — `/kaggle/temp` + `/kaggle/working` are ONE ~58–73 GB pool
   and the FP16/8bit/4bit shard sets would accumulate (~51 GB). Fix: shards on
   `/kaggle/temp/shards`, the download kept once (`delete_original=False`), **shards
   cleared between every scenario**, a Cell-1 disk guard, and a documented 7B fallback.
2. *Results stranded on failure* — the ledger was only copied to the captured output
   in the last cell. Fix: the SDK writes the ledger **directly** to
   `/kaggle/working/results` from the first record; each stage is wrapped in
   `run_stage(...)` (try/except, continue); the quant sweep runs 8bit and 4bit as
   **separate** recorded calls, so one failure cannot discard earlier scenarios.
3. *Docs/config path drift* — `SETUP.md` + `PRD_airllm.md` reconciled to
   `/kaggle/temp/shards` with the shared-overlay explanation.

## Push + Hard Stop 1

`uv run kaggle kernels push -p experiments/` → kernel
`https://www.kaggle.com/code/abdallahkhaldi07/cosmos77-ex05-airllm-benchmark`. The run is then
left to execute on the free T4 (AirLLM is slow; this takes a while). **Per the
operating contract we STOP here** and wait for "the Kaggle run is complete" before
fetching `results/*.json` and committing the measured numbers. No number is ever
fabricated — only what the kernel produces is committed.
