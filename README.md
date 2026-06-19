# COSMOS77-ex05 — Running a Massive LLM Locally with AirLLM + Quantization

> **Orchestration of AI Agents (203.3763), Dr. Yoram Segal · HW5**
> Authors: **Abdallah Khaldi** (212389712) · **Tasneem Natour** (323118794)

[![CI](https://github.com/AbdallahKhaldi/COSMOS77-ex05/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdallahKhaldi/COSMOS77-ex05/actions/workflows/ci.yml)

> ⚠️ **Placeholder README — becomes the full technical report in Phase 10.**

## What this project measures

Take a model too big for the hardware — **`Qwen/Qwen2.5-14B-Instruct`** (~29 GB in
FP16) on a free **16 GB Kaggle T4** — show the naive FP16 load **OOMs**, then make
the *same* model run with **AirLLM** (layer-by-layer mmap — "layer = page") and a
**bitsandbytes quantization sweep** (FP16 → 8bit → 4bit). Every run is measured
(**TTFT, TPOT, throughput, peak VRAM + RAM**), tied to the lecture concepts
(Prefill/Decode, compute- vs memory-bound, the virtual-memory/Paging analogy,
Roofline), and capped with an honest **On-Prem-vs-API break-even** economic
analysis. **The grade is the analysis, not the model** — every number comes from a
committed measurement ledger (`results/*.json`); nothing is fabricated.

## Status

Bootstrapped in **Phase 0**. The pipeline is built phase-by-phase per
`../CLAUDE_CODE_PLAYBOOK.md`; see [`docs/TODO.md`](docs/TODO.md) for progress and
[`CLAUDE.md`](CLAUDE.md) for the 17 binding rules.

## Where it runs

- **Our code** (measurement, analysis, economics, plots): `uv`-managed, CPU-only,
  fully unit-tested with all model/GPU/HF I/O mocked.
- **The experiment** (the heavy live runs): a free **Kaggle T4** (CUDA) via
  `experiments/airllm_benchmark.ipynb` — `airllm` + `bitsandbytes` need CUDA.
  Qwen2.5-14B is **ungated** (no HF token required).

## Quickstart

```bash
uv sync
uv run cosmos77-airllm --version
uv run cosmos77-airllm hardware     # capture this machine's spec -> results/hardware.json
```

## License

[MIT](LICENSE) © 2026 Abdallah Khaldi and Tasneem Natour.
