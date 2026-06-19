# Acceptance audit — D1–D15

Every HW5 acceptance criterion (`../CLAUDE_CODE_PLAYBOOK.md` §1.5), the artifact that
satisfies it, and its status. Audited at Phase 11; all green.

| # | Criterion | Artifact(s) | Status |
|---|---|---|---|
| **D1** | Hardware spec + quantitative model-choice math | `results/hardware.json`, `hardware/spec.py` + `model_math.py`, README §1 (29.4 GB > 16 GB) | ✅ |
| **D2** | Baseline direct run (OOM) + bottleneck identified | `results/fp16_baseline.json` (real `OutOfMemoryError`), `runners/baseline.py`, `reports/baseline.md` | ✅ |
| **D3** | AirLLM runs the same model | `results/airllm_none.json` (success, coherent output), `runners/airllm_run.py`, `reports/airllm.md` | ✅ |
| **D4** | Quantization sweep FP16/8-bit/4-bit | `results/airllm_{8bit,4bit}.json`, `runners/quant_run.py`, `reports/quantization.md` | ✅ |
| **D5** | Systematic metrics (TTFT/TPOT/throughput/peak mem/power/quality) | `measure/harness.py` + `timing.py`, every `results/*.json` | ✅ |
| **D6** | Tables + graphs | `reports/METRICS.md`, `figures/{tokens_per_sec,peak_vram,ttft_vs_tpot,quant_tradeoff}.png` | ✅ |
| **D7** | Economic break-even (On-Prem vs API vs Cloud) + assumptions + caching + privacy | `reports/ECONOMICS.md`, `figures/breakeven.png`, `economics/{model,caching,report,breakeven_plot}.py` | ✅ |
| **D8** | Lecture-concept analysis | `reports/CONCEPTS.md` (paging, compute/memory-bound, disk-BW wall, Prefill/Decode, quantization) | ✅ |
| **D9** | Roofline from measurements | `figures/roofline.png`, `analysis/roofline.py` (AI 1–4 ≪ ridge 203; ≈0.2–0.3 GB/s disk BW) | ✅ |
| **D10** | ≥1 original extension | `extensions/pareto.py`, `reports/EXTENSIONS.md`, `figures/pareto.png` (quant speed-vs-VRAM Pareto) | ✅ |
| **D11** | Report-as-README + embedded visuals | `README.md` (264 lines, 7 embedded figures) | ✅ |
| **D12** | Repo structure (src/experiments/results/reports/figures) | repo tree | ✅ |
| **D13** | Reproducible; isolated env; no HF token in code | `experiments/SETUP.md`, `.env.example`, fresh-clone test (clone→sync→pytest→regenerate) | ✅ |
| **D14** | The §4 research questions answered | README §7 (RQ-a…RQ-f), `docs/PRD.md` §3 | ✅ |
| **D15** | Honesty: real measured numbers; negatives explained | this file + ledger match; README §10 limitations; the P100→T4 reproducibility log | ✅ |

## QA gauntlet (Phase 11)

| Check | Result |
|---|---|
| `ruff check` / `ruff format --check` | clean |
| `check_line_cap.py` (≤150 lines/.py) | 0 offenders |
| `pytest --cov-fail-under=85` | 143 passed, **100% coverage**, GPU-free (all model/GPU/HF I/O mocked) |
| `uv lock --check` | consistent |
| Secrets: no `.env`/`kaggle.json`/token in tracked files; `.env.example` only | ✅ |
| Conventional Commits, both authors, no wip/tmp, no Claude trailer | 43 commits (Abdallah + Tasneem) |
| Fresh-clone reproducibility (clone → `uv sync` → pytest → `analyze`) | figures regenerate identically |
| GitHub Actions | green |

## Honesty statement (D15)

Every number in the README, the reports, and the figures is generated from the committed
measurement ledger `results/*.json` — produced by the notebook on a real Kaggle **Tesla
T4**. Nothing is hand-edited or fabricated; `uv run cosmos77-airllm analyze` re-derives
every figure and table from the ledger on a fresh clone. Where the live environment hit
real limits (a Pascal P100 with no usable kernels; AirLLM's incompatibility with
`transformers` ≥ 4.48), they were fixed or documented — never faked.
