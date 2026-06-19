# PRD — COSMOS77-ex05: Running a Massive LLM Locally with AirLLM + Quantization

> Course: **Orchestration of AI Agents (203.3763)** — Dr. Yoram Segal (UOH).
> HW5 — *Running a Massive LLM Locally: AirLLM, Quantization & Performance Benchmarking.*
> Authors: Abdallah Khaldi (212389712), Tasneem Natour (323118794). Version 1.00.

## 1. Problem statement

We deliberately pick a model that does **not** fit the hardware — `Qwen/Qwen2.5-14B-Instruct`
(~14.7 B parameters, **~29 GB in FP16**) — and try to run it on a **free 16 GB Kaggle T4**.
The naive FP16 load OOMs; we then make the *same* model run via **AirLLM** (layer-by-layer
`mmap` streaming, the "layer = page" trick) and a **bitsandbytes quantization sweep**
(FP16 → 8bit → 4bit), measuring **TTFT / TPOT / throughput / peak VRAM + RAM** for every
scenario, tying each number to a lecture concept, and closing with an honest **On-Prem-vs-API**
break-even economic analysis. The honest-measurement thesis governs everything: **the grade is
the analysis, not the model.** Every reported number flows from the committed measurement
**ledger** (`results/*.json`) — nothing is fabricated; a well-explained negative result (an OOM,
1–3 tok/s) is a valid, gradable outcome. Substance = **measured numbers + correct causal
explanation + honest economics.**

## 2. Context — Lecture 08, local inference

This work operationalises **Lecture 08** (local inference, LoRA / AirLLM). The core mental model
is the **virtual-memory / Paging analogy**: an operating system runs a program larger than physical
RAM by keeping only the *active* pages resident and paging the rest from disk on demand. AirLLM
applies the identical idea to a transformer — it keeps only the **active layer** in VRAM and
streams the others from disk (`mmap`, zero-copy), so **"a layer = a page"** and loading a layer is
a **page fault**. This is why a 29 GB model "fits" in 16 GB of VRAM: at no point is the whole model
resident. The price of paging is bandwidth: **disk BW (~3–7 GB/s) ≪ HBM BW (~320 GB/s)**, which
predicts the slow ~1–3 tok/s we expect.

Two phases of inference structure the whole report:
- **Prefill** — process the prompt in one pass. It is a big matrix-matrix multiply (**GEMM**),
  **compute-bound**, and shows up as **TTFT** (time-to-first-token).
- **Decode** — generate tokens one at a time, each a matrix-vector multiply (**GEMV**) re-reading
  all weights + the **KV-cache**, so it is **memory-bound** and shows up as **TPOT** (time-per-
  output-token / inter-token latency).

**Why a 14B model OOMs a 16 GB T4:** memory = params × bytes/param. FP16 = 2 bytes ⇒
14.7 B × 2 ≈ **29.4 GB** of weights alone, before activations / KV-cache / CUDA context. That is
**~1.8×** the T4's 16 GB VRAM ⇒ a clean `torch.cuda.OutOfMemoryError`. **Quantization** lowers
bytes/param (Q8 = 1 → ~14.7 GB; Q4 = 0.5 → ~7.4 GB), trading numerical precision for memory and
moving the model from "impossible" to "fits". The bottleneck here is **memory, not compute** —
the central diagnosis the report must defend.

## 3. Research questions (spec §4) — requirements the report MUST answer

Each is a binding requirement; the report answers each explicitly, every claim tied to a ledger
number or a lecture citation.

### RQ-a — What is the bottleneck (RAM/VRAM vs compute), and how is it identified?
The report must state the bottleneck is **memory (VRAM capacity + bandwidth)**, not compute, and
**show how it is identified**: the param→bytes math (29.4 GB > 16 GB) predicts OOM; the captured
`torch.cuda.OutOfMemoryError` (requested vs available VRAM) confirms it; AirLLM's slowness despite
idle compute confirms it is **bandwidth-bound**, not FLOP-bound. Tie to the Roofline (decode AI ≈ 1
⇒ memory-bound).

### RQ-b — How does AirLLM change resource allocation, and what is the Paging connection?
Explain that AirLLM trades **VRAM for time**: instead of one 29 GB resident allocation it holds one
layer at a time and **streams** the rest via `mmap` per-layer **SafeTensors** shards. Make the
**Paging connection explicit**: layer = page, layer load = page fault, `mmap` = zero-copy mapping,
evict-after-use = page replacement. Report peak VRAM falling from "impossible" to single-layer size.

### RQ-c — What is quantization's effect on memory/speed/quality, and where is the accuracy red line?
For FP16 / Q8 / Q4 report the **memory** drop (×2, ×4 vs FP16), the **speed** change, and a
**qualitative quality** note on the same prompt. Identify the **accuracy "red line"** — the level at
which the answer visibly degrades. Tie to the lecture: FP32→FP16 halves memory cheaply; Q8 is near-
lossless; Q4 is aggressive but usable; **NF4** (QLoRA) is the 4-bit format of choice; Q2 is smoke-
test only.

### RQ-d — How do Prefill / Decode show up as TTFT vs TPOT?
Map **Prefill → TTFT** (compute-bound GEMM) and **Decode → TPOT** (memory-bound GEMV + KV-cache).
Use the measured TTFT/TPOT split per scenario to demonstrate the two regimes and explain why AirLLM
inflates *both* (every layer is a page fault on every step).

### RQ-e — What is the Throughput/Latency price of a big model on modest hardware?
Quantify the cost honestly: AirLLM throughput (~1–3 tok/s) vs a resident-model baseline; per-token
latency (TPOT); total runtime; estimated power (watts × time). State plainly that paging buys
*feasibility* at a large *latency* cost.

### RQ-f — When does On-Prem beat API?
Deliver a **break-even**: cumulative cost (On-Prem CAPEX amortised + electricity OPEX) vs API
(tokens × price) vs optional Cloud-GPU ($/hr), with the crossover volume, all assumptions stated, the
**prompt-caching** effect on the API line, and a **privacy/security-aware** recommendation.

## 4. Functional requirements → acceptance criteria D1–D15

| FR | Requirement | Maps to | Phase | Primary artifact |
|----|-------------|---------|-------|------------------|
| FR-1 | Capture hardware (CPU/cores/RAM/GPU/VRAM/disk) + quantitative param→memory model-choice math | **D1** | 2 | `results/hardware.json`, `hardware/spec.py`, `model_math.py` |
| FR-2 | Run the naive FP16 direct load; capture OOM (or unbearable slowness); identify bottleneck = memory | **D2** | 4 | `results/fp16_baseline.json`, `reports/baseline.md` |
| FR-3 | Integrate AirLLM; run the SAME prompt on the SAME model (layer = page) | **D3** | 5 | `results/airllm_none.json`, `reports/airllm.md` |
| FR-4 | Quantization sweep FP16 / 8bit / 4bit; memory/speed/quality + accuracy red line | **D4** | 6 | `results/airllm_{8bit,4bit}.json`, `reports/quantization.md` |
| FR-5 | Systematic metrics per scenario: TTFT, TPOT/ITL, throughput, peak RAM+VRAM, runtime, est. power, quality | **D5** | 7 | `src/measure/`, `results/*.json` |
| FR-6 | Tables + graphs of all metrics | **D6** | 7 | `reports/METRICS.md`, `figures/*.png` |
| FR-7 | Economic analysis On-Prem vs API (+ optional Cloud GPU) with break-even graph + assumptions + caching + privacy | **D7** | 8 | `reports/ECONOMICS.md`, `figures/breakeven.png` |
| FR-8 | Link every result to lecture concepts (Prefill/Decode, compute/memory-bound, VRAM, Paging, quantization) | **D8** | 9 | `reports/CONCEPTS.md` |
| FR-9 | Roofline model sketched from the measurements | **D9** | 7, 9 | `figures/roofline.png`, `analysis/roofline.py` |
| FR-10 | ≥1 original extension (model sizes / QLoRA / quant Pareto / CPU-GPU-AirLLM three-way) | **D10** | 9 | `extensions/`, `reports/EXTENSIONS.md` |
| FR-11 | Report-as-README: the deep report IS `README.md`, every graph/table/screenshot embedded | **D11** | 10 | `README.md` |
| FR-12 | Repo structure: `src/`, `experiments/`, `results/`, `reports/`, `figures/` | **D12** | 0 | repo tree |
| FR-13 | Reproducible run instructions; isolated env; no HF token in code | **D13** | 3, 11 | `experiments/SETUP.md`, `.env.example` |
| FR-14 | The §4 research questions answered explicitly | **D14** | 1, 10 | `docs/PRD.md` §3, `README.md` |
| FR-15 | Honesty: real measured numbers; a well-analysed negative result is acceptable, never faked | **D15** | 11 | `docs/ACCEPTANCE.md` + ledger match |

## 5. Non-functional requirements

- **NFR-1 Honest measurement (the ledger).** Every number flows through
  `src/cosmos77_ex05/shared/gatekeeper.py` into `results/*.json`. The ledger is the **single source
  of truth** for every table and graph; nothing is "true" unless it is in the ledger. No fabrication.
- **NFR-2 Reproducibility on a free T4.** The experiment reproduces on a **free Kaggle T4** (Colab
  T4 documented fallback) from `experiments/SETUP.md`: enable GPU, set `HF_TOKEN` secret, run cells
  top-to-bottom. Analysis pipeline re-derives all figures/tables from the committed `results/*.json`.
- **NFR-3 English + lecture vocabulary.** All prose in English using the exact vocabulary: Prefill
  (GEMM, compute-bound) / Decode (GEMV, memory-bound), VRAM, KV-cache, TTFT/TPOT, throughput/latency,
  quantization (FP16/Q8/Q4, NF4), the virtual-memory/Paging analogy ("layer = page", `mmap`),
  SafeTensors, Roofline, On-Prem vs API, prompt caching.
- **NFR-4 150-line cap per `.py` file.** Hard, no exceptions; split modules that approach it.
- **NFR-5 ≥85% coverage** on the measurement / analysis / economic / plot code (the deterministic,
  GPU-free modules). The live notebook runs are not part of the coverage gate.
- **NFR-6 Tests never touch real I/O.** ALL model/GPU/HF/network I/O is mocked; the suite never
  downloads a model and never needs a GPU. Deterministic: seeded `random`, fixed prompts, no flakes.
- **NFR-7 Secrets only in `.env` / Kaggle secrets.** `.env.example` only is tracked; `HF_TOKEN`
  and any API key never appear in code, notebook, or committed files.
- **NFR-8 `uv` only for our code.** The Kaggle/Colab runtime uses its own `pip` — that is the
  experiment environment, documented separately, not part of our package management.
- **NFR-9 SDK single entry + config-driven.** One `class SDK` at `src/cosmos77_ex05/sdk/sdk.py`;
  the notebook and CLI call only the SDK. Model id, quant levels, prices, hardware assumptions, and
  paths live in `config/setup.json`, `config/pricing.json`, and `.env`.

## 6. KPIs / Definition of Done

The assignment is **Done** when all of the following hold and the README numbers match the ledger
exactly:

| # | KPI / DoD gate | Evidence |
|---|----------------|----------|
| K-1 | Hardware documented **and** param→memory math present (params × {FP16=2, Q8=1, Q4=0.5} bytes) | `results/hardware.json`, README §3 |
| K-2 | **FP16 baseline OOM shown** (captured `OutOfMemoryError`, requested vs available VRAM) | `results/fp16_baseline.json`, OOM screenshot |
| K-3 | **AirLLM runs the SAME model** on the SAME prompt and produces output | `results/airllm_none.json` |
| K-4 | **Quant sweep** FP16 / Q8 / Q4 completed | `results/airllm_{none,8bit,4bit}.json` |
| K-5 | **Full metric set per scenario** (TTFT, TPOT, throughput, peak RAM+VRAM, runtime, est. power, quality) | `results/*.json` |
| K-6 | **Tables + graphs** generated from the ledger | `reports/METRICS.md`, `figures/{tokens_per_sec,peak_vram,ttft_vs_tpot,quant_tradeoff}.png` |
| K-7 | **On-Prem-vs-API break-even graph** with stated assumptions + caching shift | `figures/breakeven.png`, `reports/ECONOMICS.md` |
| K-8 | **Concept linking** — every result tied to Prefill/Decode, compute/memory-bound, VRAM, Paging, quantization | `reports/CONCEPTS.md` |
| K-9 | **Roofline** sketched from measurements (T4 ≈65 TFLOPS / ≈320 GB/s ⇒ ridge ≈203) | `figures/roofline.png` |
| K-10 | **≥1 original extension** implemented + documented | `extensions/`, `reports/EXTENSIONS.md` |
| K-11 | **Report-as-README** with all figures/tables embedded (≥250 lines, ≥6 figures) | `README.md` |
| K-12 | **≥85% coverage** on analysis code; `ruff check` zero; line-cap zero | CI green, `docs/ACCEPTANCE.md` |
| K-13 | **≥600 TODO items** distributed P0–P12 | `docs/TODO.md` |
| K-14 | **§4 research questions answered explicitly** (RQ-a … RQ-f) | README, this PRD §3 |
| K-15 | **Honesty gate** — README numbers byte-match `results/*.json`; negatives explained, never faked | `docs/ACCEPTANCE.md` |

## 7. Out of scope

- **Output-quality optimisation / prompt engineering for a "better" answer** — the grade is the
  analysis, not the model's eloquence; a degraded Q4 answer is data, not a defect.
- **Production serving** (vLLM/TGI, batching, multi-GPU, PagedAttention serving) — referenced only
  as conceptual context for prompt caching in the economics, not built.
- **Training / full fine-tuning** — only the optional QLoRA *extension* touches adapters, as a demo.
- **Models other than `Qwen/Qwen2.5-14B-Instruct`** for the core runs (other sizes appear only in the
  optional model-size extension).
- **Running quantization on the Mac** — bitsandbytes is CUDA-only; all heavy runs are on the T4.
- **Inference inside the unit-test suite** — all model/GPU/HF/network I/O is mocked; live runs live
  in `experiments/airllm_benchmark.ipynb` on the free T4.
