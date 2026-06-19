# PRD — Quantization Sweep (mechanism) · COSMOS77-ex05

> Course: **Orchestration of AI Agents (203.3763)** — Dr. Yoram Segal (UOH).
> HW5 — *Running a Massive LLM Locally: AirLLM, Quantization & Performance Benchmarking.*
> Mechanism: **the quantization sweep** · Maps to acceptance **D4** (with D5 metrics, D8 concepts).
> Authors: Abdallah Khaldi, Tasneem Natour. Version 1.00.

## 1. Purpose

Run the **same** Qwen2.5-14B-Instruct prompt that the AirLLM mechanism runs, but sweep the
**compression** knob across `compression ∈ {None (FP16), "8bit", "4bit"}`. AirLLM forwards these
to **bitsandbytes** (CUDA-only), so the sweep happens on the free 16 GB Kaggle T4 — never on the
Mac, never in the test suite. For each level we record the **full metric set** (TTFT / TPOT /
throughput / peak VRAM + peak RAM / runtime / est. power) **plus a qualitative output-quality
note**, then answer the core trade-off question of the lecture: *how far can we push precision
down before the answer crosses the accuracy "red line"?*

This mechanism is the **memory/accuracy trade-off** made measurable. The grade is the analysis,
not the model: a Q4 answer that degrades is **data**, not a defect, provided the degradation is
captured honestly in the ledger and explained. This PRD defines the runner, its SDK surface, and a
fully **mocked** test plan (no GPU, no download) so the deterministic library stays CI-green while
the heavy runs live in `experiments/airllm_benchmark.ipynb`.

## 2. The trade-off this mechanism documents (lecture vocabulary)

Memory of the weights ≈ `params × bytes/param`. For ~14.7 B params:

| Level | bytes/param | Weights ≈ | Fits 16 GB T4? | Lecture role |
|-------|-------------|-----------|----------------|--------------|
| FP32 | 4 | ~58.8 GB | no | reference only (not run) |
| **FP16** (`None`) | 2 | **~29.4 GB** | no (resident) — AirLLM pages it | baseline precision; FP32→FP16 **halves** memory, ~lossless |
| **Q8** (`"8bit"`) | 1 | **~14.7 GB** | borderline | ~half again; near-lossless, the "safe" quant |
| **Q4** (`"4bit"`) | 0.5 | **~7.4 GB** | yes | **aggressive**; faster/smaller, quality **may degrade** |
| NF4 (QLoRA 4-bit) | 0.5 | ~7.4 GB | yes | the **high-quality** 4-bit format (normal-float); QLoRA's base |
| Q2 | 0.25 | ~3.7 GB | yes | **smoke-test only** — usually below the red line |

The **accuracy "red line"** is the precision level at which the answer on the fixed prompt
visibly degrades (loss of coherence, factual drift, repetition, truncation, or refusal). We do
**not** assume where it is — we **find and state it honestly** from the captured quality notes.
Working hypothesis to test, not to assert: FP16 ≈ Q8 (indistinguishable), Q4 usable but softer,
Q2 over the line. If our measured Q4 is already over the line, that is a valid, gradable result.

## 3. Inputs / Outputs

**Inputs**
- `config/setup.json → experiment`: `model_id`, `prompt`, `max_new_tokens`, `max_seq_len`,
  `layer_shards_saving_path`; `quant_levels` (we consume the `airllm_*` subset).
- A **model factory** (callable, injected) that returns a model-like object with `.generate(...)`;
  in production it wraps `airllm.AutoModel.from_pretrained(model_id, compression=<level>, ...)`,
  in tests it is a mock. This is the seam that keeps the runner GPU-free under test.
- The compression label per level: `None → "fp16"/"airllm_none"`, `"8bit" → "airllm_8bit"`,
  `"4bit" → "airllm_4bit"`.

**Outputs (the ledger — single source of truth)**
- `results/airllm_8bit.json`, `results/airllm_4bit.json` — one entry per level, each containing:
  `scenario`, `compression` label, `ttft_s`, `tpot_s`, `throughput_tok_s`, `peak_vram_gb`,
  `peak_ram_gb`, `total_runtime_s`, `est_power_wh`, `tokens_generated`, `output_text`, and a
  `quality_note` (short prose: does the answer degrade? where vs the red line?).
- (The `None`/FP16 level is recorded by the AirLLM mechanism as `results/airllm_none.json`; the
  sweep **reuses** it for the trade-off comparison rather than re-running it.)
- All writes go through `shared/gatekeeper.py` (the measurement ledger), never `json.dump` directly.

## 4. Module design (files ≤150 lines)

We **reuse the AirLLM run mechanism** and only add the sweep parametrisation on top, so the
quantization-specific surface stays small and well under the 150-line cap.

```
src/cosmos77_ex05/runners/
├── airllm_run.py     # (airllm mechanism) one run for a given compression; builds model via factory,
│                     #   calls the measure harness, returns a metrics dict + output text. ≤150
└── quant_run.py      # (THIS mechanism, ≤120) parametrise airllm_run over the compression levels:
                      #   for level in (None,"8bit","4bit"): smoke-test → real run → ledger entry +
                      #   quality note. No GEMM/GEMV here — it orchestrates and labels.
src/cosmos77_ex05/measure/        # TTFT/TPOT/throughput/peak-mem harness (measurement mechanism)
src/cosmos77_ex05/shared/gatekeeper.py   # ledger writer → results/*.json
src/cosmos77_ex05/sdk/sdk.py             # single class SDK entry; exposes run_quant_sweep()
```

**`quant_run.py` responsibilities (≤120 lines)**
- `compression_label(level) -> str` — map `None/"8bit"/"4bit"` to the ledger scenario label.
- `run_one_level(level, *, model_factory, cfg, ledger) -> dict` — (1) **smoke-test** with a tiny
  `max_new_tokens` (e.g. 2–4) to "prove the plumbing", (2) the **real run** at the configured
  `max_new_tokens`, (3) attach a `quality_note`, (4) write one ledger entry. Delegates the actual
  build+generate+measure to `airllm_run` so there is **no duplication** (rule 3).
- `run_quant_sweep_levels(*, model_factory, cfg, ledger) -> list[dict]` — iterate the levels in
  order, return the list of entries (the SDK wraps this).
- A `quality_note` helper: a deterministic, config-driven heuristic stub (e.g. flags empty /
  truncated / repetitive output) that the operator can override with the **human-observed** note
  from the notebook — the honest qualitative read is the human's, the stub just seeds it.

**Why split this way:** `airllm_run` owns *one* run (build → measure → output); `quant_run` owns
*the sweep* (smoke → real → label → record, ×3). Each stays single-purpose and ≤150 lines, and the
compression knob is the only new concept this file introduces.

## 5. Public SDK API

One `class SDK` (`src/cosmos77_ex05/sdk/sdk.py`) is the only entry point; the notebook and CLI call
the SDK, never the runners directly (rule 2).

```python
class SDK:
    def run_quant_sweep(
        self,
        *,
        model_factory: Callable[..., Any] | None = None,
        levels: Sequence[str | None] = (None, "8bit", "4bit"),
        smoke_first: bool = True,
    ) -> list[dict]:
        """Run the quantization sweep: for each compression level, optionally smoke-test
        with a tiny max_new_tokens, then the real run, recording ONE ledger entry per level
        with the correct compression label, the full metric set, and a quality note.

        Reuses the AirLLM run mechanism with the compression argument. `model_factory` is
        injected so tests mock it (no real GPU / no model download). Returns the ledger
        entries in run order. In production `model_factory` defaults to the AirLLM
        `AutoModel.from_pretrained(..., compression=level)` wrapper.
        """
```

Defaults pull `model_id`, `prompt`, `max_new_tokens`, paths from `config/setup.json`. The method
is **config-driven** and **idempotent** w.r.t. the ledger (re-running overwrites the level's entry).

## 6. Test plan (all heavy I/O mocked — no GPU, no download)

Tests live in `tests/unit/runners/test_quant_run.py` and `tests/unit/sdk/`. The **model factory is
mocked** (`pytest-mock`), so `.generate(...)` returns canned token ids/text and timing is injected
deterministically. Seed `random`; fix the prompt; no flakes (rule 17).

- **Happy path — one entry per level.** `run_quant_sweep` over `(None,"8bit","4bit")` produces
  **exactly 3** ledger entries (or 2 from the sweep + reused `airllm_none`), in order, each with the
  **correct `compression` label** (`airllm_none` / `airllm_8bit` / `airllm_4bit`).
- **Quality note captured.** Each entry carries a non-empty `quality_note`; assert the canned
  "degraded" mock output yields a degraded note and a clean mock yields a clean note.
- **Full metric set present.** Each entry has `ttft_s`, `tpot_s`, `throughput_tok_s`,
  `peak_vram_gb`, `peak_ram_gb`, `total_runtime_s`, `est_power_wh`, `tokens_generated`.
- **Smoke-then-real.** With `smoke_first=True`, the factory's model is invoked **twice** per level
  (tiny `max_new_tokens` then the real count); assert the tiny call uses the smoke token budget.
- **Compression is forwarded.** Assert `model_factory` was called once per level with the matching
  `compression=` kwarg (`None`, `"8bit"`, `"4bit"`).
- **No real GPU / no download (the safety assertion).** Assert neither `airllm.AutoModel` nor
  `torch.cuda` nor any HF download is touched: patch them to raise, and verify the mocked factory
  path means they are **never called**. The suite must pass on a CPU-only CI runner.
- **Ledger writes go through the gatekeeper.** Assert `gatekeeper.record(...)` is called per level
  and that the JSON lands at `results/airllm_{8bit,4bit}.json` (use a tmp results dir).
- **Error path.** A factory that raises (simulated CUDA/OOM at a level) is recorded as a failed
  ledger entry with the error captured, not swallowed silently — the sweep continues to the next
  level (or fails loudly per config), never fabricating numbers.

Coverage target for `quant_run.py` and its SDK glue: **≥85%** (rule 7). The live runs are the
notebook and are out of the coverage gate.

## 7. Acceptance-criteria mapping

| Acceptance | How this mechanism satisfies it | Evidence |
|-----------|----------------------------------|----------|
| **D4** — Quantization sweep (FP16/8bit/4bit): memory, speed, output-quality + the red line | `run_quant_sweep` runs each level, records memory+speed, and a quality note; the red line is stated from the captured notes | `results/airllm_{8bit,4bit}.json`, `reports/quantization.md` |
| **D5** — Systematic metrics per scenario | Each ledger entry carries the full metric set (TTFT/TPOT/throughput/peak RAM+VRAM/runtime/power/quality) | `results/airllm_{8bit,4bit}.json` |
| **D8** — Lecture-concept linking | Trade-off table ties bytes/param → memory, Q4 aggressive, NF4/QLoRA, the red line | this PRD §2, `reports/CONCEPTS.md` |
| **D15** — Honesty | Quality notes are the human-observed read; a degraded Q4/over-the-line result is recorded, never faked | ledger ↔ README byte-match |
| RQ-c (PRD §3) | "quantization's effect on memory/speed/quality + the accuracy red line" answered with measured numbers | README, `reports/quantization.md` |

## 8. Risks & mitigations

- **bitsandbytes is CUDA-only.** Mitigation: the sweep runs only on the T4; the **model factory is
  injected and mocked** in tests so the deterministic library never imports a GPU path.
- **Q8 may not fit / may be slow on a 16 GB T4 (~14.7 GB weights + overhead).** Mitigation: run
  every level **through AirLLM paging** (not resident), and **smoke-test first** with a tiny
  `max_new_tokens` to "prove the plumbing" before committing to the full run.
- **Disk overflow from per-layer shards across three runs.** Mitigation: reuse one
  `layer_shards_saving_path` with `delete_original=True`; document residual disk in the report.
- **The red line is subjective.** Mitigation: fix the prompt and `max_new_tokens`, record the raw
  `output_text` alongside the `quality_note`, and let the reader re-judge from the committed text.
- **AirLLM maintenance-mode / API drift on `compression`.** Mitigation: pin a supported model
  (Qwen2.5-14B) and isolate the call behind the factory seam so a signature change touches one file.
- **Temptation to "fix" a degraded answer.** Out of scope (PRD §7): the degradation is the finding.
