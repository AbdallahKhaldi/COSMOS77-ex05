# PRD — Mechanism: AirLLM runs the SAME model (the "layer = page" treatment)

> Course: **Orchestration of AI Agents (203.3763)** — Dr. Yoram Segal (UOH).
> HW5 — *Running a Massive LLM Locally: AirLLM, Quantization & Performance Benchmarking.*
> Per-mechanism PRD. Maps to **D3** (the treatment condition). Phase 5. Version 1.00.
> Companion docs: `docs/PRD.md` (project-wide), `docs/PRD_baseline.md` (D2, the FP16 OOM control).

## 1. Purpose

The FP16 baseline (D2) proves the negative: `Qwen/Qwen2.5-14B-Instruct` (~14.7 B params, **~29 GB
in FP16**) does **not** fit a free **16 GB Kaggle T4** — a clean `torch.cuda.OutOfMemoryError`. This
mechanism is the **treatment**: the *same* model, on the *same* T4, on the *same* prompt, **now
runs** via **AirLLM**. AirLLM keeps only the **active transformer layer** resident in VRAM and
**streams** the rest from disk as per-layer **SafeTensors** shards (`mmap`, zero-copy), computes,
then frees the layer. This is the operating-system **virtual-memory / Paging** mechanism applied to
a neural network: **a layer = a page**, loading a layer = a **page fault**, evict-after-use = page
replacement. It buys **feasibility** (29 GB "fits" in 16 GB) at the cost of **latency** (~1–3 tok/s),
because **disk bandwidth (~3–7 GB/s) ≪ HBM bandwidth (~320 GB/s)**. We measure that cost honestly and
write it to the ledger. **The grade is the analysis, not the model:** a slow-but-correct run with a
correct causal explanation is the deliverable.

## 2. Inputs / Outputs

**Inputs (all config-driven — no hardcoding, rule 4):**
- `config/setup.json → experiment`: `model_id` (`Qwen/Qwen2.5-14B-Instruct`), `prompt`
  (`"Explain the attention mechanism in transformers."`), `max_new_tokens` (`20`),
  `layer_shards_saving_path` (`/kaggle/temp/shards`), `max_seq_len` (`128`).
- `HF_TOKEN` from `.env` / Kaggle secrets (never in code, rule 9) — for the model download.
- Runtime: free Kaggle T4 (the *experiment env*, its own `pip`; `uv` governs our package only, rule 5).

**Outputs:**
- `results/airllm_none.json` — one ledger entry (via `shared/gatekeeper.py`, rule 13) with
  `success=True`, the full metric set (TTFT, TPOT, throughput, peak RAM + peak VRAM, total runtime,
  est. power, `n_out`), the decoded answer text, and the raw per-layer profiling timings.
- The decoded model output (the same prompt, now answered).
- `reports/airllm.md` — the narrative: it now runs; per-layer load/compute timing; the Paging
  connection; the honest latency cost.

## 3. Module design (every file ≤ 150 lines, rule 1)

| File | ≤ lines | Responsibility |
|------|--------|----------------|
| `src/cosmos77_ex05/runners/airllm_run.py` | **130** | Build the AirLLM `AutoModel`, tokenize the configured prompt, `generate`, return decoded output + raw per-layer timings. **Factory + generate injectable** for tests. |
| `src/cosmos77_ex05/runners/_base.py` | 90 | Shared runner mixin (config load, prompt/encode helpers, ledger write). Reused by `airllm_run` + the Phase-6 `quant_run` (rule 3: ≥2 users ⇒ shared module / base class). |
| `src/cosmos77_ex05/measure/harness.py` | 150 | `measure(generate_fn, ...)` — the Phase-7 harness. Imported, not duplicated. |
| `src/cosmos77_ex05/measure/timing.py` | 120 | TTFT / TPOT / throughput / est-power arithmetic split out so `harness.py` stays under cap. |
| `src/cosmos77_ex05/shared/gatekeeper.py` | 140 | The ledger writer — every number lands here → `results/*.json` (rule 13). |
| `src/cosmos77_ex05/sdk/sdk.py` | 150 | `class SDK` single entry; exposes `run_airllm()` (rule 2). |

**Split rationale.** `airllm_run.py` does **only** orchestration (resolve config → build model →
tokenize → generate → decode → return). It does **not** compute metrics (that is `measure/`) and does
**not** write JSON (that is `gatekeeper`). This keeps it small, single-purpose, and unit-testable with
the model fully mocked. The `_base.py` mixin exists because `quant_run.py` (Phase 6, D4) is
`airllm_run` parametrised by `compression ∈ {None, "8bit", "4bit"}` — DRY (rule 3).

### 3.1 `airllm_run.py` contract (the core file)

```python
from airllm import AutoModel  # imported lazily / injectable for tests

def run_airllm(
    config: ExperimentConfig,
    *,
    model_factory: Callable[..., Any] = AutoModel.from_pretrained,
    measure_fn: Callable[..., MetricRecord] = measure,
    ledger: Gatekeeper | None = None,
) -> AirllmResult: ...
```

- Builds the model via the **injected factory** (default `AutoModel.from_pretrained`):
  `model_factory(model_id, layer_shards_saving_path=cfg.layer_shards_saving_path,
  delete_original=False, profiling_mode=True)`.
  - `layer_shards_saving_path` — where AirLLM writes the per-layer SafeTensors shards (the "swap file").
  - `delete_original=False` — KEEP the monolithic HF download so each scenario (FP16/8bit/4bit)
    can re-shard from it without re-downloading; the notebook instead **clears the shards dir
    between scenarios** so the per-level shard sets never accumulate on Kaggle's shared overlay disk.
  - `profiling_mode=True` — AirLLM emits the **per-layer load + compute timings** we report as the
    direct evidence of paging.
- **Tokenize** the configured prompt with the AirLLM model's tokenizer (`max_seq_len` from config).
- **CUDA placement (the gotcha):** AirLLM keeps weights paged but expects **input ids on GPU**. We
  guard `input_ids = input_ids.cuda()` behind a `torch.cuda.is_available()` check so the *same* code
  path stays import-safe and mock-safe on a CPU/CI box.
- `generate(input_ids, max_new_tokens=cfg.max_new_tokens, use_cache=True, return_dict_in_generate=True)`
  is wrapped by the **injected `measure_fn`** so TTFT/TPOT/peak-VRAM are captured around the real call.
- **Returns** `AirllmResult(text=<decoded>, per_layer_timings=<raw>, metrics=<MetricRecord>)` and
  writes the ledger entry `results/airllm_none.json` via the gatekeeper with `success=True`.

## 4. The "layer = page" demonstration (D3 / RQ-b)

This is the intellectual payload of the mechanism. The mapping is exact and must appear in
`reports/airllm.md` with the measured per-layer timings as evidence:

| OS virtual memory (Lecture 08) | AirLLM transformer inference |
|--------------------------------|------------------------------|
| Process address space ≫ physical RAM | Model weights (29 GB) ≫ VRAM (16 GB) |
| **Page** (fixed-size chunk) | **One transformer layer** (per-layer SafeTensors shard) |
| **Page fault** → load page from disk | Load the next layer's shard from `layer_shards_saving_path` |
| **`mmap` zero-copy** mapping | SafeTensors `mmap` — map the shard, no extra copy |
| **Evict / page replacement** | Free the layer's VRAM after its forward pass |
| Resident set ≈ working set | Resident VRAM ≈ **one layer**, not the whole model |
| Thrashing when working set ≫ RAM | Every decode step re-pages **all** layers ⇒ slow |

**Why it now fits.** Peak VRAM is bounded by the **largest single layer** plus activations + KV-cache,
**not** by 29 GB. The whole model is never resident; that is precisely how an OS runs a program bigger
than RAM.

**Why it is slow — stated honestly (RQ-e).** Decode is **memory-bound** (GEMV re-reading every weight
each step). With AirLLM every weight read for every token is a **page fault to disk**, so per-token
latency is governed by **disk BW ~3–7 GB/s**, not **HBM BW ~320 GB/s** — roughly **50–100× slower**,
yielding the expected **~1–3 tok/s**. Both **TTFT** (prefill pages every layer once) and **TPOT**
(decode pages every layer *per token*) inflate. This is paging "thrashing," and it is the correct,
expected, gradable result — not a defect. AirLLM trades **VRAM for time**.

## 5. Public SDK API (rule 2 — single entry)

```python
class SDK:
    def run_airllm(self, *, write_ledger: bool = True) -> AirllmResult:
        """Run the SAME model under AirLLM (D3, the treatment).

        Resolves config/setup.json, invokes runners.airllm_run.run_airllm wrapped in the
        Phase-7 measurement harness, and (default) records results/airllm_none.json with
        success=True. The notebook and CLI call ONLY this — no business logic outside the SDK.
        """
```

- CLI: `uv run cosmos77-airllm airllm` → `SDK().run_airllm()` (stage `"airllm"` in `PIPELINE_STAGES`).
- Returns the `AirllmResult` (decoded text + per-layer timings + metrics) for notebook display.

## 6. Test plan (TDD, everything mocked — rules 6, 17)

**The suite NEVER downloads a model and NEVER needs a GPU.** AirLLM's `AutoModel.from_pretrained`
and the model's `generate` are **injected as mocks** returning **canned tokens + canned timings**.

| Test | Asserts |
|------|---------|
| `test_run_airllm_records_ledger_entry` | With mocked factory + generate, `run_airllm` writes one `results/airllm_none.json` entry with **`success=True`** and **all expected fields** (`ttft_s`, `tpot_ms`, `throughput_tok_s`, `peak_ram_gb`, `peak_vram_gb`, `total_s`, `est_power_wh`, `n_out`, `text`, `per_layer_timings`). |
| `test_no_real_download_or_gpu` | The **injected factory mock is the only model constructor called** — assert no `airllm.AutoModel.from_pretrained` real import-time download, no `torch.cuda` allocation; `.cuda()` is skipped/stubbed when `is_available()` is patched `False`. |
| `test_factory_called_with_config_args` | Factory invoked with `layer_shards_saving_path`, `delete_original=False`, `profiling_mode=True` **from config**, not literals (rule 4). |
| `test_generate_uses_config_max_new_tokens` | `generate(..., max_new_tokens=20)` driven by `config.max_new_tokens`. |
| `test_decoded_text_returned` | Canned token ids decode to the expected string via a mocked tokenizer. |
| `test_per_layer_timings_passthrough` | Canned `profiling_mode` timings are returned raw and stored in the ledger entry. |
| `test_metrics_from_canned_timings` | Harness, fed canned timings, computes the expected TTFT/TPOT/throughput (deterministic arithmetic, no I/O). |
| `test_cuda_placement_guarded` | With `torch.cuda.is_available()` patched `False`, the code path runs without calling `.cuda()` (CI-safe). |

**Fixtures:** a `FakeAutoModel` (canned `generate` returning fixed ids + a `profiling` attribute of
canned per-layer ms), a `FakeTokenizer` (deterministic encode/decode), patched `torch`/`psutil`
(rule 17: seeded, fixed prompt, no flakes). Coverage on this file + `measure/` + `gatekeeper` ≥ 85%
(rule 7). `ruff check` zero (rule 8).

## 7. Acceptance-criteria mapping

| Criterion | Evidence this mechanism produces |
|-----------|----------------------------------|
| **D3** — AirLLM runs the SAME model on the SAME prompt | `results/airllm_none.json` with `success=True` + decoded output |
| K-3 (DoD) | The ledger entry exists and the README cites it byte-exact |
| D5 — full metric set per scenario | TTFT, TPOT, throughput, peak RAM+VRAM, runtime, est. power, `n_out` in the entry |
| D8 / RQ-b — Paging connection explicit | §4 mapping table + per-layer timings narrated in `reports/airllm.md` |
| RQ-e — throughput/latency price stated honestly | measured ~1–3 tok/s with the disk-BW-vs-HBM-BW causal explanation |
| D13 / NFR-6 — tests touch no real I/O | mocked factory + generate; assert no download/GPU |
| Rules 1, 2, 4 | file ≤ 130 lines; only `SDK.run_airllm()` entry; all params from config |

## 8. Risks

- **AirLLM API drift** — `from_pretrained` kwargs (`compression`, `profiling_mode`, `layer_shards_saving_path`)
  vary across versions. *Mitigation:* the **injectable factory** isolates our code from the signature;
  pin the AirLLM version in the experiment env's documented requirements; tests assert our call shape, not AirLLM internals.
- **`.cuda()` placement error** — AirLLM's most common runtime failure is input ids left on CPU.
  *Mitigation:* the guarded `.cuda()` is the dedicated fix; `test_cuda_placement_guarded` covers it.
- **Disk pressure on T4** — `/kaggle/temp` + `/kaggle/working` are ONE shared ~58–73 GB overlay; a 14B
  download (~29 GB) + FP16 shards (~29 GB) peaks near ~58 GB. *Mitigation:* shards on
  `/kaggle/temp/shards`, the download kept once (`delete_original=False`), shards **cleared between
  scenarios**, only small JSONs saved to `/kaggle/working`; Cell 1 warns < 55 GB free and the 7B
  fallback is documented in `experiments/SETUP.md` (honesty, D15).
- **Run-time too long** — at ~1–3 tok/s, `max_new_tokens` is intentionally small (`20`); smoke-test
  with `max_new_tokens=1` first. The slowness **is the result**, not a failure (RQ-e, rule: negatives are valid).
- **Mock divergence from reality** — canned timings could mask a real-run break. *Mitigation:* the
  notebook run on the T4 is the ground truth; the ledger is the single source (rule 13); the README
  numbers must byte-match `results/airllm_none.json` (honesty gate, D15).
- **Non-determinism in `generate`** — *Mitigation:* greedy decode (no sampling) in the experiment;
  tests mock `generate` entirely, so unit determinism is guaranteed (rule 17).
