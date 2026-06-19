# PRD — FP16 Baseline (the control condition that MUST fail)

**Mechanism:** `runners/baseline.py` + `SDK.run_baseline()`
**Maps to:** D2 (baseline direct run + bottleneck identified). Playbook §1.5, §6 (Phase 4).
**Version:** 1.00 · **Status:** specified (Phase 4) · **Owner:** runners
**Binding rules:** CLAUDE.md 1–17 (150-line cap, single `class SDK` entry, config-driven, mocked TDD ≥85%, ledger-as-truth).

---

## 1. Purpose

This mechanism is the **control condition** of the experiment. Before we can claim AirLLM + quantization
"made the impossible possible," we must first demonstrate, on the real hardware, that the **naive path
fails** — and fails for a *specific, named* reason. We load `Qwen/Qwen2.5-14B-Instruct` the textbook way
(`AutoModelForCausalLM.from_pretrained(..., torch_dtype=float16, device_map="cuda")`) on a free 16 GB
Kaggle T4 and attempt `.generate(...)`. The expected outcome is `torch.cuda.OutOfMemoryError`.

The deliverable is **not** a working model — it is the *captured, explained failure*. We record the OOM
error, the **requested-vs-available VRAM** (≈29.4 GB model weights vs 16 GB T4), and the param→memory math
to `results/fp16_baseline.json` with `success=False`. We then name the bottleneck:

> **MEMORY (VRAM capacity), not compute.** The T4 never gets to run a GEMM — it cannot even *place* the
> weights in HBM. The wall is VRAM capacity (the lecture's VRAM concept), not FLOPs. The model is too big
> to *fit*, independent of how fast the GPU is.

This honest negative result is the "before" half of the before/after story the grade rewards (playbook §2:
*a clean negative result, analyzed, beats a faked positive*).

## 2. Inputs / Outputs

**Inputs (all config-driven — CLAUDE.md rule 4, no hardcoding):**
- `config/setup.json` → `experiment.model_id` (`Qwen/Qwen2.5-14B-Instruct`), `experiment.prompt`,
  `experiment.max_new_tokens` (20), `experiment.platform` (`kaggle_t4`).
- `results/hardware.json` (from Phase 2 `hardware/spec.py`) → measured T4 VRAM (`16 GB`) and model params
  (`14.7e9`). If absent, fall back to the configured T4 assumption; the ledger records which source was used.
- `BYTES_PER_PARAM` (constants.py) → `{fp16: 2.0, q8: 1.0, q4: 0.5}` for the param→memory math.
- `HF_TOKEN` from `.env` / Kaggle secrets — never in code (rule 9).

**Outputs:**
- `results/fp16_baseline.json` (the ledger entry, via `shared/gatekeeper.py`): `success=False`, the OOM
  message, `requested_vram_gb` (29.4), `available_vram_gb` (16), `deficit_gb` (13.4), `params`, `dtype`,
  `bytes_per_param`, `bottleneck="memory"`, and an optional CPU-control sub-record.
- A notebook cell render (Phase 3/4) that prints the captured OOM traceback + the memory math — the
  rendered OOM is the "before" screenshot embedded in the README (D11).
- `reports/baseline.md` narration (separate task) consumes this ledger entry.

## 3. Module design (files ≤150 lines — rule 1)

`baseline.py` stays ≤130 lines by keeping the **math** and the **ledger** out of it (reuse, rule 3):

| File | Resp. | LoC | Notes |
|---|---|---|---|
| `runners/baseline.py` | Orchestrate load→generate, catch OOM, build the result dict, hand it to the ledger | ≤130 | The only file that *touches* transformers/torch. Loader is **injected**. |
| `runners/_loaders.py` (optional) | Default real loader: `AutoModelForCausalLM.from_pretrained(...)` + tokenizer | ≤60 | Isolated so `baseline.py` has zero import-time `transformers` dependency; tests never import it. |
| `hardware/model_math.py` (Phase 2) | `model_memory(params, dtype)`, `justify(model_id, vram_gb)` → 29.4 GB verdict | ≤80 | Reused, **not** duplicated in the runner. |
| `shared/gatekeeper.py` (ported) | `record(scenario, metrics)` → appends `results/<scenario>.json`; `ledger()` aggregates | ≤150 | Single source of truth (rule 13). |

**Dependency injection is the load-bearing design choice.** `baseline.py` receives the model loader as a
parameter (default = the real `transformers` loader from `_loaders.py`, resolved lazily *inside* the
function so importing the module never imports torch). Unit tests pass a **fake loader that raises
`torch.cuda.OutOfMemoryError`** — so the suite catches the exact production code path without a GPU or a
download (rule 6). This is the seam that makes the "must fail" condition deterministically testable.

## 4. Public SDK API (single entry — rule 2)

```python
# src/cosmos77_ex05/sdk/sdk.py
class SDK:
    def run_baseline(self) -> dict:
        """Run the naive FP16 direct load on the configured GPU, capture the OOM,
        and record the `fp16_baseline` ledger entry. Returns the recorded dict
        (success=False on the T4). Why: this is D2's control condition — proof the
        naive path is VRAM-bound before AirLLM is introduced."""
```

```python
# src/cosmos77_ex05/runners/baseline.py
def run_fp16_baseline(
    config: Config,
    ledger: Gatekeeper,
    loader: ModelLoader | None = None,   # injectable; None -> real transformers loader
    *,
    include_cpu_control: bool = False,
) -> dict:
    """Attempt the naive FP16 cuda load+generate; on torch.cuda.OutOfMemoryError
    record success=False + the requested/available VRAM + the param->memory math.
    Why injectable loader: tests pass a fake that raises OOM (no GPU/download)."""

class ModelLoader(Protocol):
    def __call__(self, model_id: str, *, dtype: str, device_map: str) -> ModelBundle: ...
```

`SDK.run_baseline()` builds the `Config` + `Gatekeeper`, calls `run_fp16_baseline()` with the **real**
loader, and returns the ledger entry. The CLI stage `baseline` (constants `PIPELINE_STAGES`) dispatches to it.

**Optional CPU control** (`include_cpu_control=True`, `device_map="cpu"`): loads on CPU to show "loads but
unbearably slow," timing ~2 tokens to record `cpu_tok_per_s`. This reinforces *memory, not compute* — CPU
has the RAM to *fit* the model but is far too slow to be usable, so neither path is viable for the wrong
reasons. It is OFF by default (slow, T4-only) and fully mocked in tests.

## 5. Test plan (mocked OOM — rules 6, 17)

All transformers/torch/GPU/HF/network I/O is **mocked**; the suite NEVER downloads a model or calls a GPU.
`torch` is mocked so `torch.cuda.OutOfMemoryError` is a real raisable exception class.

| # | Test | Mock setup | Assertion |
|---|---|---|---|
| T1 | OOM is caught, not propagated | `loader` fake raises `torch.cuda.OutOfMemoryError("CUDA out of memory...")` | `run_fp16_baseline` returns a dict; no exception escapes. |
| T2 | `success=False` recorded | as T1 | returned dict + `results/fp16_baseline.json` have `success is False`. |
| T3 | Correct memory math | as T1, params=14.7e9, vram=16 | `requested_vram_gb ≈ 29.4`, `available_vram_gb == 16`, `deficit_gb ≈ 13.4`, `bytes_per_param == 2.0`. |
| T4 | Bottleneck labelled memory | as T1 | `bottleneck == "memory"`; OOM message substring stored verbatim. |
| T5 | Ledger has the entry | as T1, temp `results/` (tmp_path) | `ledger()["fp16_baseline"]["success"] is False`. |
| T6 | **No real I/O happens** | patch `transformers`/`huggingface_hub`/`torch.cuda` call-counters | assert the real `from_pretrained` / download / `.to("cuda")` are **never called** (the fake loader is the only thing invoked). |
| T7 | CPU control path | fake CPU loader returns a stub generating 2 tokens with canned timings | `cpu_tok_per_s` computed; `success` still `False` for the GPU verdict; no real torch. |
| T8 | Deterministic | seeded conftest; fixed prompt | identical dict across runs (rule 17). |

Coverage target ≥85% on `baseline.py` (rule 7); the injectable seam means every branch (OOM, CPU control,
ledger write) is reachable without hardware.

## 6. Acceptance-criteria mapping

| Criterion | How this mechanism satisfies it |
|---|---|
| **D2** baseline direct run (OOM) + bottleneck identified | `run_fp16_baseline` performs the naive FP16 cuda load; captures `torch.cuda.OutOfMemoryError`; records `success=False` + requested/available VRAM + the math; names bottleneck = **memory (VRAM capacity), not compute**. |
| D1 (supports) | Reuses `hardware/model_math.justify()` (29.4 GB > 16 GB verdict); references `results/hardware.json`. |
| D8 (concept link) | Ledger fields feed `reports/CONCEPTS.md`: this is the VRAM-capacity wall — the model cannot be *placed* in HBM, so no GEMM/decode ever runs. |
| D11 | The notebook cell render of the OOM + math is the "before" screenshot embedded in the README. |
| D13 | `HF_TOKEN` via `.env`/Kaggle secrets only; loader is injectable so CI needs no token/GPU. |
| D15 (honesty) | `success=False` is recorded truthfully; no token/sec or VRAM number is fabricated — every number flows through the ledger (rule 13). |

## 7. Risks

- **R1 — The T4 does NOT OOM (silent success).** If HF ships an unexpected smaller/sharded variant or
  `device_map="cuda"` auto-offloads, FP16 might partially load. *Mitigation:* force `device_map="cuda"`
  (no `"auto"`), assert `requested > available` from the math, and if it *does* load, record `success=True`
  with the measured peak VRAM and pivot the narrative to "unbearably slow" — still an honest D2 result.
- **R2 — OOM raised at a different layer than expected.** The error may surface during weight load or at
  the first forward. *Mitigation:* wrap the **whole** load+generate in one `try/except`; store the verbatim
  message and (if available) the layer index; the analysis does not depend on *where* it fails, only *why*.
- **R3 — Test gives false confidence by mocking too much.** If we mock `from_pretrained` to "just raise,"
  we never exercise our own try/except wiring. *Mitigation:* T6 asserts the real loader is **never** called
  *and* the fake is, so the mocked path is exactly the production path minus the GPU.
- **R4 — CPU control blows the time/disk budget on Kaggle.** A 14.7B CPU load needs ~30 GB RAM and is slow.
  *Mitigation:* OFF by default; `max_new_tokens=2`; documented as optional; never runs in the test suite.
- **R5 — Line-cap pressure (rule 1).** Math + ledger + real loader in one file would exceed 130 lines.
  *Mitigation:* `model_math.py` and `gatekeeper.py` are reused; the real loader lives in `_loaders.py`.
- **R6 — `torch` not importable in CI.** *Mitigation:* lazy import inside the default-loader factory; the
  module imports cleanly with torch absent, and tests inject a fake — the suite stays GPU/torch-free.
