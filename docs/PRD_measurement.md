# PRD — Measurement Harness (`src/cosmos77_ex05/measure/`)

> Course: **Orchestration of AI Agents (203.3763)** — Dr. Yoram Segal (UOH).
> HW5 — *Running a Massive LLM Locally: AirLLM, Quantization & Performance Benchmarking.*
> Mechanism: **the systematic measurement harness** — maps to acceptance criteria **D5** (systematic
> metrics per scenario) and **D6** (tables + graphs feed off these numbers). Version 1.00.

## 1. Purpose

This is the **heart of the grade**: the one component every runner calls to turn a model run into
honest, lecture-anchored numbers. The naive FP16 load OOMs a 16 GB T4, so we run `Qwen/Qwen2.5-14B-
Instruct` via AirLLM (layer = page) across a quant sweep (FP16 / Q8 / Q4) and **measure every
scenario identically**. The harness wraps any runner's `generate` function and emits one canonical
metric dict per run: `{ttft_s, tpot_ms, throughput_tok_s, peak_ram_gb, peak_vram_gb, total_s,
est_power_wh, n_out}`. Every dict is appended to the **measurement ledger** (`results/<scenario>.json`
via `shared/gatekeeper.py`) — the single source of truth. **Nothing is "true" unless it is in the
ledger** (CLAUDE.md rule 13; NFR-1; D15). A well-explained negative (an OOM, a 1–3 tok/s decode) is a
valid, gradable outcome — the harness records it, it never fabricates it.

The harness is **deterministic and GPU-free under test**: `torch.cuda.max_memory_allocated`, `psutil`,
and the `generate_fn` are all mocked, so CI never downloads a model or needs a GPU (NFR-6, rule 6).

## 2. Inputs / Outputs

**Inputs**
- `generate_fn: Callable[[int], GenResult]` — a runner-supplied closure. Called with `max_new_tokens`;
  returns the produced text/token count. The harness times two invocations: one with `max_new_tokens=1`
  (TTFT / Prefill) and one with the full budget `n_out` (end-to-end / Decode). The function is opaque to
  the harness — FP16, AirLLM, Q8, Q4 all share one code path.
- `n_out: int` — target output tokens (from `config.experiment.max_new_tokens`, e.g. 20). Must be ≥ 2 so
  TPOT is defined (`n_out - 1` in the denominator).
- `watts: float` — board power assumption for the est-power estimate (`config.hardware_assumptions.
  gpu_power_watts`, default 70 for a T4).
- `scenario: str` — one of `constants.SCENARIOS` (`fp16_baseline`, `airllm_none`, `airllm_8bit`,
  `airllm_4bit`); selects the ledger file `results/<scenario>.json`.
- Optional: `quality: str | None` (qualitative output note), `success: bool` (False for a captured OOM),
  `extra: dict | None` (e.g. compression label, prompt hash) — passed through to the ledger entry.

**Outputs**
- The **metric dict** (return value), used inline by the runner and by `analysis/{tables,plots,roofline}`.
- A **ledger entry** appended to `results/<scenario>.json` through the gatekeeper (idempotent schema,
  versioned, timestamped). This file — never a hand-edited number — drives `reports/METRICS.md`,
  `figures/*.png`, and the README (D6, K-5, K-6, K-15).

## 3. Module design (every file ≤ 150 lines — NFR-4 / rule 1)

Split so no file approaches the cap and so the GPU-dependent probe is isolated and mockable:

| File | Responsibility | Key symbols | ~LOC |
|------|----------------|-------------|------|
| `measure/harness.py` | Orchestration: time both phases, derive metrics, build the dict, hand it to the ledger. The public seam. | `measure(...) -> dict`, `MetricResult` (TypedDict/dataclass) | ~95 |
| `measure/timing.py` | Low-level probes + pure arithmetic. No ledger, no SDK — trivially unit-testable. | `time_call`, `peak_vram_gb`, `peak_ram_gb`, `derive_ttft_tpot`, `est_power_wh`, `reset_peak_vram` | ~110 |
| `measure/__init__.py` | Re-export `measure` and `MetricResult` only. | — | <10 |
| `shared/gatekeeper.py` | The ledger (ported HW4, repurposed): `record(scenario, metrics)` appends to `results/<scenario>.json`; `ledger()` aggregates across scenarios. Single write path. | `record`, `ledger`, `load_scenario` | ~120 |

**Boundaries.** `harness.py` orchestrates and depends on `timing.py` + `gatekeeper.py`. `timing.py`
owns every external touch (`torch`, `psutil`, `time.perf_counter`) behind named functions so a test
patches **one symbol** per probe. `torch` is an `experiment`-only optional dep (pyproject) — `timing.py`
**imports it lazily** inside the VRAM probe (`try: import torch`), returning `0.0` when CUDA is absent
so the suite and the Mac stay GPU-free. `psutil` is a core dep, always importable. No business logic in
the CLI; all of it reaches users through `SDK` (rule 2).

## 4. Metric definitions (be precise — D5 vocabulary is graded)

Let `t_ttft` be the wall-clock of `generate_fn(max_new_tokens=1)` and `total_s` the wall-clock of the
full `generate_fn(n_out)` run, both via `time.perf_counter()`.

- **TTFT — Time To First Token** (`ttft_s`, seconds). The timed `generate(max_new_tokens=1)` call.
  Corresponds to **Prefill**: process the whole prompt in one pass — a matrix-matrix multiply (**GEMM**),
  **compute-bound**. This is the latency before any streaming begins.
- **TPOT / ITL — Time Per Output Token / Inter-Token Latency** (`tpot_ms`, milliseconds).
  `tpot_ms = (total_s - ttft_s) / (n_out - 1) * 1000`. The steady-state per-token cost during
  **Decode**: each token is a matrix-vector multiply (**GEMV**) re-reading all weights + the KV-cache, so
  it is **memory-bound**. Under AirLLM every token pays a page-fault per layer, inflating TPOT.
- **throughput** (`throughput_tok_s`, tok/s). `n_out / total_s`. The aggregate rate; expected ~1–3 tok/s
  under AirLLM (disk BW ≪ HBM BW). Latency's reciprocal view of the same run.
- **peak VRAM** (`peak_vram_gb`, GB). `torch.cuda.max_memory_allocated() / 1024**3`, sampled after the
  full run, with `reset_peak_memory_stats()` called **before** the run. Returns `0.0` when CUDA is
  unavailable. This is the number that should be "impossible" (~29 GB) for FP16 and single-layer-sized
  under AirLLM (the Paging payoff, RQ-b).
- **peak RAM** (`peak_ram_gb`, GB). `psutil.Process().memory_info().rss / 1024**3` — resident set size of
  the process; AirLLM's `mmap` streaming shows up here.
- **total runtime** (`total_s`, seconds). Wall-clock of the full `n_out` generation.
- **est. power** (`est_power_wh`, watt-hours). `watts * total_s / 3600`. A coarse energy estimate
  (board TDP × time), explicitly an **estimate**, not a meter reading (honesty, D15).
- **n_out** (int). Echoed back for table joins and to make the TPOT/throughput denominators auditable.

Units are fixed by the field names: `_s` seconds, `_ms` milliseconds, `_gb` gigabytes (binary, /1024³),
`_wh` watt-hours, `_tok_s` tokens/second. Quantities are **never re-derived downstream** — tables and
plots read these stored values verbatim (single source of truth).

## 5. Public SDK API

One `class SDK` at `src/cosmos77_ex05/sdk/sdk.py` is the only entry point (rule 2, NFR-9). The harness
surfaces through:

```python
class SDK:
    def measure(
        self,
        generate_fn: Callable[[int], GenResult],
        *,
        scenario: str,
        n_out: int | None = None,        # default: config.experiment.max_new_tokens
        watts: float | None = None,      # default: config.hardware_assumptions.gpu_power_watts
        quality: str | None = None,
        success: bool = True,
        extra: dict | None = None,
    ) -> MetricResult:
        """Wrap a runner's generate fn, compute the canonical metrics, and record
        the entry in results/<scenario>.json. Returns the metric dict it stored."""

    def analyze(self) -> "DataFrame":
        """Load the ledger via gatekeeper.ledger() into the comparison table (D6).
        Reads only stored numbers — never re-measures."""
```

`SDK.measure()` resolves config defaults, calls `measure.harness.measure(...)`, and the harness records
through the gatekeeper. Runners (`runners/airllm_run.py`, `baseline.py`, `quant_run.py`) call
**`SDK.measure(...)`** — they never write `results/*.json` directly. The CLI `measure` subcommand
(already stubbed in `cli/main.py`) dispatches to `SDK` once Phase 7 lands.

## 6. Test plan (fully mocked — NFR-6, rule 17; coverage ≥ 85%, rule 7)

All tests are **GPU-free, network-free, deterministic** (seeded `random` via `conftest.py`; no real
clock dependence — `perf_counter` is patched to canned timestamps). `tests/unit/test_harness.py` and
`tests/unit/test_timing.py`; ledger I/O uses `tmp_path`.

1. **Arithmetic correctness (exact).** Mock `time.perf_counter` to return a scripted sequence so
   `ttft_s = 0.50 s` and `total_s = 10.00 s` with `n_out = 20`. Assert **exactly**:
   `tpot_ms == (10.00 - 0.50) / 19 * 1000 == 500.0`; `throughput_tok_s == 20 / 10.0 == 2.0`;
   `est_power_wh == 70 * 10.0 / 3600` (compare with `pytest.approx`). No floating-point slop on the
   integer-clean cases; `approx` only for irrational divisions.
2. **Memory probes mocked.** Patch `torch.cuda.max_memory_allocated` → `31_138_512_896`
   (≈ 29.0 GB) and `psutil.Process().memory_info().rss` → a fixed value; assert `peak_vram_gb` /
   `peak_ram_gb` equal the bytes / 1024³. Assert `reset_peak_memory_stats` was called **before** the
   timed run (ordering via a mock call-order check).
3. **CUDA-absent path.** Patch the lazy `import torch` to raise `ImportError` (or `cuda.is_available()`
   → False); assert `peak_vram_gb == 0.0` and no exception — the Mac/CI path.
4. **Ledger received the entry.** With a fake `generate_fn` returning canned tokens, assert
   `results/<scenario>.json` (under `tmp_path`) gains exactly one entry whose fields equal the returned
   dict; assert `gatekeeper.record` was called once with `scenario` + the metric dict.
5. **Negative result.** `success=False` (simulated OOM) records an entry with `success=False` and the
   captured context, and **does not fabricate** timing numbers (they're `None`/`0.0`), proving honest
   negatives survive the harness (D15).
6. **`n_out` guard.** `n_out < 2` raises a clear `ValueError` (TPOT undefined) — tested.
7. **Determinism / no flakes.** Run the arithmetic test twice in one session; assert byte-identical
   results. No `sleep`, no real GPU, no wall-clock branch.

Coverage target ≥ 85% on `measure/` and `shared/gatekeeper.py`; `ruff check` zero (selectors E,F,W,I,N,
UP,B,C4,SIM per pyproject).

## 7. Acceptance-criteria mapping

| Criterion | How this mechanism satisfies it |
|-----------|--------------------------------|
| **D5** (systematic metrics per scenario) | `measure()` emits TTFT, TPOT/ITL, throughput, peak RAM+VRAM, total runtime, est. power, quality — one identical path for all 4 scenarios. |
| **D6** (tables + graphs) | Stored ledger numbers feed `analysis/{tables,plots,roofline}` verbatim; no re-measurement. |
| **K-5** (full metric set) | The eight-field dict per `results/*.json` entry. |
| **NFR-1 / rule 13 / D15** (honest ledger) | Sole write path is `gatekeeper.record`; negatives recorded, never faked; README must byte-match the ledger. |
| **NFR-4 / rule 1** (150-line cap) | `harness.py` + `timing.py` split; both well under 150. |
| **NFR-6 / rule 6,17** (GPU-free, deterministic tests) | torch/psutil/`perf_counter`/`generate_fn` all mocked; seeded; no I/O. |
| **rule 2 / NFR-9** (single SDK entry) | Only `SDK.measure()` / `SDK.analyze()` are public; runners + CLI route through them. |

## 8. Risks & mitigations

- **`perf_counter` flakiness.** Real timing in tests would flake. *Mitigation:* patch `perf_counter` to a
  scripted sequence; assert exact arithmetic, never wall-clock thresholds.
- **`torch` import on a GPU-free box.** Importing torch at module top would break the Mac/CI path.
  *Mitigation:* lazy `try/except ImportError` inside the VRAM probe; return `0.0` when CUDA is absent;
  torch stays an `experiment`-only optional dep.
- **`n_out == 1` → divide-by-zero in TPOT.** *Mitigation:* explicit `ValueError` guard, tested.
- **Double counting / non-canonical writes.** A runner writing `results/*.json` directly would corrupt
  the single-source-of-truth. *Mitigation:* gatekeeper is the only writer; runners call `SDK.measure`.
- **Power figure mistaken for a meter reading.** *Mitigation:* named `est_power_wh`, documented as a
  board-TDP × time estimate; the report frames it as an estimate (D15 honesty).
- **Peak-VRAM not reset between scenarios.** Stale `max_memory_allocated` would over-report.
  *Mitigation:* `reset_peak_memory_stats()` before each run; ordering asserted in tests.
- **Free-tier T4 variance.** Live tok/s wobbles run-to-run. *Mitigation:* the ledger records what
  happened; variance is reported honestly, not averaged away into a fabricated "clean" number.
