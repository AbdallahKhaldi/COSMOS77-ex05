# PRD — Analysis Mechanism (tables + graphs + Roofline + concept-linking)

> Course: **Orchestration of AI Agents (203.3763)** — Dr. Yoram Segal (UOH). HW5 — *Running a
> Massive LLM Locally: AirLLM, Quantization & Performance Benchmarking.* Version 1.00.
> Maps to acceptance criteria **D6** (tables + graphs), **D8** (lecture-concept analysis),
> **D9** (Roofline). Phases **7** (build) and **9** (concept narrative). Sibling of `docs/PRD.md`
> (the umbrella PRD); this document is the **single per-mechanism PRD for the ANALYSIS layer**.

## 1. Purpose

The analysis layer turns the committed measurement **ledger** (`results/*.json`) into the graded
artifacts a reader actually sees: a comparison **table** (`reports/METRICS.md`), four metric
**figures**, and the **Roofline** plot that diagnoses *why* the numbers look the way they do. It owns
**no measurement** — it never loads a model, never touches a GPU, never invents a number. Its sole
input is the ledger (NFR-1, the single source of truth); its job is presentation + causal reading.

The mechanism encodes the course's central diagnosis as code: **decode is memory-bound** (arithmetic
intensity ≈ 1, far left of the T4 ridge ≈ 203 FLOPs/byte), and **AirLLM is slow because its effective
bandwidth is DISK (~3–7 GB/s), not HBM (~320 GB/s)** — so the AirLLM point falls *far below* the
memory roofline, not on it. The figures must make that visible; the concept-linking prose
(`reports/CONCEPTS.md`) must say it in lecture vocabulary.

## 2. Inputs / Outputs

**Inputs**
- `results/*.json` — one ledger file per scenario in `SCENARIOS` = `("fp16_baseline",
  "airllm_none", "airllm_8bit", "airllm_4bit")`. Each carries: `scenario`, `ttft_s`, `tpot_s`,
  `throughput_tok_s`, `peak_vram_gb`, `peak_ram_gb`, `total_runtime_s`, `est_power_wh`,
  `quality_note`, and (for Roofline) `model_flops`, `bytes_moved`, `effective_bw_gb_s`, `ok`/error.
  `fp16_baseline` is typically a **negative result** (OOM): metric fields may be `null` and an
  `error`/`oom` flag set — the analysis must render it without crashing (D15: negatives are data).
- `config/setup.json` → `paths` (`results_dir`, `figures_dir`, `reports_dir`) and `experiment`
  (`model_id`, `max_new_tokens`). Hardware roofline constants (T4 peak, HBM BW, disk BW range) live
  in `config/setup.json` under a `roofline` block (added Phase 7), never hardcoded (rule 4).

**Outputs**
- `reports/METRICS.md` — Markdown comparison table, one row per scenario, columns TTFT / TPOT /
  throughput / peak VRAM / peak RAM / runtime / est. power / quality.
- `figures/tokens_per_sec.png`, `figures/peak_vram.png`, `figures/ttft_vs_tpot.png`,
  `figures/quant_tradeoff.png` — metric charts (non-empty PNGs).
- `figures/roofline.png` — the T4 Roofline with each measured point + ridge line.
- `reports/CONCEPTS.md` — concept-linking narrative (D8), generated/assembled from the ledger labels.

## 3. Module design (files ≤ 150 lines; how to split)

Package `src/cosmos77_ex05/analysis/`. Each file is single-responsibility and stays under the
150-line cap (rule 1); shared logic is extracted to avoid duplication (rule 3).

| File | Responsibility | Cap |
|------|----------------|-----|
| `tables.py` | Load `results/*.json` → pandas `DataFrame`; render `reports/METRICS.md` | ≤150 |
| `plots.py` | Four metric figures from the DataFrame (matplotlib **Agg**) → `figures/` | ≤150 |
| `roofline.py` | Arithmetic intensity per scenario; T4 Roofline plot + memory/compute labels | ≤140 |
| `concepts.py` | Assemble `reports/CONCEPTS.md` narrative from ledger + labels (D8) | ≤150 |
| `loader.py` | Shared: read ledger dir → list of dicts → `DataFrame`; handle missing/OOM rows | ≤90 |

**Split rationale.** `loader.py` is shared by `tables`, `plots`, and `roofline` (the rule-of-2:
three consumers of the same load logic ⇒ one module). `plots.py` and `roofline.py` both call a tiny
shared `_save_fig(fig, path)` helper (live in `loader.py` or a 1-function `figio.py` if `plots.py`
nears the cap). `concepts.py` only assembles prose from already-loaded data, so it imports `loader`
too. No file performs I/O against a model or GPU — analysis is pure data → artifacts.

**Matplotlib Agg.** Every plotting module sets the non-interactive backend at import time
(`matplotlib.use("Agg")` before `import matplotlib.pyplot`), so figures render headless on CI and in
tests with no display (NFR-6). Figures are saved with `fig.savefig(path, dpi=...)`; never `show()`.

## 4. The Roofline math

The **Roofline** plots achievable performance (FLOPs/s) against **arithmetic intensity**
`AI = FLOPs / bytes-moved`. The hardware ceiling is the minimum of two limits:

```
attainable_FLOPs_per_s = min( peak_compute,  AI × bandwidth )
```

For the **T4** (the binding constants, read from `config/setup.json` `roofline`):
- `peak_compute` ≈ **65 TFLOPS** (FP16 tensor cores)
- HBM `bandwidth` ≈ **320 GB/s**
- **Ridge point** = `peak_compute / bandwidth` = 65e12 / 320e9 ≈ **203 FLOPs/byte**.

A workload with `AI < 203` is **memory-bound** (left of the ridge, on the sloped roof); `AI > 203` is
**compute-bound** (right of the ridge, on the flat roof). The labeling rule the code enforces:

```
label(scenario) = "memory-bound" if AI(scenario) < ridge_point else "compute-bound"
```

**Per-scenario arithmetic intensity.** Decode generates one token at a time: a matrix-vector
multiply (**GEMV**) re-reading all weights + the KV-cache. FLOPs/byte ≈ **1** (each weight byte feeds
~one multiply-add), so **decode AI ≈ 1 ≪ 203 ⇒ memory-bound** — it sits at the far left of the slope.
Prefill is a matrix-matrix multiply (**GEMM**) with high reuse, so its AI is far higher (compute-bound
regime). The decode point is what we plot per scenario, from `model_flops / bytes_moved` in the
ledger; if absent we fall back to the GEMV estimate `AI ≈ 1`.

**Why AirLLM falls FAR BELOW the roofline.** The roofline ceiling assumes weights stream from **HBM**
(~320 GB/s). AirLLM streams each layer from **disk** (~3–7 GB/s) — a page fault per layer per step
(the "layer = page" Paging analogy). So the *effective* bandwidth is ~50–100× lower than HBM, and the
measured throughput point lands **well under** the memory roof, not on it. We draw the disk-bandwidth
line as a second, much shallower slope (`AI × disk_bw`) and annotate the AirLLM point against it: the
gap between the HBM roof and the AirLLM point is the visual proof of the disk-vs-HBM bottleneck (RQ-a,
RQ-b). The plot therefore carries: the flat compute roof (65 TFLOPS), the HBM memory slope (320 GB/s),
the DISK slope (~3–7 GB/s), the ridge marker at 203, and each scenario's measured decode point.

## 5. Public SDK API

All access is through the single `class SDK` (`src/cosmos77_ex05/sdk/sdk.py`, rule 2); the notebook
and CLI call only the SDK, never the analysis modules directly.

```python
class SDK:
    def analyze(self) -> AnalysisResult:
        """Build all D6 + D9 artifacts from the ledger.

        Reads results/*.json, writes reports/METRICS.md and the five figures
        (tokens_per_sec, peak_vram, ttft_vs_tpot, quant_tradeoff, roofline),
        and returns the in-memory DataFrame + figure/report paths. Pure data →
        artifacts: no model, no GPU. Idempotent; safe to re-run from the ledger.
        """

    def concepts(self) -> Path:
        """Assemble reports/CONCEPTS.md (D8) linking each ledger row to the lecture."""
```

`AnalysisResult` (a small dataclass) exposes: `dataframe: pandas.DataFrame`, `metrics_md: Path`,
`figures: dict[str, Path]`, `roofline_labels: dict[str, str]` (scenario → "memory-bound" /
"compute-bound"). Module-level entry points the SDK calls, each typed + docstringed (rules 15, 16):
`tables.build_dataframe(results_dir) -> DataFrame`, `tables.write_metrics_md(df, out) -> Path`,
`plots.render_all(df, figures_dir) -> dict[str, Path]`, `roofline.compute_intensity(row) -> float`,
`roofline.classify(ai, ridge) -> str`, `roofline.render(df, cfg, out) -> Path`.

## 6. Test plan (fully mocked — no GPU, no model)

TDD red → green → refactor (rule 6); every public function gets a happy + error path before
implementation; ≥85% coverage on this layer (NFR-5); deterministic, seeded, no flakes (NFR-7).

**Fixtures.** `tests/fixtures/results/` holds hand-written ledger JSONs for all four scenarios — a
**fixture ledger**, never a real run: `airllm_4bit.json` with plausible numbers, and a
`fp16_baseline.json` that is an **OOM/negative** row (null metrics + `error`). `conftest.py` exposes a
`fixture_ledger_dir` and a `tmp_outputs` (tmp_path) for figures/reports. Matplotlib **Agg** is forced
in the test session so figures render with no display.

**tables**
- `build_dataframe` returns a DataFrame with **exactly one row per scenario** in `SCENARIOS` order and
  the expected columns (TTFT/TPOT/throughput/peak VRAM/peak RAM/runtime/power/quality).
- The OOM row is present, metric cells are `NaN`/`null`, and a status column flags it (D15).
- `write_metrics_md` writes a `reports/METRICS.md` containing a header row + 4 data rows; idempotent.

**roofline**
- `compute_intensity(decode_row)` ≈ 1 from the GEMV fixture; `classify(1.0, 203)` == `"memory-bound"`
  and `classify(400, 203)` == `"compute-bound"` (ridge boundary asserted).
- The decode point is asserted **left of the ridge** (`AI < ridge_point`); the AirLLM effective-BW is
  asserted **far below** HBM BW (disk-vs-HBM gap present in the rendered data).

**plots / roofline (PNG smoke)**
- `render_all` and `roofline.render` write each target PNG and the file is **non-empty**
  (`path.stat().st_size > 0`) — proves Agg actually rastered, not just touched the file.

**concepts**
- `CONCEPTS.md` mentions each scenario and the key links (TTFT↔Prefill↔GEMM↔compute-bound;
  TPOT↔Decode↔GEMV↔memory-bound; AirLLM↔disk-BW-vs-HBM); assembled purely from the fixture ledger.

No test downloads a model or imports torch/airllm; all inputs are fixture JSON.

## 7. Acceptance-criteria mapping

| Criterion | This mechanism delivers | Evidence |
|-----------|-------------------------|----------|
| **D6** tables + graphs | `tables.py` → `reports/METRICS.md`; `plots.py` → 4 figures | `reports/METRICS.md`, `figures/{tokens_per_sec,peak_vram,ttft_vs_tpot,quant_tradeoff}.png` |
| **D8** concept linking | `concepts.py` → `reports/CONCEPTS.md`; roofline labels | `reports/CONCEPTS.md`, `roofline_labels` |
| **D9** Roofline | `roofline.py` → T4 roofline, ridge ≈203, per-scenario decode points | `figures/roofline.png` |
| K-6 | Tables + graphs regenerated from the ledger (never hand-drawn) | `SDK.analyze()` output |
| K-8 | Every result tied to Prefill/Decode, compute/memory-bound, VRAM, Paging | `reports/CONCEPTS.md` |
| K-9 | Roofline from measurements (65 TFLOPS / 320 GB/s ⇒ ridge ≈203) | `figures/roofline.png` |
| NFR-4 | Each `.py` ≤150 (roofline ≤140); split via `loader.py` | line-cap check |
| NFR-5/6 | ≥85% coverage; all tests mocked, Agg, no GPU | CI green |

## 8. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| A1 | OOM `fp16_baseline` row has null metrics → DataFrame/plot crashes | Medium | High | `loader.py` coerces missing metrics to `NaN`, sets a status flag; plots skip NaN; tests assert the OOM row renders (D15). |
| A2 | Roofline mislabels decode as compute-bound (wrong inequality / unit slip) | Low | High | Single `classify(ai, ridge)` with the boundary unit-tested at AI≈1 and AI≈400; constants from config, not literals. |
| A3 | A `.py` creeps past 150 lines (esp. `plots.py` with 4 charts) | Medium | Medium | Extract `_save_fig`/load into `loader.py`/`figio.py`; one chart-builder per call; line-cap check in CI. |
| A4 | Empty/zero-byte PNG silently "passes" | Low | Medium | Smoke test asserts `st_size > 0` for every figure; Agg backend forced. |
| A5 | Figures drift from ledger if hand-edited | Low | High | Figures only ever produced by `SDK.analyze()` from `results/*.json`; never committed by hand (NFR-1, ADR-004). |
| A6 | AirLLM point drawn *on* the HBM roof, hiding the disk bottleneck | Low | High | Plot the separate DISK slope (~3–7 GB/s) and annotate the AirLLM point against it; test asserts effective-BW ≪ HBM-BW. |
| A7 | Hardcoded T4/BW constants violate rule 4 | Low | Medium | All roofline constants in `config/setup.json` `roofline` block, read via Config. |
