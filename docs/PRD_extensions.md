# PRD — Original Extension(s) (mechanism) · COSMOS77-ex05

> Course: **Orchestration of AI Agents (203.3763)** — Dr. Yoram Segal (UOH).
> HW5 — *Running a Massive LLM Locally: AirLLM, Quantization & Performance Benchmarking.*
> Mechanism: **≥1 original extension** · Maps to acceptance **D10** (drawing on D4/D5/D6/D8).
> Authors: Abdallah Khaldi, Tasneem Natour. Version 1.00.

## 1. Purpose

The baseline assignment benchmarks one model (Qwen2.5-14B-Instruct) across quantization levels on a
free 16 GB Kaggle T4. This mechanism adds the **original initiative** the spec asks for — at least
one extension that goes beyond "run and measure". We specify **three** candidate extensions under
`src/cosmos77_ex05/extensions/` and **ship the primary one**, documenting it in
`reports/EXTENSIONS.md` with its own figure. The three are deliberately ranked by cost-to-deliver:

1. **`extensions/pareto.py` (PRIMARY — no extra GPU run).** A **quant quality-vs-speed (and VRAM)
   Pareto frontier**. It is *pure analysis*: it reads the already-committed `results/*.json` ledger
   (FP16/Q8/Q4) and plots throughput (tok/s) and peak VRAM against a quality score, identifies the
   **Pareto-optimal** point(s), and writes `figures/pareto.png`. Because it needs no GPU, no
   download, and no second experiment, it is fully reproducible from fixtures and is the one we
   guarantee to ship.
2. **`extensions/model_sizes.py` (live, optional).** Run/measure **2–3 model sizes** (Qwen2.5-7B vs
   14B, optionally 32B on Kaggle) and chart the **memory/throughput scaling** — the lecture's
   *model-size scaling* concept made empirical. The runner is mocked in tests; only the notebook
   touches a GPU.
3. **`extensions/qlora.py` (live, the lecture's flagship).** A **QLoRA** demo: an **NF4-quantized
   frozen base + trainable LoRA adapters** via `peft` (`LoraConfig` / `get_peft_model`), printing
   the **trainable-vs-frozen parameter ratio** (`model.print_trainable_parameters()`) and VRAM vs
   plain LoRA — the canonical "fine-tune a huge model on one small GPU" result. The `peft`/model
   calls are injected and mocked in tests; we assert the ratio *math*, never a real train.

The grade is the analysis, not the model. The Pareto frontier turns the committed numbers into a
**decision** ("which precision wins?"); QLoRA and model-size scaling extend the story toward
fine-tuning and scaling. This PRD defines the modules, the SDK surface, and a fully **mocked** test
plan (no GPU, no download, no train) so the deterministic library stays CI-green.

## 2. The concepts this mechanism documents (lecture vocabulary)

- **The quant quality-vs-speed Pareto frontier.** Each precision (FP16, Q8, Q4) is a point in
  (quality, throughput, VRAM) space. A point is **Pareto-optimal** if no other point is better on
  *all* axes simultaneously (higher quality *and* higher tok/s *and* lower VRAM). The frontier is
  the set of non-dominated points; the **Pareto-optimal pick** is the lecture's quality-vs-speed
  trade-off resolved with the project's own data, not asserted.
- **Model-size scaling.** Weights memory ≈ `params × bytes/param`; as params grow (7B→14B→32B),
  VRAM grows ~linearly and throughput typically *falls*. Charting it makes the scaling law concrete.
- **QLoRA / NF4 / LoRA adapters / trainable-vs-frozen.** QLoRA = **NF4** (normal-float 4-bit) frozen
  base weights + low-rank **LoRA adapter** matrices that are the *only* trainable parameters. With
  rank `r`, an adapter on a `d_in × d_out` projection adds `r × (d_in + d_out)` trainable params vs
  `d_in × d_out` frozen — so the **trainable/total ratio** is typically well under 1% (often
  ~0.1–0.5%). That tiny trainable fraction + NF4 base is what lets a 14B model fine-tune on one T4.

## 3. Inputs / Outputs

**Inputs**
- **Pareto:** the committed ledger `results/fp16_baseline.json`, `results/airllm_none.json`,
  `results/airllm_8bit.json`, `results/airllm_4bit.json` (each carries `throughput_tok_s`,
  `peak_vram_gb`, and a quality field — see below); `config/setup.json → paths` for the
  figures/results dirs; an optional quality-score map in config (default: derive a 0–1 score from
  each entry's `quality_note`/`output_text`, or read an explicit `quality_score` if present).
- **model_sizes:** `config/setup.json → experiment.model_id` plus a **model-sizes list** in config
  (e.g. `extensions.model_sizes = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]`); an
  injected **runner** that, in production, wraps the AirLLM run-and-measure path, and in tests is a
  mock returning canned metrics.
- **qlora:** `model_id`; an injected **peft factory** (`get_peft_model`/`LoraConfig`-like) and a
  model-with-named-parameters stub; a `LoraConfig` block in config (`r`, `lora_alpha`,
  `target_modules`, `bias`). In production it wraps `peft`; in tests it is mocked.

**Outputs (the ledger + figures — single source of truth)**
- `figures/pareto.png` — throughput (and a second panel for peak VRAM) vs quality score across
  FP16/Q8/Q4, with the **Pareto frontier** drawn and the optimal point annotated. Non-empty PNG.
- `results/pareto.json` — the computed frontier: per-level `(quality_score, throughput_tok_s,
  peak_vram_gb, dominated: bool)` and the chosen `pareto_optimal` label.
- `results/model_sizes.json` — **one entry per size**: `model_id`, `params_b`, `peak_vram_gb`,
  `throughput_tok_s`, `total_runtime_s`.
- `results/qlora.json` — `trainable_params`, `total_params`, `trainable_ratio`,
  `base_quant` (`"nf4"`), `lora_rank`, and `peak_vram_gb` vs a plain-LoRA reference.
- `figures/model_sizes.png` / `figures/qlora.png` — the live extensions' charts (only for the
  extension actually shipped).
- All `results/*.json` writes go through `shared/gatekeeper.py` (the measurement ledger), never
  `json.dump` directly. All figures land under the configured `figures_dir`.

## 4. Module design (files ≤150 lines)

Each extension is one single-purpose file; the **analysis/plot primitives are reused** from the
existing `analysis/` package (rule 3 — no duplication), so each extension file stays small.

```
src/cosmos77_ex05/extensions/
├── __init__.py
├── pareto.py        # (PRIMARY, ≤140) load ledger → quality score → Pareto frontier → annotate
│                    #   the optimal point → write results/pareto.json + figures/pareto.png.
├── model_sizes.py   # (live, ≤130) iterate configured sizes via an INJECTED runner; one ledger
│                    #   entry per size; chart VRAM + tok/s vs params. Runner mocked in tests.
└── qlora.py         # (live, ≤140) build NF4 base + LoRA adapters via an INJECTED peft factory;
                     #   compute trainable/total ratio; record VRAM vs plain LoRA. peft mocked.
src/cosmos77_ex05/analysis/     # reused plotting + frontier helpers (load_ledger, save_fig, ...)
src/cosmos77_ex05/shared/gatekeeper.py   # ledger writer → results/*.json
src/cosmos77_ex05/sdk/sdk.py             # single class SDK entry; exposes run_extensions()
```

**`pareto.py` responsibilities (≤140 lines)**
- `quality_score(entry) -> float` — deterministic, config-driven 0–1 score: read explicit
  `quality_score` if present, else derive from `quality_note`/`output_text` (e.g. coherent=1.0,
  softer=0.7, degraded=0.3, empty/truncated=0.0). The heuristic is documented and overridable.
- `is_dominated(point, others) -> bool` — a point is dominated iff some other point is ≥ on quality
  **and** ≥ on throughput **and** ≤ on VRAM, with at least one strict. Pure function, trivially
  testable.
- `compute_frontier(entries) -> dict` — return per-level metrics + `dominated` flag + the chosen
  `pareto_optimal` (the non-dominated point that best matches the configured objective, default:
  highest quality among non-dominated).
- `plot_pareto(frontier, *, figures_dir) -> Path` — scatter quality vs throughput (and a VRAM
  panel), draw the frontier line, annotate the optimal point, save `pareto.png`. Delegates the
  actual figure save to an `analysis.save_fig` helper.

**`model_sizes.py` responsibilities (≤130 lines)** — `run_one_size(model_id, *, runner, cfg,
ledger)` (build+measure via the injected runner, one entry) and `run_size_sweep(...)` (iterate the
configured sizes, return the list); a `plot_scaling(entries)` that charts VRAM and tok/s vs
`params_b`. No model code lives here — the runner seam keeps it GPU-free under test.

**`qlora.py` responsibilities (≤140 lines)** — `count_params(model) -> tuple[int,int]`
(trainable, total via `named_parameters().requires_grad`), `trainable_ratio(trainable, total) ->
float`, `build_qlora(model_id, *, peft_factory, lora_cfg)` (NF4 base + `get_peft_model`), and
`run_qlora_demo(...)` that records `trainable_params/total_params/trainable_ratio/peak_vram_gb` to
the ledger. The `peft`/model calls are **only** reached through the injected factory.

**Why split this way:** each extension introduces exactly one new idea (frontier / scaling / QLoRA),
reuses the shared analysis + gatekeeper, and stays single-purpose and ≤150 lines.

## 5. Public SDK API

One `class SDK` (`src/cosmos77_ex05/sdk/sdk.py`) is the only entry point; the notebook and CLI call
the SDK, never the extensions directly (rule 2).

```python
class SDK:
    def run_extensions(
        self,
        *,
        which: Sequence[str] = ("pareto",),
        runner: Callable[..., Any] | None = None,        # for model_sizes (injected/mocked)
        peft_factory: Callable[..., Any] | None = None,  # for qlora (injected/mocked)
    ) -> dict[str, Any]:
        """Run the chosen original extension(s) and return their artefacts.

        - "pareto"      → compute the quant quality-vs-speed/VRAM Pareto frontier from the
                          committed results/*.json, write results/pareto.json + figures/pareto.png,
                          and return the frontier with its `pareto_optimal` pick. No GPU needed.
        - "model_sizes" → run each configured model size via the injected `runner`, record one
                          ledger entry per size, and chart memory/throughput scaling.
        - "qlora"       → build an NF4 base + LoRA adapters via the injected `peft_factory`, record
                          the trainable-vs-frozen parameter ratio and VRAM vs plain LoRA.

        `runner`/`peft_factory` are injected so tests mock them (no real GPU, no download, no train).
        Defaults pull model id, sizes, LoRA config, and paths from config/*.json.
        """
```

Defaults are config-driven; `pareto` is idempotent w.r.t. the ledger and figure (re-running
overwrites). `run_extensions(which=("pareto",))` is the shipped default and needs **no** GPU.

## 6. Test plan (all heavy I/O mocked — no GPU, no download, no train)

Tests live in `tests/unit/extensions/` and `tests/unit/sdk/`. Fixture ledgers (small JSON files in
`tests/fixtures/`) stand in for `results/*.json`; the runner and peft factory are mocked
(`pytest-mock`); `matplotlib` uses the non-interactive `Agg` backend writing to a tmp dir. Seed
`random`; fix the prompt; no flakes (rule 17).

- **Pareto — frontier correctness.** Given a fixture ledger where Q4 has highest tok/s + lowest
  VRAM but lower quality, and FP16 has highest quality but worst VRAM, assert `compute_frontier`
  marks the dominated point(s) correctly and that `is_dominated` is a pure, table-tested function
  (a strictly-worse-on-all-axes point is dominated; a trade-off point is not).
- **Pareto — optimal pick is deterministic.** Assert the chosen `pareto_optimal` label matches the
  configured objective and is reproducible across runs.
- **Pareto — writes a non-empty PNG from a fixture.** Call `plot_pareto` over the fixture and assert
  `figures/pareto.png` exists with size > 0 bytes; assert `results/pareto.json` is written **through
  the gatekeeper** (patched) and contains every level + the optimal pick.
- **Pareto — quality_score heuristic.** Table-test `quality_score`: explicit `quality_score` wins;
  a "degraded"/empty note maps low; a coherent note maps high.
- **qlora — trainable/total ratio math (mocked peft).** Build a stub model whose `named_parameters`
  yields known shapes: a large frozen base + small LoRA tensors with `requires_grad=True`. Assert
  `count_params` returns the exact `(trainable, total)` and `trainable_ratio` equals
  `trainable/total` to full precision (e.g. base 14.7e9 frozen + adapters ⇒ ratio ≪ 1%). Assert the
  injected `peft_factory` (mock `get_peft_model`) is called once with the configured `LoraConfig`
  (`r`, `lora_alpha`, `target_modules`) and `base_quant="nf4"`. **No real train**, no GPU.
- **qlora — VRAM vs plain LoRA recorded.** Assert the ledger entry carries `peak_vram_gb` for QLoRA
  and the plain-LoRA reference, both supplied by the mock (the *comparison*, not a measurement, is
  what we test).
- **model_sizes — one ledger entry per size (mocked runner).** With a mock runner returning canned
  `(params_b, peak_vram_gb, throughput_tok_s)` per `model_id`, assert `run_size_sweep` over a
  2-size config writes **exactly two** ledger entries, in order, each through the gatekeeper.
- **model_sizes — scaling chart written.** Assert `figures/model_sizes.png` is non-empty.
- **Safety assertion (no real GPU / download / train).** Patch `peft`, `torch.cuda`, `airllm`, and
  any HF download to raise; verify the mocked seams mean they are **never called**. The suite must
  pass on a CPU-only CI runner.
- **SDK wiring.** `run_extensions(which=("pareto",))` returns the frontier without touching the
  runner/peft seams; `which=("model_sizes",)` and `("qlora",)` route to their mocked seams.

Coverage target for `extensions/*.py` and their SDK glue: **≥85%** (rule 7). The live runs are the
notebook and are out of the coverage gate.

## 7. Acceptance-criteria mapping

| Acceptance | How this mechanism satisfies it | Evidence |
|-----------|----------------------------------|----------|
| **D10** — ≥1 original extension (Pareto / QLoRA / model sizes) | `run_extensions` ships the quant Pareto (primary) and specifies QLoRA + model-size scaling; the shipped one is documented with its figure | `reports/EXTENSIONS.md`, `figures/pareto.png`, `results/pareto.json` |
| **D4 / D6** — quantization trade-off + graphs | Pareto turns the FP16/Q8/Q4 sweep into a quality-vs-speed/VRAM frontier graph and a decision | `figures/pareto.png` |
| **D5** — systematic metrics reused | Pareto reads throughput + peak VRAM straight from the committed ledger; model_sizes records the metric set per size | `results/*.json` |
| **D8** — lecture-concept linking | Frontier (quality-vs-speed Pareto), model-size scaling, QLoRA/NF4/LoRA/trainable-vs-frozen all stated and computed | this PRD §2, `reports/EXTENSIONS.md` |
| **D15** — honesty | The frontier is computed from real committed numbers; QLoRA asserts the ratio math, never a faked train; degraded quality stays in the score | ledger ↔ figure ↔ README |

## 8. Risks & mitigations

- **Pareto quality axis is partly subjective.** Mitigation: the score is config-driven and
  derived from the committed `quality_note`/`output_text`; the raw text ships in the ledger so a
  reader can re-judge, and an explicit `quality_score` override is honoured.
- **Sparse data (only 3 points) makes "the frontier" thin.** Mitigation: present it as a labelled
  scatter with the frontier line + annotated optimal point; state explicitly it is a 3-point
  trade-off, not a smooth curve — honesty over polish.
- **`peft`/QLoRA is CUDA-and-train-heavy and out of CI scope.** Mitigation: the peft factory is
  injected and mocked; tests assert the **trainable/total ratio math** and the `LoraConfig`
  forwarding, never a real fine-tune; the live demo is the notebook only.
- **Model-size sweep needs multiple downloads/runs and may OOM the larger size on a T4.** Mitigation:
  the runner is injected/mocked in tests; the live sweep runs through AirLLM paging with small
  `max_new_tokens`, and 32B is **optional** (documented as a Kaggle-only stretch).
- **Scope creep — shipping all three.** Mitigation: **Pareto is the contracted deliverable** (no
  extra GPU, fully fixture-testable); model_sizes/qlora are specified and tested but shipped only if
  GPU time allows, and whichever ships is the one written up in `reports/EXTENSIONS.md`.
- **Figure non-determinism / headless CI.** Mitigation: force the `Agg` backend, fix random seeds,
  and assert on file existence + non-zero size rather than pixels.
- **Temptation to inflate QLoRA's win.** Out of scope: we report the measured/mocked ratio and VRAM
  comparison as-is; the small trainable fraction is the finding, not a number to tune.
