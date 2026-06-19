# PRD — On-Prem vs API vs Cloud-GPU Break-Even (D7)

> **Mechanism owner:** `src/cosmos77_ex05/economics/`
> **Maps to acceptance criterion:** **D7** — *Economic analysis: On-Prem (CAPEX amortized + electricity/OPEX) vs third-party API (tokens × price), optionally Cloud GPU; a break-even graph (cumulative cost vs usage volume) + all assumptions + a reasoned recommendation including privacy and Prompt/Context Caching.*
> **Phase:** 8 (economics). **Status:** specified, not yet implemented.
> **Binding rules:** CLAUDE.md §1 (17 rules). 150-line cap/file; single `class SDK` entry; `uv` only; TDD on fixtures (no network, matplotlib `Agg`); coverage ≥ 85%; config-driven (`config/pricing.json` + `config/setup.json → hardware_assumptions`); every result → `results/*.json`; English + lecture vocabulary.

---

## 1. Purpose

The grade for HW5 is the **analysis**, and D7 is the mandatory **economic** one. After benchmarking Qwen2.5-14B on a Kaggle **T4** (TTFT/TPOT/throughput from Phase 7), this mechanism answers the only question a real organisation asks next: **build (On-Prem) or buy (API), and from what volume does each win?**

We deliver a defensible **break-even** by computing **cost-per-request** under three deployment models and intersecting their cumulative-cost curves against **monthly request volume**:

1. **On-Prem** — buy the GPU once (**CAPEX**), pay electricity forever (**OPEX**). Per-request cost falls as volume rises because the fixed CAPEX amortizes over more requests. *Privacy value: nothing leaves the organisation.*
2. **API** — pay per token, no CAPEX. Per-request cost is **flat** in volume (linear total cost). Cheap at low volume, never amortizes.
3. **Cloud GPU** (optional) — rent the GPU by the hour ($/hr × runtime). No CAPEX, an OPEX floor between On-Prem and API.

The deliverable is the curve, the **break-even volume(s)**, and a recommendation that weighs **privacy/security**, not just dollars — On-Prem's real value per the lecture ("nothing leaves the org") is a reason to choose it *below* its cost break-even.

---

## 2. Inputs / Outputs

### Inputs (config-driven — **no silent defaults for graded numbers**, rule 4 / rule 13)
| Source | Key | Used for |
|---|---|---|
| `config/setup.json → hardware_assumptions` | `on_prem_gpu_price_usd` (1600) | CAPEX |
| | `hardware_life_years` (3) | amortization horizon |
| | `electricity_usd_per_kwh` (0.15) | OPEX |
| | `gpu_power_watts` (70) | OPEX |
| `config/pricing.json → providers.*` | `input_per_1m`, `output_per_1m` | API token cost (USD per 1M tokens) |
| `config/pricing.json → cloud_gpu` | `t4_usd_per_hour` (0.35), `a10g_usd_per_hour` | Cloud-GPU rate |
| `config/pricing.json → prompt_caching` | `cached_input_discount` (0.5) | caching shift (§4 caching) |
| `results/measure_*.json` (Phase 7) or caller | `runtime_s` per request, `tokens_per_req` (input+output split) | OPEX runtime + API token count |
| Caller / config | monthly volume grid, `cached_fraction` of input | the x-axis + caching scenario |

### Outputs
- **`results/economics.json`** — the D7 ledger entry: every assumption echoed back, the per-request cost of each model at a reference volume, the cost curves (volume → cost arrays), and the break-even volume(s). Single source of truth — never hand-edited.
- **`figures/breakeven.png`** — cumulative cost vs monthly volume, three lines, crossover(s) marked.
- **`reports/ECONOMICS.md`** — the table, the figure, **every assumption stated**, the break-even volume, and the privacy-weighted recommendation.

---

## 3. Module Design (files ≤ 150 lines, how to split)

```
src/cosmos77_ex05/economics/
├── __init__.py          # re-export break_even, apply_caching, build_report
├── model.py             # ≤150 — the cost math: per-request curves + crossover(s)
├── caching.py           # ≤60  — prompt/context-caching discount on the API curve
├── report.py            # ≤140 — assemble table + reports/ECONOMICS.md + results/economics.json
└── breakeven_plot.py    # ≤90  — matplotlib (Agg): the 3-line graph → figures/breakeven.png
```

**Why this split.** Four responsibilities, four failure modes: `model.py` is **pure arithmetic** (no I/O → trivially testable, no mocks); `caching.py` is one focused transform on the API curve (kept tiny so the *direction* of the shift is obvious and unit-testable in isolation); `report.py` does **impure** file writes (mock `tmp_path`); `breakeven_plot.py` is the only matplotlib file (forced `Agg`, no display). This honours single-responsibility (rule 3) and keeps every file well under 150.

### `model.py` (≤ 150) — the core
Pure functions, no I/O, raise on missing inputs:
- `break_even(api_price_per_1k, tokens_per_req, capex, life_years, kwh_price, watts, runtime_s, cloud_usd_per_hour=None, volumes=None) -> dict`
- private `_requests_in_life(monthly_volume, life_years)`, `_onprem_per_req(...)`, `_api_per_req(...)`, `_cloud_per_req(...)`, `_crossing(curve_a, curve_b, volumes)`.

### `caching.py` (≤ 60)
- `apply_caching(api_per_req, cached_fraction, cached_input_discount, input_share) -> float` — discounts the cached portion of **input** tokens, returns the new (lower) API per-request cost. Documents the direction: a cheaper API line **pushes the On-Prem↔API break-even to a higher volume** (API stays competitive longer).

### `report.py` (≤ 140)
- `build_report(cfg, measured, reports_dir, results_dir) -> dict` — load config, call `break_even` + `apply_caching`, write `results/economics.json`, render `reports/ECONOMICS.md` (with every assumption), call `breakeven_plot.plot(...)`, return the dict.

### `breakeven_plot.py` (≤ 90)
- `plot(curves, volumes, crossovers, out_path) -> Path` — `matplotlib.use("Agg")` at import; plot On-Prem / API / API-cached / Cloud; mark crossover(s) with a vertical line + annotation; save non-empty PNG.

---

## 4. The Cost Model Math

Let **V** = monthly request volume, **L** = `life_years`, lifetime requests **N(V) = V × 12 × L**.

**(a) On-Prem per request** — amortized CAPEX + electricity OPEX:
```
onprem(V) = capex / N(V)                              # amortized CAPEX, falls as V↑
          + (watts / 1000) × runtime_s × kwh_price / 3600   # OPEX per request (Wh→kWh, s→h)
```
*OPEX is volume-independent (a floor); CAPEX/N(V) → 0 as V → ∞, so On-Prem asymptotes to the electricity floor.*

**(b) API per request** — token cost, **flat in V**:
```
api = (input_tokens × input_price + output_tokens × output_price)
```
With prices given **per 1M tokens** in `pricing.json`, `price_per_token = price_per_1m / 1e6`. `tokens_per_req` carries the input/output split (e.g. 200 in + 20 out, matching `max_new_tokens`).

**(c) Cloud GPU per request** — rent by the hour, flat in V:
```
cloud = cloud_usd_per_hour × runtime_s / 3600
```

**Break-even volume** (On-Prem vs API): solve `onprem(V*) = api`. Since OPEX < api is required for a crossing to exist,
```
V* = capex / ((api − opex_per_req) × 12 × L)
```
Computed both **closed-form** (for the assertable number) and by **scanning the volume grid** for the sign-change crossing (drives the plotted marker); the two must agree on the fixture.

**Caching shift.** `cached_input_discount = 0.5` means a cached input token costs 50% of list. If a fraction *f* of input tokens are cached:
```
api_cached = api − (input_tokens × f × input_price × (1 − discount))
```
`api_cached < api` ⇒ the curve drops ⇒ **V\* moves right** (break-even at a *higher* volume — On-Prem must do *more* work to beat the now-cheaper API). The report states this direction explicitly.

---

## 5. Public SDK API

All logic is reached through the single `class SDK` (rule 2). This mechanism contributes one method:

```python
class SDK:
    def economics(self) -> dict:
        """Compute the On-Prem vs API vs Cloud-GPU break-even (D7).

        Reads hardware_assumptions from config/setup.json and prices from
        config/pricing.json; reads measured runtime_s / tokens_per_req from the
        Phase-7 ledger (results/measure_*.json). Writes results/economics.json,
        figures/breakeven.png, and reports/ECONOMICS.md, and returns the dict
        of per-request costs, cost curves, and break-even volume(s).

        Raises ValueError if any graded assumption (CAPEX, life_years, kWh
        rate, watts, token prices) is missing — no silent defaults.
        """
```

Wired to the `cosmos77-airllm economics` CLI stage (already in `PIPELINE_STAGES`). Verification: `uv run cosmos77-airllm economics` then `test -f figures/breakeven.png && grep -ci 'break-even\|assumption' reports/ECONOMICS.md`.

---

## 6. Test Plan (fixtures only — no network, matplotlib Agg)

All tests live in `tests/unit/economics/`. Pure math needs no mocks; file I/O uses `tmp_path`; the plot uses the `Agg` backend (rule 17 — deterministic).

| Test | Asserts |
|---|---|
| `test_onprem_per_req_known` | On a fixture (capex=1600, L=3, watts=70, runtime=2s, kwh=0.15, V=10k): CAPEX/N and OPEX terms match hand-computed values |
| `test_api_per_req_known` | `(200 in × 0.15/1e6) + (20 out × 0.60/1e6)` equals the closed-form cents |
| `test_cloud_per_req_known` | `0.35 × runtime_s/3600` exact |
| `test_break_even_volume` | closed-form `V*` equals the grid-scan crossing on the fixture (**crossover found at the right volume**) |
| `test_no_crossing_when_opex_exceeds_api` | when OPEX ≥ API, returns `None`/sentinel, not a bogus volume |
| `test_caching_moves_crossover_right` | `apply_caching` lowers the API curve and the new `V*` is **strictly greater** (the documented direction) |
| `test_missing_assumption_raises` | dropping `on_prem_gpu_price_usd` (or any graded key) raises a clear `ValueError` — **no silent default** |
| `test_economics_json_written` | `results/economics.json` exists, parses, echoes every assumption + curves + `V*` |
| `test_plot_writes_nonempty_png` | `figures/breakeven.png` exists and size > 0 |
| `test_report_states_assumptions` | `reports/ECONOMICS.md` contains every assumption (price, volume, life, kWh, watts) and the break-even number |

**Coverage target:** ≥ 85% on the `economics/` package. `model.py` and `caching.py` are pure → ~100% reachable; `report.py`/`breakeven_plot.py` covered via `tmp_path` + `Agg`.

---

## 7. Acceptance-Criteria Mapping

| Requirement (D7) | Satisfied by | Evidence |
|---|---|---|
| On-Prem CAPEX amortized + electricity OPEX | `model._onprem_per_req` | §4(a); `test_onprem_per_req_known` |
| API tokens × price | `model._api_per_req` | §4(b); `test_api_per_req_known` |
| Cloud GPU $/hr × runtime (optional) | `model._cloud_per_req` | §4(c); `test_cloud_per_req_known` |
| Break-even graph (cumulative cost vs volume) | `breakeven_plot.plot` | `figures/breakeven.png`, crossover marked |
| Break-even **volume** computed | `model.break_even` (closed-form + grid) | `results/economics.json → break_even`; `test_break_even_volume` |
| **All assumptions stated** | `report.build_report` | `reports/ECONOMICS.md`; `test_report_states_assumptions` |
| Prompt/Context Caching effect | `caching.apply_caching` | §4 shift; `test_caching_moves_crossover_right` |
| Reasoned recommendation incl. **privacy/security** | `reports/ECONOMICS.md` prose | "nothing leaves the organisation" weighed vs cost |
| No fabricated numbers | config + ledger pipeline | `results/economics.json` is the single source |

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Silent default fills a missing assumption | a graded number is invented (rule 13) | every key required; `test_missing_assumption_raises`; config-driven only |
| Unit slips (Wh vs kWh, s vs hr, per-token vs per-1M) | cost off by 1000× | conventions pinned in docstrings (`watts/1000`, `runtime_s/3600`, `price/1e6`); fixtures assert exact values |
| Prices are assumptions, not eternal truth | grader doubts the numbers | `pricing.json` `note` flags them as representative mid-2025 list prices; `ECONOMICS.md` cites source + date; the *method* is what D7 grades |
| Closed-form `V*` and grid scan disagree | plotted marker ≠ reported number | both computed and cross-asserted (`test_break_even_volume`); grid fine enough near the crossing |
| Caching direction stated backwards | wrong analysis | direction derived in §4 and locked by `test_caching_moves_crossover_right` (V* strictly increases) |
| Recommendation reduces to "cheapest wins" | misses D7's privacy requirement | report explicitly weighs On-Prem's "nothing leaves the org" value; recommend On-Prem below break-even when data is sensitive |
| File exceeds 150 lines | rule 1 violation | four-file split; if `report.py` grows, move the Markdown templating into a tiny `report_md.py` |
| matplotlib opens a display in CI | suite hangs (rule 17) | `matplotlib.use("Agg")` at import of `breakeven_plot.py` |

---

## 9. Definition of Done

- [ ] `model.py` ≤ 150, `caching.py` ≤ 60, `report.py` ≤ 140, `breakeven_plot.py` ≤ 90; `ruff check` clean; line-cap script passes.
- [ ] `economics()` writes `results/economics.json`, `figures/breakeven.png`, `reports/ECONOMICS.md` and returns the curves + break-even volume(s).
- [ ] Per-request costs match hand-computed fixtures for all three models; closed-form `V*` == grid crossing.
- [ ] Caching lowers the API curve and moves `V*` to a **higher** volume (documented direction, tested).
- [ ] A missing graded assumption raises a clear `ValueError` (no silent defaults).
- [ ] `reports/ECONOMICS.md` states every assumption + the break-even volume + a privacy-weighted recommendation.
- [ ] All tests fixture-driven, no network, `Agg` backend; coverage ≥ 85% on `economics/`.
- [ ] Docstrings + type hints on every public signature; all numbers flow from config → ledger (nothing hand-edited).
