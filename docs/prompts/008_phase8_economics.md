# Prompt log 008 — Phase 8: Economic break-even

**Phase:** 8 — On-Prem vs API vs Cloud-GPU break-even (D7)
**Authors:** Abdallah Khaldi, Tasneem Natour
**Date:** 2026-06-20

## Prompt issued

> Phase 8 goal: the mandatory economics — a real break-even with stated assumptions
> and a reasoned recommendation. Pure-Python + matplotlib, testable. `economics/report.py`
> (per-request costs + `reports/ECONOMICS.md`) and `economics/breakeven_plot.py`
> (`figures/breakeven.png`) on top of the Phase-2 cost-model core; wire `SDK.economics()`.

## What was done

The cost-model core (`economics/model.py` break-even math + `caching.py`) shipped in
Phase 2; Phase 8 adds the report + plot (built by a subagent), then wired the SDK.

- **`economics/report.py`** — `compute_scenario` (On-Prem amortized CAPEX + electricity
  OPEX vs API tokens×price vs Cloud-GPU $/hr) and `write_economics_md` →
  `reports/ECONOMICS.md`: an **Assumptions** table (CAPEX $1600, 3-yr life, $0.15/kWh,
  70 W, a representative 500-in/200-out-token request at a 10 s runtime basis,
  1000 req/day amortization, 0.5 cached fraction), a per-request **cost table** across
  On-Prem + 3 API providers + Cloud-GPU, the **break-even volume**, the
  **prompt-caching** shift, and a privacy-aware recommendation.
- **`economics/breakeven_plot.py`** — cumulative cost vs monthly volume for all lines
  + the caching-discounted API line, crossover marked → `figures/breakeven.png`.
- **`SDK.economics()`** + CLI `economics`, reusing `report._REPORT_REQUEST` /
  `_REPORT_PROVIDER` so the plot and the report share one set of assumptions.

## Result + verification

```bash
uv run pytest -m "not live"        # 143 passed, coverage 100%
uv run cosmos77-airllm economics   # -> reports/ECONOMICS.md + figures/breakeven.png
```

On-Prem per-request marginal (electricity) cost ≈ $2.9e-5 vs gemini-flash $9.7e-5, so
On-Prem **breaks even at ≈23.4 M requests**; prompt caching lowers the API price and
**raises** the break-even to ≈27.1 M (a cheaper API stays competitive longer). The
honest recommendation: at our measured AirLLM throughput (~0.04 tok/s) local serving is
impractical for *throughput* — **On-Prem's real value is privacy** ("nothing leaves the
organisation"), not raw cost, until volume is enormous.
