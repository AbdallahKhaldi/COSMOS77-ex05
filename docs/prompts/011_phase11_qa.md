# Prompt log 011 — Phase 11: QA gauntlet + acceptance audit

**Phase:** 11 — every gate green, the analysis reproducible, no criterion unmet
**Authors:** Abdallah Khaldi, Tasneem Natour
**Date:** 2026-06-20

## What was done

Ran the full QA gauntlet and a **4-reviewer adversarial acceptance audit** (a Workflow:
honesty / D1–D15 completeness / cyber+architecture / docs+reproducibility).

Gates (all green): `ruff check` + `ruff format --check` clean; `check_line_cap` 0
offenders; `pytest --cov-fail-under=85` → **143 passed, 100% coverage** (GPU-free);
`uv lock --check` consistent; no secrets tracked (`.env.example` only); 43 conventional
commits, both authors, 0 wip/tmp, 0 Claude trailers. **Fresh-clone reproducibility**
confirmed: clone → `uv sync` → pytest (143) → `analyze`/`economics` regenerate every
figure + table **byte-identically** from the committed ledger.

`docs/ACCEPTANCE.md` maps every D1–D15 → artifact → status (all ✅).

### Audit finding (fixed)

The honesty reviewer caught a **mislabelled "verbatim" OOM quote**: `reports/baseline.md`
and `reports/CONCEPTS.md` quoted *"15.89 GiB"* (a leftover from the earlier P100 runs),
but the committed T4 ledger records **14.56 GiB**. Fixed both to match the ledger exactly,
and standardised the usable-VRAM figure on `hardware.json`'s **15.6 GB** (was an
inconsistent ~15.9). Also corrected "51-layer" → "51-shard" (Qwen2.5-14B has 48 decoder
layers; AirLLM streams 51 per-layer shards). Three other dimensions passed clean; all
other numbers verified against `results/*.json` (throughput, TTFT/TPOT, peak VRAM, the
6× speedup, the effective-bandwidth + Roofline claims, the break-even volumes).

## Verification

```bash
uv run ruff check . && uv run pytest --cov-fail-under=85   # green
test -f docs/ACCEPTANCE.md && echo OK
```
