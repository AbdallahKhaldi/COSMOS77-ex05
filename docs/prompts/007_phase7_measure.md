# Prompt log 007 — Phase 7: tables, graphs, Roofline

**Phase:** 7 — the measurement library's analysis layer (D5/D6/D9)
**Authors:** Abdallah Khaldi, Tasneem Natour
**Date:** 2026-06-20

## Prompt issued

> Phase 7 goal: the comparison tables, graphs, and the Roofline, built from the
> committed measurement ledger (`results/*.json`). Pure-Python + matplotlib, fully
> testable, GPU-free. `analysis/tables.py` → `reports/METRICS.md`; `analysis/plots.py`
> → `figures/{tokens_per_sec,peak_vram,ttft_vs_tpot,quant_tradeoff}.png`;
> `analysis/roofline.py` → `figures/roofline.png`. Wire `SDK.analyze()` + the CLI.

## What was done

The measurement harness (`measure/`) shipped in Phases 3–6; Phase 7 adds the analysis
that turns the real T4 ledger into tables + figures. Built via three parallel
subagents over a shared `analysis/loader.py` (which reads the Gatekeeper ledger and
orders the four scenarios), then integrated:

- **`analysis/loader.py`** — `load_ledger`, `ordered`, `airllm_rows`, `label`. One
  place that reads `results/*.json`; `fp16_baseline` is the OOM control, the three
  `airllm_*` rows carry the metrics.
- **`analysis/tables.py`** — `build_dataframe` + `write_metrics_md` → `reports/METRICS.md`
  (the GPU/spec line + the 4-scenario comparison table + honest takeaways; a manual
  pipe-table fallback since `tabulate` isn't a dependency).
- **`analysis/plots.py`** — four figures: tokens/sec, peak VRAM (with the 16 GB T4
  limit + 29.4 GB FP16-OOM reference lines), TTFT vs TPOT, and the quantization
  trade-off (twin axis). matplotlib `Agg`, every value from the ledger.
- **`analysis/roofline.py`** — the T4 Roofline (peak ≈ 65 TFLOPS, BW ≈ 320 GB/s,
  ridge ≈ 203 FLOPs/byte) with the three measured AirLLM points. Computes arithmetic
  intensity (1 / 2 / 4 FLOPs/byte → all memory-bound, left of the ridge) and the
  **effective sustained bandwidth: ~0.20–0.30 GB/s — ~1000–1500× below the 320 GB/s
  HBM**. That is the headline: AirLLM trades the HBM-memory wall for a **disk-bandwidth
  wall**. (A `/1000` annotation bug found by visual inspection of the rendered PNG was
  fixed — the caption now reads ~1524× below, matching the data.)
- **`SDK.analyze()`** + CLI `analyze` build all artifacts from the ledger.

## Verification

```bash
uv run pytest -m "not live"       # 125 passed, coverage 100%
uv run cosmos77-airllm analyze    # -> reports/METRICS.md + figures/*.png (5)
```

Every figure was visually checked against the ledger; the METRICS.md numbers
byte-match `results/*.json` (0.007022 / 0.013784 / 0.040964 tok/s) — nothing hardcoded
or fabricated. The figures are generated from the single committed ledger, so the
analysis re-derives identically on a fresh clone.

## Notes

- The peak-VRAM figure shows AirLLM running the 29 GB model in **1.6–3.2 GB** of VRAM
  — the "layer = page" paging made visible. The Roofline shows *why* it's slow.
- `SDK.measure()` stays unimplemented on the Mac by design — the measurement runs
  inside the runners on the T4 (the notebook), not on a GPU-less laptop.
