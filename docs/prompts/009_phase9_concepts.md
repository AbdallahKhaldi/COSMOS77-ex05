# Prompt log 009 — Phase 9: Concept analysis + original extension

**Phase:** 9 — link every result to the lecture (D8) + ship the extension (D10)
**Authors:** Abdallah Khaldi, Tasneem Natour
**Date:** 2026-06-20

## Prompt issued

> Phase 9 goal: tie every measured result to Lecture 08 (Prefill/Decode, compute- vs
> memory-bound, VRAM/KV-cache, the virtual-memory/Paging analogy, quantization, the
> Roofline reading) in `reports/CONCEPTS.md`, and implement ≥1 original extension with
> its own figure in `reports/EXTENSIONS.md`.

## What was done

- **`reports/CONCEPTS.md`** — the concept-linking analysis, every claim tied to a
  number in the ledger: the VRAM wall (OOM at 29.4 > 15.9 GB); AirLLM = OS paging
  ("layer = page", page fault, `mmap`, evict, SafeTensors) proven by peak VRAM
  1.6–3.2 GB vs 29.4 GB needed; the disk-bandwidth wall (effective ≈0.2–0.3 GB/s,
  ~1500× below the 320 GB/s HBM) and its Roofline reading (AI ≈1–4 ≪ ridge 203,
  memory-bound, points three orders of magnitude below the memory roof). Two original
  observations the data forced out: (1) AirLLM **flattens Prefill vs Decode** into one
  disk-bound regime (TTFT ≈ TPOT for FP16/8-bit because every step is a full 51-layer
  disk sweep); (2) the 4-bit `TTFT > TPOT` is the **OS page cache warming** (cold first
  token, warm later tokens, smaller shards stay cached). Plus the quantization
  memory/speed/quality story and the counter-intuitive VRAM-rises-with-Q nuance.
- **`extensions/pareto.py` + `reports/EXTENSIONS.md`** — the original extension: a
  **quantization speed-vs-VRAM Pareto** from the committed ledger. All three points are
  Pareto-non-dominated; quality is the flat third axis (no Q4 red line). Explains why
  (resident weights are tiny under paging; quantization shrinks the *disk* footprint
  that governs speed while adding a small dequant-buffer VRAM overhead) → `figures/pareto.png`.
  Wired into `SDK.analyze()` so `cosmos77-airllm analyze` regenerates it.

## Verification

```bash
uv run pytest -m "not live"   # 143 passed, coverage 100%
ls reports/CONCEPTS.md reports/EXTENSIONS.md figures/pareto.png
```

Every figure was visually checked; every number in both reports traces to
`results/*.json`. Nothing is fabricated; the analyses re-derive from the committed ledger.
