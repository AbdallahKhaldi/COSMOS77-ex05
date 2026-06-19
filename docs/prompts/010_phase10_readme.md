# Prompt log 010 — Phase 10: README as the deep technical report

**Phase:** 10 — the report IS the README (D11)
**Authors:** Abdallah Khaldi, Tasneem Natour
**Date:** 2026-06-20

## Prompt issued

> Phase 10 goal: assemble the section reports + every figure/table into one deep
> report-as-README (spec §8): title/authors, hardware + model-choice math, experiment
> description, findings (table + graphs), Roofline, economics, concept linking, the §4
> research questions, the extension, reproduction, self-assessment. ≥250 lines, ≥6 figures.

## What was done

Wrote `README.md` (the report) by synthesising the committed ledger + the section
reports (METRICS/ECONOMICS/CONCEPTS/EXTENSIONS/baseline/airllm/quantization). Also wrote
the three per-mechanism narration reports `reports/baseline.md` (D2), `reports/airllm.md`
(D3), `reports/quantization.md` (D4).

The README has all 10 required sections, **embeds 7 figures** (peak VRAM, tokens/sec,
TTFT-vs-TPOT, quant trade-off, Roofline, break-even, Pareto), answers RQ-a…RQ-f
explicitly, includes a reproducibility log (the exact Kaggle env pins + the P100→T4
story), and a self-assessment recommending **85** against D1–D15.

## Verification

```bash
wc -l README.md            # >= 250
grep -c '!\[' README.md    # >= 6 (7 figures + CI badge)
```

Every embedded figure path exists; every quoted number was checked against
`results/*.json` (e.g. 4-bit 0.0410 tok/s, FP16 peak VRAM 1.60 GB) — nothing fabricated.
