# Prompt log 001 — Phase 1: Mandatory documentation

**Phase:** 1 — All mandatory docs BEFORE business logic
**Authors:** Abdallah Khaldi, Tasneem Natour
**Date:** 2026-06-19

## Prompt issued

> Phase 1 goal: ALL mandatory documentation before business logic. Substantive,
> not stubs. Per `../CLAUDE_CODE_PLAYBOOK.md` §3: `docs/PRD.md` (context = L08
> local inference / the virtual-memory Paging analogy / Prefill-Decode /
> quantization; the §4 research questions as requirements; FR→D1–D15; KPIs);
> eight per-mechanism PRDs (hardware, baseline, airllm, quantization,
> measurement, analysis, economics, extensions) — **parallelize**; `docs/PLAN.md`
> (C4 + experiment sequence diagram + ADR-001…006 + risk register); `docs/TODO.md`
> (**≥600** granular `T-NNNN` tasks across P0–P12); the prompt log. English + the
> lecture vocabulary throughout.

## What was done

Per the playbook's "parallelize the per-mechanism PRDs" directive, the documents
were produced by **11 parallel subagents**, each writing one file and returning a
short summary (keeping the orchestrator's context lean for the multi-phase build).
The orchestrator then verified counts, read the backbone PRD, and committed.

- **`docs/PRD.md`** (165 lines) — problem statement (the honest-measurement
  thesis: the grade is the analysis); L08 context (the Paging analogy, Prefill
  (GEMM, compute-bound)→TTFT / Decode (GEMV, memory-bound)→TPOT, the 14B→29.4GB
  OOM math); RQ-a…RQ-f research questions; FR-1…FR-15 mapped 1:1 to **D1–D15**;
  non-functional requirements; K-1…K-15 KPIs; out of scope.
- **`docs/PLAN.md`** (266 lines) — the **C4 model** (Mermaid flowchart); a Mermaid
  experiment `sequenceDiagram` (hardware → download+shard → FP16 OOM → AirLLM →
  quant sweep → measure → analyze → economics → report); **six ADRs** (ADR-001
  Kaggle T4 over the Mac [bitsandbytes is CUDA-only], ADR-002 Qwen2.5-14B,
  ADR-003 notebook + tested src library, ADR-004 honest measurement / negatives
  allowed, ADR-005 single SDK, ADR-006 150-line cap); a risk register.
- **Eight per-mechanism PRDs** (`docs/PRD_{hardware,baseline,airllm,quantization,
  measurement,analysis,economics,extensions}.md`, ~148–231 lines each) — Purpose,
  Inputs/Outputs, module design under the 150-line cap, the SDK API method, a
  fully-mocked TDD test plan (no model/GPU/HF in the suite), acceptance mapping,
  and risks. They agree on SDK method names, module layout, config keys, the
  gatekeeper-ledger contract, and the lecture vocabulary.
- **`docs/TODO.md`** — expanded from the Phase-0 seed to **670** granular `T-NNNN`
  tasks (contiguous T-0001…T-0670), distributed P0 (done) through P12.

## Verification

```bash
grep -c '^T-' docs/TODO.md     # 670  (>= 600)
ls docs/PRD_*.md | wc -l       # 8    (>= 8)
grep -c 'ADR-' docs/PLAN.md    # 11   (>= 6; six ADR sections)
uv run pytest -m "not live"    # green (no .py added this phase)
```

## Notes / decisions

- **No business logic** — Phase 1 is documentation only; the gates remain green
  from Phase 0.
- The TODO budget overshoots 600 (→670) to leave margin as tasks split during
  implementation; Phase 2 alone carries 90 tasks (shared infra + hardware capture
  + measurement ledger + economics model — the testable core).
