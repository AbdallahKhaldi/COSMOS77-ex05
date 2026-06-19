# CLAUDE.md — Project rules of engagement (binding for every prompt)

HW5 (Running a Massive LLM Locally: AirLLM, Quantization & Benchmarking) for Dr. Yoram Segal's
Orchestration of AI Agents (203.3763) course. Every prompt inherits these rules. HW5 acceptance
criteria (D1–D15) are in ../CLAUDE_CODE_PLAYBOOK.md §1.5. The grade is the ANALYSIS, not the model.

## The 17 rules
1. 150-line hard cap per .py file. Split it.
2. SDK architecture: all business logic via class SDK in src/cosmos77_ex05/sdk/sdk.py.
3. OOP, no duplication. 2 files -> shared module; 3 -> base class/mixin.
4. Zero hardcoded config (model id, quant levels, prices, hardware assumptions, paths) -> config/*.json or .env.
5. uv only for OUR code. The Kaggle/Colab runtime uses its own pip — that's the experiment env, documented separately.
6. TDD red->green->refactor. Mock ALL model/GPU/HF/network I/O. The test suite NEVER downloads a model or needs a GPU.
7. Coverage >= 85% on the measurement/analysis/economic code.
8. ruff check returns zero violations.
9. No secrets in repo. .env.example only; HF_TOKEN + any API key live in .env / Kaggle secrets, never in code.
10. Versioning starts at 1.00.
11. Conventional Commits per task; reference TODO IDs.
12. Prompt log: every session -> docs/prompts/NNN_*.md.
13. Gatekeeper/measurement ledger: every measured number flows through shared/gatekeeper.py into results/*.json.
    The ledger is the single source of truth for every table and graph. Never fabricate a number.
14. CLI only (Claude Code terminal). The deliverable is real code + a real measured experiment.
15. Docstrings on every public class/function/module (why, not what).
16. Type hints on every public signature. No bare Any.
17. Deterministic tests. Seed random; fix prompts; mock all model/GPU I/O. No flakes.

## Language & vocabulary
English only. Use the lecture vocabulary precisely: Prefill (GEMM, compute-bound) / Decode (GEMV,
memory-bound), VRAM, KV-cache, TTFT/TPOT, throughput/latency, quantization (FP16/Q8/Q4, NF4), the
virtual-memory/Paging analogy ("layer = page", mmap), SafeTensors, Roofline, On-Prem vs API, prompt caching.

## When in doubt
Less code, fewer deps, clearer docstrings. Impossible rule -> ADR in docs/PLAN.md. The honest measured
ledger + the correct causal explanation outrank a fast model. A well-analyzed negative result is acceptable.
