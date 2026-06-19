# TODO — COSMOS77-ex05 Master Task Ledger

Single source of truth for all outstanding and completed work on HW5 ("Running a
Massive LLM Locally: AirLLM, Quantization & Performance Benchmarking"). Each task is
ONE line, pipe-delimited, beginning literally with `T-NNNN ` (four digits):

`T-NNNN | <phase 0-12> | <area> | <description> | <DoD = definition of done> | <status>`

Status ∈ `todo` · `doing` · `done`. Phase headings (`## Phase N`) are prose only and
do NOT begin with `T-`. Vocabulary follows Lecture 08: Prefill/Decode, compute/memory-
bound, VRAM, TTFT/TPOT, quantization (FP16/Q8/Q4/NF4), Paging ("layer = page", `mmap`),
Roofline, On-Prem vs API, prompt caching. Every reported number flows from the
committed ledger `results/*.json`; nothing is fabricated.

## Phase 0 — Repo bootstrap

T-0001 | 0 | repo | Reconnaissance: confirm uv/gh/git/kaggle versions, ~/.kaggle creds, HW4 tooling to port | versions confirmed; port plan set | done
T-0002 | 0 | repo | Directory skeleton (src/ tests/ docs/ config/ experiments/ results/ reports/ figures/ data/) + subpackages | tree matches playbook §2.2 | done
T-0003 | 0 | build | pyproject.toml — cosmos77-ex05 v1.00, py3.11, analysis deps + experiment extra + kaggle dev dep | uv sync resolves; uv.lock committed | done
T-0004 | 0 | build | .python-version, .gitignore (data/shards/kaggle_out ignored; results/figures/ipynb kept), .env.example | .env ignored; only .env.example tracked | done
T-0005 | 0 | config | config/setup.json (Qwen2.5-14B, T4, 4 scenarios, hardware assumptions) | parses; matches playbook | done
T-0006 | 0 | config | config/pricing.json (API prices + cloud GPU + caching) + logging_config.json | parses | done
T-0007 | 0 | pkg | src/cosmos77_ex05 __init__ (v1.00), constants (SCENARIOS/BYTES_PER_PARAM), cli/main dispatcher, empty subpackages | package imports; --version works | done
T-0008 | 0 | rules | CLAUDE.md (17 rules, §16 verbatim) | byte-matches playbook §16 | done
T-0009 | 0 | docs | README placeholder, LICENSE (MIT 2026), CHANGELOG [1.00], CONTRIBUTING (two-env note) | all present; CI badge in README | done
T-0010 | 0 | ci | scripts/check_line_cap.py (port), generate_cover_pdf.py (port, ex05/exercise 5) | line-cap exits 0 | done
T-0011 | 0 | ci | .pre-commit-config.yaml + .github/workflows/ci.yml (ruff/format/line-cap/pytest-85; no GPU/model) | hooks installed; CI green | done
T-0012 | 0 | test | tests/conftest.py (seed) + tests/unit/test_constants.py smoke test | pytest green, coverage >= 85% | done
T-0013 | 0 | qa | uv sync; ruff check/format zero; check_line_cap 0; pytest >= 85% | every gate green | done
T-0014 | 0 | docs | docs/prompts/000_phase0_bootstrap.md prompt log | file present | done
T-0015 | 0 | git | git init -b main, identity + remote, conventional commits, push -u origin main, CI green | Actions green on main | done

## Phase 1 — Docs (PRD / PLAN / TODO + 8 mechanism PRDs)

T-0016 | 1 | docs | docs/PRD.md §1 problem statement (29 GB model on 16 GB T4, honest-measurement thesis) | section present, matches spec | done
T-0017 | 1 | docs | docs/PRD.md §2 context — Lecture 08 local inference, Paging analogy (layer=page) | section present | done
T-0018 | 1 | docs | docs/PRD.md §2 Prefill/Decode mental model (GEMM/GEMV, TTFT/TPOT) | section present | done
T-0019 | 1 | docs | docs/PRD.md §2 param→memory OOM math (14.7B × 2 = 29.4 GB > 16 GB) | math present | done
T-0020 | 1 | docs | docs/PRD.md §3 RQ-a bottleneck identification (memory not compute) | RQ-a written | done
T-0021 | 1 | docs | docs/PRD.md §3 RQ-b AirLLM resource reallocation + Paging connection | RQ-b written | done
T-0022 | 1 | docs | docs/PRD.md §3 RQ-c quantization effect + accuracy red line | RQ-c written | done
T-0023 | 1 | docs | docs/PRD.md §3 RQ-d Prefill/Decode → TTFT/TPOT mapping | RQ-d written | done
T-0024 | 1 | docs | docs/PRD.md §3 RQ-e throughput/latency price | RQ-e written | done
T-0025 | 1 | docs | docs/PRD.md §3 RQ-f On-Prem vs API break-even | RQ-f written | done
T-0026 | 1 | docs | docs/PRD.md §4 FR-to-acceptance table D1–D15 | table present | done
T-0027 | 1 | docs | docs/PRD.md §5 non-functional requirements NFR-1..NFR-9 | section present | done
T-0028 | 1 | docs | docs/PRD.md §6 KPIs / DoD gates K-1..K-15 | table present | done
T-0029 | 1 | docs | docs/PRD.md §7 out of scope | section present | done
T-0030 | 1 | docs | docs/PLAN.md phase roadmap P0–P12 with deliverables | roadmap present | done
T-0031 | 1 | docs | docs/PLAN.md two-environment note (uv local vs pip Kaggle) | note present | done
T-0032 | 1 | docs | docs/PLAN.md risk register (OOM, download failure, quota, flakiness) | risks listed | done
T-0033 | 1 | docs | docs/PLAN.md ledger schema + gatekeeper contract | schema present | done
T-0034 | 1 | docs | docs/PLAN.md acceptance mapping D1–D15 → phases/artifacts | mapping present | done
T-0035 | 1 | docs | docs/TODO.md seed (Phase 0) + format spec | seed present | done
T-0036 | 1 | docs | docs/TODO.md expand to ≥600 granular tasks P0–P12 | grep -c '^T-' ≥ 600 | done
T-0037 | 1 | prd | docs/PRD_hardware.md — hardware capture + param→memory model math mechanism | PRD present | done
T-0038 | 1 | prd | docs/PRD_hardware.md acceptance criteria + ledger fields | criteria present | done
T-0039 | 1 | prd | docs/PRD_baseline.md — FP16 naive load + OOM capture mechanism | PRD present | done
T-0040 | 1 | prd | docs/PRD_baseline.md OOM evidence contract (requested vs available VRAM) | contract present | done
T-0041 | 1 | prd | docs/PRD_airllm.md — AirLLM layer=page streaming mechanism | PRD present | done
T-0042 | 1 | prd | docs/PRD_airllm.md SafeTensors sharding + mmap + evict-after-use | section present | done
T-0043 | 1 | prd | docs/PRD_quantization.md — FP16/Q8/Q4 sweep mechanism + NF4 note | PRD present | done
T-0044 | 1 | prd | docs/PRD_quantization.md accuracy red line definition | definition present | done
T-0045 | 1 | prd | docs/PRD_measure.md — TTFT/TPOT/throughput/peak VRAM measurement harness | PRD present | done
T-0046 | 1 | prd | docs/PRD_measure.md metric definitions + units + sampling method | definitions present | done
T-0047 | 1 | prd | docs/PRD_economics.md — On-Prem vs API vs Cloud GPU break-even mechanism | PRD present | done
T-0048 | 1 | prd | docs/PRD_economics.md CAPEX/OPEX model + caching shift + assumptions | model present | done
T-0049 | 1 | prd | docs/PRD_roofline.md — Roofline arithmetic-intensity mechanism | PRD present | done
T-0050 | 1 | prd | docs/PRD_roofline.md ridge point math (T4 ≈65 TFLOPS / ≈320 GB/s) | math present | done
T-0051 | 1 | prd | docs/PRD_concepts.md — lecture-concept linking mechanism | PRD present | done
T-0052 | 1 | prd | docs/PRD_concepts.md concept→result→ledger traceability map | map present | done
T-0053 | 1 | prd | docs/PRD_extension.md — original extension (quant Pareto / QLoRA / three-way) | PRD present | done
T-0054 | 1 | prd | docs/PRD_extension.md extension acceptance + figure spec | spec present | done
T-0055 | 1 | docs | Cross-link all 8 mechanism PRDs from docs/PRD.md §4 table | links resolve | done
T-0056 | 1 | docs | docs/prompts/001_phase1_docs.md prompt log | file present | done
T-0057 | 1 | qa | Spell/terminology pass on all PRDs (lecture vocabulary consistent) | zero drift | done
T-0058 | 1 | qa | Verify every D1–D15 maps to ≥1 PRD + ≥1 phase | mapping complete | done
T-0059 | 1 | docs | docs/GLOSSARY.md (Prefill/Decode, AI, KV-cache, NF4, mmap, ridge point) | glossary present | done
T-0060 | 1 | git | Commit Phase 1 docs (conventional commit, CI green) | committed; CI green | done

## Phase 2 — Shared infra + hardware capture + measurement ledger + economics model

T-0061 | 2 | infra | tests/unit/test_gatekeeper.py — failing test for ledger write contract (red) | test fails | done
T-0062 | 2 | infra | src/cosmos77_ex05/shared/gatekeeper.py — minimal write_result (green) | test passes | done
T-0063 | 2 | infra | gatekeeper: atomic JSON write + schema validation | test passes | done
T-0064 | 2 | infra | gatekeeper: refactor for ≤150 lines, extract schema module | line-cap 0; tests green | done
T-0065 | 2 | infra | gatekeeper docstrings + type hints | mypy/ruff clean | done
T-0066 | 2 | infra | tests/unit/test_ledger_schema.py — failing test for results schema (red) | test fails | done
T-0067 | 2 | infra | src/cosmos77_ex05/shared/ledger_schema.py — Pydantic/TypedDict result model (green) | test passes | done
T-0068 | 2 | infra | ledger_schema: required fields (scenario, ttft, tpot, throughput, peak_vram, peak_ram, runtime, power, quality) | test passes | done
T-0069 | 2 | infra | ledger_schema: refactor + docstrings + type hints | ruff clean | done
T-0070 | 2 | infra | tests/unit/test_config_loader.py — failing test for config/setup.json loader (red) | test fails | done
T-0071 | 2 | infra | src/cosmos77_ex05/shared/config.py — load_setup() (green) | test passes | done
T-0072 | 2 | infra | config.py: load_pricing() + load_logging_config() | test passes | done
T-0073 | 2 | infra | config.py: env override + .env loading (no secrets in code) | test passes | done
T-0074 | 2 | infra | config.py: refactor + docstrings + type hints | ruff clean | done
T-0075 | 2 | infra | tests/unit/test_paths.py — failing test for path resolver (red) | test fails | done
T-0076 | 2 | infra | src/cosmos77_ex05/shared/paths.py — repo-root-relative path helpers (green) | test passes | done
T-0077 | 2 | infra | paths.py: results/ figures/ reports/ data/ shards/ accessors | test passes | done
T-0078 | 2 | infra | paths.py: refactor + docstrings + type hints | ruff clean | done
T-0079 | 2 | infra | tests/unit/test_logging.py — failing test for structured logger (red) | test fails | done
T-0080 | 2 | infra | src/cosmos77_ex05/shared/log.py — get_logger() from logging_config.json (green) | test passes | done
T-0081 | 2 | infra | log.py: refactor + docstrings + type hints | ruff clean | done
T-0082 | 2 | infra | tests/unit/test_seed.py — failing test for deterministic seeding (red) | test fails | done
T-0083 | 2 | infra | src/cosmos77_ex05/shared/seed.py — seed random/np/torch (green) | test passes | done
T-0084 | 2 | infra | seed.py: refactor + docstrings + type hints | ruff clean | done
T-0085 | 2 | hw | tests/unit/test_hw_spec.py — failing test for hardware spec capture (red) | test fails | done
T-0086 | 2 | hw | src/cosmos77_ex05/hardware/spec.py — capture CPU/cores/RAM (mocked) (green) | test passes | done
T-0087 | 2 | hw | spec.py: capture GPU name/VRAM via mocked torch.cuda | test passes | done
T-0088 | 2 | hw | spec.py: capture disk free + bandwidth assumption fields | test passes | done
T-0089 | 2 | hw | spec.py: write results/hardware.json via gatekeeper | test passes | done
T-0090 | 2 | hw | spec.py: refactor for ≤150 lines, split capture vs serialize | line-cap 0 | done
T-0091 | 2 | hw | spec.py: docstrings + type hints | ruff clean | done
T-0092 | 2 | hw | tests/unit/test_model_math.py — failing test for param→memory math (red) | test fails | done
T-0093 | 2 | hw | src/cosmos77_ex05/hardware/model_math.py — params × bytes_per_param (green) | test passes | done
T-0094 | 2 | hw | model_math.py: FP16=2, Q8=1, Q4=0.5 byte tables + GB conversion | test passes | done
T-0095 | 2 | hw | model_math.py: fits_in_vram() predicate vs T4 16 GB | test passes | done
T-0096 | 2 | hw | model_math.py: OOM ratio (29.4/16 ≈ 1.8×) helper | test passes | done
T-0097 | 2 | hw | model_math.py: KV-cache size estimate (layers × heads × dim × seq × 2) | test passes | done
T-0098 | 2 | hw | model_math.py: refactor + docstrings + type hints | ruff clean | done
T-0099 | 2 | hw | tests/unit/test_hardware_fixture.py — fixture results/hardware.sample.json | fixture loads | done
T-0100 | 2 | hw | Wire hardware.spec + model_math into SDK facade method capture_hardware() | SDK call works | done
T-0101 | 2 | econ | tests/unit/test_capex.py — failing test for On-Prem CAPEX amortisation (red) | test fails | done
T-0102 | 2 | econ | src/cosmos77_ex05/economics/capex.py — hardware cost / amortisation window (green) | test passes | done
T-0103 | 2 | econ | capex.py: refactor + docstrings + type hints | ruff clean | done
T-0104 | 2 | econ | tests/unit/test_opex.py — failing test for electricity OPEX (red) | test fails | done
T-0105 | 2 | econ | src/cosmos77_ex05/economics/opex.py — watts × hours × $/kWh (green) | test passes | done
T-0106 | 2 | econ | opex.py: refactor + docstrings + type hints | ruff clean | done
T-0107 | 2 | econ | tests/unit/test_api_cost.py — failing test for API token cost (red) | test fails | done
T-0108 | 2 | econ | src/cosmos77_ex05/economics/api_cost.py — input/output tokens × price (green) | test passes | done
T-0109 | 2 | econ | api_cost.py: prompt-caching discount on cached input tokens | test passes | done
T-0110 | 2 | econ | api_cost.py: refactor + docstrings + type hints | ruff clean | done
T-0111 | 2 | econ | tests/unit/test_cloud_gpu.py — failing test for cloud GPU $/hr cost (red) | test fails | done
T-0112 | 2 | econ | src/cosmos77_ex05/economics/cloud_gpu.py — $/hr × runtime model (green) | test passes | done
T-0113 | 2 | econ | cloud_gpu.py: refactor + docstrings + type hints | ruff clean | done
T-0114 | 2 | econ | tests/unit/test_breakeven.py — failing test for crossover volume (red) | test fails | done
T-0115 | 2 | econ | src/cosmos77_ex05/economics/breakeven.py — On-Prem vs API crossover solve (green) | test passes | done
T-0116 | 2 | econ | breakeven.py: cumulative-cost curves over token volume | test passes | done
T-0117 | 2 | econ | breakeven.py: caching-shifted API line second crossover | test passes | done
T-0118 | 2 | econ | breakeven.py: refactor for ≤150 lines + docstrings + type hints | line-cap 0; ruff clean | done
T-0119 | 2 | econ | economics fixture: config/pricing.sample.json + expected curves | fixture loads | done
T-0120 | 2 | econ | Wire economics modules into SDK facade method run_economics() | SDK call works | done
T-0121 | 2 | infra | tests/unit/test_power_est.py — failing test for power estimate (red) | test fails | done
T-0122 | 2 | infra | src/cosmos77_ex05/shared/power.py — est. energy = watts × runtime (green) | test passes | done
T-0123 | 2 | infra | power.py: refactor + docstrings + type hints | ruff clean | done
T-0124 | 2 | infra | tests/unit/test_quality_score.py — failing test for qualitative quality tag (red) | test fails | done
T-0125 | 2 | infra | src/cosmos77_ex05/shared/quality.py — quality label vs reference output (green) | test passes | done
T-0126 | 2 | infra | quality.py: refactor + docstrings + type hints | ruff clean | done
T-0127 | 2 | infra | tests/unit/test_units.py — failing test for unit conversions (bytes↔GB, s↔ms) (red) | test fails | done
T-0128 | 2 | infra | src/cosmos77_ex05/shared/units.py — conversion helpers (green) | test passes | done
T-0129 | 2 | infra | units.py: refactor + docstrings + type hints | ruff clean | done
T-0130 | 2 | qa | Run gate: ruff check/format zero on Phase 2 modules | gate green | done
T-0131 | 2 | qa | Run gate: check_line_cap.py zero on Phase 2 modules | gate green | done
T-0132 | 2 | qa | Run gate: pytest coverage ≥85% on shared/hardware/economics | coverage ≥85% | done
T-0133 | 2 | report | reports/hardware.md draft from results/hardware.json | report present | done
T-0134 | 2 | report | reports/ECONOMICS.md skeleton (assumptions table placeholder) | skeleton present | done
T-0135 | 2 | docs | docs/prompts/002_phase2_infra.md prompt log | file present | done
T-0136 | 2 | ledger | Update docs/TODO.md statuses for completed Phase 2 tasks | statuses current | done
T-0137 | 2 | git | Commit Phase 2 shared infra (conventional commit) | committed; CI green | done
T-0138 | 2 | infra | gatekeeper: reject write if required field missing (negative test) | test passes | done
T-0139 | 2 | infra | gatekeeper: append-only audit log of ledger writes | test passes | done
T-0140 | 2 | hw | model_math.py: activation memory estimate for prefill batch | test passes | done
T-0141 | 2 | hw | model_math.py: CUDA context overhead constant + total VRAM budget | test passes | done
T-0142 | 2 | econ | breakeven.py: privacy/security weight annotation field | test passes | done
T-0143 | 2 | econ | api_cost.py: per-model price table from pricing.json | test passes | done
T-0144 | 2 | infra | tests/integration/test_ledger_roundtrip.py — write→read→validate | test passes | done
T-0145 | 2 | infra | shared/__init__ exports + public API surface frozen | imports clean | done
T-0146 | 2 | qa | mypy/pyright type-check pass on Phase 2 (if configured) | type-check clean | done
T-0147 | 2 | hw | spec.py: graceful CPU-only fallback (no CUDA) path + test | test passes | done
T-0148 | 2 | econ | opex.py: PUE / cooling overhead factor (optional) + test | test passes | done
T-0149 | 2 | econ | cloud_gpu.py: spot vs on-demand price variants + test | test passes | done
T-0150 | 2 | ledger | Verify all Phase 2 results write to results/*.json deterministically | ledger deterministic | done

## Phase 3 — Kaggle notebook + model download + AirLLM sharding scripts

T-0151 | 3 | nb | experiments/airllm_benchmark.ipynb — notebook skeleton (markdown sections) | notebook opens | done
T-0152 | 3 | nb | Notebook cell: enable GPU + assert torch.cuda.is_available() | cell drafted | done
T-0153 | 3 | nb | Notebook cell: print hardware via nvidia-smi + capture to ledger | cell drafted | done
T-0154 | 3 | nb | Notebook cell: pip install airllm bitsandbytes accelerate (pinned) | cell drafted | done
T-0155 | 3 | nb | Notebook cell: load HF_TOKEN from Kaggle secrets (never hardcoded) | cell drafted | done
T-0156 | 3 | dl | tests/unit/test_downloader.py — failing test for model download wrapper (red) | test fails | done
T-0157 | 3 | dl | src/cosmos77_ex05/download/fetch.py — snapshot_download wrapper (mocked) (green) | test passes | done
T-0158 | 3 | dl | fetch.py: resume/retry on partial download + test | test passes | done
T-0159 | 3 | dl | fetch.py: verify Qwen2.5-14B-Instruct repo id + revision pin | test passes | done
T-0160 | 3 | dl | fetch.py: refactor for ≤150 lines + docstrings + type hints | line-cap 0; ruff clean | done
T-0161 | 3 | dl | tests/unit/test_disk_guard.py — failing test for free-disk precheck (red) | test fails | done
T-0162 | 3 | dl | src/cosmos77_ex05/download/disk_guard.py — assert enough disk before download (green) | test passes | done
T-0163 | 3 | dl | disk_guard.py: refactor + docstrings + type hints | ruff clean | done
T-0164 | 3 | shard | tests/unit/test_sharder.py — failing test for per-layer SafeTensors split (red) | test fails | done
T-0165 | 3 | shard | src/cosmos77_ex05/shard/splitter.py — split checkpoint into per-layer shards (mocked) (green) | test passes | done
T-0166 | 3 | shard | splitter.py: name shards layer_NNN.safetensors (layer = page) | test passes | done
T-0167 | 3 | shard | splitter.py: write shard manifest JSON (layer→file→bytes) | test passes | done
T-0168 | 3 | shard | splitter.py: verify shard count == model layer count | test passes | done
T-0169 | 3 | shard | splitter.py: refactor for ≤150 lines + docstrings + type hints | line-cap 0; ruff clean | done
T-0170 | 3 | shard | tests/unit/test_manifest.py — failing test for manifest schema (red) | test fails | done
T-0171 | 3 | shard | src/cosmos77_ex05/shard/manifest.py — manifest model + loader (green) | test passes | done
T-0172 | 3 | shard | manifest.py: refactor + docstrings + type hints | ruff clean | done
T-0173 | 3 | shard | tests/unit/test_mmap_loader.py — failing test for zero-copy mmap layer load (red) | test fails | done
T-0174 | 3 | shard | src/cosmos77_ex05/shard/mmap_loader.py — mmap a layer shard (mocked) (green) | test passes | done
T-0175 | 3 | shard | mmap_loader.py: evict-after-use (page replacement) hook + test | test passes | done
T-0176 | 3 | shard | mmap_loader.py: refactor + docstrings + type hints | ruff clean | done
T-0177 | 3 | nb | Notebook cell: call download.fetch (resumable) for Qwen2.5-14B | cell drafted | done
T-0178 | 3 | nb | Notebook cell: call shard.splitter to build per-layer shards | cell drafted | done
T-0179 | 3 | nb | Notebook cell: print shard manifest summary (count, total GB) | cell drafted | done
T-0180 | 3 | nb | Notebook cell: prompt-set definition (fixed prompts, seeded) | cell drafted | done
T-0181 | 3 | nb | Notebook markdown: explain layer=page / page-fault analogy inline | markdown present | done
T-0182 | 3 | exp | experiments/SETUP.md — reproducible Kaggle run instructions (GPU on, secret, run all) | doc present | done
T-0183 | 3 | exp | experiments/SETUP.md — Colab T4 documented fallback | fallback present | done
T-0184 | 3 | exp | experiments/SETUP.md — pinned package versions table | table present | done
T-0185 | 3 | dl | Wire download + shard into SDK facade prepare_model() | SDK call works | done
T-0186 | 3 | qa | Run gate: ruff/format/line-cap zero on Phase 3 modules | gate green | done
T-0187 | 3 | qa | Run gate: pytest coverage ≥85% on download/shard | coverage ≥85% | done
T-0188 | 3 | nb | Notebook: nbstripout / clear outputs before commit (no secrets) | outputs clean | done
T-0189 | 3 | nb | Notebook: parameterize scenarios from config/setup.json | cell drafted | done
T-0190 | 3 | shard | splitter.py: handle tied embeddings / shared weights correctly + test | test passes | done
T-0191 | 3 | shard | mmap_loader.py: dtype-aware load (FP16) + test | test passes | done
T-0192 | 3 | dl | fetch.py: allow_patterns to skip unused files (save disk) + test | test passes | done
T-0193 | 3 | dl | disk_guard.py: distinguish /kaggle/working vs /tmp quota + test | test passes | done
T-0194 | 3 | shard | manifest.py: checksum per shard for integrity + test | test passes | done
T-0195 | 3 | nb | Notebook cell: scenario loop scaffold (none/8bit/4bit) placeholder | cell drafted | done
T-0196 | 3 | report | reports/SHARDING.md — how layer shards map to pages | report present | done
T-0197 | 3 | docs | docs/prompts/003_phase3_notebook.md prompt log | file present | done
T-0198 | 3 | ledger | Update docs/TODO.md statuses for completed Phase 3 tasks | statuses current | done
T-0199 | 3 | git | Commit Phase 3 notebook + sharding scripts (conventional commit) | committed; CI green | done
T-0200 | 3 | exp | experiments/SETUP.md — troubleshooting (download stall, disk full, quota) | section present | done

## Phase 4 — FP16 baseline OOM runner

T-0201 | 4 | base | tests/unit/test_fp16_runner.py — failing test for naive FP16 load (red) | test fails | done
T-0202 | 4 | base | src/cosmos77_ex05/baseline/fp16_runner.py — naive from_pretrained FP16 (mocked) (green) | test passes | done
T-0203 | 4 | base | fp16_runner.py: device_map none → full resident allocation path | test passes | done
T-0204 | 4 | base | fp16_runner.py: refactor for ≤150 lines + docstrings + type hints | line-cap 0; ruff clean | done
T-0205 | 4 | base | tests/unit/test_oom_capture.py — failing test for OutOfMemoryError capture (red) | test fails | done
T-0206 | 4 | base | src/cosmos77_ex05/baseline/oom_capture.py — try/except torch.cuda.OutOfMemoryError (green) | test passes | done
T-0207 | 4 | base | oom_capture.py: record requested vs available VRAM bytes | test passes | done
T-0208 | 4 | base | oom_capture.py: classify failure (OOM vs unbearable slowness) | test passes | done
T-0209 | 4 | base | oom_capture.py: refactor + docstrings + type hints | ruff clean | done
T-0210 | 4 | base | tests/unit/test_baseline_ledger.py — failing test for fp16_baseline.json schema (red) | test fails | done
T-0211 | 4 | base | fp16_runner: write results/fp16_baseline.json via gatekeeper (green) | test passes | done
T-0212 | 4 | base | baseline ledger: include param→memory prediction (29.4 GB) field | test passes | done
T-0213 | 4 | base | baseline ledger: include captured error string + VRAM numbers | test passes | done
T-0214 | 4 | base | baseline ledger: bottleneck label = "memory (VRAM capacity)" | test passes | done
T-0215 | 4 | base | tests/unit/test_vram_probe.py — failing test for VRAM probe (red) | test fails | done
T-0216 | 4 | base | src/cosmos77_ex05/baseline/vram_probe.py — read total/free/used VRAM (mocked) (green) | test passes | done
T-0217 | 4 | base | vram_probe.py: peak VRAM tracker via max_memory_allocated (mocked) | test passes | done
T-0218 | 4 | base | vram_probe.py: refactor + docstrings + type hints | ruff clean | done
T-0219 | 4 | nb | Notebook cell: run FP16 baseline, expect OOM, screenshot the traceback | cell drafted | done
T-0220 | 4 | nb | Notebook cell: save OOM screenshot to figures/oom_screenshot.png | cell drafted | done
T-0221 | 4 | nb | Notebook markdown: explain why 14B FP16 OOMs a 16 GB T4 (param math) | markdown present | done
T-0222 | 4 | base | Wire fp16_runner into SDK facade run_baseline() | SDK call works | done
T-0223 | 4 | report | reports/baseline.md — narrate predicted vs captured OOM | report present | done
T-0224 | 4 | report | reports/baseline.md — tie to RQ-a (memory not compute) | section present | done
T-0225 | 4 | report | reports/baseline.md — embed OOM screenshot reference | reference present | done
T-0226 | 4 | qa | Run gate: ruff/format/line-cap zero on Phase 4 modules | gate green | done
T-0227 | 4 | qa | Run gate: pytest coverage ≥85% on baseline | coverage ≥85% | done
T-0228 | 4 | base | oom_capture.py: handle CPU-offload "fits but 0.x tok/s" slow path + test | test passes | done
T-0229 | 4 | base | fp16_runner.py: time-boxed abort if load exceeds budget + test | test passes | done
T-0230 | 4 | base | baseline ledger: runtime-to-failure field + test | test passes | done
T-0231 | 4 | base | vram_probe.py: reset_peak_memory_stats between probes + test | test passes | done
T-0232 | 4 | base | tests/unit/test_baseline_fixture.py — fixture results/fp16_baseline.sample.json | fixture loads | done
T-0233 | 4 | base | oom_capture.py: redact any path/token from error string + test | test passes | done
T-0234 | 4 | base | fp16_runner.py: deterministic prompt + seed for reproducibility + test | test passes | done
T-0235 | 4 | report | reports/baseline.md — requested-vs-available VRAM table | table present | done
T-0236 | 4 | report | reports/baseline.md — link to model_math 29.4 GB derivation | link present | done
T-0237 | 4 | nb | Notebook cell: free VRAM (del model; empty_cache) after baseline | cell drafted | done
T-0238 | 4 | base | baseline: assert OOM is the expected/honest outcome (negative result valid) | test passes | done
T-0239 | 4 | base | baseline/__init__ exports + public surface | imports clean | done
T-0240 | 4 | docs | docs/prompts/004_phase4_baseline.md prompt log | file present | done
T-0241 | 4 | ledger | Update docs/TODO.md statuses for completed Phase 4 tasks | statuses current | done
T-0242 | 4 | git | Commit Phase 4 baseline runner (conventional commit) | committed; CI green | done
T-0243 | 4 | base | tests/integration/test_baseline_pipeline.py — runner→ledger end-to-end (mocked) | test passes | done
T-0244 | 4 | base | vram_probe.py: cross-check probe vs nvidia-smi parse + test | test passes | done
T-0245 | 4 | base | oom_capture.py: structured failure taxonomy enum + test | test passes | done

## Phase 5 — AirLLM runner (layer = page)

T-0246 | 5 | air | tests/unit/test_airllm_runner.py — failing test for AirLLM model wrapper (red) | test fails | done
T-0247 | 5 | air | src/cosmos77_ex05/airllm/runner.py — AirLLMLlama2/AutoModel wrapper (mocked) (green) | test passes | done
T-0248 | 5 | air | runner.py: layer-by-layer streaming load (one layer resident) | test passes | done
T-0249 | 5 | air | runner.py: refactor for ≤150 lines + docstrings + type hints | line-cap 0; ruff clean | done
T-0250 | 5 | air | tests/unit/test_layer_pager.py — failing test for layer pager (red) | test fails | done
T-0251 | 5 | air | src/cosmos77_ex05/airllm/layer_pager.py — load→use→evict per layer (green) | test passes | done
T-0252 | 5 | air | layer_pager.py: count page faults (one per layer per step) | test passes | done
T-0253 | 5 | air | layer_pager.py: mmap zero-copy mapping path + test | test passes | done
T-0254 | 5 | air | layer_pager.py: refactor + docstrings + type hints | ruff clean | done
T-0255 | 5 | air | tests/unit/test_airllm_generate.py — failing test for generate loop (red) | test fails | done
T-0256 | 5 | air | src/cosmos77_ex05/airllm/generate.py — prefill + decode loop (mocked) (green) | test passes | done
T-0257 | 5 | air | generate.py: prefill pass (GEMM) yields first token → TTFT hook | test passes | done
T-0258 | 5 | air | generate.py: decode loop (GEMV) yields per-token → TPOT hook | test passes | done
T-0259 | 5 | air | generate.py: KV-cache handling across decode steps + test | test passes | done
T-0260 | 5 | air | generate.py: refactor for ≤150 lines + docstrings + type hints | line-cap 0; ruff clean | done
T-0261 | 5 | air | tests/unit/test_airllm_ledger.py — failing test for airllm_none.json (red) | test fails | done
T-0262 | 5 | air | runner: write results/airllm_none.json via gatekeeper (green) | test passes | done
T-0263 | 5 | air | airllm ledger: TTFT/TPOT/throughput/peak_vram/peak_ram/runtime/power/quality | test passes | done
T-0264 | 5 | air | airllm ledger: page-fault count + per-layer load time field | test passes | done
T-0265 | 5 | air | airllm ledger: output text + quality tag (same prompt as baseline) | test passes | done
T-0266 | 5 | air | tests/unit/test_peak_ram.py — failing test for peak RAM tracker (red) | test fails | done
T-0267 | 5 | air | src/cosmos77_ex05/airllm/ram_probe.py — peak RSS via psutil (mocked) (green) | test passes | done
T-0268 | 5 | air | ram_probe.py: refactor + docstrings + type hints | ruff clean | done
T-0269 | 5 | nb | Notebook cell: run AirLLM (quant=none) on the SAME prompt/model | cell drafted | done
T-0270 | 5 | nb | Notebook cell: capture peak VRAM falls to ~single-layer size | cell drafted | done
T-0271 | 5 | nb | Notebook markdown: explain VRAM-for-time trade, page-fault per layer | markdown present | done
T-0272 | 5 | air | Wire airllm.runner into SDK facade run_airllm(quant=none) | SDK call works | done
T-0273 | 5 | report | reports/airllm.md — narrate the SAME model now runs (RQ-b) | report present | done
T-0274 | 5 | report | reports/airllm.md — Paging connection table (layer=page, load=fault, mmap=zero-copy) | table present | done
T-0275 | 5 | report | reports/airllm.md — peak VRAM impossible→single-layer | section present | done
T-0276 | 5 | qa | Run gate: ruff/format/line-cap zero on Phase 5 modules | gate green | done
T-0277 | 5 | qa | Run gate: pytest coverage ≥85% on airllm | coverage ≥85% | done
T-0278 | 5 | air | generate.py: max_new_tokens + stop criteria from config + test | test passes | done
T-0279 | 5 | air | layer_pager.py: prefetch-next-layer option (overlap I/O) + test | test passes | done
T-0280 | 5 | air | runner.py: graceful handling if shard missing (manifest mismatch) + test | test passes | done
T-0281 | 5 | air | generate.py: separate prefill-time vs decode-time accounting + test | test passes | done
T-0282 | 5 | air | ram_probe.py: peak RAM includes mmap pages note + test | test passes | done
T-0283 | 5 | air | tests/unit/test_airllm_fixture.py — fixture results/airllm_none.sample.json | fixture loads | done
T-0284 | 5 | air | runner.py: disk-bandwidth log (GB/s observed per layer) + test | test passes | done
T-0285 | 5 | report | reports/airllm.md — why both TTFT and TPOT inflate (every layer a fault) | section present | done
T-0286 | 5 | report | reports/airllm.md — disk BW ≪ HBM BW explains ~1-3 tok/s | section present | done
T-0287 | 5 | nb | Notebook cell: free VRAM/RAM after AirLLM run | cell drafted | done
T-0288 | 5 | air | airllm/__init__ exports + public surface | imports clean | done
T-0289 | 5 | air | tests/integration/test_airllm_pipeline.py — runner→generate→ledger (mocked) | test passes | done
T-0290 | 5 | air | generate.py: token-by-token timestamp list for ITL distribution + test | test passes | done
T-0291 | 5 | air | layer_pager.py: evict policy = immediate (no cache) documented + test | test passes | done
T-0292 | 5 | air | runner.py: compression=none vs 8bit/4bit dispatch hook + test | test passes | done
T-0293 | 5 | air | airllm ledger: scenario id "none" + model id + prompt hash | test passes | done
T-0294 | 5 | docs | docs/prompts/005_phase5_airllm.md prompt log | file present | done
T-0295 | 5 | ledger | Update docs/TODO.md statuses for completed Phase 5 tasks | statuses current | done

## Phase 6 — Quantization sweep FP16 / Q8 / Q4

T-0296 | 6 | quant | tests/unit/test_quant_config.py — failing test for BitsAndBytesConfig builder (red) | test fails | done
T-0297 | 6 | quant | src/cosmos77_ex05/quant/config.py — build 8bit/4bit configs (green) | test passes | done
T-0298 | 6 | quant | config.py: 4bit NF4 + double-quant + compute dtype FP16 | test passes | done
T-0299 | 6 | quant | config.py: refactor + docstrings + type hints | ruff clean | done
T-0300 | 6 | quant | tests/unit/test_quant_runner.py — failing test for quantized AirLLM run (red) | test fails | done
T-0301 | 6 | quant | src/cosmos77_ex05/quant/runner.py — run AirLLM with compression=8bit (green) | test passes | done
T-0302 | 6 | quant | runner.py: run AirLLM with compression=4bit path | test passes | done
T-0303 | 6 | quant | runner.py: refactor for ≤150 lines + docstrings + type hints | line-cap 0; ruff clean | done
T-0304 | 6 | quant | tests/unit/test_quant_memory.py — failing test for memory drop ×2/×4 (red) | test fails | done
T-0305 | 6 | quant | src/cosmos77_ex05/quant/memory.py — expected vs measured VRAM per level (green) | test passes | done
T-0306 | 6 | quant | memory.py: Q8 ≈14.7 GB, Q4 ≈7.4 GB predictions + test | test passes | done
T-0307 | 6 | quant | memory.py: refactor + docstrings + type hints | ruff clean | done
T-0308 | 6 | quant | tests/unit/test_quant_ledger.py — failing test for airllm_8bit/4bit.json (red) | test fails | done
T-0309 | 6 | quant | runner: write results/airllm_8bit.json via gatekeeper (green) | test passes | done
T-0310 | 6 | quant | runner: write results/airllm_4bit.json via gatekeeper | test passes | done
T-0311 | 6 | quant | quant ledger: per-level memory/speed/quality fields | test passes | done
T-0312 | 6 | quant | tests/unit/test_red_line.py — failing test for accuracy red-line detector (red) | test fails | done
T-0313 | 6 | quant | src/cosmos77_ex05/quant/red_line.py — flag level where output visibly degrades (green) | test passes | done
T-0314 | 6 | quant | red_line.py: compare outputs FP16 vs Q8 vs Q4 on same prompt | test passes | done
T-0315 | 6 | quant | red_line.py: refactor + docstrings + type hints | ruff clean | done
T-0316 | 6 | quant | tests/unit/test_quant_quality.py — failing test for quality comparison (red) | test fails | done
T-0317 | 6 | quant | src/cosmos77_ex05/quant/quality_diff.py — qualitative diff per level (green) | test passes | done
T-0318 | 6 | quant | quality_diff.py: refactor + docstrings + type hints | ruff clean | done
T-0319 | 6 | nb | Notebook cell: sweep loop none→8bit→4bit (free VRAM between) | cell drafted | done
T-0320 | 6 | nb | Notebook cell: print same-prompt outputs side by side per level | cell drafted | done
T-0321 | 6 | nb | Notebook markdown: FP32→FP16 cheap, Q8 near-lossless, Q4 aggressive, NF4 default | markdown present | done
T-0322 | 6 | quant | Wire quant.runner into SDK facade run_quant_sweep() | SDK call works | done
T-0323 | 6 | report | reports/quantization.md — memory/speed/quality table per level | table present | done
T-0324 | 6 | report | reports/quantization.md — accuracy red line identified (RQ-c) | section present | done
T-0325 | 6 | report | reports/quantization.md — NF4 (QLoRA) note + Q2 smoke-test-only note | section present | done
T-0326 | 6 | qa | Run gate: ruff/format/line-cap zero on Phase 6 modules | gate green | done
T-0327 | 6 | qa | Run gate: pytest coverage ≥85% on quant | coverage ≥85% | done
T-0328 | 6 | quant | config.py: validate bitsandbytes is CUDA-only (skip on CPU) + test | test passes | done
T-0329 | 6 | quant | runner.py: per-level peak VRAM + peak RAM capture + test | test passes | done
T-0330 | 6 | quant | runner.py: per-level TTFT/TPOT/throughput capture + test | test passes | done
T-0331 | 6 | quant | memory.py: measured-vs-predicted delta report + test | test passes | done
T-0332 | 6 | quant | red_line.py: tag Q4 "usable", Q2 "smoke-test only" + test | test passes | done
T-0333 | 6 | quant | tests/unit/test_quant_fixture.py — fixtures airllm_8bit/4bit.sample.json | fixtures load | done
T-0334 | 6 | quant | quality_diff.py: factual-error spotting heuristic + test | test passes | done
T-0335 | 6 | report | reports/quantization.md — ×2/×4 memory drop vs FP16 table | table present | done
T-0336 | 6 | report | reports/quantization.md — speed change per level discussion | section present | done
T-0337 | 6 | nb | Notebook cell: free VRAM after each quant level + assert | cell drafted | done
T-0338 | 6 | quant | quant/__init__ exports + public surface | imports clean | done
T-0339 | 6 | quant | tests/integration/test_quant_sweep_pipeline.py — sweep→ledger (mocked) | test passes | done
T-0340 | 6 | quant | runner.py: scenario ids none/8bit/4bit consistent with config | test passes | done
T-0341 | 6 | quant | config.py: load quant levels from config/setup.json + test | test passes | done
T-0342 | 6 | quant | memory.py: bytes_per_param table sourced from constants + test | test passes | done
T-0343 | 6 | quant | red_line.py: deterministic on fixed prompts (no flakes) + test | test passes | done
T-0344 | 6 | docs | docs/prompts/006_phase6_quant.md prompt log | file present | done
T-0345 | 6 | ledger | Update docs/TODO.md statuses for completed Phase 6 tasks | statuses current | done
T-0346 | 6 | git | Commit Phase 6 quantization sweep (conventional commit) | committed; CI green | done
T-0347 | 6 | quant | runner.py: time-box each level run + abort guard + test | test passes | done
T-0348 | 6 | quant | quality_diff.py: store reference (FP16) output as quality baseline + test | test passes | done
T-0349 | 6 | quant | memory.py: KV-cache memory unchanged-by-quant note + test | test passes | done
T-0350 | 6 | quant | red_line.py: report which level crosses the red line in ledger | test passes | done

## Phase 7 — Measurement harness + tables + graphs + Roofline

T-0351 | 7 | meas | tests/unit/test_ttft.py — failing test for TTFT measurement (red) | test fails | done
T-0352 | 7 | meas | src/cosmos77_ex05/measure/ttft.py — time-to-first-token from prefill (green) | test passes | done
T-0353 | 7 | meas | ttft.py: refactor + docstrings + type hints | ruff clean | done
T-0354 | 7 | meas | tests/unit/test_tpot.py — failing test for TPOT/ITL measurement (red) | test fails | done
T-0355 | 7 | meas | src/cosmos77_ex05/measure/tpot.py — mean inter-token latency from decode (green) | test passes | done
T-0356 | 7 | meas | tpot.py: ITL distribution (p50/p95) helper + test | test passes | done
T-0357 | 7 | meas | tpot.py: refactor + docstrings + type hints | ruff clean | done
T-0358 | 7 | meas | tests/unit/test_throughput.py — failing test for tokens/sec (red) | test fails | done
T-0359 | 7 | meas | src/cosmos77_ex05/measure/throughput.py — tokens / total_decode_time (green) | test passes | done
T-0360 | 7 | meas | throughput.py: refactor + docstrings + type hints | ruff clean | done
T-0361 | 7 | meas | tests/unit/test_peakmem.py — failing test for peak VRAM+RAM aggregation (red) | test fails | done
T-0362 | 7 | meas | src/cosmos77_ex05/measure/peakmem.py — peak VRAM + peak RAM collation (green) | test passes | done
T-0363 | 7 | meas | peakmem.py: refactor + docstrings + type hints | ruff clean | done
T-0364 | 7 | meas | tests/unit/test_runtime.py — failing test for total runtime + est. power (red) | test fails | done
T-0365 | 7 | meas | src/cosmos77_ex05/measure/runtime.py — total runtime + watts×time energy (green) | test passes | done
T-0366 | 7 | meas | runtime.py: refactor + docstrings + type hints | ruff clean | done
T-0367 | 7 | meas | tests/unit/test_metric_aggregate.py — failing test for per-scenario rollup (red) | test fails | done
T-0368 | 7 | meas | src/cosmos77_ex05/measure/aggregate.py — load all results/*.json → table rows (green) | test passes | done
T-0369 | 7 | meas | aggregate.py: refactor for ≤150 lines + docstrings + type hints | line-cap 0; ruff clean | done
T-0370 | 7 | meas | measure/__init__ exports + public surface | imports clean | done
T-0371 | 7 | tables | tests/unit/test_metrics_table.py — failing test for METRICS.md generator (red) | test fails | done
T-0372 | 7 | tables | src/cosmos77_ex05/report/metrics_table.py — render reports/METRICS.md from ledger (green) | test passes | done
T-0373 | 7 | tables | metrics_table.py: columns TTFT/TPOT/tok-s/peakVRAM/peakRAM/runtime/power/quality | test passes | done
T-0374 | 7 | tables | metrics_table.py: refactor + docstrings + type hints | ruff clean | done
T-0375 | 7 | plot | tests/unit/test_plot_tokens.py — failing test for tokens_per_sec figure (red) | test fails | done
T-0376 | 7 | plot | src/cosmos77_ex05/plots/tokens_per_sec.py — bar chart per scenario (green) | test passes | done
T-0377 | 7 | plot | tokens_per_sec.py: save figures/tokens_per_sec.png deterministically | test passes | done
T-0378 | 7 | plot | tokens_per_sec.py: refactor + docstrings + type hints | ruff clean | done
T-0379 | 7 | plot | tests/unit/test_plot_vram.py — failing test for peak_vram figure (red) | test fails | done
T-0380 | 7 | plot | src/cosmos77_ex05/plots/peak_vram.py — bar chart + 16 GB T4 line (green) | test passes | done
T-0381 | 7 | plot | peak_vram.py: save figures/peak_vram.png; mark OOM baseline | test passes | done
T-0382 | 7 | plot | peak_vram.py: refactor + docstrings + type hints | ruff clean | done
T-0383 | 7 | plot | tests/unit/test_plot_ttft_tpot.py — failing test for ttft_vs_tpot figure (red) | test fails | done
T-0384 | 7 | plot | src/cosmos77_ex05/plots/ttft_vs_tpot.py — grouped bars TTFT vs TPOT (green) | test passes | done
T-0385 | 7 | plot | ttft_vs_tpot.py: save figures/ttft_vs_tpot.png | test passes | done
T-0386 | 7 | plot | ttft_vs_tpot.py: refactor + docstrings + type hints | ruff clean | done
T-0387 | 7 | plot | tests/unit/test_plot_quant.py — failing test for quant_tradeoff figure (red) | test fails | done
T-0388 | 7 | plot | src/cosmos77_ex05/plots/quant_tradeoff.py — memory vs quality scatter (green) | test passes | done
T-0389 | 7 | plot | quant_tradeoff.py: save figures/quant_tradeoff.png | test passes | done
T-0390 | 7 | plot | quant_tradeoff.py: refactor + docstrings + type hints | ruff clean | done
T-0391 | 7 | roof | tests/unit/test_roofline_math.py — failing test for arithmetic intensity (red) | test fails | done
T-0392 | 7 | roof | src/cosmos77_ex05/analysis/roofline.py — FLOPs/byte per phase (green) | test passes | done
T-0393 | 7 | roof | roofline.py: ridge point = peak_FLOPS / peak_BW (T4 ≈65T / ≈320 ⇒ ≈203) | test passes | done
T-0394 | 7 | roof | roofline.py: decode AI≈1 ⇒ memory-bound classification | test passes | done
T-0395 | 7 | roof | roofline.py: prefill higher AI ⇒ compute-bound classification | test passes | done
T-0396 | 7 | roof | roofline.py: refactor for ≤150 lines + docstrings + type hints | line-cap 0; ruff clean | done
T-0397 | 7 | roof | tests/unit/test_plot_roofline.py — failing test for roofline figure (red) | test fails | done
T-0398 | 7 | roof | src/cosmos77_ex05/plots/roofline_plot.py — log-log roofline w/ ridge + points (green) | test passes | done
T-0399 | 7 | roof | roofline_plot.py: plot prefill & decode operating points | test passes | done
T-0400 | 7 | roof | roofline_plot.py: save figures/roofline.png | test passes | done
T-0401 | 7 | roof | roofline_plot.py: refactor + docstrings + type hints | ruff clean | done
T-0402 | 7 | meas | Wire measure+plots into SDK facade build_metrics() | SDK call works | done
T-0403 | 7 | report | reports/METRICS.md — narrative around the generated table | report present | done
T-0404 | 7 | report | reports/METRICS.md — embed all four metric figures | figures embedded | done
T-0405 | 7 | qa | Run gate: ruff/format/line-cap zero on Phase 7 modules | gate green | done
T-0406 | 7 | qa | Run gate: pytest coverage ≥85% on measure/plots/analysis | coverage ≥85% | done
T-0407 | 7 | plot | All plots: fixed matplotlib backend (Agg) + seeded layout (no flakes) | tests deterministic | done
T-0408 | 7 | plot | All plots: read ONLY from results/*.json (single source of truth) | tests assert source | done
T-0409 | 7 | plot | tokens_per_sec.py: annotate ~1-3 tok/s AirLLM regime | test passes | done
T-0410 | 7 | plot | peak_vram.py: annotate impossible→single-layer drop | test passes | done
T-0411 | 7 | plot | ttft_vs_tpot.py: annotate prefill=compute / decode=memory | test passes | done
T-0412 | 7 | plot | quant_tradeoff.py: annotate red-line crossing point | test passes | done
T-0413 | 7 | roof | roofline.py: KV-cache bytes included in decode AI denominator + test | test passes | done
T-0414 | 7 | roof | roofline_plot.py: shade memory-bound vs compute-bound regions + test | test passes | done
T-0415 | 7 | meas | aggregate.py: fail loudly if a scenario ledger missing + test | test passes | done
T-0416 | 7 | meas | aggregate.py: round/format units consistently (ms, GB, tok/s) + test | test passes | done
T-0417 | 7 | tables | metrics_table.py: mark OOM baseline row distinctly + test | test passes | done
T-0418 | 7 | tables | metrics_table.py: emit machine-readable reports/metrics.csv too + test | test passes | done
T-0419 | 7 | plot | tests/unit/test_plot_fixtures.py — all plots run on sample ledgers | tests pass | done
T-0420 | 7 | roof | reports/ROOFLINE.md — explain ridge ≈203, AI per phase | report present | done
T-0421 | 7 | meas | tests/integration/test_metrics_pipeline.py — ledger→table→figures (mocked) | test passes | done
T-0422 | 7 | plot | plots/__init__ exports + public surface | imports clean | done
T-0423 | 7 | meas | measure/runtime.py: power from config watts (T4 ~70W) + test | test passes | done
T-0424 | 7 | tables | metrics_table.py: numbers byte-match ledger (no rounding drift) + test | test passes | done
T-0425 | 7 | roof | roofline.py: model FLOPs estimate (2·params·tokens) + test | test passes | done
T-0426 | 7 | plot | Figure DPI/size standardized across all plots + test | tests pass | done
T-0427 | 7 | report | reports/METRICS.md — tie each metric to a lecture concept | section present | done
T-0428 | 7 | docs | docs/prompts/007_phase7_metrics.md prompt log | file present | done
T-0429 | 7 | ledger | Update docs/TODO.md statuses for completed Phase 7 tasks | statuses current | done
T-0430 | 7 | git | Commit Phase 7 measurement harness + figures (conventional commit) | committed; CI green | done

## Phase 8 — Economic break-even (On-Prem vs API vs Cloud GPU)

T-0431 | 8 | econ | tests/unit/test_workload_model.py — failing test for token-volume workload (red) | test fails | done
T-0432 | 8 | econ | src/cosmos77_ex05/economics/workload.py — tokens/day → annual volume (green) | test passes | done
T-0433 | 8 | econ | workload.py: refactor + docstrings + type hints | ruff clean | done
T-0434 | 8 | econ | tests/unit/test_onprem_curve.py — failing test for On-Prem cumulative cost (red) | test fails | done
T-0435 | 8 | econ | src/cosmos77_ex05/economics/onprem_curve.py — CAPEX amortised + OPEX over volume (green) | test passes | done
T-0436 | 8 | econ | onprem_curve.py: include electricity from measured power + runtime | test passes | done
T-0437 | 8 | econ | onprem_curve.py: refactor + docstrings + type hints | ruff clean | done
T-0438 | 8 | econ | tests/unit/test_api_curve.py — failing test for API cumulative cost (red) | test fails | done
T-0439 | 8 | econ | src/cosmos77_ex05/economics/api_curve.py — tokens × price over volume (green) | test passes | done
T-0440 | 8 | econ | api_curve.py: split input vs output token pricing | test passes | done
T-0441 | 8 | econ | api_curve.py: refactor + docstrings + type hints | ruff clean | done
T-0442 | 8 | econ | tests/unit/test_cached_api_curve.py — failing test for cached API line (red) | test fails | done
T-0443 | 8 | econ | src/cosmos77_ex05/economics/api_curve.py — prompt-caching discount variant (green) | test passes | done
T-0444 | 8 | econ | cached api: cache-write vs cache-read price tiers + test | test passes | done
T-0445 | 8 | econ | tests/unit/test_cloud_curve.py — failing test for cloud GPU cumulative cost (red) | test fails | done
T-0446 | 8 | econ | src/cosmos77_ex05/economics/cloud_curve.py — $/hr × hours over volume (green) | test passes | done
T-0447 | 8 | econ | cloud_curve.py: refactor + docstrings + type hints | ruff clean | done
T-0448 | 8 | econ | tests/unit/test_crossover.py — failing test for break-even volume solve (red) | test fails | done
T-0449 | 8 | econ | src/cosmos77_ex05/economics/crossover.py — numeric crossover On-Prem vs API (green) | test passes | done
T-0450 | 8 | econ | crossover.py: second crossover with caching-shifted API line | test passes | done
T-0451 | 8 | econ | crossover.py: handle no-crossover (always cheaper) case + test | test passes | done
T-0452 | 8 | econ | crossover.py: refactor for ≤150 lines + docstrings + type hints | line-cap 0; ruff clean | done
T-0453 | 8 | econ | tests/unit/test_assumptions.py — failing test for assumptions table (red) | test fails | done
T-0454 | 8 | econ | src/cosmos77_ex05/economics/assumptions.py — collect all stated assumptions (green) | test passes | done
T-0455 | 8 | econ | assumptions.py: prices, $/kWh, watts, amortisation, utilisation listed | test passes | done
T-0456 | 8 | econ | assumptions.py: refactor + docstrings + type hints | ruff clean | done
T-0457 | 8 | econ | write results/economics.json via gatekeeper (curves + crossover) | test passes | done
T-0458 | 8 | plot | tests/unit/test_breakeven_plot.py — failing test for breakeven figure (red) | test fails | done
T-0459 | 8 | plot | src/cosmos77_ex05/plots/breakeven_plot.py — cost-vs-volume lines + crossover marker (green) | test passes | done
T-0460 | 8 | plot | breakeven_plot.py: On-Prem, API, cached-API, Cloud-GPU lines | test passes | done
T-0461 | 8 | plot | breakeven_plot.py: annotate crossover volumes | test passes | done
T-0462 | 8 | plot | breakeven_plot.py: save figures/breakeven.png | test passes | done
T-0463 | 8 | plot | breakeven_plot.py: refactor + docstrings + type hints | ruff clean | done
T-0464 | 8 | econ | Wire economics curves into SDK facade run_breakeven() | SDK call works | done
T-0465 | 8 | report | reports/ECONOMICS.md — assumptions table (all stated) | table present | done
T-0466 | 8 | report | reports/ECONOMICS.md — break-even narrative + crossover volume (RQ-f) | section present | done
T-0467 | 8 | report | reports/ECONOMICS.md — prompt-caching shift of the API line | section present | done
T-0468 | 8 | report | reports/ECONOMICS.md — privacy/security-aware recommendation | section present | done
T-0469 | 8 | report | reports/ECONOMICS.md — embed figures/breakeven.png | figure embedded | done
T-0470 | 8 | qa | Run gate: ruff/format/line-cap zero on Phase 8 modules | gate green | done
T-0471 | 8 | qa | Run gate: pytest coverage ≥85% on economics curves | coverage ≥85% | done
T-0472 | 8 | econ | onprem_curve.py: amortisation window sensitivity (1/2/3 yr) + test | test passes | done
T-0473 | 8 | econ | api_curve.py: per-provider price variants from pricing.json + test | test passes | done
T-0474 | 8 | econ | cloud_curve.py: spot vs on-demand crossover variants + test | test passes | done
T-0475 | 8 | econ | crossover.py: report break-even in tokens AND in days + test | test passes | done
T-0476 | 8 | econ | assumptions.py: cite source/date for every price + test | test passes | done
T-0477 | 8 | plot | breakeven_plot.py: log-x option for wide volume range + test | test passes | done
T-0478 | 8 | econ | tests/integration/test_economics_pipeline.py — pricing→curves→ledger→figure | test passes | done
T-0479 | 8 | econ | sensitivity sweep: utilisation 10/50/90% effect on crossover + test | test passes | done
T-0480 | 8 | econ | sensitivity sweep: $/kWh low/mid/high effect + test | test passes | done
T-0481 | 8 | report | reports/ECONOMICS.md — sensitivity table | table present | done
T-0482 | 8 | econ | economics fixture results/economics.sample.json | fixture loads | done
T-0483 | 8 | econ | crossover.py: assert curves monotonic increasing + test | test passes | done
T-0484 | 8 | report | reports/ECONOMICS.md — when Cloud-GPU beats both note | section present | done
T-0485 | 8 | report | reports/ECONOMICS.md — feasibility vs cost honest caveat (1-3 tok/s) | section present | done
T-0486 | 8 | econ | workload.py: map measured throughput → hours to serve volume + test | test passes | done
T-0487 | 8 | econ | onprem_curve.py: include AirLLM slow-throughput penalty (more GPU-hours) + test | test passes | done
T-0488 | 8 | econ | api_curve.py: numbers byte-match results/economics.json + test | test passes | done
T-0489 | 8 | plot | breakeven_plot.py: deterministic colors/legend (no flakes) + test | test passes | done
T-0490 | 8 | econ | economics/__init__ exports + public surface | imports clean | done
T-0491 | 8 | report | reports/ECONOMICS.md — tie to lecture On-Prem vs API framing | section present | done
T-0492 | 8 | docs | docs/prompts/008_phase8_economics.md prompt log | file present | done
T-0493 | 8 | ledger | Update docs/TODO.md statuses for completed Phase 8 tasks | statuses current | done
T-0494 | 8 | git | Commit Phase 8 economics + breakeven figure (conventional commit) | committed; CI green | done
T-0495 | 8 | econ | crossover.py: confidence interval / range on crossover + test | test passes | done
T-0496 | 8 | report | reports/ECONOMICS.md — decision matrix (volume × privacy → choice) | matrix present | done
T-0497 | 8 | econ | cloud_curve.py: idle-time billing caveat + test | test passes | done
T-0498 | 8 | econ | api_curve.py: rate-limit / latency non-cost factor note + test | test passes | done
T-0499 | 8 | econ | assumptions.py: amortisation = straight-line documented + test | test passes | done
T-0500 | 8 | econ | verify results/economics.json reproduces figure deterministically | reproducible | done

## Phase 9 — Lecture-concept analysis + original extension

T-0501 | 9 | concept | reports/CONCEPTS.md — Prefill→TTFT (GEMM, compute-bound) section | section present | done
T-0502 | 9 | concept | reports/CONCEPTS.md — Decode→TPOT (GEMV, memory-bound) section | section present | done
T-0503 | 9 | concept | reports/CONCEPTS.md — compute-bound vs memory-bound diagnosis | section present | done
T-0504 | 9 | concept | reports/CONCEPTS.md — VRAM capacity vs bandwidth distinction | section present | done
T-0505 | 9 | concept | reports/CONCEPTS.md — Paging analogy (layer=page, fault, mmap, eviction) | section present | done
T-0506 | 9 | concept | reports/CONCEPTS.md — quantization (FP16/Q8/Q4/NF4) memory math | section present | done
T-0507 | 9 | concept | reports/CONCEPTS.md — KV-cache role in decode memory traffic | section present | done
T-0508 | 9 | concept | reports/CONCEPTS.md — Roofline ridge point + operating points | section present | done
T-0509 | 9 | concept | reports/CONCEPTS.md — disk BW ≪ HBM BW explains ~1-3 tok/s | section present | done
T-0510 | 9 | concept | reports/CONCEPTS.md — On-Prem vs API + prompt caching framing | section present | done
T-0511 | 9 | concept | tests/unit/test_concept_trace.py — failing test for concept→ledger traceability (red) | test fails | done
T-0512 | 9 | concept | src/cosmos77_ex05/analysis/concept_trace.py — map each concept→result number (green) | test passes | done
T-0513 | 9 | concept | concept_trace.py: assert every claim cites a ledger field | test passes | done
T-0514 | 9 | concept | concept_trace.py: refactor + docstrings + type hints | ruff clean | done
T-0515 | 9 | concept | reports/CONCEPTS.md — answer RQ-a explicitly w/ ledger citations | RQ-a answered | done
T-0516 | 9 | concept | reports/CONCEPTS.md — answer RQ-b explicitly w/ ledger citations | RQ-b answered | done
T-0517 | 9 | concept | reports/CONCEPTS.md — answer RQ-c explicitly w/ ledger citations | RQ-c answered | done
T-0518 | 9 | concept | reports/CONCEPTS.md — answer RQ-d explicitly w/ ledger citations | RQ-d answered | done
T-0519 | 9 | concept | reports/CONCEPTS.md — answer RQ-e explicitly w/ ledger citations | RQ-e answered | done
T-0520 | 9 | concept | reports/CONCEPTS.md — answer RQ-f explicitly w/ ledger citations | RQ-f answered | done
T-0521 | 9 | ext | docs/PRD_extension.md — choose extension (quant Pareto / QLoRA / three-way CPU-GPU-AirLLM) | choice fixed | done
T-0522 | 9 | ext | tests/unit/test_extension.py — failing test for extension core (red) | test fails | done
T-0523 | 9 | ext | src/cosmos77_ex05/extensions/__init__ + module skeleton (green) | test passes | done
T-0524 | 9 | ext | extensions: model-size sweep (1.5B/7B/14B) memory/speed comparison core | test passes | done
T-0525 | 9 | ext | model-size sweep: predicted vs measured VRAM per size + test | test passes | done
T-0526 | 9 | ext | model-size sweep: write results/extension_modelsize.json + test | test passes | done
T-0527 | 9 | ext | extensions: quant Pareto frontier (memory vs quality) core + test | test passes | done
T-0528 | 9 | ext | quant Pareto: identify Pareto-optimal level + test | test passes | done
T-0529 | 9 | ext | extensions: QLoRA adapter demo (NF4 base + LoRA) core (mocked) + test | test passes | done
T-0530 | 9 | ext | QLoRA demo: adapter params vs base params ratio + test | test passes | done
T-0531 | 9 | ext | extensions: three-way CPU-only vs GPU-resident vs AirLLM comparison + test | test passes | done
T-0532 | 9 | ext | three-way: throughput/peak-mem/feasibility table + test | test passes | done
T-0533 | 9 | ext | extensions: refactor each module ≤150 lines + docstrings + type hints | line-cap 0; ruff clean | done
T-0534 | 9 | plot | tests/unit/test_extension_plot.py — failing test for extension figure (red) | test fails | done
T-0535 | 9 | plot | src/cosmos77_ex05/plots/extension_plot.py — render extension figure (green) | test passes | done
T-0536 | 9 | plot | extension_plot.py: save figures/extension.png | test passes | done
T-0537 | 9 | plot | extension_plot.py: refactor + docstrings + type hints | ruff clean | done
T-0538 | 9 | ext | Wire extension into SDK facade run_extension() | SDK call works | done
T-0539 | 9 | report | reports/EXTENSIONS.md — extension motivation + method | report present | done
T-0540 | 9 | report | reports/EXTENSIONS.md — results table + figure embedded | table+figure present | done
T-0541 | 9 | report | reports/EXTENSIONS.md — tie extension to lecture concepts | section present | done
T-0542 | 9 | qa | Run gate: ruff/format/line-cap zero on Phase 9 modules | gate green | done
T-0543 | 9 | qa | Run gate: pytest coverage ≥85% on analysis/extensions/plots | coverage ≥85% | done
T-0544 | 9 | nb | Notebook cell: run extension experiment on T4 | cell drafted | done
T-0545 | 9 | nb | Notebook markdown: explain extension hypothesis + expected result | markdown present | done
T-0546 | 9 | ext | extension fixture results/extension.sample.json | fixture loads | done
T-0547 | 9 | concept | reports/CONCEPTS.md — Amdahl/bandwidth bound note for AirLLM | section present | done
T-0548 | 9 | concept | concept_trace.py: emit concept→ledger CSV map + test | test passes | done
T-0549 | 9 | ext | extensions/__init__ exports + public surface | imports clean | done
T-0550 | 9 | ext | tests/integration/test_extension_pipeline.py — extension→ledger→figure | test passes | done
T-0551 | 9 | report | reports/CONCEPTS.md — cross-reference every figure/table | references resolve | done
T-0552 | 9 | ext | model-size sweep: ridge-point comparison across sizes + test | test passes | done
T-0553 | 9 | ext | quant Pareto: knee-point detection + test | test passes | done
T-0554 | 9 | ext | three-way: state which is feasible on free T4 + test | test passes | done
T-0555 | 9 | concept | reports/CONCEPTS.md — honest negative-result framing (OOM is data) | section present | done
T-0556 | 9 | docs | docs/prompts/009_phase9_concepts.md prompt log | file present | done
T-0557 | 9 | ledger | Update docs/TODO.md statuses for completed Phase 9 tasks | statuses current | done
T-0558 | 9 | git | Commit Phase 9 concepts + extension (conventional commit) | committed; CI green | done
T-0559 | 9 | concept | reports/CONCEPTS.md — map each RQ→D-criterion→ledger field | map present | done
T-0560 | 9 | ext | extension: reproducibility note (seed, config) in report | note present | done

## Phase 10 — README as the deep technical report

T-0561 | 10 | readme | README.md §1 title + abstract (honest-measurement thesis) | section present | todo
T-0562 | 10 | readme | README.md §2 problem statement (29 GB on 16 GB T4) | section present | todo
T-0563 | 10 | readme | README.md §3 hardware + param→memory math table | section present | todo
T-0564 | 10 | readme | README.md §3 embed results/hardware.json numbers (byte-match) | numbers match | todo
T-0565 | 10 | readme | README.md §4 FP16 baseline OOM + embed oom_screenshot.png | section present | todo
T-0566 | 10 | readme | README.md §5 AirLLM layer=page + Paging table | section present | todo
T-0567 | 10 | readme | README.md §6 quantization sweep + accuracy red line | section present | todo
T-0568 | 10 | readme | README.md §7 metrics table (from METRICS.md, byte-match) | table present | todo
T-0569 | 10 | readme | README.md §7 embed tokens_per_sec.png | figure embedded | todo
T-0570 | 10 | readme | README.md §7 embed peak_vram.png | figure embedded | todo
T-0571 | 10 | readme | README.md §7 embed ttft_vs_tpot.png | figure embedded | todo
T-0572 | 10 | readme | README.md §7 embed quant_tradeoff.png | figure embedded | todo
T-0573 | 10 | readme | README.md §8 Roofline analysis + embed roofline.png | section present | todo
T-0574 | 10 | readme | README.md §9 economics break-even + embed breakeven.png | section present | todo
T-0575 | 10 | readme | README.md §10 lecture-concept linking (Prefill/Decode etc.) | section present | todo
T-0576 | 10 | readme | README.md §11 original extension + embed extension.png | section present | todo
T-0577 | 10 | readme | README.md §12 research questions RQ-a..RQ-f answered explicitly | section present | todo
T-0578 | 10 | readme | README.md §13 reproducibility (link experiments/SETUP.md) | section present | todo
T-0579 | 10 | readme | README.md §14 honest limitations + negative results | section present | todo
T-0580 | 10 | readme | README.md §15 conclusion (feasibility vs cost) | section present | todo
T-0581 | 10 | readme | README.md — CI badge + version + authors header | header present | todo
T-0582 | 10 | readme | tests/unit/test_readme_numbers.py — failing test: README numbers ⊆ ledger (red) | test fails | todo
T-0583 | 10 | readme | src/cosmos77_ex05/report/readme_check.py — verify README numbers vs ledger (green) | test passes | todo
T-0584 | 10 | readme | readme_check.py: parse README metric mentions + cross-check | test passes | todo
T-0585 | 10 | readme | readme_check.py: refactor + docstrings + type hints | ruff clean | todo
T-0586 | 10 | readme | README.md — assert ≥250 lines | line count ≥250 | todo
T-0587 | 10 | readme | README.md — assert ≥6 embedded figures | figures ≥6 | todo
T-0588 | 10 | readme | README.md — all figure paths resolve (figures/*.png present) | paths resolve | todo
T-0589 | 10 | readme | README.md — table of contents w/ anchors | TOC present | todo
T-0590 | 10 | readme | README.md — link to each report under reports/ | links resolve | todo
T-0591 | 10 | qa | Run gate: ruff/format/line-cap zero on Phase 10 modules | gate green | todo
T-0592 | 10 | qa | Run gate: pytest coverage ≥85% incl readme_check | coverage ≥85% | todo
T-0593 | 10 | readme | README.md — every claim cites ledger or lecture | citations present | todo
T-0594 | 10 | readme | README.md — proofread lecture vocabulary consistency | zero drift | todo
T-0595 | 10 | readme | Wire readme_check into SDK facade verify_report() | SDK call works | todo
T-0596 | 10 | readme | README.md — embed metrics.csv link for raw numbers | link present | todo
T-0597 | 10 | readme | README.md — D1–D15 acceptance checklist table | checklist present | todo
T-0598 | 10 | readme | README.md — KPI K-1..K-15 status table | table present | todo
T-0599 | 10 | readme | tests/integration/test_readme_pipeline.py — ledgers→README cross-check | test passes | todo
T-0600 | 10 | readme | README.md — screenshots gallery section | section present | todo
T-0601 | 10 | report | Update CHANGELOG.md [1.00] with report completion | changelog updated | todo
T-0602 | 10 | readme | README.md — author contribution statement | statement present | todo
T-0603 | 10 | readme | readme_check.py: fail if README cites a number absent from ledger + test | test passes | todo
T-0604 | 10 | docs | docs/prompts/010_phase10_readme.md prompt log | file present | todo
T-0605 | 10 | ledger | Update docs/TODO.md statuses for completed Phase 10 tasks | statuses current | todo

## Phase 11 — QA gauntlet + acceptance audit D1–D15

T-0606 | 11 | qa | tests/acceptance/test_d1_hardware.py — D1 hardware+math present (red→green) | test passes | todo
T-0607 | 11 | qa | tests/acceptance/test_d2_oom.py — D2 FP16 OOM captured | test passes | todo
T-0608 | 11 | qa | tests/acceptance/test_d3_airllm.py — D3 AirLLM runs SAME model | test passes | todo
T-0609 | 11 | qa | tests/acceptance/test_d4_quant.py — D4 quant sweep complete | test passes | todo
T-0610 | 11 | qa | tests/acceptance/test_d5_metrics.py — D5 full metric set per scenario | test passes | todo
T-0611 | 11 | qa | tests/acceptance/test_d6_tables_graphs.py — D6 tables+graphs from ledger | test passes | todo
T-0612 | 11 | qa | tests/acceptance/test_d7_economics.py — D7 breakeven+assumptions+caching | test passes | todo
T-0613 | 11 | qa | tests/acceptance/test_d8_concepts.py — D8 concept linking complete | test passes | todo
T-0614 | 11 | qa | tests/acceptance/test_d9_roofline.py — D9 roofline from measurements | test passes | todo
T-0615 | 11 | qa | tests/acceptance/test_d10_extension.py — D10 ≥1 extension present | test passes | todo
T-0616 | 11 | qa | tests/acceptance/test_d11_readme.py — D11 report-as-README ≥250 lines ≥6 figs | test passes | todo
T-0617 | 11 | qa | tests/acceptance/test_d12_structure.py — D12 repo structure present | test passes | todo
T-0618 | 11 | qa | tests/acceptance/test_d13_reproducible.py — D13 SETUP.md + no token in code | test passes | todo
T-0619 | 11 | qa | tests/acceptance/test_d14_rqs.py — D14 RQ-a..RQ-f answered | test passes | todo
T-0620 | 11 | qa | tests/acceptance/test_d15_honesty.py — D15 README numbers byte-match ledger | test passes | todo
T-0621 | 11 | qa | docs/ACCEPTANCE.md — D1–D15 audit results table | table present | todo
T-0622 | 11 | qa | docs/ACCEPTANCE.md — K-1..K-15 KPI evidence links | links present | todo
T-0623 | 11 | qa | Full ruff check across repo zero violations | zero violations | todo
T-0624 | 11 | qa | Full ruff format check zero diffs | zero diffs | todo
T-0625 | 11 | qa | scripts/check_line_cap.py across all .py zero over 150 | zero over cap | todo
T-0626 | 11 | qa | Full pytest suite green; coverage ≥85% overall | coverage ≥85% | todo
T-0627 | 11 | qa | grep secrets scan: no HF_TOKEN/API keys in tracked files | zero hits | todo
T-0628 | 11 | qa | Verify all results/*.json validate against ledger schema | all valid | todo
T-0629 | 11 | qa | Verify every figure regenerates deterministically from ledger | reproducible | todo
T-0630 | 11 | qa | Verify notebook outputs stripped + no secrets committed | clean | todo
T-0631 | 11 | qa | Verify .env.example only (no .env tracked) | confirmed | todo
T-0632 | 11 | qa | Verify CHANGELOG + CONTRIBUTING + LICENSE present and current | confirmed | todo
T-0633 | 11 | qa | Verify docs/TODO.md grep -c '^T-' ≥ 600 | count ≥600 | todo
T-0634 | 11 | qa | Verify all 8 mechanism PRDs cross-linked from PRD.md | links resolve | todo
T-0635 | 11 | qa | Tests never touch real I/O audit (all mocked) | confirmed | todo
T-0636 | 11 | qa | Determinism audit: seeded, no flaky tests over 3 runs | stable | todo
T-0637 | 11 | qa | CI workflow green on main (final run) | CI green | todo
T-0638 | 11 | qa | docs/ACCEPTANCE.md — honesty statement (no fabricated numbers) | statement present | todo
T-0639 | 11 | qa | Self-review against PRD §6 DoD gates K-1..K-15 | all gates pass | todo
T-0640 | 11 | qa | Address any reviewer findings from code-review pass | findings resolved | todo
T-0641 | 11 | docs | docs/prompts/011_phase11_qa.md prompt log | file present | todo
T-0642 | 11 | ledger | Update docs/TODO.md statuses for completed Phase 11 tasks | statuses current | todo
T-0643 | 11 | git | Commit Phase 11 acceptance audit (conventional commit) | committed; CI green | todo
T-0644 | 11 | qa | Final smoke: SDK end-to-end on sample ledgers reproduces all outputs | reproduces | todo
T-0645 | 11 | qa | Verify line cap, coverage, and acceptance all green together | all green | todo

## Phase 12 — Cover PDF + tag v1.00 + release + Moodle

T-0646 | 12 | cover | scripts/generate_cover_pdf.py — set exercise=5, course 203.3763 | exercise=5 set | todo
T-0647 | 12 | cover | Cover PDF: authors (Abdallah Khaldi 212389712, Tasneem Natour 323118794) | authors set | todo
T-0648 | 12 | cover | Cover PDF: title "Running a Massive LLM Locally: AirLLM, Quantization & Benchmarking" | title set | todo
T-0649 | 12 | cover | Generate docs/cover.pdf | PDF generated | todo
T-0650 | 12 | cover | tests/unit/test_cover_pdf.py — failing test for cover metadata (red→green) | test passes | todo
T-0651 | 12 | cover | Verify cover.pdf exercise field == 5 | field correct | todo
T-0652 | 12 | release | Update CHANGELOG.md [1.00] release date 2026 | changelog finalised | todo
T-0653 | 12 | release | Bump version strings to 1.00 across package + pyproject | versions consistent | todo
T-0654 | 12 | release | Verify pyproject version == package __version__ == 1.00 | versions match | todo
T-0655 | 12 | release | Final commit (conventional) of release prep | committed; CI green | todo
T-0656 | 12 | release | git tag v1.00 (annotated) | tag created | todo
T-0657 | 12 | release | git push origin main --tags | pushed | todo
T-0658 | 12 | release | Verify CI green on tagged commit | CI green | todo
T-0659 | 12 | release | Create GitHub release v1.00 with notes | release published | todo
T-0660 | 12 | release | Attach cover.pdf + key figures to GitHub release | assets attached | todo
T-0661 | 12 | release | Release notes link README report + ledger | links present | todo
T-0662 | 12 | moodle | Prepare Moodle submission bundle (repo link + cover.pdf) | bundle ready | todo
T-0663 | 12 | moodle | Verify submission matches course 203.3763 exercise 5 requirements | verified | todo
T-0664 | 12 | moodle | Submit to Moodle | submitted | todo
T-0665 | 12 | release | Verify release tag reproduces all figures from committed ledger | reproducible | todo
T-0666 | 12 | release | Verify repo public/visible per submission policy | confirmed | todo
T-0667 | 12 | docs | docs/prompts/012_phase12_release.md prompt log | file present | todo
T-0668 | 12 | ledger | Final docs/TODO.md status pass (all done where applicable) | statuses final | todo
T-0669 | 12 | release | Tag message references D1–D15 acceptance + KPIs | message complete | todo
T-0670 | 12 | release | Post-release smoke: clone fresh, uv sync, pytest green | fresh clone green | todo
