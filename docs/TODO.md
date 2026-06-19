# TODO — COSMOS77-ex05

Single source of truth for outstanding work (CONTRIBUTING.md). Format:

`T-NNNN | phase | area | description | DoD (definition of done) | status`

Status ∈ `todo` · `doing` · `done`. Completed tasks carry the commit SHA. Phase 1
expands this seed into the **≥ 600** granular tasks the playbook requires
(`../CLAUDE_CODE_PLAYBOOK.md` §0 / §3); this file currently tracks Phase 0 only.

## Phase 0 — Repo bootstrap

T-0001 | 0 | repo | Reconnaissance: tools (uv/gh/git/kaggle), ~/.kaggle creds, HW4 tooling to port | versions confirmed; port plan set | done
T-0002 | 0 | repo | Directory skeleton (src/ tests/ docs/ config/ experiments/ results/ reports/ figures/ data/) + subpackages | tree matches playbook §2.2 | done
T-0003 | 0 | build | pyproject.toml — cosmos77-ex05 v1.00, py3.11, analysis deps + experiment optional extra + kaggle dev dep | uv sync resolves; uv.lock committed | done
T-0004 | 0 | build | .python-version, .gitignore (data/shards/kaggle_out ignored; results/figures/ipynb kept), .env.example (no secrets) | .env ignored; only .env.example tracked | done
T-0005 | 0 | config | config/setup.json (Qwen2.5-14B, T4, 4 scenarios, hardware assumptions) | parses; matches playbook | done
T-0006 | 0 | config | config/pricing.json (API prices + cloud GPU + caching) + logging_config.json | parses | done
T-0007 | 0 | pkg | src/cosmos77_ex05 __init__ (v1.00), constants (SCENARIOS/BYTES_PER_PARAM), cli/main dispatcher, empty subpackages | package imports; cosmos77-airllm --version works | done
T-0008 | 0 | rules | CLAUDE.md (17 rules, §16 verbatim) | byte-matches playbook §16 | done
T-0009 | 0 | docs | README placeholder, LICENSE (MIT 2026), CHANGELOG [1.00], CONTRIBUTING (two-env note) | all present; CI badge in README | done
T-0010 | 0 | ci | scripts/check_line_cap.py (port), generate_cover_pdf.py (port, ex05/exercise 5) | line-cap exits 0 | done
T-0011 | 0 | ci | .pre-commit-config.yaml + .github/workflows/ci.yml (ruff/format/line-cap/pytest-85; no GPU/model) | hooks installed; CI green | done
T-0012 | 0 | test | tests/conftest.py (seed) + tests/unit/test_constants.py smoke test | uv run pytest green, coverage >= 85% | done
T-0013 | 0 | qa | uv sync; ruff check/format zero; check_line_cap 0; pytest >= 85% | every gate green | done
T-0014 | 0 | docs | docs/prompts/000_phase0_bootstrap.md prompt log | file present | done
T-0015 | 0 | git | git init -b main, identity + remote, conventional commits, git push -u origin main, CI green | Actions green on main | done

## Phase 1 → 12 — placeholder

`T-0100`+ are authored in Phase 1 (PRD/PLAN/TODO + 8 mechanism PRDs), bringing the
list to ≥ 600 items across P1–P12. See `../CLAUDE_CODE_PLAYBOOK.md` §18 for the
per-phase deliverables.
