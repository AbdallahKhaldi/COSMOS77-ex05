# Contributing to COSMOS77-ex05

Working agreement for the two student contributors (Abdallah Khaldi and Tasneem
Natour) and any future maintainers. Mirrors the 17 binding rules in
[CLAUDE.md](CLAUDE.md) and the master playbook (`../CLAUDE_CODE_PLAYBOOK.md`).

## Local setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # one-time
uv sync
uv run pre-commit install
```

`uv sync` materialises `.venv/` and `uv.lock`. **Never** invoke `pip`,
`python -m venv`, or `python script.py` directly for OUR code (rule 5). The
Kaggle/Colab T4 runtime is the one exception: it uses its own pip to install the
heavy experiment deps — that is the **experiment environment**, separate from ours.

### Two environments

- **Our repo (this machine / CI):** CPU-only, `uv`-managed; the measurement,
  analysis, economic, and plot code. The test suite **never** downloads a model
  or needs a GPU — all model/GPU/HF I/O is mocked (rule 6).
- **The experiment (Kaggle/Colab T4):** `airllm`, `transformers`, `torch`,
  `accelerate`, `bitsandbytes`, `huggingface-hub` — installed by the notebook.
  `bitsandbytes` quantization is **CUDA-only** (it does not run on a Mac).

### System prerequisites (not pip dependencies)

- A free **Kaggle** account (phone-verified to unlock GPU + internet), or a Colab
  T4 fallback. `~/.kaggle/kaggle.json` lives in your home dir — **never** commit it.
- Optionally a free **HuggingFace read token** in `.env` / a Kaggle secret (Qwen2.5
  is ungated, so it only avoids rate limits) — never in code (rule 9).

## Commits

Conventional Commits (rule 11): `type(scope): summary` + a body referencing the
TODO ID (`Closes T-NNNN`). Multiple commits per phase; no `wip`/`tmp`/`fixup`.
Two-person team — work is authored by both partners across the history.

## Quality gates (run before pushing)

```bash
uv run ruff check .                 # zero issues
uv run ruff format --check .        # zero diffs
uv run python scripts/check_line_cap.py
uv run pytest -m "not live" --cov-fail-under=85
```

The same gates run in [GitHub Actions](.github/workflows/ci.yml) on every push.
Live runs (real GPU/model/HF/Kaggle) are marked `live` and excluded from CI.

## Honesty

Every measured number flows through the **measurement ledger** (`results/*.json`)
— the single source of truth for every table and graph (rule 13). Never fabricate
a token/sec or a memory number; a well-analyzed negative result is acceptable and
must be explained. If a rule genuinely cannot be followed for a module, document
the exception as an ADR in `docs/PLAN.md`.
