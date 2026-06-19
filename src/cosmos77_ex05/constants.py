"""Project-wide structural constants (not tunable config — see CLAUDE.md rule 4).

Fixed enumerations the runners, measurement harness, and analysis share. Tunable
values (model id, quant levels, max tokens, prices, hardware assumptions, paths)
live in ``config/*.json`` and ``.env`` and are read via the Config loader (Phase 2).
"""

from __future__ import annotations

#: Default text encoding for all file I/O across the project.
DEFAULT_ENCODING: str = "utf-8"

#: The importable package name (mirrors pyproject ``name``, underscored).
PACKAGE_NAME: str = "cosmos77_ex05"

#: The version string — kept in lockstep with pyproject and every config file.
PROJECT_VERSION: str = "1.00"

#: The four measured scenarios, in run order (each becomes a ``results/<name>.json``).
SCENARIOS: tuple[str, ...] = (
    "fp16_baseline",
    "airllm_none",
    "airllm_8bit",
    "airllm_4bit",
)

#: Bytes per parameter at each precision — the param→memory math (D1).
BYTES_PER_PARAM: dict[str, float] = {"fp16": 2.0, "q8": 1.0, "q4": 0.5}

#: The ``cosmos77-airllm`` CLI stages (wired to the SDK per phase).
PIPELINE_STAGES: tuple[str, ...] = (
    "hardware",
    "baseline",
    "airllm",
    "quant",
    "measure",
    "analyze",
    "economics",
    "report",
)
