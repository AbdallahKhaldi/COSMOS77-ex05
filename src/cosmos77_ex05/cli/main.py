"""Command-line entry point for ``cosmos77-airllm``.

A thin dispatcher over the SDK. Each subcommand (hardware, baseline, airllm,
quant, measure, analyze, economics, report) is wired to the SDK in its phase;
until then it prints guidance. No business logic lives here (CLAUDE.md rule 2 —
all logic flows through the SDK).
"""

from __future__ import annotations

import argparse
import sys

from cosmos77_ex05.constants import PIPELINE_STAGES


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="cosmos77-airllm",
        description="AirLLM local massive-LLM benchmarking for Orchestration of AI Agents (203.3763).",
    )
    parser.add_argument("command", nargs="?", choices=PIPELINE_STAGES, help="pipeline stage to run")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns a process exit code."""
    from cosmos77_ex05 import __version__

    args = build_parser().parse_args(argv)
    if args.version:
        print(f"cosmos77-airllm {__version__}")
        return 0
    if args.command is None:
        build_parser().print_help()
        return 0
    print(f"`{args.command}` is not wired yet — it lands in its phase (see docs/TODO.md).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
