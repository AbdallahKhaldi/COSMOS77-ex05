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
    return _dispatch(args.command)


def _dispatch(command: str) -> int:
    """Run one pipeline stage via the SDK and print a short summary."""
    from cosmos77_ex05.sdk.sdk import SDK

    sdk = SDK()
    if command == "hardware":
        out = sdk.capture_hardware()
        spec, math = out["spec"], out["model_math"]
        cpu = spec.get("cpu", {})
        gpu = spec.get("gpu", {})
        print(f"CPU: {cpu.get('model', '?')} ({cpu.get('cores_physical', '?')} physical cores)")
        print(f"RAM: {spec.get('ram_gb', '?')} GB   GPU: {gpu.get('name', 'none / CPU-only')}")
        print(f"model: {math.get('model_id')} -> FP16 {math.get('memory_gb', {}).get('fp16')} GB")
        print(f"verdict: {math.get('verdict')}")
        return 0
    if command == "analyze":
        out = sdk.analyze()
        print(f"wrote {out['metrics_md']}")
        for fig in out["figures"]:
            print(f"figure: {fig}")
        return 0
    if command == "economics":
        out = sdk.economics()
        print(f"wrote {out['economics_md']}")
        print(f"figure: {out['figure']}")
        return 0
    print(f"`{command}` is not wired yet — it lands in its phase (see docs/TODO.md).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
