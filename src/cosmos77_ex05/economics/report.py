"""Phase-8 ECONOMICS report writer (D7): On-Prem vs API vs Cloud-GPU.

Turns the cost-model CORE (:mod:`cosmos77_ex05.economics.model`) plus the graded
config into a per-request *scenario* and a Markdown ``ECONOMICS.md`` stating every
assumption, the cost table, and the break-even volume + caching shift. Nothing is
hard-coded — prices, CAPEX, life, kWh rate and watts all come from ``config``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cosmos77_ex05.economics.caching import apply_caching_discount, break_even_with_caching
from cosmos77_ex05.economics.model import (
    api_cost_per_request,
    break_even_requests,
    cloud_gpu_cost_per_request,
    on_prem_opex_per_request,
)

#: Volume amortizing CAPEX, cached input fraction, days/yr, the report request +
#: provider, and the Markdown destination. Stated assumptions, never hard-coded math.
REQUESTS_PER_DAY: int = 1000
CACHED_FRACTION: float = 0.5
_DAYS_PER_YEAR: int = 365
_REPORT_REQUEST = {"tokens_in": 500, "tokens_out": 200, "runtime_s": 10.0}
_REPORT_PROVIDER = "google_gemini_flash"
_DEFAULT_REPORT: Path = Path("reports/ECONOMICS.md")


def compute_scenario(
    config: Any,
    *,
    tokens_in: float,
    tokens_out: float,
    runtime_s: float,
    provider: str,
) -> dict[str, Any]:
    """Per-request On-Prem (CAPEX+OPEX) / API / Cloud-GPU costs + break-even volume."""
    hw = config.hardware_assumptions()
    pricing = config.pricing()
    prices = config.provider_prices(provider)
    out_per_1m, in_per_1m = prices["output_per_1m"], prices["input_per_1m"]
    capex = float(hw["on_prem_gpu_price_usd"])
    api_per_req = api_cost_per_request(tokens_in, tokens_out, in_per_1m, out_per_1m)
    opex = on_prem_opex_per_request(hw["gpu_power_watts"], runtime_s, hw["electricity_usd_per_kwh"])
    capex_per_req = capex / (hw["hardware_life_years"] * _DAYS_PER_YEAR * REQUESTS_PER_DAY)
    discount = pricing["prompt_caching"]["cached_input_discount"]
    cached_in = apply_caching_discount(in_per_1m, CACHED_FRACTION, discount)
    cached_api = api_cost_per_request(tokens_in, tokens_out, cached_in, out_per_1m)
    cloud = cloud_gpu_cost_per_request(pricing["cloud_gpu"]["t4_usd_per_hour"], runtime_s)
    return {
        "provider": provider,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "runtime_s": runtime_s,
        "capex": capex,
        "on_prem_opex_per_req": opex,
        "on_prem_per_req": capex_per_req + opex,
        "api_per_req": api_per_req,
        "cached_api_per_req": cached_api,
        "cloud_per_req": cloud,
        "break_even_requests": break_even_requests(capex, api_per_req, opex),
        "cached_break_even_requests": break_even_with_caching(capex, cached_api, opex),
    }


def _cost_table(config: Any, scenario: dict[str, Any]) -> str:
    """A pipe table of per-request USD costs: On-Prem, each API, Cloud-GPU."""
    tin, tout = scenario["tokens_in"], scenario["tokens_out"]
    providers = []
    for name, p in config.pricing()["providers"].items():
        cost = api_cost_per_request(tin, tout, p["input_per_1m"], p["output_per_1m"])
        providers.append((cost, f"| API `{name}` | {cost:.6f} |"))
    rows = [row for _, row in sorted(providers)]
    on_prem = f"| On-Prem (amortized CAPEX + electricity) | {scenario['on_prem_per_req']:.6f} |"
    cloud = f"| Cloud-GPU (T4 rental) | {scenario['cloud_per_req']:.6f} |"
    head = "| Option | USD / request |\n| --- | --- |"
    return "\n".join([head, on_prem, *rows, cloud])


def _assumptions_table(config: Any, scenario: dict[str, Any]) -> str:
    """A pipe table stating every graded assumption behind the numbers."""
    hw = config.hardware_assumptions()
    rows = [
        ("CAPEX (On-Prem GPU)", f"${hw['on_prem_gpu_price_usd']}"),
        ("Hardware life", f"{hw['hardware_life_years']} years"),
        ("Electricity", f"${hw['electricity_usd_per_kwh']} / kWh"),
        ("GPU power draw", f"{hw['gpu_power_watts']} W"),
        ("Request", f"{scenario['tokens_in']:g} in + {scenario['tokens_out']:g} out tokens"),
        ("Runtime basis (per request)", f"{scenario['runtime_s']:g} s"),
        ("Amortization volume", f"{REQUESTS_PER_DAY} requests/day"),
        ("Cached input fraction", f"{CACHED_FRACTION:g}"),
    ]
    head = "| Assumption | Value |\n| --- | --- |\n"
    return head + "\n".join(f"| {k} | {v} |" for k, v in rows)


def _break_even_section(scenario: dict[str, Any]) -> str:
    """Monthly break-even volume (On-Prem vs cheapest API) + the caching shift."""
    base, cached = scenario["break_even_requests"], scenario["cached_break_even_requests"]
    if base is None:
        return "On-Prem never breaks even: its electricity floor exceeds the API price."
    return (
        f"On-Prem pays for itself after **{base:,.0f} requests** (~{base / 30:,.0f}/day over "
        f"a month) versus the cheapest API. Prompt/context caching cuts the effective API "
        f"price, **raising** the break-even to **{cached:,.0f} requests** — a cheaper API "
        f"stays competitive longer, so On-Prem must do *more* work to win."
    )


def _recommendation(ledger: dict[str, dict[str, Any]]) -> str:
    """Honest On-Prem-vs-API recommendation weighing privacy + AirLLM latency."""
    tok_s = ledger.get("airllm_4bit", {}).get("throughput_tok_s", 0.0)
    return (
        f"**Recommendation.** Our measured AirLLM 4-bit throughput is only ~{tok_s:.3f} tok/s "
        f"on a constrained 16 GB GPU, so local serving is impractical for high throughput — "
        f"for cost-per-token at low volume the API wins. On-Prem's real value is **privacy**: "
        f"nothing leaves the organisation, justifying the build for sensitive data regardless "
        f"of the break-even. Verdict: API for throughput/low volume; On-Prem for privacy."
    )


def write_economics_md(
    config: Any,
    ledger: dict[str, dict[str, Any]],
    out_path: Path | str = _DEFAULT_REPORT,
) -> Path:
    """Write the D7 ECONOMICS Markdown report (Assumptions, cost table, break-even
    + caching shift, privacy recommendation) and return its path; makes parents.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scenario = compute_scenario(config, provider=_REPORT_PROVIDER, **_REPORT_REQUEST)
    body = "\n\n".join(
        [
            "# On-Prem vs API economics (break-even)",
            "## Assumptions",
            _assumptions_table(config, scenario),
            "## Per-request cost",
            _cost_table(config, scenario),
            "## Break-even and prompt/context caching",
            _break_even_section(scenario),
            "## Recommendation (privacy vs cost)",
            _recommendation(ledger),
        ]
    )
    out.write_text(body + "\n", encoding="utf-8")
    return out
