"""The break-even cost math (D7): On-Prem vs API vs Cloud-GPU per request.

Why: an organisation building on an LLM must decide **build (On-Prem)** or
**buy (API)** — and from what request volume each wins. On-Prem pays the GPU
once (CAPEX) then electricity forever (OPEX), so its per-request cost falls as
volume amortizes the fixed cost; the API pays per token and stays flat. This
module is pure arithmetic (no I/O) so the crossover is trivially testable, and
it **raises** on a missing graded assumption rather than inventing a number.
"""

from __future__ import annotations

_REQUIRED = (
    "on_prem_gpu_price_usd",
    "hardware_life_years",
    "electricity_usd_per_kwh",
    "gpu_power_watts",
    "runtime_s",
    "tokens_in",
    "tokens_out",
    "input_per_1m",
    "output_per_1m",
)


def api_cost_per_request(
    tokens_in: float,
    tokens_out: float,
    input_per_1m: float,
    output_per_1m: float,
) -> float:
    """API cost (USD) of one request — tokens x price, flat in volume.

    Prices are USD per 1,000,000 tokens. Returns
    ``(tokens_in * input_per_1m + tokens_out * output_per_1m) / 1e6``.
    """
    return (tokens_in * input_per_1m + tokens_out * output_per_1m) / 1e6


def on_prem_opex_per_request(watts: float, runtime_s: float, kwh_price: float) -> float:
    """On-Prem electricity OPEX (USD) for one request (CAPEX handled separately).

    Wh -> kWh via /1000, s -> h via /3600. Returns
    ``(watts * runtime_s / 3600 / 1000) * kwh_price``.
    """
    return (watts * runtime_s / 3600 / 1000) * kwh_price


def cloud_gpu_cost_per_request(usd_per_hour: float, runtime_s: float) -> float:
    """Cloud-GPU rental cost (USD) for one request — flat in volume.

    Returns ``usd_per_hour * runtime_s / 3600``.
    """
    return usd_per_hour * runtime_s / 3600


def break_even_requests(
    capex: float,
    api_per_req: float,
    on_prem_opex_per_req: float,
) -> float | None:
    """Request volume where cumulative On-Prem equals cumulative API.

    Solves ``capex + N*opex = N*api_per_req`` for ``N``. A crossing exists only
    when the API costs more per request than On-Prem's electricity floor.

    Returns:
        ``capex / (api_per_req - on_prem_opex_per_req)`` when
        ``api_per_req > on_prem_opex_per_req``; otherwise ``None`` (On-Prem
        never breaks even — the caller must report that honestly).
    """
    if api_per_req <= on_prem_opex_per_req:
        return None
    return capex / (api_per_req - on_prem_opex_per_req)


def cumulative_costs(
    volumes: list[float],
    capex: float,
    api_per_req: float,
    on_prem_opex_per_req: float,
    cloud_per_req: float | None = None,
) -> dict:
    """Cumulative cost curves over ``volumes`` for the Phase-8 plot.

    On-Prem = CAPEX (upfront) + N*opex; API = N*api_per_req; Cloud = N*cloud.

    Returns:
        Dict with ``"on_prem"`` and ``"api"`` arrays, plus ``"cloud_gpu"``
        when ``cloud_per_req`` is given — each aligned to ``volumes``.
    """
    curves: dict[str, list[float]] = {
        "on_prem": [capex + n * on_prem_opex_per_req for n in volumes],
        "api": [n * api_per_req for n in volumes],
    }
    if cloud_per_req is not None:
        curves["cloud_gpu"] = [n * cloud_per_req for n in volumes]
    return curves


def break_even(assumptions: dict) -> dict:
    """Compute the full D7 break-even from a dict of assumptions.

    Requires every key in ``_REQUIRED`` (CAPEX, life, kWh rate, watts, runtime,
    token counts, token prices). Optional ``cloud_usd_per_hour`` enables the
    Cloud-GPU line.

    Returns:
        Dict with ``api_per_req``, ``on_prem_opex_per_req``, ``cloud_per_req``,
        ``break_even_requests`` and the echoed ``assumptions``.

    Raises:
        ValueError: If any required (graded) assumption is missing — no silent
            default is ever substituted for a graded number.
    """
    missing = [key for key in _REQUIRED if key not in assumptions]
    if missing:
        raise ValueError(f"Missing required economics assumption(s): {missing}")
    api_per_req = api_cost_per_request(
        assumptions["tokens_in"],
        assumptions["tokens_out"],
        assumptions["input_per_1m"],
        assumptions["output_per_1m"],
    )
    opex = on_prem_opex_per_request(
        assumptions["gpu_power_watts"],
        assumptions["runtime_s"],
        assumptions["electricity_usd_per_kwh"],
    )
    cloud_rate = assumptions.get("cloud_usd_per_hour")
    cloud_per_req = (
        cloud_gpu_cost_per_request(cloud_rate, assumptions["runtime_s"])
        if cloud_rate is not None
        else None
    )
    return {
        "api_per_req": api_per_req,
        "on_prem_opex_per_req": opex,
        "cloud_per_req": cloud_per_req,
        "break_even_requests": break_even_requests(
            assumptions["on_prem_gpu_price_usd"], api_per_req, opex
        ),
        "assumptions": assumptions,
    }
