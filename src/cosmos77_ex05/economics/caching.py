"""Prompt/Context-Caching discount on the API price (D7, caching shift).

A cached input token (a repeated system prompt / context) is billed at only
``discount`` of the list price. Caching therefore *lowers* the effective API
price — which pushes the On-Prem<->API break-even to a **higher** volume: a
cheaper API stays competitive longer, so On-Prem must do more work to beat it.
"""

from __future__ import annotations

from cosmos77_ex05.economics.model import break_even_requests


def apply_caching_discount(
    input_per_1m: float,
    cached_fraction: float,
    discount: float,
) -> float:
    """Return the effective input price once a fraction is billed at ``discount``.

    Args:
        input_per_1m: List input price in USD per 1,000,000 tokens.
        cached_fraction: Fraction (0..1) of input tokens served from cache.
        discount: Fraction of list price paid per cached token (e.g. ``0.5``).

    Returns:
        ``input_per_1m * ((1 - cached_fraction) + cached_fraction * discount)``
        — always <= ``input_per_1m`` (caching only lowers the price).
    """
    return input_per_1m * ((1 - cached_fraction) + cached_fraction * discount)


def break_even_with_caching(
    capex: float,
    cached_api_per_req: float,
    on_prem_opex_per_req: float,
) -> float | None:
    """Recompute the break-even volume using a caching-discounted API price.

    Because ``cached_api_per_req`` is lower, the returned volume is *higher*
    than the uncached one — the documented direction of the caching shift.

    Args:
        capex: Upfront On-Prem hardware cost (USD).
        cached_api_per_req: API cost per request after the caching discount.
        on_prem_opex_per_req: On-Prem electricity cost per request (USD).

    Returns:
        The break-even request volume, or ``None`` when no crossing exists.
    """
    return break_even_requests(capex, cached_api_per_req, on_prem_opex_per_req)
