"""Tests for the Prompt/Context-Caching discount and its break-even direction.

Worked truth: apply_caching_discount(0.15, 0.8, 0.5) = 0.15*(0.2 + 0.8*0.5) =
0.09 < 0.15 (caching lowers the price). A lower API price moves the On-Prem
break-even to a HIGHER request volume — locked by an explicit direction assert.
"""

from __future__ import annotations

from cosmos77_ex05.economics.caching import (
    apply_caching_discount,
    break_even_with_caching,
)
from cosmos77_ex05.economics.model import (
    api_cost_per_request,
    break_even_requests,
    on_prem_opex_per_request,
)


def test_apply_caching_discount_lowers_price():
    effective = apply_caching_discount(0.15, 0.8, 0.5)
    assert effective == 0.15 * ((1 - 0.8) + 0.8 * 0.5)
    assert effective == 0.09
    assert effective < 0.15  # caching only ever lowers the input price


def test_apply_caching_discount_extremes():
    assert apply_caching_discount(0.15, 0.0, 0.5) == 0.15  # nothing cached
    assert apply_caching_discount(0.15, 1.0, 0.5) == 0.15 * 0.5  # all cached


def test_caching_moves_break_even_to_higher_volume():
    capex = 1600
    tokens_in, tokens_out = 1000.0, 200.0
    output_per_1m = 0.60
    opex = on_prem_opex_per_request(70, 10, 0.15)

    api_full = api_cost_per_request(tokens_in, tokens_out, 0.15, output_per_1m)
    cached_input = apply_caching_discount(0.15, 0.8, 0.5)
    api_cached = api_cost_per_request(tokens_in, tokens_out, cached_input, output_per_1m)

    star_full = break_even_requests(capex, api_full, opex)
    star_cached = break_even_with_caching(capex, api_cached, opex)

    assert api_cached < api_full  # the API line dropped
    assert star_cached > star_full  # ...so break-even moved to a higher volume


def test_break_even_with_caching_none_when_no_crossing():
    # An aggressively cached, tiny-token request can fall below the OPEX floor.
    assert break_even_with_caching(1600, 0.00001, 0.00005) is None
