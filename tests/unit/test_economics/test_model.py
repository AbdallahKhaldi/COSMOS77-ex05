"""Tests for the D7 cost math: per-request costs, break-even, cumulative curves.

Worked truths used below (all hand-computed):
- api_cost_per_request(1000, 200, 0.15, 0.60) = (150 + 120)/1e6 = 0.00027 USD.
- on_prem_opex_per_request(70, 10, 0.15) = (70*10/3600/1000)*0.15 electricity.
- break_even_requests(1600, api, opex) = 1600/(api - opex) when api > opex.
"""

from __future__ import annotations

import pytest

from cosmos77_ex05.economics.model import (
    api_cost_per_request,
    break_even,
    break_even_requests,
    cloud_gpu_cost_per_request,
    cumulative_costs,
    on_prem_opex_per_request,
)


def _assumptions(**overrides: float) -> dict:
    """A complete, valid assumptions dict; ``overrides`` tweak single keys."""
    base = {
        "on_prem_gpu_price_usd": 1600,
        "hardware_life_years": 3,
        "electricity_usd_per_kwh": 0.15,
        "gpu_power_watts": 70,
        "runtime_s": 10.0,
        "tokens_in": 1000.0,
        "tokens_out": 200.0,
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
    }
    base.update(overrides)
    return base


def test_api_cost_per_request_exact():
    assert api_cost_per_request(1000, 200, 0.15, 0.60) == (1000 * 0.15 + 200 * 0.60) / 1e6
    assert api_cost_per_request(1000, 200, 0.15, 0.60) == 0.00027


def test_on_prem_opex_per_request_exact():
    expected = (70 * 10 / 3600 / 1000) * 0.15
    assert on_prem_opex_per_request(70, 10, 0.15) == expected


def test_cloud_gpu_cost_per_request_exact():
    assert cloud_gpu_cost_per_request(0.35, 3600) == 0.35
    assert cloud_gpu_cost_per_request(0.35, 10) == 0.35 * 10 / 3600


def test_break_even_requests_crossover():
    api = api_cost_per_request(1000, 200, 0.15, 0.60)
    opex = on_prem_opex_per_request(70, 10, 0.15)
    assert api > opex
    result = break_even_requests(1600, api, opex)
    assert result == 1600 / (api - opex)


def test_break_even_requests_none_when_opex_exceeds_api():
    api = 0.0001
    opex = 0.0002  # OPEX >= API => On-Prem never beats the API.
    assert break_even_requests(1600, api, opex) is None
    assert break_even_requests(1600, api, api) is None  # equal => no crossing


def test_cumulative_costs_onprem_starts_at_capex_api_starts_at_zero():
    api = api_cost_per_request(1000, 200, 0.15, 0.60)
    opex = on_prem_opex_per_request(70, 10, 0.15)
    star = break_even_requests(1600, api, opex)
    volumes = [0.0, star, 2 * star]

    curves = cumulative_costs(volumes, 1600, api, opex, cloud_per_req=0.001)

    assert curves["on_prem"][0] == 1600  # CAPEX upfront at zero volume
    assert curves["api"][0] == 0.0  # no CAPEX for the API line
    # API is below On-Prem before the crossing, above after it => crosses once.
    assert curves["api"][0] < curves["on_prem"][0]
    assert curves["api"][2] > curves["on_prem"][2]
    assert curves["api"][1] == pytest.approx(curves["on_prem"][1])
    assert curves["cloud_gpu"][2] == 2 * star * 0.001


def test_cumulative_costs_omits_cloud_when_none():
    curves = cumulative_costs([0.0, 10.0], 1600, 0.001, 0.0001)
    assert "cloud_gpu" not in curves


def test_break_even_full_pipeline():
    result = break_even(_assumptions(cloud_usd_per_hour=0.35))
    assert result["api_per_req"] == 0.00027
    assert result["on_prem_opex_per_req"] == (70 * 10 / 3600 / 1000) * 0.15
    assert result["cloud_per_req"] == 0.35 * 10 / 3600
    assert result["break_even_requests"] == 1600 / (
        result["api_per_req"] - result["on_prem_opex_per_req"]
    )
    assert result["assumptions"]["on_prem_gpu_price_usd"] == 1600


def test_break_even_cloud_per_req_none_without_rate():
    result = break_even(_assumptions())
    assert result["cloud_per_req"] is None


def test_break_even_missing_capex_raises():
    bad = _assumptions()
    del bad["on_prem_gpu_price_usd"]
    with pytest.raises(ValueError, match="on_prem_gpu_price_usd"):
        break_even(bad)


@pytest.mark.parametrize(
    "key",
    ["hardware_life_years", "gpu_power_watts", "tokens_in", "input_per_1m", "runtime_s"],
)
def test_break_even_any_missing_graded_key_raises(key: str):
    bad = _assumptions()
    del bad[key]
    with pytest.raises(ValueError, match=key):
        break_even(bad)
