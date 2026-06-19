"""Tests for the Phase-8 ECONOMICS report writer (D7).

The report layer turns the cost-model CORE + config into a per-request scenario
and a Markdown break-even report. We assert on the *direction* of the caching
shift (it raises the break-even volume) and on the presence of every graded
number/keyword the report must state — never the GPU, never the network.
"""

from __future__ import annotations

from pathlib import Path

from cosmos77_ex05.economics import report

#: A tiny, real-shaped ledger citing the measured AirLLM 4-bit per-request fact.
_LEDGER = {
    "airllm_4bit": {
        "success": True,
        "throughput_tok_s": 0.040964,
        "tpot_ms": 23294.021335,
        "total_s": 488.239252,
        "est_power_wh": 9.493541,
        "peak_vram_gb": 3.153576,
        "n_out": 20,
    },
}


def _scenario(config):
    """Compute the representative scenario from the config fixture."""
    return report.compute_scenario(
        config,
        tokens_in=500,
        tokens_out=200,
        runtime_s=10.0,
        provider="google_gemini_flash",
    )


def test_compute_scenario_returns_positive_costs(config) -> None:
    """On-Prem, API and Cloud-GPU per-request costs are all positive."""
    scenario = _scenario(config)
    assert scenario["on_prem_per_req"] > 0
    assert scenario["api_per_req"] > 0
    assert scenario["cloud_per_req"] > 0
    assert scenario["on_prem_opex_per_req"] > 0


def test_compute_scenario_break_even_finite_when_api_above_opex(config) -> None:
    """A finite break-even exists when API/req exceeds On-Prem electricity/req."""
    scenario = _scenario(config)
    assert scenario["api_per_req"] > scenario["on_prem_opex_per_req"]
    assert scenario["break_even_requests"] is not None
    assert scenario["break_even_requests"] > 0


def test_compute_scenario_caching_raises_break_even(config) -> None:
    """Caching lowers the API price, so the break-even moves to a HIGHER volume."""
    scenario = _scenario(config)
    assert scenario["cached_api_per_req"] < scenario["api_per_req"]
    assert scenario["cached_break_even_requests"] > scenario["break_even_requests"]


def test_compute_scenario_includes_chosen_provider(config) -> None:
    """The scenario echoes the chosen provider for traceability."""
    scenario = _scenario(config)
    assert scenario["provider"] == "google_gemini_flash"


def _read_report(config, tmp_path: Path) -> str:
    out = report.write_economics_md(config, _LEDGER, out_path=tmp_path / "ECONOMICS.md")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert len(text) > 0
    return text


def test_write_economics_md_states_assumptions_and_keywords(config, tmp_path: Path) -> None:
    """The report names its assumptions, break-even, privacy, and caching."""
    text = _read_report(config, tmp_path)
    lowered = text.lower()
    assert "assumption" in lowered
    assert "break-even" in lowered
    assert "privacy" in lowered
    assert "caching" in lowered


def test_write_economics_md_contains_graded_numbers(config, tmp_path: Path) -> None:
    """CAPEX $1600, 3-year life and $0.15/kWh appear verbatim in the report."""
    text = _read_report(config, tmp_path)
    assert "1600" in text
    assert "3" in text
    assert "0.15" in text


def test_write_economics_md_renders_a_cost_table(config, tmp_path: Path) -> None:
    """A Markdown pipe table lists On-Prem, the API providers and Cloud-GPU."""
    text = _read_report(config, tmp_path)
    assert "| " in text and " |" in text  # at least one pipe table row
    assert "On-Prem" in text
    assert "Cloud-GPU" in text
    for provider in ("openai_gpt4o_mini", "anthropic_claude_haiku", "google_gemini_flash"):
        assert provider in text


def test_write_economics_md_creates_missing_parent(config, tmp_path: Path) -> None:
    """The writer creates a missing parent directory for the report path."""
    out_path = tmp_path / "nested" / "reports" / "ECONOMICS.md"
    out = report.write_economics_md(config, _LEDGER, out_path=out_path)
    assert out.exists()
    assert out.parent.is_dir()


def test_compute_scenario_no_break_even_when_opex_exceeds_api(config) -> None:
    """A long runtime makes On-Prem electricity exceed the API price => no crossing."""
    scenario = report.compute_scenario(
        config, tokens_in=500, tokens_out=200, runtime_s=100.0, provider="google_gemini_flash"
    )
    assert scenario["on_prem_opex_per_req"] > scenario["api_per_req"]
    assert scenario["break_even_requests"] is None


def test_break_even_section_reports_when_never_breaks_even(config) -> None:
    """The report honestly states when On-Prem never breaks even (None branch)."""
    section = report._break_even_section(
        {"break_even_requests": None, "cached_break_even_requests": None}
    )
    assert "never breaks even" in section
