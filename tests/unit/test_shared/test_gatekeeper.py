"""Tests for the measurement-ledger Gatekeeper (rule 13)."""

from __future__ import annotations

from cosmos77_ex05.shared.gatekeeper import Gatekeeper


def test_record_writes_scenario_json(tmp_path):
    gk = Gatekeeper(tmp_path)
    path = gk.record("airllm_4bit", {"throughput_tok_s": 2.1, "peak_vram_gb": 7.4})
    assert path.name == "airllm_4bit.json"
    data = gk.read("airllm_4bit")
    assert data["scenario"] == "airllm_4bit"
    assert data["throughput_tok_s"] == 2.1


def test_record_merges_into_existing(tmp_path):
    gk = Gatekeeper(tmp_path)
    gk.record("airllm_none", {"ttft_s": 1.0})
    gk.record("airllm_none", {"throughput_tok_s": 1.5})
    data = gk.read("airllm_none")
    assert data["ttft_s"] == 1.0
    assert data["throughput_tok_s"] == 1.5


def test_read_missing_is_empty(tmp_path):
    assert Gatekeeper(tmp_path).read("nope") == {}


def test_ledger_aggregates_all_scenarios(tmp_path):
    gk = Gatekeeper(tmp_path)
    gk.record("fp16_baseline", {"success": False})
    gk.record("airllm_8bit", {"throughput_tok_s": 2.0})
    ledger = gk.ledger()
    assert set(ledger) == {"fp16_baseline", "airllm_8bit"}
    assert ledger["fp16_baseline"]["success"] is False


def test_ledger_missing_dir_is_empty(tmp_path):
    assert Gatekeeper(tmp_path / "nope").ledger() == {}


def test_ledger_skips_bad_json(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    Gatekeeper(tmp_path).record("ok", {"x": 1})
    ledger = Gatekeeper(tmp_path).ledger()
    assert "ok" in ledger
    assert "broken" not in ledger


def test_scrub_redacts_hf_token():
    scrubbed = Gatekeeper.scrub("token=hf_AbCdEfGhIjKlMnOpQrStUvWx now")
    assert "hf_AbCd" not in scrubbed
    assert "[REDACTED]" in scrubbed
