"""Tests for the cover-PDF field values (Phase 12) — exercise 5, ex05 URL."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "generate_cover_pdf.py"


def _load():
    spec = importlib.util.spec_from_file_location("gen_cover", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exercise_number_is_five():
    fields = dict(_load().build_field_values(5, 85))
    assert fields["Submitting an exercise number"] == "5"


def test_repo_url_is_ex05():
    fields = dict(_load().build_field_values(5, 85))
    assert fields["Link to GITHUB"].endswith("COSMOS77-ex05")
    assert "ex04" not in fields["Link to GITHUB"]


def test_self_score_and_group():
    fields = dict(_load().build_field_values(5, 85))
    assert fields["Recommendation for self-scoring"] == "85"
    assert fields["Group ID code"] == "COSMOS77"
    assert fields["A late submission confirmation"] == "no"


def test_two_students_with_ids():
    ids = [s[1][0] for s in _load()._STUDENTS]
    assert ids == ["212389712", "323118794"]
