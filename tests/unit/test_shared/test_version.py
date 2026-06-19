"""Tests for the version constant + config-version guard (rule 10)."""

from __future__ import annotations

import pytest

from cosmos77_ex05.shared import version


def test_version_is_one_zero():
    assert version.VERSION == "1.00"


def test_validate_accepts_matching():
    version.validate_config_version("1.00")  # must not raise


def test_validate_rejects_drift():
    with pytest.raises(ValueError):
        version.validate_config_version("0.9")
