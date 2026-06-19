"""Phase-0 smoke test: the package imports and core constants are sane."""

from __future__ import annotations

import cosmos77_ex05
from cosmos77_ex05 import constants


def test_package_version_is_one_zero():
    assert cosmos77_ex05.__version__ == "1.00"


def test_default_encoding_is_utf8():
    assert constants.DEFAULT_ENCODING == "utf-8"


def test_package_name_matches_module():
    assert constants.PACKAGE_NAME == "cosmos77_ex05"


def test_project_version_matches_package():
    assert constants.PROJECT_VERSION == cosmos77_ex05.__version__


def test_scenarios_and_bytes_per_param():
    assert constants.SCENARIOS[0] == "fp16_baseline"
    assert len(constants.SCENARIOS) == 4
    assert constants.BYTES_PER_PARAM["fp16"] == 2.0
    assert constants.BYTES_PER_PARAM["q4"] == 0.5


def test_pipeline_stages_well_formed():
    from cosmos77_ex05.cli.main import build_parser

    build_parser()
    assert "hardware" in constants.PIPELINE_STAGES
    assert "economics" in constants.PIPELINE_STAGES
