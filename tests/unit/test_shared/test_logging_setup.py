"""Tests for logging initialisation (rule 13 support)."""

from __future__ import annotations

import json
import logging

from cosmos77_ex05.shared.logging_setup import (
    _ensure_handler_dirs,
    get_logger,
    init_logging,
)


def test_get_logger_namespace():
    logger = get_logger("cosmos77_ex05.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "cosmos77_ex05.test"


def test_ensure_handler_dirs_creates_paths(tmp_path):
    file_target = tmp_path / "logs" / "app.log"
    dir_target = tmp_path / "spool"
    _ensure_handler_dirs(
        {
            "handlers": {
                "file": {"filename": str(file_target)},
                "dir": {"directory": str(dir_target)},
                "console": {},
            }
        }
    )
    assert file_target.parent.exists()
    assert dir_target.exists()


def test_init_logging_applies_console_config(tmp_path):
    cfg = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"s": {"format": "%(message)s"}},
        "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "s"}},
        "root": {"level": "WARNING", "handlers": ["console"]},
    }
    path = tmp_path / "logging.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    init_logging(path)  # must not raise
    get_logger().warning("ok")
