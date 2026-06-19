"""Shared pytest fixtures and deterministic-seed setup (CLAUDE.md rule 17)."""

from __future__ import annotations

import random

import pytest


@pytest.fixture(autouse=True)
def _seed_random() -> None:
    """Seed `random` before every test so nothing flakes."""
    random.seed(1729)
