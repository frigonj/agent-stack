"""
tests/perf/conftest.py
───────────────────────
Shared fixtures and CLI options for the perf test suite.

Adds:
  --update-perf   Overwrite perf_baseline.json with current measurements
  --perf-url-redis  Override Redis URL (default: redis://localhost:6379)
  --perf-url-db     Override DATABASE_URL (default: postgresql://agent:agent@localhost:5432/agentmem)
  --perf-url-lm     Override LM Studio URL (default: http://localhost:1234)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_BASELINE_FILE = Path(__file__).parent / "perf_baseline.json"


def pytest_addoption(parser):
    parser.addoption(
        "--update-perf",
        action="store_true",
        default=False,
        help="Overwrite tests/perf/perf_baseline.json with current measurements.",
    )
    parser.addoption(
        "--perf-url-redis",
        default="redis://localhost:6379",
        help="Redis URL for perf tests (default: redis://localhost:6379)",
    )
    parser.addoption(
        "--perf-url-db",
        default="postgresql://agent:agent@localhost:5432/agentmem",
        help="DATABASE_URL for perf tests",
    )
    parser.addoption(
        "--perf-url-lm",
        default="http://localhost:1234",
        help="LM Studio base URL for perf tests",
    )


@pytest.fixture(scope="session")
def update_perf(request) -> bool:
    return request.config.getoption("--update-perf")


@pytest.fixture(scope="session")
def perf_redis_url(request) -> str:
    return request.config.getoption("--perf-url-redis")


@pytest.fixture(scope="session")
def perf_db_url(request) -> str:
    return request.config.getoption("--perf-url-db")


@pytest.fixture(scope="session")
def perf_lm_url(request) -> str:
    return request.config.getoption("--perf-url-lm")


@pytest.fixture(scope="session")
def perf_baseline() -> dict:
    """
    Load the shared perf_baseline.json. Tests update this dict in-place
    and call _save_baseline() to persist changes.
    Session-scoped so all tests in a run share the same in-memory dict.
    """
    if _BASELINE_FILE.exists():
        return json.loads(_BASELINE_FILE.read_text())
    return {}
