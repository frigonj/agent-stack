"""
tests/unit/test_config.py
──────────────────────────
Unit tests for Settings — env var loading and defaults.
"""

import os
import pytest
from core.config import Settings


def test_default_values():
    """Settings have sensible defaults."""
    s = Settings()
    assert s.redis_url == "redis://localhost:6379"
    assert s.lm_studio_url == "http://host.docker.internal:1234"
    assert s.lm_studio_model == "qwen2.5-14b"
    assert s.log_level == "INFO"


def test_env_override(monkeypatch):
    """Environment variables override defaults."""
    monkeypatch.setenv("REDIS_URL", "redis://myredis:6380")
    monkeypatch.setenv("LM_STUDIO_MODEL", "mistral-nemo-12b")
    monkeypatch.setenv("AGENT_ROLE", "code_search")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    s = Settings()
    assert s.redis_url == "redis://myredis:6380"
    assert s.lm_studio_model == "mistral-nemo-12b"
    assert s.agent_role == "code_search"
    assert s.log_level == "DEBUG"


def test_case_insensitive_env(monkeypatch):
    """Settings are case-insensitive for env vars."""
    monkeypatch.setenv("log_level", "WARNING")
    s = Settings()
    assert s.log_level == "WARNING"
