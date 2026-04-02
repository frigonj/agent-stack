"""
tests/conftest.py
──────────────────
Shared test fixtures and import-time patches.

The psycopg C extension and native libpq are not available in all CI
environments. This conftest stubs them out before any test module is
imported so that unit/integration tests that mock the DB layer still
collect and run correctly.

Tests that require a REAL database connection should be marked
@pytest.mark.integration_db and the CI step should provide Postgres.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock


def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = None
    return mod


# ── Stub psycopg native extensions if not installed ──────────────────────────
# This allows the test suite to collect and run without a Postgres/libpq
# installation.  Tests that actually hit the DB must ensure psycopg is
# available (e.g., inside the Docker dev stack).

if "psycopg" not in sys.modules:
    psycopg = _make_stub("psycopg")
    psycopg.__path__ = []   # mark as package so sub-imports work
    # Attrs referenced at import time in long_term.py
    psycopg.AsyncConnection = MagicMock()
    psycopg.AsyncClientCursor = MagicMock()
    psycopg.OperationalError = type("OperationalError", (Exception,), {})
    sys.modules["psycopg"] = psycopg

    psycopg_errors = _make_stub("psycopg.errors")
    psycopg_errors.UniqueViolation = type("UniqueViolation", (Exception,), {})
    sys.modules["psycopg.errors"] = psycopg_errors
    psycopg.errors = psycopg_errors

    psycopg_rows = _make_stub("psycopg.rows")
    psycopg_rows.dict_row = MagicMock()
    sys.modules["psycopg.rows"] = psycopg_rows
    sys.modules["psycopg_pool"] = _make_stub("psycopg_pool")

    pool_mod = sys.modules["psycopg_pool"]
    pool_mod.AsyncConnectionPool = MagicMock()

if "pgvector" not in sys.modules:
    pgvector = _make_stub("pgvector")
    sys.modules["pgvector"] = pgvector
    sys.modules["pgvector.psycopg"] = _make_stub("pgvector.psycopg")

# discord.py — optional; only the discord_bridge needs it
if "discord" not in sys.modules:
    discord_stub = _make_stub("discord")
    discord_stub.Client = MagicMock()
    discord_stub.Intents = MagicMock()
    sys.modules["discord"] = discord_stub
    sys.modules["discord.ext"] = _make_stub("discord.ext")
    sys.modules["discord.ext.commands"] = _make_stub("discord.ext.commands")

# langchain_anthropic — optional Claude fallback
if "langchain_anthropic" not in sys.modules:
    sys.modules["langchain_anthropic"] = _make_stub("langchain_anthropic")
