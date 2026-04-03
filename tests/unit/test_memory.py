"""
tests/unit/test_memory.py
──────────────────────────
Unit tests for LongTermMemory.
Uses psycopg + psycopg_pool mocking — no live PostgreSQL required.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.memory.long_term import LongTermMemory


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def mem():
    m = LongTermMemory(
        database_url="postgresql://agent:agent@localhost:5432/agentmem",
        agent_name="test_agent",
    )
    yield m
    # Don't call close() — pool was never opened in unit tests


def _make_pool_mock(fetchone_result=None, fetchall_result=None):
    """Return a mock AsyncConnectionPool whose context manager yields a mock conn."""
    mock_cur = AsyncMock()
    mock_cur.fetchone = AsyncMock(return_value=fetchone_result)
    mock_cur.fetchall = AsyncMock(return_value=fetchall_result or [])

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_cur)
    mock_conn.commit = AsyncMock()
    mock_conn.rollback = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    mock_pool = AsyncMock()
    mock_pool.connection = MagicMock(return_value=mock_conn)
    mock_pool.__aenter__ = AsyncMock(return_value=mock_pool)
    mock_pool.__aexit__ = AsyncMock(return_value=False)
    return mock_pool, mock_conn, mock_cur


# ── Constructor ───────────────────────────────────────────────────────────────


def test_init():
    """LongTermMemory stores url and agent name."""
    m = LongTermMemory(
        database_url="postgresql://user:pass@host/db",
        agent_name="executor",
    )
    assert m._url == "postgresql://user:pass@host/db"
    assert m._agent == "executor"
    assert m._pool is None


# ── open_session ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_open_session_clean(mem):
    """open_session returns status=OK when agent was cleanly shut down."""
    mock_pool, mock_conn, mock_cur = _make_pool_mock(
        fetchone_result={"status": "CLEAN"}
    )

    with patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)):
        result = await mem.open_session()

    assert result["status"] == "OK"
    assert result["agent"] == "test_agent"


@pytest.mark.asyncio
async def test_open_session_crash(mem):
    """open_session returns status=CRASH when previous session was OPEN."""
    mock_pool, mock_conn, mock_cur = _make_pool_mock(fetchone_result={"status": "OPEN"})

    with patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)):
        result = await mem.open_session()

    assert result["status"] == "CRASH"


@pytest.mark.asyncio
async def test_open_session_new_agent(mem):
    """open_session returns OK when no prior row exists."""
    mock_pool, mock_conn, mock_cur = _make_pool_mock(fetchone_result=None)

    with patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)):
        result = await mem.open_session()

    assert result["status"] == "OK"


# ── store ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store(mem):
    """store() inserts a knowledge row and returns status=stored."""
    mock_pool, mock_conn, _ = _make_pool_mock()

    with (
        patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)),
        patch("core.memory.long_term._embed_texts", return_value=[[0.1] * 384]),
    ):
        result = await mem.store(
            content="Auth retry uses exponential backoff",
            topic="bugfix",
            tags=["auth", "retry"],
        )

    assert result == {"status": "stored"}
    mock_conn.execute.assert_called()
    call_args = mock_conn.execute.call_args_list
    sql = call_args[-1][0][0]
    assert "INSERT INTO knowledge" in sql


# ── close_session ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_session(mem):
    """close_session inserts a handoff and sets status to CLEAN."""
    mock_pool, mock_conn, _ = _make_pool_mock()
    mem._pool = mock_pool  # skip _get_pool

    with patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)):
        await mem.close_session(summary="Task complete")

    calls = [c[0][0] for c in mock_conn.execute.call_args_list]
    assert any("INSERT INTO handoffs" in sql for sql in calls)
    assert any("status = 'CLEAN'" in sql for sql in calls)


# ── batch_store ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_batch_store_empty(mem):
    """batch_store with empty list does nothing and returns 0."""
    mock_pool, _, _ = _make_pool_mock()

    with patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)):
        result = await mem.batch_store([])

    assert result == {"status": "stored", "count": 0}


@pytest.mark.asyncio
async def test_batch_store_entries(mem):
    """batch_store inserts all entries and returns count."""
    entries = [
        {"content": "Finding 1", "topic": "bugfix", "tags": ["auth"]},
        {"content": "Finding 2", "topic": "performance", "tags": ["db"]},
    ]
    mock_pool, mock_conn, _ = _make_pool_mock()
    mock_conn.executemany = AsyncMock()

    with (
        patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)),
        patch(
            "core.memory.long_term._embed_texts",
            return_value=[[0.1] * 384, [0.2] * 384],
        ),
    ):
        result = await mem.batch_store(entries)

    assert result["count"] == 2
    assert result["status"] == "stored"
