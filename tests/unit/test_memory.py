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


# ── TTL / expiry ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_with_ttl(mem):
    """store() with ttl_days passes the TTL value into the SQL."""
    mock_pool, mock_conn, _ = _make_pool_mock()

    with (
        patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)),
        patch("core.memory.long_term._embed_texts", return_value=[[0.1] * 384]),
    ):
        result = await mem.store(
            content="Temporary finding",
            topic="debug",
            ttl_days=7,
        )

    assert result == {"status": "stored"}
    call_args = mock_conn.execute.call_args_list
    # Last call is the INSERT; its params tuple must include ttl_days value (7) twice
    params = call_args[-1][0][1]
    ttl_positions = [p for p in params if p == 7]
    assert len(ttl_positions) == 2, (
        "ttl_days should appear twice in INSERT params (CASE WHEN)"
    )


@pytest.mark.asyncio
async def test_store_without_ttl_is_permanent(mem):
    """store() with no ttl_days passes NULL for expires_at."""
    mock_pool, mock_conn, _ = _make_pool_mock()

    with (
        patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)),
        patch("core.memory.long_term._embed_texts", return_value=[[0.1] * 384]),
    ):
        await mem.store(content="Permanent finding", topic="core")

    params = mock_conn.execute.call_args_list[-1][0][1]
    # Both ttl_days slots should be None
    assert params[-2] is None
    assert params[-1] is None


@pytest.mark.asyncio
async def test_batch_store_with_ttl(mem):
    """batch_store propagates ttl_days from each entry dict."""
    captured_rows = []

    async def _fake_executemany(sql, rows):
        captured_rows.extend(rows)

    entries = [
        {"content": "Ephemeral", "topic": "debug", "ttl_days": 3},
        {"content": "Permanent", "topic": "core"},
    ]
    mock_pool, mock_conn, _ = _make_pool_mock()

    # cursor() is used as an async context manager in batch_store;
    # AsyncMock auto-creates child mocks, so wire executemany on the cursor.
    cur_mock = mock_conn.cursor.return_value.__aenter__.return_value
    cur_mock.executemany = _fake_executemany

    with (
        patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)),
        patch(
            "core.memory.long_term._embed_texts",
            return_value=[[0.1] * 384, [0.2] * 384],
        ),
    ):
        result = await mem.batch_store(entries)

    assert result["count"] == 2
    assert len(captured_rows) == 2
    ephemeral_row, permanent_row = captured_rows
    # ttl_days=3 appears at positions -2 and -1 in each row tuple
    assert ephemeral_row[-2] == 3
    assert ephemeral_row[-1] == 3
    assert permanent_row[-2] is None
    assert permanent_row[-1] is None


@pytest.mark.asyncio
async def test_recall_excludes_expired(mem):
    """recall() SQL must filter out expired entries."""
    mock_pool, mock_conn, _ = _make_pool_mock(fetchall_result=[])

    with patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)):
        await mem.recall("some query")

    sql = mock_conn.execute.call_args[0][0]
    assert "expires_at" in sql
    assert "expires_at > NOW()" in sql


@pytest.mark.asyncio
async def test_expire_knowledge_deletes_stale(mem):
    """expire_knowledge() issues a DELETE for expired rows and returns the count."""
    expired_rows = [{"id": 1}, {"id": 2}]
    mock_pool, mock_conn, mock_cur = _make_pool_mock(fetchall_result=expired_rows)

    with patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)):
        deleted = await mem.expire_knowledge()

    assert deleted == 2
    sql = mock_conn.execute.call_args[0][0]
    assert "DELETE FROM knowledge" in sql
    assert "expires_at <= NOW()" in sql


@pytest.mark.asyncio
async def test_cleanup_stale_includes_expired_knowledge(mem):
    """cleanup_stale() calls expire_knowledge() and includes its count in the result."""
    mock_pool, mock_conn, mock_cur = _make_pool_mock(fetchall_result=[])

    with (
        patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)),
        patch.object(mem, "expire_knowledge", new=AsyncMock(return_value=5)),
    ):
        result = await mem.cleanup_stale()

    assert "knowledge_expired" in result
    assert result["knowledge_expired"] == 5


# ── set_expiry ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_expiry_timed(mem):
    """set_expiry() issues an UPDATE with an INTERVAL for timed sizes."""
    mock_pool, mock_conn, mock_cur = _make_pool_mock(fetchone_result={"id": 42})

    with patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)):
        updated = await mem.set_expiry(42, "medium")

    assert updated is True
    sql = mock_conn.execute.call_args[0][0]
    assert "UPDATE knowledge" in sql
    assert "expires_at" in sql
    params = mock_conn.execute.call_args[0][1]
    assert "7 days" in params  # medium maps to "7 days"
    assert 42 in params


@pytest.mark.asyncio
async def test_set_expiry_permanent(mem):
    """set_expiry('permanent') sets expires_at to NULL."""
    mock_pool, mock_conn, mock_cur = _make_pool_mock(fetchone_result={"id": 7})

    with patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)):
        updated = await mem.set_expiry(7, "permanent")

    assert updated is True
    sql = mock_conn.execute.call_args[0][0]
    assert "expires_at = NULL" in sql


@pytest.mark.asyncio
async def test_set_expiry_missing_row(mem):
    """set_expiry() returns False when the row no longer exists."""
    mock_pool, mock_conn, mock_cur = _make_pool_mock(fetchone_result=None)

    with patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)):
        updated = await mem.set_expiry(999, "short")

    assert updated is False


# ── store returns id ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_returns_id(mem):
    """store() now returns the inserted row id."""
    mock_pool, mock_conn, mock_cur = _make_pool_mock(fetchone_result={"id": 55})

    with (
        patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)),
        patch("core.memory.long_term._embed_texts", return_value=[[0.1] * 384]),
    ):
        result = await mem.store(content="some finding", topic="test")

    assert result["id"] == 55
    assert result["status"] == "stored"


@pytest.mark.asyncio
async def test_batch_store_returns_ids(mem):
    """batch_store() returns all inserted row ids."""
    entries = [
        {"content": "A", "topic": "x"},
        {"content": "B", "topic": "y"},
    ]
    # Each execute() call returns a cursor whose fetchone returns the next id
    id_sequence = [{"id": 10}, {"id": 11}]
    call_count = 0

    async def fetchone_side_effect():
        nonlocal call_count
        result = id_sequence[call_count % len(id_sequence)]
        call_count += 1
        return result

    mock_pool, mock_conn, mock_cur = _make_pool_mock()
    mock_cur.fetchone = fetchone_side_effect

    with (
        patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)),
        patch(
            "core.memory.long_term._embed_texts",
            return_value=[[0.1] * 384, [0.2] * 384],
        ),
    ):
        result = await mem.batch_store(entries)

    assert result["count"] == 2
    assert result["ids"] == [10, 11]


# ── get_unclassified ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_unclassified_returns_old_null_rows(mem):
    """get_unclassified() queries for expires_at IS NULL rows older than 1 minute."""
    rows = [
        {
            "id": 1,
            "topic": "t",
            "content": "c",
            "tags": [],
            "agent": "x",
            "created_at": None,
        }
    ]
    mock_pool, mock_conn, mock_cur = _make_pool_mock(fetchall_result=rows)

    with patch.object(mem, "_get_pool", new=AsyncMock(return_value=mock_pool)):
        result = await mem.get_unclassified()

    assert result == rows
    sql = mock_conn.execute.call_args[0][0]
    assert "expires_at IS NULL" in sql
    assert "1 minute" in sql
