"""
tests/unit/test_memory.py
──────────────────────────
Unit tests for long-term memory client logic.
Uses httpx mocking — no Emrys server required.
"""

import json
import pytest
import pytest_asyncio
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from core.memory.long_term import LongTermMemory


@pytest_asyncio.fixture
async def mem():
    m = LongTermMemory(emrys_url="http://mock-emrys:8000", agent_name="test_agent")
    yield m
    await m.close()


@pytest.mark.asyncio
async def test_open_session(mem):
    """open_session calls the correct Emrys endpoint."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"session_id": "sess-001", "status": "CLEAN"}
    mock_response.raise_for_status = MagicMock()

    with patch.object(mem._client, "post", new=AsyncMock(return_value=mock_response)) as mock_post:
        result = await mem.open_session()

    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert "open_session" in call_args[0][0]
    assert result["session_id"] == "sess-001"
    assert mem._session_id == "sess-001"


@pytest.mark.asyncio
async def test_store_knowledge(mem):
    """store() calls store_knowledge with correct payload."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "kb-123", "stored": True}
    mock_response.raise_for_status = MagicMock()

    with patch.object(mem._client, "post", new=AsyncMock(return_value=mock_response)) as mock_post:
        result = await mem.store(
            content="Auth retry uses exponential backoff",
            topic="bugfix",
            tags=["auth", "retry"],
        )

    call_kwargs = mock_post.call_args[1]
    body = call_kwargs["json"]
    assert body["arguments"]["content"] == "Auth retry uses exponential backoff"
    assert body["arguments"]["topic"] == "bugfix"
    assert "auth" in body["arguments"]["tags"]


@pytest.mark.asyncio
async def test_recall(mem):
    """recall() returns parsed entries from Emrys."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "entries": [
            {"id": "kb-1", "content": "Auth retry fix", "topic": "bugfix"},
            {"id": "kb-2", "content": "Auth token refresh", "topic": "bugfix"},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(mem._client, "post", new=AsyncMock(return_value=mock_response)):
        entries = await mem.recall("auth token")

    assert len(entries) == 2
    assert entries[0]["id"] == "kb-1"


@pytest.mark.asyncio
async def test_search_falls_back_to_fulltext_on_vector_error(mem):
    """search() falls back to full-text recall when vector search fails."""
    mock_recall_response = MagicMock()
    mock_recall_response.status_code = 200
    mock_recall_response.json.return_value = {"entries": [{"id": "kb-1", "content": "fallback result"}]}
    mock_recall_response.raise_for_status = MagicMock()

    mock_vector_response = MagicMock()
    mock_vector_response.raise_for_status = MagicMock(side_effect=Exception("vectors not installed"))

    responses = [mock_vector_response, mock_recall_response]

    with patch.object(mem._client, "post", new=AsyncMock(side_effect=responses)):
        entries = await mem.search("some query", semantic=True)

    assert len(entries) == 1
    assert entries[0]["content"] == "fallback result"


@pytest.mark.asyncio
async def test_close_session(mem):
    """close_session calls write_handoff and clears session_id."""
    mem._session_id = "sess-001"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"written": True}
    mock_response.raise_for_status = MagicMock()

    with patch.object(mem._client, "post", new=AsyncMock(return_value=mock_response)) as mock_post:
        await mem.close_session(summary="Task complete")

    call_kwargs = mock_post.call_args[1]
    assert "write_handoff" in mock_post.call_args[0][0]
    assert mem._session_id is None


@pytest.mark.asyncio
async def test_batch_store(mem):
    """batch_store() sends all entries in a single call."""
    entries = [
        {"content": "Finding 1", "topic": "bugfix", "tags": ["auth"]},
        {"content": "Finding 2", "topic": "performance", "tags": ["db"]},
        {"content": "Finding 3", "topic": "architecture", "tags": ["design"]},
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"stored": 3}
    mock_response.raise_for_status = MagicMock()

    with patch.object(mem._client, "post", new=AsyncMock(return_value=mock_response)) as mock_post:
        result = await mem.batch_store(entries)

    assert mock_post.call_count == 1
    call_kwargs = mock_post.call_args[1]
    assert len(call_kwargs["json"]["arguments"]["entries"]) == 3
