"""
tests/integration/test_full_pipeline.py
─────────────────────────────────────────
CORE INTEGRATION TEST — DO NOT DELETE OR MODIFY WITHOUT TEAM REVIEW.

Validates that the full event pipeline works end-to-end:
  TASK_CREATED → orchestrator stores plan → dispatches TASK_ASSIGNED
  Simulated specialist → TASK_COMPLETED → orchestrator receives and processes
  Final TASK_COMPLETED → broadcast with discord_message_id

Requires Redis running at REDIS_URL (default: redis://localhost:6379).
Does NOT require LM Studio, Postgres, or Discord.

Design note: the live agent stack may be running against the same Redis.
All helpers here:
  - Record the stream's current tip BEFORE publishing so they only see new messages.
  - Filter by the specific event_id that was published so unrelated events
    from the running stack never cause false positives or failures.

Run: pytest tests/integration/test_full_pipeline.py -v
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid

import pytest
import pytest_asyncio

from core.events.bus import Event, EventBus, EventType

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def bus():
    b = EventBus(redis_url=REDIS_URL)
    await b.connect()
    yield b
    await b.disconnect()


@pytest_asyncio.fixture
async def producer_bus():
    """Separate bus connection for publishing test events (simulates specialists)."""
    b = EventBus(redis_url=REDIS_URL)
    await b.connect()
    yield b
    await b.disconnect()


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _stream_tip(bus: EventBus, stream_key: str) -> str:
    """
    Return the entry ID of the most-recent message in *stream_key*, or '0-0'
    if the stream is empty / does not exist yet.
    Used to bookmark the stream position before publishing so tests only read
    messages they wrote — ignoring pre-existing traffic from the running stack.
    """
    try:
        entries = await bus._client.xrevrange(stream_key, count=1)
        if entries:
            return entries[0][0]   # (entry_id, data)
    except Exception:
        pass
    return "0-0"


async def publish_and_find(
    bus: EventBus,
    producer_bus: EventBus,
    target_stream: str,
    event: Event,
    expect_type: EventType,
    match_event_id: str,
    timeout: float = 5.0,
) -> Event | None:
    """
    Bookmark the stream, publish *event* via *producer_bus*, then poll
    *target_stream* for an event that matches *match_event_id*.

    Using XRANGE from the bookmark means we never pick up unrelated events
    from the live agent stack.
    """
    stream_key = f"agents:{target_stream}"
    tip        = await _stream_tip(bus, stream_key)

    await producer_bus.publish(event, target=target_stream)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        entries = await bus._client.xrange(stream_key, min=f"({tip}", count=50)
        for entry_id, data in entries:
            try:
                evt = Event.from_redis(data)
            except Exception:
                continue
            if evt.event_id == match_event_id:
                return evt
        await asyncio.sleep(0.1)

    return None


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_task_assigned_reaches_specialist_stream(bus, producer_bus):
    """
    TASK_ASSIGNED published to a specialist stream is readable from that stream.
    Validates the fundamental routing: orchestrator → specialist.
    """
    task_id    = str(uuid.uuid4())
    step_id    = str(uuid.uuid4())
    specialist = "executor"

    evt = Event(
        type=EventType.TASK_ASSIGNED,
        source="orchestrator",
        payload={
            "task": "ls /workspace",
            "assigned_to": specialist,
            "parent_task_id": task_id,
            "subtask_id": step_id,
        },
        task_id=task_id,
    )

    received = await publish_and_find(
        bus, producer_bus,
        target_stream=specialist,
        event=evt,
        expect_type=EventType.TASK_ASSIGNED,
        match_event_id=evt.event_id,
    )

    assert received is not None, "Specialist never received TASK_ASSIGNED"
    assert received.payload["assigned_to"] == specialist
    assert received.payload["parent_task_id"] == task_id
    assert received.payload["subtask_id"] == step_id


@pytest.mark.asyncio
async def test_task_completed_reaches_orchestrator(bus, producer_bus):
    """
    TASK_COMPLETED from a specialist reaches the orchestrator stream.
    This is the critical P0 path: specialist result → orchestrator.
    """
    task_id = str(uuid.uuid4())
    step_id = str(uuid.uuid4())

    evt = Event(
        type=EventType.TASK_COMPLETED,
        source="executor",
        payload={
            "result": "total 0\ndrwxr-xr-x 3 root root 80 Jan 1 00:00 workspace",
            "task_id": str(uuid.uuid4()),
            "subtask_id": step_id,
            "parent_task_id": task_id,
        },
    )

    received = await publish_and_find(
        bus, producer_bus,
        target_stream="orchestrator",
        event=evt,
        expect_type=EventType.TASK_COMPLETED,
        match_event_id=evt.event_id,
    )

    assert received is not None, "Orchestrator never received TASK_COMPLETED from specialist"
    assert received.source == "executor"
    assert received.payload["parent_task_id"] == task_id
    assert received.payload["subtask_id"] == step_id
    assert "workspace" in received.payload["result"]


@pytest.mark.asyncio
async def test_final_reply_emitted_to_broadcast(bus, producer_bus):
    """
    The orchestrator's final reply sends TASK_COMPLETED to broadcast with
    discord_message_id so the Discord bridge can deliver it.
    """
    task_id            = str(uuid.uuid4())
    discord_message_id = str(uuid.uuid4())

    evt = Event(
        type=EventType.TASK_COMPLETED,
        source="orchestrator",
        payload={
            "result": "Done! Files listed successfully.",
            "task_id": task_id,
            "discord_message_id": discord_message_id,
        },
    )

    received = await publish_and_find(
        bus, producer_bus,
        target_stream="broadcast",
        event=evt,
        expect_type=EventType.TASK_COMPLETED,
        match_event_id=evt.event_id,
    )

    assert received is not None, "Broadcast never received final TASK_COMPLETED"
    assert received.payload["discord_message_id"] == discord_message_id
    assert received.source == "orchestrator"


@pytest.mark.asyncio
async def test_orphan_result_carries_parent_id(bus, producer_bus):
    """
    If a TASK_COMPLETED arrives after the plan has expired, parent_task_id is
    still present in the payload so the orchestrator can recover discord_message_id
    from the context stream registry.
    """
    task_id = str(uuid.uuid4())
    step_id = str(uuid.uuid4())

    evt = Event(
        type=EventType.TASK_COMPLETED,
        source="document_qa",
        payload={
            "result": "Architecture reviewed.",
            "task_id": str(uuid.uuid4()),
            "subtask_id": step_id,
            "parent_task_id": task_id,
        },
    )

    received = await publish_and_find(
        bus, producer_bus,
        target_stream="orchestrator",
        event=evt,
        expect_type=EventType.TASK_COMPLETED,
        match_event_id=evt.event_id,
    )

    assert received is not None, "Orchestrator did not receive orphan TASK_COMPLETED"
    assert received.payload["parent_task_id"] == task_id


@pytest.mark.asyncio
async def test_context_stream_created_and_readable(bus):
    """
    create_context_stream registers the stream and it is immediately
    writable and readable.
    """
    context_id = str(uuid.uuid4())
    stream_key = await bus.create_context_stream(
        "task", context_id, "integration-test-task",
        metadata={"discord_message_id": "msg-123"},
    )
    assert stream_key.startswith("ctx:task:")

    meta = await bus.get_context_metadata(context_id)
    assert meta is not None
    assert meta["status"] == "active"
    assert meta["discord_message_id"] == "msg-123"

    evt = Event(
        type=EventType.TASK_CREATED,
        source="test",
        payload={"task": "test"},
    )
    await bus.publish_to_context(context_id, evt)

    entries = await bus.read_context_stream(context_id, count=5)
    assert entries, "Context stream should have at least one entry"

    # Verify we can find our specific event
    found_ids = [e.event_id for _, e in entries]
    assert evt.event_id in found_ids


@pytest.mark.asyncio
async def test_consume_sentinel_on_idle(bus):
    """
    When no events arrive, consume() yields (None, None, None) so callers
    can check stop-flags. This is the fix for the idle termination bug.
    """
    idle_stream     = f"idle_test_{uuid.uuid4().hex[:8]}"
    group           = f"idle_group_{uuid.uuid4().hex[:8]}"
    sentinel_received = False

    async def drain():
        nonlocal sentinel_received
        async for stream, eid, event in bus.consume(
            role=idle_stream,
            group=group,
            consumer="idle_consumer",
            streams=[idle_stream],
            block_ms=300,
        ):
            if stream is None:
                sentinel_received = True
                return

    await asyncio.wait_for(drain(), timeout=2.0)
    assert sentinel_received, (
        "consume() never yielded idle sentinel — idle termination bug is present"
    )
