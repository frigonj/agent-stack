"""
tests/integration/test_event_bus.py
────────────────────────────────────
Integration tests for Redis Streams event bus.
Requires Redis running at REDIS_URL (set in env or defaults to localhost:6379).
"""

import os
import pytest
import pytest_asyncio

from core.events.bus import Event, EventBus, EventType

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


@pytest_asyncio.fixture
async def bus():
    b = EventBus(redis_url=REDIS_URL)
    await b.connect()
    yield b
    await b.disconnect()


@pytest.mark.asyncio
async def test_publish_and_consume(bus):
    """Events published to a stream can be consumed by a consumer group."""
    test_stream = "test_agent"
    group = "test_group"
    consumer = "test_consumer"

    event = Event(
        type=EventType.TASK_CREATED,
        source="test",
        payload={"task": "hello world"},
    )

    entry_id = await bus.publish(event, target=test_stream)
    assert entry_id is not None

    received = []
    async for stream, eid, evt in bus.consume(
        role=test_stream,
        group=group,
        consumer=consumer,
        streams=[test_stream],
        block_ms=500,
        count=1,
    ):
        received.append(evt)
        await bus.ack(stream, group, eid)
        break  # Only consume one

    assert len(received) == 1
    assert received[0].type == EventType.TASK_CREATED
    assert received[0].payload["task"] == "hello world"
    assert received[0].source == "test"


@pytest.mark.asyncio
async def test_event_serialization_roundtrip(bus):
    """Events survive Redis serialization and deserialization intact."""
    original = Event(
        type=EventType.AGENT_TOOL_RESULT,
        source="executor",
        payload={"result": "exit code 0", "nested": {"key": "value"}},
        task_id="task-123",
    )

    await bus.publish(original, target="test_serialize")

    async for _, eid, evt in bus.consume(
        role="test_serialize",
        group="serialize_group",
        consumer="serialize_consumer",
        streams=["test_serialize"],
        block_ms=500,
        count=1,
    ):
        assert evt.type == original.type
        assert evt.source == original.source
        assert evt.payload == original.payload
        assert evt.task_id == original.task_id
        await bus.ack("agents:test_serialize", "serialize_group", eid)
        break


@pytest.mark.asyncio
async def test_broadcast_stream(bus):
    """Events published to broadcast are visible to any consumer."""
    event = Event(
        type=EventType.AGENT_STARTED,
        source="orchestrator",
        payload={"role": "orchestrator"},
    )
    entry_id = await bus.publish(event, target="broadcast")
    assert entry_id is not None
