"""
tests/unit/test_events.py
──────────────────────────
Unit tests for event models and serialization.
No Redis required — pure logic tests.
"""

import json
import pytest
from core.events.bus import Event, EventType


def test_event_to_redis_all_strings():
    """Redis requires all values to be strings."""
    event = Event(
        type=EventType.TASK_CREATED,
        source="orchestrator",
        payload={"task": "do something", "priority": 1},
        task_id="task-abc",
    )
    serialized = event.to_redis()

    for key, val in serialized.items():
        assert isinstance(val, str), f"Field '{key}' is not a string: {type(val)}"


def test_event_roundtrip():
    """Events serialize and deserialize without data loss."""
    original = Event(
        type=EventType.AGENT_TOOL_CALL,
        source="executor",
        payload={"command": "ls -la", "nested": {"a": 1, "b": [1, 2, 3]}},
        task_id="task-xyz",
    )
    serialized = original.to_redis()
    restored = Event.from_redis(serialized)

    assert restored.type == original.type
    assert restored.source == original.source
    assert restored.task_id == original.task_id
    assert restored.payload == original.payload
    assert restored.event_id == original.event_id


def test_event_type_values():
    """EventType values are stable strings (not auto-generated)."""
    assert EventType.TASK_CREATED == "task.created"
    assert EventType.MEMORY_PROMOTED == "memory.promoted"
    assert EventType.AGENT_TOOL_RESULT == "agent.tool_result"


def test_event_unique_ids():
    """Each event gets a unique event_id and task_id by default."""
    e1 = Event(type=EventType.HEARTBEAT, source="system", payload={})
    e2 = Event(type=EventType.HEARTBEAT, source="system", payload={})

    assert e1.event_id != e2.event_id
    assert e1.task_id != e2.task_id


def test_event_payload_preserves_types():
    """Nested payload types survive JSON roundtrip."""
    payload = {
        "string": "hello",
        "integer": 42,
        "float": 3.14,
        "list": [1, 2, 3],
        "nested": {"key": "value"},
        "bool": True,
        "null": None,
    }
    event = Event(type=EventType.TASK_COMPLETED, source="agent", payload=payload)
    restored = Event.from_redis(event.to_redis())

    assert restored.payload == payload


def test_event_from_redis_missing_optional_fields():
    """Events can be restored even if optional fields are absent."""
    minimal = {
        "event_id": "eid-001",
        "type": "task.created",
        "source": "orchestrator",
        "task_id": "tid-001",
        "timestamp": "1700000000.0",
        "payload": json.dumps({"task": "minimal"}),
    }
    event = Event.from_redis(minimal)
    assert event.type == EventType.TASK_CREATED
    assert event.payload["task"] == "minimal"
