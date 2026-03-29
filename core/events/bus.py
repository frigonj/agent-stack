"""
core/events/bus.py
──────────────────
Redis Streams event bus for short-term agent memory and inter-agent messaging.

Every agent action, tool call, and result is an event. Events are ephemeral
by design — meaningful findings get promoted to Emrys long-term memory.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, AsyncIterator, Optional

import redis.asyncio as aioredis
import structlog

log = structlog.get_logger()


class EventType(str, Enum):
    # Lifecycle
    TASK_CREATED      = "task.created"
    TASK_ASSIGNED     = "task.assigned"
    TASK_COMPLETED    = "task.completed"
    TASK_FAILED       = "task.failed"

    # Agent actions
    AGENT_STARTED     = "agent.started"
    AGENT_THINKING    = "agent.thinking"
    AGENT_TOOL_CALL   = "agent.tool_call"
    AGENT_TOOL_RESULT = "agent.tool_result"
    AGENT_RESPONSE    = "agent.response"

    # Memory
    MEMORY_PROMOTED   = "memory.promoted"   # Short → long-term (Emrys)
    CONTEXT_RECOVERED = "context.recovered"

    # Approval gates
    APPROVAL_REQUIRED = "approval.required"
    APPROVAL_GRANTED  = "approval.granted"
    APPROVAL_DENIED   = "approval.denied"

    # Parallel tasks
    TASK_SPAWNED         = "task.spawned"         # sub-task dispatched by orchestrator

    # Proactive reasoning
    THINK_CYCLE          = "think.cycle"           # orchestrator autonomous think cycle

    # Self-modification
    SELF_MODIFY_PROPOSED = "self.modify.proposed"  # agent proposes a code change
    SELF_MODIFY_APPLIED  = "self.modify.applied"   # code change applied + restart requested

    # System
    HEARTBEAT         = "system.heartbeat"
    ERROR             = "system.error"


@dataclass
class Event:
    type: EventType
    source: str                          # Agent role that produced this event
    payload: dict[str, Any]
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: Optional[int] = None   # None = use stream MAXLEN, else explicit

    def to_redis(self) -> dict[str, str]:
        """Serialize for Redis XADD (all values must be strings)."""
        return {
            "event_id":   self.event_id,
            "type":       self.type.value,
            "source":     self.source,
            "task_id":    self.task_id,
            "timestamp":  str(self.timestamp),
            "payload":    json.dumps(self.payload),
        }

    @classmethod
    def from_redis(cls, data: dict[str, Any]) -> "Event":
        return cls(
            event_id=data["event_id"],
            type=EventType(data["type"]),
            source=data["source"],
            task_id=data["task_id"],
            timestamp=float(data["timestamp"]),
            payload=json.loads(data["payload"]),
        )


class EventBus:
    """
    Thin async wrapper around Redis Streams.

    Streams are per-agent-role:
        agents:orchestrator
        agents:document_qa
        agents:code_search
        agents:executor

    Plus a global broadcast stream:
        agents:broadcast
    """

    STREAM_PREFIX = "agents"
    BROADCAST_STREAM = "agents:broadcast"
    MAX_STREAM_LEN = 10_000          # Per stream — keeps Redis footprint bounded

    def __init__(self, redis_url: str):
        self._url = redis_url
        self._client: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        self._client = await aioredis.from_url(
            self._url,
            encoding="utf-8",
            decode_responses=True,
        )
        log.info("event_bus.connected", url=self._url)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()

    def _stream_key(self, target: str) -> str:
        return f"{self.STREAM_PREFIX}:{target}"

    async def publish(
        self,
        event: Event,
        target: str = "broadcast",
    ) -> str:
        """Publish an event to a stream. Returns the Redis stream entry ID."""
        key = self._stream_key(target)
        entry_id = await self._client.xadd(
            key,
            event.to_redis(),
            maxlen=self.MAX_STREAM_LEN,
            approximate=True,
        )
        log.debug(
            "event_bus.published",
            stream=key,
            event_type=event.type,
            event_id=event.event_id,
            entry_id=entry_id,
        )
        return entry_id

    async def consume(
        self,
        role: str,
        group: str,
        consumer: str,
        streams: Optional[list[str]] = None,
        block_ms: int = 1000,
        count: int = 10,
    ) -> AsyncIterator[tuple[str, str, Event]]:
        """
        Yield (stream, entry_id, Event) tuples from a consumer group.
        Creates the group and stream if they don't exist.
        """
        target_streams = streams or [role, "broadcast"]
        keys = [self._stream_key(s) for s in target_streams]

        # Ensure consumer groups exist
        for key in keys:
            try:
                await self._client.xgroup_create(key, group, id="0", mkstream=True)
            except aioredis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise

        stream_ids = {k: ">" for k in keys}

        while True:
            results = await self._client.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams=stream_ids,
                count=count,
                block=block_ms,
            )
            if not results:
                continue
            for stream_key, messages in results:
                for entry_id, data in messages:
                    event = Event.from_redis(data)
                    yield stream_key, entry_id, event

    async def ack(self, stream: str, group: str, entry_id: str) -> None:
        """Acknowledge a processed message."""
        await self._client.xack(stream, group, entry_id)

    async def publish_and_ack(
        self,
        event: Event,
        target: str,
        ack_stream: str,
        ack_group: str,
        ack_entry_id: str,
    ) -> None:
        """Publish a response event and ack the triggering event atomically."""
        async with self._client.pipeline(transaction=True) as pipe:
            key = self._stream_key(target)
            await pipe.xadd(key, event.to_redis(), maxlen=self.MAX_STREAM_LEN, approximate=True)
            await pipe.xack(ack_stream, ack_group, ack_entry_id)
            await pipe.execute()

    async def wait_for_approval(self, approval_id: str, timeout: float = 300.0) -> str:
        """
        Poll for an approval decision set by the Discord bridge.
        Returns 'approved' or 'denied'. Defaults to 'denied' on timeout.
        """
        key = f"approval:{approval_id}"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            val = await self._client.get(key)
            if val is not None:
                await self._client.delete(key)
                return val
            await asyncio.sleep(2)
        return "denied"

    async def get_stream_info(self, role: str) -> dict:
        key = self._stream_key(role)
        info = await self._client.xinfo_stream(key)
        return info
