"""
core/events/bus.py
──────────────────
Redis Streams event bus for short-term agent memory and inter-agent messaging.

Every agent action, tool call, and result is an event. Events are ephemeral
by design — meaningful findings get promoted to Emrys long-term memory.

Context streams
───────────────
In addition to per-role streams (agents:{role}), the bus manages named
context streams that represent a single task, chat session, or other unit
of work.  These are the source of truth for correlation — no more
parent_task_id threading through payloads.

  ctx:task:{id}:{slug}   — all events for one task
  ctx:chat:{id}:{slug}   — all events for one chat session
  ctx:plan:{id}:{slug}   — all events for one execution plan

A lightweight registry (Redis Hash  ctx:registry) maps context_id → metadata
so the orchestrator and agents can enumerate and resume contexts.

Configuration
─────────────
Runtime-adjustable values live in Redis under the key prefix  config: .
Any agent or the orchestrator may call get_config / set_config to read or
update them.  Changes take effect on the next read — no restart needed.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

import redis.asyncio as aioredis
import structlog

log = structlog.get_logger()


# ── Defaults for runtime-configurable values ──────────────────────────────────

CONFIG_DEFAULTS: dict[str, Any] = {
    "vote_timeout_ms": 20_000,  # ms agents have to vote on a plan (allows LLM evaluation)
    "vote_max_extension_ms": 15_000,  # max extra ms one agent can request (allows clarification round-trip)
    # vote_min_quorum is computed dynamically as ceil(n_voters / 2) — not a config value
    "vote_max_retries": 3,  # retry attempts before escalating to the user
    "vote_user_timeout_hours": 48,  # hours before a user-escalated vote auto-rejects
    "chat_idle_gap_secs": 1_800,  # 30 min idle → new chat session
    "chat_keyword_overlap": 0.4,  # Jaccard threshold for same session
    "max_concurrent_contexts": 5,  # per-agent context stream pool size
    "topic_confidence_ask": 0.8,  # below this → ask user for category
    "topic_min_sessions": 50,  # sessions before asking user at all
    "max_fix_depth": 10,  # max strategy-rotating fix attempts
}


def _slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9\-_ ]", "", name)
    name = re.sub(r"\s+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name[:40] or "context"


class EventType(str, Enum):
    # Lifecycle
    TASK_CREATED = "task.created"
    TASK_ASSIGNED = "task.assigned"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"

    # Agent actions
    AGENT_STARTED = "agent.started"
    AGENT_THINKING = "agent.thinking"
    AGENT_TOOL_CALL = "agent.tool_call"
    AGENT_TOOL_RESULT = "agent.tool_result"
    AGENT_RESPONSE = "agent.response"

    # Memory
    MEMORY_PROMOTED = "memory.promoted"  # Short → long-term (Emrys)
    CONTEXT_RECOVERED = "context.recovered"

    # Approval gates
    APPROVAL_REQUIRED = "approval.required"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_DENIED = "approval.denied"

    # Parallel tasks
    TASK_SPAWNED = "task.spawned"  # sub-task dispatched by orchestrator

    # Fix subtask (strategy-rotating retry on failure)
    TASK_FIX_SPAWNED = "task.fix_spawned"  # failed step spawned a fix subtask

    # Proactive reasoning
    THINK_CYCLE = "think.cycle"  # orchestrator autonomous think cycle

    # Self-modification
    SELF_MODIFY_PROPOSED = "self.modify.proposed"  # agent proposes a code change
    SELF_MODIFY_APPLIED = (
        "self.modify.applied"  # code change applied + restart requested
    )

    # Session management
    SESSION_RESET = "session.reset"  # wipe conversation/chat context

    # Discord management (agent → bridge)
    DISCORD_ACTION = "discord.action"  # request a Discord API action
    DISCORD_ACTION_DONE = "discord.action.done"  # bridge confirms action completed

    # Memory lifecycle
    MEMORY_PRUNED = "memory.pruned"  # long-term memory pruned; user warned
    MEMORY_CLASSIFY = "memory.classify"  # unclassified knowledge row needs TTL assigned

    # Plan execution lifecycle
    PLAN_STATUS = "plan.status"  # step/phase progress update → Discord
    PLAN_PROPOSED = "plan.proposed"  # orchestrator broadcasts plan before executing
    AGENT_VOTE = "agent.vote"  # agent approves/rejects a proposed plan
    VOTE_EXTENSION_REQUESTED = (
        "vote.extension_requested"  # agent needs more deliberation time
    )
    VOTE_RESULT = "vote.result"  # orchestrator posts tally after voting closes
    VOTE_CLARIFICATION_REQUEST = (
        "vote.clarification_request"  # agent asks orchestrator to explain a plan step
    )
    VOTE_CLARIFICATION_RESPONSE = (
        "vote.clarification_response"  # orchestrator answers the clarification
    )
    PEER_VOTE_REQUESTED = (
        "peer.vote_requested"  # any agent requests a general peer vote
    )
    VOTE_INITIATED = (
        "vote.initiated"  # orchestrator broadcasts a peer vote to all agents
    )
    VOTE_ESCALATED_TO_USER = (
        "vote.escalated_to_user"  # quorum not met after retries → user decides
    )
    VOTE_USER_RESULT = "vote.user_result"  # user cast their vote via Discord

    # Pause / resume lifecycle
    TASK_PAUSED = (
        "task.paused"  # research agent finished current iteration, loop stopped
    )
    TASK_RESUMED = "task.resumed"  # user requested continuation from last checkpoint
    TASK_UPDATED = "task.updated"  # user edited the original Discord message while task is in-flight

    # Knowledge gap lifecycle
    KNOWLEDGE_GAP = "knowledge.gap"  # agent lacks info needed to complete a task
    KNOWLEDGE_TEACH = "knowledge.teach"  # user supplies a source to resolve a gap

    # Memory approval lifecycle
    # Fired when an agent wants to store a memory with TTL > 24h.
    # The orchestrator routes this to Discord; the user approves/denies via reaction.
    # On timeout (2 min) the entry defaults to "short" (24h).
    MEMORY_APPROVAL_REQUESTED = "memory.approval_requested"
    MEMORY_APPROVAL_RESULT = "memory.approval_result"

    # Context stream lifecycle
    CONTEXT_CREATED = (
        "context.created"  # orchestrator announces new named context stream
    )
    CONTEXT_CLOSED = "context.closed"  # context stream closed + snapshot saved
    CONTEXT_SNAPSHOT = "context.snapshot"  # rolling mid-execution snapshot

    # Chat session lifecycle
    CHAT_SESSION_STARTED = "chat.session_started"
    CHAT_SESSION_CLOSED = "chat.session_closed"

    # Topic classification (agent-learned categories)
    TOPIC_CLASSIFIED = (
        "topic.classified"  # agent assigned a topic category to a context
    )

    # Runtime configuration
    CONFIG_UPDATED = "config.updated"  # agent/orchestrator updated a config value

    # Lifecycle control
    AGENT_SHUTDOWN = "agent.shutdown"  # orchestrator tells an ephemeral agent to exit

    # Checkpoint / fork
    CHECKPOINT_REACHED = "checkpoint.reached"  # agent marks a named task boundary

    # System
    HEARTBEAT = "system.heartbeat"
    ERROR = "system.error"


@dataclass
class Event:
    type: EventType
    source: str  # Agent role that produced this event
    payload: dict[str, Any]
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: Optional[int] = None  # None = use stream MAXLEN, else explicit

    def to_redis(self) -> dict[str, str]:
        """Serialize for Redis XADD (all values must be strings)."""
        return {
            "event_id": self.event_id,
            "type": self.type.value,
            "source": self.source,
            "task_id": self.task_id,
            "timestamp": str(self.timestamp),
            "payload": json.dumps(self.payload),
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

    Role streams (per-agent dispatch):
        agents:orchestrator
        agents:document_qa
        agents:code_search
        agents:executor

    Broadcast stream:
        agents:broadcast

    Context streams (per-task/session source of truth):
        ctx:task:{id}:{slug}
        ctx:chat:{id}:{slug}
        ctx:plan:{id}:{slug}

    Context registry (Redis Hash):
        ctx:registry  →  context_id → JSON metadata
    """

    STREAM_PREFIX = "agents"
    BROADCAST_STREAM = "agents:broadcast"
    CONTEXT_PREFIX = "ctx"
    CONTEXT_REGISTRY = "ctx:registry"
    MAX_STREAM_LEN = 10_000  # Per stream — keeps Redis footprint bounded
    CONFIG_PREFIX = "config"

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

    # ── Role stream publish / consume ─────────────────────────────────────────

    async def publish(
        self,
        event: Event,
        target: str = "broadcast",
    ) -> str:
        """Publish an event to a role/broadcast stream. Returns the Redis entry ID."""
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
    ) -> AsyncIterator[tuple[str | None, str | None, "Event | None"]]:
        """
        Yield (stream, entry_id, Event) tuples from a consumer group.
        Creates the group and stream if they don't exist.

        When no messages arrive within *block_ms* a ``(None, None, None)``
        sentinel is yielded so callers can check a stop-flag without being
        permanently blocked in the ``async for`` loop.  Callers must guard:

            if stream is None:   # idle sentinel
                continue
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

        _reconnect_delay = 1.0
        while True:
            try:
                results = await self._client.xreadgroup(
                    groupname=group,
                    consumername=consumer,
                    streams=stream_ids,
                    count=count,
                    block=block_ms,
                )
                _reconnect_delay = 1.0  # reset on success
            except (aioredis.ConnectionError, aioredis.TimeoutError) as exc:
                log.warning(
                    "event_bus.consume_reconnect",
                    role=role,
                    error=str(exc),
                    retry_in=_reconnect_delay,
                )
                yield None, None, None  # let caller check _running
                await asyncio.sleep(_reconnect_delay)
                _reconnect_delay = min(_reconnect_delay * 2, 30.0)
                # Re-ensure consumer groups after reconnect
                for key in keys:
                    try:
                        await self._client.xgroup_create(
                            key, group, id="0", mkstream=True
                        )
                    except aioredis.ResponseError as e:
                        if "BUSYGROUP" not in str(e):
                            raise
                continue
            if not results:
                # Yield a heartbeat sentinel so the caller can react to stop
                # signals without waiting for a real event to unblock the loop.
                yield None, None, None
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
            await pipe.xadd(
                key, event.to_redis(), maxlen=self.MAX_STREAM_LEN, approximate=True
            )
            await pipe.xack(ack_stream, ack_group, ack_entry_id)
            await pipe.execute()

    # ── Context stream management ─────────────────────────────────────────────

    def _context_stream_key(self, context_type: str, context_id: str, name: str) -> str:
        return f"{self.CONTEXT_PREFIX}:{context_type}:{context_id}:{_slugify(name)}"

    async def create_context_stream(
        self,
        context_type: str,
        context_id: str,
        name: str,
        metadata: dict | None = None,
    ) -> str:
        """
        Register a new named context stream and return its stream key.
        Idempotent — safe to call multiple times with the same context_id.

        context_type: 'task' | 'chat' | 'plan'
        context_id:   stable UUID for this context
        name:         human-readable slug (will be sanitised)
        metadata:     arbitrary extra fields stored in the registry
        """
        stream_key = self._context_stream_key(context_type, context_id, name)
        existing = await self._client.hget(self.CONTEXT_REGISTRY, context_id)
        if existing:
            return json.loads(existing)["stream"]

        entry = {
            "type": context_type,
            "id": context_id,
            "name": name,
            "stream": stream_key,
            "status": "active",
            "created_at": time.time(),
            **(metadata or {}),
        }
        await self._client.hset(self.CONTEXT_REGISTRY, context_id, json.dumps(entry))
        # Touch the stream so it exists even before the first event
        try:
            await self._client.xgroup_create(
                stream_key, "readers", id="0", mkstream=True
            )
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        log.debug(
            "event_bus.context_created", type=context_type, id=context_id, name=name
        )
        return stream_key

    async def publish_to_context(self, context_id: str, event: Event) -> str:
        """
        Append an event to the context stream identified by context_id.
        Returns the Redis entry ID, or empty string if context not found.
        """
        meta = await self.get_context_metadata(context_id)
        if not meta:
            log.warning("event_bus.context_not_found", context_id=context_id)
            return ""
        stream_key = meta["stream"]
        entry_id = await self._client.xadd(
            stream_key,
            event.to_redis(),
            maxlen=self.MAX_STREAM_LEN,
            approximate=True,
        )
        return entry_id

    async def get_context_metadata(self, context_id: str) -> dict | None:
        """Return registry metadata for a context, or None if not found."""
        data = await self._client.hget(self.CONTEXT_REGISTRY, context_id)
        return json.loads(data) if data else None

    async def update_context_metadata(self, context_id: str, **fields) -> None:
        """Merge fields into an existing context registry entry."""
        data = await self._client.hget(self.CONTEXT_REGISTRY, context_id)
        if not data:
            return
        entry = json.loads(data)
        entry.update(fields)
        await self._client.hset(self.CONTEXT_REGISTRY, context_id, json.dumps(entry))

    async def close_context(self, context_id: str) -> None:
        """Mark a context as closed in the registry."""
        await self.update_context_metadata(
            context_id, status="closed", closed_at=time.time()
        )
        log.debug("event_bus.context_closed", context_id=context_id)

    async def list_active_contexts(self, context_type: str | None = None) -> list[dict]:
        """
        Return all registered contexts sorted newest-first.
        Optionally filter by context_type ('task', 'chat', 'plan').
        """
        all_data = await self._client.hgetall(self.CONTEXT_REGISTRY)
        contexts = [json.loads(v) for v in all_data.values()]
        if context_type:
            contexts = [c for c in contexts if c.get("type") == context_type]
        return sorted(contexts, key=lambda c: c.get("created_at", 0), reverse=True)

    async def read_context_stream(
        self,
        context_id: str,
        count: int = 50,
        last_id: str = "0",
    ) -> list[tuple[str, Event]]:
        """
        Read up to *count* messages from a context stream.
        Returns list of (entry_id, Event) tuples.

        Used by agents to check history before asking the user for clarification.
        """
        meta = await self.get_context_metadata(context_id)
        if not meta:
            return []
        stream_key = meta["stream"]
        try:
            results = await self._client.xrange(stream_key, min=last_id, count=count)
            return [(entry_id, Event.from_redis(data)) for entry_id, data in results]
        except Exception as exc:
            log.warning(
                "event_bus.context_read_failed", context_id=context_id, error=str(exc)
            )
            return []

    async def consume_context_stream(
        self,
        context_id: str,
        group: str,
        consumer: str,
        block_ms: int = 1000,
        count: int = 10,
    ) -> AsyncIterator[tuple[str, str, Event]]:
        """
        Consume new messages from a context stream via a consumer group.
        Yields (stream_key, entry_id, Event).  Used by agents subscribing to
        a specific context as part of the multi-context pool.
        """
        meta = await self.get_context_metadata(context_id)
        if not meta:
            return
        stream_key = meta["stream"]
        try:
            await self._client.xgroup_create(stream_key, group, id="$", mkstream=False)
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        stream_ids = {stream_key: ">"}
        while True:
            results = await self._client.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams=stream_ids,
                count=count,
                block=block_ms,
            )
            if not results:
                yield  # signal idle; caller can break if context is closed
                continue
            for sk, messages in results:
                for entry_id, data in messages:
                    event = Event.from_redis(data)
                    yield sk, entry_id, event

    # ── Runtime configuration ─────────────────────────────────────────────────

    async def get_config(self, key: str, default: Any = None) -> Any:
        """
        Read a runtime-configurable value from Redis.
        Falls back to CONFIG_DEFAULTS, then to *default*.
        """
        val = await self._client.get(f"{self.CONFIG_PREFIX}:{key}")
        if val is not None:
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return val
        return CONFIG_DEFAULTS.get(key, default)

    async def set_config(self, key: str, value: Any) -> None:
        """
        Write a runtime-configurable value to Redis.
        Persists until explicitly cleared.  Publish CONFIG_UPDATED so listeners
        can react immediately without polling.
        """
        await self._client.set(f"{self.CONFIG_PREFIX}:{key}", json.dumps(value))
        log.info("event_bus.config_updated", key=key, value=value)

    # ── Approval helpers ──────────────────────────────────────────────────────

    async def wait_for_approval(self, approval_id: str, timeout: float = 300.0) -> str:
        """
        Block until an approval decision is pushed by the Discord bridge.

        The bridge calls set_approval() which writes the decision to a Redis
        list (`approval:notify:{id}`). This method uses BLPOP to wake up
        immediately instead of polling every 2 s — reduces Redis round-trips
        from ~150 to 1 per approval, and cuts median latency to near zero.

        Returns 'approved' or 'denied'. Defaults to 'denied' on timeout.
        """
        notify_key = f"approval:notify:{approval_id}"
        result = await self._client.blpop(notify_key, timeout=int(timeout))
        if result is not None:
            _key, decision = result
            return decision
        return "denied"

    async def set_approval(
        self, approval_id: str, decision: str, ex: int = 600
    ) -> None:
        """
        Record an approval decision and unblock any waiter.
        """
        key = f"approval:{approval_id}"
        notify_key = f"approval:notify:{approval_id}"
        await self._client.set(key, decision, ex=ex)
        await self._client.lpush(notify_key, decision)
        await self._client.expire(notify_key, ex)

    # ── Peer-vote result helpers ──────────────────────────────────────────────

    async def wait_for_vote_result(self, vote_id: str, timeout: float = 120.0) -> str:
        """
        Block until the orchestrator pushes the final outcome for *vote_id*.

        notify_vote_result() is called by the orchestrator after VOTE_RESULT is
        broadcast.  Returns 'approved' or 'rejected'.  Defaults to 'rejected'
        on timeout.
        """
        notify_key = f"vote:notify:{vote_id}"
        result = await self._client.blpop(notify_key, timeout=int(timeout))
        if result is not None:
            _key, outcome = result
            return outcome.decode() if isinstance(outcome, bytes) else outcome
        return "rejected"

    async def notify_vote_result(
        self, vote_id: str, outcome: str, ex: int = 300
    ) -> None:
        """
        Push the final outcome so any wait_for_vote_result() waiter unblocks.
        Called by the orchestrator at the end of _run_peer_vote().
        """
        notify_key = f"vote:notify:{vote_id}"
        await self._client.lpush(notify_key, outcome)
        await self._client.expire(notify_key, ex)

    # ── Checkpoint / fork helpers ─────────────────────────────────────────────

    async def copy_context_stream_to(
        self,
        source_context_id: str,
        dest_context_id: str,
        cutoff_timestamp: float,
    ) -> int:
        """
        Copy events from a source context stream into a destination context stream,
        stopping at (and including) the last event at or before *cutoff_timestamp*.

        Copied events are written verbatim with an extra payload field
        ``_copied_from`` so the consuming agent knows they are immutable history
        and must not be re-executed.

        Returns the number of events copied.
        """
        src_meta = await self.get_context_metadata(source_context_id)
        dst_meta = await self.get_context_metadata(dest_context_id)
        if not src_meta or not dst_meta:
            log.warning(
                "event_bus.copy_stream_missing_context",
                source=source_context_id,
                dest=dest_context_id,
            )
            return 0

        src_key = src_meta["stream"]
        dst_key = dst_meta["stream"]

        # Read all entries from the source stream (up to MAX_STREAM_LEN)
        entries = await self._client.xrange(src_key, min="-", max="+")
        copied = 0
        for entry_id, data in entries:
            event = Event.from_redis(data)
            if event.timestamp > cutoff_timestamp:
                break
            # Mark as copied history so agents skip re-execution
            payload_with_marker = {**event.payload, "_copied_from": source_context_id}
            copied_event = Event(
                type=event.type,
                source=event.source,
                payload=payload_with_marker,
                task_id=event.task_id,
                event_id=event.event_id,
                timestamp=event.timestamp,
            )
            await self._client.xadd(
                dst_key,
                copied_event.to_redis(),
                maxlen=self.MAX_STREAM_LEN,
                approximate=True,
            )
            copied += 1

        log.info(
            "event_bus.stream_copied",
            source=source_context_id,
            dest=dest_context_id,
            events_copied=copied,
            cutoff=cutoff_timestamp,
        )
        return copied

    # ── Misc helpers ──────────────────────────────────────────────────────────

    async def get_stream_info(self, role: str) -> dict:
        key = self._stream_key(role)
        info = await self._client.xinfo_stream(key)
        return info
