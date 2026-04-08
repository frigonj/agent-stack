"""
core/base_agent.py
──────────────────
Base class for all agents in the stack.

Every agent:
  1. Connects to Redis Streams (short-term memory / event bus)
  2. Connects to Emrys (long-term memory)
  3. Connects to LM Studio (inference)
  4. Opens a session with crash detection
  5. Listens for events on its role stream
  6. Promotes meaningful findings to long-term memory on session close
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
import uuid
from abc import ABC, abstractmethod
import re
from typing import Any, Awaitable, Callable, Optional

import httpx
import structlog
from langchain_core.messages import (
    AIMessage,
    HumanMessage as _HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

from core.events.bus import Event, EventBus, EventType
from core.errors import AgentError
from core.memory.long_term import LongTermMemory
from core.config import Settings
from core.context import (
    truncate_memory_entries,
    MEMORY_WARN_THRESHOLD,
    MEMORY_HARD_LIMIT,
    MEMORY_PRUNE_TARGET,
)

# ── Token estimation ──────────────────────────────────────────────────────────
# Try tiktoken for accurate counts; fall back to chars//4 if unavailable.
try:
    import tiktoken

    _tok_enc = tiktoken.get_encoding("cl100k_base")  # good approximation for Qwen3

    def _count_tokens(text: str) -> int:
        return len(_tok_enc.encode(text, disallowed_special=()))
except ImportError:

    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return len(text) // 4


log = structlog.get_logger()

# ── Agent loop action parsing ─────────────────────────────────────────────────
# Matches lines like "CMD: docker ps" or "SEARCH: EventType" or "READ: /workspace/docs/x.pdf"
_ACTION_RE = re.compile(
    r"^(?P<prefix>CMD|SEARCH|READ):\s*(?P<payload>.+)$", re.MULTILINE
)
# Matches "DONE: <answer>" — the LLM signals it is finished
_DONE_RE = re.compile(r"^DONE:\s*(?P<answer>.+)", re.MULTILINE | re.DOTALL)


class BaseAgent(ABC):
    """
    All agents inherit from this. Subclasses implement:
      - handle_event(event) → the agent's core logic per event
      - on_startup()        → optional async init
      - on_shutdown()       → optional async cleanup + memory promotion
      - think()             → optional proactive reasoning (called every think_interval seconds)
    """

    # Seconds between proactive think() calls. Override in subclasses.
    think_interval: int = 300

    # Seconds of inactivity before the agent self-exits (0 = never).
    # Overridden at runtime by the IDLE_TIMEOUT environment variable.
    idle_timeout: int = 0

    def __init__(self, settings: Settings):
        self.settings = settings
        self.role = settings.agent_role
        self.consumer_name = f"{self.role}_consumer"
        self.group_name = f"{self.role}_group"

        self.bus = EventBus(redis_url=settings.redis_url)
        self.memory = LongTermMemory(
            database_url=settings.database_url,
            agent_name=self.role,
        )
        self.llm = ChatOpenAI(
            base_url=f"{settings.lm_studio_url}/v1",
            api_key="lm-studio",  # LM Studio doesn't require a real key
            model=settings.lm_studio_model,
            temperature=0.1,
            streaming=True,
            extra_body={"thinking": False},
        )

        self._running = False
        self._stopped = False
        self._findings: list[dict] = []  # Staged for promotion on shutdown
        self._stop_event: Optional[asyncio.Event] = None
        # context_id → asyncio.Task for multi-context consumer pool
        self._context_tasks: dict[str, asyncio.Task] = {}
        # Tracks fire-and-forget _handle_and_ack tasks so stop() can drain them
        self._handler_tasks: set[asyncio.Task] = set()
        self._model_context_limit: Optional[int] = None  # cached from LM Studio
        self._model_context_limit_ts: float = 0.0  # monotonic time of last fetch
        self._last_event_time: float = time.monotonic()
        # Counts concurrently running event-handler tasks.
        # Status is only set to "idle" when this reaches zero.
        self._active_tasks: int = 0

        # IDLE_TIMEOUT env var overrides the class-level default
        env_idle = os.environ.get("IDLE_TIMEOUT", "")
        self._idle_timeout = int(env_idle) if env_idle.isdigit() else self.idle_timeout

        # ── Pending vote clarification requests ───────────────────────────────
        # plan_id → asyncio.Event (set when response arrives)
        self._clarification_events: dict[str, asyncio.Event] = {}
        # plan_id → answer string
        self._clarification_answers: dict[str, str] = {}

        # ── Circuit breaker state ─────────────────────────────────────────────
        # Tracks consecutive LM Studio failures. After CIRCUIT_THRESHOLD failures
        # the circuit opens and llm_invoke() fails fast (or uses Claude fallback).
        self._circuit_failures: int = 0
        self._circuit_open: bool = False
        self._circuit_open_since: float = 0.0

        # Optional Claude API fallback (only active when ANTHROPIC_API_KEY is set)
        _api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if _api_key:
            try:
                from langchain_anthropic import ChatAnthropic

                self._claude_fallback: Optional[Any] = ChatAnthropic(
                    model=settings.claude_fallback_model,
                    api_key=_api_key,
                    temperature=0.1,
                    max_tokens=2048,
                )
                log.info("agent.claude_fallback_ready", role=self.role)
            except Exception:
                self._claude_fallback = None
        else:
            self._claude_fallback = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        log.info("agent.starting", role=self.role)

        self._stop_event = asyncio.Event()

        await self.bus.connect()
        await self.memory.open_session()

        # Recover context if crashed
        recovery = await self.memory.recover_context()
        if recovery.get("status") == "CRASH":
            log.warning("agent.crash_recovery", role=self.role, context=recovery)

        await self.on_startup()
        self._running = True

        # ACK any stale PEL entries from a previous crashed run so they don't
        # permanently block the think loop via _queue_is_idle()
        await self._drain_stale_pel()

        # Announce presence
        await self.bus.publish(
            Event(
                type=EventType.AGENT_STARTED,
                source=self.role,
                payload={"role": self.role},
            ),
            target="broadcast",
        )

        log.info("agent.ready", role=self.role)
        await self._set_status("idle")
        await asyncio.gather(
            self._event_loop(),
            self._think_loop(),
            self._idle_watchdog_loop(),
        )

    async def stop(self, summary: Optional[str] = None) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._running = False
        if self._stop_event:
            self._stop_event.set()
        log.info("agent.stopping", role=self.role)

        await self.on_shutdown()

        # Cancel all active context stream consumer tasks
        for ctx_id, task in list(self._context_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._context_tasks.clear()

        # Drain in-flight _handle_and_ack tasks — give them up to 10s to finish
        # before cancelling so ACKs and error events are not lost.
        if self._handler_tasks:
            pending = [t for t in self._handler_tasks if not t.done()]
            if pending:
                _, still_pending = await asyncio.wait(pending, timeout=10.0)
                for t in still_pending:
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
        self._handler_tasks.clear()

        # Promote any staged findings to long-term memory
        if self._findings:
            await self.memory.batch_store(self._findings)
            log.info("agent.promoted_findings", count=len(self._findings))

        await self.memory.close_session(
            summary=summary or f"{self.role} session ended cleanly"
        )
        await self.bus.disconnect()
        await self.memory.close()
        log.info("agent.stopped", role=self.role)

    # ── Multi-context consumer pool ──────────────────────────────────────────
    # In addition to the primary role stream, each agent can subscribe to up to
    # MAX_CONCURRENT_CONTEXTS individual context streams simultaneously.
    # Non-LLM work (I/O, shell commands) genuinely parallelises; LLM calls still
    # serialize via the distributed llm:lock so the GPU is never overloaded.

    async def subscribe_to_context(self, context_id: str) -> bool:
        """
        Add a context stream to this agent's active consumer pool.
        Returns False if the pool is already at capacity or context not found.
        """
        max_ctx = int(await self.bus.get_config("max_concurrent_contexts", 5))
        if len(self._context_tasks) >= max_ctx:
            log.warning(
                "agent.context_pool_full",
                role=self.role,
                active=len(self._context_tasks),
                max=max_ctx,
            )
            return False
        if context_id in self._context_tasks:
            return True  # already subscribed
        meta = await self.bus.get_context_metadata(context_id)
        if not meta:
            return False
        task = asyncio.create_task(
            self._context_consumer_loop(context_id),
            name=f"{self.role}-ctx-{context_id[:8]}",
        )
        self._context_tasks[context_id] = task
        log.info("agent.context_subscribed", role=self.role, context_id=context_id)
        return True

    async def unsubscribe_from_context(self, context_id: str) -> None:
        """Remove a context stream from the pool (e.g. when the context closes)."""
        task = self._context_tasks.pop(context_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        log.debug("agent.context_unsubscribed", role=self.role, context_id=context_id)

    async def _context_consumer_loop(self, context_id: str) -> None:
        """
        Consume events from a specific context stream.
        Calls handle_context_event() (override in subclasses) for each event.
        Exits when the context is closed or the agent stops.
        """
        group = f"{self.role}_ctx_group"
        consumer = f"{self.role}_ctx_{context_id[:8]}"
        async for item in self.bus.consume_context_stream(
            context_id, group, consumer, block_ms=500
        ):
            if not self._running:
                break
            if item is None:
                # idle yield — check if context was closed
                meta = await self.bus.get_context_metadata(context_id)
                if meta and meta.get("status") == "closed":
                    log.debug("agent.context_stream_closed", context_id=context_id)
                    break
                continue
            stream_key, entry_id, event = item
            self._last_event_time = time.monotonic()
            asyncio.create_task(
                self._handle_context_and_ack(stream_key, entry_id, event, context_id)
            )

    async def _handle_context_and_ack(
        self, stream: str, entry_id: str, event: "Event", context_id: str
    ) -> None:
        group = f"{self.role}_ctx_group"
        try:
            await self.handle_context_event(event, context_id)
        except Exception as exc:
            log.error(
                "agent.context_event_error",
                role=self.role,
                context_id=context_id,
                error=str(exc),
            )
            await self._emit_error(event, exc)
            return
        await self.bus.ack(stream, group, entry_id)

    async def handle_context_event(self, event: "Event", context_id: str) -> None:
        """
        Override in subclasses to react to events on a subscribed context stream.
        Default: forward to handle_event() so agents without specific context
        handling still process the event normally.
        """
        await self.handle_event(event)

    # ── Event loop ───────────────────────────────────────────────────────────

    async def _drain_stale_pel(self) -> None:
        """
        ACK any messages sitting in this consumer's PEL (pending entry list) that
        were delivered to the previous incarnation of this container but never
        ACKed (e.g. due to a crash or SIGKILL). Without this, pel-count stays > 0
        and _queue_is_idle() always returns False, permanently blocking the think loop.

        Uses XAUTOCLAIM with min-idle-time=0 to reclaim and immediately ACK all
        such entries. The events themselves were almost certainly partially-processed
        and are safe to drop — the task that generated them will time-out and be retried.
        """
        streams = [self.role, "broadcast"]
        for stream_name in streams:
            key = self.bus._stream_key(stream_name)
            try:
                # XAUTOCLAIM <key> <group> <consumer> <min-idle-ms> <start-id>
                # Returns (next_id, [[entry_id, fields], ...], [deleted_ids])
                result = await self.bus._client.xautoclaim(
                    key,
                    self.group_name,
                    self.consumer_name,
                    min_idle_time=0,
                    start_id="0-0",
                )
                # result[1] is the list of claimed entries
                claimed = result[1] if result and len(result) > 1 else []
                if claimed:
                    entry_ids = [e[0] for e in claimed if e]
                    await self.bus._client.xack(key, self.group_name, *entry_ids)
                    log.info(
                        "agent.stale_pel_drained",
                        role=self.role,
                        stream=stream_name,
                        count=len(entry_ids),
                    )
            except Exception as exc:
                # XAUTOCLAIM requires Redis 6.2+; fall back gracefully
                log.debug(
                    "agent.stale_pel_drain_skipped", stream=stream_name, error=str(exc)
                )

    async def _queue_is_idle(self) -> bool:
        """
        Return True if there are no unprocessed messages waiting in this
        agent's stream. Uses XINFO GROUPS to read consumer group lag.
        A non-zero lag means there is already work to do — skip the think cycle.
        """
        try:
            key = f"agents:{self.role}"
            groups = await self.bus._client.xinfo_groups(key)
            for g in groups:
                lag = g.get("lag") or g.get("pel-count") or 0
                if int(lag) > 0:
                    return False
            return True
        except Exception:
            return True  # assume idle if we can't check

    async def _think_loop(self) -> None:
        """
        Background loop that calls think() every think_interval seconds.
        Skips the think cycle if there is pending work in the queue —
        avoids making LLM calls that collide with active event processing.
        Exits immediately when stop() is called (via _stop_event).
        """
        while self._running:
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.think_interval,
                )
                break  # stop was requested
            except asyncio.TimeoutError:
                pass  # normal — interval elapsed

            if not self._running:
                break

            # Only run memory health check + think() when the queue is idle
            if not await self._queue_is_idle():
                depth = await self._queue_depth()
                log.debug(
                    "agent.think_skipped_busy",
                    role=self.role,
                    queue_depth=depth,
                    reason=f"{depth} message(s) pending in agents:{self.role}",
                )
                continue

            try:
                await self._apply_log_level_override()
                await self._check_memory_health()
                await self.think()
            except Exception as exc:
                log.error("agent.think_error", role=self.role, error=str(exc))

    async def _idle_watchdog_loop(self) -> None:
        """Exit cleanly after _idle_timeout seconds with no events (0 = disabled)."""
        if not self._idle_timeout:
            return
        # Check interval: at most every 5 s, but never longer than the timeout itself.
        # This means short-timeout agents (e.g. 1 s in tests) are checked frequently
        # without burning CPU in production (e.g. 600 s → check every 5 s).
        check_interval = max(0.5, min(5.0, self._idle_timeout))
        log.info(
            "agent.idle_watchdog_active",
            role=self.role,
            idle_timeout=self._idle_timeout,
            check_interval=check_interval,
        )
        while self._running:
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=check_interval)
                return  # stop was requested
            except asyncio.TimeoutError:
                pass
            if not self._running:
                return
            idle_secs = time.monotonic() - self._last_event_time
            if idle_secs >= self._idle_timeout:
                log.info("agent.idle_exit", role=self.role, idle_secs=int(idle_secs))
                self._running = False
                self._stop_event.set()
                return

    async def _event_loop(self) -> None:
        async for stream, entry_id, event in self.bus.consume(
            role=self.role,
            group=self.group_name,
            consumer=self.consumer_name,
        ):
            if not self._running:
                break
            # consume() yields (None, None, None) when no events arrive within
            # block_ms — this is the heartbeat sentinel that lets us react to a
            # stop signal without being permanently stuck waiting for real events.
            if stream is None:
                continue
            # Only reset the idle clock for events targeted at this agent, not
            # broadcast events.  Broadcast traffic (plan.status, plan.proposed,
            # agent.vote, …) is informational and must not prevent an ephemeral
            # agent from timing out after its own work is done.
            if "broadcast" not in stream:
                self._last_event_time = time.monotonic()
            # Fire-and-forget each event so the consumer loop stays unblocked.
            # Handlers that take a long time (LLM calls, docker spawning) won't
            # delay acknowledgement of subsequent events.
            t = asyncio.create_task(self._handle_and_ack(stream, entry_id, event))
            self._handler_tasks.add(t)
            t.add_done_callback(self._handler_tasks.discard)

    async def _handle_and_ack(self, stream: str, entry_id: str, event: Event) -> None:
        # Orchestrator can send a targeted shutdown to any ephemeral agent.
        # Handle it here so subclasses don't need to; ack before stopping.
        if event.type == EventType.AGENT_SHUTDOWN:
            await self.bus.ack(stream, self.group_name, entry_id)
            log.info("agent.shutdown_received", role=self.role)
            # Wait for any in-flight task handlers to finish before tearing down.
            drain_deadline = time.monotonic() + 120
            while self._active_tasks > 0 and time.monotonic() < drain_deadline:
                await asyncio.sleep(0.5)
            if self._active_tasks > 0:
                log.warning(
                    "agent.shutdown_drain_timeout",
                    role=self.role,
                    active=self._active_tasks,
                )
            await self.stop(summary="shutdown requested by orchestrator")
            return

        if event.type == EventType.PLAN_PROPOSED:
            await self.bus.ack(stream, self.group_name, entry_id)
            asyncio.create_task(self._handle_plan_proposed(event))
            return

        if event.type == EventType.VOTE_INITIATED:
            await self.bus.ack(stream, self.group_name, entry_id)
            asyncio.create_task(self._handle_vote_initiated(event))
            return

        if event.type == EventType.VOTE_CLARIFICATION_RESPONSE:
            await self.bus.ack(stream, self.group_name, entry_id)
            self._handle_clarification_response(event)
            return

        self._active_tasks += 1
        await self._set_status(
            "busy", task=event.payload.get("task", event.type.value)[:80]
        )
        try:
            await self.handle_event(event)
        except Exception as exc:
            log.error(
                "agent.event_error",
                role=self.role,
                error=str(exc),
                event_id=event.event_id,
            )
            await self._emit_error(event, exc)
        finally:
            await self.bus.ack(stream, self.group_name, entry_id)
            self._active_tasks = max(0, self._active_tasks - 1)
            if self._active_tasks == 0:
                await self._set_status("idle")

    _STATUS_KEY_PREFIX = "agent:status"
    _STATUS_TTL = 3600  # 1 hour — auto-expires so stale entries don't linger

    async def _set_status(self, status: str, task: str = "") -> None:
        """Publish agent status to Redis for the control script to query."""
        try:
            payload = json.dumps(
                {
                    "status": status,
                    "task": task,
                    "since": time.time(),
                    "queue": await self._queue_depth(),
                }
            )
            await self.bus._client.setex(
                f"{self._STATUS_KEY_PREFIX}:{self.role}", self._STATUS_TTL, payload
            )
        except Exception as exc:
            log.warning("agent.status_publish_failed", role=self.role, error=str(exc))

    async def _queue_depth(self) -> int:
        """Return the number of pending (unprocessed) messages in this agent's stream."""
        try:
            key = f"agents:{self.role}"
            groups = await self.bus._client.xinfo_groups(key)
            for g in groups:
                lag = g.get("lag") or g.get("pel-count") or 0
                return int(lag)
        except Exception as exc:
            log.warning("agent.queue_depth_failed", role=self.role, error=str(exc))
        return 0

    # ── LLM call management ─────────────────────────────────────────────────

    # Redis key used as a distributed mutex across all agent containers.
    # Only one agent may hold this lock at a time, preventing KV cache exhaustion.
    _LLM_LOCK_KEY = "llm:lock"
    _LLM_LOCK_TTL = 120  # seconds — auto-releases if agent crashes mid-call
    _LLM_LOCK_POLL = 1.5  # seconds between acquire attempts
    _MODEL_CTX_CACHE_TTL = 300  # re-fetch context limit after 5 min (model may reload)

    async def _get_model_context_limit(self) -> int:
        """
        Fetch the loaded model's actual runtime context length from LM Studio.

        Strategy (in priority order):
        1. /v1/models  → context_window field (reflects actual loaded n_ctx)
        2. /api/v0/models → context_length (prefer over max_context_length which
           is the server maximum, not what the model was loaded with)
        3. Fallback: 4096 (conservative safe default)

        Cached after the first successful call.
        """
        if (
            self._model_context_limit is not None
            and time.monotonic() - self._model_context_limit_ts
            < self._MODEL_CTX_CACHE_TTL
        ):
            return self._model_context_limit
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                # Preferred: /v1/models returns context_window = actual loaded n_ctx
                try:
                    r1 = await client.get(f"{self.settings.lm_studio_url}/v1/models")
                    r1.raise_for_status()
                    data1 = r1.json()
                    models1 = (
                        data1 if isinstance(data1, list) else data1.get("data", [])
                    )
                    for m in models1:
                        ctx = m.get("context_window") or m.get("context_length")
                        if ctx:
                            self._model_context_limit = int(ctx)
                            self._model_context_limit_ts = time.monotonic()
                            log.info(
                                "agent.model_context_limit",
                                role=self.role,
                                limit=self._model_context_limit,
                                source="/v1/models",
                            )
                            return self._model_context_limit
                except Exception:
                    pass

                # Fallback: /api/v0/models — use context_length, not max_context_length
                # max_context_length is the server's ceiling, not the loaded model's n_ctx
                r2 = await client.get(f"{self.settings.lm_studio_url}/api/v0/models")
                r2.raise_for_status()
                data2 = r2.json()
                models2 = data2 if isinstance(data2, list) else data2.get("data", [])
                for m in models2:
                    ctx = m.get("context_length") or m.get("max_context_length")
                    if ctx:
                        self._model_context_limit = int(ctx)
                        self._model_context_limit_ts = time.monotonic()
                        log.info(
                            "agent.model_context_limit",
                            role=self.role,
                            limit=self._model_context_limit,
                            source="/api/v0/models",
                        )
                        return self._model_context_limit
        except Exception as exc:
            log.warning("agent.context_limit_fetch_failed", error=str(exc))
        # Conservative fallback: 4096 tokens
        self._model_context_limit = 4096
        self._model_context_limit_ts = time.monotonic()
        return self._model_context_limit

    # ── Circuit breaker constants ────────────────────────────────────────────
    CIRCUIT_THRESHOLD = 3  # consecutive failures before opening
    CIRCUIT_RESET_SECS = 60  # seconds before attempting half-open probe

    def _estimate_tokens(self, messages: list) -> int:
        """Token estimate using tiktoken when available, chars//4 as fallback."""
        total = 0
        for m in messages:
            content = getattr(m, "content", "") or ""
            total += _count_tokens(content)
        return total

    async def _budget_content_chars(
        self, system_msg: str, fixed_content: str = ""
    ) -> int:
        """
        Return the character budget available for variable document/code content.

        Subtracts the system message and any fixed content (task description,
        static instructions) from the model's input budget, leaving room for
        the reply (25% of context window).

        Use this before loading large content (files, source trees, code snippets)
        to avoid sending more tokens than the model can accept.

        Returns at least 1 000 chars so there's always something useful to send.
        """
        ctx_limit = await self._get_model_context_limit()
        input_cap = int(ctx_limit * 0.70)  # reserve 25% for reply + 5% overhead
        sys_tokens = self._estimate_tokens([SystemMessage(content=system_msg)])
        fixed_toks = _count_tokens(fixed_content) if fixed_content else 0
        token_budget = max(250, input_cap - sys_tokens - fixed_toks)
        # Convert token budget to chars conservatively (code/structured text ≈ 3 chars/token)
        available = token_budget * 3
        log.debug(
            "agent.content_budget",
            role=self.role,
            ctx_limit=ctx_limit,
            sys_tokens=sys_tokens,
            fixed_tokens=fixed_toks,
            available_chars=available,
        )
        return available

    # ── Agent action loop (ReAct) ─────────────────────────────────────────────

    # Patterns that indicate an action observation is an error/failure worth retrying
    _ERROR_OBS_RE = re.compile(
        r"\b(error|exception|traceback|command not found|permission denied|"
        r"no such file|failed|exit code [^0]|returncode [^0])\b",
        re.I,
    )

    async def agent_loop(
        self,
        messages: list,
        action_handler: Callable[[str, str], Awaitable[str]],
        max_steps: int = 10,
    ) -> str:
        """
        ReAct-style multi-step loop: Reason → Act → Observe → repeat.

        Self-correction: when an observation indicates failure, the loop injects
        an explicit "that failed, try a different approach" prompt so the LLM
        can diagnose and recover rather than give up or loop blindly.

        The loop ends when the LLM emits ``DONE: <answer>`` (preferred),
        produces a response with no action lines (implicit done), or
        exhausts ``max_steps``.
        """
        msgs = list(messages)
        consecutive_errors = 0
        _MAX_CONSECUTIVE_ERRORS = 3

        for step in range(max_steps):
            response = await self.llm_invoke(msgs)
            content: str = response.content

            # ── Parse action lines ─────────────────────────────────────────
            actions = _ACTION_RE.findall(content)

            # ── Explicit DONE termination ──────────────────────────────────
            # Check DONE *after* actions so that a response with both
            # "CMD: ..." and "DONE: ..." executes the command first.
            done_m = _DONE_RE.search(content)
            if done_m and not actions:
                log.debug("agent_loop.done", role=self.role, step=step)
                return done_m.group("answer").strip()

            if not actions:
                # No action lines and no DONE → treat full response as the final answer
                log.debug("agent_loop.implicit_done", role=self.role, step=step)
                return content.strip()

            # Append the assistant turn
            msgs = msgs + [AIMessage(content=content)]

            # Execute actions sequentially, collect observations
            obs_parts: list[str] = []
            step_had_error = False
            for prefix, payload in actions:
                payload = payload.strip()
                log.info(
                    "agent_loop.action",
                    role=self.role,
                    step=step,
                    action=prefix,
                    payload=payload[:80],
                )
                try:
                    obs = await action_handler(prefix, payload)
                except Exception as exc:
                    obs = f"Error executing {prefix}: {exc}"
                obs_parts.append(f"{prefix}: {payload}\nOBSERVATION: {obs}")
                if self._ERROR_OBS_RE.search(obs):
                    step_had_error = True

            obs_content = "\n\n".join(obs_parts)

            # If DONE was present alongside actions, return now.
            if done_m:
                log.debug("agent_loop.done_after_actions", role=self.role, step=step)
                return obs_parts[-1].split("OBSERVATION:", 1)[-1].strip()

            # ── Self-correction: inject diagnostic nudge on persistent errors ──
            if step_had_error:
                consecutive_errors += 1
                if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    # Too many consecutive failures — ask LLM to try a completely
                    # different strategy rather than repeating the same mistake.
                    obs_content += (
                        "\n\n[SYSTEM] The previous approach has failed multiple times. "
                        "Stop and diagnose: what is the root cause? "
                        "Try a completely different approach or command. "
                        "If the problem is environmental (missing tool, wrong path, "
                        "permissions), say so in DONE: rather than retrying."
                    )
                    consecutive_errors = 0
                    log.warning(
                        "agent_loop.self_correction_nudge",
                        role=self.role,
                        step=step,
                    )
                else:
                    obs_content += (
                        "\n\n[SYSTEM] The last action produced an error. "
                        "Diagnose what went wrong and try a corrected approach."
                    )
            else:
                consecutive_errors = 0

            # Prune old observations if needed to stay within context
            msgs = await self._prune_loop_observations(msgs, obs_content)
            msgs = msgs + [_HumanMessage(content=obs_content)]

        # Exhausted max_steps — force a final synthesis
        log.warning("agent_loop.max_steps_reached", role=self.role, max_steps=max_steps)
        msgs = msgs + [
            _HumanMessage(
                content=(
                    "You have reached the step limit. "
                    "Summarise what you found, what worked, what didn't, and your best answer now. "
                    "Start your response with DONE: "
                )
            )
        ]
        final = await self.llm_invoke(msgs)
        done_m = _DONE_RE.search(final.content)
        return done_m.group("answer").strip() if done_m else final.content.strip()

    async def _prune_loop_observations(self, messages: list, incoming_obs: str) -> list:
        """
        Compress then drop old OBSERVATION HumanMessages (index ≥ 2) when the
        conversation is approaching the model's context limit.

        First pass: replace the oldest full observation with a one-line summary
        (first non-empty line of the OBSERVATION body, capped to 200 chars).
        This preserves the signal that a tool ran and what it returned without
        keeping the full output in context.

        Second pass: if still over budget after all observations are compressed,
        drop them oldest-first.
        """
        limit = await self._get_model_context_limit()
        budget = int(limit * 0.70)
        candidate = messages + [_HumanMessage(content=incoming_obs)]

        def _compress_obs(content: str) -> str:
            """Return a one-line compressed form of an observation message."""
            if "OBSERVATION:" not in content:
                return content
            prefix, _, body = content.partition("OBSERVATION:")
            first_line = next(
                (ln.strip() for ln in body.splitlines() if ln.strip()), body.strip()
            )
            summary = first_line[:200] + ("…" if len(first_line) > 200 else "")
            return f"{prefix}OBSERVATION: [compressed] {summary}"

        # Pass 1: compress oldest full observations.
        while len(messages) > 2 and self._estimate_tokens(candidate) > budget:
            compressed = False
            for i in range(2, len(messages)):
                msg = messages[i]
                content = msg.content or ""
                if (
                    isinstance(msg, _HumanMessage)
                    and "OBSERVATION:" in content
                    and "[compressed]" not in content
                ):
                    new_content = _compress_obs(content)
                    log.debug("agent_loop.obs_compressed", role=self.role, index=i)
                    messages = (
                        messages[:i]
                        + [_HumanMessage(content=new_content)]
                        + messages[i + 1 :]
                    )
                    candidate = messages + [_HumanMessage(content=incoming_obs)]
                    compressed = True
                    break
            if not compressed:
                break

        # Pass 2: drop compressed (or any remaining) observations if still over budget.
        while len(messages) > 2 and self._estimate_tokens(candidate) > budget:
            pruned = False
            for i in range(2, len(messages)):
                msg = messages[i]
                if isinstance(msg, _HumanMessage) and "OBSERVATION:" in (
                    msg.content or ""
                ):
                    log.debug("agent_loop.obs_dropped", role=self.role, index=i)
                    messages = messages[:i] + messages[i + 1 :]
                    candidate = messages + [_HumanMessage(content=incoming_obs)]
                    pruned = True
                    break
            if not pruned:
                break

        return messages

    def _circuit_check(self) -> bool:
        """
        Return True if the circuit is currently open (LM Studio considered down).
        Automatically transitions to HALF_OPEN after CIRCUIT_RESET_SECS.
        """
        if not self._circuit_open:
            return False
        elapsed = time.monotonic() - self._circuit_open_since
        if elapsed >= self.CIRCUIT_RESET_SECS:
            log.info("agent.circuit_half_open", role=self.role, elapsed=int(elapsed))
            self._circuit_open = False  # probe one call; re-opens on failure
            return False
        return True

    async def llm_invoke(self, messages: list) -> Any:
        """
        Serialised, context-checked LLM call with circuit breaker.

        1. Checks the circuit breaker — fails fast if LM Studio is down.
        2. Acquires a Redis distributed lock via pub/sub notification so only
           one agent calls LM Studio at a time (no busy polling).
        3. Checks estimated token count against the model's context limit.
        4. Releases the lock and publishes a wake-up to unblock waiters.

        Falls back to Claude Haiku when ANTHROPIC_API_KEY is set and the
        circuit is open. Raises RuntimeError otherwise.

        Use this instead of self.llm.ainvoke() everywhere.
        """
        # ── Circuit breaker fast-fail ────────────────────────────────────────
        if self._circuit_check():
            if self._claude_fallback is not None:
                log.warning("agent.circuit_open_using_fallback", role=self.role)
                return await self._claude_fallback.ainvoke(messages)
            raise RuntimeError(AgentError.LLM_CIRCUIT_OPEN.value)

        # ── Context limit pre-check ──────────────────────────────────────────
        limit = await self._get_model_context_limit()
        budget = int(limit * 0.75)  # reserve 25% for reply
        estimated = self._estimate_tokens(messages)
        if estimated > budget:
            last = messages[-1]
            # Remaining token budget for the last message → chars
            remaining_tokens = max(50, budget - self._estimate_tokens(messages[:-1]))
            # Approximate chars from tokens (tiktoken averages ~4 chars/token for English)
            max_chars = remaining_tokens * 4
            truncated_content = (
                last.content[:max_chars] + "\n[…input truncated to fit context window]"
            )
            messages = messages[:-1] + [_HumanMessage(content=truncated_content)]
            log.warning(
                "agent.input_truncated",
                role=self.role,
                estimated_tokens=estimated,
                budget=budget,
                limit=limit,
            )

        # ── Distributed lock with priority queue ────────────────────────────
        #
        # Priority: orchestrator (score 0) always cuts the queue ahead of
        # specialist agents (score 1). This keeps user-facing response latency
        # low even when specialists are busy with multi-call pipelines.
        #
        # Mechanism:
        #   llm:queue  — Redis sorted set; members are "role:nonce", score is
        #                priority (0 = high, 1 = normal).  ZADD NX adds our
        #                entry; we hold the lock when we are rank-0 AND the
        #                SET NX lock key is ours.
        #   llm:lock   — Redis string SET NX; actual mutual-exclusion token.
        #   llm:lock:released — pub/sub channel; PUBLISH wakes up waiters.
        #
        lock_priority = 0 if self.role == "orchestrator" else 1
        nonce = uuid.uuid4().hex
        member = f"{self.role}:{nonce}"
        lock_value = member
        queue_key = f"{self._LLM_LOCK_KEY}:queue"
        notify_chan = f"{self._LLM_LOCK_KEY}:released"
        acquired = False

        # Register in the priority queue
        await self.bus._client.zadd(queue_key, {member: lock_priority}, nx=True)
        # Set a TTL on the queue key itself so crashed agents don't litter it
        await self.bus._client.expire(queue_key, self._LLM_LOCK_TTL * 10)

        try:
            while not acquired:
                # We can attempt the lock only when we are at the front of the queue
                front = await self.bus._client.zrange(queue_key, 0, 0)
                if front and front[0] == member:
                    ok = await self.bus._client.set(
                        self._LLM_LOCK_KEY,
                        lock_value,
                        nx=True,
                        ex=self._LLM_LOCK_TTL,
                    )
                    if ok:
                        acquired = True
                        continue

                front_member = front[0] if front else "unknown"
                blocking_role = (
                    front_member.split(":")[0] if front_member else "unknown"
                )
                log.debug(
                    "agent.llm_lock_waiting",
                    role=self.role,
                    priority=lock_priority,
                    blocked_by=blocking_role,
                    reason=f"LLM lock held by {blocking_role}",
                )
                pubsub = self.bus._client.pubsub()
                await pubsub.subscribe(notify_chan)
                try:
                    # get_message() is non-blocking on async redis-py and returns
                    # None immediately, causing a busy-spin. Use listen() with
                    # asyncio.wait_for() to actually block until a message arrives.
                    async def _wait_for_release():
                        async for msg in pubsub.listen():
                            if msg["type"] == "message":
                                return

                    await asyncio.wait_for(
                        _wait_for_release(), timeout=float(self._LLM_LOCK_TTL)
                    )
                except asyncio.TimeoutError:
                    pass  # Lock TTL expired; retry acquisition
                finally:
                    await pubsub.unsubscribe(notify_chan)
                    await pubsub.aclose()
        except Exception:
            # Clean up queue entry if we never acquired the lock
            await self.bus._client.zrem(queue_key, member)
            raise

        try:
            try:
                result = await self.llm.ainvoke(messages)
            except Exception as exc:
                # On context overflow: correct the cached limit and retry once
                # with a truncated last message, so callers never see the error.
                err_str = str(exc)
                _ctx_match = re.search(r"n_ctx:\s*(\d+)", err_str)
                if not _ctx_match:
                    raise
                actual_ctx = int(_ctx_match.group(1))
                log.warning(
                    "agent.context_overflow_retry",
                    role=self.role,
                    was=self._model_context_limit,
                    actual_ctx=actual_ctx,
                )
                self._model_context_limit = actual_ctx
                self._model_context_limit_ts = time.monotonic()
                # Truncate last message to fit inside the corrected limit
                retry_budget = int(actual_ctx * 0.75)
                prior_tokens = self._estimate_tokens(messages[:-1])
                remaining_tokens = max(50, retry_budget - prior_tokens)
                last = messages[-1]
                truncated_content = (
                    last.content[: remaining_tokens * 4]
                    + "\n[…input truncated to fit context window]"
                )
                messages = messages[:-1] + [_HumanMessage(content=truncated_content)]
                result = await self.llm.ainvoke(messages)

            # Successful call — reset circuit breaker
            self._circuit_failures = 0
            self._circuit_open = False
            return result
        except Exception as exc:
            # Track failures for circuit breaker
            self._circuit_failures += 1
            if self._circuit_failures >= self.CIRCUIT_THRESHOLD:
                self._circuit_open = True
                self._circuit_open_since = time.monotonic()
                log.error(
                    "agent.circuit_opened",
                    role=self.role,
                    failures=self._circuit_failures,
                    error=str(exc),
                )
            raise
        finally:
            # Remove from queue and release lock, then wake up waiters
            await self.bus._client.zrem(queue_key, member)
            current = await self.bus._client.get(self._LLM_LOCK_KEY)
            if current == lock_value:
                await self.bus._client.delete(self._LLM_LOCK_KEY)
                await self.bus._client.publish(notify_chan, "released")

    # ── Memory helpers ───────────────────────────────────────────────────────

    def stage_finding(
        self, content: str, topic: str, tags: Optional[list[str]] = None
    ) -> None:
        """
        Stage a finding for promotion to Emrys on shutdown.
        Use this instead of immediate store() for batching efficiency.
        """
        self._findings.append(
            {
                "content": content,
                "topic": topic,
                "tags": tags or [self.role],
                "kind": "finding",
            }
        )

    async def promote_now(
        self, content: str, topic: str, tags: Optional[list[str]] = None
    ) -> None:
        """Immediately promote a critical finding to long-term memory."""
        await self.memory.store(content, topic, tags or [self.role])
        await self.bus.publish(
            Event(
                type=EventType.MEMORY_PROMOTED,
                source=self.role,
                payload={"topic": topic, "preview": content[:120]},
            ),
            target="broadcast",
        )

    async def recall(
        self, query: str, semantic: bool = True, limit: int = 5
    ) -> list[dict]:
        """Query long-term memory before starting a task. Results are auto-truncated."""
        results = await self.memory.search(query, semantic=semantic, limit=limit)
        return truncate_memory_entries(results)

    async def search_tools(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search the shared tool registry by semantic similarity.
        Call this before making an LLM call — the results can either drive
        direct execution (for shell: tools) or inform the LLM as context.
        """
        return await self.memory.search_tools(query, limit)

    async def recall_and_search_tools(
        self, query: str, tools_limit: int = 5, memory_limit: int = 5
    ) -> tuple[list[dict], list[dict]]:
        """
        Embed the query once and run memory + tool searches in parallel.
        Returns (memory_results, tool_results). Prefer this over calling
        recall() and search_tools() separately to avoid two embedding calls.
        """
        memory_results, tool_results = await self.memory.search_memory_and_tools(
            query, tools_limit=tools_limit, memory_limit=memory_limit
        )
        return truncate_memory_entries(memory_results), tool_results

    def format_tools_context(self, tools: list[dict]) -> str:
        """
        Format tool search results as a compact prompt section.
        Prepend to LLM SystemMessages so the model knows what tools exist.
        """
        if not tools:
            return ""
        lines = ["\n## Relevant tools from shared registry"]
        for t in tools:
            sim = t.get("similarity", 0)
            if sim and sim < 0.3:
                continue  # skip low-relevance hits
            lines.append(
                f"- **{t['name']}** (owner: {t['owner_agent']}): {t['description']}"
            )
            lines.append(f"  invoke: `{t['invocation']}`")
        if len(lines) == 1:
            return ""  # all hits filtered out
        return "\n".join(lines) + "\n"

    def format_memory_context(
        self, memories: list[dict], label: str = "Prior knowledge"
    ) -> str:
        """
        Format long-term memory search results as a compact prompt section.
        Inject before LLM calls so the agent is primed with what it already knows.

        Filters out low-similarity hits (<0.35) and caps each entry at 300 chars
        to stay within token budget.
        """
        if not memories:
            return ""
        lines = [f"\n## {label}"]
        for m in memories:
            sim = m.get("similarity", 1.0)
            if sim and sim < 0.35:
                continue
            content = (m.get("content") or "").strip()
            if not content:
                continue
            topic = m.get("topic", "")
            snippet = content[:300] + ("…" if len(content) > 300 else "")
            line = f"- [{topic}] {snippet}" if topic else f"- {snippet}"
            lines.append(line)
        if len(lines) == 1:
            return ""  # all hits filtered
        return "\n".join(lines) + "\n"

    async def build_task_context(
        self, task: str, tools_limit: int = 5, memory_limit: int = 5
    ) -> str:
        """
        Build a combined context string (memory + tools) for a given task.
        Single embedding call shared across both lookups.
        Call this before constructing LLM messages to prime the model with
        relevant prior knowledge and available tools.
        """
        memories, tools = await self.recall_and_search_tools(
            task, tools_limit=tools_limit, memory_limit=memory_limit
        )
        return self.format_memory_context(memories) + self.format_tools_context(tools)

    # Redis key used to rate-limit memory warning notifications (per agent).
    _MEMORY_WARN_KEY_PREFIX = "memory:warn_sent"
    _MEMORY_WARN_INTERVAL = 86_400  # 24 hours — don't spam; one warning per day

    async def _apply_log_level_override(self) -> None:
        """Check Redis for a dynamic log level override and apply it if set."""
        try:
            level_str = await self.bus._client.get("config:log_level")
            if level_str:
                level = getattr(logging, level_str.upper(), None)
                if level is not None:
                    root = logging.getLogger()
                    if root.level != level:
                        root.setLevel(level)
                        log.info(
                            "agent.log_level_applied",
                            role=self.role,
                            level=level_str.upper(),
                        )
        except Exception:
            pass

    async def _check_memory_health(self) -> None:
        """
        Check memory count. Emit MEMORY_PRUNED warning if over threshold,
        and prune if over hard limit.  Warnings are throttled to once per 24h.
        """
        try:
            count = await self.memory.count()
            if count >= MEMORY_HARD_LIMIT:
                deleted = await self.memory.prune(MEMORY_PRUNE_TARGET)
                await self.bus.publish(
                    Event(
                        type=EventType.MEMORY_PRUNED,
                        source=self.role,
                        payload={
                            "deleted": deleted,
                            "remaining": count - deleted,
                            "reason": "hard_limit",
                            "message": (
                                f"⚠️ Long-term memory pruned: {deleted} oldest entries removed "
                                f"({count - deleted} remaining). Hard limit of {MEMORY_HARD_LIMIT} was reached."
                            ),
                        },
                    ),
                    target="broadcast",
                )
            elif count >= MEMORY_WARN_THRESHOLD:
                # Rate-limit: only warn once per 24h to avoid constant noise
                warn_key = f"{self._MEMORY_WARN_KEY_PREFIX}:{self.role}"
                already_warned = await self.bus._client.get(warn_key)
                if not already_warned:
                    await self.bus._client.setex(
                        warn_key, self._MEMORY_WARN_INTERVAL, "1"
                    )
                    await self.bus.publish(
                        Event(
                            type=EventType.MEMORY_PRUNED,
                            source=self.role,
                            payload={
                                "deleted": 0,
                                "remaining": count,
                                "reason": "warn_threshold",
                                "message": (
                                    f"⚠️ Long-term memory at {count} entries "
                                    f"(warn threshold: {MEMORY_WARN_THRESHOLD}). "
                                    f"Will auto-prune at {MEMORY_HARD_LIMIT}."
                                ),
                            },
                        ),
                        target="broadcast",
                    )
        except Exception as exc:
            import psycopg  # noqa: PLC0415

            if isinstance(exc, psycopg.OperationalError):
                log.error(
                    "agent.memory_db_unreachable",
                    role=self.role,
                    error=str(exc),
                )
                await self.emit(
                    EventType.ERROR,
                    payload={
                        "error": f"PostgreSQL unreachable during memory health check: {exc}"
                    },
                )
            else:
                log.warning("agent.memory_health_check_failed", error=str(exc))

    # ── Emit helpers ─────────────────────────────────────────────────────────

    async def emit(
        self, event_type: EventType, payload: dict, target: str = "broadcast"
    ) -> None:
        await self.bus.publish(
            Event(type=event_type, source=self.role, payload=payload),
            target=target,
        )

    async def _emit_error(self, triggering_event: Event, exc: Exception) -> None:
        await self.emit(
            EventType.ERROR,
            payload={
                "error": str(exc),
                "triggering_event_id": triggering_event.event_id,
                "triggering_event_type": triggering_event.type,
            },
        )

    # ── Self-modification context ────────────────────────────────────────────

    def self_modify_context(self) -> str:
        """
        Compact prompt section for self-modification. Appended to LLM SystemMessages.
        Kept short to preserve context window budget.
        """
        role = self.role
        src = f"/workspace/src/agents/{role}/main.py"
        return (
            f"\n## Self-modification\n"
            f"Your source: {src} | Container: agent_{role}\n"
            f"1. Read: ask executor 'cat {src}'\n"
            f"2. Write (approval required): ask executor to tee the new content to {src}\n"
            f"3. Restart (approval required): ask executor 'docker restart agent_{role}'\n"
            f"\n## Workspace layout\n"
            f"/workspace/src/          — full agent stack source (agents/, core/, docker-compose.yml)\n"
            f"/workspace/emrys/        — Emrys library source (README.md, src/, pyproject.toml)\n"
            f"/workspace/tools/        — reusable scripts saved by executor\n"
            f"/workspace/docs/         — documents for document_qa\n"
            f"/workspace/repos/        — repos for code_search\n"
            f"/workspace/projects/     — user coding projects (read-only)\n"
            f"\n## Memory API (self.memory)\n"
            f"store(content, topic, tags)        — persist a finding to Postgres\n"
            f"search(query, semantic=True)       — recall relevant knowledge\n"
            f"enqueue_task(task, priority=5)     — queue async work for think loop\n"
            f"register_tool(name, desc, owner, invocation, tags, created_by)  — add tool to shared registry\n"
            f"search_tools(query, limit=5)       — search shared tool registry before LLM\n"
            f"Full Emrys docs: cat /workspace/emrys/README.md\n"
            f"\n## Shared tool registry\n"
            f"All agents share a PostgreSQL tools table. Always call search_tools(task) before llm_invoke().\n"
            f"Executor 'owns' shell: tools and can run them directly. Other agents reference them as context.\n"
            f"After solving a new problem, register the solution: await self.memory.register_tool(...)\n"
        )

    def task_queue_context(self) -> str:
        """
        Compact prompt section describing the persistent task queue.
        Appended to LLM SystemMessages so every agent knows how to schedule work.
        """
        return (
            "\n## Task queue\n"
            "You have a persistent task queue backed by the open_tasks Postgres table.\n"
            "Use it to schedule work that should happen asynchronously or in a future think cycle.\n"
            "Priority: 1=urgent, 5=normal, 10=low. Lower number runs first.\n"
            "To enqueue a task from code: await self.memory.enqueue_task('task text', priority=5)\n"
            "The orchestrator think loop drains this queue automatically — one task per cycle.\n"
            "Use this instead of doing long-running work inline when:\n"
            "  - The task doesn't need an immediate reply\n"
            "  - The task is a follow-up to the current one\n"
            "  - You want to schedule recurring or deferred work\n"
        )

    # ── Voting ───────────────────────────────────────────────────────────────

    def _handle_clarification_response(self, event: "Event") -> None:
        """Unblock any coroutine waiting on a vote clarification for this plan."""
        plan_id = event.payload.get("plan_id", "")
        answer = event.payload.get("answer", "")
        ev = self._clarification_events.get(plan_id)
        if ev is not None:
            self._clarification_answers[plan_id] = answer
            ev.set()

    async def _handle_plan_proposed(self, event: "Event") -> None:
        """
        Called when a PLAN_PROPOSED broadcast is received.
        Delegates to on_plan_proposed() so subclasses can inspect the plan
        and cast a vote.  Passes a request_clarification() coroutine the
        subclass can await to ask the orchestrator for more context before voting.
        """
        plan_id = event.payload.get("plan_id", "")
        steps = event.payload.get("steps", [])
        original_task = event.payload.get("original_task", "")
        steps_summary = "\n".join(
            f"  Phase {s.get('phase', 1)}: [{s.get('agent', '?')}] {s.get('task', '')}"
            for s in steps
        )

        async def request_clarification(question: str, timeout: float = 8.0) -> str:
            """
            Ask the orchestrator to clarify something about this plan.
            Returns the orchestrator's answer, or an empty string on timeout.
            """
            ev = asyncio.Event()
            self._clarification_events[plan_id] = ev
            self._clarification_answers.pop(plan_id, None)
            await self.emit(
                EventType.VOTE_CLARIFICATION_REQUEST,
                payload={
                    "plan_id": plan_id,
                    "question": question[:300],
                    "original_task": original_task,
                    "steps_summary": steps_summary,
                },
                target="orchestrator",
            )
            try:
                await asyncio.wait_for(ev.wait(), timeout=timeout)
                return self._clarification_answers.get(plan_id, "")
            except asyncio.TimeoutError:
                log.warning(
                    "agent.vote_clarification_timeout",
                    role=self.role,
                    plan_id=plan_id[:8],
                )
                return ""
            finally:
                self._clarification_events.pop(plan_id, None)
                self._clarification_answers.pop(plan_id, None)

        try:
            approve, reason, confidence = await self.on_plan_proposed(
                plan_id, steps, event.payload, request_clarification
            )
        except Exception as exc:
            log.warning(
                "agent.plan_proposed_handler_error", role=self.role, error=str(exc)
            )
            return
        if approve is None:
            return  # explicit opt-out — don't vote
        await self.emit_vote(
            plan_id, approve=approve, reason=reason, confidence=confidence
        )

    async def on_plan_proposed(
        self,
        plan_id: str,
        steps: list[dict],
        payload: dict,
        request_clarification=None,
    ) -> tuple[bool | None, str, float]:
        """
        Override in subclasses to inspect a proposed plan and return a vote.

        Return a 3-tuple: (approve, reason, confidence)
          approve:    True to support, False to object, None to abstain (no vote emitted)
          reason:     brief human-readable explanation
          confidence: 0.0–1.0; only rejections with confidence ≥ 0.7 trigger plan revision

        request_clarification(question) — await this coroutine to ask the orchestrator
        to explain something about the plan before you vote. It returns the answer as a
        string, or "" on timeout.

        The default implementation abstains so agents that don't override this
        method don't pollute the vote tally with unconsidered approvals.
        """
        return None, "", 0.0

    async def emit_vote(
        self,
        plan_id: str,
        approve: bool,
        reason: str = "",
        confidence: float = 1.0,
    ) -> None:
        """
        Cast a vote on a proposed plan.  Publish to broadcast so the orchestrator
        and Discord bridge both receive it.

        approve:    True = support the plan, False = object
        reason:     brief human-readable justification
        confidence: 0.0–1.0; high-confidence rejections carry more weight
        """
        await self.emit(
            EventType.AGENT_VOTE,
            payload={
                "plan_id": plan_id,
                "agent": self.role,
                "approve": approve,
                "reason": reason[:300],
                "confidence": round(confidence, 3),
            },
            target="broadcast",
        )
        log.info(
            "agent.vote_cast",
            role=self.role,
            plan_id=plan_id[:8],
            approve=approve,
            confidence=confidence,
        )

    async def request_vote_extension(
        self, plan_id: str, extra_ms: int, reason: str = ""
    ) -> None:
        """
        Ask the orchestrator for more deliberation time before votes are tallied.
        The orchestrator will grant up to vote_max_extension_ms (from config).
        """
        await self.emit(
            EventType.VOTE_EXTENSION_REQUESTED,
            payload={
                "plan_id": plan_id,
                "agent": self.role,
                "requested_ms": extra_ms,
                "reason": reason[:200],
            },
            target="broadcast",
        )

    async def initiate_peer_vote(
        self,
        question: str,
        context: str = "",
        vote_id: str | None = None,
    ) -> str:
        """
        Request a general peer vote from all agents on any question.
        The orchestrator will start all ephemeral agents, broadcast the vote,
        collect responses, tally results, and shut down vote-only agents.

        Returns the vote_id so the caller can correlate the VOTE_RESULT event.
        """
        import uuid as _uuid

        vid = vote_id or _uuid.uuid4().hex
        await self.emit(
            EventType.PEER_VOTE_REQUESTED,
            payload={
                "vote_id": vid,
                "question": question[:400],
                "context": context[:800],
                "initiator": self.role,
            },
            target="orchestrator",
        )
        log.info(
            "agent.peer_vote_requested",
            role=self.role,
            vote_id=vid[:8],
            question=question[:80],
        )
        return vid

    async def _handle_vote_initiated(self, event: "Event") -> None:
        """
        Called when a VOTE_INITIATED broadcast is received (peer vote in progress).
        Delegates to on_vote_initiated() so subclasses can reason and cast a vote.
        Uses the same clarification mechanism as plan votes.
        """
        vote_id = event.payload.get("vote_id", "")
        question = event.payload.get("question", "")
        context = event.payload.get("context", "")
        initiator = event.payload.get("initiator", "orchestrator")

        async def request_clarification(q: str, timeout: float = 8.0) -> str:
            ev = asyncio.Event()
            self._clarification_events[vote_id] = ev
            self._clarification_answers.pop(vote_id, None)
            await self.emit(
                EventType.VOTE_CLARIFICATION_REQUEST,
                payload={
                    "plan_id": vote_id,  # reuse plan_id field for routing
                    "question": q[:300],
                    "original_task": question,
                    "steps_summary": context,
                },
                target="orchestrator",
            )
            try:
                await asyncio.wait_for(ev.wait(), timeout=timeout)
                return self._clarification_answers.get(vote_id, "")
            except asyncio.TimeoutError:
                log.warning(
                    "agent.vote_clarification_timeout",
                    role=self.role,
                    vote_id=vote_id[:8],
                )
                return ""
            finally:
                self._clarification_events.pop(vote_id, None)
                self._clarification_answers.pop(vote_id, None)

        try:
            approve, reason, confidence = await self.on_vote_initiated(
                vote_id, question, context, initiator, request_clarification
            )
        except Exception as exc:
            log.warning(
                "agent.vote_initiated_handler_error", role=self.role, error=str(exc)
            )
            return
        if approve is None:
            return
        await self.emit_vote(
            vote_id, approve=approve, reason=reason, confidence=confidence
        )

    async def on_vote_initiated(
        self,
        vote_id: str,
        question: str,
        context: str,
        initiator: str,
        request_clarification=None,
    ) -> tuple[bool | None, str, float]:
        """
        Called for every peer vote broadcast.  The base implementation uses
        the LLM to reason about the question from this agent's perspective,
        asks for clarification if needed, then votes.

        Subclasses can override for specialised behaviour or to abstain.
        """
        from langchain_core.messages import HumanMessage as _HM, SystemMessage as _SM

        try:
            response = await self.llm_invoke(
                [
                    _SM(
                        content=(
                            f"You are the {self.role} agent participating in a peer vote. "
                            "Read the question and any context, then decide whether to vote yay or nay "
                            "based on your role's domain knowledge and concerns.\n\n"
                            "Reply in exactly this format:\n"
                            "UNDERSTOOD: yes/no\n"
                            "CLARIFICATION_NEEDED: <one focused question if UNDERSTOOD=no, else 'none'>\n"
                            "VOTE: yay/nay\n"
                            "REASON: <one sentence>"
                        )
                    ),
                    _HM(
                        content=(
                            f"Question: {question}\n"
                            + (f"Context: {context}" if context else "")
                        )
                    ),
                ]
            )
            lines = {
                k.strip(): v.strip()
                for line in response.content.splitlines()
                if ":" in line
                for k, v in [line.split(":", 1)]
            }
        except Exception as exc:
            log.warning("agent.peer_vote_llm_error", role=self.role, error=str(exc))
            return None, "", 0.0  # abstain on LLM failure

        understood = lines.get("UNDERSTOOD", "yes").lower() == "yes"
        clarification_q = lines.get("CLARIFICATION_NEEDED", "none")
        vote_str = lines.get("VOTE", "yay").lower()
        reason = lines.get("REASON", "")

        if (
            not understood
            and clarification_q
            and clarification_q.lower() != "none"
            and request_clarification
        ):
            answer = await request_clarification(clarification_q)
            if answer:
                try:
                    r2 = await self.llm_invoke(
                        [
                            _SM(
                                content=(
                                    f"You are the {self.role} agent re-evaluating a peer vote after clarification. "
                                    "Reply in exactly this format:\n"
                                    "VOTE: yay/nay\n"
                                    "REASON: <one sentence>"
                                )
                            ),
                            _HM(
                                content=(
                                    f"Question: {question}\n"
                                    + (f"Context: {context}\n" if context else "")
                                    + f"Clarification: {answer}"
                                )
                            ),
                        ]
                    )
                    lines2 = {
                        k.strip(): v.strip()
                        for line in r2.content.splitlines()
                        if ":" in line
                        for k, v in [line.split(":", 1)]
                    }
                    vote_str = lines2.get("VOTE", vote_str).lower()
                    reason = lines2.get("REASON", reason)
                except Exception:
                    pass

        approve = vote_str == "yay"
        return approve, reason[:200], 0.8

    # ── Topic classification ──────────────────────────────────────────────────

    async def classify_topic(self, text: str) -> tuple[str | None, float]:
        """
        Classify *text* into a topic category using patterns stored in long-term memory.

        Returns (category, confidence).  Returns (None, 0.0) when no pattern
        matches well enough or when not enough sessions have been seen yet.

        When confidence < topic_confidence_ask threshold AND enough sessions have
        accumulated, the caller should ask the user for the correct label.
        """
        import re as _re

        keywords = [
            w for w in _re.sub(r"[^a-z0-9\s]", "", text.lower()).split() if len(w) > 2
        ]
        if not keywords:
            return None, 0.0

        try:
            matches = await self.memory.search_topic_patterns(keywords, limit=3)
        except Exception:
            return None, 0.0

        if not matches:
            return None, 0.0

        best = matches[0]
        confidence = best.get("confidence", 0.0)
        category = best.get("category")

        # Reward the pattern for this match
        try:
            await self.memory.save_topic_pattern(
                category, keywords, confidence, created_by=self.role
            )
        except Exception:
            pass

        return category, confidence

    async def should_ask_user_for_topic(self, confidence: float) -> bool:
        """
        Returns True when the system should prompt the user for a topic label.
        Only kicks in once enough sessions have accumulated so early-learning
        noise doesn't bother the user constantly.
        """
        try:
            threshold = float(await self.bus.get_config("topic_confidence_ask", 0.8))
            min_sessions = int(await self.bus.get_config("topic_min_sessions", 50))
            if confidence >= threshold:
                return False
            session_count = await self.memory.get_closed_session_count()
            return session_count >= min_sessions
        except Exception:
            return False

    # ── Context history resolution ────────────────────────────────────────────

    async def resolve_from_context_history(
        self,
        question: str,
        context_id: str,
        lookback: int = 20,
    ) -> str | None:
        """
        Before asking the user a clarifying question, check whether the answer
        can be found in the recent history of a context stream.

        Returns a relevant excerpt if found, or None if the history is empty
        or no content overlaps with the question keywords.
        """
        import re as _re

        entries = await self.bus.read_context_stream(context_id, count=lookback)
        if not entries:
            return None

        q_words = set(_re.sub(r"[^a-z0-9\s]", "", question.lower()).split())
        best_score = 0
        best_text = None

        for _entry_id, event in reversed(entries):
            payload_text = " ".join(str(v) for v in event.payload.values())
            words = set(_re.sub(r"[^a-z0-9\s]", "", payload_text.lower()).split())
            union = q_words | words
            if not union:
                continue
            score = len(q_words & words) / len(union)
            if score > best_score:
                best_score = score
                best_text = payload_text[:500]

        # Only surface if there is meaningful overlap
        if best_score >= 0.25 and best_text:
            return best_text
        return None

    # ── Subclass interface ───────────────────────────────────────────────────

    @abstractmethod
    async def handle_event(self, event: Event) -> None:
        """Process a single event. Called for every event on the agent's stream."""
        ...

    async def on_startup(self) -> None:
        """Optional: run after connections established, before event loop."""
        pass

    async def on_shutdown(self) -> None:
        """Optional: run before session close and memory promotion."""
        pass

    async def think(self) -> None:
        """
        Proactive reasoning cycle called every think_interval seconds.
        Override in subclasses to add autonomous background behaviour.
        """
        pass


def run_agent(agent: BaseAgent) -> None:
    """Entry point for running an agent with graceful shutdown."""
    loop = asyncio.new_event_loop()

    async def _main():
        def _handle_signal():
            asyncio.create_task(agent.stop())

        loop.add_signal_handler(signal.SIGTERM, _handle_signal)
        loop.add_signal_handler(signal.SIGINT, _handle_signal)
        await agent.start()
        # Runs cleanup whether we exited via signal or idle timeout
        await agent.stop()

    loop.run_until_complete(_main())

    # Drain any tasks that were created but not awaited before the loop closed.
    # This prevents "Task was destroyed but it is pending!" warnings from
    # fire-and-forget tasks (psycopg pool workers, Redis ACKs, etc.) that
    # outlived the main coroutine.
    pending = asyncio.all_tasks(loop)
    if pending:
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    loop.close()
