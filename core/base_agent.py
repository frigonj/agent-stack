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
import os
import signal
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx
import structlog
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from core.events.bus import Event, EventBus, EventType
from core.errors import AgentError, error_payload
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
    _tok_enc = tiktoken.get_encoding("cl100k_base")  # good approximation for Qwen2.5

    def _count_tokens(text: str) -> int:
        return len(_tok_enc.encode(text, disallowed_special=()))
except ImportError:
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return len(text) // 4

log = structlog.get_logger()


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
            api_key="lm-studio",           # LM Studio doesn't require a real key
            model=settings.lm_studio_model,
            temperature=0.1,
            streaming=True,
            extra_body={"thinking": False},
        )

        self._running = False
        self._stopped = False
        self._findings: list[dict] = []    # Staged for promotion on shutdown
        self._stop_event: Optional[asyncio.Event] = None
        # context_id → asyncio.Task for multi-context consumer pool
        self._context_tasks: dict[str, asyncio.Task] = {}
        self._model_context_limit: Optional[int] = None   # cached from LM Studio
        self._last_event_time: float = time.monotonic()
        # Counts concurrently running event-handler tasks.
        # Status is only set to "idle" when this reaches zero.
        self._active_tasks: int = 0

        # IDLE_TIMEOUT env var overrides the class-level default
        env_idle = os.environ.get("IDLE_TIMEOUT", "")
        self._idle_timeout = int(env_idle) if env_idle.isdigit() else self.idle_timeout

        # ── Circuit breaker state ─────────────────────────────────────────────
        # Tracks consecutive LM Studio failures. After CIRCUIT_THRESHOLD failures
        # the circuit opens and llm_invoke() fails fast (or uses Claude fallback).
        self._circuit_failures:   int   = 0
        self._circuit_open:       bool  = False
        self._circuit_open_since: float = 0.0

        # Optional Claude API fallback (only active when ANTHROPIC_API_KEY is set)
        _api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if _api_key:
            try:
                from langchain_anthropic import ChatAnthropic
                self._claude_fallback: Optional[Any] = ChatAnthropic(
                    model="claude-haiku-4-5-20251001",   # cheapest/fastest for fallback
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

        # Announce presence
        await self.bus.publish(
            Event(type=EventType.AGENT_STARTED, source=self.role, payload={"role": self.role}),
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
            except asyncio.CancelledError:
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
            asyncio.create_task(self._handle_context_and_ack(stream_key, entry_id, event, context_id))

    async def _handle_context_and_ack(
        self, stream: str, entry_id: str, event: "Event", context_id: str
    ) -> None:
        try:
            group = f"{self.role}_ctx_group"
            await self.handle_context_event(event, context_id)
            await self.bus.ack(stream, group, entry_id)
        except Exception as exc:
            log.error(
                "agent.context_event_error",
                role=self.role,
                context_id=context_id,
                error=str(exc),
            )

    async def handle_context_event(self, event: "Event", context_id: str) -> None:
        """
        Override in subclasses to react to events on a subscribed context stream.
        Default: forward to handle_event() so agents without specific context
        handling still process the event normally.
        """
        await self.handle_event(event)

    # ── Event loop ───────────────────────────────────────────────────────────

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
                pass   # normal — interval elapsed

            if not self._running:
                break

            # Only run memory health check + think() when the queue is idle
            if not await self._queue_is_idle():
                log.debug("agent.think_skipped_busy", role=self.role)
                continue

            try:
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
            self._last_event_time = time.monotonic()
            # Fire-and-forget each event so the consumer loop stays unblocked.
            # Handlers that take a long time (LLM calls, docker spawning) won't
            # delay acknowledgement of subsequent events.
            asyncio.create_task(self._handle_and_ack(stream, entry_id, event))

    async def _handle_and_ack(self, stream: str, entry_id: str, event: Event) -> None:
        self._active_tasks += 1
        await self._set_status("busy", task=event.payload.get("task", event.type.value)[:80])
        try:
            await self.handle_event(event)
            await self.bus.ack(stream, self.group_name, entry_id)
        except Exception as exc:
            log.error("agent.event_error", role=self.role, error=str(exc), event_id=event.event_id)
            await self._emit_error(event, exc)
        finally:
            self._active_tasks -= 1
            if self._active_tasks <= 0:
                self._active_tasks = 0
                await self._set_status("idle")

    _STATUS_KEY_PREFIX = "agent:status"
    _STATUS_TTL = 3600  # 1 hour — auto-expires so stale entries don't linger

    async def _set_status(self, status: str, task: str = "") -> None:
        """Publish agent status to Redis for the control script to query."""
        try:
            payload = json.dumps({
                "status":  status,
                "task":    task,
                "since":   time.time(),
                "queue":   await self._queue_depth(),
            })
            await self.bus._client.setex(
                f"{self._STATUS_KEY_PREFIX}:{self.role}", self._STATUS_TTL, payload
            )
        except Exception:
            pass  # status publishing is best-effort; never block the event loop

    async def _queue_depth(self) -> int:
        """Return the number of pending (unprocessed) messages in this agent's stream."""
        try:
            key = f"agents:{self.role}"
            groups = await self.bus._client.xinfo_groups(key)
            for g in groups:
                lag = g.get("lag") or g.get("pel-count") or 0
                return int(lag)
        except Exception:
            pass
        return 0

    # ── LLM call management ─────────────────────────────────────────────────

    # Redis key used as a distributed mutex across all agent containers.
    # Only one agent may hold this lock at a time, preventing KV cache exhaustion.
    _LLM_LOCK_KEY  = "llm:lock"
    _LLM_LOCK_TTL  = 120          # seconds — auto-releases if agent crashes mid-call
    _LLM_LOCK_POLL = 1.5          # seconds between acquire attempts

    async def _get_model_context_limit(self) -> int:
        """
        Fetch the loaded model's max context length from LM Studio's REST API.
        Cached after the first successful call. Returns a conservative default on failure.
        """
        if self._model_context_limit is not None:
            return self._model_context_limit
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.settings.lm_studio_url}/api/v0/models")
                r.raise_for_status()
                data = r.json()
                models = data if isinstance(data, list) else data.get("data", [])
                for m in models:
                    ctx = m.get("max_context_length") or m.get("context_length")
                    if ctx:
                        self._model_context_limit = int(ctx)
                        log.info(
                            "agent.model_context_limit",
                            role=self.role,
                            limit=self._model_context_limit,
                        )
                        return self._model_context_limit
        except Exception as exc:
            log.warning("agent.context_limit_fetch_failed", error=str(exc))
        # Conservative fallback: 4096 tokens → ~16 000 chars
        self._model_context_limit = 4096
        return self._model_context_limit

    # ── Circuit breaker constants ────────────────────────────────────────────
    CIRCUIT_THRESHOLD   = 3     # consecutive failures before opening
    CIRCUIT_RESET_SECS  = 60    # seconds before attempting half-open probe

    def _estimate_tokens(self, messages: list) -> int:
        """Token estimate using tiktoken when available, chars//4 as fallback."""
        total = 0
        for m in messages:
            content = getattr(m, "content", "") or ""
            total += _count_tokens(content)
        return total

    async def _budget_content_chars(self, system_msg: str, fixed_content: str = "") -> int:
        """
        Return the character budget available for variable document/code content.

        Subtracts the system message and any fixed content (task description,
        static instructions) from the model's input budget, leaving room for
        the reply (25% of context window).

        Use this before loading large content (files, source trees, code snippets)
        to avoid sending more tokens than the model can accept.

        Returns at least 1 000 chars so there's always something useful to send.
        """
        ctx_limit   = await self._get_model_context_limit()
        input_cap   = int(ctx_limit * 0.70)          # 25% reply + 5% overhead
        sys_tokens  = self._estimate_tokens([SystemMessage(content=system_msg)])
        fixed_toks  = _count_tokens(fixed_content) if fixed_content else 0
        available   = max(1_000, (input_cap - sys_tokens - fixed_toks) * 4)
        log.debug(
            "agent.content_budget",
            role=self.role,
            ctx_limit=ctx_limit,
            sys_tokens=sys_tokens,
            fixed_tokens=fixed_toks,
            available_chars=available,
        )
        return available

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
        limit    = await self._get_model_context_limit()
        budget   = int(limit * 0.75)   # reserve 25% for reply
        estimated = self._estimate_tokens(messages)
        if estimated > budget:
            from langchain_core.messages import HumanMessage
            last      = messages[-1]
            # Remaining token budget for the last message → chars
            remaining_tokens = max(50, budget - self._estimate_tokens(messages[:-1]))
            # Approximate chars from tokens (tiktoken averages ~4 chars/token for English)
            max_chars = remaining_tokens * 4
            truncated_content = last.content[:max_chars] + "\n[…input truncated to fit context window]"
            messages  = messages[:-1] + [HumanMessage(content=truncated_content)]
            log.warning(
                "agent.input_truncated",
                role=self.role,
                estimated_tokens=estimated,
                budget=budget,
                limit=limit,
            )

        # ── Distributed lock via Redis SET NX + pub/sub wake-up ─────────────
        lock_value  = f"{self.role}:{id(messages)}"
        notify_chan  = f"{self._LLM_LOCK_KEY}:released"
        acquired    = False

        while not acquired:
            ok = await self.bus._client.set(
                self._LLM_LOCK_KEY, lock_value,
                nx=True, ex=self._LLM_LOCK_TTL,
            )
            if ok:
                acquired = True
            else:
                log.debug("agent.llm_lock_waiting", role=self.role)
                # Subscribe and block until the current holder releases the lock.
                # This replaces the 1.5 s busy-poll loop — agents wake up instantly.
                pubsub = self.bus._client.pubsub()
                await pubsub.subscribe(notify_chan)
                try:
                    # Wait up to _LLM_LOCK_TTL seconds so we never stall longer
                    # than the lock's own TTL even if the PUBLISH is missed.
                    await pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=float(self._LLM_LOCK_TTL),
                    )
                finally:
                    await pubsub.unsubscribe(notify_chan)
                    await pubsub.aclose()

        try:
            result = await self.llm.ainvoke(messages)
            # Successful call — reset circuit breaker
            self._circuit_failures = 0
            self._circuit_open     = False
            return result
        except Exception as exc:
            # Track failures for circuit breaker
            self._circuit_failures += 1
            if self._circuit_failures >= self.CIRCUIT_THRESHOLD:
                self._circuit_open       = True
                self._circuit_open_since = time.monotonic()
                log.error(
                    "agent.circuit_opened",
                    role=self.role,
                    failures=self._circuit_failures,
                    error=str(exc),
                )
            raise
        finally:
            # Release lock only if we still own it (guard against TTL expiry)
            current = await self.bus._client.get(self._LLM_LOCK_KEY)
            if current == lock_value:
                await self.bus._client.delete(self._LLM_LOCK_KEY)
                # Wake up all waiters immediately
                await self.bus._client.publish(notify_chan, "released")

    # ── Memory helpers ───────────────────────────────────────────────────────

    def stage_finding(self, content: str, topic: str, tags: Optional[list[str]] = None) -> None:
        """
        Stage a finding for promotion to Emrys on shutdown.
        Use this instead of immediate store() for batching efficiency.
        """
        self._findings.append({
            "content": content,
            "topic": topic,
            "tags": tags or [self.role],
            "kind": "finding",
        })

    async def promote_now(self, content: str, topic: str, tags: Optional[list[str]] = None) -> None:
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

    async def recall(self, query: str, semantic: bool = True) -> list[dict]:
        """Query long-term memory before starting a task. Results are auto-truncated."""
        results = await self.memory.search(query, semantic=semantic)
        return truncate_memory_entries(results)

    async def search_tools(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search the shared tool registry by semantic similarity.
        Call this before making an LLM call — the results can either drive
        direct execution (for shell: tools) or inform the LLM as context.
        """
        return await self.memory.search_tools(query, limit)

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
            lines.append(f"- **{t['name']}** (owner: {t['owner_agent']}): {t['description']}")
            lines.append(f"  invoke: `{t['invocation']}`")
        if len(lines) == 1:
            return ""  # all hits filtered out
        return "\n".join(lines) + "\n"

    async def _check_memory_health(self) -> None:
        """
        Check memory count. Emit MEMORY_PRUNED warning if over threshold,
        and prune if over hard limit.
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
            log.warning("agent.memory_health_check_failed", error=str(exc))

    # ── Emit helpers ─────────────────────────────────────────────────────────

    async def emit(self, event_type: EventType, payload: dict, target: str = "broadcast") -> None:
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
                "plan_id":    plan_id,
                "agent":      self.role,
                "approve":    approve,
                "reason":     reason[:300],
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
                "plan_id":      plan_id,
                "agent":        self.role,
                "requested_ms": extra_ms,
                "reason":       reason[:200],
            },
            target="broadcast",
        )

    # ── Topic classification ──────────────────────────────────────────────────

    async def classify_topic(
        self, text: str
    ) -> tuple[str | None, float]:
        """
        Classify *text* into a topic category using patterns stored in long-term memory.

        Returns (category, confidence).  Returns (None, 0.0) when no pattern
        matches well enough or when not enough sessions have been seen yet.

        When confidence < topic_confidence_ask threshold AND enough sessions have
        accumulated, the caller should ask the user for the correct label.
        """
        import re as _re
        keywords = [
            w for w in _re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
            if len(w) > 2
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
        category   = best.get("category")

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
        best_text  = None

        for _entry_id, event in reversed(entries):
            payload_text = " ".join(str(v) for v in event.payload.values())
            words = set(_re.sub(r"[^a-z0-9\s]", "", payload_text.lower()).split())
            union = q_words | words
            if not union:
                continue
            score = len(q_words & words) / len(union)
            if score > best_score:
                best_score = score
                best_text  = payload_text[:500]

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
