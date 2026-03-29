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
import signal
from abc import ABC, abstractmethod
from typing import Any, Optional

import structlog
from langchain_openai import ChatOpenAI

from core.events.bus import Event, EventBus, EventType
from core.memory.long_term import LongTermMemory
from core.config import Settings

log = structlog.get_logger()


class BaseAgent(ABC):
    """
    All agents inherit from this. Subclasses implement:
      - handle_event(event) → the agent's core logic per event
      - on_startup()        → optional async init
      - on_shutdown()       → optional async cleanup + memory promotion
    """

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
        )

        self._running = False
        self._findings: list[dict] = []    # Staged for promotion on shutdown

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        log.info("agent.starting", role=self.role)

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
        await self._event_loop()

    async def stop(self, summary: Optional[str] = None) -> None:
        self._running = False
        log.info("agent.stopping", role=self.role)

        await self.on_shutdown()

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

    # ── Event loop ───────────────────────────────────────────────────────────

    async def _event_loop(self) -> None:
        async for stream, entry_id, event in self.bus.consume(
            role=self.role,
            group=self.group_name,
            consumer=self.consumer_name,
        ):
            if not self._running:
                break
            try:
                await self.handle_event(event)
                await self.bus.ack(stream, self.group_name, entry_id)
            except Exception as exc:
                log.error("agent.event_error", role=self.role, error=str(exc), event_id=event.event_id)
                await self._emit_error(event, exc)

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
        """Query long-term memory before starting a task."""
        return await self.memory.search(query, semantic=semantic)

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


def run_agent(agent: BaseAgent) -> None:
    """Entry point for running an agent with graceful shutdown."""
    loop = asyncio.new_event_loop()

    async def _main():
        def _handle_signal():
            asyncio.create_task(agent.stop())

        loop.add_signal_handler(signal.SIGTERM, _handle_signal)
        loop.add_signal_handler(signal.SIGINT, _handle_signal)
        await agent.start()

    loop.run_until_complete(_main())
