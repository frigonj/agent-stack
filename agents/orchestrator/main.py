"""
agents/orchestrator/main.py
────────────────────────────
Orchestrator agent — plans tasks, delegates to specialist agents in parallel,
aggregates results, and proactively generates work via a background think loop.

Discord replies are always the highest priority: every TASK_CREATED event
gets a synthesised, meaningful response — never a raw specialist relay.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from core.base_agent import BaseAgent, run_agent
from core.config import Settings
from core.events.bus import Event, EventType

log = structlog.get_logger()

SYSTEM_PROMPT = """You are the orchestrator of a local AI agent stack running on a developer's \
machine. You coordinate a team of specialist agents, maintain persistent memory across sessions, \
and are the sole point of contact between the user (via Discord) and the agent pipeline.

## Your specialists

- **document_qa**: answers questions from PDFs and text files stored in /workspace/docs.
  Use for: summarisation, Q&A, fact extraction from documents.

- **code_search**: indexes and searches code repositories in /workspace/repos.
  Use for: finding functions/classes, understanding patterns, code analysis.

- **executor**: runs shell commands and can modify the agent stack's own source code.
  Safe commands (ls, cat, grep, find, etc.) run immediately. Privileged commands
  (git, pip, curl, python, tee, docker) require Discord approval before running.
  Source files are at /workspace/src/. Docker containers can be restarted via
  `docker restart <container_name>`.
  Use for: file operations, running scripts, self-modification, service restarts.

- **direct**: YOU answer this yourself, right now. Use for: greetings, status checks,
  conversational questions, anything you can answer without specialist help.
  This is the HIGHEST PRIORITY path — prefer it whenever possible.

## Parallel execution

Break tasks into 1–4 independent subtasks. Return your plan as JSON only — no markdown:
{"subtasks": [{"task": "...", "agent": "executor|document_qa|code_search|direct"}]}

Use agent="direct" for:
  - Greetings and social messages ("hello", "good morning")
  - Status questions ("how are you", "what are you doing")
  - Questions about your own capabilities, memory, or agents
  - Any task you can answer from your knowledge + recalled memory alone

Use specialists only when you genuinely need them (file search, code search, shell commands).

## Your memory

1. **Long-term memory (PostgreSQL + pgvector)** — recall() before every task.
2. **Short-term memory (Redis Streams)** — live task state. Ephemeral.

Call `stage_finding()` or `promote_now()` for anything worth remembering long-term.

## Proactive think loop

Every 2 minutes you scan memory for open problems or improvements. If you find clear
actionable work, spawn a task. Otherwise respond IDLE.

## Your responsibilities

1. recall() — always check long-term memory first.
2. Plan   — JSON plan with 1–4 subtasks. Default to direct when no specialist is needed.
3. Delegate — dispatch concurrently; or answer directly.
4. Synthesise — ALWAYS use your LLM to form the final reply. Never relay raw specialist output.
5. Promote — stage findings worth keeping.
6. Reply  — final synthesised answer goes to broadcast → Discord.
"""


@dataclass
class ParallelPlan:
    parent_task_id: str
    parent_task: str
    pending: set = field(default_factory=set)   # subtask_ids not yet received
    results: list = field(default_factory=list)  # (source, result) tuples
    created_at: float = field(default_factory=time.time)


class OrchestratorAgent(BaseAgent):

    # Think cycle runs every 2 minutes
    think_interval = 120

    # Timeout for stale parallel plans (10 minutes)
    PLAN_TIMEOUT = 600

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._pending: dict[str, ParallelPlan] = {}

    async def on_startup(self) -> None:
        log.info("orchestrator.startup")

    # ── Event handling ───────────────────────────────────────────────────────

    async def handle_event(self, event: Event) -> None:
        if event.type == EventType.TASK_CREATED:
            await self._handle_task(event)

        elif event.type == EventType.TASK_COMPLETED:
            await self._handle_result(event)

        elif event.type == EventType.AGENT_STARTED:
            log.info("orchestrator.agent_joined", agent=event.source)

    # ── Task intake ──────────────────────────────────────────────────────────

    async def _handle_task(self, event: Event) -> None:
        task = event.payload.get("task", "")
        task_id = event.task_id
        discord_message_id = event.payload.get("discord_message_id")
        log.info("orchestrator.task_received", task=task[:80], task_id=task_id)

        await self._run_task(task, task_id, discord_message_id=discord_message_id)

    async def _run_task(
        self,
        task: str,
        task_id: str,
        discord_message_id: str | None = None,
    ) -> None:
        """Full planning + dispatch pipeline. Always produces a synthesised reply."""
        # Check long-term memory first
        prior_knowledge = await self.recall(task)
        context = ""
        if prior_knowledge:
            context = "\n\nRelevant prior knowledge:\n" + "\n".join(
                f"- [{e['topic']}] {e['content'][:200]}" for e in prior_knowledge
            )
            log.info("orchestrator.prior_knowledge_found", count=len(prior_knowledge))

        # Generate a structured parallel plan
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Task: {task}{context}\n\n"
                "Return your subtask plan as JSON only — no markdown, no explanation:\n"
                '{"subtasks": [{"task": "...", "agent": "executor|document_qa|code_search|direct"}]}'
            )),
        ]
        response = await self.llm.ainvoke(messages)
        raw = response.content.strip()

        # Extract JSON even if the LLM wraps it in markdown fences
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            plan = json.loads(raw)
            subtasks = plan["subtasks"]
            if not isinstance(subtasks, list) or not subtasks:
                raise ValueError("empty subtasks")
        except Exception:
            log.warning("orchestrator.plan_parse_failed", raw=raw[:200])
            subtasks = [{"task": task, "agent": "direct"}]

        log.info("orchestrator.plan", subtask_count=len(subtasks), task_id=task_id)

        # Check if all subtasks are "direct" — handle entirely here
        direct_subtasks = [s for s in subtasks if s.get("agent") == "direct"]
        specialist_subtasks = [s for s in subtasks if s.get("agent") != "direct"]

        if not specialist_subtasks:
            # Pure direct answer — no specialist delegation needed
            await self._answer_directly(task, task_id, context, discord_message_id)
            return

        # Dispatch specialist subtasks
        await self._dispatch_subtasks(
            task_id, task, specialist_subtasks, context,
            direct_tasks=direct_subtasks,
            discord_message_id=discord_message_id,
        )

    # ── Direct answer ────────────────────────────────────────────────────────

    async def _answer_directly(
        self,
        task: str,
        task_id: str,
        context: str = "",
        discord_message_id: str | None = None,
    ) -> None:
        """Answer the task directly using the LLM without delegating."""
        agent_status = (
            "Agents online: orchestrator, document_qa, code_search, executor, discord_bridge. "
            "Long-term memory: PostgreSQL + pgvector. Short-term memory: Redis Streams."
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Task (answer directly): {task}\n"
                f"Agent status: {agent_status}"
                f"{context}"
            )),
        ]
        response = await self.llm.ainvoke(messages)
        result = response.content

        await self._publish_reply(result, task_id, discord_message_id)

    # ── Parallel dispatch ────────────────────────────────────────────────────

    async def _dispatch_subtasks(
        self,
        parent_task_id: str,
        parent_task: str,
        subtasks: list[dict],
        context: str,
        direct_tasks: list[dict] | None = None,
        discord_message_id: str | None = None,
    ) -> None:
        pending_ids: set[str] = set()
        direct_results: list[tuple[str, str]] = []

        # Resolve any direct subtasks immediately
        for dt in (direct_tasks or []):
            direct_results.append(("orchestrator_direct", dt["task"]))

        for st in subtasks:
            subtask_id = str(uuid.uuid4())
            pending_ids.add(subtask_id)
            agent = st.get("agent", "executor")

            await self.emit(
                EventType.TASK_ASSIGNED,
                payload={
                    "task": st["task"],
                    "assigned_to": agent,
                    "parent_task_id": parent_task_id,
                    "subtask_id": subtask_id,
                },
                target=agent,
            )
            log.info("orchestrator.dispatched", agent=agent, subtask_id=subtask_id)

        plan = ParallelPlan(
            parent_task_id=parent_task_id,
            parent_task=parent_task,
            pending=pending_ids,
            results=direct_results,
        )
        plan._context = context  # type: ignore[attr-defined]
        plan._discord_message_id = discord_message_id  # type: ignore[attr-defined]
        self._pending[parent_task_id] = plan

    # ── Result aggregation ───────────────────────────────────────────────────

    SPECIALIST_ROLES = {"document_qa", "code_search", "executor"}

    async def _handle_result(self, event: Event) -> None:
        if event.source not in self.SPECIALIST_ROLES:
            return  # Ignore our own re-broadcasts to avoid relay loop

        result = event.payload.get("result", "")
        subtask_id = event.payload.get("subtask_id")
        parent_task_id = event.payload.get("parent_task_id")

        log.info("orchestrator.result_received", source=event.source, subtask_id=subtask_id)

        # Match to a pending parallel plan using parent_task_id
        plan = self._pending.get(parent_task_id) if parent_task_id else None

        if plan:
            plan.results.append((event.source, result))
            if subtask_id:
                plan.pending.discard(subtask_id)
            else:
                # Specialist didn't echo subtask_id — clear all pending for this agent
                # so we don't block forever
                plan.pending.clear()

            if not plan.pending:
                del self._pending[parent_task_id]
                discord_message_id = getattr(plan, "_discord_message_id", None)
                await self._aggregate_and_respond(plan, discord_message_id)

            if event.payload.get("promote", False):
                topic = event.payload.get("topic", event.source)
                await self.promote_now(result, topic=topic, tags=[event.source, "result"])
            return

        # No matching plan — synthesise a reply from the raw result
        # (handles legacy single-subtask results and agents that don't pass IDs)
        await self._synthesise_and_reply(
            original_task=event.payload.get("task", ""),
            raw_results=[(event.source, result)],
            task_id=event.task_id,
        )

        if event.payload.get("promote", False):
            topic = event.payload.get("topic", event.source)
            await self.promote_now(result, topic=topic, tags=[event.source, "result"])

    async def _aggregate_and_respond(
        self, plan: ParallelPlan, discord_message_id: str | None = None
    ) -> None:
        """Synthesise results from one or more specialists into a final reply."""
        await self._synthesise_and_reply(
            original_task=plan.parent_task,
            raw_results=plan.results,
            task_id=plan.parent_task_id,
            context=getattr(plan, "_context", ""),
            discord_message_id=discord_message_id,
        )

    async def _synthesise_and_reply(
        self,
        original_task: str,
        raw_results: list[tuple[str, str]],
        task_id: str,
        context: str = "",
        discord_message_id: str | None = None,
    ) -> None:
        """Always run results through LLM before sending to Discord."""
        combined = "\n\n".join(
            f"[{source}]: {result}" for source, result in raw_results
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Original task: {original_task}{context}\n\n"
                f"Information gathered:\n{combined}\n\n"
                "Based on the above, write a clear, concise, helpful response for the user. "
                "If the specialists found nothing useful, say so honestly and offer alternatives."
            )),
        ]
        response = await self.llm.ainvoke(messages)
        await self._publish_reply(response.content, task_id, discord_message_id)

    async def _publish_reply(
        self,
        result: str,
        task_id: str,
        discord_message_id: str | None = None,
    ) -> None:
        """Send the final reply to broadcast (→ Discord)."""
        payload = {"result": result, "task_id": task_id}
        if discord_message_id:
            payload["discord_message_id"] = discord_message_id
        await self.emit(EventType.TASK_COMPLETED, payload=payload, target="broadcast")

    # ── Proactive think loop ─────────────────────────────────────────────────

    async def think(self) -> None:
        """
        Every think_interval seconds: scan memory for open problems or improvement
        opportunities. If something actionable is found, spawn a task autonomously.
        """
        # Expire stale plans
        now = time.time()
        stale = [
            tid for tid, p in self._pending.items()
            if now - p.created_at > self.PLAN_TIMEOUT
        ]
        for tid in stale:
            plan = self._pending.pop(tid)
            log.warning("orchestrator.plan_expired", task_id=tid, task=plan.parent_task[:60])

        # Scan long-term memory for open issues and improvement candidates
        findings = await self.recall(
            "recent errors bugs improvements open tasks suggestions self-modification"
        )
        if not findings:
            return

        context = "\n".join(
            f"- [{e['topic']}] {e['content'][:200]}" for e in findings[:6]
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Recent context from long-term memory:\n{context}\n\n"
                "Is there something I should proactively work on right now?\n"
                "Consider: fixing a logged error, improving agent code, running diagnostics.\n"
                "If nothing is urgent or actionable, respond with exactly: IDLE"
            )),
        ]
        response = await self.llm.ainvoke(messages)
        proposal = response.content.strip()

        if "IDLE" in proposal.upper()[:20]:
            log.debug("orchestrator.think_idle")
            return

        task_id = str(uuid.uuid4())
        log.info("orchestrator.proactive_task", task=proposal[:120], task_id=task_id)

        await self.emit(
            EventType.THINK_CYCLE,
            payload={"proposal": proposal[:300], "task_id": task_id},
            target="broadcast",
        )
        await self._run_task(proposal, task_id)

    async def on_shutdown(self) -> None:
        log.info("orchestrator.shutdown", pending_plans=len(self._pending))


if __name__ == "__main__":
    settings = Settings()
    agent = OrchestratorAgent(settings)
    run_agent(agent)
