"""
agents/orchestrator/main.py
────────────────────────────
Orchestrator agent — plans tasks, delegates to specialist agents,
aggregates results. The brain of the pipeline.
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from core.base_agent import BaseAgent, run_agent
from core.config import Settings
from core.events.bus import Event, EventType

log = structlog.get_logger()

SYSTEM_PROMPT = """You are the orchestrator of a local AI agent stack. You coordinate a team \
of specialist agents, maintain persistent memory across sessions, and are the sole point of \
contact between the user (via Discord) and the agent pipeline.

## Your specialists

- **document_qa**: answers questions from PDFs and text files stored in /workspace/docs.
  Route here for: summarisation, Q&A, fact extraction from documents.

- **code_search**: indexes and searches code repositories in /workspace/repos.
  Route here for: finding functions/classes, understanding patterns, code analysis.

- **executor**: runs shell commands on an allowlist. Safe commands (ls, cat, grep, find, etc.)
  run immediately. Privileged commands (git, pip, curl, python) require the user's Discord
  approval before executing. Route here for: file operations, running scripts, package installs.

## Your memory

You have two memory layers:

1. **Long-term memory (PostgreSQL + pgvector)** — knowledge the team has accumulated across
   all sessions. ALWAYS call `recall(task)` before starting any task. If relevant prior
   knowledge exists, use it to inform your plan and avoid redundant work.
   Call `stage_finding()` or `promote_now()` for anything worth remembering.

2. **Short-term memory (Redis Streams)** — live task state and inter-agent events in the
   current session. Ephemeral by design.

## Your responsibilities

1. Check long-term memory — recall before you plan.
2. Plan — decide which specialist(s) to involve and why.
3. Delegate — emit task.assigned events to the right specialists.
4. Aggregate — collect task.completed results and form a final response.
5. Promote — stage findings worth keeping for future sessions.
6. Reply — your final answer goes to broadcast and back to the user on Discord.

Be concise and specific when delegating. Tell each specialist exactly what you need.
When results come back, synthesise them into a clear answer rather than forwarding raw output.
"""


class OrchestratorAgent(BaseAgent):

    async def on_startup(self) -> None:
        log.info("orchestrator.startup")

    async def handle_event(self, event: Event) -> None:
        if event.type == EventType.TASK_CREATED:
            await self._handle_task(event)

        elif event.type == EventType.TASK_COMPLETED:
            await self._handle_result(event)

        elif event.type == EventType.AGENT_STARTED:
            log.info("orchestrator.agent_joined", agent=event.source)

    async def _handle_task(self, event: Event) -> None:
        task = event.payload.get("task", "")
        task_id = event.task_id
        log.info("orchestrator.task_received", task=task[:80], task_id=task_id)

        # Check long-term memory first
        prior_knowledge = await self.recall(task)
        context = ""
        if prior_knowledge:
            context = "\n\nRelevant prior knowledge:\n" + "\n".join(
                f"- [{e['topic']}] {e['content']}" for e in prior_knowledge
            )
            log.info("orchestrator.prior_knowledge_found", count=len(prior_knowledge))

        # Plan via LLM
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Task: {task}{context}\n\nWhich agent(s) should handle this and how?"),
        ]
        response = await self.llm.ainvoke(messages)
        plan = response.content

        log.info("orchestrator.plan", plan=plan[:200])

        # Route to specialist(s) based on plan
        await self._route_task(task, task_id, plan)

    async def _route_task(self, task: str, task_id: str, plan: str) -> None:
        """Simple keyword routing — extend with LLM-based routing for complex cases."""
        plan_lower = plan.lower()

        routed = False
        if any(kw in plan_lower for kw in ["document", "pdf", "file", "read"]):
            await self.emit(
                EventType.TASK_ASSIGNED,
                payload={"task": task, "plan": plan, "assigned_to": "document_qa"},
                target="document_qa",
            )
            routed = True

        if any(kw in plan_lower for kw in ["code", "repo", "search", "function", "class"]):
            await self.emit(
                EventType.TASK_ASSIGNED,
                payload={"task": task, "plan": plan, "assigned_to": "code_search"},
                target="code_search",
            )
            routed = True

        if any(kw in plan_lower for kw in ["run", "execute", "shell", "tool", "command"]):
            await self.emit(
                EventType.TASK_ASSIGNED,
                payload={"task": task, "plan": plan, "assigned_to": "executor"},
                target="executor",
            )
            routed = True

        if not routed:
            # Default: send to executor as general handler
            await self.emit(
                EventType.TASK_ASSIGNED,
                payload={"task": task, "plan": plan, "assigned_to": "executor"},
                target="executor",
            )

    SPECIALIST_ROLES = {"document_qa", "code_search", "executor"}

    async def _handle_result(self, event: Event) -> None:
        # Only handle results from specialists — ignore our own re-broadcasts
        # to prevent the orchestrator from looping on events it put on broadcast.
        if event.source not in self.SPECIALIST_ROLES:
            return

        result = event.payload.get("result", "")
        source = event.source
        task_id = event.task_id

        log.info("orchestrator.result_received", source=source, task_id=task_id)

        # Relay the final result to broadcast so the Discord bridge picks it up
        await self.emit(
            EventType.TASK_COMPLETED,
            payload={"result": result, "task_id": task_id},
            target="broadcast",
        )

        # Promote to long-term memory if flagged
        if event.payload.get("promote", False):
            topic = event.payload.get("topic", source)
            await self.promote_now(result, topic=topic, tags=[source, "result"])

    async def on_shutdown(self) -> None:
        log.info("orchestrator.shutdown")


if __name__ == "__main__":
    settings = Settings()
    agent = OrchestratorAgent(settings)
    run_agent(agent)
