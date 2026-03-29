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

SYSTEM_PROMPT = """You are an orchestrator agent managing a team of specialist AI agents.

Your specialists:
- document_qa: answers questions from PDF/text documents
- code_search: searches and analyzes codebases
- executor: runs tools, shell commands, and file operations

Your job:
1. Receive a task from the user or broadcast stream
2. Break it into sub-tasks
3. Delegate each sub-task to the right specialist via the event bus
4. Aggregate results and produce a final response

Always check long-term memory before starting a task — the team may have solved this before.
Be explicit about which agent you are routing to and why.
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
