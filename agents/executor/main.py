"""
agents/executor/main.py
────────────────────────
Tool executor agent — runs shell commands, reads/writes files,
calls external APIs. The hands of the pipeline.

Approval gates: commands in REQUIRES_APPROVAL are held until the user
approves or denies via Discord. Read-only commands in SAFE_COMMANDS run
without approval. All other commands are blocked entirely.
"""

from __future__ import annotations

import asyncio
import shlex
import uuid

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from core.base_agent import BaseAgent, run_agent
from core.config import Settings
from core.events.bus import Event, EventType

log = structlog.get_logger()

SYSTEM_PROMPT = """You are a tool execution specialist.
You receive tasks that require running commands, reading files, or calling APIs.
Before executing anything, confirm what you are about to do.
Report results clearly, including stdout, stderr, and exit codes.
Flag any errors or unexpected outputs for the orchestrator.
"""

# Read-only commands — run without approval
SAFE_COMMANDS = {
    "ls", "cat", "find", "grep", "echo", "pwd", "wc",
    "head", "tail", "diff", "sort", "uniq",
}

# Commands that can modify state — require explicit user approval via Discord
REQUIRES_APPROVAL = {
    "curl", "python", "python3", "pip", "git",
}


class ExecutorAgent(BaseAgent):

    async def on_startup(self) -> None:
        log.info("executor.startup")

    async def handle_event(self, event: Event) -> None:
        if event.type == EventType.TASK_ASSIGNED:
            if event.payload.get("assigned_to") == "executor":
                await self._handle_task(event)

    async def _handle_task(self, event: Event) -> None:
        task = event.payload.get("task", "")
        task_id = event.task_id
        log.info("executor.task", task=task[:80])

        await self.emit(
            EventType.AGENT_THINKING,
            payload={"task": task, "task_id": task_id},
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Task: {task}\n\n"
                "If this requires a shell command, respond with exactly:\n"
                "CMD: <the command>\n\n"
                "Otherwise, respond with your analysis directly."
            )),
        ]
        response = await self.llm.ainvoke(messages)
        content = response.content

        result = content
        if content.strip().startswith("CMD:"):
            cmd = content.split("CMD:", 1)[1].strip().splitlines()[0]
            result = await self._run_command(cmd, task, task_id)

        await self.emit(
            EventType.AGENT_TOOL_RESULT,
            payload={"result": result[:2000], "task_id": task_id},
        )

        if any(kw in task.lower() for kw in ["bug", "error", "pattern", "found", "discovered"]):
            self.stage_finding(
                content=f"Executor task: {task}\nResult: {result[:500]}",
                topic="execution_findings",
                tags=["executor", "tool"],
            )

        await self.emit(
            EventType.TASK_COMPLETED,
            payload={"result": result, "task_id": task_id},
            target="orchestrator",
        )

    async def _run_command(self, cmd: str, task: str, task_id: str, timeout: int = 30) -> str:
        """Run a shell command after allowlist and approval checks."""
        try:
            parts = shlex.split(cmd)
        except ValueError as e:
            return f"Error parsing command: {e}"

        base_cmd = parts[0] if parts else ""

        if base_cmd in REQUIRES_APPROVAL:
            approved = await self._request_approval(cmd, task, task_id)
            if not approved:
                log.info("executor.command_denied", cmd=cmd)
                return f"Command denied: `{cmd}`"
        elif base_cmd not in SAFE_COMMANDS:
            log.warning("executor.blocked_command", cmd=base_cmd)
            return f"Command '{base_cmd}' is not permitted."

        log.info("executor.running", cmd=cmd)
        await self.emit(EventType.AGENT_TOOL_CALL, payload={"command": cmd})

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            out = stdout.decode(errors="replace")
            err = stderr.decode(errors="replace")
            code = proc.returncode

            result = f"Exit code: {code}\n"
            if out:
                result += f"STDOUT:\n{out[:1500]}\n"
            if err:
                result += f"STDERR:\n{err[:500]}\n"
            return result

        except asyncio.TimeoutError:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Execution error: {e}"

    async def _request_approval(self, cmd: str, task: str, task_id: str) -> bool:
        """
        Emit an approval.required event and wait for the user's decision
        via Discord (set as a Redis key by the discord_bridge).
        Returns True if approved, False if denied or timed out.
        """
        approval_id = str(uuid.uuid4())
        log.info("executor.approval_requested", cmd=cmd, approval_id=approval_id)

        await self.emit(
            EventType.APPROVAL_REQUIRED,
            payload={
                "approval_id": approval_id,
                "command": cmd,
                "task": task,
                "task_id": task_id,
            },
        )

        decision = await self.bus.wait_for_approval(approval_id, timeout=300)
        log.info("executor.approval_decision", decision=decision, approval_id=approval_id)
        return decision == "approved"

    async def on_shutdown(self) -> None:
        log.info("executor.shutdown")


if __name__ == "__main__":
    settings = Settings()
    agent = ExecutorAgent(settings)
    run_agent(agent)
