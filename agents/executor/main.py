"""
agents/executor/main.py
────────────────────────
Tool executor agent — runs shell commands, reads/writes files, restarts services.
The hands of the pipeline, including self-modification of the agent stack.

Approval gates:
  SAFE_COMMANDS     — read-only, run immediately.
  REQUIRES_APPROVAL — state-changing; held for Discord approval before running.
  Anything else     — blocked entirely.
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

SYSTEM_PROMPT = """You are a tool execution specialist with full access to the agent stack \
source code and runtime environment.

## Shell commands
If a task requires a shell command, respond with exactly:
CMD: <the command>

Report results clearly: stdout, stderr, exit codes. Flag unexpected outputs for the orchestrator.

## Self-modification
The agent stack source is mounted at /workspace/src/. You can read any agent file and propose
improvements by writing new versions. All writes, patches, and restarts require human approval.

Read a file:
  CMD: cat /workspace/src/agents/executor/main.py

List all agents:
  CMD: ls /workspace/src/agents/

Write a new version of a file (requires approval):
  CMD: tee /workspace/src/agents/executor/main.py << 'EOF'
  <full file content>
  EOF

Restart a service after a code change (requires approval):
  CMD: docker restart agent_executor

Container names: agent_orchestrator, agent_executor, agent_document_qa,
                 agent_code_search, agent_discord_bridge

## Self-improvement workflow
1. cat the file you want to improve.
2. Propose the change in your response.
3. Use tee to write the updated file (will trigger approval gate).
4. Use docker restart <name> to reload (will trigger approval gate).

When writing a self-modification, include a brief explanation of what changed and why.
"""

# Read-only commands — run without approval
SAFE_COMMANDS = {
    "ls", "cat", "find", "grep", "echo", "pwd", "wc",
    "head", "tail", "diff", "sort", "uniq", "stat",
    "env", "printenv", "which", "type", "id",
}

# Commands that can modify state — require explicit user approval via Discord
REQUIRES_APPROVAL = {
    # Network / package managers
    "curl", "wget", "pip", "pip3", "apt", "apt-get", "npm", "yarn",
    # Code execution
    "python", "python3",
    # Version control
    "git",
    # File write / patching (self-modification)
    "tee", "patch", "cp", "mv", "rm", "chmod", "chown", "mkdir", "touch",
    # Container management (self-restart)
    "docker", "docker-compose",
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
        subtask_id = event.payload.get("subtask_id")
        parent_task_id = event.payload.get("parent_task_id")

        log.info("executor.task", task=task[:80], subtask_id=subtask_id)

        await self.emit(
            EventType.AGENT_THINKING,
            payload={"task": task, "task_id": task_id},
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Task: {task}"),
        ]
        response = await self.llm.ainvoke(messages)
        content = response.content

        result = content
        if "CMD:" in content:
            # Extract the first CMD: line
            cmd_line = next(
                (ln.split("CMD:", 1)[1].strip() for ln in content.splitlines() if "CMD:" in ln),
                None,
            )
            if cmd_line:
                result = await self._run_command(cmd_line, task, task_id)

        await self.emit(
            EventType.AGENT_TOOL_RESULT,
            payload={"result": result[:2000], "task_id": task_id},
        )

        # Stage findings for long-term memory
        if any(kw in task.lower() for kw in ["bug", "error", "pattern", "found", "discovered", "improve"]):
            self.stage_finding(
                content=f"Executor task: {task}\nResult: {result[:500]}",
                topic="execution_findings",
                tags=["executor", "tool"],
            )

        # Mark self-modifications in memory immediately
        if any(kw in task.lower() for kw in ["modify", "update", "patch", "rewrite", "fix code"]):
            await self.promote_now(
                content=f"Self-modification applied.\nTask: {task}\nOutcome: {result[:400]}",
                topic="self_modification",
                tags=["executor", "self_modify"],
            )

        await self.emit(
            EventType.TASK_COMPLETED,
            payload={
                "result": result,
                "task_id": task_id,
                "subtask_id": subtask_id,
                "parent_task_id": parent_task_id,
            },
            target="orchestrator",
        )

    async def _run_command(self, cmd: str, task: str, task_id: str, timeout: int = 60) -> str:
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
                return f"Command denied by user: `{cmd}`"
        elif base_cmd not in SAFE_COMMANDS:
            log.warning("executor.blocked_command", cmd=base_cmd)
            return f"Command '{base_cmd}' is not on the allowlist and cannot be run."

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
                result += f"STDOUT:\n{out[:2000]}\n"
            if err:
                result += f"STDERR:\n{err[:500]}\n"
            return result

        except asyncio.TimeoutError:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Execution error: {e}"

    async def _request_approval(self, cmd: str, task: str, task_id: str) -> bool:
        """
        Emit an approval.required event and wait for the user's decision via Discord.
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
