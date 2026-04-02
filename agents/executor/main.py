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
import re
import shlex
import uuid
from pathlib import Path

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from core.base_agent import BaseAgent, run_agent
from core.config import Settings
from core.events.bus import Event, EventType
from core.context import truncate_task, truncate_command_output

TOOLS_DIR = Path("/workspace/tools")

log = structlog.get_logger()

# ── Built-in tool definitions seeded into the shared registry on startup ─────
# All agents can find and reference these. Executor owns + executes shell: tools.
_BUILTIN_TOOLS: list[tuple[str, str, str, list[str]]] = [
    # (name, description, shell_invocation, tags)
    ("list-docker-containers",
     "List all running Docker containers with status",
     "shell:docker ps",
     ["docker", "containers", "status"]),
    ("list-docker-all-containers",
     "List all Docker containers including stopped ones",
     "shell:docker ps -a",
     ["docker", "containers"]),
    ("list-workspace-tools",
     "List all saved reusable scripts in /workspace/tools/",
     "shell:ls -la /workspace/tools/",
     ["tools", "scripts", "workspace"]),
    ("list-agent-source",
     "List all agent source directories in the stack",
     "shell:ls /workspace/src/agents/",
     ["agents", "source"]),
    ("show-disk-usage",
     "Show disk usage of workspace volumes",
     "shell:df -h /workspace",
     ["disk", "workspace"]),
    ("list-python-processes",
     "Show running Python processes inside containers",
     "shell:ps aux | grep python",
     ["processes", "python"]),
    ("show-redis-streams",
     "List all Redis streams and their lengths",
     "shell:docker exec agent_redis redis-cli --no-auth-warning keys 'agents:*'",
     ["redis", "streams", "events"]),
]


# ── Inline command extraction (no LLM) ───────────────────────────────────────
# If the task payload is already a concrete shell command, extract it directly.

_CMD_PREFIXES = tuple(
    sorted(
        {"ls", "cat", "find", "grep", "echo", "pwd", "wc", "head", "tail", "diff",
         "sort", "uniq", "stat", "env", "which", "docker", "docker-compose",
         "git", "pip", "pip3", "python", "python3", "npm", "yarn", "apt", "apt-get",
         "tee", "cp", "mv", "rm", "chmod", "mkdir", "touch", "curl", "wget",
         "patch", "chown"},
        key=len, reverse=True,  # longer prefixes first to avoid shadowing
    )
)

_INLINE_PATTERN = re.compile(
    r'^(?:run|execute|exec(?:ute)?|please\s+run|can you\s+run)?\s*'
    r'((?:' + '|'.join(re.escape(p) for p in _CMD_PREFIXES) + r')\s+\S.{0,500})$',
    re.IGNORECASE,
)


def _extract_cmd(content: str) -> str | None:
    """
    Extract the first CMD: line from an LLM response.
    Returns the command string, or None if not present.
    """
    for line in content.splitlines():
        if "CMD:" in line:
            return line.split("CMD:", 1)[1].strip()
    return None


def _extract_inline_command(task: str) -> str | None:
    """
    If the task is a bare shell command or a thin wrapper around one
    (e.g. "run ls /workspace"), extract and return the command string.
    Returns None if the task is prose that needs LLM interpretation.
    """
    t = task.strip()
    m = _INLINE_PATTERN.match(t)
    return m.group(1).strip() if m else None


def _command_tags(cmd: str) -> list[str]:
    """Extract meaningful tags from a command string for capability indexing."""
    base = cmd.split()[0].lower() if cmd.split() else "unknown"
    tags = ["executor", "capability", base]
    if "docker" in cmd:
        tags.append("docker")
    if any(kw in cmd for kw in ("tee", "patch", "cp ", "mv ", "rm ")):
        tags.append("file_write")
    return tags


SYSTEM_PROMPT = """You are a tool execution specialist with full access to the agent stack \
source code and runtime environment.

## Shell commands
If a task requires a shell command, respond with exactly:
CMD: <the command>

Report results clearly: stdout, stderr, exit codes. Flag unexpected outputs for the orchestrator.

## Tool-building (PRIORITY)
Agents maintain a growing toolset in /workspace/tools/. Before generating a new shell command:
1. Check if a script already exists:  CMD: ls /workspace/tools/
2. If a matching script exists, run it instead of generating a new command.

After solving a new type of problem via a shell command, build a reusable script:
  CMD: tee /workspace/tools/<descriptive-name>.sh << 'EOF'
  #!/bin/bash
  # Description: <what this script does>
  <command>
  EOF
  CMD: chmod +x /workspace/tools/<descriptive-name>.sh

Name scripts clearly: list-docker-containers.sh, find-large-files.sh, check-agent-logs.sh.
Over time this toolset replaces LLM calls for routine operations.

## Command trust tiers
SAFE (instant, no log):    ls, cat, grep, find, echo, stat, head, tail, diff, env, which
AUTO_APPROVED (instant, audit log to #agent-logs):
  docker start/restart/stop/logs/exec/inspect
  tee, mkdir, touch, chmod, cp  (within /workspace only)
  python, python3, bash, sh
  git log/diff/status/show/fetch/pull
REQUIRES_APPROVAL (Discord gate):
  docker rm/rmi/prune  |  git push/commit/reset/rebase
  pip/npm/yarn/apt      |  curl/wget
  rm, mv, patch, chown

Prefer AUTO_APPROVED commands — operate autonomously. Only escalate for truly
destructive or external operations.

## Self-modification
The agent stack source is at /workspace/src/. Read, write, and restart autonomously.

Read a file:  CMD: cat /workspace/src/agents/executor/main.py
List agents:  CMD: ls /workspace/src/agents/
Write a file: CMD: tee /workspace/src/agents/executor/main.py << 'EOF'
              <content>
              EOF
Restart:      CMD: docker restart agent_executor

Container names: agent_orchestrator, agent_executor, agent_document_qa,
                 agent_code_search, agent_discord_bridge, agent_claude_code

## Self-improvement workflow
1. cat the file  2. propose change  3. tee (auto-approved)  4. docker restart (auto-approved)

## document_qa capabilities (delegate via orchestrator)
The document_qa agent can:
- Answer questions from /workspace/docs (PDF, markdown, text) via pypdf + LLM
- Review the full agent-stack source at /workspace/src and produce architecture summaries
- Generate LaTeX documents and compile them to PDF using texlive/latexmk
- Output files land in /workspace/docs/generated/
Route tasks involving documentation, architecture review, or PDF/LaTeX generation to document_qa.

## research agent (delegate via orchestrator)
The research agent searches the internet via SearXNG (Google + Bing + DDG, no API key):
- Decomposes questions → multi-query search → fact extraction → cross-source consensus
- Commits sourced facts to Postgres (table: research_sources) for future recall
- Zero Claude API calls — uses local LLM (Qwen) throughout
Route "what is X", "latest version of Y", "current news/status" queries to research.
Container name: agent_research
"""

# ── Three-tier command trust model ───────────────────────────────────────────
#
# SAFE            — read-only; run immediately, no gate, no audit.
# AUTO_APPROVED   — state-changing but bounded to the container/workspace;
#                   run immediately, emit an AUDIT event to the log channel.
# REQUIRES_APPROVAL — destructive, external, or irreversible; gate via Discord.
#
# The goal: agents operate autonomously for routine self-modification,
# container management, and scripting.  The approval gate is reserved for
# operations that could affect things outside the container or that cannot
# be rolled back (force-push, package installs that change the image, deletions).

SAFE_COMMANDS = {
    "ls", "cat", "find", "grep", "echo", "pwd", "wc",
    "head", "tail", "diff", "sort", "uniq", "stat",
    "env", "printenv", "which", "type", "id",
}

# Run automatically; emits AUDIT event so every action is visible in #agent-logs
AUTO_APPROVED_COMMANDS = {
    # Container lifecycle within the stack
    "docker",           # start/restart/stop/logs/exec/inspect (NOT rm/rmi)
    "docker-compose",
    # Workspace file writes
    "tee",              # write files inside /workspace — blocked for /workspace/user outside tests
    "mkdir", "touch", "chmod", "cp",
    # In-container code execution
    "python", "python3",
    # Non-destructive git operations
    "git",              # log/diff/status/show/fetch/pull — push/commit still blocked
    # Tool scripts
    "bash", "sh",
}

# Require explicit Discord approval — destructive, network-external, or irreversible
REQUIRES_APPROVAL = {
    # Package management (modifies the container image effectively)
    "pip", "pip3", "npm", "yarn",
    "apt", "apt-get",
    # External network calls
    "curl", "wget",
    # Destructive file operations (can delete user data)
    "rm", "mv", "patch",
    # Privileged container operations (permanent removal)
    "chown",
}


class ExecutorAgent(BaseAgent):

    async def on_startup(self) -> None:
        log.info("executor.startup")
        for name, desc, inv, tags in _BUILTIN_TOOLS:
            await self.memory.register_tool(name, desc, "executor", inv, tags, "executor")
        log.info("executor.tools_seeded", count=len(_BUILTIN_TOOLS))

    async def handle_event(self, event: Event) -> None:
        if event.type == EventType.TASK_ASSIGNED:
            if event.payload.get("assigned_to") == "executor":
                await self._handle_task(event)

    async def _handle_task(self, event: Event) -> None:
        task = truncate_task(event.payload.get("task", ""))
        task_id = event.task_id
        subtask_id = event.payload.get("subtask_id")
        parent_task_id = event.payload.get("parent_task_id")
        # Orchestrator may pass a timeout hint for long-running operations
        # (e.g. docker build, npm install). Falls back to the 60 s default.
        timeout = int(event.payload.get("timeout_hint", 60))

        log.info("executor.task", task=task[:80], subtask_id=subtask_id)

        await self.emit(
            EventType.AGENT_THINKING,
            payload={"task": task, "task_id": task_id},
        )

        # ── 1. Try inline command extraction (no LLM) ─────────────────────
        inline_cmd = _extract_inline_command(task)
        if inline_cmd:
            log.debug("executor.inline_command", cmd=inline_cmd[:80])
            result = await self._run_command(inline_cmd, task, task_id, timeout=timeout)
        else:
            # ── 2. Check local tool scripts ───────────────────────────────
            local_tool = self._find_local_tool(task)
            if local_tool:
                log.info("executor.local_tool_hit", tool=local_tool, task=task[:80])
                result = await self._run_command(f"bash {local_tool}", task, task_id, timeout=timeout)
            else:
                # ── 3. Search shared tool registry ────────────────────────
                tool_hits = await self.search_tools(task)
                shell_hit = next(
                    (
                        t for t in tool_hits
                        if t["invocation"].startswith("shell:")
                        and t.get("similarity", 0) >= 0.82
                        and "{" not in t["invocation"]  # skip parameterised templates
                    ),
                    None,
                )
                if shell_hit:
                    cmd = shell_hit["invocation"][6:]  # strip "shell:" prefix
                    log.info("executor.tool_registry_hit",
                             tool=shell_hit["name"], sim=round(shell_hit.get("similarity", 0), 3))
                    await self.memory.increment_tool_usage(shell_hit["name"])
                    result = await self._run_command(cmd, task, task_id, timeout=timeout)
                else:
                    # ── 4. Try capability registry (no LLM) ───────────────────
                    cached_cmd = await self.memory.lookup_capability(task)
                    if cached_cmd:
                        log.info("executor.capability_hit", task=task[:80], cmd=cached_cmd[:80])
                        result = await self._run_command(cached_cmd, task, task_id, timeout=timeout)
                    else:
                        # ── 5. Fall back to LLM ───────────────────────────────
                        tools_ctx = self.format_tools_context(tool_hits)
                        messages = [
                            SystemMessage(content=SYSTEM_PROMPT + tools_ctx),
                            HumanMessage(content=f"Task: {task}"),
                        ]
                        response = await self.llm_invoke(messages)
                        content = response.content

                        cmd_line = self._extract_cmd(content)

                        # ── 5a. Retry with a strict format prompt if no CMD: found ──
                        if not cmd_line:
                            log.warning(
                                "executor.no_cmd_in_response",
                                task=task[:80],
                                preview=content[:120].replace("\n", " "),
                            )
                            retry_messages = [
                                SystemMessage(content=(
                                    "You are a shell command executor. "
                                    "Respond with EXACTLY one line in this format:\n"
                                    "CMD: <shell command>\n\n"
                                    "No explanation. No markdown. Just CMD: followed by the command."
                                )),
                                HumanMessage(content=f"Task: {task}"),
                            ]
                            try:
                                retry_resp = await self.llm_invoke(retry_messages)
                                cmd_line = self._extract_cmd(retry_resp.content)
                            except Exception as exc:
                                log.warning("executor.cmd_retry_failed", error=str(exc))

                        result = content if not cmd_line else ""
                        if cmd_line:
                            result = await self._run_command(cmd_line, task, task_id, timeout=timeout)
                            # Store successful command for future reuse
                            if not result.startswith("Command denied") and not result.startswith("Error"):
                                tool_tags = _command_tags(cmd_line)
                                await self.memory.store_capability(task, cmd_line, tool_tags)
                                # Also register as a named tool for cross-agent visibility
                                words = [w for w in re.sub(r"[^a-z0-9\s]", "", task.lower()).split() if len(w) > 2][:4]
                                tool_name = "-".join(words)[:50]
                                if tool_name:
                                    await self.memory.register_tool(
                                        tool_name, task[:120], "executor",
                                        f"shell:{cmd_line}", tool_tags, self.role,
                                    )
                                # Promote complex commands to named tool scripts
                                self._maybe_save_tool(task, cmd_line)
                        else:
                            # No command could be extracted — mark the result so the
                            # orchestrator's validator can detect it as incomplete.
                            result = f"[EXECUTOR_NO_CMD] {content}"

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

    # ── Local toolset ────────────────────────────────────────────────────────

    def _find_local_tool(self, task: str) -> str | None:
        """
        Scan /workspace/tools/ for a script whose filename keywords overlap
        with the task keywords. Returns the script path if a good match is found.
        """
        if not TOOLS_DIR.exists():
            return None
        task_words = set(re.sub(r"[^a-z0-9\s]", "", task.lower()).split()) - {"the", "a", "an", "in", "of", "for", "to"}
        best_path: str | None = None
        best_score = 0
        for script in TOOLS_DIR.glob("*.sh"):
            name_words = set(re.sub(r"[^a-z0-9\s]", " ", script.stem.replace("-", " ").replace("_", " ")).split())
            score = len(task_words & name_words)
            if score >= 2 and score > best_score:
                best_score = score
                best_path = str(script)
        return best_path

    def _maybe_save_tool(self, task: str, cmd: str) -> None:
        """
        Save a successful LLM-generated command as a named tool script for
        future reuse without LLM. Only saves non-trivial multi-token commands.
        """
        if len(cmd.split()) < 3:
            return  # too simple to be worth a script
        try:
            TOOLS_DIR.mkdir(parents=True, exist_ok=True)
            # Derive a descriptive name from task keywords
            words = [w for w in re.sub(r"[^a-z0-9\s]", "", task.lower()).split() if len(w) > 2][:5]
            name = "-".join(words)[:50]
            if not name:
                return
            script_path = TOOLS_DIR / f"{name}.sh"
            if script_path.exists():
                return  # never overwrite existing tools
            content = f"#!/bin/bash\n# Generated from task: {task[:120]}\n{cmd}\n"
            script_path.write_text(content)
            script_path.chmod(0o755)
            log.info("executor.tool_saved", script=str(script_path), cmd=cmd[:80])
        except Exception as exc:
            log.debug("executor.tool_save_failed", error=str(exc))

    def _try_command_recovery(
        self,
        base_cmd: str,
        parts: list[str],
        stdout: str,
        stderr: str,
    ) -> tuple[str, str] | None:
        """
        Return (alternative_command, reason) for common deterministic failure patterns.
        Returns None when no safe recovery is known.
        """
        combined = (stdout + stderr).lower()

        # Command not found → try known binary aliases
        if "not found" in combined or "command not found" in combined:
            aliases: dict[str, str] = {
                "python":  "python3",
                "python3": "python",
                "pip":     "pip3",
                "pip3":    "pip",
            }
            alt = aliases.get(base_cmd)
            if alt:
                rest = (" " + " ".join(parts[1:])) if len(parts) > 1 else ""
                return alt + rest, f"`{base_cmd}` not found — retrying with `{alt}`"

        return None

    # ── Safety guardrails for AUTO_APPROVED docker/git subcommands ───────────────
    # Certain subcommands within AUTO_APPROVED bases are still destructive.
    # If the full command matches any of these patterns, escalate to REQUIRES_APPROVAL.
    _ESCALATE_PATTERNS: list[re.Pattern] = [
        re.compile(r'\bdocker\s+(rm|rmi|volume\s+rm|network\s+rm|system\s+prune)\b', re.I),
        re.compile(r'\bgit\s+(push|commit|reset|rebase|merge|cherry-pick|force)\b', re.I),
        re.compile(r'\btee\b.*/workspace/(user|projects)/', re.I),  # outside container workspace
    ]

    def _needs_escalation(self, cmd: str) -> bool:
        return any(p.search(cmd) for p in self._ESCALATE_PATTERNS)

    async def _run_command(
        self, cmd: str, task: str, task_id: str, timeout: int = 60, _recovery: bool = False
    ) -> str:
        """Run a shell command after three-tier trust check."""
        try:
            parts = shlex.split(cmd)
        except ValueError as e:
            return f"Error parsing command: {e}"

        base_cmd = parts[0] if parts else ""

        if base_cmd in REQUIRES_APPROVAL or (base_cmd in AUTO_APPROVED_COMMANDS and self._needs_escalation(cmd)):
            approved = await self._request_approval(cmd, task, task_id)
            if not approved:
                log.info("executor.command_denied", cmd=cmd)
                return f"Command denied by user: `{cmd}`"
        elif base_cmd in AUTO_APPROVED_COMMANDS:
            # Run immediately — emit audit event so the action is visible in #agent-logs
            await self._emit_audit(cmd, task_id)
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

            output = f"Exit code: {code}\n"
            if out:
                output += f"STDOUT:\n{out}\n"
            if err:
                output += f"STDERR:\n{err}\n"

            # Auto-recover from common failures (one level deep only)
            if code != 0 and not _recovery:
                recovery = self._try_command_recovery(base_cmd, parts, out, err)
                if recovery:
                    alt_cmd, reason = recovery
                    log.info("executor.auto_recovery", original=cmd, recovery=alt_cmd, reason=reason)
                    return await self._run_command(alt_cmd, task, task_id, timeout, _recovery=True)

            return truncate_command_output(output)

        except asyncio.TimeoutError:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Execution error: {e}"

    async def _emit_audit(self, cmd: str, task_id: str) -> None:
        """
        Emit an audit log event for AUTO_APPROVED commands.
        The discord bridge posts these to #agent-logs so every automated
        action remains visible without blocking execution.
        """
        await self.emit(
            EventType.AGENT_TOOL_CALL,
            payload={
                "command": cmd,
                "task_id": task_id,
                "tier": "auto_approved",
                "audit": True,
            },
        )
        log.info("executor.auto_approved", cmd=cmd[:120])

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
