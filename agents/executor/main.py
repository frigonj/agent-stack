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
# Format: (name, description-with-example, shell_invocation, tags)
# The description is embedded for semantic search — include usage examples so the
# planner can generate correct CMD: lines without guessing syntax.
_BUILTIN_TOOLS: list[tuple[str, str, str, list[str]]] = [
    # ── Docker ────────────────────────────────────────────────────────────────
    (
        "list-docker-containers",
        "List all running Docker containers with name, status, ports. "
        "Example: CMD: docker ps",
        "shell:docker ps",
        ["docker", "containers", "status", "list"],
    ),
    (
        "list-all-docker-containers",
        "List all Docker containers including stopped ones. Example: CMD: docker ps -a",
        "shell:docker ps -a",
        ["docker", "containers", "stopped"],
    ),
    (
        "docker-logs",
        "Tail logs of a Docker container. "
        "Example: CMD: docker logs --tail 100 agent_orchestrator",
        "shell:docker logs --tail 100 <container_name>",
        ["docker", "logs", "debug"],
    ),
    (
        "docker-restart",
        "Restart a Docker container by name. Requires AUTO_APPROVED tier. "
        "Example: CMD: docker restart agent_executor",
        "shell:docker restart <container_name>",
        ["docker", "restart", "container"],
    ),
    (
        "docker-exec",
        "Run a command inside a running container. "
        "Example: CMD: docker exec agent_postgres psql -U agent -d agentmem -c 'SELECT COUNT(*) FROM knowledge;'",
        "shell:docker exec <container> <command>",
        ["docker", "exec", "container"],
    ),
    (
        "docker-inspect",
        "Show detailed container metadata (env vars, mounts, network). "
        "Example: CMD: docker inspect agent_orchestrator",
        "shell:docker inspect <container_name>",
        ["docker", "inspect", "metadata"],
    ),
    (
        "docker-compose-ps",
        "Show status of all compose services. Example: CMD: docker compose ps",
        "shell:docker compose ps",
        ["docker", "compose", "services", "status"],
    ),
    # ── File operations ───────────────────────────────────────────────────────
    (
        "read-file",
        "Print file contents to stdout. "
        "Example: CMD: cat /workspace/src/core/config.py",
        "shell:cat <file_path>",
        ["file", "read", "cat"],
    ),
    (
        "list-directory",
        "List files and directories. Use -la for details, -R for recursive. "
        "Example: CMD: ls -la /workspace/src/agents/",
        "shell:ls -la <directory>",
        ["file", "list", "directory", "ls"],
    ),
    (
        "find-files",
        "Find files by name pattern recursively. "
        "Example: CMD: find /workspace -name '*.py' -not -path '*/__pycache__/*'",
        "shell:find <dir> -name '<pattern>'",
        ["file", "find", "search", "pattern"],
    ),
    (
        "grep-in-files",
        "Search file contents for a pattern. -r recursive, -n line numbers, -l files only. "
        "Example: CMD: grep -rn 'EventType' /workspace/src/core/",
        "shell:grep -rn '<pattern>' <directory>",
        ["grep", "search", "text", "pattern", "code"],
    ),
    (
        "head-file",
        "Print first N lines of a file. "
        "Example: CMD: head -50 /workspace/src/agents/orchestrator/main.py",
        "shell:head -<N> <file_path>",
        ["file", "read", "head"],
    ),
    (
        "tail-file",
        "Print last N lines of a file (useful for logs). "
        "Example: CMD: tail -100 /workspace/logs/orchestrator.log",
        "shell:tail -<N> <file_path>",
        ["file", "read", "tail", "logs"],
    ),
    (
        "write-file",
        "Write content to a file using tee. Requires AUTO_APPROVED tier. "
        "Example: CMD: tee /workspace/tools/my-script.sh << 'EOF'\\n#!/bin/bash\\n...\\nEOF",
        "shell:tee <file_path> << 'EOF'\\n<content>\\nEOF",
        ["file", "write", "create", "tee"],
    ),
    (
        "make-executable",
        "Make a script executable with chmod. "
        "Example: CMD: chmod +x /workspace/tools/my-script.sh",
        "shell:chmod +x <file_path>",
        ["file", "chmod", "executable", "script"],
    ),
    (
        "disk-usage",
        "Show disk space usage of a directory or volume. "
        "Example: CMD: df -h /workspace",
        "shell:df -h <path>",
        ["disk", "storage", "usage", "workspace"],
    ),
    (
        "directory-size",
        "Show the size of each subdirectory. Example: CMD: du -sh /workspace/*/",
        "shell:du -sh <path>/*/",
        ["disk", "size", "directory"],
    ),
    # ── Git ───────────────────────────────────────────────────────────────────
    (
        "git-log",
        "Show recent git commit history with author and message. "
        "Example: CMD: git -C /workspace/src log --oneline -20",
        "shell:git -C <repo_path> log --oneline -<N>",
        ["git", "log", "history", "commits"],
    ),
    (
        "git-status",
        "Show working tree status — modified, staged, untracked files. "
        "Example: CMD: git -C /workspace/src status",
        "shell:git -C <repo_path> status",
        ["git", "status", "changes"],
    ),
    (
        "git-diff",
        "Show unstaged changes in the working tree. "
        "Example: CMD: git -C /workspace/src diff",
        "shell:git -C <repo_path> diff",
        ["git", "diff", "changes"],
    ),
    (
        "git-pull",
        "Pull latest changes from remote. Example: CMD: git -C /workspace/src pull",
        "shell:git -C <repo_path> pull",
        ["git", "pull", "update", "sync"],
    ),
    # ── Python / pip ──────────────────────────────────────────────────────────
    (
        "run-python-script",
        "Execute a Python script. "
        "Example: CMD: python3 /workspace/tools/check_health.py",
        "shell:python3 <script_path>",
        ["python", "script", "run", "execute"],
    ),
    (
        "pip-install",
        "Install a Python package. Requires REQUIRES_APPROVAL tier. "
        "Example: CMD: pip install requests",
        "shell:pip install <package_name>",
        ["pip", "install", "python", "package"],
    ),
    (
        "pip-list",
        "List installed Python packages and versions. Example: CMD: pip list",
        "shell:pip list",
        ["pip", "python", "packages", "installed"],
    ),
    (
        "run-pytest",
        "Run the test suite with pytest. "
        "Example: CMD: pytest /workspace/src/tests/unit/ -v",
        "shell:pytest <test_path> -v",
        ["pytest", "test", "unit", "testing"],
    ),
    # ── Redis ─────────────────────────────────────────────────────────────────
    (
        "redis-list-keys",
        "List all Redis keys matching a pattern. "
        "Example: CMD: docker exec agent_redis redis-cli keys 'agents:*'",
        "shell:docker exec agent_redis redis-cli keys '<pattern>'",
        ["redis", "keys", "list"],
    ),
    (
        "redis-get-key",
        "Get the value of a Redis key. "
        "Example: CMD: docker exec agent_redis redis-cli get 'agent:status:orchestrator'",
        "shell:docker exec agent_redis redis-cli get '<key>'",
        ["redis", "get", "value"],
    ),
    (
        "redis-stream-length",
        "Get the number of entries in a Redis stream. "
        "Example: CMD: docker exec agent_redis redis-cli xlen agents:orchestrator",
        "shell:docker exec agent_redis redis-cli xlen <stream_name>",
        ["redis", "stream", "length", "queue"],
    ),
    (
        "redis-stream-read",
        "Read recent entries from a Redis stream. "
        "Example: CMD: docker exec agent_redis redis-cli xrevrange agents:broadcast + - COUNT 10",
        "shell:docker exec agent_redis redis-cli xrevrange <stream> + - COUNT <N>",
        ["redis", "stream", "read", "events"],
    ),
    # ── PostgreSQL ────────────────────────────────────────────────────────────
    (
        "postgres-query",
        "Run an SQL query against the agent memory database. "
        "Example: CMD: docker exec agent_postgres psql -U agent -d agentmem -c 'SELECT topic, COUNT(*) FROM knowledge GROUP BY topic;'",
        "shell:docker exec agent_postgres psql -U agent -d agentmem -c '<SQL>'",
        ["postgres", "sql", "database", "query"],
    ),
    (
        "postgres-count-knowledge",
        "Count knowledge entries in the agent long-term memory database. "
        "Example: CMD: docker exec agent_postgres psql -U agent -d agentmem -c 'SELECT COUNT(*) FROM knowledge;'",
        "shell:docker exec agent_postgres psql -U agent -d agentmem -c 'SELECT COUNT(*) FROM knowledge;'",
        ["postgres", "memory", "knowledge", "count"],
    ),
    (
        "postgres-list-tools",
        "List all registered tools in the shared tool registry. "
        "Example: CMD: docker exec agent_postgres psql -U agent -d agentmem -c 'SELECT name, owner_agent FROM tools ORDER BY name;'",
        "shell:docker exec agent_postgres psql -U agent -d agentmem -c 'SELECT name, owner_agent FROM tools ORDER BY name;'",
        ["postgres", "tools", "registry", "list"],
    ),
    # ── Process / system ──────────────────────────────────────────────────────
    (
        "list-processes",
        "List running processes. Use grep to filter. "
        "Example: CMD: ps aux | grep python",
        "shell:ps aux | grep <process_name>",
        ["processes", "system", "ps"],
    ),
    (
        "check-port",
        "Check if a port is listening. Example: CMD: ss -tlnp | grep 6379",
        "shell:ss -tlnp | grep <port>",
        ["network", "port", "listening"],
    ),
    (
        "curl-endpoint",
        "Make an HTTP request to an endpoint. Requires REQUIRES_APPROVAL. "
        "Example: CMD: curl -s http://localhost:1234/api/v0/models | python3 -m json.tool",
        "shell:curl -s <url>",
        ["curl", "http", "network", "api"],
    ),
    # ── Workspace / agent stack ───────────────────────────────────────────────
    (
        "list-workspace-tools",
        "List all saved reusable scripts in /workspace/tools/. "
        "Example: CMD: ls -la /workspace/tools/",
        "shell:ls -la /workspace/tools/",
        ["tools", "scripts", "workspace", "list"],
    ),
    (
        "list-agent-sources",
        "List all agent source directories in the stack. "
        "Example: CMD: ls /workspace/src/agents/",
        "shell:ls /workspace/src/agents/",
        ["agents", "source", "list"],
    ),
    (
        "show-redis-streams",
        "List all Redis event streams and their lengths. "
        "Example: CMD: docker exec agent_redis redis-cli --no-auth-warning keys 'agents:*'",
        "shell:docker exec agent_redis redis-cli --no-auth-warning keys 'agents:*'",
        ["redis", "streams", "events", "agents"],
    ),
    (
        "show-agent-status",
        "Show the current status (idle/busy) of all agents from Redis. "
        "Example: CMD: docker exec agent_redis redis-cli keys 'agent:status:*'",
        "shell:docker exec agent_redis redis-cli keys 'agent:status:*'",
        ["agents", "status", "redis", "monitoring"],
    ),
]


# ── Inline command extraction (no LLM) ───────────────────────────────────────
# If the task payload is already a concrete shell command, extract it directly.

_CMD_PREFIXES = tuple(
    sorted(
        {
            "ls",
            "cat",
            "find",
            "grep",
            "echo",
            "pwd",
            "wc",
            "head",
            "tail",
            "diff",
            "sort",
            "uniq",
            "stat",
            "env",
            "which",
            "docker",
            "docker-compose",
            "git",
            "pip",
            "pip3",
            "python",
            "python3",
            "pytest",
            "npm",
            "yarn",
            "apt",
            "apt-get",
            "tee",
            "cp",
            "mv",
            "rm",
            "chmod",
            "mkdir",
            "touch",
            "curl",
            "wget",
            "patch",
            "chown",
            "df",
            "du",
            "ps",
            "ss",
            "lsof",
            "netstat",
            "uname",
            "uptime",
            "free",
        },
        key=len,
        reverse=True,  # longer prefixes first to avoid shadowing
    )
)

_INLINE_PATTERN = re.compile(
    r"^(?:run|execute|exec(?:ute)?|please\s+run|can you\s+run)?\s*"
    r"((?:" + "|".join(re.escape(p) for p in _CMD_PREFIXES) + r")\s+\S.{0,500})$",
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
SAFE (instant, no log):
  ls, cat, grep, find, echo, stat, head, tail, diff, env, which
  df, du, ps, ss, lsof, netstat, uname, uptime, free
  pip list/show/freeze/check (read-only pip sub-commands)
AUTO_APPROVED (instant, audit log to #agent-logs):
  docker start/restart/stop/logs/exec/inspect
  tee, mkdir, touch, chmod, cp  (within /workspace only)
  python, python3, pytest, bash, sh
  git log/diff/status/show/fetch/pull
REQUIRES_APPROVAL (Discord gate):
  docker rm/rmi/prune  |  git push/commit/reset/rebase
  pip install/npm/yarn/apt  |  curl/wget
  rm, mv, patch, chown

Prefer AUTO_APPROVED commands — operate autonomously. Only escalate for truly
destructive or external operations.

## Multi-step execution (ReAct loop)
You may issue multiple commands to complete a task. After each CMD: line you will
receive an OBSERVATION: with the output. Use the observation to decide your next step.

Example:
  CMD: cat /workspace/src/agents/executor/main.py
  (receive OBSERVATION with file content)
  CMD: tee /workspace/src/agents/executor/main.py << 'EOF'
  <patched content>
  EOF
  (receive OBSERVATION confirming write)
  CMD: docker restart agent_executor
  DONE: Patched executor/main.py and restarted the container.

When the task is fully complete, respond with:
  DONE: <summary of what was done>

DONE: must always be the last line. Never put a CMD: after DONE:.
If only one command is needed, issue it followed by DONE: on the same response.
If no command is needed, respond directly — the loop treats any response without CMD: as done.

## Self-modification
The agent stack source is at /workspace/src/. Read, write, and restart autonomously.

Read a file:  CMD: cat /workspace/src/agents/executor/main.py
List agents:  CMD: ls /workspace/src/agents/
Write a file: CMD: tee /workspace/src/agents/executor/main.py << 'EOF'
              <content>
              EOF
Restart:      CMD: docker restart agent_executor

Container names: agent_orchestrator, agent_executor, agent_document_qa,
                 agent_code_search, agent_discord_bridge, agent_claude_code,
                 agent_developer

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

## developer agent (delegate via orchestrator)
The developer agent writes, edits, refactors, and fixes code:
- Implements features, fixes bugs, scaffolds new agents, writes tests, reviews code
- Works across /workspace/src (agent stack) and /workspace/projects (user projects)
- Delegates file writes back to executor (approval-gated) and searches to code_search
Route tasks that require *writing or modifying* source code to developer.
Container name: agent_developer
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
    "ls",
    "cat",
    "find",
    "grep",
    "echo",
    "pwd",
    "wc",
    "head",
    "tail",
    "diff",
    "sort",
    "uniq",
    "stat",
    "env",
    "printenv",
    "which",
    "type",
    "id",
    # Read-only system introspection
    "df",
    "du",
    "ps",
    "ss",
    "lsof",
    "netstat",
    "uname",
    "uptime",
    "free",
}

# Run automatically; emits AUDIT event so every action is visible in #agent-logs
AUTO_APPROVED_COMMANDS = {
    # Container lifecycle within the stack
    "docker",  # start/restart/stop/logs/exec/inspect (NOT rm/rmi)
    "docker-compose",
    # Workspace file writes
    "tee",  # write files inside /workspace — blocked for /workspace/user outside tests
    "mkdir",
    "touch",
    "chmod",
    "cp",
    # In-container code execution
    "python",
    "python3",
    "pytest",  # run test suite — bounded to container, no external effects
    # Non-destructive git operations
    "git",  # log/diff/status/show/fetch/pull — push/commit still blocked
    # Tool scripts
    "bash",
    "sh",
}

# Require explicit Discord approval — destructive, network-external, or irreversible
REQUIRES_APPROVAL = {
    # Package management (modifies the container image effectively)
    "pip",
    "pip3",
    "npm",
    "yarn",
    "apt",
    "apt-get",
    # External network calls
    "curl",
    "wget",
    # Destructive file operations (can delete user data)
    "rm",
    "mv",
    "patch",
    # Privileged container operations (permanent removal)
    "chown",
}

# pip sub-commands that are read-only — skip the approval gate for these
_PIP_SAFE_SUBCOMMANDS = {"list", "show", "freeze", "check", "inspect"}


class ExecutorAgent(BaseAgent):
    async def on_startup(self) -> None:
        log.info("executor.startup")
        for name, desc, inv, tags in _BUILTIN_TOOLS:
            await self.memory.register_tool(
                name, desc, "executor", inv, tags, "executor"
            )
        log.info("executor.tools_seeded", count=len(_BUILTIN_TOOLS))
        await self._seed_workspace_scripts()
        discovered = await self._scan_workspace_tools()
        log.info("executor.workspace_tools_scanned", discovered=discovered)

    async def _seed_workspace_scripts(self) -> None:
        """
        Copy seed tool scripts from /workspace/src/workspace/tools/ into
        /workspace/tools/ if they do not already exist there.

        This makes repo-committed scripts available on the named Docker volume
        on first start without needing a separate bind mount.
        """
        import shutil

        src_tools = Path("/workspace/src/workspace/tools")
        if not src_tools.exists():
            return
        TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        copied = 0
        for script in src_tools.glob("*.sh"):
            dest = TOOLS_DIR / script.name
            if not dest.exists():
                try:
                    shutil.copy2(script, dest)
                    dest.chmod(0o755)
                    copied += 1
                    log.info("executor.seed_script_copied", script=script.name)
                except Exception as exc:
                    log.warning(
                        "executor.seed_script_copy_failed",
                        script=script.name,
                        error=str(exc),
                    )
        if copied:
            log.info("executor.seed_scripts_installed", count=copied)

    async def _scan_workspace_tools(self) -> int:
        """
        Scan /workspace/tools/ for shell scripts and register each one in the
        shared tool registry.  Scripts are expected to carry a header like:

            #!/bin/bash
            # Description: What this script does (used as the registry description)
            # Usage: ./script-name.sh [arg1] [arg2]      (optional, appended to desc)
            # Tags: tag1, tag2, tag3                       (optional)

        Idempotent — register_tool() upserts on name, so re-scanning on restart
        refreshes descriptions without creating duplicates.

        Returns the number of scripts discovered and registered.
        """
        if not TOOLS_DIR.exists():
            return 0

        count = 0
        for script in sorted(TOOLS_DIR.glob("*.sh")):
            try:
                lines = script.read_text(errors="ignore").splitlines()
            except Exception as exc:
                log.warning(
                    "executor.scan_tool_read_error", path=str(script), error=str(exc)
                )
                continue

            description = ""
            usage = ""
            tags: list[str] = ["workspace-tool", "script"]

            for line in lines[:20]:  # only parse the header block
                stripped = line.strip()
                if stripped.startswith("# Description:"):
                    description = stripped[len("# Description:") :].strip()
                elif stripped.startswith("# Usage:"):
                    usage = stripped[len("# Usage:") :].strip()
                elif stripped.startswith("# Tags:"):
                    extra = [
                        t.strip()
                        for t in stripped[len("# Tags:") :].split(",")
                        if t.strip()
                    ]
                    tags.extend(extra)

            if not description:
                # Fall back to the script filename as a human-readable description
                description = script.stem.replace("-", " ").replace("_", " ")

            if usage:
                description = f"{description}. Usage: {usage}"

            # Tool name = script stem (e.g. check-agent-logs → check-agent-logs)
            tool_name = script.stem[:60]
            invocation = f"shell:{script}"

            await self.memory.register_tool(
                tool_name, description, "executor", invocation, tags, "workspace-scan"
            )
            log.debug(
                "executor.workspace_tool_registered", name=tool_name, script=script.name
            )
            count += 1

        return count

    async def on_plan_proposed(
        self, plan_id: str, steps: list[dict], payload: dict, request_clarification=None
    ) -> tuple[bool | None, str, float]:
        """
        Use the LLM to evaluate executor steps. Ask for clarification if any
        step is ambiguous before casting a vote.
        """
        my_steps = [s for s in steps if s.get("agent") == "executor"]
        if not my_steps:
            return None, "", 0.0  # abstain — not involved in this plan

        # Hard safety check first (no LLM needed).
        _DANGEROUS = ("rm -rf /", "mkfs", "dd if=", ":(){:|:&};:", "chmod 777 /")
        for step in my_steps:
            task_text = step.get("task", "").lower()
            for pattern in _DANGEROUS:
                if pattern in task_text:
                    return False, f"step contains potentially destructive command: {pattern!r}", 0.9

        steps_txt = "\n".join(
            f"  Phase {s.get('phase', 1)}: {s.get('task', '')}" for s in my_steps
        )
        original_task = payload.get("original_task", "")

        try:
            response = await self.llm_invoke([
                SystemMessage(content=(
                    "You are the executor agent. You run shell commands and manage files. "
                    "You are reviewing steps assigned to you in a proposed execution plan. "
                    "Decide if you can execute them safely and whether you understand them fully.\n\n"
                    "Reply in exactly this format:\n"
                    "UNDERSTOOD: yes/no\n"
                    "CLARIFICATION_NEEDED: <one focused question if UNDERSTOOD=no, else 'none'>\n"
                    "APPROVE: yes/no\n"
                    "REASON: <one sentence>"
                )),
                HumanMessage(content=(
                    f"Overall task: {original_task}\n"
                    f"My steps:\n{steps_txt}"
                )),
            ])
            lines = {
                k.strip(): v.strip()
                for line in response.content.splitlines()
                if ":" in line
                for k, v in [line.split(":", 1)]
            }
        except Exception as exc:
            log.warning("executor.vote_llm_error", error=str(exc))
            return True, "could not evaluate — defaulting to approve", 0.5

        understood = lines.get("UNDERSTOOD", "yes").lower() == "yes"
        clarification_q = lines.get("CLARIFICATION_NEEDED", "none")
        approve_str = lines.get("APPROVE", "yes").lower()
        reason = lines.get("REASON", "")

        # Ask for clarification if the LLM flagged something as unclear.
        if not understood and clarification_q and clarification_q.lower() != "none" and request_clarification:
            answer = await request_clarification(clarification_q)
            if answer:
                # Re-evaluate with the clarification folded in.
                try:
                    response2 = await self.llm_invoke([
                        SystemMessage(content=(
                            "You are the executor agent re-evaluating a plan step after receiving clarification."
                            " Reply in exactly this format:\n"
                            "APPROVE: yes/no\n"
                            "REASON: <one sentence>"
                        )),
                        HumanMessage(content=(
                            f"Overall task: {original_task}\n"
                            f"My steps:\n{steps_txt}\n\n"
                            f"Clarification received: {answer}"
                        )),
                    ])
                    lines2 = {
                        k.strip(): v.strip()
                        for line in response2.content.splitlines()
                        if ":" in line
                        for k, v in [line.split(":", 1)]
                    }
                    approve_str = lines2.get("APPROVE", approve_str).lower()
                    reason = lines2.get("REASON", reason)
                except Exception:
                    pass  # keep original decision

        approve = approve_str == "yes"
        confidence = 0.85 if approve else 0.8
        return approve, reason[:200], confidence

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
                result = await self._run_command(
                    f"bash {local_tool}", task, task_id, timeout=timeout
                )
            else:
                # ── 3. Search shared tool registry ────────────────────────
                tool_hits = await self.search_tools(task)
                shell_hit = next(
                    (
                        t
                        for t in tool_hits
                        if t["invocation"].startswith("shell:")
                        and t.get("similarity", 0) >= 0.82
                        and "{" not in t["invocation"]  # skip parameterised templates
                    ),
                    None,
                )
                if shell_hit:
                    cmd = shell_hit["invocation"][6:]  # strip "shell:" prefix
                    log.info(
                        "executor.tool_registry_hit",
                        tool=shell_hit["name"],
                        sim=round(shell_hit.get("similarity", 0), 3),
                    )
                    await self.memory.increment_tool_usage(shell_hit["name"])
                    result = await self._run_command(
                        cmd, task, task_id, timeout=timeout
                    )
                else:
                    # ── 4. Try capability registry (no LLM) ───────────────────
                    cached_cmd = await self.memory.lookup_capability(task)
                    if cached_cmd:
                        log.info(
                            "executor.capability_hit",
                            task=task[:80],
                            cmd=cached_cmd[:80],
                        )
                        result = await self._run_command(
                            cached_cmd, task, task_id, timeout=timeout
                        )
                    else:
                        # ── 5. Fall back to LLM with multi-step ReAct loop ────
                        tools_ctx = self.format_tools_context(tool_hits)
                        messages = [
                            SystemMessage(content=SYSTEM_PROMPT + tools_ctx),
                            HumanMessage(content=f"Task: {task}"),
                        ]

                        async def _exec_action(action_type: str, payload: str) -> str:
                            if action_type == "CMD":
                                return await self._run_command(
                                    payload, task, task_id, timeout=timeout
                                )
                            return f"Unknown action: {action_type}"

                        result = await self.agent_loop(
                            messages,
                            action_handler=_exec_action,
                            max_steps=5,
                        )

                        # Store successful result in capability cache for future reuse
                        if (
                            result
                            and not result.startswith("Command denied")
                            and not result.startswith("Error")
                        ):
                            tool_tags = ["executor", "multi-step"]
                            await self.memory.store_capability(
                                task, result[:200], tool_tags
                            )
                            words = [
                                w
                                for w in re.sub(
                                    r"[^a-z0-9\s]", "", task.lower()
                                ).split()
                                if len(w) > 2
                            ][:4]
                            tool_name = "-".join(words)[:50]
                            if tool_name:
                                await self.memory.register_tool(
                                    tool_name,
                                    task[:120],
                                    "executor",
                                    f"shell:{result[:200]}",
                                    tool_tags,
                                    self.role,
                                )

        await self.emit(
            EventType.AGENT_TOOL_RESULT,
            payload={"result": result[:2000], "task_id": task_id},
        )

        # Stage findings for long-term memory
        if any(
            kw in task.lower()
            for kw in ["bug", "error", "pattern", "found", "discovered", "improve"]
        ):
            self.stage_finding(
                content=f"Executor task: {task}\nResult: {result[:500]}",
                topic="execution_findings",
                tags=["executor", "tool"],
            )

        # Mark self-modifications in memory immediately
        if any(
            kw in task.lower()
            for kw in ["modify", "update", "patch", "rewrite", "fix code"]
        ):
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
        task_words = set(re.sub(r"[^a-z0-9\s]", "", task.lower()).split()) - {
            "the",
            "a",
            "an",
            "in",
            "of",
            "for",
            "to",
        }
        best_path: str | None = None
        best_score = 0
        for script in TOOLS_DIR.glob("*.sh"):
            name_words = set(
                re.sub(
                    r"[^a-z0-9\s]", " ", script.stem.replace("-", " ").replace("_", " ")
                ).split()
            )
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
            words = [
                w
                for w in re.sub(r"[^a-z0-9\s]", "", task.lower()).split()
                if len(w) > 2
            ][:5]
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
                "python": "python3",
                "python3": "python",
                "pip": "pip3",
                "pip3": "pip",
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
        re.compile(
            r"\bdocker\s+(rm|rmi|volume\s+rm|network\s+rm|system\s+prune)\b", re.I
        ),
        re.compile(
            r"\bgit\s+(push|commit|reset|rebase|merge|cherry-pick|force)\b", re.I
        ),
        re.compile(
            r"\btee\b.*/workspace/(user|projects)/", re.I
        ),  # outside container workspace
    ]

    def _needs_escalation(self, cmd: str) -> bool:
        return any(p.search(cmd) for p in self._ESCALATE_PATTERNS)

    async def _run_command(
        self,
        cmd: str,
        task: str,
        task_id: str,
        timeout: int = 60,
        _recovery: bool = False,
    ) -> str:
        """Run a shell command after three-tier trust check."""
        try:
            parts = shlex.split(cmd)
        except ValueError as e:
            return f"Error parsing command: {e}"

        base_cmd = parts[0] if parts else ""

        # pip list/show/freeze/check are read-only — treat as AUTO_APPROVED
        pip_read_only = (
            base_cmd in ("pip", "pip3")
            and len(parts) >= 2
            and parts[1] in _PIP_SAFE_SUBCOMMANDS
        )

        if not pip_read_only and (
            base_cmd in REQUIRES_APPROVAL
            or (base_cmd in AUTO_APPROVED_COMMANDS and self._needs_escalation(cmd))
        ):
            approved = await self._request_approval(cmd, task, task_id)
            if not approved:
                log.info("executor.command_denied", cmd=cmd)
                return f"Command denied by user: `{cmd}`"
        elif pip_read_only or base_cmd in AUTO_APPROVED_COMMANDS:
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
                    log.info(
                        "executor.auto_recovery",
                        original=cmd,
                        recovery=alt_cmd,
                        reason=reason,
                    )
                    return await self._run_command(
                        alt_cmd, task, task_id, timeout, _recovery=True
                    )

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
        log.info(
            "executor.approval_decision", decision=decision, approval_id=approval_id
        )
        return decision == "approved"

    async def on_shutdown(self) -> None:
        log.info("executor.shutdown")


if __name__ == "__main__":
    settings = Settings()
    agent = ExecutorAgent(settings)
    run_agent(agent)
