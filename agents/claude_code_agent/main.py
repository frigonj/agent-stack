"""
agents/claude_code_agent/main.py
─────────────────────────────────
Claude API-powered agent for interactive environment management.

Subscribes to agents:claude_code stream, runs an agentic tool-use loop
using the Anthropic API, and emits results to agents:broadcast.

Tools available to Claude:
  read_file(path)                            — read file contents
  write_file(path, content)                  — write / create a file
  list_dir(path)                             — list directory entries
  run_shell(command, timeout_seconds)        — run a shell command
  search_files(pattern, directory)           — glob file search
  search_content(pattern, path, file_glob)   — ripgrep-style content search
"""

from __future__ import annotations

import asyncio
import glob as globlib
import json
import os
import time
import uuid
from pathlib import Path

import anthropic
import redis.asyncio as aioredis
import structlog

log = structlog.get_logger()

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "1024"))
STREAM = "agents:claude_code"
GROUP = "claude_code_group"
CONSUMER = "claude_code_agent"
BROADCAST = "agents:broadcast"

# Hard caps on tool output returned to Claude.
# The account rate limit is 10,000 input tokens/minute.
# Tool definitions + system prompt + history already consume ~800 tokens,
# so individual tool outputs must stay small to avoid 429s.
_MAX_FILE_CHARS = 6_000  # ~1,500 tokens
_MAX_SHELL_CHARS = 4_000  # ~1,000 tokens
_MAX_SEARCH_CHARS = 3_000  # ~750 tokens
_MAX_GLOB_RESULTS = 50

# Maximum tool-use iterations per task. Each iteration re-sends the full
# conversation history, so cost compounds quickly. Keep this low.
_MAX_ITERATIONS = 8

SYSTEM_PROMPT = """\
You are Claude Code Agent — an AI assistant with direct access to the agent stack environment.

TOKEN EFFICIENCY IS CRITICAL. The account has a strict 10,000 input token/minute rate limit.
Follow these rules on every response:
- Read only the specific lines you need. Never read a file speculatively.
- Use search_content to locate relevant sections before reading whole files.
- Write responses as short as possible. No preamble, no summaries of what you did.
- Prefer targeted shell one-liners (grep, head, tail) over reading whole files.
- Do not repeat tool output in your text reply — just act on it.

Workspace mounts (Docker):
  /workspace/src      — agent stack source (read-write)
  /workspace/user     — user's Windows home
  /workspace/projects — user's coding projects (read-write)
  /workspace/docs     — documents

When editing files: search → read only affected section → write minimal change → one-line summary.\
"""

TOOLS: list[dict] = [
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or /workspace/src-relative path",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file, creating parent directories as needed",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "List files and subdirectories at a path",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "run_shell",
        "description": "Run a shell command and return combined stdout/stderr",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30)",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "search_files",
        "description": "Find files matching a glob pattern",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, e.g. '**/*.py'",
                },
                "directory": {
                    "type": "string",
                    "description": "Base directory (default /workspace/src)",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "search_content",
        "description": "Search file contents for a pattern (grep-style)",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex or literal string"},
                "path": {
                    "type": "string",
                    "description": "Directory or file to search (default /workspace/src)",
                },
                "file_glob": {
                    "type": "string",
                    "description": "File pattern filter, e.g. '*.py'",
                },
            },
            "required": ["pattern"],
        },
    },
]


# ── Tool implementations ──────────────────────────────────────────────────────


def _resolve(path: str) -> str:
    """Resolve workspace-relative paths to absolute."""
    if not path.startswith("/"):
        return f"/workspace/src/{path}"
    return path


async def _read_file(path: str) -> str:
    path = _resolve(path)
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        if len(content) > _MAX_FILE_CHARS:
            content = (
                content[:_MAX_FILE_CHARS]
                + f"\n... (truncated — {len(content)} chars total; use search_content to find specific sections)"
            )
        return content
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except Exception as e:
        return f"Error reading {path}: {e}"


async def _write_file(path: str, content: str) -> str:
    path = _resolve(path)
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


async def _list_dir(path: str) -> str:
    path = _resolve(path)
    try:
        entries = []
        for entry in sorted(os.scandir(path), key=lambda e: (e.is_file(), e.name)):
            entries.append(("  " if entry.is_file() else "📁 ") + entry.name)
        return "\n".join(entries) if entries else "(empty)"
    except FileNotFoundError:
        return f"Error: directory not found: {path}"
    except Exception as e:
        return f"Error listing {path}: {e}"


async def _run_shell(command: str, timeout_seconds: int = 30) -> str:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            proc.kill()
            return f"Error: command timed out after {timeout_seconds}s"
        output = stdout.decode("utf-8", errors="replace")
        if len(output) > _MAX_SHELL_CHARS:
            output = (
                output[:_MAX_SHELL_CHARS]
                + "\n... (truncated — use tail/grep to see more)"
            )
        return output or "(no output)"
    except Exception as e:
        return f"Error: {e}"


async def _search_files(pattern: str, directory: str = "/workspace/src") -> str:
    if not directory.startswith("/"):
        directory = _resolve(directory)
    try:
        matches = globlib.glob(os.path.join(directory, pattern), recursive=True)
        if not matches:
            return "No files found"
        results = sorted(matches)
        out = "\n".join(results[:_MAX_GLOB_RESULTS])
        if len(results) > _MAX_GLOB_RESULTS:
            out += (
                f"\n... ({len(results) - _MAX_GLOB_RESULTS} more — refine your pattern)"
            )
        return out
    except Exception as e:
        return f"Error: {e}"


async def _search_content(
    pattern: str,
    path: str = "/workspace/src",
    file_glob: str | None = None,
) -> str:
    if not path.startswith("/"):
        path = _resolve(path)
    cmd = (
        ["grep", "-rn", "--include=" + file_glob, pattern, path]
        if file_glob
        else ["grep", "-rn", pattern, path]
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        output = stdout.decode("utf-8", errors="replace")
        if len(output) > _MAX_SEARCH_CHARS:
            output = (
                output[:_MAX_SEARCH_CHARS]
                + "\n... (truncated — narrow pattern or use file_glob)"
            )
        return output or "No matches found"
    except asyncio.TimeoutError:
        return "Search timed out"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS: dict = {
    "read_file": lambda i: _read_file(i["path"]),
    "write_file": lambda i: _write_file(i["path"], i["content"]),
    "list_dir": lambda i: _list_dir(i["path"]),
    "run_shell": lambda i: _run_shell(i["command"], i.get("timeout_seconds", 30)),
    "search_files": lambda i: _search_files(
        i["pattern"], i.get("directory", "/workspace/src")
    ),
    "search_content": lambda i: _search_content(
        i["pattern"], i.get("path", "/workspace/src"), i.get("file_glob")
    ),
}


# ── Agentic loop ──────────────────────────────────────────────────────────────


async def _api_call_with_retry(
    client: anthropic.AsyncAnthropic, **kwargs
) -> anthropic.types.Message:
    """Call client.messages.create with exponential backoff on 429."""
    delay = 10  # seconds — start conservatively given the tight TPM window
    for attempt in range(5):
        try:
            return await client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            if attempt == 4:
                raise
            log.warning(
                "claude_code_agent.rate_limited",
                attempt=attempt + 1,
                retry_in=delay,
                error=str(exc),
            )
            await asyncio.sleep(delay)
            delay *= 2  # 10 → 20 → 40 → 80s


async def run_task(client: anthropic.AsyncAnthropic, task: str) -> str:
    """Drive a tool-use loop until Claude reaches end_turn or _MAX_ITERATIONS."""
    messages: list[dict] = [{"role": "user", "content": task}]

    for _iteration in range(_MAX_ITERATIONS):
        response = await _api_call_with_retry(
            client,
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "type") and block.type == "text":
                    return block.text
            return "(task completed)"

        if response.stop_reason != "tool_use":
            break

        # Execute all tool calls in this response
        tool_results = []
        for block in response.content:
            if not (hasattr(block, "type") and block.type == "tool_use"):
                continue
            handler = TOOL_HANDLERS.get(block.name)
            if handler:
                result = await handler(block.input)
            else:
                result = f"Unknown tool: {block.name}"
            log.info(
                "claude_code_agent.tool", tool=block.name, result_len=len(str(result))
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return "(task completed — max iterations reached)"


# ── Event handling ────────────────────────────────────────────────────────────


async def _handle_event(
    redis: aioredis.Redis,
    client: anthropic.AsyncAnthropic,
    data: dict,
) -> None:
    if data.get("type") not in (
        "task.created",
        "task.assigned",
        "claude.task",
        "claude.escalation",
    ):
        return

    payload = json.loads(data.get("payload", "{}"))
    task = payload.get("task", "").strip()
    task_id = data.get("task_id", str(uuid.uuid4()))
    parent_task_id = payload.get("parent_task_id", "")
    subtask_id = payload.get("subtask_id", "")

    if not task:
        return

    log.info("claude_code_agent.task_start", task=task[:80], task_id=task_id)

    result = await run_task(client, task)

    log.info("claude_code_agent.task_done", task_id=task_id, result_len=len(result))

    await redis.xadd(
        BROADCAST,
        {
            "event_id": str(uuid.uuid4()),
            "type": "task.completed",
            "source": "claude_code_agent",
            "task_id": parent_task_id or task_id,
            "timestamp": str(time.time()),
            "payload": json.dumps(
                {
                    "task_id": parent_task_id or task_id,
                    "subtask_id": subtask_id,
                    "parent_task_id": parent_task_id,
                    "result": result,
                }
            ),
        },
        maxlen=10_000,
        approximate=True,
    )


# ── Main loop ─────────────────────────────────────────────────────────────────


async def main_loop() -> None:
    redis_url = os.environ["REDIS_URL"]
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    redis = await aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    client = anthropic.AsyncAnthropic(api_key=api_key)

    try:
        await redis.xgroup_create(STREAM, GROUP, id="$", mkstream=True)
    except aioredis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise

    log.info("claude_code_agent.ready", model=MODEL, stream=STREAM)

    while True:
        try:
            results = await redis.xreadgroup(
                groupname=GROUP,
                consumername=CONSUMER,
                streams={STREAM: ">"},
                count=1,
                block=1000,
            )
            if not results:
                continue
            for _stream_key, messages in results:
                for entry_id, data in messages:
                    try:
                        await _handle_event(redis, client, data)
                    except Exception as exc:
                        log.error("claude_code_agent.event_error", error=str(exc))
                    finally:
                        await redis.xack(STREAM, GROUP, entry_id)

        except aioredis.ResponseError as exc:
            if "NOGROUP" in str(exc):
                log.warning("claude_code_agent.rebuilding_group")
                try:
                    await redis.xgroup_create(STREAM, GROUP, id="$", mkstream=True)
                except aioredis.ResponseError:
                    pass
            else:
                log.error("claude_code_agent.redis_error", error=str(exc))
                await asyncio.sleep(3)
        except Exception as exc:
            log.error("claude_code_agent.loop_error", error=str(exc))
            await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main_loop())
