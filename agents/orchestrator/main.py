"""
agents/orchestrator/main.py
────────────────────────────
Orchestrator agent — plans tasks in phases, executes, validates each result,
retries failed phases up to MAX_RETRIES times, then escalates to the user.

Loop per task:
  1. Build ExecutionPlan  (phases → sequential, steps within a phase → parallel)
  2. Dispatch phase 1
  3. On result: validate → mark done/failed
  4. If phase complete:
       - All passed  → advance to next phase (or finalise)
       - Any failed  → retry phase with error context (up to MAX_RETRIES)
       - Retries exhausted → escalate to user with full error summary
  5. Status updates emitted as plan.status events → Discord per-agent channels
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from core.base_agent import BaseAgent, run_agent
from core.config import Settings
from core.events.bus import Event, EventType
from core.context import truncate_task, truncate_context, truncate_result

log = structlog.get_logger()

MAX_RETRIES = 3   # max retry attempts per phase before escalating to user

# ── Architecture change watcher ───────────────────────────────────────────────
# Checked every think_interval (900 s). When any file hash changes the
# orchestrator spins up document_qa to rebuild + upload docs.

_ARCH_HASH_KEY           = "doc:arch_file_hashes"
_ARCH_LAST_FULL_BUILD    = "doc:arch_last_full_build"   # Unix timestamp (str) in Redis
_ARCH_CHANGELOG_PATH     = Path("/workspace/docs/generated/CHANGELOG.md")
# How often a full LLM+LaTeX build is allowed (seconds). Configurable via env.
_FULL_BUILD_INTERVAL     = int(os.getenv("ARCH_FULL_BUILD_INTERVAL", str(86400)))

WATCHED_PATHS: list[str] = [
    "/workspace/src/docker-compose.yml",
    "/workspace/src/core/base_agent.py",
    "/workspace/src/core/events/bus.py",
    "/workspace/src/core/config.py",
    "/workspace/src/core/memory/long_term.py",
    "/workspace/src/core/context.py",
    "/workspace/src/core/errors.py",
    "/workspace/src/agents/orchestrator/main.py",
    "/workspace/src/agents/executor/main.py",
    "/workspace/src/agents/document_qa/main.py",
    "/workspace/src/agents/code_search/main.py",
    "/workspace/src/agents/research/main.py",
    "/workspace/src/agents/discord_bridge/main.py",
    "/workspace/src/agents/claude_code_agent/main.py",
    "/workspace/src/docker/agent.Dockerfile",
    "/workspace/src/docker/document_qa.Dockerfile",
    "/workspace/src/docker/claude_code.Dockerfile",
]


# ── Keyword router ────────────────────────────────────────────────────────────

_KEYWORD_ROUTES: list[tuple[re.Pattern, str]] = [
    # Discord — most-specific first
    (re.compile(r'\b(duplicate|same\s+name|identical)\b.{0,40}\bchannels?\b',        re.I), 'discord'),
    (re.compile(r'\bchannels?\b.{0,40}\b(duplicate|same\s+name|identical)\b',        re.I), 'discord'),
    (re.compile(r'\b(list|show|display)\b.{0,20}\bchannels?\b',                       re.I), 'discord'),
    (re.compile(r'\bcreate\s+(?:a\s+)?(?:text\s+)?channels?\b',                       re.I), 'discord'),
    (re.compile(r'\bcreate\s+(?:a\s+)?categor\w+',                                    re.I), 'discord'),
    (re.compile(r'\b(delete|remove|clean\s+up)\b.{0,30}\bchannels?\b',                re.I), 'discord'),
    (re.compile(r'\brename\b.{0,30}\bchannels?\b|\brename\s+#',                       re.I), 'discord'),
    (re.compile(r'\bset\b.{0,15}\btopic\b',                                           re.I), 'discord'),
    (re.compile(r'\b(send|post)\b.{0,20}\b(message|in #|to #)',                       re.I), 'discord'),
    (re.compile(r'\bpin\s+(message|msg)\b',                                           re.I), 'discord'),
    (re.compile(r'\b(set\s*up|organis[ez]|orchestrate|configure)\b.{0,30}\bdiscord\b',re.I), 'discord'),
    (re.compile(r'\bdiscord\b.{0,30}\b(channel|category|message|server)\b',          re.I), 'discord'),
    # Executor — shell / file / container
    (re.compile(r'\b(cat|ls|grep|find|diff|stat|wc|head|tail)\s+[/\w]',              re.I), 'executor'),
    (re.compile(r'\b(docker|pip|pip3|git|python3?|npm|yarn|apt)\s',                   re.I), 'executor'),
    (re.compile(r'\b(run|execute|exec)\b.{0,15}\b(command|script|cmd)\b',             re.I), 'executor'),
    (re.compile(r'\brestart\s+(the\s+)?(agent|container|service|orchestrator|executor)',re.I), 'executor'),
    (re.compile(r'\b(read|write|edit|modify|update)\b.{0,20}\bfile\b',                re.I), 'executor'),
    (re.compile(r'\b(install|uninstall|upgrade)\b.{0,20}\b(package|library|module)\b',re.I), 'executor'),
    # Document QA — Q&A, PDF, architecture review, LaTeX generation
    (re.compile(r'\b(summaris[ez]|what does|tell me about)\b.{0,30}\bdoc',            re.I), 'document_qa'),
    (re.compile(r'\bin (the |my )?(docs?|document|pdf)\b',                            re.I), 'document_qa'),
    (re.compile(r'\b(read|extract|parse|open)\b.{0,15}\bpdf\b',                       re.I), 'document_qa'),
    (re.compile(r'\b(generate|create|write|produce)\b.{0,30}\b(latex|pdf\s+report|doc(?:ument)?|report)\b', re.I), 'document_qa'),
    (re.compile(r'\b(architecture|arch\s*doc|document\s+the\s+stack|review.{0,20}(source|stack|agents?))\b', re.I), 'document_qa'),
    (re.compile(r'\b(compile\s+latex|latexmk|tex\s+file|\.tex\b)',                    re.I), 'document_qa'),
    (re.compile(r'\b(system\s+design|component\s+diagram|stack\s+overview)\b',        re.I), 'document_qa'),
    # Code search
    (re.compile(r'\b(find|search|where is|look for)\b.{0,20}\b(function|class|method)',re.I), 'code_search'),
    (re.compile(r'\bin (the |this )?(repo|codebase|source code)\b',                   re.I), 'code_search'),
    # Research agent — internet lookups, current info, multi-source fact gathering
    (re.compile(r'\b(research|look up|look\s+up|find\s+out|what is|who is|when did|where is)\b.{0,40}\b(current|latest|recent|now|today|price|news|update)\b', re.I), 'research'),
    (re.compile(r'\b(search the (web|internet|net)|google|bing|look online)\b',        re.I), 'research'),
    (re.compile(r'\b(latest|current|recent|up[\s-]to[\s-]date)\b.{0,40}\b(version|release|news|status|price|info|docs)\b', re.I), 'research'),
    (re.compile(r'\bwhat (is|are|was|were) .{3,60}\??\s*$',                            re.I), 'research'),
    # Direct — never reached; kept as fallback sentinel only
    # Conversational classification is handled by _classify_intent() before routing.
]


# ── Intent classification patterns ───────────────────────────────────────────
# Tier-1 fast path: no LLM needed for these.

_CHAT_RE = re.compile(
    r'^('
    r'hi|hello|hey|good\s+(morning|evening|afternoon|night)|'
    r'thanks?|thank you|cheers|np|no prob\w*|'
    r'ok|okay|got it|sounds good|cool|great|perfect|nice|sure|'
    r'what can you (do|help)|your capabilities|what are you|how are you|'
    r'are you (there|running|up|alive|ready)|'
    r'status|ping|'
    r'what (agents?|tools?|services?) (are|do you have|exist)|'
    r'how does .{3,80} work\??|'
    r'explain .{3,80}|'
    r'what (is|was|happened to|did you do)\b'
    r')',
    re.I,
)

# Vague imperatives that need clarification — only fire when there is no
# conversation context that could resolve the reference.
_CLARIFY_RE = re.compile(
    r'^(fix|update|change|modify|improve|adjust|tweak|clean\s*up|refactor|'
    r'make\s+it|make\s+that|do\s+(it|that|this)|handle\s+(it|that)|'
    r'sort\s+(it|that)\s+out)\b.{0,50}$',
    re.I,
)


# ── Seed intent examples ──────────────────────────────────────────────────────
# Canonical labelled examples stored to PostgreSQL on first startup.
# Cover patterns that fall through the fast-path regexes above.
# Format: (text, intent)  — intent is "task", "chat", or "clarify"
_SEED_INTENTS: list[tuple[str, str]] = [
    # ── task ──────────────────────────────────────────────────────────────────
    ("list all running docker containers",                                  "task"),
    ("show me the contents of config/.env.example",                        "task"),
    ("search the codebase for the EventBus class",                         "task"),
    ("find all files that import from core.events.bus",                    "task"),
    ("run pytest tests/unit and show me the results",                      "task"),
    ("restart the executor container",                                     "task"),
    ("write a bash script that monitors redis queue depth",                 "task"),
    ("create a new discord channel called agent-logs",                     "task"),
    ("grep for TODO comments across the codebase",                         "task"),
    ("check git log for the last 10 commits",                              "task"),
    ("install the redis python package",                                   "task"),
    ("generate a PDF report of the agent architecture",                    "task"),
    ("back up the postgres database to /workspace/backups",                "task"),
    ("search online for the latest langchain release notes",               "task"),
    ("what is the current version of python in the executor container",    "task"),
    ("read the docker-compose.yml and summarise the services",             "task"),
    ("build the orchestrator docker image",                                "task"),
    ("show disk usage in the workspace directory",                         "task"),
    ("tail the last 50 lines of the orchestrator logs",                    "task"),
    ("pin the last message in the announcements channel",                  "task"),
    # ── chat ──────────────────────────────────────────────────────────────────
    ("what did you just do",                                               "chat"),
    ("remind me what this project does",                                   "chat"),
    ("how does the event bus work",                                        "chat"),
    ("what is the difference between the executor and the orchestrator",   "chat"),
    ("good morning",                                                       "chat"),
    ("nice work",                                                          "chat"),
    ("that worked perfectly thanks",                                       "chat"),
    ("can you walk me through what just happened",                         "chat"),
    ("what agents are currently available",                                "chat"),
    ("which agent handles shell commands",                                 "chat"),
    # ── clarify ───────────────────────────────────────────────────────────────
    ("update the config",                                                  "clarify"),
    ("make it faster",                                                     "clarify"),
    ("run the thing",                                                      "clarify"),
    ("clean it up",                                                        "clarify"),
    ("do that again but better",                                           "clarify"),
    ("fix the bug",                                                        "clarify"),
    ("add more logging",                                                   "clarify"),
    ("deploy it",                                                          "clarify"),
]


def _route_by_keyword(task: str) -> list[dict] | None:
    tl = task.lower().strip()
    for pattern, agent in _KEYWORD_ROUTES:
        if pattern.search(tl):
            log.debug("orchestrator.keyword_routed", agent=agent, pattern=pattern.pattern[:40])
            return [{"task": task, "agent": agent, "phase": 1, "expected": ""}]
    return None


# ── Discord action parser ─────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r'[^a-z0-9\-_ ]', '', name)
    name = re.sub(r'\s+', '-', name)
    name = re.sub(r'-{2,}', '-', name).strip('-')
    return name[:100]


def _parse_discord_actions(task: str) -> list[dict] | None:
    t  = task.strip()
    tl = t.lower()
    actions: list[dict] = []

    if re.search(r'\b(duplicate|same\s+name|identical)\b.{0,60}\bchannels?\b', tl) or \
       re.search(r'\bchannels?\b.{0,60}\b(duplicate|same\s+name|identical)\b', tl):
        return [{"action": "find_and_delete_duplicates"}]

    if re.search(r'\b(list|show|display)\b.{0,20}\bchannels?\b', tl):
        return [{"action": "list_channels"}]

    m = re.search(
        r'\bcreate\s+(?:a\s+)?categor\w+\s+(?:called|named)?\s*["\']?([^\'"#,\n]{2,60})["\']?', tl)
    if m:
        actions.append({"action": "create_category", "name": m.group(1).strip()})

    m = re.search(r'\bcreate\s+(?:a\s+)?(?:text\s+)?channels?\s+(?:called|named)?\s*(.+)', tl)
    if m:
        names_raw = re.split(r'[,]|\band\b', m.group(1))
        for raw in names_raw:
            raw = re.sub(r'\bwith\s+(topic|description)\b.*', '', raw).strip()
            name = _slugify(raw)
            if not name:
                continue
            entry: dict = {"action": "create_channel", "name": name}
            topic_m = re.search(r'\bwith\s+topic\s+["\']?([^"\']{2,120})["\']?', tl)
            if topic_m:
                entry["topic"] = topic_m.group(1).strip()
            cat_m = re.search(r'\b(?:in|under)\s+(?:category\s+)?["\']?([^"\'#,\n]{2,60})["\']?', tl)
            if cat_m:
                entry["category"] = cat_m.group(1).strip()
            actions.append(entry)

    m = re.search(r'\b(?:delete|remove)\s+(?:the\s+)?(?:channel\s+)?#?([a-z0-9_\-]{2,60})', tl)
    if m and re.search(r'\bchannel\b', tl):
        actions.append({"action": "delete_channel", "channel_name": m.group(1).strip()})

    m = re.search(
        r'\brename\s+(?:channel\s+|the\s+)?#?([a-z0-9_\-]{2,60})\s+to\s+#?([a-z0-9_\-]{2,60})', tl)
    if m:
        actions.append({
            "action": "rename_channel",
            "channel_name": m.group(1).strip(),
            "name": _slugify(m.group(2)),
        })

    m = re.search(
        r'\bset\s+(?:the\s+)?topic\s+(?:of\s+|for\s+)?#?([a-z0-9_\-]{2,60})\s+to\s+["\']?(.{2,200}?)["\']?\s*$', tl)
    if m:
        actions.append({
            "action": "set_topic",
            "channel_name": m.group(1).strip(),
            "topic": m.group(2).strip(),
        })

    m = re.search(
        r'\b(?:send|post)\s+(?:a\s+)?(?:message\s+)?["\']([^"\']{1,1800})["\']?\s+(?:to|in)\s+#?([a-z0-9_\-]{2,60})', tl)
    if m:
        actions.append({
            "action": "send_message",
            "content": m.group(1).strip(),
            "channel_name": m.group(2).strip(),
        })

    m = re.search(r'\bpin\s+(?:message\s+)?(\d+)\s+(?:in|from)\s+#?([a-z0-9_\-]{2,60})', tl)
    if m:
        actions.append({
            "action": "pin_message",
            "message_id": m.group(1),
            "channel_name": m.group(2).strip(),
        })

    return actions if actions else None


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the technical project manager of an AI agent stack.

## Your three modes
- CHAT    — answer questions, give status, explain how things work. No task needed.
- PLAN    — break actionable work into phases, delegate to specialists, validate results.
- CLARIFY — ask one focused question when the request is too vague to act on safely.

Never create a plan for conversational messages. When in doubt, respond directly.
Always resolve pronouns and references ("the tool", "it", "that") from conversation context before deciding to clarify.

## Specialists
- executor          : shell commands, file R/W, docker operations, running Python/bash scripts,
                      web fetches (curl/wget), package installs. Paths available:
                      /workspace/src (agent source — read/write), /workspace/user (user home),
                      /workspace/projects (coding projects), /workspace/tools (reusable scripts).
                      Safe without approval: ls, cat, grep, find, curl, python3 -c.
                      Requires approval: git push/commit, docker restart/rm, tee (file writes), pip install.
- document_qa       : answer questions from documents in /workspace/docs (PDF, markdown, text);
                      review agent-stack source architecture at /workspace/src;
                      generate LaTeX documents and compile them to PDF via latexmk;
                      output goes to /workspace/docs/generated/.
- code_search       : search and explain code in /workspace/repos (function definitions, usages, patterns).
- research          : multi-source internet research via SearXNG (Google+Bing+DDG aggregated).
                      Decomposes questions into sub-queries, searches, extracts facts, checks consensus
                      across independent sources, then synthesises a cited answer. Zero Claude API calls.
                      Use for: "what is X", "latest version of Y", "current status of Z", web lookups.
- claude_code_agent : full Anthropic Claude API agent with tool use — best for complex multi-step reasoning,
                      code generation, analysis, or tasks requiring very deep intelligence.
                      Do NOT use for simple web lookups — use research agent instead.
- discord           : full Discord server management — channels, categories, topics, messages, pins.

## Stack architecture (for context)
- Redis Streams     : event bus between agents (streams: agents:{role}, agents:broadcast)
- Context streams   : per-task/chat source of truth (ctx:task:{id}:{slug}, ctx:chat:{id}:{slug})
                      All events for a single task or chat session flow through one stream.
                      Use self.bus.publish_to_context(context_id, event) to append.
                      Use self.bus.read_context_stream(context_id) to check history.
                      Use self.bus.list_active_contexts(context_type) to enumerate.
- PostgreSQL        : long-term memory via Emrys library (tables: knowledge, open_tasks,
                      active_plans, context_snapshots, topic_patterns)
- Emrys             : Python library wrapping the Postgres memory. Methods on self.memory:
                      store(content, topic, tags) — save a finding
                      search(query, semantic=True) — recall relevant knowledge
                      enqueue_task(task, priority) — add to the persistent work queue
                      save_context_snapshot / close_context_snapshot / search_context_snapshots
                      save_topic_pattern / search_topic_patterns — topic category learning
                      Full source: /workspace/emrys/ (README, src/, pyproject.toml)
- LM Studio         : local LLM inference at http://host.docker.internal:1234 (model: qwen2.5-14b)

## Voting
Before executing a plan, broadcast PLAN_PROPOSED. Specialist agents may reply with AGENT_VOTE
(approve=True/False, reason, confidence 0–1) within the vote_timeout_ms window (default 3 s,
configurable via self.bus.set_config). Agents may request more time via VOTE_EXTENSION_REQUESTED.
High-confidence rejections (confidence ≥ 0.7) trigger plan revision before execution.

## Context recall by user
Users can ask "recall <id>" or "show task <id>" to retrieve any past task, chat, or plan.
Call await self.memory.search_context_snapshots(query) to find by description, or
await self.memory.get_context_snapshot(context_id) for direct ID lookup.

## Topic classification
Agents learn keyword→category patterns over sessions (table: topic_patterns).
Call await self.classify_topic(text) → (category, confidence).
If confidence < 0.8 and enough sessions have run, ask the user for the correct label.

## Tool-building
Agents maintain reusable scripts in /workspace/tools/. Before planning new work, check
if an existing tool covers the need. When completed work produces a reusable pattern,
instruct executor to save it as a named script there.

## Self-modification
Source: /workspace/src/agents/<name>/main.py
1. cat the file  2. tee new content (approval)  3. docker restart agent_<name> (approval)
"""

# ── Plan data model ───────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    step_id: str
    phase: int        # phases are sequential; steps in the same phase run in parallel
    task: str
    agent: str
    expected: str     # brief description of what success looks like
    status: str = "pending"   # pending | running | done | failed
    result: str = ""
    depth: int = 0             # fix-attempt depth (0 = original, 1–10 = fix subtask)
    error_chain: list = field(default_factory=list)  # errors from prior fix attempts


@dataclass
class ExecutionPlan:
    plan_id: str
    task_id: str
    original_task: str
    steps: list[PlanStep]
    retry_count: int = 0
    discord_message_id: Optional[str] = None
    context: str = ""
    context_id: str = ""       # context stream ID (same as task_id for tasks)
    created_at: float = field(default_factory=time.time)
    # Phases currently being advanced — guards against concurrent _check_phase_complete
    # calls (possible because asyncio.create_task fires multiple result handlers).
    _phases_advancing: set = field(default_factory=set)

    def steps_in_phase(self, phase: int) -> list[PlanStep]:
        return [s for s in self.steps if s.phase == phase]

    def current_phase(self) -> int:
        """Lowest phase with any pending or running steps."""
        for step in sorted(self.steps, key=lambda s: s.phase):
            if step.status in ("pending", "running"):
                return step.phase
        return self.steps[-1].phase if self.steps else 1

    def phase_complete(self, phase: int) -> bool:
        return all(s.status in ("done", "failed") for s in self.steps_in_phase(phase))

    def all_complete(self) -> bool:
        return all(s.status in ("done", "failed") for s in self.steps)

    def max_phase(self) -> int:
        return max((s.phase for s in self.steps), default=1)

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "original_task": self.original_task,
            "retry_count": self.retry_count,
            "discord_message_id": self.discord_message_id,
            "context": self.context,
            "created_at": self.created_at,
            "steps": [
                {
                    "step_id": s.step_id, "phase": s.phase, "agent": s.agent,
                    "task": s.task, "expected": s.expected,
                    "status": s.status, "result": s.result,
                }
                for s in self.steps
            ],
        }


# ── Chat session ─────────────────────────────────────────────────────────────

@dataclass
class ChatSession:
    """
    Represents a continuous chat conversation scoped by topic continuity + time.
    A new session starts when the idle gap exceeds chat_idle_gap_secs OR the
    incoming message has low keyword overlap with the current session.
    """
    session_id: str
    name: str
    stream_key: str
    keywords: set = field(default_factory=set)
    topic_category: str = ""
    topic_confidence: float = 0.0
    message_count: int = 0
    last_activity: float = field(default_factory=time.time)


# ── Vote tracking ─────────────────────────────────────────────────────────────

@dataclass
class VoteState:
    """Tracks votes cast on a PLAN_PROPOSED event."""
    plan_id: str
    deadline: float                 # monotonic time when voting closes
    extended_until: float = 0.0    # extended deadline if any agent requested more time
    votes: list = field(default_factory=list)   # list of vote payload dicts
    resolved: bool = False


# ── Orchestrator ──────────────────────────────────────────────────────────────

class OrchestratorAgent(BaseAgent):

    think_interval         = 900   # 15 minutes
    PLAN_TIMEOUT           = 1200  # stale plan expiry (20 min — accounts for agent startup time)
    _CONVERSATION_TIMEOUT  = 900   # 15 min gap → reset context window

    # Planning prompt — requests phases + expected outcomes
    _PLAN_PROMPT = """\
You are a technical project manager. Translate a user request into a structured execution plan.

Phases run SEQUENTIALLY. Steps in the same phase run in PARALLEL.
Use multiple phases when later steps depend on earlier results.

Agents:
- executor     : shell commands, file R/W, docker (ls/cat/grep safe; git/docker/tee need approval)
- document_qa  : answer questions from /workspace/docs; review agent-stack architecture at /workspace/src;
                 generate LaTeX documents compiled to PDF (output: /workspace/docs/generated/)
- code_search  : search /workspace/repos for functions/classes/patterns
- research     : internet research via SearXNG — "what is X", "latest Y", "current status of Z"
- discord      : Discord server management (channels, categories, messages, topics, file uploads from /workspace)

Return ONLY valid JSON, no markdown fences. Two possible responses:

1. When requirements are clear — execution plan:
{"steps": [{"phase": 1, "task": "...", "agent": "...", "expected": "..."}, ...]}

2. When requirements are too vague to act on safely — clarification request:
{"clarify": "Your single focused question here"}

Rules for plans:
- expected = one short phrase describing success (e.g. "exit code 0", "file written", "list returned")
- 1–4 phases, 1–8 steps total
- discord agent for ANYTHING involving Discord channels/categories/messages/topics
- When unsure which agent, use executor
- Steps that depend on results of earlier steps MUST use a later phase number
- Prefer checking /workspace/tools/ for existing scripts before creating new executor steps

Step quality rules (CRITICAL — failure to follow causes task failures):
- PREFER two focused steps over one ambiguous step. "Read file X then update Y" → two steps.
- Each executor step must describe ONE concrete operation: a single command, a single file read,
  or a single file write. Never combine "read AND write" or "find AND update" in one step.
- Include the exact file path, command flags, or search query in the task string whenever known.
  BAD:  "update the config file"
  GOOD: "read /workspace/src/config/.env.example with cat"
  GOOD: "write updated REDIS_URL to /workspace/src/.env using tee"
- For multi-file or multi-command tasks, assign each file/command its own step (same phase if independent).
- code_search steps must name the specific function, class, or pattern to find.
- research steps must state the exact question to research."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        # Active execution plans: task_id → ExecutionPlan
        self._plans: dict[str, ExecutionPlan] = {}
        # Discord action tracking: task_id → {pending, results, discord_message_id, parent_task}
        self._pending_discord: dict[str, dict] = {}
        # Short-term conversation window — tuples of (task, result, timestamp)
        self._conversation: deque = deque(maxlen=10)
        # Learned intent examples: list of {"text": ..., "intent": ...}
        # Loaded from long-term memory on startup and augmented at runtime.
        self._learned_intents: list[dict] = []
        # Active chat sessions: session_id → ChatSession
        self._chat_sessions: dict[str, ChatSession] = {}
        # Active vote rounds: plan_id → VoteState
        self._active_votes: dict[str, VoteState] = {}

    _OWN_TOOLS = [
        ("route-to-executor",
         "Execute shell commands, read/write files, run scripts, restart containers",
         "event:task.assigned:executor",
         ["executor", "shell", "command", "file"]),
        ("route-to-code-search",
         "Search codebases for functions, classes, patterns, or explain how code works",
         "event:task.assigned:code_search",
         ["code", "search", "codebase", "grep"]),
        ("route-to-document-qa",
         "Answer questions from documents/PDFs, review agent-stack architecture, generate LaTeX PDF reports",
         "event:task.assigned:document_qa",
         ["documents", "qa", "summarize", "latex", "architecture", "pdf", "report"]),
        ("route-to-discord",
         "Manage Discord server: create/delete channels, send messages, pin posts",
         "event:task.assigned:discord",
         ["discord", "channel", "message"]),
        ("route-to-claude-code",
         "Complex coding tasks, code editing, multi-file changes using Claude API",
         "event:task.assigned:claude_code_agent",
         ["claude", "code", "edit", "programming"]),
        ("route-to-research",
         "Research factual questions across the internet — returns sourced, consensus-checked answers",
         "event:task.assigned:research",
         ["research", "web", "search", "lookup", "facts", "current", "latest"]),
    ]

    async def on_startup(self) -> None:
        log.info("orchestrator.startup")
        for name, desc, inv, tags in self._OWN_TOOLS:
            await self.memory.register_tool(name, desc, "orchestrator", inv, tags, "orchestrator")
        log.info("orchestrator.tools_seeded", count=len(self._OWN_TOOLS))
        await self._load_learned_intents()
        await self._seed_intent_examples()
        await self._check_version_reset()
        # Warn about any plans that were in-flight when the process last exited
        interrupted = await self.memory.load_active_plans()
        if interrupted:
            log.warning(
                "orchestrator.interrupted_plans_found",
                count=len(interrupted),
                tasks=[r["original_task"][:60] for r in interrupted],
            )
            for row in interrupted:
                await self.memory.upsert_plan(
                    row["task_id"], row["plan_id"], row["original_task"],
                    "interrupted", row["plan_json"],
                )

    # ── Event dispatch ───────────────────────────────────────────────────────

    async def handle_event(self, event: Event) -> None:
        if event.type == EventType.TASK_CREATED:
            await self._handle_task(event)
        elif event.type == EventType.TASK_COMPLETED:
            await self._handle_result(event)
        elif event.type == EventType.DISCORD_ACTION_DONE:
            await self._handle_discord_done(event)
        elif event.type == EventType.AGENT_VOTE:
            await self._handle_agent_vote(event)
        elif event.type == EventType.VOTE_EXTENSION_REQUESTED:
            await self._handle_vote_extension(event)
        elif event.type == EventType.AGENT_STARTED:
            log.info("orchestrator.agent_joined", agent=event.source)
        elif event.type == EventType.SESSION_RESET:
            await self._handle_session_reset()

    # ── Task intake ──────────────────────────────────────────────────────────

    async def _handle_task(self, event: Event) -> None:
        task = truncate_task(event.payload.get("task", ""))
        task_id = event.task_id
        discord_message_id = event.payload.get("discord_message_id")
        # session_id may be provided by the discord bridge after local classification
        hint_session_id = event.payload.get("session_id")
        log.info("orchestrator.task_received", task=task[:80], task_id=task_id)

        intent = await self._classify_intent(task)
        log.info("orchestrator.intent", intent=intent, task=task[:60])

        if intent == "chat":
            session = await self._get_or_create_chat_session(task, hint_session_id)
            # Publish incoming message to the chat context stream
            await self.bus.publish_to_context(
                session.session_id,
                Event(
                    type=EventType.TASK_CREATED,
                    source="discord",
                    payload={"task": task, "discord_message_id": discord_message_id or ""},
                    task_id=task_id,
                ),
            )
            session.message_count += 1
            session.last_activity = time.time()
            await self._respond_chat(task, task_id, discord_message_id, session_id=session.session_id)
        elif intent == "clarify":
            session = await self._get_or_create_chat_session(task, hint_session_id)
            session.message_count += 1
            session.last_activity = time.time()
            await self._respond_clarify(task, task_id, discord_message_id, session_id=session.session_id)
        else:
            # Detach planning so the event-handler returns immediately and acks
            # the task.created event.  The orchestrator can then receive and begin
            # handling new incoming tasks concurrently while this plan is being
            # built and dispatched.  Error handling is local to the background task.
            async def _plan_and_run() -> None:
                try:
                    await self._run_task(task, task_id, discord_message_id=discord_message_id)
                except Exception as exc:
                    log.error("orchestrator.plan_error", task=task[:80], error=str(exc))
                    await self._publish_reply(
                        f"⚠️ Planning error: {exc}",
                        task_id, discord_message_id, original_task=task,
                    )
            asyncio.create_task(_plan_and_run())

    # ── Intent classification ─────────────────────────────────────────────────

    async def _classify_intent(self, task: str) -> str:
        """
        Returns 'chat', 'clarify', or 'task'.

        Tier 1 — fast keyword patterns (no LLM).
        Tier 2 — LLM classifier with a minimal prompt (fallback for ambiguous input).
        """
        t = task.strip()

        # Fast: clearly conversational
        if _CHAT_RE.match(t):
            return "chat"

        # Fast: keyword router has a clear non-task route for this
        route = _route_by_keyword(t)
        if route and route[0]["agent"] == "direct":
            return "chat"

        # Fast: vague imperative with nothing in conversation to resolve it
        if _CLARIFY_RE.match(t) and not self._conversation:
            return "clarify"

        # Tier 2: keyword router found a concrete specialist → it's a task
        if route:
            return "task"

        # Tier 3: learned intent examples (persisted from prior LLM classifications)
        learned = self._lookup_learned_intent(t)
        if learned:
            log.debug("orchestrator.learned_intent_hit", intent=learned, task=t[:60])
            return learned

        # Tier 4: LLM classifier (fallback for genuinely ambiguous input)
        # Include recent conversation so pronouns like "the tool" / "it" resolve correctly.
        conv_ctx = ""
        if self._conversation:
            lines = [f"  [{i+1}] User said: {entry[0][:120]}" for i, entry in enumerate(self._conversation)]
            conv_ctx = "\nRecent conversation (use this to resolve references):\n" + "\n".join(lines) + "\n\n"
        messages = [
            SystemMessage(content=(
                "Classify the user message into exactly one word: chat, task, or clarify.\n"
                "chat    — conversational; no action needed (questions, greetings, status).\n"
                "task    — clear, actionable work that can be delegated to specialist agents.\n"
                "clarify — actionable in principle but too vague or ambiguous to execute safely.\n"
                "IMPORTANT: use the conversation context to resolve any references ('the tool', 'it', 'that').\n"
                "If the reference resolves via context, classify as 'task' not 'clarify'.\n"
                "Reply with exactly one word."
            )),
            HumanMessage(content=conv_ctx + t[:400]),
        ]
        try:
            response = await self.llm_invoke(messages)
            word = response.content.strip().lower().split()[0]
            if word in ("chat", "task", "clarify"):
                # Persist so the same pattern is recognised instantly next time
                await self._store_learned_intent(t, word)
                return word
        except Exception as exc:
            log.warning("orchestrator.classify_failed", error=str(exc))

        return "task"   # safe fallback — always attempt to help

    async def _respond_chat(
        self,
        task: str,
        task_id: str,
        discord_message_id: str | None,
        session_id: str = "",
    ) -> None:
        """Respond directly to a conversational message. No plan created."""
        conversation_context = ""
        if self._conversation:
            now = time.time()
            lines = [
                f"  [{i+1}] ({int((now - ts) / 60)}m ago) You: {t[:150]}\n       Me: {r[:150]}"
                for i, (t, r, ts) in enumerate(self._conversation)
            ]
            conversation_context = "\n\nRecent conversation:\n" + "\n".join(lines)

        # If we still lack context, check the chat stream history before replying
        if session_id and not conversation_context:
            hist = await self.resolve_from_context_history(task, session_id, lookback=15)
            if hist:
                conversation_context = f"\n\nFrom chat history:\n{hist}"

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=task + conversation_context),
        ]
        response = await self.llm_invoke(messages)
        reply = response.content

        # Publish reply to chat context stream for future recall
        if session_id:
            await self.bus.publish_to_context(
                session_id,
                Event(
                    type=EventType.AGENT_RESPONSE,
                    source=self.role,
                    payload={"reply": reply[:500], "task": task[:200]},
                    task_id=task_id,
                ),
            )
            # Keep snapshot current
            session = self._chat_sessions.get(session_id)
            if session:
                await self.memory.save_context_snapshot(
                    session_id, "chat", session.name,
                    topic_category=session.topic_category,
                    keywords=list(session.keywords)[:30],
                    message_count=session.message_count,
                    snapshot_json={"last_task": task[:200], "last_reply": reply[:200]},
                )

        await self._publish_reply(reply, task_id, discord_message_id, original_task=task)
        log.info("orchestrator.chat_replied", task=task[:60])

    async def _respond_clarify(
        self,
        task: str,
        task_id: str,
        discord_message_id: str | None,
        session_id: str = "",
    ) -> None:
        """
        Ask one focused clarifying question — but first check conversation and
        chat stream history so we don't ask for something already established.
        """
        conv_ctx = ""
        if self._conversation:
            now = time.time()
            lines = [
                f"  [{i+1}] ({int((now - entry[2]) / 60)}m ago) User: {entry[0][:120]}\n       Me: {entry[1][:120]}"
                for i, entry in enumerate(self._conversation)
            ]
            conv_ctx = "\n\nRecent conversation:\n" + "\n".join(lines)

        # Check chat stream history before prompting the user
        if session_id and not conv_ctx:
            hist = await self.resolve_from_context_history(task, session_id, lookback=20)
            if hist:
                conv_ctx = f"\n\nFrom chat history:\n{hist}"

        messages = [
            SystemMessage(content=(
                "The user's request may need clarification. "
                "First check the conversation context — if the subject is already established there, "
                "treat this as a follow-up and answer or act directly instead of asking again. "
                "Otherwise ask exactly one short, specific question. "
                "No preamble. No bullet lists. One sentence."
            )),
            HumanMessage(content=task[:400] + conv_ctx),
        ]
        response = await self.llm_invoke(messages)
        await self._publish_reply(response.content, task_id, discord_message_id, original_task=task)
        log.info("orchestrator.clarify_asked", task=task[:60])

    # ── Session reset ─────────────────────────────────────────────────────────

    async def _handle_session_reset(self) -> None:
        """
        Wipe the in-memory conversation window and all active chat sessions.
        Called when a  session.reset  event arrives on the broadcast stream
        (typically from a  `reset session`  control channel command).
        """
        cleared_conv  = len(self._conversation)
        cleared_chats = len(self._chat_sessions)
        self._conversation.clear()
        self._chat_sessions.clear()
        # Reload learned intents fresh from DB (drops any runtime noise)
        await self._load_learned_intents()
        log.info(
            "orchestrator.session_reset",
            cleared_conversation=cleared_conv,
            cleared_chat_sessions=cleared_chats,
            intents_reloaded=len(self._learned_intents),
        )

    async def _check_version_reset(self) -> None:
        """
        Compare the running AGENT_VERSION env var against what was stored in Redis
        on the previous run.  When the version changes, flush non-seed learned intents
        so stale classification examples trained on old behaviour do not pollute
        the new version.  DB contents (knowledge, tasks, plans) are preserved.
        """
        current_version = os.environ.get("AGENT_VERSION", "")
        if not current_version:
            return  # version pinning not configured — skip

        redis_key = "config:agent_version"
        try:
            stored_version = await self.bus._client.get(redis_key)
            if stored_version and stored_version != current_version:
                # Version changed — drop non-seed intents from in-memory cache
                before = len(self._learned_intents)
                self._learned_intents = [
                    e for e in self._learned_intents if "seed" in e.get("tags", [])
                ]
                after = len(self._learned_intents)
                log.warning(
                    "orchestrator.version_changed_intent_flush",
                    old_version=stored_version,
                    new_version=current_version,
                    flushed=before - after,
                    kept=after,
                )
            await self.bus._client.set(redis_key, current_version)
        except Exception as exc:
            log.warning("orchestrator.version_check_failed", error=str(exc))

    # ── Learned intent store ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_intent_entry(e: dict) -> dict:
        """
        Normalize a raw DB row (content/tags keys) into the shape expected by
        _lookup_learned_intent (text/intent keys).

        Two content formats exist:
          - Seed entries:    plain natural-language text, intent encoded in tags list
          - Runtime entries: "INTENT_EXAMPLE intent=X text=Y…" string
        """
        content = e.get("content", "")
        tags    = e.get("tags") or []
        if isinstance(tags, str):
            import json as _json
            try:
                tags = _json.loads(tags)
            except Exception:
                tags = []

        if content.startswith("INTENT_EXAMPLE intent="):
            # Parse "INTENT_EXAMPLE intent=task text=list all running…"
            try:
                rest   = content[len("INTENT_EXAMPLE intent="):]
                intent = rest.split()[0]
                text   = rest[len(intent):].lstrip()
                if text.startswith("text="):
                    text = text[5:]
            except Exception:
                intent = "task"
                text   = content
        else:
            # Seed / organic batch_store format: content is the plain text
            text   = content
            # intent is the tag that is neither "intent", "seed", nor "classification"
            intent = next(
                (t for t in tags if t not in ("intent", "seed", "classification")),
                "task",
            )

        return {"text": text, "intent": intent, "tags": tags}

    async def _load_learned_intents(self) -> None:
        """
        Load persisted intent classifications from long-term memory.
        These augment the hard-coded patterns and grow over time.
        """
        try:
            entries = await self.recall("intent classification examples", limit=50)
            self._learned_intents = [
                self._normalize_intent_entry(e)
                for e in entries
                if e.get("topic") == "learned_intent"
            ]
            log.info("orchestrator.learned_intents_loaded", count=len(self._learned_intents))
        except Exception as exc:
            log.warning("orchestrator.learned_intents_load_failed", error=str(exc))

    async def _seed_intent_examples(self) -> None:
        """
        On first startup (or after a DB wipe), seed _SEED_INTENTS into PostgreSQL
        so Tier 3 classification is useful immediately without waiting for organic
        LLM-classified examples to accumulate.

        Idempotent: skips seeding when ≥ 20 seed-tagged entries are already loaded.
        """
        seed_already_present = sum(
            1 for e in self._learned_intents if "seed" in e.get("tags", [])
        )
        if seed_already_present >= 20:
            log.debug("orchestrator.intent_seed_skipped", existing=seed_already_present)
            return

        entries = [
            {
                "content": text,
                "topic": "learned_intent",
                "tags": ["intent", intent, "seed"],
                "metadata": {"intent": intent},
            }
            for text, intent in _SEED_INTENTS
        ]
        try:
            result = await self.memory.batch_store(entries)
            # Also populate the in-memory cache immediately
            for text, intent in _SEED_INTENTS:
                self._learned_intents.append({"text": text, "intent": intent, "tags": ["seed"]})
            if len(self._learned_intents) > 200:
                self._learned_intents = self._learned_intents[-200:]
            log.info(
                "orchestrator.intent_seed_done",
                stored=result.get("count", 0),
                total_in_memory=len(self._learned_intents),
            )
        except Exception as exc:
            log.warning("orchestrator.intent_seed_failed", error=str(exc))

    async def _store_learned_intent(self, text: str, intent: str) -> None:
        """
        Persist a (text, intent) classification to long-term memory so it can
        be recalled in future sessions and used as a fast-path lookup.
        Also store custom intent patterns emitted by agents via the memory API.
        """
        try:
            await self.promote_now(
                content=f"INTENT_EXAMPLE intent={intent} text={text[:200]}",
                topic="learned_intent",
                tags=["intent", intent, "classification"],
            )
            # Append to in-memory cache so it takes effect immediately
            self._learned_intents.append({"text": text[:200], "intent": intent})
            # Keep cache bounded
            if len(self._learned_intents) > 200:
                self._learned_intents = self._learned_intents[-200:]
            log.debug("orchestrator.intent_stored", intent=intent, text=text[:60])
        except Exception as exc:
            log.warning("orchestrator.intent_store_failed", error=str(exc))

    def _lookup_learned_intent(self, task: str) -> str | None:
        """
        Check learned examples for a close match.
        Uses simple word-overlap similarity — no embeddings needed.
        Returns the intent string if a sufficiently similar example is found.
        """
        if not self._learned_intents:
            return None
        task_words = set(re.sub(r"[^a-z0-9\s]", "", task.lower()).split())
        best_intent: str | None = None
        best_score = 0.0
        for ex in self._learned_intents:
            ex_words = set(re.sub(r"[^a-z0-9\s]", "", ex.get("text", "").lower()).split())
            union = task_words | ex_words
            if not union:
                continue
            score = len(task_words & ex_words) / len(union)  # Jaccard similarity
            if score > best_score:
                best_score = score
                best_intent = ex.get("intent")
        # Only trust high-confidence matches
        if best_score >= 0.6 and best_intent in ("chat", "task", "clarify"):
            return best_intent
        return None

    # ── Chat session management ───────────────────────────────────────────────

    async def _get_or_create_chat_session(
        self, task: str, hint_session_id: str | None = None
    ) -> ChatSession:
        """
        Find or create a chat session for this message.

        Session continuity rules:
          1. If the discord bridge provided a hint_session_id that is still
             active, reuse it unconditionally.
          2. Otherwise scan active sessions for one whose keywords overlap
             the incoming message's keywords (Jaccard ≥ chat_keyword_overlap)
             AND whose last_activity is within chat_idle_gap_secs.
          3. If nothing matches, start a fresh session.
        """
        import re as _re

        idle_gap   = float(await self.bus.get_config("chat_idle_gap_secs", 1800))
        kw_thresh  = float(await self.bus.get_config("chat_keyword_overlap", 0.4))
        now        = time.time()

        # Hint from the bridge
        if hint_session_id and hint_session_id in self._chat_sessions:
            return self._chat_sessions[hint_session_id]

        # Extract keywords from current message
        msg_words = set(
            w for w in _re.sub(r"[^a-z0-9\s]", "", task.lower()).split()
            if len(w) > 2
        )

        best_session: ChatSession | None = None
        best_score = 0.0

        for session in list(self._chat_sessions.values()):
            # Expired by idle gap?
            if now - session.last_activity > idle_gap:
                await self._close_chat_session(session)
                continue
            # Keyword overlap
            union = msg_words | session.keywords
            if not union:
                continue
            score = len(msg_words & session.keywords) / len(union)
            if score >= kw_thresh and score > best_score:
                best_score = score
                best_session = session

        if best_session:
            # Merge new keywords in
            best_session.keywords |= msg_words
            return best_session

        # New session
        session_id = str(uuid.uuid4())
        slug_words = [w for w in msg_words if len(w) > 3][:4]
        name       = "-".join(slug_words) or "chat"
        stream_key = await self.bus.create_context_stream(
            "chat", session_id, name,
            metadata={"message_count": 0, "keywords": list(msg_words)[:30]},
        )
        session = ChatSession(
            session_id=session_id,
            name=name,
            stream_key=stream_key,
            keywords=msg_words,
        )

        # Classify topic
        category, confidence = await self.classify_topic(task)
        session.topic_category   = category or ""
        session.topic_confidence = confidence

        # Ask user for topic label if confidence is low and we have enough sessions
        if category is None or (await self.should_ask_user_for_topic(confidence)):
            log.debug("orchestrator.topic_unknown", task=task[:60], confidence=confidence)
        else:
            await self.memory.save_topic_pattern(category, list(msg_words)[:10], confidence)

        # Persist initial snapshot
        await self.memory.save_context_snapshot(
            session_id, "chat", name,
            topic_category=category,
            keywords=list(msg_words)[:30],
            message_count=0,
            status="active",
        )

        self._chat_sessions[session_id] = session
        await self.emit(
            EventType.CHAT_SESSION_STARTED,
            payload={"session_id": session_id, "name": name, "topic": category or ""},
            target="broadcast",
        )
        log.info("orchestrator.chat_session_started", session_id=session_id[:8], name=name)
        return session

    async def _close_chat_session(self, session: ChatSession) -> None:
        """
        Score and archive a chat session to long-term memory.
        Sessions with ≥3 messages or a resolved topic get a high value_score
        and are stored for later recall.
        """
        self._chat_sessions.pop(session.session_id, None)

        # Value heuristic: longer conversations + having a known topic are worth storing
        value_score = min(1.0, session.message_count / 10.0)
        if session.topic_category:
            value_score = min(1.0, value_score + 0.3)

        if value_score < 0.2:
            # Too brief — just close the context stream
            await self.bus.close_context(session.session_id)
            log.debug("orchestrator.chat_session_discarded", session_id=session.session_id[:8])
            return

        # Generate a brief summary via LLM (only for valuable sessions)
        summary = f"Chat session: {session.name}. Messages: {session.message_count}."
        if session.topic_category:
            summary += f" Topic: {session.topic_category}."

        await self.memory.close_context_snapshot(
            session.session_id,
            summary=summary,
            snapshot_json={
                "name":          session.name,
                "keywords":      list(session.keywords)[:30],
                "topic":         session.topic_category,
                "message_count": session.message_count,
            },
            value_score=value_score,
        )
        await self.bus.close_context(session.session_id)
        await self.emit(
            EventType.CHAT_SESSION_CLOSED,
            payload={
                "session_id":  session.session_id,
                "name":        session.name,
                "messages":    session.message_count,
                "value_score": round(value_score, 2),
                "summary":     summary,
            },
            target="broadcast",
        )
        log.info(
            "orchestrator.chat_session_closed",
            session_id=session.session_id[:8],
            messages=session.message_count,
            value_score=round(value_score, 2),
        )

    # ── Voting helpers ────────────────────────────────────────────────────────

    async def _handle_agent_vote(self, event: Event) -> None:
        plan_id = event.payload.get("plan_id", "")
        vs = self._active_votes.get(plan_id)
        if not vs or vs.resolved:
            return
        vs.votes.append(event.payload)
        log.info(
            "orchestrator.vote_received",
            agent=event.source,
            plan_id=plan_id[:8],
            approve=event.payload.get("approve"),
            confidence=event.payload.get("confidence"),
        )

    async def _handle_vote_extension(self, event: Event) -> None:
        plan_id    = event.payload.get("plan_id", "")
        vs         = self._active_votes.get(plan_id)
        if not vs or vs.resolved:
            return
        requested  = int(event.payload.get("requested_ms", 0))
        max_ext    = int(await self.bus.get_config("vote_max_extension_ms", 10_000))
        granted    = min(requested, max_ext)
        new_ext    = time.monotonic() + granted / 1000
        if new_ext > vs.extended_until:
            vs.extended_until = new_ext
        log.info(
            "orchestrator.vote_extension_granted",
            agent=event.source,
            plan_id=plan_id[:8],
            granted_ms=granted,
        )

    async def _propose_plan_and_vote(self, plan: ExecutionPlan) -> tuple[bool, list[str]]:
        """
        Broadcast a proposed plan, collect agent votes for vote_timeout_ms
        (plus any granted extensions), then decide whether to proceed.

        Returns (proceed, rejection_reasons).
        proceed=False means the plan should be revised before execution.
        """
        vote_ms  = int(await self.bus.get_config("vote_timeout_ms", 3_000))
        deadline = time.monotonic() + vote_ms / 1000

        vs = VoteState(plan_id=plan.plan_id, deadline=deadline)
        self._active_votes[plan.plan_id] = vs

        await self.emit(
            EventType.PLAN_PROPOSED,
            payload={
                "plan_id":       plan.plan_id,
                "task_id":       plan.task_id,
                "original_task": plan.original_task[:200],
                "steps": [
                    {"phase": s.phase, "agent": s.agent, "task": s.task[:80]}
                    for s in plan.steps
                ],
            },
            target="broadcast",
        )

        # Wait for votes with possible extension
        while True:
            effective = max(vs.deadline, vs.extended_until)
            remaining = effective - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(0.2, remaining))
            if time.monotonic() >= effective:
                break

        vs.resolved = True
        self._active_votes.pop(plan.plan_id, None)

        if not vs.votes:
            return True, []  # silence = unanimous approval

        rejections = [
            v for v in vs.votes
            if not v.get("approve", True) and float(v.get("confidence", 0)) >= 0.7
        ]
        if rejections:
            reasons = [v.get("reason", "no reason") for v in rejections]
            reason_str = "; ".join(reasons)
            await self._emit_plan_status(
                plan,
                f"🗳️ Plan contested by {len(rejections)} agent(s): {reason_str[:200]}. Revising…",
            )
            log.info(
                "orchestrator.plan_contested",
                plan_id=plan.plan_id[:8],
                rejections=len(rejections),
                reasons=reason_str[:120],
            )
            return False, reasons

        approvals = [v for v in vs.votes if v.get("approve", True)]
        log.info(
            "orchestrator.plan_approved",
            plan_id=plan.plan_id[:8],
            approvals=len(approvals),
        )
        return True, []

    # ── Planning ─────────────────────────────────────────────────────────────

    async def _run_task(
        self,
        task: str,
        task_id: str,
        discord_message_id: str | None = None,
        retry_context: str = "",
        _plan_attempt: int = 0,
    ) -> None:
        """Build an ExecutionPlan and start phase 1."""
        # Conversation context (resolve follow-ups like "clean it up", "yes proceed")
        conversation_context = ""
        if self._conversation:
            lines = [f"  [{i+1}] {t[:160]}" for i, (t, _r, _ts) in enumerate(self._conversation)]
            conversation_context = "\nRecent conversation (oldest→newest):\n" + "\n".join(lines)

        # Long-term memory — cap at 2 to avoid pollution
        prior_knowledge = (await self.recall(task))[:2]
        context = ""
        if prior_knowledge:
            raw_context = "\n\nRelevant prior knowledge:\n" + "\n".join(
                f"- [{e['topic']}] {e['content'][:200]}" for e in prior_knowledge
            )
            context = truncate_context(raw_context)

        # Tool registry — find relevant tools before calling the LLM planner.
        # This lets the planner reference concrete tool names instead of guessing.
        tool_hits = await self.search_tools(task, limit=6)
        tools_context = self.format_tools_context(tool_hits)
        if tools_context:
            context += tools_context

        # Build plan steps
        keyword_plan = _route_by_keyword(task)
        if keyword_plan and not retry_context:
            steps = [
                PlanStep(
                    step_id=str(uuid.uuid4()),
                    phase=s.get("phase", 1),
                    task=s["task"],
                    agent=s["agent"],
                    expected=s.get("expected", ""),
                )
                for s in keyword_plan
            ]
        else:
            steps = await self._llm_plan(
                task, context, conversation_context, retry_context,
                task_id=task_id, discord_message_id=discord_message_id,
            )

        # Planner asked for clarification — no plan to build
        if not steps:
            return

        # Create the task context stream (source of truth for all events in this task)
        import re as _re
        slug_words = [w for w in _re.sub(r"[^a-z0-9\s]", "", task.lower()).split() if len(w) > 3][:5]
        task_slug = "-".join(slug_words) or "task"
        await self.bus.create_context_stream(
            "task", task_id, task_slug,
            metadata={"original_task": task[:200], "discord_message_id": discord_message_id or ""},
        )
        # Save initial snapshot so the task is immediately recallable by ID
        await self.memory.save_context_snapshot(
            task_id, "task", task_slug,
            snapshot_json={"task": task[:500]},
            status="active",
        )

        plan = ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            task_id=task_id,
            original_task=task,
            steps=steps,
            discord_message_id=discord_message_id,
            context=context,
            context_id=task_id,
        )

        phases = sorted(set(s.phase for s in steps))
        plan_summary = "\n".join(
            f"  Phase {p}: " + " | ".join(
                f"[{s.agent}] {s.task[:60]}" for s in steps if s.phase == p
            )
            for p in phases
        )
        await self._emit_plan_status(
            plan,
            f"📋 Plan ({len(steps)} step(s), {len(phases)} phase(s))\n{plan_summary}",
        )

        # Voting round — specialists can contest before execution begins
        if _plan_attempt < 2:   # max 2 revision cycles to avoid loops
            proceed, rejection_reasons = await self._propose_plan_and_vote(plan)
            if not proceed:
                reason_ctx = "; ".join(rejection_reasons)
                await self._run_task(
                    task, task_id, discord_message_id,
                    retry_context=f"\n[Plan revision {_plan_attempt+1}: agents objected. {reason_ctx}]",
                    _plan_attempt=_plan_attempt + 1,
                )
                return

        self._plans[task_id] = plan
        await self.memory.upsert_plan(task_id, plan.plan_id, task, "running", plan.to_dict())
        await self._dispatch_phase(plan, min(phases))

    async def _fit_plan_context(
        self,
        task: str,
        context: str,
        conversation_context: str,
        retry_context: str,
    ) -> str:
        """
        Fit context components within the available token budget so the LLM
        planner always receives the full task description.

        Priority (highest → lowest):
          1. task + retry_context  — always preserved
          2. conversation_context — trimmed to oldest N entries
          3. context (memory/tools) — dropped last
        """
        limit   = await self._get_model_context_limit()
        # Reserve 25% for the reply and ~10% for the system prompt overhead
        budget  = int(limit * 0.65)
        base_tokens = self._estimate_tokens([
            SystemMessage(content=self._PLAN_PROMPT),
            HumanMessage(content=f"Task: {task}{retry_context}"),
        ])
        available = budget - base_tokens

        if available <= 0:
            # Task alone overflows — nothing we can add; let llm_invoke truncate
            log.warning(
                "orchestrator.plan_context_overflow",
                task_len=len(task),
                base_tokens=base_tokens,
                budget=budget,
            )
            return retry_context

        # Try adding full context + conversation
        full = context + conversation_context
        if self._estimate_tokens([HumanMessage(content=full)]) <= available:
            return context + conversation_context + retry_context

        # Trim conversation to 3 most recent turns
        trimmed_conv = ""
        if self._conversation:
            recent = list(self._conversation)[-3:]
            lines = [f"  [{i+1}] {t[:160]}" for i, (t, _r, _ts) in enumerate(recent)]
            trimmed_conv = "\nRecent conversation (last 3):\n" + "\n".join(lines)
        if self._estimate_tokens([HumanMessage(content=context + trimmed_conv)]) <= available:
            log.info("orchestrator.plan_context_trimmed", kept="conv_3", task=task[:60])
            return context + trimmed_conv + retry_context

        # Drop conversation entirely, keep memory/tools context
        if self._estimate_tokens([HumanMessage(content=context)]) <= available:
            log.info("orchestrator.plan_context_trimmed", kept="context_only", task=task[:60])
            return context + retry_context

        # Context alone is too large — truncate it to fit
        max_ctx_chars = available * 4   # ~4 chars/token
        truncated_ctx = context[:max_ctx_chars] + "\n[…context truncated]" if len(context) > max_ctx_chars else context
        log.info("orchestrator.plan_context_trimmed", kept="context_truncated", task=task[:60])
        return truncated_ctx + retry_context

    async def _llm_plan(
        self,
        task: str,
        context: str,
        conversation_context: str,
        retry_context: str,
        task_id: str = "",
        discord_message_id: str | None = None,
    ) -> list[PlanStep]:
        """
        Call the LLM to build a structured phased plan.
        If the LLM returns {"clarify": "..."}, emit the question and return [].
        Falls back to a single executor step on parse failure.
        """
        fitted_context = await self._fit_plan_context(
            task, context, conversation_context, retry_context
        )
        messages = [
            SystemMessage(content=self._PLAN_PROMPT),
            HumanMessage(content=f"Task: {task}{fitted_context}"),
        ]
        try:
            response = await self.llm_invoke(messages)
            raw = response.content.strip()
            if "```" in raw:
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else parts[0]
                if raw.lower().startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)

            # Planner is asking for clarification
            if "clarify" in data and task_id:
                question = str(data["clarify"])
                log.info("orchestrator.planner_clarify", question=question[:100])
                await self._publish_reply(question, task_id, discord_message_id, original_task=task)
                return []

            steps = self._parse_plan_steps(data)
            if steps:
                return steps
        except Exception as exc:
            log.warning("orchestrator.plan_parse_failed", error=str(exc))
        # Hard fallback
        return [PlanStep(step_id=str(uuid.uuid4()), phase=1, task=task, agent="executor", expected="")]

    _MAX_PLAN_STEPS = 8   # hard cap — prevents runaway plans from the LLM

    def _parse_plan_steps(self, data: dict) -> list[PlanStep]:
        raw = data.get("steps", [])
        if not raw or not isinstance(raw, list):
            raise ValueError("no steps in plan")

        # Clamp to hard cap — take the first N steps rather than failing
        if len(raw) > self._MAX_PLAN_STEPS:
            log.warning(
                "orchestrator.plan_steps_clamped",
                original=len(raw),
                clamped=self._MAX_PLAN_STEPS,
            )
            raw = raw[: self._MAX_PLAN_STEPS]

        steps = []
        for s in raw:
            task = str(s.get("task", "")).strip()
            if not task:
                continue
            steps.append(PlanStep(
                step_id=str(uuid.uuid4()),
                phase=max(1, int(s.get("phase", 1))),
                task=task[:600],          # hard cap on step task length
                agent=str(s.get("agent", "executor")).strip(),
                expected=str(s.get("expected", "")).strip()[:200],
            ))
        return steps

    # ── Phase dispatch ───────────────────────────────────────────────────────

    async def _dispatch_phase(self, plan: ExecutionPlan, phase: int) -> None:
        """Dispatch all pending steps in the given phase."""
        steps = [s for s in plan.steps if s.phase == phase and s.status == "pending"]
        if not steps:
            await self._check_phase_complete(plan)
            return

        step_desc = " | ".join(f"[{s.agent}] {s.task[:50]}" for s in steps)
        await self._emit_plan_status(plan, f"⚡ Phase {phase}: {step_desc}")

        # Pre-warm all specialist containers concurrently before dispatching any tasks.
        # This avoids serial 20-30s startup delays that would stall the event loop.
        specialist_steps = [s for s in steps if s.agent not in ("discord", "direct")]
        if specialist_steps:
            start_results = await asyncio.gather(
                *[self._ensure_agent_running(s.agent, plan) for s in specialist_steps]
            )
            for step, ok in zip(specialist_steps, start_results):
                if not ok:
                    step.status = "failed"
                    step.result = f"Failed to start agent container: {step.agent}"

        discord_count = 0
        for step in steps:
            if step.status == "failed":
                continue  # already marked failed during pre-warm
            step.status = "running"

            if step.agent == "discord":
                n = await self._execute_discord_subtask(step.task, plan.task_id)
                discord_count += n
                if n == 0:
                    step.status = "failed"
                    step.result = "Could not parse Discord action from task description"

            elif step.agent == "direct":
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT + self.self_modify_context() + self.task_queue_context()),
                    HumanMessage(content=f"Answer directly: {step.task}"),
                ]
                resp = await self.llm_invoke(messages)
                step.result = resp.content
                step.status = "done"

            else:
                stream_target = self._AGENT_STREAM.get(step.agent, step.agent)
                await self.emit(
                    EventType.TASK_ASSIGNED,
                    payload={
                        "task": step.task,
                        "assigned_to": step.agent,
                        "parent_task_id": plan.task_id,
                        "subtask_id": step.step_id,
                    },
                    target=stream_target,
                )
                await self._emit_plan_status(
                    plan,
                    f"→ `[{step.agent}]` {step.task[:120]}",
                )
                log.info("orchestrator.dispatched", agent=step.agent, step_id=step.step_id)

        # Register discord action tracking for this phase
        if discord_count > 0:
            existing = self._pending_discord.get(plan.task_id)
            if existing:
                existing["pending"] += discord_count
            else:
                self._pending_discord[plan.task_id] = {
                    "pending": discord_count,
                    "results": [],
                    "discord_message_id": plan.discord_message_id,
                    "parent_task": plan.original_task,
                }

        # Check completion only when no steps are waiting for async results.
        # Phases with remote agent steps will be advanced when their results arrive.
        has_async = any(
            s.status == "running" and s.agent not in ("direct",)
            for s in plan.steps if s.phase == phase
        )
        if not has_async:
            await self._check_phase_complete(plan)

    # ── Result handling ──────────────────────────────────────────────────────

    SPECIALIST_ROLES = {"document_qa", "code_search", "executor", "claude_code_agent", "research"}

    # Some agents listen on a different stream than their role name.
    _AGENT_STREAM = {
        "claude_code_agent": "claude_code",  # legacy stream name kept for discord-bridge compat
    }

    async def _handle_result(self, event: Event) -> None:
        if event.source not in self.SPECIALIST_ROLES:
            return

        result    = event.payload.get("result", "")
        subtask_id = event.payload.get("subtask_id")
        parent_id  = event.payload.get("parent_task_id")
        log.info(
            "orchestrator.result_received",
            source=event.source,
            subtask_id=subtask_id,
            parent_id=parent_id,
            result_len=len(result),
        )

        plan = self._plans.get(parent_id) if parent_id else None
        if plan:
            preview = result[:120].replace("\n", " ")
            await self._emit_plan_status(
                plan,
                f"← `[{event.source}]` result ({len(result)} chars): {preview}{'…' if len(result) > 120 else ''}",
            )
            step = next((s for s in plan.steps if s.step_id == subtask_id), None)
            if step:
                passed, reason = self._validate_result(step.expected, result, source=event.source)
                step.result = result
                step.status = "done" if passed else "failed"
                if not passed:
                    await self._emit_plan_status(
                        plan, f"⚠️ `[{event.source}]` validation failed: {reason}"
                    )
                else:
                    await self._emit_plan_status(plan, f"✅ `[{event.source}]` step complete")
            else:
                # Fallback: mark first running step for this agent
                for s in plan.steps:
                    if s.status == "running" and s.agent == event.source:
                        passed, _ = self._validate_result(s.expected, result, source=event.source)
                        s.result = result
                        s.status = "done" if passed else "failed"
                        break

            if event.payload.get("promote", False):
                await self.promote_now(result, topic=event.source, tags=[event.source])

            await self._check_phase_complete(plan)
            return

        # No matching plan — the plan was either expired or this is an orphan result.
        # Try to recover discord_message_id from the context stream registry so the
        # user still gets a reply even after plan expiry.
        log.warning(
            "orchestrator.orphan_result",
            source=event.source,
            parent_id=parent_id,
            subtask_id=subtask_id,
        )
        discord_message_id = None
        effective_task_id  = parent_id or event.task_id
        if parent_id:
            ctx_meta = await self.bus.get_context_metadata(parent_id)
            if ctx_meta:
                discord_message_id = ctx_meta.get("discord_message_id") or None

        await self._synthesise_and_reply(
            original_task=event.payload.get("task", ""),
            raw_results=[(event.source, result)],
            task_id=effective_task_id,
            discord_message_id=discord_message_id,
        )

    async def _handle_discord_done(self, event: Event) -> None:
        payload = event.payload
        task_id = payload.get("task_id")
        if not task_id or task_id not in self._pending_discord:
            return

        entry = self._pending_discord[task_id]
        action = payload.get("action", "unknown")
        result = payload.get("result", "")
        ok     = bool(payload.get("ok", True))
        entry["results"].append((action, result, ok))
        entry["pending"] = max(0, entry["pending"] - 1)

        if entry["pending"] > 0:
            return

        del self._pending_discord[task_id]

        plan = self._plans.get(task_id)
        if plan:
            # Update discord steps in the current phase
            composite = "\n".join(
                f"{'✅' if ok else '❌'} {act}: {res}"
                for act, res, ok in entry["results"]
            )
            all_ok = all(ok for _, _, ok in entry["results"])
            for step in plan.steps:
                if step.agent == "discord" and step.status == "running":
                    step.result = composite
                    step.status = "done" if all_ok else "failed"
            await self._check_phase_complete(plan)
            return

        # Discord-only task (no associated plan)
        parts = [f"{'✅' if ok else '❌'} `{act}`: {res}" for act, res, ok in entry["results"]]
        await self._publish_reply(
            "\n".join(parts) or "Discord actions completed.",
            task_id,
            entry.get("discord_message_id"),
            original_task=entry.get("parent_task", ""),
        )

    # ── Validation ───────────────────────────────────────────────────────────

    # Tags injected by specialist agents to prove execution actually happened.
    # A result that is pure LLM prose without one of these is treated as a
    # planning reply, not completed work.
    _EXECUTION_PROOF_MARKERS = (
        "Exit code:",          # executor ran a shell command
        "[EXECUTOR_NO_CMD]",   # executor flagged no command — will fail below
        "STDOUT:",             # executor ran a command with output
        "STDERR:",             # executor ran a command (even if it errored)
        "✅",                  # discord action confirmed
        "❌",                  # discord action failed (still ran)
        "[research]",          # research agent citation block
        "[doc]",               # document_qa citation block
        "code_search:",        # code search result header
    )

    def _validate_result(self, expected: str, result: str, source: str = "") -> tuple[bool, str]:
        """
        Rule-based result validation. Returns (passed, reason).

        Core rule: a successful result must prove work was actually done.
        For executor steps: an exit code must be present.
        For all agents: [EXECUTOR_NO_CMD] and hard error markers are failures.
        """
        if not result:
            return False, "Empty result"

        # Executor couldn't extract or run a command — it only chatted
        if result.startswith("[EXECUTOR_NO_CMD]"):
            return False, "Executor produced no command — LLM described the task instead of running it"

        # Hard failure markers
        fail_markers = [
            "Exit code: 1\n", "Exit code: 2\n", "Exit code: 127\n",
            "Traceback (most recent", "Exception:", "not on the allowlist",
            "Execution error:", "Command timed out",
        ]
        for m in fail_markers:
            if m in result:
                return False, m.strip()

        # Discord action failures
        if "'ok': False" in result or '"ok": false' in result.lower():
            return False, "Discord action reported failure"

        # Executor-specific: no evidence of command execution = chatted instead of worked
        if source == "executor" and "Exit code:" not in result:
            return False, "Executor result has no 'Exit code:' — command was not run"

        # Expected outcome check: if the expected description names a concrete
        # success signal, verify it appears somewhere in the result.
        if expected:
            exp_lower = expected.lower()
            result_lower = result.lower()
            # "exit code 0" → the result must contain "exit code: 0"
            if "exit code 0" in exp_lower and "exit code: 0" not in result_lower:
                return False, f"Expected '{expected}' but result has no 'Exit code: 0'"
            # "file written" → result should mention a path or "written"
            if "file written" in exp_lower and not any(
                kw in result_lower for kw in ("written", "created", "saved", "/workspace")
            ):
                return False, f"Expected '{expected}' but result shows no file write confirmation"

        return True, ""

    # ── Internet fix-research ────────────────────────────────────────────────

    _SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")

    async def _research_fix(self, task: str, error: str, max_results: int = 5) -> str:
        """
        Query SearXNG directly for solutions to a failed task/error.
        Returns a compact string of (domain, snippet) pairs ready to inject
        into a fix step's task description, or "" if search is unavailable.
        """
        # Build a concise, search-engine-friendly query from the error
        # Strip ANSI codes and long tracebacks; keep the last meaningful line
        clean_error = re.sub(r'\x1b\[[0-9;]*m', '', error)
        error_lines = [l.strip() for l in clean_error.splitlines() if l.strip()]
        # Prefer lines that look like error messages (contain "Error", "error", "failed", etc.)
        error_hint = next(
            (l for l in reversed(error_lines) if re.search(r'error|failed|exception|not found|cannot', l, re.I)),
            error_lines[-1] if error_lines else error[:120],
        )
        query = f"{error_hint[:100]} fix"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._SEARXNG_URL}/search",
                    params={"q": query, "format": "json", "language": "en", "safesearch": "0"},
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])[:max_results]
        except Exception as exc:
            log.debug("orchestrator.fix_research_unavailable", error=str(exc))
            return ""

        if not results:
            return ""

        lines = []
        for r in results:
            domain  = urlparse(r.get("url", "")).netloc.lstrip("www.")
            snippet = r.get("content", r.get("snippet", "")).strip()[:200]
            if snippet:
                lines.append(f"  [{domain}] {snippet}")

        if not lines:
            return ""

        log.info("orchestrator.fix_research_found",
                 query=query[:60], hits=len(lines))
        return (
            f"\n\n[Internet search for '{query[:80]}' found these hints:]\n"
            + "\n".join(lines)
        )

    # ── Phase completion / retry / escalation ────────────────────────────────

    async def _check_phase_complete(self, plan: ExecutionPlan) -> None:
        """
        Called after each step completes. Advances phases, retries on failure,
        or escalates to the user once MAX_RETRIES is exhausted.

        Guard: because event handlers run concurrently (asyncio.create_task), two
        results arriving in the same phase can both pass the "running" check before
        either one has awaited anything. We record which phases are already being
        advanced and bail immediately if this phase is already claimed.
        """
        phase = plan.current_phase()

        # Still waiting for discord actions or running steps?
        if plan.task_id in self._pending_discord:
            return
        running = [s for s in plan.steps if s.phase == phase and s.status == "running"]
        if running:
            return

        # Atomically claim this phase — bail if another coroutine already has it.
        # This must happen before the first `await` to be effective.
        if phase in plan._phases_advancing:
            log.debug("orchestrator.phase_advance_skipped", plan_id=plan.plan_id, phase=phase)
            return
        plan._phases_advancing.add(phase)

        failed = [s for s in plan.steps if s.phase == phase and s.status == "failed"]

        if failed:
            max_depth = int(await self.bus.get_config("max_fix_depth", 10))
            # Check if any failed step has capacity for another fix attempt
            fixable = [s for s in failed if s.depth < max_depth]

            if fixable:
                next_phase = max(s.phase for s in plan.steps) + 1
                fix_steps_added = 0
                for step in fixable:
                    new_depth  = step.depth + 1
                    error_text = step.result[:300]
                    error_chain = step.error_chain + [f"[depth {step.depth}] {error_text}"]

                    # ── Strategy rotation ──────────────────────────────────
                    # depth 1–3: same agent, error context
                    # depth 4–6: escalate to claude_code_agent for deeper reasoning
                    # depth 7–9: orchestrator reformulates the task entirely
                    # depth 10:  exhausted — escalated below (not fixable)
                    if new_depth <= 3:
                        fix_agent = step.agent
                        chain_ctx = "\n".join(f"  - {e}" for e in error_chain[-3:])
                        fix_task  = (
                            f"{step.task}\n\n"
                            f"[Fix attempt {new_depth}: previous attempt failed.]\n"
                            f"Error history:\n{chain_ctx}\n"
                            "Try a different approach."
                        )
                    elif new_depth <= 6:
                        fix_agent = "claude_code_agent"
                        chain_ctx = "\n".join(f"  - {e}" for e in error_chain[-3:])
                        web_hints = await self._research_fix(step.task, error_text)
                        fix_task  = (
                            f"A previous agent ({step.agent}) failed to complete this task.\n"
                            f"Original task: {step.task}\n\n"
                            f"Error history (most recent first):\n{chain_ctx}"
                            f"{web_hints}\n\n"
                            "Please provide a correct solution using your full reasoning capability."
                        )
                    else:
                        fix_agent = step.agent
                        # Reformulate from scratch via LLM, enriched with web search hints
                        web_hints = await self._research_fix(step.task, error_text)
                        try:
                            rephrase_messages = [
                                SystemMessage(content=(
                                    "A specialist agent repeatedly failed a subtask. "
                                    "Reformulate the task in a completely different way that avoids the known failure mode. "
                                    "Output only the reformulated task description — no preamble."
                                )),
                                HumanMessage(content=(
                                    f"Original task: {step.task}\n"
                                    f"Errors so far:\n" +
                                    "\n".join(f"  - {e}" for e in error_chain[-3:]) +
                                    web_hints
                                )),
                            ]
                            resp     = await self.llm_invoke(rephrase_messages)
                            fix_task = resp.content.strip()
                        except Exception:
                            fix_task = step.task  # fallback to original

                    fix_step = PlanStep(
                        step_id=str(uuid.uuid4()),
                        phase=next_phase,
                        task=fix_task[:2000],
                        agent=fix_agent,
                        expected=step.expected,
                        depth=new_depth,
                        error_chain=error_chain,
                    )
                    plan.steps.append(fix_step)
                    fix_steps_added += 1
                    step.status = "fixed"   # mark original as superseded

                    await self.emit(
                        EventType.TASK_FIX_SPAWNED,
                        payload={
                            "original_step_id": step.step_id,
                            "fix_step_id":      fix_step.step_id,
                            "depth":            new_depth,
                            "agent":            fix_agent,
                            "strategy":         "same" if new_depth <= 3 else ("escalate" if new_depth <= 6 else "reformulate"),
                            "plan_id":          plan.plan_id,
                        },
                        target="broadcast",
                    )

                if fix_steps_added:
                    await self._emit_plan_status(
                        plan,
                        f"🔧 {fix_steps_added} fix subtask(s) spawned → phase {next_phase}",
                    )
                    plan._phases_advancing.discard(phase)
                    await self.memory.upsert_plan(plan.task_id, plan.plan_id, plan.original_task, "running", plan.to_dict())
                    await self._dispatch_phase(plan, next_phase)
                    return

            # All failed steps exhausted their fix budget — do one last web search then escalate
            exhausted = [s for s in failed if s.depth >= max_depth]
            error_lines = "\n".join(
                f"• [{s.agent}] {s.task[:80]}\n  "
                f"Final error (depth {s.depth}): {s.result[:200]}"
                for s in exhausted
            )
            # Search for each unique error; collect hints for the user
            web_sections: list[str] = []
            seen_queries: set[str] = set()
            for s in exhausted:
                hints = await self._research_fix(s.task, s.result[:300])
                # _research_fix returns "" or a string starting with "\n\n[Internet search..."
                # De-duplicate by query (first 60 chars of the bracketed label)
                if hints and hints[:80] not in seen_queries:
                    seen_queries.add(hints[:80])
                    web_sections.append(hints.strip())
            web_block = ("\n\n**Possible fixes found online:**\n" + "\n".join(web_sections)) if web_sections else ""
            reply = (
                f"❌ Could not complete after {max_depth} fix attempt(s).\n\n"
                f"Failed steps:\n{error_lines}"
                f"{web_block}\n\n"
                "Please review the hints above or provide more specific instructions."
            )
            await self._emit_plan_status(
                plan, f"❌ Escalating to user — fix depth {max_depth} exhausted"
            )
            await self._publish_reply(
                reply, plan.task_id, plan.discord_message_id, original_task=plan.original_task
            )
            self._plans.pop(plan.task_id, None)
            await self._close_task_context(plan, success=False)
            await self.memory.upsert_plan(plan.task_id, plan.plan_id, plan.original_task, "failed", plan.to_dict())
        else:
            # Phase succeeded
            next_phase = phase + 1
            if next_phase <= plan.max_phase():
                await self._emit_plan_status(plan, f"✅ Phase {phase} complete → phase {next_phase}")
                await self.memory.upsert_plan(plan.task_id, plan.plan_id, plan.original_task, "running", plan.to_dict())
                await self._dispatch_phase(plan, next_phase)
            else:
                # All phases done — synthesise final reply
                await self._emit_plan_status(plan, "✅ All phases complete")
                all_results = [
                    (s.agent, s.result)
                    for s in plan.steps
                    if s.status == "done" and s.result
                ]
                await self._synthesise_and_reply(
                    original_task=plan.original_task,
                    raw_results=all_results or [("orchestrator", "Task completed.")],
                    task_id=plan.task_id,
                    context=plan.context,
                    discord_message_id=plan.discord_message_id,
                )
                self._plans.pop(plan.task_id, None)
                await self._close_task_context(plan, success=True)
                await self.memory.upsert_plan(plan.task_id, plan.plan_id, plan.original_task, "completed", plan.to_dict())

    # ── Context lifecycle ────────────────────────────────────────────────────

    async def _close_task_context(self, plan: ExecutionPlan, success: bool) -> None:
        """
        Archive the task context stream to long-term memory.
        Called on plan completion (success or failure).

        Value scoring:
          - Successful multi-step plans score highest
          - Failed tasks still score >0 (failure cases are worth remembering)
          - Single-step trivial tasks score low
        """
        steps_done   = sum(1 for s in plan.steps if s.status == "done")
        steps_total  = len(plan.steps)
        value_score  = (steps_done / max(steps_total, 1)) * (0.9 if success else 0.4)
        if steps_total > 1:
            value_score = min(1.0, value_score + 0.2)

        # Summarise step outcomes
        step_lines = "\n".join(
            f"  {'✅' if s.status == 'done' else '❌'} [{s.agent}] {s.task[:80]}"
            for s in plan.steps
        )
        summary = (
            f"{'✅' if success else '❌'} Task: {plan.original_task[:200]}\n"
            f"Steps:\n{step_lines}"
        )

        await self.memory.close_context_snapshot(
            plan.context_id or plan.task_id,
            summary=summary,
            snapshot_json=plan.to_dict(),
            value_score=value_score,
        )
        await self.bus.close_context(plan.context_id or plan.task_id)
        await self.emit(
            EventType.CONTEXT_CLOSED,
            payload={
                "context_id":  plan.context_id or plan.task_id,
                "task_id":     plan.task_id,
                "success":     success,
                "value_score": round(value_score, 2),
                "summary":     summary[:300],
            },
            target="broadcast",
        )
        log.info(
            "orchestrator.task_context_closed",
            task_id=plan.task_id[:8],
            success=success,
            value_score=round(value_score, 2),
        )

    # ── Status events ────────────────────────────────────────────────────────

    async def _emit_plan_status(self, plan: ExecutionPlan, message: str) -> None:
        """Broadcast a plan status update. Bridge routes these to the appropriate channel."""
        await self.emit(
            EventType.PLAN_STATUS,
            payload={
                "message": message,
                "plan_id": plan.plan_id,
                "task_id": plan.task_id,
                "retry_count": plan.retry_count,
                "original_task": plan.original_task[:120],
            },
            target="broadcast",
        )
        log.info("orchestrator.plan_status", message=message[:120])

    # ── On-demand agent lifecycle ─────────────────────────────────────────────

    _EPHEMERAL_AGENTS = {"document_qa", "code_search", "executor", "claude_code_agent", "research"}

    # Maps agent role name → Docker container_name (for docker start/stop).
    # Defaults to "agent_{role}" for anything not listed here.
    _CONTAINER_NAME = {
        "claude_code_agent": "agent_claude_code",
    }

    async def _ensure_agent_running(self, agent: str, plan: "ExecutionPlan | None" = None) -> bool:
        """Start an ephemeral agent container if it isn't already running.
        Uses `docker start` — no compose plugin required.
        Returns True on success, False if the container could not be started."""
        if agent not in self._EPHEMERAL_AGENTS:
            return True
        container = self._CONTAINER_NAME.get(agent, f"agent_{agent}")
        if plan:
            await self._emit_plan_status(plan, f"🚀 Starting `{container}`…")
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "start", container,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            if proc.returncode != 0:
                err = stderr.decode().strip()
                # Container doesn't exist yet (never been run) — fall back to compose up
                if "No such container" in err:
                    log.info("orchestrator.agent_compose_up", agent=agent, container=container)
                    compose_proc = await asyncio.create_subprocess_exec(
                        "docker", "compose", "up", "-d", "--no-deps", agent,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=self.settings.compose_project_dir,
                    )
                    _, compose_err = await asyncio.wait_for(compose_proc.communicate(), timeout=120)
                    if compose_proc.returncode == 0:
                        log.info("orchestrator.agent_created_and_started", agent=agent, container=container)
                        if plan:
                            await self._emit_plan_status(plan, f"✅ `{container}` created and started")
                        return True
                    err = compose_err.decode().strip()[:200]
                log.error("orchestrator.agent_start_failed", agent=agent, container=container, error=err[:200])
                if plan:
                    await self._emit_plan_status(plan, f"❌ Failed to start `{container}`: {err[:200]}")
                return False
            log.info("orchestrator.agent_started", agent=agent, container=container)
            if plan:
                await self._emit_plan_status(plan, f"✅ `{container}` started")
            return True
        except asyncio.TimeoutError:
            log.error("orchestrator.agent_start_timeout", agent=agent, container=container)
            if plan:
                await self._emit_plan_status(plan, f"❌ Timed out starting `{container}` (60s)")
            return False
        except Exception as exc:
            log.error("orchestrator.agent_start_error", agent=agent, error=str(exc))
            if plan:
                await self._emit_plan_status(plan, f"❌ Error starting `{container}`: {exc}")
            return False

    # ── Discord action dispatch ───────────────────────────────────────────────

    async def _execute_discord_subtask(self, task: str, task_id: str) -> int:
        """
        Convert a natural-language Discord task into discord.action events.
        Returns number of actions emitted.
        """
        actions = _parse_discord_actions(task)

        if actions is None:
            log.info("orchestrator.discord_action_llm_fallback", task=task[:80])
            messages = [
                SystemMessage(content=(
                    "Translate a Discord management request into a JSON array of action objects. No markdown. Output ONLY valid JSON.\n"
                    "Valid actions: send_message, send_file, create_channel, delete_channel, rename_channel, "
                    "set_topic, create_category, pin_message, list_channels, find_and_delete_duplicates\n"
                    "Rules:\n"
                    "- send_message MUST have a non-empty 'content' field with the literal text to send. NEVER emit send_message without concrete content.\n"
                    "- send_file sends a file from /workspace to Discord. Required field: file_path (absolute path). Optional: channel_name or channel_id, content (caption).\n"
                    "- To read/list channels use list_channels only.\n"
                    "- To set a channel description use set_topic with a 'topic' field.\n"
                    "- Do NOT combine list_channels with send_message in the same action array.\n"
                    'Example: [{"action":"send_file","file_path":"/workspace/docs/report.pdf","channel_name":"agent-tasks","content":"Here is the report"}]'
                )),
                HumanMessage(content=f"Discord task: {task}"),
            ]
            response = await self.llm_invoke(messages)
            raw = response.content.strip()
            if "```" in raw:
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else parts[0]
                if raw.lower().startswith("json"):
                    raw = raw[4:]
            try:
                parsed = json.loads(raw.strip())
                actions = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                log.warning("orchestrator.discord_action_parse_failed", task=task[:100])
                return 0

        emitted = 0
        for action_payload in actions:
            if not isinstance(action_payload, dict) or "action" not in action_payload:
                continue
            # Skip send_message actions with no content — they'd fail in the bridge
            # and break the whole plan (LLMs sometimes emit these speculatively).
            if action_payload.get("action") == "send_message" and not action_payload.get("content", "").strip():
                log.warning("orchestrator.discord_action_skipped_empty_content", action=action_payload)
                continue
            action_payload["task_id"] = task_id
            await self.emit(EventType.DISCORD_ACTION, payload=action_payload, target="broadcast")
            log.info("orchestrator.discord_action_emitted", action=action_payload.get("action"))
            emitted += 1
        return emitted

    # ── Reply synthesis ──────────────────────────────────────────────────────

    async def _synthesise_and_reply(
        self,
        original_task: str,
        raw_results: list[tuple[str, str]],
        task_id: str,
        context: str = "",
        discord_message_id: str | None = None,
    ) -> None:
        """Pass-through for single clean results; LLM synthesis for command output or multiple."""
        if len(raw_results) == 1:
            source, result = raw_results[0]
            result = result.strip()
            if not result.startswith("Exit code:") and len(result) <= 1800:
                await self._publish_reply(result, task_id, discord_message_id, original_task=original_task)
                return

        summary_system = (
            "You are a concise assistant. Summarise the gathered information into a clear reply. "
            "Interpret command output plainly. If nothing useful was found, say so honestly."
        )
        fixed = f"Task: {original_task}\n\nInformation gathered:\n"
        budget = await self._budget_content_chars(summary_system, fixed)
        raw_combined = "\n\n".join(f"[{src}]: {res}" for src, res in raw_results)
        combined = raw_combined[:budget]
        messages = [
            SystemMessage(content=summary_system),
            HumanMessage(content=fixed + combined),
        ]
        response = await self.llm_invoke(messages)
        await self._publish_reply(response.content, task_id, discord_message_id, original_task=original_task)

    async def _publish_reply(
        self,
        result: str,
        task_id: str,
        discord_message_id: str | None = None,
        original_task: str = "",
    ) -> None:
        payload = {"result": result, "task_id": task_id}
        if discord_message_id:
            payload["discord_message_id"] = discord_message_id
        await self.emit(EventType.TASK_COMPLETED, payload=payload, target="broadcast")
        if original_task:
            now = time.time()
            # Reset context window if the user has been idle for > 15 minutes
            if self._conversation and (now - self._conversation[-1][2]) > self._CONVERSATION_TIMEOUT:
                self._conversation.clear()
                log.info("orchestrator.conversation_reset", reason="idle_timeout")
            self._conversation.append((original_task, result[:300], now))

    # ── Proactive think loop ─────────────────────────────────────────────────

    async def think(self) -> None:
        """DB queue drain + stale chat session cleanup."""
        now = time.time()

        # Close chat sessions that have gone idle
        idle_gap = float(await self.bus.get_config("chat_idle_gap_secs", 1800))
        for session in list(self._chat_sessions.values()):
            if now - session.last_activity > idle_gap:
                log.info("orchestrator.chat_session_idle_close", session_id=session.session_id[:8])
                await self._close_chat_session(session)
        stale = [
            tid for tid, p in list(self._plans.items())
            if now - p.created_at > self.PLAN_TIMEOUT
        ]
        for tid in stale:
            plan = self._plans.pop(tid)
            log.warning("orchestrator.plan_expired", task_id=tid, task=plan.original_task[:60])
            await self.memory.upsert_plan(tid, plan.plan_id, plan.original_task, "expired", plan.to_dict())
            running_steps = [s for s in plan.steps if s.status == "running"]
            await self._emit_plan_status(
                plan,
                f"⌛ Plan expired after {self.PLAN_TIMEOUT}s — "
                f"{len(running_steps)} step(s) still running: "
                + ", ".join(f"`{s.agent}`" for s in running_steps),
            )

        await self.memory.expire_plans()

        # Purge completed/failed tasks, resolved plans, and old handoffs older
        # than 24 h. Keeps the DB lean without manual intervention.
        cleaned = await self.memory.cleanup_stale(max_age_hours=24)
        if any(cleaned.values()):
            log.info("orchestrator.stale_cleanup", **cleaned)

        # ── Architecture change detection ─────────────────────────────────────
        changed = await self._detect_arch_changes()
        if changed:
            await self._dispatch_arch_review(changed)

        pending = await self.memory.get_pending_tasks(limit=1)
        if not pending:
            log.debug("orchestrator.think_idle")
            return

        row     = pending[0]
        claimed = await self.memory.claim_task(row["id"])
        if not claimed:
            return

        task_text = row["task"]
        task_id   = str(uuid.uuid4())
        log.info("orchestrator.think_dequeued", db_task_id=row["id"], task=task_text[:80])

        await self.emit(
            EventType.THINK_CYCLE,
            payload={"task": task_text[:300], "task_id": task_id, "db_task_id": row["id"]},
            target="broadcast",
        )
        try:
            await self._run_task(task_text, task_id)
            await self.memory.complete_task(row["id"])
        except Exception as exc:
            log.error("orchestrator.think_task_failed", error=str(exc))
            await self.memory.fail_task(row["id"])

    # ── Architecture change watcher ───────────────────────────────────────────

    async def _detect_arch_changes(self) -> list[str]:
        """
        SHA256-hash each WATCHED_PATHS entry, compare to values stored in
        the Redis hash _ARCH_HASH_KEY.

        Returns a list of changed (or new) file paths.
        On the very first run (no stored hashes) just baselines and returns []
        so we don't trigger a review on every fresh container start.
        """
        current: dict[str, str] = {}
        for path_str in WATCHED_PATHS:
            p = Path(path_str)
            if not p.exists():
                continue
            try:
                current[path_str] = hashlib.sha256(p.read_bytes()).hexdigest()
            except Exception:
                pass

        if not current:
            return []

        stored_raw = await self.bus._client.hgetall(_ARCH_HASH_KEY)
        stored: dict[str, str] = {
            (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
            for k, v in stored_raw.items()
        }

        # Persist current hashes regardless of outcome
        await self.bus._client.hset(_ARCH_HASH_KEY, mapping=current)

        if not stored:
            log.info("orchestrator.arch_watcher_baselined", files=len(current))
            return []

        return [p for p, h in current.items() if stored.get(p) != h]

    async def _dispatch_arch_review(self, changed: list[str]) -> None:
        """
        Two-tier documentation strategy:

        MINOR UPDATE (default, every think cycle a change is detected):
          Write a timestamped entry to CHANGELOG.md directly — no agent spin-up,
          no LLM call, no LaTeX compilation.

        FULL BUILD (at most once per _FULL_BUILD_INTERVAL, default 24 h):
          Spin up document_qa for the full LLM architecture review → LaTeX → PDF
          → Google Drive upload.  The daily build incorporates everything that
          accumulated in the changelog since the last build.
        """
        short_names = [Path(p).name for p in changed]

        last_raw = await self.bus._client.get(_ARCH_LAST_FULL_BUILD)
        last_build = float(last_raw) if last_raw else 0.0
        due_for_full_build = (time.time() - last_build) >= _FULL_BUILD_INTERVAL

        # Always write the changelog entry (cheap, always useful)
        await self._write_arch_changelog(short_names)

        if not due_for_full_build:
            log.info("orchestrator.arch_minor_update", files=short_names,
                     next_full_build_in_h=round((_FULL_BUILD_INTERVAL - (time.time() - last_build)) / 3600, 1))
            return

        # ── Full build ────────────────────────────────────────────────────────
        log.info("orchestrator.arch_full_build_triggered", files=short_names)
        # Mark the build time before dispatching so a crash doesn't cause a
        # rapid retry storm on the next think cycle.
        await self.bus._client.set(_ARCH_LAST_FULL_BUILD, str(time.time()))

        task_id   = str(uuid.uuid4())
        task_desc = (
            "Full architecture review triggered by accumulated source changes. "
            "Recent changes include: " + ", ".join(short_names[:8]) + ". "
            "Generate updated architecture documentation and upload to Google Drive."
        )
        plan = ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            task_id=task_id,
            original_task=task_desc,
            steps=[PlanStep(
                step_id=str(uuid.uuid4()),
                phase=1,
                task=task_desc,
                agent="document_qa",
                expected="architecture PDF compiled and synced to Drive",
            )],
        )
        self._plans[task_id] = plan
        await self.memory.upsert_plan(
            task_id, plan.plan_id, task_desc, "running", plan.to_dict()
        )
        await self._emit_plan_status(
            plan, f"📐 Daily arch build — spinning up document_qa: {', '.join(short_names[:5])}"
        )
        await self._dispatch_phase(plan, 1)

    async def _write_arch_changelog(self, short_names: list[str]) -> None:
        """
        Append a one-line timestamped entry to CHANGELOG.md.
        Pure file I/O — no LLM, no agent spin-up.
        """
        try:
            _ARCH_CHANGELOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            ts    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            entry = f"- `{ts}` — changed: {', '.join(short_names)}\n"
            with _ARCH_CHANGELOG_PATH.open("a") as f:
                f.write(entry)
            log.info("orchestrator.arch_changelog_appended", files=short_names)
        except Exception as exc:
            log.warning("orchestrator.arch_changelog_failed", error=str(exc))

    async def on_shutdown(self) -> None:
        log.info("orchestrator.shutdown", active_plans=len(self._plans))


if __name__ == "__main__":
    settings = Settings()
    agent = OrchestratorAgent(settings)
    run_agent(agent)
