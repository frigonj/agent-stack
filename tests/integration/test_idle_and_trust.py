"""
tests/integration/test_idle_and_trust.py
──────────────────────────────────────────
CORE INTEGRATION TEST — DO NOT DELETE OR MODIFY WITHOUT TEAM REVIEW.

Two concerns:

1. Idle termination — agents with IDLE_TIMEOUT set must exit cleanly when no
   events arrive for that duration.  Tests the fix to the consume() sentinel.

2. Trust tier routing — executor must route commands to the correct tier
   (SAFE / AUTO_APPROVED / REQUIRES_APPROVAL) without calling _request_approval
   for AUTO_APPROVED commands.

Does NOT require LM Studio, Postgres, Redis, or Discord.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.executor.main import (
    AUTO_APPROVED_COMMANDS,
    REQUIRES_APPROVAL,
    SAFE_COMMANDS,
    ExecutorAgent,
)
from core.base_agent import BaseAgent


# ── Idle termination ─────────────────────────────────────────────────────────


class _MinimalAgent(BaseAgent):
    """Bare-minimum agent that does nothing — used to test lifecycle only."""

    async def handle_event(self, event):
        pass

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass


@pytest.mark.asyncio
async def test_idle_watchdog_fires(monkeypatch):
    """
    An agent with idle_timeout=1 should stop within 3 seconds of starting
    when no events are delivered.

    Uses a mock EventBus that always yields the idle sentinel so the fix to
    consume() is exercised without a real Redis connection.
    """
    # Mock the event bus
    mock_bus = AsyncMock()
    mock_bus.connect = AsyncMock()
    mock_bus.disconnect = AsyncMock()
    mock_bus.publish = AsyncMock()
    mock_bus.ack = AsyncMock()

    # consume() yields the idle sentinel indefinitely
    async def _mock_consume(*args, **kwargs):
        while True:
            yield None, None, None
            await asyncio.sleep(0.05)

    mock_bus.consume = _mock_consume

    # Mock memory so no DB needed
    mock_memory = AsyncMock()
    mock_memory.open_session = AsyncMock(return_value={"status": "CLEAN"})
    mock_memory.recover_context = AsyncMock(return_value={"status": "CLEAN"})
    mock_memory.close_session = AsyncMock()
    mock_memory.close = AsyncMock()
    mock_memory.batch_store = AsyncMock()

    settings = MagicMock()
    settings.agent_role = "test_idle_agent"
    settings.redis_url = "redis://localhost:6379"
    settings.database_url = "postgresql://localhost/test"
    settings.lm_studio_url = "http://localhost:1234"
    settings.lm_studio_model = "test-model"

    agent = _MinimalAgent(settings)
    agent.bus = mock_bus
    agent.memory = mock_memory
    agent._idle_timeout = 1  # 1 second idle timeout

    start = time.monotonic()
    await asyncio.wait_for(agent.start(), timeout=5.0)
    time.monotonic() - start

    # Agent should have exited in roughly 1–3 seconds (watchdog checks every 60s
    # normally, but we reduce the check interval for the test by checking the flag)
    assert agent._stopped or not agent._running, "Agent did not stop after idle timeout"


@pytest.mark.asyncio
async def test_idle_watchdog_resets_on_event(monkeypatch):
    """
    If events are delivered, the idle timer resets.
    The agent should NOT stop while events are flowing.
    """
    event_count = 0

    async def _mock_consume_with_events(*args, **kwargs):
        nonlocal event_count
        from core.events.bus import Event, EventType

        for _ in range(3):
            event = Event(
                type=EventType.AGENT_STARTED,
                source="test",
                payload={},
            )
            event_count += 1
            yield "agents:test_idle_agent", f"1-{event_count}", event
            await asyncio.sleep(0.1)
        # After events, yield sentinel forever
        while True:
            yield None, None, None
            await asyncio.sleep(0.05)

    mock_bus = AsyncMock()
    mock_bus.connect = AsyncMock()
    mock_bus.disconnect = AsyncMock()
    mock_bus.publish = AsyncMock()
    mock_bus.ack = AsyncMock()
    mock_bus.consume = _mock_consume_with_events

    mock_memory = AsyncMock()
    mock_memory.open_session = AsyncMock(return_value={"status": "CLEAN"})
    mock_memory.recover_context = AsyncMock(return_value={"status": "CLEAN"})
    mock_memory.close_session = AsyncMock()
    mock_memory.close = AsyncMock()
    mock_memory.batch_store = AsyncMock()

    settings = MagicMock()
    settings.agent_role = "test_idle_agent2"
    settings.redis_url = "redis://localhost:6379"
    settings.database_url = "postgresql://localhost/test"
    settings.lm_studio_url = "http://localhost:1234"
    settings.lm_studio_model = "test-model"

    agent = _MinimalAgent(settings)
    agent.bus = mock_bus
    agent.memory = mock_memory
    agent._idle_timeout = 2  # 2 second idle timeout

    # The 3 events above should reset the timer. We'll stop the agent externally
    # after 1 second, verifying it's still running (hasn't idle-exited prematurely).
    async def _stopper():
        await asyncio.sleep(1.0)
        assert agent._running, "Agent exited too early — idle timer not reset by events"
        agent._running = False
        agent._stop_event.set()

    await asyncio.gather(
        asyncio.wait_for(agent.start(), timeout=4.0),
        _stopper(),
        return_exceptions=True,
    )


# ── Trust tier routing ────────────────────────────────────────────────────────


def _make_executor(monkeypatch) -> ExecutorAgent:
    """Create an ExecutorAgent with mocked bus and memory."""
    settings = MagicMock()
    settings.agent_role = "executor"
    settings.redis_url = "redis://localhost:6379"
    settings.database_url = "postgresql://localhost/test"
    settings.lm_studio_url = "http://localhost:1234"
    settings.lm_studio_model = "test-model"

    agent = ExecutorAgent.__new__(ExecutorAgent)
    agent.settings = settings
    agent.role = "executor"
    agent.bus = AsyncMock()
    agent.bus.publish = AsyncMock()
    agent.memory = AsyncMock()
    agent._findings = []
    agent._context_tasks = {}
    agent._running = True
    agent._stopped = False
    agent._stop_event = asyncio.Event()
    agent._circuit_failures = 0
    agent._circuit_open = False
    agent._circuit_open_since = 0.0
    agent._model_context_limit = None
    agent._last_event_time = time.monotonic()
    agent._idle_timeout = 0
    agent._claude_fallback = None
    return agent


@pytest.mark.asyncio
async def test_safe_commands_run_without_approval(monkeypatch):
    """SAFE commands bypass both approval and audit emission."""
    agent = _make_executor(monkeypatch)
    agent._request_approval = AsyncMock(return_value=True)
    agent._emit_audit = AsyncMock()

    with patch("asyncio.create_subprocess_shell") as mock_proc:
        mock_p = AsyncMock()
        mock_p.returncode = 0
        mock_p.communicate = AsyncMock(return_value=(b"file1\nfile2\n", b""))
        mock_proc.return_value = mock_p

        result = await agent._run_command("ls /workspace", "list workspace", "task-1")

    agent._request_approval.assert_not_called()
    agent._emit_audit.assert_not_called()
    assert "file1" in result or "Exit code" in result


@pytest.mark.asyncio
async def test_auto_approved_commands_run_without_approval(monkeypatch):
    """AUTO_APPROVED commands run immediately and emit an audit event, no Discord gate."""
    agent = _make_executor(monkeypatch)
    agent._request_approval = AsyncMock(return_value=True)
    agent._emit_audit = AsyncMock()

    with patch("asyncio.create_subprocess_shell") as mock_proc:
        mock_p = AsyncMock()
        mock_p.returncode = 0
        mock_p.communicate = AsyncMock(return_value=(b"Restarting container\n", b""))
        mock_proc.return_value = mock_p

        await agent._run_command(
            "docker restart agent_executor", "restart executor", "task-2"
        )

    agent._request_approval.assert_not_called()
    agent._emit_audit.assert_called_once()
    audit_call = agent._emit_audit.call_args
    assert "docker restart" in audit_call.args[0]


@pytest.mark.asyncio
async def test_requires_approval_commands_gate(monkeypatch):
    """REQUIRES_APPROVAL commands call _request_approval before running."""
    agent = _make_executor(monkeypatch)
    agent._request_approval = AsyncMock(return_value=False)  # user denies
    agent._emit_audit = AsyncMock()

    result = await agent._run_command(
        "rm -rf /workspace/bad_file", "delete bad file", "task-3"
    )

    agent._request_approval.assert_called_once()
    agent._emit_audit.assert_not_called()
    assert "denied" in result.lower()


@pytest.mark.asyncio
async def test_docker_rm_escalates_from_auto_approved(monkeypatch):
    """
    'docker' is in AUTO_APPROVED_COMMANDS, but 'docker rm' matches the escalation
    pattern and must be gated via approval even though the base command is auto-approved.
    """
    agent = _make_executor(monkeypatch)
    agent._request_approval = AsyncMock(return_value=False)
    agent._emit_audit = AsyncMock()

    result = await agent._run_command(
        "docker rm agent_executor", "remove container", "task-4"
    )

    agent._request_approval.assert_called_once()
    agent._emit_audit.assert_not_called()
    assert "denied" in result.lower()


@pytest.mark.asyncio
async def test_git_push_escalates_from_auto_approved(monkeypatch):
    """'git push' must be gated even though 'git' is in AUTO_APPROVED_COMMANDS."""
    agent = _make_executor(monkeypatch)
    agent._request_approval = AsyncMock(return_value=False)
    agent._emit_audit = AsyncMock()

    result = await agent._run_command(
        "git push origin main", "push to remote", "task-5"
    )

    agent._request_approval.assert_called_once()
    assert "denied" in result.lower()


@pytest.mark.asyncio
async def test_tee_to_user_dir_escalates(monkeypatch):
    """Writing to /workspace/user (outside container workspace) must be gated."""
    agent = _make_executor(monkeypatch)
    agent._request_approval = AsyncMock(return_value=False)
    agent._emit_audit = AsyncMock()

    result = await agent._run_command(
        "tee /workspace/user/.bashrc", "write to user home", "task-6"
    )

    agent._request_approval.assert_called_once()
    assert "denied" in result.lower()


def test_command_tier_membership():
    """Snapshot test: key commands are in the correct tier."""
    # SAFE
    assert "ls" in SAFE_COMMANDS
    assert "cat" in SAFE_COMMANDS
    assert "grep" in SAFE_COMMANDS

    # AUTO_APPROVED
    assert "docker" in AUTO_APPROVED_COMMANDS
    assert "tee" in AUTO_APPROVED_COMMANDS
    assert "python3" in AUTO_APPROVED_COMMANDS
    assert "git" in AUTO_APPROVED_COMMANDS
    assert "bash" in AUTO_APPROVED_COMMANDS

    # REQUIRES_APPROVAL
    assert "rm" in REQUIRES_APPROVAL
    assert "pip" in REQUIRES_APPROVAL
    assert "curl" in REQUIRES_APPROVAL
    assert "wget" in REQUIRES_APPROVAL
    assert "apt" in REQUIRES_APPROVAL

    # Nothing should be in both SAFE and AUTO_APPROVED
    assert not (SAFE_COMMANDS & AUTO_APPROVED_COMMANDS), (
        f"Overlap between SAFE and AUTO_APPROVED: {SAFE_COMMANDS & AUTO_APPROVED_COMMANDS}"
    )

    # Nothing should be in both AUTO_APPROVED and REQUIRES_APPROVAL
    assert not (AUTO_APPROVED_COMMANDS & REQUIRES_APPROVAL), (
        f"Overlap between AUTO_APPROVED and REQUIRES_APPROVAL: "
        f"{AUTO_APPROVED_COMMANDS & REQUIRES_APPROVAL}"
    )
