"""
agents/discord_bridge/main.py
──────────────────────────────
Discord ↔ Redis Streams bridge.

One bot, not per-agent accounts. Agent identity is shown via embed
color and icon. The bot does three things:

  1. Inbound (#agent-tasks):  Messages → task.created on agents:orchestrator
  2. Inbound (#claude):       Messages (human only) → task.created on agents:claude_code
  3. Inbound (#control):      Restart commands → HTTP POST to host_restart_helper
  4. Outbound: broadcast stream events → formatted Discord embeds

Access control for #claude:
  - Only human Discord users may post tasks directly to #claude.
  - Agents that need Claude's help emit a `claude.escalation` event to
    agents:broadcast. The bridge posts an approval embed in #claude.
    The user approves → task routed to agents:claude_code.
    The user denies  → `claude.escalation.denied` acked back to broadcast.

Approval gates:
  - approval.required   → embed with ✅/❌ buttons (5 min timeout)
    Button click → sets Redis key `approval:{id}` → unblocks executor
  - claude.escalation   → embed with ✅/❌ buttons (5 min timeout)
    On approve → pushes task to agents:claude_code
    On deny    → emits claude.escalation.denied to broadcast

claude.escalation event payload fields:
  task           — the task text to send to Claude (required)
  reason         — why the agent is escalating (shown in embed)
  source_task_id — originating task id (echoed back in denial event)
  escalation_id  — unique id for this escalation request

Environment variables:
  DISCORD_BOT_TOKEN                 — required
  DISCORD_TASK_CHANNEL_ID           — required (#agent-tasks)
  DISCORD_CLAUDE_CHANNEL_ID         — optional (#claude → Claude API agent)
  DISCORD_CONTROL_CHANNEL_ID        — optional (#control → restart commands)
  DISCORD_USER_APPROVALS_CHANNEL_ID — optional (#user-approvals → user-escalated votes)
  DISCORD_KNOWLEDGE_GAP_CHANNEL_ID  — optional (#knowledge-gap → agent knowledge gap escalations)
  CONTROL_HELPER_URL                — optional (default http://host.docker.internal:7799)
  DISCORD_GUILD_ID                  — optional (auto-detected)
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import time
import uuid

import httpx

import discord
import redis.asyncio as aioredis
import structlog

log = structlog.get_logger()


# ── Local message classification (no LLM) ────────────────────────────────────
# Keeps the Discord bridge thin — most work stays in the orchestrator.

_TASK_RE = re.compile(
    r"\b(run|execute|create|delete|update|find|search|list|show|install|restart|"
    r"fix|build|deploy|generate|write|edit|modify|refactor|analyse|analyze|"
    r"move|rename|copy|check|test|debug|grep|cat|ls|docker|git|pip|npm)\b",
    re.I,
)
_CHAT_RE = re.compile(
    r"^(hi|hello|hey|good\s+\w+|thanks?|thank you|ok|okay|got it|"
    r"what|how|why|when|who|is |are |can |could |would |should |"
    r"tell me|explain|status|ping|what can you|are you)",
    re.I,
)


def _classify_message_locally(text: str) -> str:
    """
    Fast keyword-based intent classification for the bridge.
    Returns 'task' or 'chat'.  The orchestrator refines this with context.
    """
    t = text.strip()
    if _CHAT_RE.match(t) and not _TASK_RE.search(t):
        return "chat"
    if _TASK_RE.search(t):
        return "task"
    return "chat"  # default to chat so ambiguous messages don't create spurious plans


def _extract_keywords(text: str) -> list[str]:
    """Return meaningful words from a message for session matching."""
    words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
    return [w for w in words if len(w) > 2]


# ── Visual identity per agent ─────────────────────────────────────────────────

AGENT_COLOR = {
    "orchestrator": discord.Color.blue(),
    "document_qa": discord.Color.green(),
    "code_search": discord.Color.orange(),
    "executor": discord.Color.gold(),
    "discord": discord.Color.blurple(),
    "claude_code_agent": discord.Color.from_rgb(138, 43, 226),  # purple
}

AGENT_ICON = {
    "orchestrator": "🧠",
    "document_qa": "📄",
    "code_search": "🔍",
    "executor": "⚙️",
    "claude_code_agent": "✨",
}

# ── Control channel command config ────────────────────────────────────────────

_KNOWN_SERVICES = {
    "lm-studio",
    "all",
    "orchestrator",
    "executor",
    "code-search",
    "document-qa",
    "discord",
    "claude",
    "redis",
    "postgres",
}

_CONTROL_HELP = (
    "**Control commands:**\n"
    "`restart lm-studio` — restart LM Studio on the host\n"
    "`restart <service>` — restart a Docker container\n"
    "`restart all` — restart all agent containers\n"
    "`status` — show running containers\n"
    "`relaunch code` — restart always-on services, no rebuild\n"
    "`relaunch rebuild` — rebuild all images, restart always-on services\n"
    "`relaunch rebuild <svc>` — rebuild and restart a single service\n"
    "`relaunch flush` — rebuild + clear Redis streams (Postgres preserved)\n"
    "`relaunch reset` — **destructive** full reset, destroys all data\n"
    "`build docs` — force a full architecture document rebuild now\n"
    "`verbose on` — forward all Redis events to #agent-logs\n"
    "`verbose off` — return to normal (only high-level events)\n"
    "`loglevel <DEBUG|INFO|WARNING|ERROR>` — change agent log level across all containers\n"
    "`reset session` — clear orchestrator conversation history and reload intents\n"
    "`pause <task_id>` — pause a running research task after its current iteration\n"
    "`resume <task_id>` — resume a paused research task from its last checkpoint\n\n"
    f"Services: {', '.join(sorted(_KNOWN_SERVICES))}"
)

# Event types we surface to Discord (others are silently ack'd)
VISIBLE_EVENTS = {
    "task.completed",
    "task.created",
    "task.failed",
    "approval.required",
    "claude.escalation",
    "system.error",
    "plan.proposed",
    "agent.vote",
    "vote.initiated",
    "vote.result",
    "vote.escalated_to_user",
    "context.closed",
    "agent.response",
    "discord.action",
    "memory.pruned",
    "plan.status",
    "knowledge.gap",
}

# Extra events forwarded only when verbose mode is active
# (Redis key  config:verbose_events = "1"  or Discord command "verbose on")
VERBOSE_EVENTS = {
    "task.created",
    "task.assigned",
    "task.failed",
    "agent.started",
    "agent.thinking",
    "agent.tool_call",
    "agent.tool_result",
    "memory.promoted",
    "context.created",
    "think.cycle",
    "task.spawned",
    "task.fix_spawned",
    "self.modify.proposed",
    "self.modify.applied",
}


# ── Approval gate UI ──────────────────────────────────────────────────────────


class ApprovalView(discord.ui.View):
    """Approve / Deny buttons for executor approval gates."""

    def __init__(self, approval_id: str, redis_client: aioredis.Redis):
        super().__init__(timeout=300)  # 5-minute window
        self.approval_id = approval_id
        self.redis = redis_client
        self._decided = False

    async def _resolve(self, interaction: discord.Interaction, decision: str) -> None:
        if self._decided:
            await interaction.response.send_message("Already decided.", ephemeral=True)
            return
        self._decided = True

        # Use set_approval() so the BLPOP waiter in bus.wait_for_approval() wakes
        # up immediately instead of waiting for the next poll interval.
        from core.events.bus import EventBus

        bus = EventBus.__new__(EventBus)
        bus._client = self.redis
        await bus.set_approval(self.approval_id, decision, ex=600)

        color = discord.Color.green() if decision == "approved" else discord.Color.red()
        label = "✅ Approved" if decision == "approved" else "❌ Denied"

        embed = interaction.message.embeds[0]
        embed.color = color
        embed.set_footer(text=f"{label} by {interaction.user.display_name}")

        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]

        await interaction.response.edit_message(embed=embed, view=self)
        self.stop()

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.success, emoji="✅")
    async def approve(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self._resolve(interaction, "approved")

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.danger, emoji="❌")
    async def deny(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self._resolve(interaction, "denied")

    async def on_timeout(self) -> None:
        # Auto-deny — executor's wait_for_approval will get "denied"
        if not self._decided:
            from core.events.bus import EventBus

            bus = EventBus.__new__(EventBus)
            bus._client = self.redis
            await bus.set_approval(self.approval_id, "denied", ex=60)


# ── Memory retention approval UI ─────────────────────────────────────────────


class MemoryApprovalView(discord.ui.View):
    """
    Approve / Deny buttons for memory retention requests (TTL > 24h).
    On resolution, publishes MEMORY_APPROVAL_RESULT to the event bus so the
    waiting coroutine in base_agent._request_memory_approval() can unblock.
    Timeout = 120s to match _MEMORY_APPROVAL_TIMEOUT in base_agent.
    """

    def __init__(self, approval_id: str, redis_client: aioredis.Redis):
        super().__init__(timeout=120)
        self.approval_id = approval_id
        self.redis = redis_client
        self._decided = False

    async def _resolve(self, interaction: discord.Interaction, approved: bool) -> None:
        if self._decided:
            await interaction.response.send_message("Already decided.", ephemeral=True)
            return
        self._decided = True

        from core.events.bus import Event, EventBus, EventType

        bus = EventBus.__new__(EventBus)
        bus._client = self.redis
        await bus.publish(
            Event(
                type=EventType.MEMORY_APPROVAL_RESULT,
                source="discord_bridge",
                payload={"approval_id": self.approval_id, "approved": approved},
            ),
            target="broadcast",
        )

        color = discord.Color.green() if approved else discord.Color.red()
        label = "✅ Approved" if approved else "❌ Denied"
        embed = interaction.message.embeds[0]
        embed.color = color
        embed.set_footer(text=f"{label} by {interaction.user.display_name}")
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        await interaction.response.edit_message(embed=embed, view=self)
        self.stop()

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.success, emoji="✅")
    async def approve(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self._resolve(interaction, approved=True)

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.danger, emoji="❌")
    async def deny(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self._resolve(interaction, approved=False)

    async def on_timeout(self) -> None:
        # Timeout = deny; the agent already defaults to "short" on timeout
        if not self._decided:
            self._decided = True


# ── Allowlist approval UI (one-time / permanent / deny) ─────────────────────


class AllowlistApprovalView(discord.ui.View):
    """
    Three-button approval gate for unknown-command allowlist requests.

    Approve once      → sends 'approved'          (command runs, not remembered)
    Approve permanent → sends 'approved_permanent' (command added to Redis allowlist)
    Deny              → sends 'denied'
    """

    def __init__(self, approval_id: str, redis_client: aioredis.Redis):
        super().__init__(timeout=300)
        self.approval_id = approval_id
        self.redis = redis_client
        self._decided = False

    async def _resolve(self, interaction: discord.Interaction, decision: str) -> None:
        if self._decided:
            await interaction.response.send_message("Already decided.", ephemeral=True)
            return
        self._decided = True

        # Defer immediately so Discord doesn't time out the interaction (3 s window).
        await interaction.response.defer()

        try:
            from core.events.bus import EventBus

            bus = EventBus.__new__(EventBus)
            bus._client = self.redis
            await bus.set_approval(self.approval_id, decision, ex=600)

            if decision == "approved_permanent":
                label = "♾️ Approved permanently"
                color = discord.Color.green()
            elif decision == "approved":
                label = "✅ Approved once"
                color = discord.Color.blurple()
            else:
                label = "❌ Denied"
                color = discord.Color.red()

            embed = interaction.message.embeds[0]
            embed.color = color
            embed.set_footer(text=f"{label} by {interaction.user.display_name}")

            for item in self.children:
                item.disabled = True  # type: ignore[attr-defined]

            await interaction.edit_original_response(embed=embed, view=self)
        except Exception:
            logger.exception(
                "AllowlistApprovalView._resolve failed (approval_id=%s)",
                self.approval_id,
            )
        finally:
            self.stop()

    @discord.ui.button(label="Once", style=discord.ButtonStyle.primary, emoji="✅")
    async def approve_once(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self._resolve(interaction, "approved")

    @discord.ui.button(label="Always", style=discord.ButtonStyle.success, emoji="♾️")
    async def approve_permanent(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self._resolve(interaction, "approved_permanent")

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.danger, emoji="❌")
    async def deny(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self._resolve(interaction, "denied")

    async def on_timeout(self) -> None:
        if not self._decided:
            from core.events.bus import EventBus

            bus = EventBus.__new__(EventBus)
            bus._client = self.redis
            await bus.set_approval(self.approval_id, "denied", ex=60)


# ── Claude escalation gate UI ────────────────────────────────────────────────


class ClaudeEscalationView(discord.ui.View):
    """
    Approve / Deny buttons for agent → Claude escalation requests.

    Approve → task pushed directly to agents:claude_code stream.
    Deny    → claude.escalation.denied event emitted to agents:broadcast
               so the requesting agent receives a clean failure.
    """

    def __init__(
        self,
        escalation_id: str,
        source_task_id: str,
        source: str,
        task: str,
        redis_client: aioredis.Redis,
    ):
        super().__init__(timeout=300)  # 5-minute window
        self.escalation_id = escalation_id
        self.source_task_id = source_task_id
        self.source = source
        self.task = task
        self.redis = redis_client
        self._decided = False

    async def _resolve(self, interaction: discord.Interaction, approved: bool) -> None:
        if self._decided:
            await interaction.response.send_message("Already decided.", ephemeral=True)
            return
        self._decided = True

        color = discord.Color.green() if approved else discord.Color.red()
        label = "✅ Approved" if approved else "❌ Denied"

        embed = interaction.message.embeds[0]
        embed.color = color
        embed.set_footer(text=f"{label} by {interaction.user.display_name}")
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        await interaction.response.edit_message(embed=embed, view=self)
        self.stop()

        if approved:
            # Route the task straight to the Claude agent
            task_id = str(uuid.uuid4())
            await self.redis.xadd(
                "agents:claude_code",
                {
                    "event_id": str(uuid.uuid4()),
                    "type": "claude.escalation",
                    "source": self.source,
                    "task_id": task_id,
                    "timestamp": str(time.time()),
                    "payload": json.dumps(
                        {
                            "task": self.task,
                            "escalation_id": self.escalation_id,
                            "source_task_id": self.source_task_id,
                            "approved_by": interaction.user.display_name,
                        }
                    ),
                },
            )
            log.info(
                "discord_bridge.claude_escalation_approved",
                escalation_id=self.escalation_id,
                task_id=task_id,
                approved_by=interaction.user.display_name,
            )
        else:
            # Ack the denial back to broadcast so the originating agent can handle it
            await self.redis.xadd(
                "agents:broadcast",
                {
                    "event_id": str(uuid.uuid4()),
                    "type": "claude.escalation.denied",
                    "source": "discord_bridge",
                    "task_id": self.source_task_id,
                    "timestamp": str(time.time()),
                    "payload": json.dumps(
                        {
                            "escalation_id": self.escalation_id,
                            "source_task_id": self.source_task_id,
                            "denied_by": interaction.user.display_name,
                        }
                    ),
                },
                maxlen=10_000,
                approximate=True,
            )
            log.info(
                "discord_bridge.claude_escalation_denied",
                escalation_id=self.escalation_id,
                denied_by=interaction.user.display_name,
            )

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.success, emoji="✅")
    async def approve(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self._resolve(interaction, approved=True)

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.danger, emoji="❌")
    async def deny(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self._resolve(interaction, approved=False)

    async def on_timeout(self) -> None:
        if not self._decided:
            self._decided = True
            await self.redis.xadd(
                "agents:broadcast",
                {
                    "event_id": str(uuid.uuid4()),
                    "type": "claude.escalation.denied",
                    "source": "discord_bridge",
                    "task_id": self.source_task_id,
                    "timestamp": str(time.time()),
                    "payload": json.dumps(
                        {
                            "escalation_id": self.escalation_id,
                            "source_task_id": self.source_task_id,
                            "denied_by": "timeout",
                        }
                    ),
                },
                maxlen=10_000,
                approximate=True,
            )
            log.info(
                "discord_bridge.claude_escalation_timeout",
                escalation_id=self.escalation_id,
            )


# ── User vote gate UI ─────────────────────────────────────────────────────────


class UserVoteView(discord.ui.View):
    """
    Approve / Reject buttons for votes escalated to the user when agent quorum
    was not met.  Auto-rejects on timeout (default 48 h).

    On resolution the view emits vote.user_result to agents:orchestrator so the
    waiting _escalate_vote_to_user coroutine can resolve its Future.
    """

    def __init__(
        self,
        vote_id: str,
        vote_type: str,
        redis_client: aioredis.Redis,
        timeout_seconds: float,
    ):
        super().__init__(timeout=timeout_seconds)
        self.vote_id = vote_id
        self.vote_type = vote_type
        self.redis = redis_client
        self._decided = False

    async def _resolve(self, interaction: discord.Interaction, approved: bool) -> None:
        if self._decided:
            await interaction.response.send_message("Already decided.", ephemeral=True)
            return
        self._decided = True

        color = discord.Color.green() if approved else discord.Color.red()
        label = "✅ Approved" if approved else "❌ Rejected"

        embed = interaction.message.embeds[0]
        embed.color = color
        embed.set_footer(text=f"{label} by {interaction.user.display_name}")
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        await interaction.response.edit_message(embed=embed, view=self)
        self.stop()

        await self.redis.xadd(
            "agents:orchestrator",
            {
                "event_id": str(uuid.uuid4()),
                "type": "vote.user_result",
                "source": "discord_bridge",
                "task_id": str(uuid.uuid4()),
                "timestamp": str(time.time()),
                "payload": json.dumps(
                    {
                        "vote_id": self.vote_id,
                        "approved": approved,
                        "decided_by": interaction.user.display_name,
                        "vote_type": self.vote_type,
                    }
                ),
            },
        )
        log.info(
            "discord_bridge.user_vote_resolved",
            vote_id=self.vote_id,
            approved=approved,
            decided_by=interaction.user.display_name,
        )

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.success, emoji="✅")
    async def approve(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self._resolve(interaction, approved=True)

    @discord.ui.button(label="Reject", style=discord.ButtonStyle.danger, emoji="❌")
    async def reject(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self._resolve(interaction, approved=False)

    async def on_timeout(self) -> None:
        if not self._decided:
            self._decided = True
            await self.redis.xadd(
                "agents:orchestrator",
                {
                    "event_id": str(uuid.uuid4()),
                    "type": "vote.user_result",
                    "source": "discord_bridge",
                    "task_id": str(uuid.uuid4()),
                    "timestamp": str(time.time()),
                    "payload": json.dumps(
                        {
                            "vote_id": self.vote_id,
                            "approved": False,
                            "decided_by": "timeout",
                            "vote_type": self.vote_type,
                        }
                    ),
                },
            )
            log.info(
                "discord_bridge.user_vote_timeout",
                vote_id=self.vote_id,
            )


# ── Bridge client ─────────────────────────────────────────────────────────────


class DiscordBridgeClient(discord.Client):
    """
    discord.Client subclass that owns the Redis connection and background
    consumer task. setup_hook() is the correct place to launch background
    tasks in discord.py 2.x — it runs before the bot starts processing events.
    """

    def __init__(
        self,
        redis_url: str,
        general_channel_id: str,
        intake_channel_id: str | None = None,
        claude_channel_id: str | None = None,
        control_channel_id: str | None = None,
        logs_channel_id: str | None = None,
        errors_channel_id: str | None = None,
        task_tracking_channel_id: str | None = None,
        votes_channel_id: str | None = None,
        approvals_channel_id: str | None = None,
        knowledge_gap_channel_id: str | None = None,
        control_helper_url: str = "http://host.docker.internal:7799",
        guild_id: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.redis_url = redis_url
        self.general_channel_id = (
            general_channel_id  # #general — conversational tasks + reactions
        )
        self.intake_channel_id = (
            intake_channel_id  # #intake — async backlog, reactions but no replies
        )
        self.claude_channel_id = (
            claude_channel_id  # #claude — direct Claude chat + escalations
        )
        self.control_channel_id = (
            control_channel_id  # #control — bot restart/status commands
        )
        self.logs_channel_id = logs_channel_id  # #logs — plan status, verbose events
        self.errors_channel_id = (
            errors_channel_id  # #errors — system errors, memory warnings
        )
        self.task_tracking_channel_id = (
            task_tracking_channel_id  # #task-tracking — lifecycle events
        )
        self.votes_channel_id = (
            votes_channel_id  # #votes — agent deliberation, plan votes
        )
        self.approvals_channel_id = (
            approvals_channel_id  # #approvals — approval.required, vote escalations
        )
        self.knowledge_gap_channel_id = (
            knowledge_gap_channel_id  # #knowledge-gap — agent knowledge gap escalations
        )
        self.control_helper_url = control_helper_url.rstrip("/")
        self.guild_id = int(guild_id) if guild_id else None
        self.redis: aioredis.Redis | None = None
        # Active vote embeds: plan_id → discord.Message (for editing when votes arrive)
        self._vote_messages: dict[str, discord.Message] = {}
        # task_id → (channel_id, message_id, enqueued_at) for ⏳→✅ reaction update.
        # Entries are evicted on task completion OR after _PENDING_TTL_SECS seconds
        # to prevent unbounded growth when tasks time out or agents crash.
        self._pending_messages: dict[str, tuple[int, int, float]] = {}
        self._PENDING_TTL_SECS = 1800  # 30 minutes — generous for slow tasks
        # Reverse index: message_id → task_id (for edit detection)
        self._message_to_task: dict[int, str] = {}
        # cached channel objects — fetched and stored on first use
        self._logs_channel = None
        self._errors_channel = None
        self._task_tracking_channel = None
        # Pending relaunch confirmations: message_id → (mode, svc, expires_at)
        self._relaunch_pending: dict[int, tuple[str, str | None, float]] = {}
        # Deduplication: recently-dispatched event_ids (capped at 2000 entries)
        # Prevents double-posting when the consumer group redelivers unacked messages
        # after a bridge restart or Redis reconnect.
        self._seen_event_ids: dict[str, float] = {}
        self._SEEN_TTL_SECS = 300  # 5 minutes — long enough to catch redeliveries
        # Per-vote guard: (plan_id, agent) → True — prevents a single agent's vote
        # from being appended to the embed twice if the event is redelivered.
        self._posted_votes: set[str] = set()
        # Knowledge gap embeds: gap_embed_message_id → task_id
        # Used to correlate a user's reply-to-embed with the right frozen task.
        self._gap_messages: dict[int, str] = {}

    async def setup_hook(self) -> None:
        self.redis = await aioredis.from_url(
            self.redis_url, encoding="utf-8", decode_responses=True
        )
        log.info("discord_bridge.redis_connected")
        self.loop.create_task(self._broadcast_consumer())

    async def on_ready(self) -> None:
        log.info(
            "discord_bridge.ready",
            user=str(self.user),
            guilds=[g.name for g in self.guilds],
        )
        try:
            channel = await self.fetch_channel(int(self.general_channel_id))
            log.info(
                "discord_bridge.channel_found",
                channel_id=self.general_channel_id,
                channel_name=channel.name,
                guild=channel.guild.name,
            )
            # Auto-detect guild_id from the general channel if not explicitly set
            if not self.guild_id and hasattr(channel, "guild"):
                self.guild_id = channel.guild.id
                log.info(
                    "discord_bridge.guild_detected",
                    guild_id=self.guild_id,
                    guild=channel.guild.name,
                )
        except Exception as exc:
            log.error(
                "discord_bridge.channel_fetch_failed",
                channel_id=self.general_channel_id,
                error=str(exc),
            )

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        channel_str = str(message.channel.id)

        log.info(
            "discord_bridge.message_received",
            channel_id=channel_str,
            has_content=bool(message.content),
            content_preview=message.content[:40]
            if message.content
            else "(empty — Message Content Intent may be disabled)",
        )

        # ── #control channel ──────────────────────────────────────────────────
        if self.control_channel_id and channel_str == self.control_channel_id:
            if not message.content:
                return
            await self._handle_control_command(message)
            return

        # ── #knowledge-gap channel ────────────────────────────────────────────
        if (
            self.knowledge_gap_channel_id
            and channel_str == self.knowledge_gap_channel_id
        ):
            await self._handle_knowledge_gap_reply(message)
            return

        # ── #claude channel ───────────────────────────────────────────────────
        if self.claude_channel_id and channel_str == self.claude_channel_id:
            if not message.content:
                return
            await self._queue_task(message, stream="agents:claude_code")
            return

        # ── #intake channel ───────────────────────────────────────────────────
        # Queue silently (no reply). Skip if the user has already reacted ✅ or ❌.
        if self.intake_channel_id and channel_str == self.intake_channel_id:
            if not message.content:
                return
            existing = {str(r.emoji) for r in message.reactions}
            if "✅" in existing or "❌" in existing:
                return  # user marked as done/cancelled before bot saw it
            await self._queue_task(message, stream="agents:orchestrator", silent=True)
            return

        # ── #general channel (default) ────────────────────────────────────────
        if channel_str != self.general_channel_id:
            return

        if not message.content:
            await message.reply(
                "⚠️ Message Content Intent is not enabled. "
                "Go to discord.com/developers/applications → your app → Bot → "
                "Privileged Gateway Intents → enable **Message Content Intent**."
            )
            return

        await self._queue_task(message, stream="agents:orchestrator")

    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent) -> None:
        """
        Detect edits to task-channel messages that are still in-flight.
        If the user edits a message that mapped to an active task_id, publish a
        task.updated event to the orchestrator so it can interrupt or amend the task.
        """
        channel_str = str(payload.channel_id)
        if channel_str not in (self.general_channel_id, self.claude_channel_id or ""):
            return

        task_id = self._message_to_task.get(payload.message_id)
        if not task_id:
            return  # message not in-flight, ignore

        new_content = (payload.data.get("content") or "").strip()
        if not new_content:
            return

        await self.redis.xadd(
            "agents:orchestrator",
            {
                "event_id": str(uuid.uuid4()),
                "type": "task.updated",
                "source": "discord",
                "task_id": task_id,
                "timestamp": str(time.time()),
                "payload": json.dumps(
                    {
                        "task_id": task_id,
                        "updated_task": new_content[:500],
                        "message_id": str(payload.message_id),
                        "channel_id": str(payload.channel_id),
                    }
                ),
            },
        )
        log.info(
            "discord_bridge.task_updated",
            task_id=task_id,
            new_content=new_content[:80],
        )

    async def on_raw_reaction_add(
        self, payload: discord.RawReactionActionEvent
    ) -> None:
        """
        Watch for user ✅ / ❌ reactions on #intake messages.
        If the reacting user is not the bot and the message is still in-flight,
        cancel it so the agent stops working on it.
        """
        if not self.intake_channel_id:
            return
        if str(payload.channel_id) != self.intake_channel_id:
            return
        if payload.user_id == self.user.id:
            return  # ignore the bot's own reactions
        if str(payload.emoji) not in ("✅", "❌"):
            return

        task_id = self._message_to_task.get(payload.message_id)
        if not task_id:
            return  # not an in-flight intake task

        label = "done" if str(payload.emoji) == "✅" else "cancelled"
        await self.redis.xadd(
            "agents:orchestrator",
            {
                "event_id": str(uuid.uuid4()),
                "type": "task.cancelled",
                "source": "discord",
                "task_id": task_id,
                "timestamp": str(time.time()),
                "payload": json.dumps(
                    {
                        "task_id": task_id,
                        "reason": f"User marked intake item as {label} via reaction",
                        "message_id": str(payload.message_id),
                        "channel_id": str(payload.channel_id),
                    }
                ),
            },
        )
        # Evict from pending so completion logic doesn't try to react afterwards
        self._pending_messages.pop(task_id, None)
        self._message_to_task.pop(payload.message_id, None)
        log.info(
            "discord_bridge.intake_item_marked",
            task_id=task_id,
            label=label,
        )

    async def _detect_chat_session(self, keywords: list[str]) -> str | None:
        """
        Check the context registry for an active chat session whose keywords
        overlap enough with this message.  Returns the session_id or None.
        No LLM — purely keyword-set intersection via Redis.
        """
        try:
            raw = await self.redis.hgetall("ctx:registry")
            if not raw:
                return None
            now = time.time()
            idle_gap = 1800  # 30 min default (kept in sync with config)
            try:
                raw_cfg = await self.redis.get("config:chat_idle_gap_secs")
                if raw_cfg:
                    idle_gap = float(raw_cfg.strip('"'))
            except Exception:
                pass
            thresh = 0.4
            try:
                raw_cfg = await self.redis.get("config:chat_keyword_overlap")
                if raw_cfg:
                    thresh = float(raw_cfg.strip('"'))
            except Exception:
                pass

            query_set = set(keywords)
            best_id: str | None = None
            best_score = 0.0

            for _ctx_id, meta_raw in raw.items():
                try:
                    meta = json.loads(meta_raw)
                except Exception:
                    continue
                if meta.get("type") != "chat" or meta.get("status") != "active":
                    continue
                last_active = (
                    meta.get("last_active")
                    or meta.get("updated_at")
                    or meta.get("created_at", 0)
                )
                if now - last_active > idle_gap:
                    continue
                pat_set = set(meta.get("keywords") or [])
                union = query_set | pat_set
                if not union:
                    continue
                score = len(query_set & pat_set) / len(union)
                if score >= thresh and score > best_score:
                    best_score = score
                    best_id = meta.get("id")

            return best_id
        except Exception as exc:
            log.warning("discord_bridge.session_detect_failed", error=str(exc))
            return None

    async def _queue_task(
        self, message: discord.Message, stream: str, silent: bool = False
    ) -> None:
        """
        Publish a task.created event to the given Redis stream.
        Performs local classification and session detection so the orchestrator
        receives a session_id hint without needing an extra LLM call.

        silent=True suppresses replies (used for #intake — reactions only).
        """
        task_id = str(uuid.uuid4())
        message_id = str(message.id)
        text = message.content

        # Local classification (no LLM)
        local_intent = (
            _classify_message_locally(text)
            if stream == "agents:orchestrator"
            else "task"
        )
        keywords = _extract_keywords(text)

        # Try to find an existing session this message belongs to
        session_id = (
            await self._detect_chat_session(keywords)
            if local_intent == "chat"
            else None
        )

        event = {
            "event_id": str(uuid.uuid4()),
            "type": "task.created",
            "source": "discord",
            "task_id": task_id,
            "timestamp": str(time.time()),
            "payload": json.dumps(
                {
                    "task": text,
                    "discord_user": message.author.display_name,
                    "discord_message_id": message_id,
                    "discord_channel_id": str(message.channel.id),
                    "local_intent": local_intent,
                    "session_id": session_id or "",
                }
            ),
        }
        await self.redis.xadd(stream, event)
        self._pending_messages[task_id] = (message.channel.id, message.id, time.time())
        self._message_to_task[message.id] = task_id
        await message.add_reaction("⏳")
        log.info(
            "discord_bridge.task_queued",
            stream=stream,
            task=text[:80],
            task_id=task_id,
            local_intent=local_intent,
            session_id=session_id or "new",
        )

    async def _handle_knowledge_gap_reply(self, message: discord.Message) -> None:
        """
        Handle a message in #knowledge-gap.

        Resolution priority:
        1. Reply-to-embed  — message.reference points to a known gap embed
        2. teach <task_id> <source>  — explicit command form
        3. Freeform in channel with open gaps  — if exactly one gap is open,
           attribute it automatically; otherwise ask the user to reply to the embed.
        """
        content = message.content.strip() if message.content else ""
        attachments = message.attachments

        # ── Determine task_id ─────────────────────────────────────────────────
        task_id = None

        # 1. Reply-to-embed
        if message.reference and message.reference.message_id:
            task_id = self._gap_messages.get(message.reference.message_id)

        # 2. Explicit "teach <task_id> <source>" form
        if task_id is None and content.lower().startswith("teach "):
            parts = content.split(None, 2)
            if len(parts) >= 3:
                task_id = parts[1]
                content = parts[2]

        # 3. Single open gap — auto-attribute
        if task_id is None and len(self._gap_messages) == 1:
            task_id = next(iter(self._gap_messages.values()))

        if not task_id:
            await message.reply(
                "❓ Couldn't match this to an open knowledge gap. "
                "Please **reply directly to the gap embed**, or use "
                "`teach <task_id> <source>`."
            )
            return

        # ── Determine source ──────────────────────────────────────────────────
        source = ""
        source_type = ""

        if attachments:
            # Discord CDN URLs are fetchable — treat as a URL source
            source = attachments[0].url
            source_type = "url"
        elif content:
            import re as _re

            if _re.match(r"https?://", content.split()[0]):
                source = content.split()[0]
                source_type = "url"
            elif content.startswith("/workspace/"):
                source = content.strip()
                source_type = "file"
            else:
                source = content
                source_type = "text"

        if not source:
            await message.reply(
                "❓ No source found. Please include a URL, a `/workspace/` path, or text."
            )
            return

        # ── Emit knowledge.teach to orchestrator ──────────────────────────────
        await self.redis.xadd(
            "agents:orchestrator",
            {
                "event_id": str(uuid.uuid4()),
                "type": "knowledge.teach",
                "source": "discord_bridge",
                "task_id": task_id,
                "timestamp": str(time.time()),
                "payload": json.dumps(
                    {"task_id": task_id, "source": source, "source_type": source_type}
                ),
            },
        )

        # Clean up the gap entry
        self._gap_messages = {
            mid: tid for mid, tid in self._gap_messages.items() if tid != task_id
        }

        type_label = {
            "url": "URL",
            "file": "workspace file",
            "text": "inline text",
        }.get(source_type, source_type)
        await message.reply(
            f"✅ Got it — sending {type_label} to the agent and resuming task `{task_id}`."
        )
        log.info(
            "discord_bridge.knowledge_gap_resolved",
            task_id=task_id,
            source_type=source_type,
        )

    async def _handle_control_command(self, message: discord.Message) -> None:
        """Parse and dispatch a control channel command."""
        text = message.content.strip()
        lower = text.lower()

        # help
        if lower in ("help", "?"):
            await message.reply(_CONTROL_HELP)
            return

        # status
        if lower == "status":
            await message.add_reaction("⏳")
            result = await self._call_helper("GET", "/status")
            await message.remove_reaction("⏳", self.user)
            output = result.get("output") or result.get("message") or str(result)
            await message.reply(f"```\n{output[:1900]}\n```")
            return

        # restart <service>
        parts = lower.split()
        if len(parts) == 2 and parts[0] == "restart":
            service = parts[1]
            await message.add_reaction("⏳")
            result = await self._call_helper("POST", f"/restart/{service}")
            await message.remove_reaction("⏳", self.user)
            ok = result.get("ok", False)
            msg = result.get("message", str(result))
            icon = "✅" if ok else "❌"
            # Include details for "restart all"
            details = result.get("details")
            if details:
                lines = [f"{icon} {msg}"]
                for d in details:
                    lines.append(("  ✅ " if d["ok"] else "  ❌ ") + d["message"])
                await message.reply("\n".join(lines)[:2000])
            else:
                await message.reply(f"{icon} {msg}")
            log.info("discord_bridge.control_command", service=service, ok=ok)
            return

        # build docs — force an immediate full architecture rebuild
        if lower in ("build docs", "build doc", "build arch", "rebuild docs"):
            await message.add_reaction("⏳")
            try:
                # Clear the 24 h gate so the orchestrator treats this as overdue
                await self.redis.delete("doc:arch_last_full_build")
                # Publish directly to orchestrator — no need to wait for the next think cycle
                task_id = str(uuid.uuid4())
                event = {
                    "event_id": str(uuid.uuid4()),
                    "type": "task.created",
                    "source": "control",
                    "task_id": task_id,
                    "timestamp": str(__import__("time").time()),
                    "payload": __import__("json").dumps(
                        {
                            "task": "Force full architecture documentation rebuild. "
                            "Review the agent-stack source, generate updated "
                            "architecture docs, compile to PDF, and upload to Google Drive.",
                            "discord_message_id": str(message.id),
                        }
                    ),
                }
                await self.redis.xadd("agents:orchestrator", event)
                await message.remove_reaction("⏳", self.user)
                await message.reply(
                    "📐 Architecture rebuild queued — document_qa will be spun up shortly."
                )
                log.info("discord_bridge.force_arch_build", task_id=task_id)
            except Exception as exc:
                await message.remove_reaction("⏳", self.user)
                await message.reply(f"❌ Failed to queue rebuild: {exc}")
            return

        # verbose on/off
        if lower in ("verbose on", "verbose 1", "verbose true"):
            await self.redis.set("config:verbose_events", "1")
            await message.reply(
                "🔊 Verbose mode **on** — all Redis events will be forwarded to #agent-logs."
            )
            log.info("discord_bridge.verbose_enabled")
            return

        if lower in ("verbose off", "verbose 0", "verbose false"):
            await self.redis.delete("config:verbose_events")
            await message.reply(
                "🔇 Verbose mode **off** — returning to normal event filtering."
            )
            log.info("discord_bridge.verbose_disabled")
            return

        # loglevel <level>
        if len(parts) == 2 and parts[0] == "loglevel":
            level = parts[1].upper()
            if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
                await message.reply(
                    "❌ Invalid level. Use: `loglevel DEBUG|INFO|WARNING|ERROR`"
                )
                return
            await self.redis.set("config:log_level", level)
            # Apply immediately to the bridge process itself
            import logging as _logging

            _logging.getLogger().setLevel(getattr(_logging, level))
            await message.reply(
                f"📋 Log level set to **{level}** — agents will pick this up on their next think cycle."
            )
            log.info("discord_bridge.loglevel_set", level=level)
            return

        # reset session
        if lower in ("reset session", "reset context", "new session", "clear context"):
            await self.redis.xadd(
                "agents:broadcast",
                {
                    "type": "session.reset",
                    "source": "discord_control",
                    "payload": "{}",
                },
            )
            await message.reply(
                "🔄 Session reset sent — orchestrator will clear conversation history "
                "and reload intent examples."
            )
            log.info("discord_bridge.session_reset_sent")
            return

        # pause <task_id>
        if len(parts) == 2 and parts[0] == "pause":
            task_id = parts[1]
            pause_key = f"research:pause:{task_id}"
            await self.redis.set(pause_key, "1", ex=3600)
            await self.redis.xadd(
                "agents:broadcast",
                {
                    "event_id": str(uuid.uuid4()),
                    "type": "task.paused",
                    "source": "discord_control",
                    "task_id": task_id,
                    "timestamp": str(time.time()),
                    "payload": json.dumps({"task_id": task_id}),
                },
            )
            await message.reply(
                f"⏸ Pause signal sent for task `{task_id}`. "
                "The research agent will stop after the current iteration completes."
            )
            log.info("discord_bridge.task_pause_requested", task_id=task_id)
            return

        # resume <task_id>
        if len(parts) == 2 and parts[0] == "resume":
            task_id = parts[1]
            await self.redis.xadd(
                "agents:orchestrator",
                {
                    "event_id": str(uuid.uuid4()),
                    "type": "task.resumed",
                    "source": "discord_control",
                    "task_id": task_id,
                    "timestamp": str(time.time()),
                    "payload": json.dumps({"task_id": task_id}),
                },
            )
            await message.reply(
                f"▶ Resume signal sent for task `{task_id}`. "
                "The orchestrator will reload the plan and continue from the last checkpoint."
            )
            log.info("discord_bridge.task_resume_requested", task_id=task_id)
            return

        # teach <task_id> <source>  — resolve a knowledge gap from the control channel
        if len(parts) >= 3 and parts[0] == "teach":
            task_id = parts[1]
            source = parts[2]
            import re as _re

            if _re.match(r"https?://", source):
                source_type = "url"
            elif source.startswith("/workspace/"):
                source_type = "file"
            else:
                source_type = "text"
            await self.redis.xadd(
                "agents:orchestrator",
                {
                    "event_id": str(uuid.uuid4()),
                    "type": "knowledge.teach",
                    "source": "discord_control",
                    "task_id": task_id,
                    "timestamp": str(time.time()),
                    "payload": json.dumps(
                        {
                            "task_id": task_id,
                            "source": source,
                            "source_type": source_type,
                        }
                    ),
                },
            )
            # Clean up gap embed tracking if present
            self._gap_messages = {
                mid: tid for mid, tid in self._gap_messages.items() if tid != task_id
            }
            await message.reply(
                f"✅ Teaching source sent for task `{task_id}`. Agent will resume shortly."
            )
            log.info(
                "discord_bridge.knowledge_teach_sent",
                task_id=task_id,
                source_type=source_type,
            )
            return

        # relaunch [mode] [svc]
        # Modes: code, rebuild, rebuild <svc>, flush, reset
        # flush and reset are destructive — require a confirm reply before executing.
        if parts and parts[0] == "relaunch":
            mode = parts[1] if len(parts) >= 2 else "rebuild"
            svc = parts[2] if len(parts) >= 3 else None

            valid_modes = {"code", "rebuild", "flush", "reset"}
            if mode not in valid_modes:
                await message.reply(
                    f"❓ Unknown relaunch mode `{mode}`. "
                    f"Valid modes: {', '.join(sorted(valid_modes))}"
                )
                return

            # Destructive modes require explicit confirmation.
            if mode in ("flush", "reset"):
                expires_at = time.time() + 120  # 2-minute window
                self._relaunch_pending[message.id] = (mode, svc, expires_at)
                warnings = {
                    "flush": "⚠️ This will **clear all Redis streams** (pending events lost). Postgres data is preserved.",
                    "reset": "🔴 **DESTRUCTIVE** — this will delete ALL data (Redis + Postgres volumes). All agent knowledge and task history will be lost.",
                }
                await message.reply(
                    f"{warnings[mode]}\n\n"
                    f"Reply `confirm` (within 2 minutes) to proceed with `relaunch {mode}`."
                )
                return

            # Non-destructive modes run immediately.
            await self._run_relaunch(message, mode, svc)
            return

        # confirm — resolve a pending relaunch
        if lower == "confirm" and message.reference:
            ref_id = message.reference.message_id
            entry = self._relaunch_pending.pop(ref_id, None)
            if entry is None:
                await message.reply("❓ No pending relaunch to confirm.")
                return
            mode, svc, expires_at = entry
            if time.time() > expires_at:
                await message.reply(
                    "⏰ Confirmation window expired. Re-issue the relaunch command."
                )
                return
            await self._run_relaunch(message, mode, svc)
            return

        # unrecognised
        await message.reply(f"❓ Unknown command. {_CONTROL_HELP}")

    async def _run_relaunch(
        self, message: discord.Message, mode: str, svc: str | None
    ) -> None:
        """Call /relaunch/<mode>[/<svc>] on the host helper and post the result."""
        path = f"/relaunch/{mode}" + (f"/{svc}" if svc else "")
        await message.add_reaction("⏳")
        result = await self._call_helper("POST", path)
        await message.remove_reaction("⏳", self.user)
        ok = result.get("ok", False)
        icon = "✅" if ok else "❌"
        msg = result.get("message", str(result))
        output = result.get("output", "").strip()
        # Trim output to fit Discord's 2000-char limit alongside the header line.
        if output:
            header = f"{icon} {msg}\n```\n"
            footer = "\n```"
            max_out = 2000 - len(header) - len(footer)
            if len(output) > max_out:
                output = "..." + output[-max_out + 3 :]
            await message.reply(f"{header}{output}{footer}")
        else:
            await message.reply(f"{icon} {msg}")
        log.info("discord_bridge.relaunch", mode=mode, svc=svc, ok=ok)

    async def _call_helper(self, method: str, path: str) -> dict:
        """Call the host restart helper over HTTP using async httpx."""
        url = self.control_helper_url + path
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.request(method, url)
                return r.json()
        except Exception as e:
            return {"ok": False, "message": str(e)}

    # ── Broadcast stream consumer ─────────────────────────────────────────────

    async def _broadcast_consumer(self) -> None:
        group = "discord_bridge_group"
        stream = "agents:broadcast"
        consumer = "discord_bridge"

        try:
            await self.redis.xgroup_create(stream, group, id="$", mkstream=True)
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        # Default channel for broadcast events is #logs; fall back to
        # #general only if no log channel is configured.
        channel_id = int(self.logs_channel_id or self.general_channel_id)

        while True:
            try:
                results = await self.redis.xreadgroup(
                    groupname=group,
                    consumername=consumer,
                    streams={stream: ">"},
                    count=10,
                    block=1000,
                )
                if not results:
                    # Evict stale pending messages and seen event ids on idle ticks
                    now = time.time()
                    expired = [
                        k
                        for k, v in self._pending_messages.items()
                        if now - v[2] > self._PENDING_TTL_SECS
                    ]
                    for k in expired:
                        del self._pending_messages[k]
                    if expired:
                        log.debug("discord_bridge.pending_evicted", count=len(expired))
                    expired_seen = [
                        k
                        for k, ts in self._seen_event_ids.items()
                        if now - ts > self._SEEN_TTL_SECS
                    ]
                    for k in expired_seen:
                        del self._seen_event_ids[k]
                    continue
                for _stream_key, messages in results:
                    for entry_id, data in messages:
                        await self._dispatch(data, channel_id)
                        await self.redis.xack(stream, group, entry_id)
            except aioredis.ResponseError as exc:
                if "NOGROUP" in str(exc):
                    # Stream was deleted and recreated — rebuild the consumer group
                    log.warning(
                        "discord_bridge.rebuilding_consumer_group", stream=stream
                    )
                    try:
                        await self.redis.xgroup_create(
                            stream, group, id="$", mkstream=True
                        )
                    except aioredis.ResponseError:
                        pass  # BUSYGROUP is fine — another instance beat us to it
                else:
                    log.error("discord_bridge.consumer_error", error=str(exc))
                    await asyncio.sleep(3)
            except Exception as exc:
                log.error("discord_bridge.consumer_error", error=str(exc))
                await asyncio.sleep(3)

    async def _get_logs_channel(self):
        """Return the #logs channel, fetching and caching it on first use."""
        if self._logs_channel is not None:
            return self._logs_channel
        if not self.logs_channel_id:
            return None
        try:
            self._logs_channel = await self.fetch_channel(int(self.logs_channel_id))
        except Exception as exc:
            log.warning("discord_bridge.logs_channel_fetch_failed", error=str(exc))
            self._logs_channel = None
        return self._logs_channel

    async def _get_errors_channel(self):
        """Return the #errors channel, falling back to #logs."""
        if self._errors_channel is not None:
            return self._errors_channel
        if self.errors_channel_id:
            try:
                self._errors_channel = await self.fetch_channel(
                    int(self.errors_channel_id)
                )
                return self._errors_channel
            except Exception as exc:
                log.warning(
                    "discord_bridge.errors_channel_fetch_failed", error=str(exc)
                )
        # Fall back to #logs
        return await self._get_logs_channel()

    async def _get_task_tracking_channel(self):
        """Return the #task-tracking channel, falling back to #logs."""
        if self._task_tracking_channel is not None:
            return self._task_tracking_channel
        if self.task_tracking_channel_id:
            try:
                self._task_tracking_channel = await self.fetch_channel(
                    int(self.task_tracking_channel_id)
                )
                return self._task_tracking_channel
            except Exception as exc:
                log.warning(
                    "discord_bridge.task_tracking_channel_fetch_failed", error=str(exc)
                )
        return await self._get_logs_channel()

    async def _send_long_result(
        self,
        channel: discord.abc.Messageable,
        embed: discord.Embed,
        text: str,
        filename: str = "result.txt",
    ) -> None:
        """
        Send `embed` to `channel`. If `text` exceeds 3800 chars (safe embed limit),
        attach the full text as a file so nothing is silently cut off.
        Discord embed descriptions cap at 4096; we use 3800 to leave room for title/footer.
        """
        if len(text) <= 3800:
            await channel.send(embed=embed)
        else:
            # Truncate embed description with note, attach full text
            embed.description = text[:3800] + "\n\n*[full output attached below]*"
            file = discord.File(
                io.BytesIO(text.encode("utf-8", errors="replace")),
                filename=filename,
            )
            await channel.send(embed=embed, file=file)

    async def _is_verbose(self) -> bool:
        """Return True when verbose mode is enabled (config:verbose_events = '1')."""
        try:
            val = await self.redis.get("config:verbose_events")
            return val in ("1", "true", "on")
        except Exception:
            return False

    async def _dispatch(self, data: dict, channel_id: int) -> None:
        event_type = data.get("type", "")

        # Dedup: skip events we've already dispatched (redeliveries after restart/reconnect)
        event_id = data.get("event_id", "")
        if event_id:
            now = time.time()
            if event_id in self._seen_event_ids:
                log.debug(
                    "discord_bridge.dedup_skip",
                    event_id=event_id,
                    event_type=event_type,
                )
                return
            self._seen_event_ids[event_id] = now
            # Evict oldest entries if the table grows too large
            if len(self._seen_event_ids) > 2000:
                oldest = sorted(self._seen_event_ids.items(), key=lambda x: x[1])[:500]
                for k, _ in oldest:
                    del self._seen_event_ids[k]

        verbose = event_type in VERBOSE_EVENTS
        if event_type not in VISIBLE_EVENTS:
            if not verbose:
                return
            # Verbose-only event: only forward when verbose mode is active
            if not await self._is_verbose():
                return

        try:
            channel = await self.fetch_channel(channel_id)
        except Exception as exc:
            log.warning(
                "discord_bridge.channel_not_found",
                channel_id=channel_id,
                error=str(exc),
            )
            return

        source = data.get("source", "system")
        payload = json.loads(data.get("payload", "{}"))
        icon = AGENT_ICON.get(source, "🤖")
        color = AGENT_COLOR.get(source, discord.Color.blurple())

        # Verbose-only events: compact one-liner to #logs, never to the main channel
        if verbose and event_type not in VISIBLE_EVENTS:
            log_ch = await self._get_logs_channel()
            dest = log_ch or channel
            task_id = data.get("task_id", "")[:8]
            ts = float(data.get("timestamp", time.time()))
            ts_str = time.strftime("%H:%M:%S", time.localtime(ts))
            # Summarise payload concisely
            summary = ""
            if event_type == "agent.tool_call":
                summary = f"`{payload.get('command', payload.get('tool', ''))[:120]}`"
            elif event_type == "agent.tool_result":
                summary = payload.get("result", "")[:200]
            elif event_type in ("task.created", "task.assigned"):
                summary = payload.get("task", "")[:120]
            elif event_type == "memory.promoted":
                summary = (
                    f"[{payload.get('topic', '')}] {payload.get('preview', '')[:100]}"
                )
            elif event_type == "agent.thinking":
                summary = payload.get("task", "")[:120]
            elif event_type in ("self.modify.proposed", "self.modify.applied"):
                summary = payload.get("file", payload.get("description", ""))[:120]
            else:
                summary = str(payload)[:200]
            embed = discord.Embed(
                description=f"**{ts_str}** `{event_type}` · {icon} **{source}** · task `{task_id}`\n{summary}",
                color=discord.Color.greyple(),
            )
            await dest.send(embed=embed)
            return

        if event_type == "task.created":
            # Post to tasks-tracking channel when a new task enters the queue
            task_text = payload.get("task", "")[:200]
            task_id_short = data.get("task_id", "")[:8]
            embed = discord.Embed(
                title="📋 Task Queued",
                description=task_text,
                color=discord.Color.blurple(),
            )
            embed.set_footer(text=f"task {task_id_short} | {source}")
            tracking_ch = await self._get_task_tracking_channel()
            if tracking_ch:
                await tracking_ch.send(embed=embed)

        elif event_type == "task.failed":
            task_text = payload.get("task", "")[:200]
            error = payload.get("error", "")[:300]
            task_id_short = data.get("task_id", "")[:8]
            embed = discord.Embed(
                title="❌ Task Failed",
                description=f"**Task:** {task_text}\n**Error:** {error}",
                color=discord.Color.red(),
            )
            embed.set_footer(text=f"task {task_id_short} | {source}")
            tracking_ch = await self._get_task_tracking_channel()
            if tracking_ch:
                await tracking_ch.send(embed=embed)

        elif event_type == "task.completed":
            # Memory approval requests piggyback on task.completed — intercept early
            if payload.get("memory_approval"):
                approval_id = payload.get("approval_id", "")
                topic = payload.get("topic", "")
                approvals_ch = await self._get_approvals_channel()
                target_ch = approvals_ch or channel
                embed = discord.Embed(
                    title="🧠 Memory Retention Approval",
                    description=payload.get("result", ""),
                    color=discord.Color.blurple(),
                )
                embed.set_footer(text="Auto-denies in 2 minutes")
                view = MemoryApprovalView(approval_id, self.redis)
                await target_ch.send(embed=embed, view=view)
                return

            result = payload.get("result", "(no output)")
            task_id = payload.get("task_id")
            is_chat = payload.get("is_chat", False)
            embed = discord.Embed(
                title=f"{icon} {source.replace('_', ' ').title()}",
                description=result[:3800],
                color=color,
            )

            # Post completion to #task-tracking (skip for simple chat responses)
            if not is_chat:
                tracking_ch = await self._get_task_tracking_channel()
                if tracking_ch:
                    task_id_short = (task_id or "")[:8]
                    t_embed = discord.Embed(
                        title="✅ Task Completed",
                        description=result[:500],
                        color=discord.Color.green(),
                    )
                    t_embed.set_footer(text=f"task {task_id_short} | {source}")
                    await tracking_ch.send(embed=t_embed)

            # Try to reply to the original user message so it threads nicely
            origin = self._pending_messages.pop(task_id, None) if task_id else None
            if origin:
                ch_id, msg_id, _enqueued_at = origin
                self._message_to_task.pop(msg_id, None)
                try:
                    ch = await self.fetch_channel(ch_id)
                    msg = await ch.fetch_message(msg_id)
                    await msg.remove_reaction("⏳", self.user)
                    await msg.add_reaction("✅")
                    if len(result) > 3800:
                        embed.description = (
                            result[:3800] + "\n\n*[full output attached below]*"
                        )
                        file = discord.File(
                            io.BytesIO(result.encode("utf-8", errors="replace")),
                            filename=f"result_{(task_id or 'task')[:8]}.txt",
                        )
                        await msg.reply(embed=embed, file=file, mention_author=False)
                    else:
                        await msg.reply(embed=embed, mention_author=False)
                    log.info("discord_bridge.reply_sent", task_id=task_id)
                    return
                except Exception as exc:
                    log.warning("discord_bridge.reply_failed", error=str(exc))

            await self._send_long_result(
                channel, embed, result, filename=f"result_{(task_id or 'task')[:8]}.txt"
            )
            log.info(
                "discord_bridge.message_sent", event_type=event_type, source=source
            )

        elif event_type == "approval.required":
            approvals_ch = await self._get_approvals_channel()
            await self._post_approval(payload, approvals_ch or channel, source, icon)

        elif event_type == "claude.escalation":
            await self._post_claude_escalation(payload, source, icon)

        elif event_type in ("system.error", "agent.response"):
            text = payload.get("error") or payload.get("response") or str(payload)
            is_error = event_type == "system.error"
            embed = discord.Embed(
                title=f"{icon} {source.replace('_', ' ').title()}",
                description=f"```{text[:1500]}```" if is_error else text[:3800],
                color=discord.Color.dark_red() if is_error else color,
            )
            if is_error:
                err_ch = await self._get_errors_channel()
                await (err_ch or channel).send(embed=embed)
            else:
                await self._send_long_result(channel, embed, text)

        elif event_type == "memory.pruned":
            message = payload.get("message", "Memory pruned.")
            reason = payload.get("reason", "")
            embed = discord.Embed(
                title="⚠️ Memory Warning",
                description=message,
                color=discord.Color.yellow(),
            )
            embed.set_footer(text=f"Source: {source} | Reason: {reason}")
            # Warnings go to #errors channel
            err_ch = await self._get_errors_channel()
            await (err_ch or channel).send(embed=embed)
            log.info(
                "discord_bridge.memory_pruned_notified", source=source, reason=reason
            )

        elif event_type == "plan.status":
            message = payload.get("message", "")
            plan_id = payload.get("plan_id", "")[:8]
            retry_count = payload.get("retry_count", 0)
            original = payload.get("original_task", "")

            # Pick color by message prefix
            if message.startswith("❌"):
                msg_color = discord.Color.red()
            elif message.startswith("⚠️"):
                msg_color = discord.Color.yellow()
            elif message.startswith("✅"):
                msg_color = discord.Color.green()
            elif message.startswith("⚡"):
                msg_color = discord.Color.blurple()
            elif message.startswith("🚀"):
                msg_color = discord.Color.teal()
            elif message.startswith("→"):
                msg_color = discord.Color.og_blurple()
            elif message.startswith("←"):
                msg_color = discord.Color.from_rgb(100, 200, 100)
            else:
                msg_color = discord.Color.greyple()

            embed = discord.Embed(description=message[:2000], color=msg_color)
            footer = f"Plan {plan_id}" if plan_id else "orchestrator"
            if retry_count:
                footer += f" | retry {retry_count}/{3}"
            if original:
                footer += f" | {original[:80]}"
            embed.set_footer(text=footer)

            # Always route plan.status to #logs if configured;
            # fall back to the broadcast channel so nothing is lost.
            log_ch = await self._get_logs_channel()
            await (log_ch or channel).send(embed=embed)
            log.info("discord_bridge.plan_status", message=message[:80])

        elif event_type == "plan.proposed":
            await self._post_plan_proposed(payload)

        elif event_type == "agent.vote":
            await self._post_agent_vote(payload)

        elif event_type == "vote.initiated":
            await self._post_vote_initiated(payload)

        elif event_type == "vote.result":
            await self._post_vote_result(payload)

        elif event_type == "vote.escalated_to_user":
            await self._post_vote_escalated(payload)

        elif event_type == "context.closed":
            await self._post_context_closed(payload, channel)

        elif event_type == "discord.action":
            await self._execute_discord_action(payload, data.get("event_id", ""))

        elif event_type == "knowledge.gap":
            await self._post_knowledge_gap(payload)

    async def _post_knowledge_gap(self, payload: dict) -> None:
        """
        Post a knowledge gap escalation embed to #knowledge-gap.
        Stores the embed message_id → task_id so a user reply-to-embed
        can be correlated back to the frozen task.
        """
        ch = await self._get_knowledge_gap_channel()
        if not ch:
            return

        task_id = payload.get("task_id", "")
        what = payload.get("what", "unknown")
        agent = payload.get("agent", "unknown")

        embed = discord.Embed(
            title="❓ Agent needs information",
            color=discord.Color.yellow(),
        )
        embed.add_field(name="Agent", value=f"`{agent}`", inline=True)
        embed.add_field(name="Task ID", value=f"`{task_id}`", inline=True)
        embed.add_field(name="What it needs", value=what[:1000], inline=False)
        embed.add_field(
            name="How to resolve",
            value=(
                "**Reply to this message** with one of:\n"
                "• A URL — agent will fetch and learn from it\n"
                "• A `/workspace/docs/` path — agent will read the file\n"
                "• Plain text — injected directly as context\n\n"
                f"Or use the control channel: `teach {task_id} <source>`"
            ),
            inline=False,
        )

        try:
            msg = await ch.send(embed=embed)
            self._gap_messages[msg.id] = task_id
            log.info(
                "discord_bridge.knowledge_gap_posted",
                task_id=task_id,
                agent=agent,
                message_id=msg.id,
            )
        except Exception as exc:
            log.warning("discord_bridge.knowledge_gap_post_failed", error=str(exc))

    async def _execute_discord_action(self, payload: dict, event_id: str) -> None:
        """
        Execute a Discord API action on behalf of an agent.

        Supported actions (payload["action"]):
          send_message     — send text/embed to a channel
          create_channel   — create a text channel in the guild
          delete_channel   — delete a channel by id or name
          rename_channel   — rename a channel
          set_topic        — set channel topic
          create_category  — create a category
          pin_message      — pin a message in a channel
          list_channels    — reply with a list of guild channels
        """
        action = payload.get("action", "")
        task_id = payload.get("task_id", "")

        async def _ack(result: str, ok: bool = True) -> None:
            await self.redis.xadd(
                "agents:broadcast",
                {
                    "event_id": str(uuid.uuid4()),
                    "type": "discord.action.done",
                    "source": "discord_bridge",
                    "task_id": task_id,
                    "timestamp": str(time.time()),
                    "payload": json.dumps(
                        {
                            "action": action,
                            "result": result,
                            "ok": ok,
                            "triggering_event_id": event_id,
                        }
                    ),
                },
                maxlen=10_000,
                approximate=True,
            )
            log.info(
                "discord_bridge.action_done", action=action, ok=ok, result=result[:100]
            )

        guild = self.get_guild(self.guild_id) if self.guild_id else None
        if not guild and action not in ("send_message", "send_file"):
            await _ack("Guild not found — set DISCORD_GUILD_ID in .env", ok=False)
            return

        async def _resolve_channel(p: dict):
            """Resolve channel_id or channel_name to a channel object."""
            ch_id = p.get("channel_id")
            ch_name = (p.get("channel_name") or p.get("name", "")).lstrip("#").strip()
            if ch_id:
                return await self.fetch_channel(int(ch_id))
            if ch_name and guild:
                ch = discord.utils.get(guild.text_channels, name=ch_name)
                if not ch:
                    # case-insensitive fallback
                    ch = next(
                        (
                            c
                            for c in guild.text_channels
                            if c.name.lower() == ch_name.lower()
                        ),
                        None,
                    )
                return ch
            return None

        try:
            if action == "send_message":
                ch_id = payload.get("channel_id")
                text = payload.get("content", "")
                if not text:
                    await _ack("send_message requires non-empty content", ok=False)
                    return
                if ch_id:
                    ch = await self.fetch_channel(int(ch_id))
                else:
                    ch = await _resolve_channel(payload)
                    if ch is None:
                        ch = await self.fetch_channel(int(self.general_channel_id))
                if not ch:
                    await _ack("Channel not found for send_message", ok=False)
                    return
                await ch.send(text[:2000])
                await _ack(f"Message sent to #{ch.name}")

            elif action == "send_file":
                # Send a file from /workspace to a Discord channel.
                # payload fields:
                #   file_path    — absolute path inside the container (required)
                #   channel_id / channel_name — destination (defaults to task channel)
                #   content      — optional caption text
                from pathlib import Path as _Path

                file_path = payload.get("file_path", "")
                if not file_path:
                    await _ack("send_file requires a file_path", ok=False)
                    return
                p = _Path(file_path)
                if not p.exists():
                    await _ack(f"File not found: {file_path}", ok=False)
                    return
                if p.stat().st_size > 8 * 1024 * 1024:  # 8 MB Discord limit
                    await _ack(
                        f"File too large for Discord (>8 MB): {file_path}", ok=False
                    )
                    return
                ch_id = payload.get("channel_id")
                if ch_id:
                    ch = await self.fetch_channel(int(ch_id))
                else:
                    ch = await _resolve_channel(payload)
                    if ch is None:
                        ch = await self.fetch_channel(int(self.task_channel_id))
                if not ch:
                    await _ack("Channel not found for send_file", ok=False)
                    return
                caption = payload.get("content", "")
                await ch.send(
                    content=caption[:2000] if caption else discord.utils.MISSING,
                    file=discord.File(str(p), filename=p.name),
                )
                await _ack(f"File '{p.name}' sent to #{ch.name}")

            elif action == "create_channel":
                name = payload.get("name", "new-channel")
                topic = payload.get("topic", "")
                category_name = payload.get("category")
                category = None
                if category_name:
                    category = discord.utils.get(guild.categories, name=category_name)
                    if not category:
                        category = next(
                            (
                                c
                                for c in guild.categories
                                if c.name.lower() == category_name.lower()
                            ),
                            None,
                        )
                ch = await guild.create_text_channel(
                    name=name, topic=topic, category=category
                )
                await _ack(f"Created #{ch.name} (id={ch.id})")

            elif action == "delete_channel":
                ch = await _resolve_channel(payload)
                if not ch:
                    await _ack("Channel not found for delete_channel", ok=False)
                    return
                await ch.delete(reason=payload.get("reason", "Deleted by agent"))
                await _ack(f"Deleted #{ch.name}")

            elif action == "rename_channel":
                ch = await _resolve_channel(payload)
                if not ch:
                    await _ack("Channel not found for rename_channel", ok=False)
                    return
                old = ch.name
                new_name = payload.get("name", old)
                await ch.edit(name=new_name)
                await _ack(f"Renamed #{old} → #{new_name}")

            elif action == "set_topic":
                ch = await _resolve_channel(payload)
                topic = payload.get("topic", "")
                if not ch:
                    await _ack("Channel not found for set_topic", ok=False)
                    return
                await ch.edit(topic=topic)
                await _ack(f"Topic set on #{ch.name}")

            elif action == "create_category":
                name = payload.get("name", "New Category")
                cat = await guild.create_category(name=name)
                await _ack(f"Created category '{cat.name}' (id={cat.id})")

            elif action == "pin_message":
                ch = await _resolve_channel(payload)
                msg_id = payload.get("message_id")
                if not ch:
                    await _ack("Channel not found for pin_message", ok=False)
                    return
                msg = await ch.fetch_message(int(msg_id))
                await msg.pin()
                await _ack(f"Pinned message {msg_id} in #{ch.name}")

            elif action == "list_channels":
                lines = [f"#{ch.name} (id={ch.id})" for ch in guild.text_channels]
                await _ack("Channels:\n" + "\n".join(lines))

            elif action == "scan_task_queue":
                # Scan the last N messages in #general and #intake.
                # For each human message: react ✅ if completed, 📓 if still pending,
                # or skip if already has one of these reactions.
                # #intake messages with ❌ are skipped (user-cancelled).
                limit = int(payload.get("limit", 50))

                channel_ids_to_scan = [self.general_channel_id]
                if self.intake_channel_id:
                    channel_ids_to_scan.append(self.intake_channel_id)

                pending_msg_ids = {
                    msg_id for (_ch, msg_id, _ts) in self._pending_messages.values()
                }
                checked = reacted_done = reacted_pending = 0

                for cid in channel_ids_to_scan:
                    try:
                        scan_ch = await self.fetch_channel(int(cid))
                    except Exception as exc:
                        log.warning(
                            "discord_bridge.scan_channel_fetch_failed",
                            channel_id=cid,
                            error=str(exc),
                        )
                        continue
                    is_intake = cid == self.intake_channel_id
                    async for msg in scan_ch.history(limit=limit):
                        if msg.author.bot:
                            continue
                        checked += 1
                        existing_reactions = {str(r.emoji) for r in msg.reactions}
                        # Skip already-marked or user-cancelled intake items
                        if "✅" in existing_reactions or "📓" in existing_reactions:
                            continue
                        if is_intake and "❌" in existing_reactions:
                            continue  # user cancelled

                        if msg.id in pending_msg_ids:
                            if "⏳" not in existing_reactions:
                                await msg.add_reaction("📓")
                                reacted_pending += 1
                        else:
                            if "✅" not in existing_reactions:
                                await msg.add_reaction("✅")
                                reacted_done += 1

                await _ack(
                    f"Scanned {checked} messages across {len(channel_ids_to_scan)} channel(s): "
                    f"{reacted_done} marked ✅, {reacted_pending} marked 📓."
                )
                log.info(
                    "discord_bridge.task_queue_scanned",
                    checked=checked,
                    done=reacted_done,
                    pending=reacted_pending,
                )

            elif action == "find_and_delete_duplicates":
                # Find text channels that share a name (case-insensitive), keep the oldest (lowest id)
                seen: dict[str, discord.TextChannel] = {}
                deleted: list[str] = []
                errors: list[str] = []
                for ch in sorted(guild.text_channels, key=lambda c: c.id):
                    key = ch.name.lower()
                    if key in seen:
                        # This channel has a later id → it's the duplicate; delete it
                        try:
                            await ch.delete(reason="Duplicate channel removed by agent")
                            deleted.append(f"#{ch.name} (id={ch.id})")
                            log.info(
                                "discord_bridge.duplicate_deleted",
                                channel=ch.name,
                                id=ch.id,
                            )
                        except Exception as exc:
                            errors.append(f"#{ch.name}: {exc}")
                    else:
                        seen[key] = ch
                if deleted:
                    msg = f"Deleted {len(deleted)} duplicate(s): {', '.join(deleted)}"
                    if errors:
                        msg += f"\nFailed: {', '.join(errors)}"
                    await _ack(msg)
                else:
                    await _ack("No duplicate channels found.")

            else:
                await _ack(f"Unknown action: {action}", ok=False)

        except Exception as exc:
            log.error("discord_bridge.action_error", action=action, error=str(exc))
            await _ack(f"Error: {exc}", ok=False)

    async def _get_votes_channel(self):
        """Return the #votes channel, falling back to #logs, then #general."""
        for cid in filter(None, [self.votes_channel_id, self.logs_channel_id]):
            try:
                return await self.fetch_channel(int(cid))
            except Exception:
                pass
        try:
            return await self.fetch_channel(int(self.general_channel_id))
        except Exception:
            return None

    async def _get_approvals_channel(self):
        """Return the #approvals channel, falling back to #votes."""
        if self.approvals_channel_id:
            try:
                return await self.fetch_channel(int(self.approvals_channel_id))
            except Exception:
                pass
        return await self._get_votes_channel()

    async def _get_knowledge_gap_channel(self):
        """Return the #knowledge-gap channel, falling back to #general."""
        if self.knowledge_gap_channel_id:
            try:
                return await self.fetch_channel(int(self.knowledge_gap_channel_id))
            except Exception:
                pass
        try:
            return await self.fetch_channel(int(self.general_channel_id))
        except Exception:
            return None

    async def _post_plan_proposed(self, payload: dict) -> None:
        """Post a brief vote-initiated notice to #votes."""
        ch = await self._get_votes_channel()
        if not ch:
            return
        plan_id = payload.get("plan_id", "")[:8]
        task = payload.get("original_task", "")
        embed = discord.Embed(
            title="🗳️ Vote initiated",
            description=f"**Task:** {task[:300]}",
            color=discord.Color.og_blurple(),
        )
        embed.set_footer(text=f"Plan {plan_id} | awaiting agent votes…")
        msg = await ch.send(embed=embed)
        self._vote_messages[payload.get("plan_id", "")] = msg

    async def _post_vote_initiated(self, payload: dict) -> None:
        """Post a brief notice when a peer vote is opened by an agent."""
        ch = await self._get_votes_channel()
        if not ch:
            return
        vote_id = payload.get("vote_id", "")[:8]
        question = payload.get("question", "")
        initiator = payload.get("initiator", "unknown")
        embed = discord.Embed(
            title="🗳️ Peer vote opened",
            description=f"**{initiator}** has called a vote.\n\n**Question:** {question[:400]}",
            color=discord.Color.og_blurple(),
        )
        embed.set_footer(text=f"Vote {vote_id} | awaiting agent responses…")
        msg = await ch.send(embed=embed)
        self._vote_messages[payload.get("vote_id", "")] = msg

    async def _post_vote_result(self, payload: dict) -> None:
        """Post vote tally to #votes after voting closes."""
        ch = await self._get_votes_channel()
        if not ch:
            return
        plan_id = payload.get("plan_id", "")[:8]
        task = payload.get("task", "")
        outcome = payload.get("outcome", "approved")
        yay = payload.get("yay", 0)
        nay = payload.get("nay", 0)
        total = payload.get("total", 0)
        votes: list[dict] = payload.get("votes", [])

        vote_type = payload.get("vote_type", "plan")
        initiator = payload.get("initiator", "")

        tie_broken = payload.get("tie_broken", False)
        if outcome == "approved":
            color = discord.Color.green()
            title = "✅ Vote passed" + (
                " (tie-broken by orchestrator)" if tie_broken else ""
            )
        elif outcome == "no_quorum":
            color = discord.Color.yellow()
            title = "⏳ Quorum not met — retrying vote…"
        else:
            color = discord.Color.red()
            title = "❌ Vote failed" + (
                " — plan will be revised" if vote_type == "plan" else ""
            )

        if total == 0:
            distribution = "No agents voted (silent approval)."
        else:
            yay_pct = int(yay / total * 100)
            nay_pct = int(nay / total * 100)
            distribution = (
                f"**Yay:** {yay} ({yay_pct}%)  |  **Nay:** {nay} ({nay_pct}%)"
            )

        lines = []
        for v in votes:
            icon = "✅" if v.get("approve", True) else "❌"
            agent = v.get("agent", "?")
            conf = float(v.get("confidence", 1.0))
            reason = v.get("reason", "") or "no reason"
            lines.append(f"{icon} **{agent}** (conf={conf:.2f}): {reason[:120]}")

        body = distribution
        if lines:
            body += "\n\n" + "\n".join(lines)

        label = "Question" if vote_type == "peer" else "Task"
        header = f"**{label}:** {task[:200]}"
        if vote_type == "peer" and initiator:
            header += f"\n**Initiated by:** {initiator}"

        embed = discord.Embed(
            title=title,
            description=f"{header}\n\n{body[:3800]}",
            color=color,
        )
        embed.set_footer(text=f"{'Vote' if vote_type == 'peer' else 'Plan'} {plan_id}")
        await ch.send(embed=embed)

    async def _post_vote_escalated(self, payload: dict) -> None:
        """Post an interactive user-vote embed when agent quorum was not met."""
        ch = await self._get_approvals_channel()
        if not ch:
            return
        vote_id = payload.get("vote_id", "")
        question = payload.get("question", "")
        context = payload.get("context", "")
        vote_type = payload.get("vote_type", "peer")
        timeout_hours = int(payload.get("timeout_hours", 48))

        label = "Plan" if vote_type == "plan" else "Question"
        desc = f"**{label}:** {question[:400]}"
        if context:
            desc += f"\n\n**Context:**\n{context[:800]}"
        desc += f"\n\n⏳ Auto-rejects in **{timeout_hours} hours** if no response."

        embed = discord.Embed(
            title="🧑‍⚖️ Vote escalated — agent quorum not reached",
            description=desc,
            color=discord.Color.yellow(),
        )
        embed.set_footer(text=f"Vote {vote_id[:8]}")

        view = UserVoteView(
            vote_id=vote_id,
            vote_type=vote_type,
            redis_client=self.redis,
            timeout_seconds=timeout_hours * 3600,
        )
        await ch.send(embed=embed, view=view)
        log.info(
            "discord_bridge.vote_escalated_posted",
            vote_id=vote_id[:8],
            timeout_hours=timeout_hours,
        )

    async def _post_agent_vote(self, payload: dict) -> None:
        """Append a vote line to the existing plan-proposed embed (or post standalone)."""
        ch = await self._get_votes_channel()
        if not ch:
            return
        plan_id = payload.get("plan_id", "")
        agent = payload.get("agent", "?")
        approve = payload.get("approve", True)
        reason = payload.get("reason", "")
        confidence = payload.get("confidence", 1.0)

        # Guard: don't post the same agent's vote on the same plan twice
        vote_key = f"{plan_id}:{agent}"
        if vote_key in self._posted_votes:
            log.debug(
                "discord_bridge.vote_dedup_skip", plan_id=plan_id[:8], agent=agent
            )
            return
        self._posted_votes.add(vote_key)
        # Prune old vote keys to prevent unbounded growth (keep last 500)
        if len(self._posted_votes) > 500:
            self._posted_votes = set(list(self._posted_votes)[-400:])

        icon = "✅" if approve else "❌"
        line = f"{icon} **{agent}** (conf={confidence:.2f}): {reason or 'no reason'}"

        existing = self._vote_messages.get(plan_id)
        if existing:
            try:
                embed = existing.embeds[0]
                current = embed.description or ""
                embed.description = (current + f"\n{line}")[:4000]
                await existing.edit(embed=embed)
                return
            except Exception:
                pass
        # Fallback standalone post
        embed = discord.Embed(
            description=line,
            color=discord.Color.green() if approve else discord.Color.red(),
        )
        embed.set_footer(text=f"Vote on plan {plan_id[:8]}")
        await ch.send(embed=embed)

    async def _post_context_closed(self, payload: dict, fallback_channel) -> None:
        """
        Post a brief recap when a task or chat context closes.
        Valuable contexts (value_score ≥ 0.4) get a summary embed.
        """
        value_score = float(payload.get("value_score", 0))
        if value_score < 0.4:
            return  # too trivial to surface
        context_id = payload.get("context_id", "")[:8]
        summary = payload.get("summary", "")
        success = payload.get("success", True)
        color = discord.Color.green() if success else discord.Color.orange()
        embed = discord.Embed(
            title="📋 Context Closed",
            description=summary[:2000],
            color=color,
        )
        embed.set_footer(
            text=f"ID: {context_id} | value={value_score:.2f} | Use /recall {context_id} to revisit"
        )
        log_ch = await self._get_logs_channel()
        await (log_ch or fallback_channel).send(embed=embed)

    async def _post_approval(
        self, payload: dict, channel, source: str, icon: str
    ) -> None:
        approval_id = payload.get("approval_id", "")
        command = payload.get("command", "")
        task = payload.get("task", "")
        is_allowlist = payload.get("allowlist_request", False)
        base_cmd = payload.get("base_cmd", command.split()[0] if command else "")

        if is_allowlist:
            embed = discord.Embed(
                title="🛡️ Allowlist Exception Requested",
                description=(
                    f"**{icon} {source}** tried to run `{base_cmd}`, which is not on "
                    f"the command allowlist.\n\nAgents **voted to permit** it. "
                    f"Do you want to allow it?"
                ),
                color=discord.Color.orange(),
            )
            embed.add_field(
                name="Command", value=f"```{command[:900]}```", inline=False
            )
            if task:
                embed.add_field(name="Task context", value=task[:300], inline=False)
            embed.add_field(
                name="Options",
                value=(
                    "**✅ Once** — run this time only, not remembered\n"
                    "**♾️ Always** — add to allowlist permanently\n"
                    "**❌ Deny** — block this command"
                ),
                inline=False,
            )
            embed.set_footer(text="No response within 5 minutes = auto-denied")
            view = AllowlistApprovalView(approval_id, self.redis)
        else:
            embed = discord.Embed(
                title="🔐 Approval Required",
                description=f"**{icon} {source}** wants to run a privileged command.",
                color=discord.Color.yellow(),
            )
            embed.add_field(
                name="Command", value=f"```{command[:900]}```", inline=False
            )
            if task:
                embed.add_field(name="Task context", value=task[:300], inline=False)
            embed.set_footer(text="No response within 5 minutes = auto-denied")
            view = ApprovalView(approval_id, self.redis)

        await channel.send(embed=embed, view=view)

    async def _post_claude_escalation(
        self, payload: dict, source: str, icon: str
    ) -> None:
        """
        Post an agent → Claude escalation request in #claude (or #agent-tasks
        if #claude is not configured). The user approves or denies within 5 min.
        """
        escalation_id = payload.get("escalation_id", str(uuid.uuid4()))
        source_task_id = payload.get("source_task_id", "")
        task = payload.get("task", "(no task provided)")
        reason = payload.get("reason", "")

        # Prefer the dedicated #claude channel so escalations appear alongside
        # direct Claude conversations; fall back to #agent-tasks.
        target_channel_id = (
            int(self.claude_channel_id) if self.claude_channel_id else None
        )
        try:
            ch = (
                await self.fetch_channel(target_channel_id)
                if target_channel_id
                else None
            )
        except Exception:
            ch = None
        if ch is None:
            try:
                ch = await self.fetch_channel(int(self.general_channel_id))
            except Exception:
                log.error("discord_bridge.escalation_channel_not_found")
                return

        embed = discord.Embed(
            title="🤖→✨ Agent Requesting Claude",
            description=(
                f"**{icon} {source.replace('_', ' ').title()}** wants to escalate a task to Claude.\n\n"
                f"**Task:**\n{task[:1500]}"
            ),
            color=discord.Color.from_rgb(138, 43, 226),
        )
        if reason:
            embed.add_field(
                name="Reason for escalation", value=reason[:500], inline=False
            )
        embed.set_footer(text="No response within 5 minutes = auto-denied")

        view = ClaudeEscalationView(
            escalation_id=escalation_id,
            source_task_id=source_task_id,
            source=source,
            task=task,
            redis_client=self.redis,
        )
        await ch.send(embed=embed, view=view)
        log.info(
            "discord_bridge.claude_escalation_posted",
            source=source,
            escalation_id=escalation_id,
            channel=ch.name,
        )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    redis_url = os.environ["REDIS_URL"]
    bot_token = os.environ["DISCORD_BOT_TOKEN"]
    general_channel_id = os.environ["DISCORD_GENERAL_CHANNEL_ID"]
    intake_channel_id = os.environ.get("DISCORD_INTAKE_CHANNEL_ID")
    claude_channel_id = os.environ.get("DISCORD_CLAUDE_CHANNEL_ID")
    control_channel_id = os.environ.get("DISCORD_CONTROL_CHANNEL_ID")
    logs_channel_id = os.environ.get("DISCORD_LOGS_CHANNEL_ID")
    errors_channel_id = os.environ.get("DISCORD_ERRORS_CHANNEL_ID")
    task_tracking_channel_id = os.environ.get("DISCORD_TASK_TRACKING_CHANNEL_ID")
    votes_channel_id = os.environ.get("DISCORD_VOTES_CHANNEL_ID")
    approvals_channel_id = os.environ.get("DISCORD_APPROVALS_CHANNEL_ID")
    knowledge_gap_channel_id = os.environ.get("DISCORD_KNOWLEDGE_GAP_CHANNEL_ID")
    control_helper_url = os.environ.get(
        "CONTROL_HELPER_URL", "http://host.docker.internal:7799"
    )
    guild_id = os.environ.get("DISCORD_GUILD_ID")  # Optional — auto-detected if absent

    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True
    intents.reactions = (
        True  # Required for on_raw_reaction_add (#intake ✅/❌ handling)
    )

    client = DiscordBridgeClient(
        redis_url=redis_url,
        general_channel_id=general_channel_id,
        intake_channel_id=intake_channel_id,
        claude_channel_id=claude_channel_id,
        control_channel_id=control_channel_id,
        logs_channel_id=logs_channel_id,
        errors_channel_id=errors_channel_id,
        task_tracking_channel_id=task_tracking_channel_id,
        votes_channel_id=votes_channel_id,
        approvals_channel_id=approvals_channel_id,
        knowledge_gap_channel_id=knowledge_gap_channel_id,
        control_helper_url=control_helper_url,
        guild_id=guild_id,
        intents=intents,
    )

    log.info(
        "discord_bridge.starting",
        general_channel=general_channel_id,
        intake_channel=intake_channel_id or "(not configured)",
        claude_channel=claude_channel_id or "(not configured)",
        control_channel=control_channel_id or "(not configured)",
        logs_channel=logs_channel_id or "(not configured)",
        errors_channel=errors_channel_id or "(not configured — falling back to #logs)",
        task_tracking_channel=task_tracking_channel_id
        or "(not configured — falling back to #logs)",
        votes_channel=votes_channel_id or "(not configured — falling back to #logs)",
        approvals_channel=approvals_channel_id
        or "(not configured — falling back to #votes)",
        knowledge_gap_channel=knowledge_gap_channel_id
        or "(not configured — falling back to #general)",
    )
    client.run(bot_token)


if __name__ == "__main__":
    main()
