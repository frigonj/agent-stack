"""
agents/discord_bridge/main.py
──────────────────────────────
Discord ↔ Redis Streams bridge.

One bot, not per-agent accounts. Agent identity is shown via embed
color and icon. The bot does two things:

  1. Inbound:  Messages in #agent-tasks → task.created on agents:orchestrator
  2. Outbound: broadcast stream events → formatted Discord embeds

Approval gates:
  - approval.required event → embed with ✅/❌ buttons (5 min timeout)
  - Button click → sets Redis key `approval:{id}` → unblocks executor
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

import discord
import redis.asyncio as aioredis
import structlog

log = structlog.get_logger()

# ── Visual identity per agent ─────────────────────────────────────────────────

AGENT_COLOR = {
    "orchestrator": discord.Color.blue(),
    "document_qa":  discord.Color.green(),
    "code_search":  discord.Color.orange(),
    "executor":     discord.Color.gold(),
    "discord":      discord.Color.blurple(),
}

AGENT_ICON = {
    "orchestrator": "🧠",
    "document_qa":  "📄",
    "code_search":  "🔍",
    "executor":     "⚙️",
}

# Event types we surface to Discord (others are silently ack'd)
VISIBLE_EVENTS = {
    "task.completed",
    "approval.required",
    "system.error",
    "agent.response",
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
            await interaction.response.send_message(
                "Already decided.", ephemeral=True
            )
            return
        self._decided = True

        await self.redis.set(f"approval:{self.approval_id}", decision, ex=600)

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
            await self.redis.set(f"approval:{self.approval_id}", "denied", ex=60)


# ── Bridge client ─────────────────────────────────────────────────────────────

class DiscordBridgeClient(discord.Client):
    """
    discord.Client subclass that owns the Redis connection and background
    consumer task. setup_hook() is the correct place to launch background
    tasks in discord.py 2.x — it runs before the bot starts processing events.
    """

    def __init__(self, redis_url: str, task_channel_id: str, **kwargs):
        super().__init__(**kwargs)
        self.redis_url = redis_url
        self.task_channel_id = task_channel_id
        self.redis: aioredis.Redis | None = None

    async def setup_hook(self) -> None:
        self.redis = await aioredis.from_url(
            self.redis_url, encoding="utf-8", decode_responses=True
        )
        log.info("discord_bridge.redis_connected")
        self.loop.create_task(self._broadcast_consumer())

    async def on_ready(self) -> None:
        log.info("discord_bridge.ready", user=str(self.user), guilds=[g.name for g in self.guilds])
        try:
            channel = await self.fetch_channel(int(self.task_channel_id))
            log.info(
                "discord_bridge.channel_found",
                channel_id=self.task_channel_id,
                channel_name=channel.name,
                guild=channel.guild.name,
            )
        except Exception as exc:
            log.error(
                "discord_bridge.channel_fetch_failed",
                channel_id=self.task_channel_id,
                error=str(exc),
            )

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        log.info(
            "discord_bridge.message_received",
            channel_id=str(message.channel.id),
            expected_channel_id=self.task_channel_id,
            has_content=bool(message.content),
            content_preview=message.content[:40] if message.content else "(empty — Message Content Intent may be disabled)",
        )

        if str(message.channel.id) != self.task_channel_id:
            return

        if not message.content:
            await message.reply(
                "⚠️ Message Content Intent is not enabled. "
                "Go to discord.com/developers/applications → your app → Bot → "
                "Privileged Gateway Intents → enable **Message Content Intent**."
            )
            return

        task_id = str(uuid.uuid4())
        event = {
            "event_id":  str(uuid.uuid4()),
            "type":      "task.created",
            "source":    "discord",
            "task_id":   task_id,
            "timestamp": str(time.time()),
            "payload":   json.dumps({
                "task":               message.content,
                "discord_user":       message.author.display_name,
                "discord_message_id": str(message.id),
                "discord_channel_id": str(message.channel.id),
            }),
        }
        await self.redis.xadd("agents:orchestrator", event)
        await message.add_reaction("⏳")
        log.info("discord_bridge.task_queued", task=message.content[:80])

    # ── Broadcast stream consumer ─────────────────────────────────────────────

    async def _broadcast_consumer(self) -> None:
        group    = "discord_bridge_group"
        stream   = "agents:broadcast"
        consumer = "discord_bridge"

        try:
            await self.redis.xgroup_create(stream, group, id="$", mkstream=True)
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        channel_id = int(self.task_channel_id)

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
                    continue
                for _stream_key, messages in results:
                    for entry_id, data in messages:
                        await self._dispatch(data, channel_id)
                        await self.redis.xack(stream, group, entry_id)
            except aioredis.ResponseError as exc:
                if "NOGROUP" in str(exc):
                    # Stream was deleted and recreated — rebuild the consumer group
                    log.warning("discord_bridge.rebuilding_consumer_group", stream=stream)
                    try:
                        await self.redis.xgroup_create(stream, group, id="$", mkstream=True)
                    except aioredis.ResponseError:
                        pass  # BUSYGROUP is fine — another instance beat us to it
                else:
                    log.error("discord_bridge.consumer_error", error=str(exc))
                    await asyncio.sleep(3)
            except Exception as exc:
                log.error("discord_bridge.consumer_error", error=str(exc))
                await asyncio.sleep(3)

    async def _dispatch(self, data: dict, channel_id: int) -> None:
        event_type = data.get("type", "")
        if event_type not in VISIBLE_EVENTS:
            return

        try:
            channel = await self.fetch_channel(channel_id)
        except Exception as exc:
            log.warning("discord_bridge.channel_not_found", channel_id=channel_id, error=str(exc))
            return

        source  = data.get("source", "system")
        payload = json.loads(data.get("payload", "{}"))
        icon    = AGENT_ICON.get(source, "🤖")
        color   = AGENT_COLOR.get(source, discord.Color.blurple())

        if event_type == "task.completed":
            result = payload.get("result", "(no output)")
            embed = discord.Embed(
                title=f"{icon} {source.replace('_', ' ').title()} — Done",
                description=result[:4000],
                color=color,
            )
            await channel.send(embed=embed)

        elif event_type == "approval.required":
            await self._post_approval(payload, channel, source, icon)

        elif event_type in ("system.error", "agent.response"):
            text = payload.get("error") or payload.get("response") or str(payload)
            embed = discord.Embed(
                title=f"{icon} {source.replace('_', ' ').title()}",
                description=f"```{text[:1500]}```" if event_type == "system.error" else text[:4000],
                color=discord.Color.dark_red() if event_type == "system.error" else color,
            )
            await channel.send(embed=embed)

    async def _post_approval(
        self, payload: dict, channel, source: str, icon: str
    ) -> None:
        approval_id = payload.get("approval_id", "")
        command     = payload.get("command", "")
        task        = payload.get("task", "")

        embed = discord.Embed(
            title="🔐 Approval Required",
            description=f"**{icon} {source}** wants to run a privileged command.",
            color=discord.Color.yellow(),
        )
        embed.add_field(name="Command", value=f"```{command[:900]}```", inline=False)
        if task:
            embed.add_field(name="Task context", value=task[:300], inline=False)
        embed.set_footer(text="No response within 5 minutes = auto-denied")

        view = ApprovalView(approval_id, self.redis)
        await channel.send(embed=embed, view=view)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    redis_url        = os.environ["REDIS_URL"]
    bot_token        = os.environ["DISCORD_BOT_TOKEN"]
    task_channel_id  = os.environ["DISCORD_TASK_CHANNEL_ID"]

    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True

    client = DiscordBridgeClient(
        redis_url=redis_url,
        task_channel_id=task_channel_id,
        intents=intents,
    )

    log.info("discord_bridge.starting")
    client.run(bot_token)


if __name__ == "__main__":
    main()
