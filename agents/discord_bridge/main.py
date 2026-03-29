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
    "discord.action",
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

    def __init__(self, redis_url: str, task_channel_id: str, guild_id: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.redis_url = redis_url
        self.task_channel_id = task_channel_id
        self.guild_id = int(guild_id) if guild_id else None
        self.redis: aioredis.Redis | None = None
        # task_id → (channel_id, message_id) for ⏳→✅ reaction update
        self._pending_messages: dict[str, tuple[int, int]] = {}

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
            # Auto-detect guild_id from the task channel if not explicitly set
            if not self.guild_id and hasattr(channel, "guild"):
                self.guild_id = channel.guild.id
                log.info("discord_bridge.guild_detected", guild_id=self.guild_id, guild=channel.guild.name)
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
        message_id = str(message.id)

        event = {
            "event_id":  str(uuid.uuid4()),
            "type":      "task.created",
            "source":    "discord",
            "task_id":   task_id,
            "timestamp": str(time.time()),
            "payload":   json.dumps({
                "task":               message.content,
                "discord_user":       message.author.display_name,
                "discord_message_id": message_id,
                "discord_channel_id": str(message.channel.id),
            }),
        }
        await self.redis.xadd("agents:orchestrator", event)
        # Track pending message for reaction update when reply arrives
        self._pending_messages[task_id] = (message.channel.id, message.id)
        await message.add_reaction("⏳")
        log.info("discord_bridge.task_queued", task=message.content[:80], task_id=task_id)

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
            task_id = payload.get("task_id")
            embed = discord.Embed(
                title=f"{icon} {source.replace('_', ' ').title()}",
                description=result[:4000],
                color=color,
            )

            # Try to reply to the original user message so it threads nicely
            origin = self._pending_messages.pop(task_id, None) if task_id else None
            if origin:
                ch_id, msg_id = origin
                try:
                    ch = await self.fetch_channel(ch_id)
                    msg = await ch.fetch_message(msg_id)
                    await msg.remove_reaction("⏳", self.user)
                    await msg.add_reaction("✅")
                    await msg.reply(embed=embed, mention_author=False)
                    log.info("discord_bridge.reply_sent", task_id=task_id)
                    return
                except Exception as exc:
                    log.warning("discord_bridge.reply_failed", error=str(exc))

            await channel.send(embed=embed)
            log.info("discord_bridge.message_sent", event_type=event_type, source=source)

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

        elif event_type == "discord.action":
            await self._execute_discord_action(payload, data.get("event_id", ""))

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
        action  = payload.get("action", "")
        task_id = payload.get("task_id", "")

        async def _ack(result: str, ok: bool = True) -> None:
            await self.redis.xadd(
                "agents:broadcast",
                {
                    "event_id":  str(uuid.uuid4()),
                    "type":      "discord.action.done",
                    "source":    "discord_bridge",
                    "task_id":   task_id,
                    "timestamp": str(time.time()),
                    "payload":   json.dumps({
                        "action": action, "result": result, "ok": ok,
                        "triggering_event_id": event_id,
                    }),
                },
                maxlen=10_000,
                approximate=True,
            )
            log.info("discord_bridge.action_done", action=action, ok=ok, result=result[:100])

        guild = self.get_guild(self.guild_id) if self.guild_id else None
        if not guild and action not in ("send_message",):
            await _ack("Guild not found — set DISCORD_GUILD_ID in .env", ok=False)
            return

        try:
            if action == "send_message":
                ch_id  = payload.get("channel_id") or self.task_channel_id
                text   = payload.get("content", "")
                ch = await self.fetch_channel(int(ch_id))
                await ch.send(text[:2000])
                await _ack(f"Message sent to <#{ch_id}>")

            elif action == "create_channel":
                name      = payload.get("name", "new-channel")
                topic     = payload.get("topic", "")
                category_name = payload.get("category")
                category  = None
                if category_name:
                    category = discord.utils.get(guild.categories, name=category_name)
                ch = await guild.create_text_channel(name=name, topic=topic, category=category)
                await _ack(f"Created #{ch.name} (id={ch.id})")

            elif action == "delete_channel":
                ch_id  = payload.get("channel_id")
                ch_name = payload.get("name")
                if ch_id:
                    ch = await self.fetch_channel(int(ch_id))
                elif ch_name:
                    ch = discord.utils.get(guild.text_channels, name=ch_name)
                else:
                    await _ack("Provide channel_id or name", ok=False)
                    return
                if not ch:
                    await _ack("Channel not found", ok=False)
                    return
                await ch.delete(reason=payload.get("reason", "Deleted by agent"))
                await _ack(f"Deleted #{ch.name}")

            elif action == "rename_channel":
                ch_id  = payload.get("channel_id")
                new_name = payload.get("name", "")
                ch = await self.fetch_channel(int(ch_id))
                old = ch.name
                await ch.edit(name=new_name)
                await _ack(f"Renamed #{old} → #{new_name}")

            elif action == "set_topic":
                ch_id  = payload.get("channel_id")
                topic  = payload.get("topic", "")
                ch = await self.fetch_channel(int(ch_id))
                await ch.edit(topic=topic)
                await _ack(f"Topic set on #{ch.name}")

            elif action == "create_category":
                name = payload.get("name", "New Category")
                cat  = await guild.create_category(name=name)
                await _ack(f"Created category '{cat.name}' (id={cat.id})")

            elif action == "pin_message":
                ch_id  = payload.get("channel_id")
                msg_id = payload.get("message_id")
                ch  = await self.fetch_channel(int(ch_id))
                msg = await ch.fetch_message(int(msg_id))
                await msg.pin()
                await _ack(f"Pinned message {msg_id} in #{ch.name}")

            elif action == "list_channels":
                lines = [f"#{ch.name} (id={ch.id})" for ch in guild.text_channels]
                await _ack("Channels:\n" + "\n".join(lines))

            else:
                await _ack(f"Unknown action: {action}", ok=False)

        except Exception as exc:
            log.error("discord_bridge.action_error", action=action, error=str(exc))
            await _ack(f"Error: {exc}", ok=False)

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
    guild_id         = os.environ.get("DISCORD_GUILD_ID")  # Optional — auto-detected if absent

    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True

    client = DiscordBridgeClient(
        redis_url=redis_url,
        task_channel_id=task_channel_id,
        guild_id=guild_id,
        intents=intents,
    )

    log.info("discord_bridge.starting")
    client.run(bot_token)


if __name__ == "__main__":
    main()
