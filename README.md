# Agent Stack

A self-hosted, event-driven multi-agent RAG system that runs entirely on local hardware.
LM Studio handles inference, Redis Streams handle short-term memory, PostgreSQL + pgvector
handle long-term memory and semantic search, and a Discord bot is the human interface.

---

## Architecture

```
Discord (#agent-tasks)
        │  user message
        ▼
┌───────────────────┐
│  discord_bridge   │  ← single bot, bridges Discord ↔ Redis
└───────┬───────────┘
        │ task.created
        ▼
┌───────────────────┐     task.assigned      ┌─────────────────┐
│   orchestrator    │ ─────────────────────► │   document_qa   │
│                   │ ─────────────────────► │   code_search   │
│  plans, routes,   │ ─────────────────────► │   executor      │
│  aggregates       │                        └────────┬────────┘
└───────┬───────────┘                                 │ task.completed
        │ task.completed (broadcast)                  │
        ▼                                             │
Discord embed ◄──────────────────────────────────────┘

Short-term memory          Long-term memory
─────────────────          ────────────────
Redis Streams              PostgreSQL + pgvector
• Live task state          • Session handoffs
• Inter-agent events       • Knowledge base (FTS + semantic)
• Tool results             • Crash recovery
• Ephemeral by design      • Persistent across restarts

Inference: LM Studio (local, OpenAI-compatible API, port 1234)
```

---

## Services

| Container | Image | Purpose |
|---|---|---|
| `agent_redis` | `redis:7-alpine` | Event bus, short-term memory |
| `agent_postgres` | `pgvector/pgvector:pg16` | Long-term memory, vector search |
| `agent_orchestrator` | Built from `agent.Dockerfile` | Plans tasks, routes to specialists, aggregates results |
| `agent_document_qa` | Built from `agent.Dockerfile` | Answers questions from docs in `/workspace/docs` |
| `agent_code_search` | Built from `agent.Dockerfile` | Searches and analyses code in `/workspace/repos` |
| `agent_executor` | Built from `agent.Dockerfile` | Runs shell commands with approval gates |
| `agent_discord_bridge` | Built from `bridge.Dockerfile` | Discord ↔ Redis bridge, approval UI |

---

## Hardware

| Component | Spec |
|---|---|
| CPU | Intel Core i7-11700F @ 2.50GHz (8c/16t) |
| RAM | 64GB DDR4-3200 |
| GPU | NVIDIA RTX 3070 (8GB VRAM) |
| Storage | 2TB WD Black SN7100 NVMe PCIe 4.0 |
| OS | Windows 11 + WSL2 + Docker Desktop |

Recommended models:
- **Orchestrator / Doc QA / Executor:** Qwen 3.5 8B Q4_K_M
- **Code Search:** Qwen2.5-Coder 14B Q4_K_M

---

## Setup

### Prerequisites
- Docker Desktop (Windows)
- LM Studio with a model loaded and local server started (port 1234)
- A Discord bot token and channel ID (see Discord setup below)
- Git

### First-time setup

```bash
git clone https://github.com/frigonj/agent-stack.git
cd agent-stack

# Clone Emrys source (required for Docker builds)
git clone https://github.com/NuAvalon/emrys.git emrys-src

# Copy and fill in environment variables
cp config/.env.example .env
# Edit .env with your values

# Build and start
docker compose up --build
```

### Environment variables (`.env`)

```env
LM_STUDIO_URL=http://host.docker.internal:1234
LM_STUDIO_MODEL=qwen2.5-14b

REDIS_URL=redis://redis:6379
DATABASE_URL=postgresql://agent:agent@postgres:5432/agentmem

DISCORD_BOT_TOKEN=your-token
DISCORD_TASK_CHANNEL_ID=your-channel-id

HF_TOKEN=                     # optional — prevents HuggingFace rate limits
LOG_LEVEL=INFO
```

---

## Discord Setup

1. Go to [discord.com/developers/applications](https://discord.com/developers/applications) → New Application → Bot
2. Under **Bot → Privileged Gateway Intents** enable all three (Presence, Server Members, Message Content)
3. Generate invite URL with permissions integer `3967583347801328` and scopes `bot`
4. Invite the bot to your server
5. Enable **Developer Mode** in Discord → right-click your task channel → **Copy Channel ID**
6. Add `DISCORD_BOT_TOKEN` and `DISCORD_TASK_CHANNEL_ID` to `.env`

Usage: send any message in `#agent-tasks` — the bot will ⏳ react, process via the agent stack, and reply with an embed.

---

## Approval Gates

The executor requires explicit approval before running privileged commands. When triggered, the Discord bridge posts an embed with **✅ Approve** / **❌ Deny** buttons. No response within 5 minutes auto-denies.

| Tier | Commands | Behaviour |
|---|---|---|
| Safe | `ls cat find grep head tail wc pwd diff sort uniq` | Run immediately |
| Gated | `curl python python3 pip git` | Held for Discord approval |
| Blocked | anything else | Rejected outright |

---

## Memory Architecture

### Two-layer model

| Layer | Technology | Lifetime | Purpose |
|---|---|---|---|
| Short-term | Redis Streams | Ephemeral (TTL / MAXLEN) | Live task state, inter-agent events |
| Long-term | PostgreSQL + pgvector | Permanent | Knowledge base, session handoffs, crash recovery |

### Recall flow

Agents call `await self.recall(query)` before starting a task. This runs a pgvector cosine
similarity search (semantic) with FTS fallback. Meaningful findings are staged via
`self.stage_finding()` and batch-promoted to PostgreSQL on session close.

### Schema

```sql
agent_status   -- active session state per agent
handoffs       -- session summaries written on clean shutdown
knowledge      -- promoted findings (content + 384-dim embedding)
knowledge_fts  -- GIN index for full-text search
```

---

## Event Bus

All inter-agent communication flows through Redis Streams.

| Stream | Consumers |
|---|---|
| `agents:orchestrator` | orchestrator |
| `agents:document_qa` | document_qa |
| `agents:code_search` | code_search |
| `agents:executor` | executor |
| `agents:broadcast` | all agents + discord_bridge |

### Event types

| Type | Producer | Consumer |
|---|---|---|
| `task.created` | discord_bridge | orchestrator |
| `task.assigned` | orchestrator | specialist agents |
| `task.completed` | specialists → orchestrator → broadcast | discord_bridge |
| `approval.required` | executor | discord_bridge |
| `agent.started` | any agent | orchestrator |
| `memory.promoted` | any agent | broadcast |
| `system.error` | any agent | discord_bridge |

---

## Adding a New Agent

```python
# agents/my_agent/main.py
from core.base_agent import BaseAgent, run_agent
from core.config import Settings
from core.events.bus import Event, EventType

class MyAgent(BaseAgent):
    async def handle_event(self, event: Event) -> None:
        if event.type == EventType.TASK_ASSIGNED:
            if event.payload.get("assigned_to") == "my_agent":
                result = "my result"
                self.stage_finding(result, topic="my_topic")
                await self.emit(EventType.TASK_COMPLETED,
                                payload={"result": result, "task_id": event.task_id},
                                target="orchestrator")

if __name__ == "__main__":
    run_agent(MyAgent(Settings()))
```

Then add a service block to `docker-compose.yml` (copy any existing agent, change `AGENT_MODULE` and `AGENT_ROLE`) and add the role to `SPECIALIST_ROLES` in `agents/orchestrator/main.py`.

---

## Running Tests

```bash
# Unit tests (no infrastructure needed)
pytest tests/unit/ -v

# Integration tests (Redis required)
docker compose up -d redis
pytest tests/integration/ -v
```

---

## Common Commands

```bash
# Start everything
docker compose up -d

# Tail logs for a specific agent
docker compose logs -f orchestrator

# Purge all Redis streams (clears message backlog)
docker exec agent_redis redis-cli DEL agents:broadcast agents:orchestrator \
  agents:document_qa agents:code_search agents:executor

# Inspect knowledge base
docker exec agent_postgres psql -U agent agentmem \
  -c "SELECT agent, topic, left(content,80) FROM knowledge ORDER BY created_at DESC LIMIT 10;"

# Hard reset (destroys all data)
docker compose down -v

# Rebuild a single service
docker compose build orchestrator && docker compose up -d orchestrator
```
