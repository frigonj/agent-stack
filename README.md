# Agent Stack

A local, event-driven multi-agent RAG system built on LM Studio, Redis Streams, and Emrys.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AGENT PIPELINE                       │
├──────────────────────────┬──────────────────────────────────┤
│   SHORT-TERM MEMORY      │   LONG-TERM MEMORY               │
│   Redis Streams          │   Emrys (SQLite + vectors)       │
│                          │                                  │
│   • Live task state      │   • Session handoffs             │
│   • Inter-agent events   │   • Knowledge base               │
│   • Tool call results    │   • Crash recovery               │
│   • Ephemeral by design  │   • Persistent by design         │
└──────────────────────────┴──────────────────────────────────┘
         │                              ▲
         │   Memory Promotion           │
         └──────────────────────────────┘
                (meaningful findings only)

Agents:
  orchestrator  →  plans tasks, delegates, aggregates results
  document_qa   →  answers questions from PDFs and text files
  code_search   →  indexes and searches codebases
  executor      →  runs tools, shell commands, file operations

Inference:
  LM Studio (local, OpenAI-compatible API)
  Recommended model: Qwen2.5 14B Q4_K_M
```

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Docker Desktop / Engine | 24+ | Docker Compose v2 included |
| Python | 3.11+ | For local dev and tests |
| LM Studio | Latest | Running with a model loaded |

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/YOUR_USERNAME/agent-stack.git
cd agent-stack
bash scripts/setup.sh
```

### 2. Configure LM Studio

- Open LM Studio and load a model (recommended: Qwen2.5 14B Q4_K_M)
- Enable the local server: **Local Server → Start Server**
- Default port: `1234`

**WSL2 note:** Docker containers reach your Windows host via `host.docker.internal`.
This is set as the default in `.env`. No changes needed for most setups.

### 3. Start the stack

```bash
docker compose up
```

Or start just the infrastructure while developing agents locally:

```bash
docker compose up redis emrys
```

### 4. Watch agent logs

```bash
docker compose logs -f orchestrator
docker compose logs -f document_qa
```

## Sending a Task

The orchestrator listens on the `agents:orchestrator` Redis stream for `task.created` events.

```python
import asyncio
import redis.asyncio as aioredis
import json

async def send_task(task: str):
    r = await aioredis.from_url("redis://localhost:6379", decode_responses=True)
    await r.xadd("agents:orchestrator", {
        "event_id": "manual-001",
        "type": "task.created",
        "source": "user",
        "task_id": "task-001",
        "timestamp": "1700000000.0",
        "payload": json.dumps({"task": task}),
    })
    await r.aclose()

asyncio.run(send_task("Search the codebase for authentication patterns"))
```

## Memory Promotion

Short-term events that matter get promoted to Emrys long-term memory:

```
Redis Stream Event              Emrys Knowledge Base
──────────────────              ────────────────────
task.created         →  agent processes
agent.tool_result    →  if meaningful → store_knowledge()
task.completed       →  session handoff written on shutdown
                        knowledge searchable next session
```

Agents use two promotion patterns:
- `stage_finding()` — batches findings, promotes on clean shutdown
- `promote_now()` — immediately promotes critical findings mid-session

## Project Structure

```
agent-stack/
├── .github/workflows/
│   └── ci.yml                  # Full CI: lint → unit → integration → docker → smoke
├── agents/
│   ├── orchestrator/main.py    # Planner and router
│   ├── document_qa/main.py     # PDF and text Q&A
│   ├── code_search/main.py     # Repository search and analysis
│   └── executor/main.py        # Tool and command execution
├── core/
│   ├── base_agent.py           # Base class all agents inherit
│   ├── config.py               # Pydantic settings (env-driven)
│   ├── events/bus.py           # Redis Streams abstraction
│   └── memory/long_term.py     # Emrys MCP client
├── docker/
│   ├── agent.Dockerfile        # Shared agent image
│   └── emrys.Dockerfile        # Emrys MCP server image
├── tests/
│   ├── unit/                   # Pure logic tests (no infra)
│   └── integration/            # Redis + Emrys required
├── config/
│   └── .env.example            # Environment variable template
├── scripts/
│   └── setup.sh                # One-shot setup script
├── docker-compose.yml
├── requirements.txt
└── pytest.ini
```

## Adding a New Agent

1. Create `agents/my_agent/main.py`
2. Inherit from `BaseAgent` and implement `handle_event()`
3. Add the service to `docker-compose.yml` (copy an existing agent block)
4. Add `AGENT_ROLE=my_agent` to its environment
5. Run `docker compose up --build my_agent`

```python
from core.base_agent import BaseAgent, run_agent
from core.config import Settings
from core.events.bus import Event, EventType

class MyAgent(BaseAgent):
    async def handle_event(self, event: Event) -> None:
        if event.type == EventType.TASK_ASSIGNED:
            if event.payload.get("assigned_to") == "my_agent":
                # do work
                await self.emit(EventType.TASK_COMPLETED, payload={"result": "done"}, target="orchestrator")

if __name__ == "__main__":
    run_agent(MyAgent(Settings()))
```

## Running Tests

```bash
source .venv/bin/activate

# Unit tests only (no infra needed)
pytest tests/unit/ -v

# All tests (requires Redis running)
docker compose up -d redis emrys
pytest tests/ -v
```

## CI Pipeline

Every push to `main` or `develop` runs:

1. **Lint** — Ruff + Mypy
2. **Unit tests** — no infrastructure required
3. **Integration tests** — spins up Redis + Emrys via GitHub Actions services
4. **Docker build** — validates all four agent images build cleanly
5. **Stack smoke test** — starts Redis + Emrys via Compose, verifies health

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LM_STUDIO_URL` | `http://host.docker.internal:1234` | LM Studio API endpoint |
| `LM_STUDIO_MODEL` | `qwen2.5-14b` | Model name as shown in LM Studio |
| `REDIS_URL` | `redis://redis:6379` | Redis connection string |
| `EMRYS_URL` | `http://emrys:8000` | Emrys MCP server URL |
| `AGENT_ROLE` | `agent` | This agent's role identifier |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## License

MIT
