# MemPalace

This project uses MemPalace for persistent codebase memory. The palace is stored at `~/.mempalace/palace` (WSL) and indexed under the `agent_stack` wing.

## Using MemPalace

Run these from the project root in WSL with the venv active:

```bash
cd '/mnt/c/Users/frigo/Documents/Coding Projects/agent-stack'
source .venv/bin/activate
```

**At session start — get oriented:**
```bash
python -m mempalace wake-up --wing agent_stack
```

**Search for anything:**
```bash
python -m mempalace search "your query here"
python -m mempalace search "your query" --wing agent_stack --room agents
```

**Check coverage:**
```bash
python -m mempalace status
```

**Manually re-index (normally runs automatically after git commit):**
```bash
python -m mempalace mine . --wing agent_stack
python -m mempalace compress
```

## Rooms

| Room | Contents |
|------|----------|
| agents | orchestrator, executor, discord_bridge, ephemeral agents |
| core | event bus, long-term memory, schema, Redis/Postgres logic |
| emrys_src | LLM inference layer, qwen model integration |
| docker | docker-compose, Dockerfiles, service configs |
| scripts | setup, deploy, relaunch scripts |
| testing | unit, integration, kpi, perf test suites |
| workspace | generated docs, artifacts |
| general | config, env files, README |

## Auto-indexing

MemPalace mine + compress runs automatically after every `git commit` via a PostToolUse hook in `.claude/settings.json`.
