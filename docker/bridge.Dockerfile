# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Minimal deps — no LM Studio, no psycopg, no sentence-transformers
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        "discord.py>=2.3.0" \
        "redis>=5.0.0" \
        "structlog>=24.0.0" \
        "httpx>=0.27.0"

COPY agents/discord_bridge/ ./agent/

CMD ["python", "-m", "agent.main"]
