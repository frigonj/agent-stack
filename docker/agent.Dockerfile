# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Dependency layer ──────────────────────────────────────────────────────────
# IMPORTANT: ARG AGENT_MODULE is declared AFTER this block.
# Docker keys the layer cache on all preceding ARG values — if AGENT_MODULE
# were declared before pip install, each agent would get its own cache entry
# and packages would reinstall 4x on every build.
COPY requirements.txt .
COPY emrys-src/ /tmp/emrys-src/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt && \
    pip install "/tmp/emrys-src[vectors]" && \
    rm -rf /tmp/emrys-src

# ── Agent-specific layer (declared after deps so cache above is shared) ───────
ARG AGENT_MODULE
ENV AGENT_MODULE=${AGENT_MODULE}

COPY core/ ./core/
COPY agents/${AGENT_MODULE}/ ./agent/

CMD ["python", "-m", "agent.main"]
