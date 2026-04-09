FROM python:3.11-slim

WORKDIR /app

# System deps + Docker CLI (for docker commands via run_shell)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential ca-certificates gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
       | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
       https://download.docker.com/linux/debian bookworm stable" \
       > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        "anthropic>=0.40.0" \
        "redis>=5.0.0" \
        "structlog>=24.0.0"

COPY agents/claude_code_agent/ ./agent/

CMD ["python", "-m", "agent.main"]
