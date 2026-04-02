# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# TeX Live (minimal complete LaTeX installation) + build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-fonts-recommended \
    texlive-latex-extra \
    latexmk \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Dependency layer ──────────────────────────────────────────────────────────
COPY requirements.txt .
COPY emrys-src/ /tmp/emrys-src/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt && \
    pip install "/tmp/emrys-src[vectors]" && \
    pip install pylatex pypdf google-api-python-client google-auth && \
    rm -rf /tmp/emrys-src

COPY core/ ./core/
COPY agents/document_qa/ ./agent/

CMD ["python", "-m", "agent.main"]
