#!/usr/bin/env bash
# scripts/setup.sh
# ─────────────────
# One-shot setup for the agent stack.
# Run this once after cloning the repo.
# Works on Linux, macOS, and WSL2.

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
error()   { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ── Prereq checks ─────────────────────────────────────────────────────────────

info "Checking prerequisites..."

command -v docker  >/dev/null 2>&1 || error "Docker not found. Install Docker Desktop (Windows/macOS) or Docker Engine (Linux/WSL2)."
command -v python3 >/dev/null 2>&1 || error "Python 3 not found."

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Python version: $PYTHON_VERSION"
if [[ "$PYTHON_VERSION" < "3.10" ]]; then
  error "Python 3.10+ required. Found $PYTHON_VERSION."
fi

# ── .env setup ────────────────────────────────────────────────────────────────

if [ ! -f .env ]; then
  info "Creating .env from template..."
  cp config/.env.example .env
  warn "Edit .env and set LM_STUDIO_URL to your machine's IP if running in WSL2."
  warn "Default: LM_STUDIO_URL=http://host.docker.internal:1234"
else
  info ".env already exists, skipping."
fi

# ── Python venv (for local dev / testing) ────────────────────────────────────

if [ ! -d .venv ]; then
  info "Creating Python virtual environment..."
  python3 -m venv .venv
fi

info "Installing Python dependencies..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

info "Installing Emrys from GitHub (not yet on PyPI)..."
.venv/bin/pip install --quiet "emrys[vectors] @ git+https://github.com/NuAvalon/emrys.git"

# Also clone Emrys source for Docker builds (avoids network issues inside build context)
if [ ! -d emrys-src ]; then
  info "Cloning Emrys source for Docker builds..."
  git clone --quiet https://github.com/NuAvalon/emrys.git emrys-src
else
  info "Updating Emrys source..."
  git -C emrys-src pull --quiet
fi

# ── Emrys init ────────────────────────────────────────────────────────────────

info "Initializing Emrys persistent memory..."
mkdir -p .persist
.venv/bin/emrys init --mode tool --persist-dir .persist 2>/dev/null || true

# ── Workspace directories ─────────────────────────────────────────────────────

info "Creating workspace directories..."
mkdir -p workspace/{docs,repos}

# ── Docker build ─────────────────────────────────────────────────────────────

info "Building Docker images..."
docker compose build --quiet

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Next steps (Windows):"
echo "  1. Make sure LM Studio is running with a model loaded"
echo "  2. Double-click  scripts\\start.bat  (or run it from a terminal)"
echo "     • Starts the host restart helper on port 7799"
echo "     • Starts Redis, Postgres, Discord bridge, and Orchestrator"
echo "  3. Send a task from Discord — the orchestrator spins up agents as needed"
echo ""
echo "  Next steps (Linux / WSL2):"
echo "  1. Make sure LM Studio is running with a model loaded"
echo "  2. Start the stack:"
echo "       docker compose up -d"
echo "  3. (Optional) start the host restart helper:"
echo "       python scripts/host_restart_helper.py &"
echo "  4. Send a task from Discord — the orchestrator spins up agents as needed"
echo ""
echo "  Logs:"
echo "    docker compose logs -f orchestrator"
echo "    docker compose logs -f discord_bridge"
echo ""
echo "  To run tests:"
echo "    source .venv/bin/activate"
echo "    pytest tests/"
echo ""
echo "  Windows auto-start tip:"
echo "    Win+R → shell:startup → drop a shortcut to scripts\\start.bat there"
echo ""
