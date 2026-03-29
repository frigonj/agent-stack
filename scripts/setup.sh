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
echo "  Next steps:"
echo "  1. Make sure LM Studio is running with a model loaded"
echo "  2. Run:  docker compose up"
echo "  3. Check logs:  docker compose logs -f orchestrator"
echo ""
echo "  To run tests:"
echo "    source .venv/bin/activate"
echo "    pytest tests/"
echo ""
