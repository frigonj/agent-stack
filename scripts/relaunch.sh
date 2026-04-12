#!/usr/bin/env bash
# =============================================================================
# relaunch.sh — Agent Stack relaunch helper
#
# Usage:
#   ./scripts/relaunch.sh [mode]
#
# Modes:
#   code      Code-only changes (volume-mounted agents). Restarts always-on
#             services. No rebuild. Ephemeral agents pick up changes on next
#             spawn automatically.
#
#   rebuild   Dockerfile or dependency changes. Rebuilds all images, then
#             restarts always-on services.
#
#   rebuild <svc>
#             Rebuild and restart a single service only.
#             e.g.  ./scripts/relaunch.sh rebuild orchestrator
#
#   flush     Rebuild + clear Redis stream backlog. Use after schema changes,
#             event type renames, or any change that could cause stale events
#             to be mishandled on replay.
#
#   reset     Full hard reset — destroys all data (Redis + Postgres volumes),
#             then rebuilds and starts fresh. Use for breaking schema migrations
#             or when you want a clean slate.
#
# Default (no argument): same as "rebuild"
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."   # always run from repo root

# Always-on services that need an explicit restart.
# Ephemeral agents (executor, research, document_qa, code_search,
# claude_code_agent, developer) are spun up on demand by the orchestrator
# and will automatically use the new image on their next spawn.
ALWAYS_ON="redis postgres orchestrator discord_bridge optimizer"

# Redis streams to flush. Add any new agent streams here if you create one.
REDIS_STREAMS="agents:broadcast agents:orchestrator agents:document_qa \
               agents:code_search agents:executor agents:research \
               agents:developer agents:claude_code agents:optimizer"

MODE="${1:-rebuild}"

# ── helpers ──────────────────────────────────────────────────────────────────

log()  { echo -e "\n\033[1;36m>>> $*\033[0m"; }
warn() { echo -e "\033[1;33mWARN: $*\033[0m"; }
ok()   { echo -e "\033[1;32m OK\033[0m"; }

flush_redis() {
    log "Flushing Redis streams (clears stale event backlog)..."
    # shellcheck disable=SC2086
    docker exec agent_redis redis-cli DEL $REDIS_STREAMS
    ok
}

# ── modes ─────────────────────────────────────────────────────────────────────

case "$MODE" in

  code)
    # -------------------------------------------------------------------------
    # CODE-ONLY: agent source is volume-mounted, so no rebuild is needed.
    # Just restart the always-on services to reload their in-memory state.
    # Ephemeral agents will pick up changes on their next spawn.
    # -------------------------------------------------------------------------
    log "Mode: code — restarting always-on services (no rebuild)"
    # shellcheck disable=SC2086
    docker compose restart $ALWAYS_ON
    ok
    ;;

  rebuild)
    # -------------------------------------------------------------------------
    # REBUILD: Dockerfile changed, new dependency added, or core/ modified.
    # Rebuilds all images (uses Docker layer cache where possible, so fast for
    # unchanged layers). Then brings always-on services up with the new image.
    # -------------------------------------------------------------------------
    SVC="${2:-}"
    if [[ -n "$SVC" ]]; then
      log "Mode: rebuild — single service: $SVC"
      docker compose build "$SVC"
      docker compose up -d "$SVC"
    else
      log "Mode: rebuild — all services"
      docker compose build
      # shellcheck disable=SC2086
      docker compose up -d $ALWAYS_ON
    fi
    ok
    ;;

  flush)
    # -------------------------------------------------------------------------
    # FLUSH: Significant change to event types, payload structure, or Redis
    # stream consumers. Rebuilds everything, then deletes all Redis streams
    # so no stale events are replayed. Postgres data is preserved.
    # -------------------------------------------------------------------------
    log "Mode: flush — rebuild + clear Redis streams"
    warn "This deletes all pending Redis events. Postgres data is preserved."
    read -rp "Continue? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

    docker compose build
    flush_redis
    # shellcheck disable=SC2086
    docker compose up -d $ALWAYS_ON
    ok
    ;;

  reset)
    # -------------------------------------------------------------------------
    # RESET: Full hard reset. Destroys ALL data — Redis streams, Postgres
    # knowledge base, agent_status, handoffs. Use for breaking schema changes
    # or when you want a completely clean slate.
    # -------------------------------------------------------------------------
    log "Mode: reset — full hard reset (destroys all data)"
    warn "This will DELETE all Redis and Postgres data (volumes)."
    warn "All agent knowledge, task history, and session state will be lost."
    read -rp "Type 'yes' to confirm: " confirm
    [[ "$confirm" == "yes" ]] || { echo "Aborted."; exit 0; }

    docker compose down -v
    docker compose build
    # shellcheck disable=SC2086
    docker compose up -d $ALWAYS_ON
    ok
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [code|rebuild [svc]|flush|reset]"
    exit 1
    ;;

esac

# ── post-launch status ────────────────────────────────────────────────────────

log "Service status:"
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
