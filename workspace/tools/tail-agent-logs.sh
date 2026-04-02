#!/bin/bash
# Description: Tail the last N log lines from an agent container (default: orchestrator, 100 lines)
# Usage: ./tail-agent-logs.sh [container_name] [lines]
# Tags: logs, docker, debug, monitoring
set -euo pipefail

CONTAINER="${1:-agent_orchestrator}"
LINES="${2:-100}"

echo "=== Last $LINES lines from $CONTAINER ==="
docker logs --tail "$LINES" "$CONTAINER" 2>&1
