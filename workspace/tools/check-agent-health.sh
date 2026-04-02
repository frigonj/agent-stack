#!/bin/bash
# Description: Check the health and status of all agent containers
# Usage: ./check-agent-health.sh
# Tags: health, agents, docker, monitoring
set -euo pipefail

echo "=== Agent Container Status ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.RunningFor}}" | grep -E "agent_|NAMES"

echo ""
echo "=== Redis Agent Status Keys ==="
docker exec agent_redis redis-cli keys 'agent:status:*' 2>/dev/null | while read key; do
    val=$(docker exec agent_redis redis-cli get "$key" 2>/dev/null)
    echo "  $key = $val"
done

echo ""
echo "=== Event Stream Lengths ==="
docker exec agent_redis redis-cli --no-auth-warning keys 'agents:*' 2>/dev/null | while read stream; do
    len=$(docker exec agent_redis redis-cli xlen "$stream" 2>/dev/null)
    echo "  $stream: $len messages"
done
