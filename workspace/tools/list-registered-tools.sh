#!/bin/bash
# Description: List all tools registered in the shared PostgreSQL tool registry, grouped by owner agent
# Usage: ./list-registered-tools.sh
# Tags: tools, registry, postgres, agents
set -euo pipefail

echo "=== Registered Tools by Agent ==="
docker exec agent_postgres psql -U agent -d agentmem -c \
    "SELECT owner_agent, name, LEFT(description, 80) AS description FROM tools ORDER BY owner_agent, name;"

echo ""
echo "=== Total tool count per agent ==="
docker exec agent_postgres psql -U agent -d agentmem -c \
    "SELECT owner_agent, COUNT(*) AS tools FROM tools GROUP BY owner_agent ORDER BY tools DESC;"
