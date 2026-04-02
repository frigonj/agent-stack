#!/bin/bash
# Description: Query the agent long-term memory PostgreSQL database with a custom SQL statement
# Usage: ./query-memory-db.sh "SELECT topic, COUNT(*) FROM knowledge GROUP BY topic ORDER BY count DESC;"
# Tags: postgres, memory, database, query, knowledge
set -euo pipefail

SQL="${1:-SELECT topic, COUNT(*) FROM knowledge GROUP BY topic ORDER BY count DESC LIMIT 20;}"

echo "=== Executing: $SQL ==="
docker exec agent_postgres psql -U agent -d agentmem -c "$SQL"
