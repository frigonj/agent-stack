#!/bin/bash
# Description: Dump the agent memory PostgreSQL database to a timestamped backup file in /workspace/docs/
# Usage: ./backup-memory-db.sh
# Tags: postgres, backup, database, memory
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="/workspace/docs/agentmem_backup_${TIMESTAMP}.sql"

echo "=== Backing up agentmem to $OUTFILE ==="
docker exec agent_postgres pg_dump -U agent agentmem > "$OUTFILE"

SIZE=$(du -sh "$OUTFILE" | cut -f1)
echo "Done. Backup size: $SIZE"
echo "File: $OUTFILE"
