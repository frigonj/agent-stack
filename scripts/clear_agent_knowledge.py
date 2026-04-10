#!/usr/bin/env python3
"""
scripts/clear_agent_knowledge.py
─────────────────────────────────
Delete all knowledge rows written by over-eager agents
(before the LT memory discipline was enforced).

Run from the project root with venv active:
    python scripts/clear_agent_knowledge.py [--agents code_search developer document_qa] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from psycopg import AsyncConnection
from psycopg.rows import dict_row

DEFAULT_AGENTS = ["code_search", "developer", "document_qa"]


async def clear_knowledge(agents: list[str], dry_run: bool) -> None:
    db_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/agentmemory",
    )
    print(f"Connecting to: {db_url}")
    async with await AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
        # Show counts before deletion
        for agent in agents:
            cur = await conn.execute(
                "SELECT COUNT(*) AS cnt FROM knowledge WHERE agent = %s", (agent,)
            )
            row = await cur.fetchone()
            print(f"  {agent}: {row['cnt']} rows")

        if dry_run:
            print("\n[dry-run] No rows deleted.")
            return

        confirm = input(f"\nDelete ALL knowledge rows for {agents}? [yes/N] ").strip()
        if confirm.lower() != "yes":
            print("Aborted.")
            return

        for agent in agents:
            cur = await conn.execute(
                "DELETE FROM knowledge WHERE agent = %s RETURNING id", (agent,)
            )
            deleted = await cur.fetchall()
            print(f"  Deleted {len(deleted)} rows for agent '{agent}'")

        await conn.commit()
        print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clear knowledge rows for specified agents"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=DEFAULT_AGENTS,
        help="Agent names to clear (default: code_search developer document_qa)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show row counts without deleting",
    )
    args = parser.parse_args()
    asyncio.run(clear_knowledge(args.agents, args.dry_run))


if __name__ == "__main__":
    main()
