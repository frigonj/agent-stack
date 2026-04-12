"""
core/brave_quota.py
────────────────────
Brave Search API quota gate and ledger.

Flow
----
1. Research agent calls `request_search_approval(bus, memory, task_id, queries, purpose)`
   before firing any Brave requests.
2. This module calculates the request count, posts a Discord approval embed via the
   event bus, and blocks until the user approves or denies (5-minute timeout).
3. On approval, the ledger row is written to Postgres and a Redis counter for the
   current UTC month is incremented so the running total is always fast to read.
4. After the search completes, the agent calls `record_actual_usage(memory, ledger_id, n)`
   to fill in how many requests were actually fired.

Quota constants
---------------
BRAVE_COST_PER_1000 = $5.00  (informational, for the embed display)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog

log = structlog.get_logger()

BRAVE_COST_PER_1000 = 5.00  # USD — used for display only


# ── Redis key helpers ─────────────────────────────────────────────────────────


def _monthly_key() -> str:
    """Redis key for the current UTC month's request counter."""
    now = datetime.now(timezone.utc)
    return f"brave_quota:monthly:{now.year}:{now.month:02d}"


def _daily_key() -> str:
    """Redis key for today's UTC request counter."""
    now = datetime.now(timezone.utc)
    return f"brave_quota:daily:{now.date().isoformat()}"


# ── Public API ────────────────────────────────────────────────────────────────


async def get_quota_summary(bus, memory) -> dict:
    """
    Return a dict with current month/day usage and estimated cost.
    Safe to call at any time — returns zeros on Redis miss.
    """
    monthly = int(await bus._client.get(_monthly_key()) or 0)
    daily = int(await bus._client.get(_daily_key()) or 0)
    return {
        "monthly_requests": monthly,
        "daily_requests": daily,
        "monthly_cost_usd": round(monthly / 1000 * BRAVE_COST_PER_1000, 4),
    }


async def request_search_approval(
    bus,
    memory,
    task_id: str,
    queries: list[str],
    purpose: str,
) -> tuple[bool, Optional[int]]:
    """
    Gate: show the user how many Brave API requests the upcoming search will
    consume and wait for approval.

    Parameters
    ----------
    bus      : EventBus instance
    memory   : LongTermMemory instance
    task_id  : current research task ID
    queries  : the sub-queries that will be sent to Brave
    purpose  : short human-readable description of the research task

    Returns
    -------
    (approved: bool, ledger_id: int | None)
      ledger_id is the Postgres row ID to pass to record_actual_usage() later.
    """
    from core.events.bus import Event, EventType

    n_requests = len(queries)
    approval_id = f"brave_search:{task_id}:{uuid.uuid4().hex[:8]}"
    quota = await get_quota_summary(bus, memory)
    projected_monthly = quota["monthly_requests"] + n_requests
    projected_cost = round(projected_monthly / 1000 * BRAVE_COST_PER_1000, 4)

    log.info(
        "brave_quota.gate_requested",
        task_id=task_id,
        n_requests=n_requests,
        purpose=purpose[:80],
    )

    # Publish the approval request — Discord bridge renders the embed
    await bus.publish(
        Event(
            type=EventType.BRAVE_SEARCH_APPROVAL_REQUIRED,
            source="research",
            payload={
                "approval_id": approval_id,
                "task_id": task_id,
                "purpose": purpose[:300],
                "queries": queries,
                "n_requests": n_requests,
                "monthly_used": quota["monthly_requests"],
                "monthly_cost_usd": quota["monthly_cost_usd"],
                "projected_monthly": projected_monthly,
                "projected_cost_usd": projected_cost,
            },
        ),
        target="discord_bridge",
    )

    # Block until the user decides (5-minute window)
    decision = await bus.wait_for_approval(approval_id, timeout=300.0)
    approved = decision == "approved"

    if not approved:
        log.info("brave_quota.denied", task_id=task_id, approval_id=approval_id)
        return False, None

    # Write the ledger row
    ledger_id = await _write_ledger(memory, task_id, n_requests, purpose, queries)

    # Increment Redis counters (INCRBY + 30-day TTL to auto-expire old months)
    pipe = bus._client.pipeline()
    pipe.incrby(_monthly_key(), n_requests)
    pipe.expire(_monthly_key(), 32 * 86400)
    pipe.incrby(_daily_key(), n_requests)
    pipe.expire(_daily_key(), 2 * 86400)
    await pipe.execute()

    log.info(
        "brave_quota.approved",
        task_id=task_id,
        n_requests=n_requests,
        ledger_id=ledger_id,
    )
    return True, ledger_id


async def record_actual_usage(memory, ledger_id: int, actual_reqs: int) -> None:
    """
    Update the ledger row with how many Brave requests were actually fired.
    Call this after the search loop completes (or fails).
    """
    try:
        pool = await memory._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """
                UPDATE brave_quota
                SET actual_reqs = %s, completed_at = NOW()
                WHERE id = %s
                """,
                (actual_reqs, ledger_id),
            )
    except Exception as exc:
        log.warning(
            "brave_quota.record_actual_failed", ledger_id=ledger_id, error=str(exc)
        )


# ── Internal helpers ──────────────────────────────────────────────────────────


async def _write_ledger(
    memory, task_id: str, approved_reqs: int, purpose: str, queries: list[str]
) -> int:
    """Insert a ledger row and return its id."""
    query_summary = "; ".join(queries[:6])[:500]
    pool = await memory._get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            INSERT INTO brave_quota (task_id, approved_reqs, query_summary)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (task_id, approved_reqs, query_summary),
        )
        row = await cur.fetchone()
    return row[0]
