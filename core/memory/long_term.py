"""
core/memory/long_term.py
────────────────────────
PostgreSQL + pgvector client for long-term persistent memory.

Uses an async connection pool (psycopg3) with pgvector for semantic search
and PostgreSQL FTS (tsvector) for full-text recall. Schema is initialized
on first connection — no migration tooling required for initial deployment.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Optional

import structlog
from psycopg import AsyncConnection
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

log = structlog.get_logger()

# ── TTL size tiers ────────────────────────────────────────────────────────────
# Agents pick a t-shirt size; the background classifier writes expires_at.
# NULL means no expiry (permanent).

TTL_SIZES: dict[str, Optional[str]] = {
    "session": "12 hours",
    "short": "24 hours",
    "medium": "7 days",
    "long": "30 days",
    "permanent": None,
    "extend": "7 days",  # granted when user approves an extension request
}

TTL_SIZE_NAMES = list(TTL_SIZES.keys())  # ordered, for prompts

# Stream key consumed by the classify background pass
MEMORY_CLASSIFY_STREAM = "agents:memory:classify"
MEMORY_CLASSIFY_GROUP = "memory_classify_group"

# ── Embedding model (lazy-loaded, sentence-transformers already in container) ─

_encoder = None

# In-process LRU cache keyed on SHA-256(text).
# Prevents re-encoding identical queries across the lifetime of one agent process.
# 500 entries ≈ 750 KB of float32 vectors — negligible memory cost.
_EMBED_CACHE_MAX = 500
_embed_cache: dict[str, list[float]] = {}


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def _embed_texts(texts: list[str]) -> list[list[float] | None]:
    """
    Batch-encode texts to 384-dim vectors with in-process caching.

    Cached texts skip the model entirely — only novel texts are encoded.
    Returns None entries for any text that fails to encode.
    """
    results: list[list[float] | None] = [None] * len(texts)
    to_encode_indices: list[int] = []
    to_encode_texts: list[str] = []

    for i, text in enumerate(texts):
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in _embed_cache:
            results[i] = _embed_cache[key]
        else:
            to_encode_indices.append(i)
            to_encode_texts.append(text)

    if not to_encode_texts:
        return results  # all cache hits

    try:
        model = _get_encoder()
        vectors = model.encode(to_encode_texts)
        for idx, (orig_i, text, vec) in enumerate(
            zip(to_encode_indices, to_encode_texts, vectors)
        ):
            v = vec.tolist()
            key = hashlib.sha256(text.encode()).hexdigest()
            # Simple FIFO eviction when cache is full
            if len(_embed_cache) >= _EMBED_CACHE_MAX:
                oldest = next(iter(_embed_cache))
                del _embed_cache[oldest]
            _embed_cache[key] = v
            results[orig_i] = v
    except Exception as exc:
        log.warning("memory.embed_failed", error=str(exc))
        # results already has None for un-encoded indices

    cache_hits = len(texts) - len(to_encode_texts)
    if cache_hits:
        log.debug("memory.embed_cache_hits", hits=cache_hits, total=len(texts))

    return results


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS agent_status (
        agent        TEXT PRIMARY KEY,
        status       TEXT NOT NULL DEFAULT 'idle',
        current_task TEXT DEFAULT '',
        updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS handoffs (
        id      SERIAL PRIMARY KEY,
        agent   TEXT NOT NULL,
        ts      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        summary TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_handoffs_agent ON handoffs(agent)",
    """
    CREATE TABLE IF NOT EXISTS knowledge (
        id         SERIAL PRIMARY KEY,
        agent      TEXT NOT NULL DEFAULT 'default',
        topic      TEXT NOT NULL DEFAULT 'general',
        title      TEXT NOT NULL DEFAULT '',
        content    TEXT NOT NULL,
        tags       JSONB NOT NULL DEFAULT '[]',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        expires_at TIMESTAMPTZ DEFAULT NULL,
        embedding  vector(384)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_knowledge_agent ON knowledge(agent)",
    """
    CREATE INDEX IF NOT EXISTS idx_knowledge_fts ON knowledge
        USING GIN(to_tsvector('english', title || ' ' || content))
    """,
    # expires_at column — added after initial schema; safe to re-run
    "ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ DEFAULT NULL",
    "CREATE INDEX IF NOT EXISTS idx_knowledge_expires ON knowledge(expires_at) WHERE expires_at IS NOT NULL",
    # ── Open task queue ───────────────────────────────────────────────────
    # Used by the orchestrator think loop as a persistent, ordered work queue.
    # Tasks stored here are picked up on the next think cycle without any LLM call.
    """
    CREATE TABLE IF NOT EXISTS open_tasks (
        id         SERIAL PRIMARY KEY,
        agent      TEXT NOT NULL DEFAULT 'orchestrator',
        task       TEXT NOT NULL,
        priority   INT  NOT NULL DEFAULT 5,   -- lower = higher priority
        status     TEXT NOT NULL DEFAULT 'pending',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_open_tasks_queue ON open_tasks(status, priority, created_at)",
    # ── Active plan ledger ────────────────────────────────────────────────
    # One row per in-flight or recently-completed orchestrator plan.
    # Enables crash recovery and post-mortem inspection.
    # Rows expire automatically via expire_plans(): 7 days after last update.
    """
    CREATE TABLE IF NOT EXISTS active_plans (
        task_id       TEXT PRIMARY KEY,
        plan_id       TEXT NOT NULL,
        original_task TEXT NOT NULL,
        status        TEXT NOT NULL DEFAULT 'planning',
        plan_json     JSONB NOT NULL DEFAULT '{}',
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_active_plans_status ON active_plans(status, updated_at)",
    # ── Shared tool registry ──────────────────────────────────────────────
    # All agents register capabilities here; all agents search before calling the LLM.
    # Executor is the primary owner/runner but any agent may contribute.
    """
    CREATE TABLE IF NOT EXISTS tools (
        id          SERIAL PRIMARY KEY,
        name        TEXT UNIQUE NOT NULL,
        description TEXT NOT NULL,
        owner_agent TEXT NOT NULL DEFAULT 'executor',
        invocation  TEXT NOT NULL,
        tags        JSONB NOT NULL DEFAULT '[]',
        usage_count INT  NOT NULL DEFAULT 0,
        created_by  TEXT NOT NULL DEFAULT 'system',
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        embedding   vector(384)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tools_owner ON tools(owner_agent)",
    """
    CREATE INDEX IF NOT EXISTS idx_tools_fts ON tools
        USING GIN(to_tsvector('english', name || ' ' || description))
    """,
    # ── Context snapshots ─────────────────────────────────────────────────
    # Unified store for task, chat session, and plan summaries.
    # Serves both long-term recall ("what happened in session X?") and
    # user-facing context lookup ("/recall <id>").
    # Rolling mid-execution snapshots allow rollback to any checkpoint.
    """
    CREATE TABLE IF NOT EXISTS context_snapshots (
        context_id     TEXT NOT NULL,
        snapshot_seq   INT  NOT NULL DEFAULT 0,  -- 0 = latest, >0 = rolling checkpoints
        context_type   TEXT NOT NULL,            -- 'task' | 'chat' | 'plan'
        name           TEXT NOT NULL,
        status         TEXT NOT NULL DEFAULT 'active',
        topic_category TEXT DEFAULT NULL,
        keywords       JSONB NOT NULL DEFAULT '[]',
        summary        TEXT DEFAULT NULL,
        snapshot_json  JSONB NOT NULL DEFAULT '{}',
        value_score    FLOAT NOT NULL DEFAULT 0.0,
        message_count  INT   NOT NULL DEFAULT 0,
        created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        closed_at      TIMESTAMPTZ DEFAULT NULL,
        embedding      vector(384),
        PRIMARY KEY (context_id, snapshot_seq)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ctx_snapshots_type ON context_snapshots(context_type, status)",
    "CREATE INDEX IF NOT EXISTS idx_ctx_snapshots_updated ON context_snapshots(updated_at DESC)",
    """
    CREATE INDEX IF NOT EXISTS idx_ctx_snapshots_fts ON context_snapshots
        USING GIN(to_tsvector('english', name || ' ' || COALESCE(summary, '') || ' ' || COALESCE(topic_category, '')))
    """,
    # checkpoint_label column — added after initial schema; safe to re-run
    "ALTER TABLE context_snapshots ADD COLUMN IF NOT EXISTS checkpoint_label TEXT DEFAULT NULL",
    "CREATE INDEX IF NOT EXISTS idx_ctx_snapshots_checkpoint ON context_snapshots(context_id, checkpoint_label) WHERE checkpoint_label IS NOT NULL",
    # ── Topic patterns ────────────────────────────────────────────────────
    # Agent-learned keyword→category mappings that grow over sessions.
    # Confidence rises as more sessions confirm the pattern.
    # When confidence < threshold the orchestrator may ask the user.
    """
    CREATE TABLE IF NOT EXISTS topic_patterns (
        id           SERIAL PRIMARY KEY,
        category     TEXT NOT NULL,
        keywords     JSONB NOT NULL DEFAULT '[]',
        match_count  INT  NOT NULL DEFAULT 1,
        confidence   FLOAT NOT NULL DEFAULT 0.5,
        created_by   TEXT NOT NULL DEFAULT 'orchestrator',
        created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_topic_patterns_category ON topic_patterns(category)",
    "CREATE INDEX IF NOT EXISTS idx_topic_patterns_confidence ON topic_patterns(confidence DESC)",
    # ── Research sources ──────────────────────────────────────────────────
    # Committed facts gathered by the research agent after cross-source consensus.
    # Each row is one source URL + extracted snippet pair for a given research query.
    # Rows with confidence >= 0.75 are considered reliable and used in replies.
    """
    CREATE TABLE IF NOT EXISTS research_sources (
        id           SERIAL PRIMARY KEY,
        query_id     TEXT NOT NULL,           -- matches the ctx:research:{id} stream key
        query_text   TEXT NOT NULL,           -- original user research question
        url          TEXT NOT NULL,
        domain       TEXT NOT NULL DEFAULT '',
        title        TEXT NOT NULL DEFAULT '',
        snippet      TEXT NOT NULL,
        fact         TEXT NOT NULL DEFAULT '', -- extracted single-sentence fact
        confidence   FLOAT NOT NULL DEFAULT 0.0,
        source_rank  INT   NOT NULL DEFAULT 0, -- position in search results (0-indexed)
        committed    BOOLEAN NOT NULL DEFAULT FALSE,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        embedding    vector(384)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_research_sources_query ON research_sources(query_id)",
    "CREATE INDEX IF NOT EXISTS idx_research_sources_committed ON research_sources(committed, confidence DESC)",
    """
    CREATE INDEX IF NOT EXISTS idx_research_sources_fts ON research_sources
        USING GIN(to_tsvector('english', query_text || ' ' || fact || ' ' || snippet))
    """,
    # ── Research staging ──────────────────────────────────────────────────
    # Intermediate facts accumulated during a pauseable research loop.
    # Survives process restarts and explicit pauses; deleted on final commit.
    """
    CREATE TABLE IF NOT EXISTS research_staging (
        id         BIGSERIAL PRIMARY KEY,
        task_id    TEXT NOT NULL,
        iteration  INT  NOT NULL,
        fact_json  JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_research_staging_task ON research_staging(task_id, iteration)",
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_research_staging_dedup
        ON research_staging(task_id, iteration, (fact_json->>'url'), (fact_json->>'fact'))
    """,
    # ── active_plans additions ─────────────────────────────────────────────
    # paused status is enforced at application level; column additions are
    # safe to re-run (IF NOT EXISTS / IF NOT EXISTS equivalent via ADD COLUMN IF NOT EXISTS).
    "ALTER TABLE active_plans ADD COLUMN IF NOT EXISTS paused_at_phase INT DEFAULT NULL",
    # ── content_hash deduplication on knowledge ────────────────────────────
    # SHA-256 of (agent || topic || content).  Upsert on conflict refreshes the
    # row in-place instead of creating a duplicate entry.
    "ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS content_hash TEXT DEFAULT NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_content_hash ON knowledge(content_hash) WHERE content_hash IS NOT NULL",
    # ── Memory approval decisions ──────────────────────────────────────────
    # Stores every approve/deny decision the user makes on a memory retention
    # request.  Used to bias future TTL classification so agents learn what is
    # and is not worth keeping long-term.
    """
    CREATE TABLE IF NOT EXISTS memory_approval_decisions (
        id            SERIAL PRIMARY KEY,
        agent         TEXT NOT NULL,
        topic         TEXT NOT NULL,
        tags          JSONB NOT NULL DEFAULT '[]',
        content       TEXT NOT NULL,
        proposed_size TEXT NOT NULL,
        approved      BOOLEAN NOT NULL,
        decided_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        embedding     vector(384)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_mad_agent ON memory_approval_decisions(agent)",
    "CREATE INDEX IF NOT EXISTS idx_mad_approved ON memory_approval_decisions(approved, proposed_size)",
    """
    CREATE INDEX IF NOT EXISTS idx_mad_fts ON memory_approval_decisions
        USING GIN(to_tsvector('english', topic || ' ' || content))
    """,
    # ── classified_at on knowledge ─────────────────────────────────────────
    # Tracks when a row's TTL was last assigned.  get_unclassified() uses this
    # instead of expires_at IS NULL so permanent rows (expires_at = NULL) are
    # never re-queued for classification on restart.
    "ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS classified_at TIMESTAMPTZ DEFAULT NULL",
    "CREATE INDEX IF NOT EXISTS idx_knowledge_classified ON knowledge(classified_at) WHERE classified_at IS NULL",
    # ── Pending memory approvals ───────────────────────────────────────────
    # One row per in-flight approval request.  Inserted before the Discord
    # embed is sent; deleted (or marked decided) after the user responds or
    # the 48-hour window expires.  Survives container restarts.
    """
    CREATE TABLE IF NOT EXISTS pending_memory_approvals (
        approval_id   TEXT PRIMARY KEY,
        knowledge_id  INTEGER NOT NULL,
        agent         TEXT NOT NULL,
        topic         TEXT NOT NULL,
        tags          JSONB NOT NULL DEFAULT '[]',
        content       TEXT NOT NULL,
        proposed_size TEXT NOT NULL,
        requested_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        expires_at    TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '48 hours',
        result        BOOLEAN DEFAULT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_pma_knowledge ON pending_memory_approvals(knowledge_id)",
    "CREATE INDEX IF NOT EXISTS idx_pma_result ON pending_memory_approvals(result, expires_at) WHERE result IS NULL",
    # ── Eval results ──────────────────────────────────────────────────────
    # One row per evaluation run against a training task output.
    # Tier 1 = structural checks (rule-based), Tier 2 = local LLM judge,
    # Tier 3 = external judge (Gemini Flash).
    # user_verdict is set when the user thumbs-up/down from #eval-queue.
    """
    CREATE TABLE IF NOT EXISTS eval_results (
        id              SERIAL PRIMARY KEY,
        task_id         TEXT NOT NULL,
        plan_id         TEXT NOT NULL DEFAULT '',
        task_type       TEXT NOT NULL,              -- 'arch_doc' | 'research' | 'code_analysis' | ...
        original_task   TEXT NOT NULL,
        artifact_path   TEXT DEFAULT NULL,          -- path to output file if any
        tier1_passed    BOOLEAN NOT NULL DEFAULT FALSE,
        tier1_reasons   JSONB NOT NULL DEFAULT '[]',
        tier2_score     FLOAT DEFAULT NULL,         -- 0-10, NULL if not run
        tier2_breakdown JSONB DEFAULT NULL,         -- per-criterion scores
        tier2_flags     JSONB NOT NULL DEFAULT '[]',
        tier3_score     FLOAT DEFAULT NULL,         -- 0-10, NULL if not run
        tier3_model     TEXT DEFAULT NULL,
        final_score     FLOAT DEFAULT NULL,         -- resolved score after all tiers
        review_status   TEXT NOT NULL DEFAULT 'pending',  -- 'pending'|'approved'|'rejected'|'auto_approved'
        user_verdict    BOOLEAN DEFAULT NULL,       -- TRUE=thumbs up, FALSE=thumbs down
        user_feedback   TEXT DEFAULT NULL,
        plan_steps      INT NOT NULL DEFAULT 0,
        plan_retries    INT NOT NULL DEFAULT 0,
        approval_requested BOOLEAN NOT NULL DEFAULT FALSE,
        agents_used     JSONB NOT NULL DEFAULT '[]',
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        reviewed_at     TIMESTAMPTZ DEFAULT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_eval_task ON eval_results(task_id)",
    "CREATE INDEX IF NOT EXISTS idx_eval_type ON eval_results(task_type, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_eval_review ON eval_results(review_status, created_at DESC)",
    # ── Brave Search quota ledger ────────────────────────────────────────────
    # One row per approved search batch. Tracks approved vs actual spend so
    # the quota dashboard is accurate even if a task is interrupted mid-run.
    """
    CREATE TABLE IF NOT EXISTS brave_quota (
        id              SERIAL PRIMARY KEY,
        task_id         TEXT NOT NULL,
        approved_reqs   INT NOT NULL,           -- how many requests the user approved
        actual_reqs     INT NOT NULL DEFAULT 0, -- filled in after the search completes
        query_summary   TEXT NOT NULL DEFAULT '',
        approved_by     TEXT NOT NULL DEFAULT 'user',
        approved_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        completed_at    TIMESTAMPTZ DEFAULT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_brave_quota_task ON brave_quota(task_id)",
    "CREATE INDEX IF NOT EXISTS idx_brave_quota_date ON brave_quota(approved_at DESC)",
]


async def _configure_conn(conn: AsyncConnection) -> None:
    """Register pgvector codec on each new pool connection."""
    from pgvector.psycopg import register_vector_async

    await register_vector_async(conn)
    await conn.commit()


# ── Client ────────────────────────────────────────────────────────────────────


class LongTermMemory:
    """
    Async PostgreSQL + pgvector client for agent persistent memory.

    All agents share the same database. Recall uses PostgreSQL FTS by default;
    vector_search uses pgvector cosine similarity when embeddings are present.
    """

    def __init__(self, database_url: str, agent_name: str):
        self._url = database_url
        self._agent = agent_name
        self._pool: Optional[AsyncConnectionPool] = None

    async def _get_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            # Create the pgvector extension BEFORE opening the pool.
            # The pool's configure callback (register_vector_async) queries PG
            # for the vector type OID — it fails if the extension doesn't exist
            # yet. A plain connection has no codec registered, so this is safe.
            # Catch UniqueViolation: all agents start concurrently and may race
            # on CREATE EXTENSION — IF NOT EXISTS is not atomic under concurrency.
            async with await AsyncConnection.connect(self._url) as conn:
                try:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    await conn.commit()
                except UniqueViolation:
                    await conn.rollback()  # extension was created by another agent

            self._pool = AsyncConnectionPool(
                self._url,
                min_size=1,
                max_size=5,
                open=False,
                configure=_configure_conn,
                kwargs={"row_factory": dict_row},
            )
            await self._pool.open()
            async with self._pool.connection() as conn:
                # Serialize schema migrations across all agents using a
                # PostgreSQL advisory lock. ALTER TABLE statements acquire
                # AccessExclusiveLock; without serialization, concurrent agents
                # deadlock on the same relation in conflicting order.
                await conn.execute("SELECT pg_advisory_lock(8675309)")
                try:
                    for stmt in _SCHEMA:
                        await conn.execute(stmt)
                finally:
                    await conn.execute("SELECT pg_advisory_unlock(8675309)")
        return self._pool

    # ── Session lifecycle ────────────────────────────────────────────────────

    async def open_session(self) -> dict:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT status FROM agent_status WHERE agent = %s", (self._agent,)
            )
            row = await cur.fetchone()
            crashed = row is not None and row["status"] == "OPEN"
            await conn.execute(
                """
                INSERT INTO agent_status (agent, status, current_task, updated_at)
                VALUES (%s, 'OPEN', '', NOW())
                ON CONFLICT (agent) DO UPDATE SET
                    status = 'OPEN', current_task = '', updated_at = NOW()
            """,
                (self._agent,),
            )
        status = "CRASH" if crashed else "OK"
        log.info("memory.session_opened", agent=self._agent, status=status)
        return {"status": status, "agent": self._agent}

    async def recover_context(self) -> dict:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT status, current_task FROM agent_status WHERE agent = %s",
                (self._agent,),
            )
            row = await cur.fetchone()
            cur = await conn.execute(
                "SELECT summary FROM handoffs WHERE agent = %s ORDER BY id DESC LIMIT 1",
                (self._agent,),
            )
            handoff = await cur.fetchone()
        result = {
            "status": row["status"] if row else "NEW",
            "current_task": row["current_task"] if row else "",
            "last_handoff": handoff["summary"] if handoff else "",
        }
        log.info("memory.context_recovered", agent=self._agent)
        return result

    async def checkpoint(self, note: str) -> None:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE agent_status SET current_task = %s, updated_at = NOW() WHERE agent = %s",
                (note, self._agent),
            )
        log.debug("memory.checkpoint", agent=self._agent, note=note)

    async def close_session(self, summary: str) -> None:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO handoffs (agent, summary) VALUES (%s, %s)",
                (self._agent, summary),
            )
            await conn.execute(
                "UPDATE agent_status SET status = 'CLEAN', updated_at = NOW() WHERE agent = %s",
                (self._agent,),
            )
        log.info("memory.session_closed", agent=self._agent)

    # ── Knowledge store ──────────────────────────────────────────────────────

    async def store(
        self,
        content: str,
        topic: str,
        tags: Optional[list[str]] = None,
        kind: str = "finding",
        ttl_days: Optional[int] = None,
    ) -> dict:
        """
        Upsert a knowledge entry.

        Uses a SHA-256(agent||topic||content) hash to detect duplicates.
        On collision: updates tags, refreshes embedding, resets expires_at,
        and bumps updated_at instead of inserting a second row.
        """
        content_hash = hashlib.sha256(
            f"{self._agent}||{topic}||{content}".encode()
        ).hexdigest()
        [embedding] = await asyncio.to_thread(_embed_texts, [content])
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                INSERT INTO knowledge
                    (agent, topic, title, content, tags, embedding, expires_at, content_hash)
                VALUES (%s, %s, %s, %s, %s, %s,
                        CASE WHEN %s::INTEGER IS NOT NULL
                             THEN NOW() + (%s || ' days')::INTERVAL
                             ELSE NULL END,
                        %s)
                ON CONFLICT (content_hash) WHERE content_hash IS NOT NULL
                DO UPDATE SET
                    tags       = EXCLUDED.tags,
                    embedding  = EXCLUDED.embedding,
                    expires_at = EXCLUDED.expires_at
                RETURNING id
                """,
                (
                    self._agent,
                    topic,
                    topic,
                    content,
                    json.dumps(tags or []),
                    embedding,
                    ttl_days,
                    ttl_days,
                    content_hash,
                ),
            )
            row = await cur.fetchone()
        knowledge_id = row["id"]
        log.info(
            "memory.promoted",
            topic=topic,
            kind=kind,
            ttl_days=ttl_days,
            id=knowledge_id,
        )
        return {"status": "stored", "id": knowledge_id}

    async def record_approval_decision(
        self,
        agent: str,
        topic: str,
        tags: list[str],
        content: str,
        proposed_size: str,
        approved: bool,
    ) -> None:
        """
        Persist a memory approval/denial decision so agents can learn over time
        what qualifies as long-term knowledge.
        """
        [embedding] = await asyncio.to_thread(
            _embed_texts, [f"{topic} {content[:200]}"]
        )
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO memory_approval_decisions
                    (agent, topic, tags, content, proposed_size, approved, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    agent,
                    topic,
                    json.dumps(tags),
                    content[:500],
                    proposed_size,
                    approved,
                    embedding,
                ),
            )
        log.info(
            "memory.approval_decision_recorded",
            agent=agent,
            topic=topic,
            proposed_size=proposed_size,
            approved=approved,
        )

    async def find_similar_approval_decisions(
        self, topic: str, content: str, limit: int = 5
    ) -> list[dict]:
        """
        Return past approval decisions semantically similar to the given topic+content.
        Used by _classify_ttl_by_rules to apply learned patterns before falling back to defaults.
        """
        query_text = f"{topic} {content[:200]}"
        [embedding] = await asyncio.to_thread(_embed_texts, [query_text])
        pool = await self._get_pool()
        async with pool.connection() as conn:
            if embedding is not None:
                cur = await conn.execute(
                    """
                    SELECT topic, tags, proposed_size, approved,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM memory_approval_decisions
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding, embedding, limit),
                )
            else:
                cur = await conn.execute(
                    """
                    SELECT topic, tags, proposed_size, approved,
                           ts_rank(
                               to_tsvector('english', topic || ' ' || content),
                               plainto_tsquery('english', %s)
                           ) AS similarity
                    FROM memory_approval_decisions
                    WHERE to_tsvector('english', topic || ' ' || content)
                          @@ plainto_tsquery('english', %s)
                    ORDER BY similarity DESC
                    LIMIT %s
                    """,
                    (query_text, query_text, limit),
                )
            rows = await cur.fetchall()
        return list(rows)

    async def batch_store(self, entries: list[dict]) -> dict:
        if not entries:
            return {"status": "stored", "count": 0, "ids": []}
        texts = [e.get("content", "") for e in entries]
        embeddings = await asyncio.to_thread(_embed_texts, texts)
        pool = await self._get_pool()
        ids: list[int] = []
        try:
            async with pool.connection() as conn:
                for i, e in enumerate(entries):
                    ttl = e.get("ttl_days")
                    cur = await conn.execute(
                        """
                        INSERT INTO knowledge (agent, topic, title, content, tags, embedding, expires_at)
                        VALUES (%s, %s, %s, %s, %s, %s,
                                CASE WHEN %s::INTEGER IS NOT NULL THEN NOW() + (%s || ' days')::INTERVAL ELSE NULL END)
                        RETURNING id
                        """,
                        (
                            self._agent,
                            e.get("topic", "general"),
                            e.get("topic", "general"),
                            e.get("content", ""),
                            json.dumps(e.get("tags") or []),
                            embeddings[i],
                            ttl,
                            ttl,
                        ),
                    )
                    row = await cur.fetchone()
                    ids.append(row["id"])
                await conn.commit()
        except Exception as exc:
            log.error("memory.batch_store_failed", count=len(entries), error=str(exc))
            raise
        log.info("memory.batch_promoted", count=len(entries))
        return {"status": "stored", "count": len(entries), "ids": ids}

    # ── Recall / search ──────────────────────────────────────────────────────

    async def recall(self, query: str, limit: int = 5) -> list[dict]:
        """Full-text search via PostgreSQL tsvector. Excludes expired entries."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, agent, topic, title, content, tags, created_at, expires_at
                FROM knowledge
                WHERE (expires_at IS NULL OR expires_at > NOW())
                  AND topic != 'learned_intent'
                  AND to_tsvector('english', title || ' ' || content)
                      @@ plainto_tsquery('english', %s)
                ORDER BY ts_rank(
                    to_tsvector('english', title || ' ' || content),
                    plainto_tsquery('english', %s)
                ) DESC
                LIMIT %s
            """,
                (query, query, limit),
            )
            rows = await cur.fetchall()
        log.debug("memory.recalled", query=query, results=len(rows))
        return list(rows)

    async def vector_search(self, query: str, limit: int = 5) -> list[dict]:
        """Semantic search via pgvector cosine similarity."""
        [embedding] = await asyncio.to_thread(_embed_texts, [query])
        if embedding is None:
            return await self.recall(query, limit)
        return await self._vector_search_with_embedding(query, embedding, limit)

    async def _vector_search_with_embedding(
        self, query: str, embedding: list[float], limit: int = 5
    ) -> list[dict]:
        """Semantic search using a pre-computed embedding vector. Excludes expired entries
        and internal orchestrator topics (e.g. learned_intent) that are only useful to
        the orchestrator itself and pollute recall for other agents."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, agent, topic, title, content, tags, created_at, expires_at,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM knowledge
                WHERE embedding IS NOT NULL
                  AND (expires_at IS NULL OR expires_at > NOW())
                  AND topic != 'learned_intent'
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """,
                (embedding, embedding, limit),
            )
            rows = await cur.fetchall()
        log.debug("memory.vector_recalled", query=query, results=len(rows))
        return list(rows)

    async def search(
        self, query: str, semantic: bool = False, limit: int = 5
    ) -> list[dict]:
        if semantic:
            try:
                return await self.vector_search(query, limit)
            except Exception:
                log.warning("memory.vector_search_failed", fallback="fts")
        return await self.recall(query, limit)

    async def search_memory_and_tools(
        self, query: str, tools_limit: int = 5, memory_limit: int = 5
    ) -> tuple[list[dict], list[dict]]:
        """
        Embed the query once, then run knowledge search and tool search in
        parallel. Returns (memory_results, tool_results).

        Use this instead of calling recall() + search_tools() separately to
        avoid paying the embedding cost twice.
        """
        [embedding] = await asyncio.to_thread(_embed_texts, [query])
        if embedding is not None:
            memory_task = asyncio.create_task(
                self._vector_search_with_embedding(query, embedding, memory_limit)
            )
            tools_task = asyncio.create_task(
                self._search_tools_with_embedding(query, embedding, tools_limit)
            )
        else:
            memory_task = asyncio.create_task(self.recall(query, memory_limit))
            tools_task = asyncio.create_task(
                self._search_tools_with_embedding(query, None, tools_limit)
            )
        memory_results, tool_results = await asyncio.gather(memory_task, tools_task)
        return memory_results, tool_results

    async def set_expiry(self, knowledge_id: int, size: str) -> bool:
        """
        Write expires_at for a knowledge row based on a t-shirt size label.
        Also stamps classified_at = NOW() so the row is never re-queued for
        classification on restart (even when size='permanent', expires_at=NULL).
        Returns True if the row was found and updated, False if it no longer exists.
        Size must be one of TTL_SIZE_NAMES; 'permanent' sets expires_at to NULL.
        """
        interval = TTL_SIZES.get(size)
        pool = await self._get_pool()
        async with pool.connection() as conn:
            if interval is None:
                cur = await conn.execute(
                    "UPDATE knowledge SET expires_at = NULL, classified_at = NOW() WHERE id = %s RETURNING id",
                    (knowledge_id,),
                )
            else:
                cur = await conn.execute(
                    "UPDATE knowledge SET expires_at = NOW() + %s::INTERVAL, classified_at = NOW() WHERE id = %s RETURNING id",
                    (interval, knowledge_id),
                )
            row = await cur.fetchone()
        updated = row is not None
        log.debug("memory.expiry_set", id=knowledge_id, size=size, updated=updated)
        return updated

    async def get_unclassified(self, limit: int = 50) -> list[dict]:
        """
        Return knowledge rows that have never been classified.
        Uses classified_at IS NULL (not expires_at IS NULL) so permanent rows
        with expires_at = NULL are not incorrectly re-queued on restart.
        Excludes rows that already have a pending approval in-flight.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT k.id, k.agent, k.topic, k.content, k.tags, k.created_at
                FROM knowledge k
                WHERE k.classified_at IS NULL
                  AND k.created_at < NOW() - INTERVAL '1 minute'
                  AND NOT EXISTS (
                      SELECT 1 FROM pending_memory_approvals p
                      WHERE p.knowledge_id = k.id
                        AND p.result IS NULL
                        AND p.expires_at > NOW()
                  )
                ORDER BY k.created_at ASC
                LIMIT %s
                """,
                (limit,),
            )
            rows = await cur.fetchall()
        return list(rows)

    async def get_expiring_soon(
        self, within_hours: int = 6, limit: int = 50
    ) -> list[dict]:
        """
        Return knowledge rows expiring within `within_hours` hours that do not
        already have a live pending approval (extension already in flight).
        Used by the expiry-review loop to decide whether to request an extension.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT k.id, k.agent, k.topic, k.content, k.tags, k.expires_at
                FROM knowledge k
                WHERE k.expires_at IS NOT NULL
                  AND k.expires_at > NOW()
                  AND k.expires_at <= NOW() + (%s * INTERVAL '1 hour')
                  AND NOT EXISTS (
                      SELECT 1 FROM pending_memory_approvals p
                      WHERE p.knowledge_id = k.id
                        AND p.result IS NULL
                        AND p.expires_at > NOW()
                  )
                ORDER BY k.expires_at ASC
                LIMIT %s
                """,
                (within_hours, limit),
            )
            rows = await cur.fetchall()
        return list(rows)

    async def insert_pending_approval(
        self,
        approval_id: str,
        knowledge_id: int,
        agent: str,
        topic: str,
        tags: list[str],
        content: str,
        proposed_size: str,
    ) -> bool:
        """
        Insert a pending approval row.  Returns False if one already exists
        for this knowledge_id (prevents duplicate Discord embeds).
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT approval_id FROM pending_memory_approvals WHERE knowledge_id = %s AND result IS NULL AND expires_at > NOW()",
                (knowledge_id,),
            )
            existing = await cur.fetchone()
            if existing:
                return False
            await conn.execute(
                """
                INSERT INTO pending_memory_approvals
                    (approval_id, knowledge_id, agent, topic, tags, content, proposed_size)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (approval_id) DO NOTHING
                """,
                (
                    approval_id,
                    knowledge_id,
                    agent,
                    topic,
                    json.dumps(tags),
                    content[:500],
                    proposed_size,
                ),
            )
        return True

    async def resolve_pending_approval(
        self, approval_id: str, approved: bool
    ) -> dict | None:
        """
        Mark a pending approval as decided.  Returns the full row so the caller
        can apply set_expiry and record_approval_decision.  Returns None if the
        approval_id is not found or already decided.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE pending_memory_approvals
                SET result = %s
                WHERE approval_id = %s AND result IS NULL
                RETURNING knowledge_id, agent, topic, tags, content, proposed_size
                """,
                (approved, approval_id),
            )
            row = await cur.fetchone()
        return dict(row) if row else None

    async def get_expired_pending_approvals(self) -> list[dict]:
        """Return pending approval rows whose 48h window has elapsed."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                DELETE FROM pending_memory_approvals
                WHERE result IS NULL AND expires_at <= NOW()
                RETURNING approval_id, knowledge_id, agent, topic, tags, content, proposed_size
                """,
            )
            rows = await cur.fetchall()
        return list(rows)

    async def delete_pending_approval(self, approval_id: str) -> None:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "DELETE FROM pending_memory_approvals WHERE approval_id = %s",
                (approval_id,),
            )

    async def count(self) -> int:
        """Return number of non-expired knowledge entries across all agents."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT COUNT(*) AS n FROM knowledge WHERE expires_at IS NULL OR expires_at > NOW()"
            )
            row = await cur.fetchone()
        return row["n"] if row else 0

    async def expire_knowledge(self) -> int:
        """Delete knowledge entries whose expires_at has passed. Returns count deleted."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM knowledge WHERE expires_at IS NOT NULL AND expires_at <= NOW() RETURNING id"
            )
            rows = await cur.fetchall()
        deleted = len(rows)
        if deleted:
            log.info("memory.knowledge_expired", deleted=deleted)
        return deleted

    async def expire_by_task_id(self, task_id: str) -> int:
        """
        Immediately expire all knowledge rows tagged with task_id.
        Called when the user says 'try again' — the failed attempt's memories
        should not influence the retry.  Returns the number of rows expired.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE knowledge
                SET expires_at = NOW()
                WHERE tags @> %s::jsonb
                  AND (expires_at IS NULL OR expires_at > NOW())
                RETURNING id
                """,
                (json.dumps([task_id]),),
            )
            rows = await cur.fetchall()
        expired = len(rows)
        if expired:
            log.info("memory.task_expired", task_id=task_id, expired=expired)
        return expired

    async def prune(self, target: int) -> int:
        """
        Delete the oldest knowledge entries until count <= target.
        Returns the number of entries deleted.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT COUNT(*) AS n FROM knowledge")
            row = await cur.fetchone()
            total = row["n"] if row else 0

            to_delete = max(0, total - target)
            if to_delete == 0:
                return 0

            await conn.execute(
                """
                DELETE FROM knowledge
                WHERE id IN (
                    SELECT id FROM knowledge
                    ORDER BY created_at ASC
                    LIMIT %s
                )
            """,
                (to_delete,),
            )

        log.info("memory.pruned", deleted=to_delete, remaining=total - to_delete)
        return to_delete

    # ── Active plan ledger ───────────────────────────────────────────────────

    async def upsert_plan(
        self,
        task_id: str,
        plan_id: str,
        original_task: str,
        status: str,
        plan_json: dict,
    ) -> None:
        """Create or update a plan row. Called on plan creation and every status change."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO active_plans (task_id, plan_id, original_task, status, plan_json)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (task_id) DO UPDATE SET
                    status     = EXCLUDED.status,
                    plan_json  = EXCLUDED.plan_json,
                    updated_at = NOW()
            """,
                (task_id, plan_id, original_task[:500], status, json.dumps(plan_json)),
            )
        log.debug("memory.plan_upserted", task_id=task_id, status=status)

    async def load_active_plans(self) -> list[dict]:
        """Return all plans that were still running when the process last exited.
        Paused plans are included so they survive restarts, but the orchestrator
        will NOT auto-resume them — the user must send /resume explicitly."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute("""
                SELECT task_id, plan_id, original_task, status, plan_json, created_at, updated_at
                FROM active_plans
                WHERE status IN ('planning', 'running', 'paused')
                ORDER BY created_at ASC
            """)
            rows = await cur.fetchall()
        return list(rows)

    async def expire_plans(self) -> int:
        """Delete plan rows older than 7 days. Returns number deleted."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute("""
                DELETE FROM active_plans
                WHERE updated_at < NOW() - INTERVAL '7 days'
                RETURNING task_id
            """)
            rows = await cur.fetchall()
        deleted = len(rows)
        if deleted:
            log.info("memory.plans_expired", deleted=deleted)
        return deleted

    # ── Research staging helpers ──────────────────────────────────────────────

    async def save_research_staging(
        self, task_id: str, iteration: int, facts: list[dict]
    ) -> None:
        """Persist a batch of staging facts for a pauseable research task."""
        if not facts:
            return
        pool = await self._get_pool()
        async with pool.connection() as conn:
            for f in facts:
                await conn.execute(
                    """
                    INSERT INTO research_staging (task_id, iteration, fact_json)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (task_id, iteration, json.dumps(f)),
                )
            await conn.commit()
        log.debug(
            "memory.research_staging_saved",
            task_id=task_id,
            iteration=iteration,
            count=len(facts),
        )

    async def load_research_staging(self, task_id: str) -> list[dict]:
        """Return all staged facts for a task, ordered by iteration."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT fact_json FROM research_staging
                WHERE task_id = %s
                ORDER BY iteration ASC, created_at ASC
                """,
                (task_id,),
            )
            rows = await cur.fetchall()
        return [r[0] if isinstance(r[0], dict) else json.loads(r[0]) for r in rows]

    async def delete_research_staging(self, task_id: str) -> None:
        """Remove all staging rows for a task (called after final commit)."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "DELETE FROM research_staging WHERE task_id = %s", (task_id,)
            )
            await conn.commit()
        log.debug("memory.research_staging_deleted", task_id=task_id)

    async def cleanup_stale(self, max_age_hours: int = 24) -> dict:
        """
        Delete completed/failed tasks, resolved plans, and old handoffs that are
        older than *max_age_hours*. Called by the orchestrator think loop.

        Returns a dict with counts of each type deleted so the caller can log
        or emit a status event if anything was pruned.
        """
        interval = f"{max_age_hours} hours"
        pool = await self._get_pool()
        async with pool.connection() as conn:
            # Completed / failed tasks older than max_age
            cur = await conn.execute(
                """
                DELETE FROM open_tasks
                WHERE status IN ('done', 'failed')
                  AND updated_at < NOW() - INTERVAL %s
                RETURNING id
                """,
                (interval,),
            )
            tasks = len(await cur.fetchall())

            # Terminal plans older than max_age
            cur = await conn.execute(
                """
                DELETE FROM active_plans
                WHERE status IN ('completed', 'expired', 'failed')
                  AND updated_at < NOW() - INTERVAL %s
                RETURNING task_id
                """,
                (interval,),
            )
            plans = len(await cur.fetchall())

            # Handoffs (session summaries) older than max_age
            cur = await conn.execute(
                """
                DELETE FROM handoffs
                WHERE ts < NOW() - INTERVAL %s
                RETURNING id
                """,
                (interval,),
            )
            handoffs = len(await cur.fetchall())

        # Expired knowledge entries (TTL-based)
        knowledge = await self.expire_knowledge()

        total = tasks + plans + handoffs + knowledge
        if total:
            log.info(
                "memory.stale_cleanup",
                tasks=tasks,
                plans=plans,
                handoffs=handoffs,
                knowledge_expired=knowledge,
                max_age_hours=max_age_hours,
            )
        return {
            "tasks": tasks,
            "plans": plans,
            "handoffs": handoffs,
            "knowledge_expired": knowledge,
        }

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    # ── Open task queue ──────────────────────────────────────────────────────

    async def enqueue_task(
        self, task: str, priority: int = 5, agent: str = "orchestrator"
    ) -> int:
        """Add a task to the persistent work queue. Returns the new task id."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                INSERT INTO open_tasks (agent, task, priority)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (agent, task, priority),
            )
            row = await cur.fetchone()
        task_id = row["id"]
        log.info("memory.task_enqueued", task_id=task_id, priority=priority)
        return task_id

    async def get_pending_tasks(
        self, limit: int = 5, agent: str = "orchestrator"
    ) -> list[dict]:
        """
        Return pending tasks ordered by priority then age.
        These are consumed by the think loop without any LLM call.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, agent, task, priority, created_at
                FROM open_tasks
                WHERE status = 'pending' AND agent = %s
                ORDER BY priority ASC, created_at ASC
                LIMIT %s
                """,
                (agent, limit),
            )
            rows = await cur.fetchall()
        return list(rows)

    async def claim_task(self, task_id: int) -> bool:
        """Mark a task as running. Returns False if already claimed (concurrent safety)."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE open_tasks
                SET status = 'running', updated_at = NOW()
                WHERE id = %s AND status = 'pending'
                RETURNING id
                """,
                (task_id,),
            )
            row = await cur.fetchone()
        return row is not None

    async def complete_task(self, task_id: int) -> None:
        """Mark a task as done."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE open_tasks SET status = 'done', updated_at = NOW() WHERE id = %s",
                (task_id,),
            )

    async def fail_task(self, task_id: int) -> None:
        """Return a claimed task to pending so it can be retried."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE open_tasks SET status = 'pending', updated_at = NOW() WHERE id = %s",
                (task_id,),
            )

    # ── Shared tool registry ─────────────────────────────────────────────────
    # All agents register capabilities here and search before calling the LLM.
    # Executor is the primary runner but any agent may contribute tools.

    async def register_tool(
        self,
        name: str,
        description: str,
        owner_agent: str,
        invocation: str,
        tags: Optional[list[str]] = None,
        created_by: str = "system",
    ) -> None:
        """
        Add or update a tool in the shared registry.
        Upserts on name so re-registering on startup refreshes the embedding.
        """
        [embedding] = await asyncio.to_thread(_embed_texts, [description])
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO tools (name, description, owner_agent, invocation, tags, created_by, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    description = EXCLUDED.description,
                    owner_agent = EXCLUDED.owner_agent,
                    invocation  = EXCLUDED.invocation,
                    tags        = EXCLUDED.tags,
                    embedding   = EXCLUDED.embedding
                """,
                (
                    name,
                    description,
                    owner_agent,
                    invocation,
                    json.dumps(tags or []),
                    created_by,
                    embedding,
                ),
            )
        log.debug("memory.tool_registered", name=name, owner=owner_agent)

    async def search_tools(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search the shared tool registry using semantic (pgvector) similarity.
        Falls back to FTS when no embeddings are available.
        Returns rows ordered by relevance (most similar first).
        """
        [embedding] = await asyncio.to_thread(_embed_texts, [query])
        return await self._search_tools_with_embedding(query, embedding, limit)

    async def _search_tools_with_embedding(
        self, query: str, embedding: list[float] | None, limit: int = 5
    ) -> list[dict]:
        """Search tools using a pre-computed embedding vector."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            if embedding is not None:
                cur = await conn.execute(
                    """
                    SELECT name, description, owner_agent, invocation, tags, usage_count,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM tools
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding, embedding, limit),
                )
            else:
                cur = await conn.execute(
                    """
                    SELECT name, description, owner_agent, invocation, tags, usage_count,
                           ts_rank(
                               to_tsvector('english', name || ' ' || description),
                               plainto_tsquery('english', %s)
                           ) AS similarity
                    FROM tools
                    WHERE to_tsvector('english', name || ' ' || description)
                          @@ plainto_tsquery('english', %s)
                    ORDER BY similarity DESC
                    LIMIT %s
                    """,
                    (query, query, limit),
                )
            rows = await cur.fetchall()
        log.debug("memory.tools_searched", query=query[:60], results=len(rows))
        return list(rows)

    async def get_tool(self, name: str) -> Optional[dict]:
        """Exact lookup by tool name."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT name, description, owner_agent, invocation, tags, usage_count "
                "FROM tools WHERE name = %s",
                (name,),
            )
            return await cur.fetchone()

    async def increment_tool_usage(self, name: str) -> None:
        """Bump usage_count so frequently-used tools sort higher over time."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE tools SET usage_count = usage_count + 1 WHERE name = %s",
                (name,),
            )

    # ── Executor capability registry ─────────────────────────────────────────
    # Successful task→command mappings are stored here so the executor can
    # re-run known operations without an LLM call.

    async def store_capability(
        self, task: str, command: str, tags: Optional[list[str]] = None
    ) -> None:
        """
        Persist a task→command mapping so the executor can reuse it.
        Uses an upsert on title so re-running the same task refreshes the entry.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO knowledge (agent, topic, title, content, tags)
                VALUES ('executor', 'capability', %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    task.lower().strip(),
                    command,
                    json.dumps(tags or ["executor", "capability"]),
                ),
            )
        log.debug("memory.capability_stored", task=task[:80], command=command[:80])

    async def lookup_capability(self, task: str) -> Optional[str]:
        """
        Look up a previously stored command for a task using full-text search.
        Returns the shell command string, or None if no confident match found.
        """
        pool = await self._get_pool()
        normalized = task.lower().strip()
        async with pool.connection() as conn:
            # Exact title match first (fastest, highest confidence)
            cur = await conn.execute(
                "SELECT content FROM knowledge WHERE topic = 'capability' AND title = %s LIMIT 1",
                (normalized,),
            )
            row = await cur.fetchone()
            if row:
                log.debug("memory.capability_hit_exact", task=task[:80])
                return row["content"]

            # FTS match — require all query terms to be present
            cur = await conn.execute(
                """
                SELECT content,
                       ts_rank(to_tsvector('english', title), plainto_tsquery('english', %s)) AS rank
                FROM knowledge
                WHERE topic = 'capability'
                  AND to_tsvector('english', title) @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT 1
                """,
                (normalized, normalized),
            )
            row = await cur.fetchone()
            # Require a high FTS rank to avoid confusing partial matches from
            # previous tasks (e.g. "list docker containers" ≠ "list docker logs").
            if row and row["rank"] > 0.3:
                log.debug("memory.capability_hit_fts", task=task[:80], rank=row["rank"])
                return row["content"]

        return None

    # ── Context snapshots ────────────────────────────────────────────────────
    # Unified recall + long-term memory for task, chat, and plan contexts.
    # snapshot_seq=0 is always the canonical "current" row; rolling checkpoints
    # use increasing seq numbers so users can roll back to any point.

    async def save_context_snapshot(
        self,
        context_id: str,
        context_type: str,
        name: str,
        *,
        summary: Optional[str] = None,
        snapshot_json: Optional[dict] = None,
        keywords: Optional[list[str]] = None,
        value_score: float = 0.0,
        topic_category: Optional[str] = None,
        message_count: int = 0,
        status: str = "active",
        snapshot_seq: int = 0,
        checkpoint_label: Optional[str] = None,
    ) -> None:
        """
        Create or update a context snapshot.  snapshot_seq=0 is the live row;
        call with a higher seq to save a rolling checkpoint.

        Pass *checkpoint_label* to mark this snapshot as a named task boundary
        that can later be used as a fork / step-off point.
        """
        embed_text = f"{name} {summary or ''} {topic_category or ''}"
        [embedding] = await asyncio.to_thread(_embed_texts, [embed_text])
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO context_snapshots
                    (context_id, snapshot_seq, context_type, name, status,
                     topic_category, keywords, summary, snapshot_json,
                     value_score, message_count, embedding, checkpoint_label)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (context_id, snapshot_seq) DO UPDATE SET
                    name             = EXCLUDED.name,
                    status           = EXCLUDED.status,
                    topic_category   = EXCLUDED.topic_category,
                    keywords         = EXCLUDED.keywords,
                    summary          = EXCLUDED.summary,
                    snapshot_json    = EXCLUDED.snapshot_json,
                    value_score      = EXCLUDED.value_score,
                    message_count    = EXCLUDED.message_count,
                    embedding        = EXCLUDED.embedding,
                    checkpoint_label = EXCLUDED.checkpoint_label,
                    updated_at       = NOW()
                """,
                (
                    context_id,
                    snapshot_seq,
                    context_type,
                    name,
                    status,
                    topic_category,
                    json.dumps(keywords or []),
                    summary,
                    json.dumps(snapshot_json or {}),
                    value_score,
                    message_count,
                    embedding,
                    checkpoint_label,
                ),
            )
        log.debug(
            "memory.context_snapshot_saved",
            context_id=context_id,
            seq=snapshot_seq,
            checkpoint_label=checkpoint_label,
        )

    async def close_context_snapshot(
        self,
        context_id: str,
        summary: str,
        snapshot_json: dict,
        value_score: float,
        *,
        checkpoint: bool = True,
    ) -> None:
        """
        Mark the live snapshot (seq=0) as closed and optionally save a final
        checkpoint with the full state for rollback.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """
                UPDATE context_snapshots
                SET status    = 'closed',
                    summary   = COALESCE(%s, summary),
                    value_score = %s,
                    closed_at = NOW(),
                    updated_at = NOW()
                WHERE context_id = %s AND snapshot_seq = 0
                """,
                (summary, value_score, context_id),
            )
            if checkpoint:
                # Determine next seq
                cur = await conn.execute(
                    "SELECT MAX(snapshot_seq) AS mx FROM context_snapshots WHERE context_id = %s",
                    (context_id,),
                )
                row = await cur.fetchone()
                next_seq = (row["mx"] or 0) + 1
                # Read current live row to copy metadata
                cur = await conn.execute(
                    "SELECT context_type, name, topic_category, keywords, message_count "
                    "FROM context_snapshots WHERE context_id = %s AND snapshot_seq = 0",
                    (context_id,),
                )
                live = await cur.fetchone()
                if live:
                    await conn.execute(
                        """
                        INSERT INTO context_snapshots
                            (context_id, snapshot_seq, context_type, name, status,
                             topic_category, keywords, summary, snapshot_json,
                             value_score, message_count, closed_at)
                        VALUES (%s, %s, %s, %s, 'closed', %s, %s, %s, %s, %s, %s, NOW())
                        """,
                        (
                            context_id,
                            next_seq,
                            live["context_type"],
                            live["name"],
                            live["topic_category"],
                            live["keywords"],
                            summary,
                            json.dumps(snapshot_json),
                            value_score,
                            live["message_count"],
                        ),
                    )
        log.info(
            "memory.context_closed", context_id=context_id, value_score=value_score
        )

    async def get_context_snapshot(
        self, context_id: str, snapshot_seq: int = 0
    ) -> Optional[dict]:
        """Return a specific snapshot row, or None if not found."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM context_snapshots WHERE context_id = %s AND snapshot_seq = %s",
                (context_id, snapshot_seq),
            )
            return await cur.fetchone()

    async def list_context_checkpoints(self, context_id: str) -> list[dict]:
        """Return all rolling snapshots for a context ordered oldest→newest."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT context_id, snapshot_seq, status, summary, value_score, "
                "message_count, created_at, closed_at "
                "FROM context_snapshots WHERE context_id = %s ORDER BY snapshot_seq ASC",
                (context_id,),
            )
            return list(await cur.fetchall())

    async def search_context_snapshots(
        self,
        query: str,
        context_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Semantic + FTS search over closed and active context snapshots.
        Returns snapshot_seq=0 rows only (latest state per context).
        """
        [embedding] = await asyncio.to_thread(_embed_texts, [query])
        pool = await self._get_pool()
        type_filter = "AND context_type = %s" if context_type else ""
        async with pool.connection() as conn:
            if embedding is not None:
                params = [embedding, embedding]
                if context_type:
                    params.append(context_type)
                params.append(limit)
                cur = await conn.execute(
                    f"""
                    SELECT context_id, context_type, name, status, topic_category,
                           keywords, summary, value_score, message_count,
                           created_at, closed_at,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM context_snapshots
                    WHERE snapshot_seq = 0 {type_filter}
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    params,
                )
            else:
                params = [query, query]
                if context_type:
                    params.append(context_type)
                params.append(limit)
                cur = await conn.execute(
                    f"""
                    SELECT context_id, context_type, name, status, topic_category,
                           keywords, summary, value_score, message_count,
                           created_at, closed_at
                    FROM context_snapshots
                    WHERE snapshot_seq = 0
                      AND to_tsvector('english', name || ' ' || COALESCE(summary,'') || ' ' || COALESCE(topic_category,''))
                          @@ plainto_tsquery('english', %s)
                      {type_filter}
                    ORDER BY ts_rank(
                        to_tsvector('english', name || ' ' || COALESCE(summary,'') || ' ' || COALESCE(topic_category,'')),
                        plainto_tsquery('english', %s)
                    ) DESC
                    LIMIT %s
                    """,
                    params,
                )
            rows = await cur.fetchall()
        return list(rows)

    async def save_named_checkpoint(
        self,
        context_id: str,
        label: str,
        snapshot_json: dict,
        *,
        summary: Optional[str] = None,
        message_count: int = 0,
    ) -> int:
        """
        Write a named checkpoint snapshot for *context_id*.

        Reads the live row (seq=0) for metadata, allocates the next seq, and
        writes a new snapshot row tagged with *label*.  The label is the
        agent-meaningful task-boundary name (e.g. ``"plan_approved"``,
        ``"step_2_complete"``).

        Returns the snapshot_seq that was assigned.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT MAX(snapshot_seq) AS mx FROM context_snapshots WHERE context_id = %s",
                (context_id,),
            )
            row = await cur.fetchone()
            next_seq = (row["mx"] or 0) + 1

            cur = await conn.execute(
                "SELECT context_type, name, topic_category, keywords, embedding "
                "FROM context_snapshots WHERE context_id = %s AND snapshot_seq = 0",
                (context_id,),
            )
            live = await cur.fetchone()
            if not live:
                log.warning(
                    "memory.checkpoint_no_live_row", context_id=context_id, label=label
                )
                return 0

            await conn.execute(
                """
                INSERT INTO context_snapshots
                    (context_id, snapshot_seq, context_type, name, status,
                     topic_category, keywords, summary, snapshot_json,
                     message_count, embedding, checkpoint_label)
                VALUES (%s, %s, %s, %s, 'checkpoint', %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    context_id,
                    next_seq,
                    live["context_type"],
                    live["name"],
                    live["topic_category"],
                    live["keywords"],
                    summary,
                    json.dumps(snapshot_json),
                    message_count,
                    live["embedding"],
                    label,
                ),
            )
        log.info(
            "memory.named_checkpoint_saved",
            context_id=context_id,
            seq=next_seq,
            label=label,
        )
        return next_seq

    async def get_latest_named_checkpoint(
        self,
        context_id: str,
        label: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Return the most recent named checkpoint for *context_id*.

        If *label* is provided, return the latest checkpoint with that exact label.
        Otherwise return the latest checkpoint regardless of label.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            if label:
                cur = await conn.execute(
                    """
                    SELECT context_id, snapshot_seq, context_type, name, summary,
                           snapshot_json, message_count, checkpoint_label, updated_at
                    FROM context_snapshots
                    WHERE context_id = %s AND checkpoint_label = %s
                    ORDER BY snapshot_seq DESC
                    LIMIT 1
                    """,
                    (context_id, label),
                )
            else:
                cur = await conn.execute(
                    """
                    SELECT context_id, snapshot_seq, context_type, name, summary,
                           snapshot_json, message_count, checkpoint_label, updated_at
                    FROM context_snapshots
                    WHERE context_id = %s AND checkpoint_label IS NOT NULL
                    ORDER BY snapshot_seq DESC
                    LIMIT 1
                    """,
                    (context_id,),
                )
            return await cur.fetchone()

    async def fork_from_checkpoint(
        self,
        source_context_id: str,
        new_context_id: str,
        new_context_name: str,
        *,
        label: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Create a new context snapshot row pre-populated from the latest named
        checkpoint of *source_context_id*.

        The new row (seq=0) carries:
        - ``forked_from_context_id`` and ``forked_from_seq`` in snapshot_json
          so the agent knows its provenance
        - ``checkpoint_label`` cleared (the fork is a fresh live row)

        Returns the checkpoint row that was used as the step-off point,
        or None if no matching checkpoint exists.
        """
        checkpoint = await self.get_latest_named_checkpoint(
            source_context_id, label=label
        )
        if not checkpoint:
            log.warning(
                "memory.fork_no_checkpoint",
                source=source_context_id,
                label=label,
            )
            return None

        fork_json = {
            **(
                json.loads(checkpoint["snapshot_json"])
                if isinstance(checkpoint["snapshot_json"], str)
                else (checkpoint["snapshot_json"] or {})
            ),
            "forked_from_context_id": source_context_id,
            "forked_from_seq": checkpoint["snapshot_seq"],
            "forked_from_label": checkpoint["checkpoint_label"],
        }

        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO context_snapshots
                    (context_id, snapshot_seq, context_type, name, status,
                     summary, snapshot_json, message_count, checkpoint_label)
                VALUES (%s, 0, %s, %s, 'active', %s, %s, %s, NULL)
                ON CONFLICT (context_id, snapshot_seq) DO UPDATE SET
                    context_type  = EXCLUDED.context_type,
                    name          = EXCLUDED.name,
                    status        = 'active',
                    summary       = EXCLUDED.summary,
                    snapshot_json = EXCLUDED.snapshot_json,
                    message_count = EXCLUDED.message_count,
                    checkpoint_label = NULL,
                    updated_at    = NOW()
                """,
                (
                    new_context_id,
                    checkpoint["context_type"],
                    new_context_name,
                    checkpoint["summary"],
                    json.dumps(fork_json),
                    checkpoint["message_count"],
                ),
            )
        log.info(
            "memory.context_forked",
            source=source_context_id,
            dest=new_context_id,
            step_off_seq=checkpoint["snapshot_seq"],
            step_off_label=checkpoint["checkpoint_label"],
        )
        return dict(checkpoint)

    async def get_closed_session_count(self) -> int:
        """Total number of closed contexts — used to decide when to start asking users for topic labels."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT COUNT(*) AS n FROM context_snapshots WHERE snapshot_seq = 0 AND status = 'closed'"
            )
            row = await cur.fetchone()
        return row["n"] if row else 0

    # ── Topic patterns ───────────────────────────────────────────────────────
    # Agents observe keyword→category correlations over sessions.
    # Confidence grows as patterns repeat.  Low-confidence patterns prompt
    # the user for a label (after enough sessions have accumulated).

    async def save_topic_pattern(
        self,
        category: str,
        keywords: list[str],
        confidence: float = 0.5,
        created_by: str = "orchestrator",
    ) -> None:
        """Insert a new topic pattern or update match_count + confidence if it already exists."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            # Check for an existing pattern with overlapping keywords
            cur = await conn.execute(
                "SELECT id, match_count FROM topic_patterns WHERE category = %s AND keywords = %s::jsonb",
                (category, json.dumps(sorted(keywords))),
            )
            row = await cur.fetchone()
            if row:
                new_count = row["match_count"] + 1
                # Confidence converges toward 1.0 as match_count grows (log scale)
                import math

                new_conf = min(0.99, 0.5 + 0.5 * math.log1p(new_count) / math.log1p(50))
                await conn.execute(
                    "UPDATE topic_patterns SET match_count = %s, confidence = %s, updated_at = NOW() WHERE id = %s",
                    (new_count, new_conf, row["id"]),
                )
            else:
                await conn.execute(
                    "INSERT INTO topic_patterns (category, keywords, confidence, created_by) VALUES (%s, %s::jsonb, %s, %s)",
                    (category, json.dumps(sorted(keywords)), confidence, created_by),
                )
        log.debug(
            "memory.topic_pattern_saved", category=category, keywords=keywords[:5]
        )

    async def search_topic_patterns(
        self, keywords: list[str], limit: int = 5
    ) -> list[dict]:
        """
        Find stored patterns whose keyword sets overlap most with *keywords*.
        Returns rows ordered by Jaccard-style overlap (approximated via array intersection).
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            # Retrieve all patterns and compute overlap in Python (table stays small)
            cur = await conn.execute(
                "SELECT id, category, keywords, match_count, confidence FROM topic_patterns ORDER BY confidence DESC"
            )
            rows = await cur.fetchall()

        if not rows:
            return []

        query_set = set(kw.lower() for kw in keywords)
        scored: list[tuple[float, dict]] = []
        for row in rows:
            pat_set = set(kw.lower() for kw in json.loads(row["keywords"]))
            union = query_set | pat_set
            if not union:
                continue
            jaccard = len(query_set & pat_set) / len(union)
            scored.append((jaccard * row["confidence"], dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

    # ── Eval results ─────────────────────────────────────────────────────────

    async def save_eval_result(self, result: dict) -> int:
        """
        Insert a new eval result row. Returns the generated id.

        Expected keys in result:
          task_id, plan_id, task_type, original_task, artifact_path,
          tier1_passed, tier1_reasons, tier2_score, tier2_breakdown, tier2_flags,
          tier3_score, tier3_model, final_score, review_status,
          plan_steps, plan_retries, approval_requested, agents_used
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                INSERT INTO eval_results (
                    task_id, plan_id, task_type, original_task, artifact_path,
                    tier1_passed, tier1_reasons, tier2_score, tier2_breakdown, tier2_flags,
                    tier3_score, tier3_model, final_score, review_status,
                    plan_steps, plan_retries, approval_requested, agents_used
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s
                ) RETURNING id
                """,
                (
                    result.get("task_id", ""),
                    result.get("plan_id", ""),
                    result.get("task_type", "unknown"),
                    result.get("original_task", ""),
                    result.get("artifact_path"),
                    result.get("tier1_passed", False),
                    json.dumps(result.get("tier1_reasons", [])),
                    result.get("tier2_score"),
                    json.dumps(result.get("tier2_breakdown"))
                    if result.get("tier2_breakdown")
                    else None,
                    json.dumps(result.get("tier2_flags", [])),
                    result.get("tier3_score"),
                    result.get("tier3_model"),
                    result.get("final_score"),
                    result.get("review_status", "pending"),
                    result.get("plan_steps", 0),
                    result.get("plan_retries", 0),
                    result.get("approval_requested", False),
                    json.dumps(result.get("agents_used", [])),
                ),
            )
            row = await cur.fetchone()
            return row["id"] if row else -1

    async def resolve_eval_result(
        self,
        eval_id: int,
        verdict: bool,
        feedback: str = "",
    ) -> None:
        """Record a user thumbs-up (True) / thumbs-down (False) on an eval result."""
        pool = await self._get_pool()
        status = "approved" if verdict else "rejected"
        async with pool.connection() as conn:
            await conn.execute(
                """
                UPDATE eval_results
                SET user_verdict = %s, user_feedback = %s,
                    review_status = %s, reviewed_at = NOW()
                WHERE id = %s
                """,
                (verdict, feedback, status, eval_id),
            )

    async def get_pending_eval_reviews(self, limit: int = 20) -> list[dict]:
        """Return eval results awaiting user review, newest first."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, task_id, task_type, original_task, artifact_path,
                       tier1_passed, tier2_score, tier3_score, final_score,
                       tier2_flags, plan_retries, approval_requested, created_at
                FROM eval_results
                WHERE review_status = 'pending'
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return list(await cur.fetchall())

    async def get_eval_stats(self, task_type: str | None = None) -> dict:
        """Return aggregate eval stats, optionally filtered by task_type."""
        pool = await self._get_pool()
        type_filter = "WHERE task_type = %s" if task_type else ""
        params = (task_type,) if task_type else ()
        async with pool.connection() as conn:
            cur = await conn.execute(
                f"""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN tier1_passed THEN 1 ELSE 0 END) AS tier1_passed,
                    ROUND(AVG(tier2_score)::numeric, 2) AS avg_tier2,
                    ROUND(AVG(final_score)::numeric, 2) AS avg_final,
                    SUM(CASE WHEN approval_requested THEN 1 ELSE 0 END) AS approval_interrupts,
                    SUM(CASE WHEN user_verdict = TRUE THEN 1 ELSE 0 END) AS user_approved,
                    SUM(CASE WHEN user_verdict = FALSE THEN 1 ELSE 0 END) AS user_rejected
                FROM eval_results
                {type_filter}
                """,
                params,
            )
            row = await cur.fetchone()
            return dict(row) if row else {}

    async def confirm_topic_pattern(
        self, category: str, keywords: list[str], confirmed_by: str = "user"
    ) -> None:
        """
        Boost confidence of a pattern to 0.99 after user confirmation.
        Also resets match_count to 50 so future variations inherit high base confidence.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """
                UPDATE topic_patterns
                SET confidence = 0.99, match_count = 50, updated_at = NOW()
                WHERE category = %s AND keywords = %s::jsonb
                """,
                (category, json.dumps(sorted(keywords))),
            )
        log.info(
            "memory.topic_pattern_confirmed",
            category=category,
            confirmed_by=confirmed_by,
        )
