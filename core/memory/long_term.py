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
import json
from typing import Optional

import structlog
from psycopg import AsyncConnection
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

log = structlog.get_logger()

# ── Embedding model (lazy-loaded, sentence-transformers already in container) ─

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def _embed_texts(texts: list[str]) -> list[list[float] | None]:
    """Batch-encode texts to 384-dim vectors. Returns None entries on failure."""
    try:
        model = _get_encoder()
        vectors = model.encode(texts)
        return [v.tolist() for v in vectors]
    except Exception as exc:
        log.warning("memory.embed_failed", error=str(exc))
        return [None] * len(texts)


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
        embedding  vector(384)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_knowledge_agent ON knowledge(agent)",
    """
    CREATE INDEX IF NOT EXISTS idx_knowledge_fts ON knowledge
        USING GIN(to_tsvector('english', title || ' ' || content))
    """,
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
                for stmt in _SCHEMA:
                    await conn.execute(stmt)
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
            await conn.execute("""
                INSERT INTO agent_status (agent, status, current_task, updated_at)
                VALUES (%s, 'OPEN', '', NOW())
                ON CONFLICT (agent) DO UPDATE SET
                    status = 'OPEN', current_task = '', updated_at = NOW()
            """, (self._agent,))
        status = "CRASH" if crashed else "OK"
        log.info("memory.session_opened", agent=self._agent, status=status)
        return {"status": status, "agent": self._agent}

    async def recover_context(self) -> dict:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT status, current_task FROM agent_status WHERE agent = %s",
                (self._agent,)
            )
            row = await cur.fetchone()
            cur = await conn.execute(
                "SELECT summary FROM handoffs WHERE agent = %s ORDER BY id DESC LIMIT 1",
                (self._agent,)
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
                (note, self._agent)
            )
        log.debug("memory.checkpoint", agent=self._agent, note=note)

    async def close_session(self, summary: str) -> None:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO handoffs (agent, summary) VALUES (%s, %s)",
                (self._agent, summary)
            )
            await conn.execute(
                "UPDATE agent_status SET status = 'CLEAN', updated_at = NOW() WHERE agent = %s",
                (self._agent,)
            )
        log.info("memory.session_closed", agent=self._agent)

    # ── Knowledge store ──────────────────────────────────────────────────────

    async def store(self, content: str, topic: str,
                    tags: Optional[list[str]] = None, kind: str = "finding") -> dict:
        [embedding] = await asyncio.to_thread(_embed_texts, [content])
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute("""
                INSERT INTO knowledge (agent, topic, title, content, tags, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (self._agent, topic, topic, content, json.dumps(tags or []), embedding))
        log.info("memory.promoted", topic=topic, kind=kind)
        return {"status": "stored"}

    async def batch_store(self, entries: list[dict]) -> dict:
        if not entries:
            return {"status": "stored", "count": 0}
        texts = [e.get("content", "") for e in entries]
        embeddings = await asyncio.to_thread(_embed_texts, texts)
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.executemany("""
                INSERT INTO knowledge (agent, topic, title, content, tags, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, [
                (
                    self._agent,
                    e.get("topic", "general"),
                    e.get("topic", "general"),
                    e.get("content", ""),
                    json.dumps(e.get("tags") or []),
                    embeddings[i],
                )
                for i, e in enumerate(entries)
            ])
        log.info("memory.batch_promoted", count=len(entries))
        return {"status": "stored", "count": len(entries)}

    # ── Recall / search ──────────────────────────────────────────────────────

    async def recall(self, query: str, limit: int = 5) -> list[dict]:
        """Full-text search via PostgreSQL tsvector."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute("""
                SELECT id, agent, topic, title, content, tags, created_at
                FROM knowledge
                WHERE to_tsvector('english', title || ' ' || content)
                      @@ plainto_tsquery('english', %s)
                ORDER BY ts_rank(
                    to_tsvector('english', title || ' ' || content),
                    plainto_tsquery('english', %s)
                ) DESC
                LIMIT %s
            """, (query, query, limit))
            rows = await cur.fetchall()
        log.debug("memory.recalled", query=query, results=len(rows))
        return list(rows)

    async def vector_search(self, query: str, limit: int = 5) -> list[dict]:
        """Semantic search via pgvector cosine similarity."""
        [embedding] = await asyncio.to_thread(_embed_texts, [query])
        if embedding is None:
            return await self.recall(query, limit)
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute("""
                SELECT id, agent, topic, title, content, tags, created_at,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM knowledge
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding, embedding, limit))
            rows = await cur.fetchall()
        log.debug("memory.vector_recalled", query=query, results=len(rows))
        return list(rows)

    async def search(self, query: str, semantic: bool = False, limit: int = 5) -> list[dict]:
        if semantic:
            try:
                return await self.vector_search(query, limit)
            except Exception:
                log.warning("memory.vector_search_failed", fallback="fts")
        return await self.recall(query, limit)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
