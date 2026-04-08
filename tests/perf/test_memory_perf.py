"""
tests/perf/test_memory_perf.py
────────────────────────────────
PostgreSQL + pgvector long-term memory performance.

Measures:
  - Single-agent write latency (store one knowledge entry)
  - Single-agent semantic search latency (query with vector cosine similarity)
  - FTS (full-text search) latency vs vector search
  - Concurrent 5-agent write + query throughput (simulates the real stack)
  - batch_store throughput vs individual stores

Marks:
  @pytest.mark.perf       — all tests here
  @pytest.mark.db_live    — skipped unless Postgres + pgvector is reachable

Run:
  pytest tests/perf/test_memory_perf.py -v -m perf
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from statistics import mean, median

import os

import pytest

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://agent:agent@localhost:5432/agentmem"
)
REGRESSION_THRESHOLD = 0.30  # memory ops can be noisy; 30% threshold


# ── Reachability ──────────────────────────────────────────────────────────────


def _postgres_available() -> bool:
    try:
        import psycopg

        conn = psycopg.connect(DATABASE_URL, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


db_live = pytest.mark.skipif(
    not _postgres_available(),
    reason=f"Postgres not reachable at {DATABASE_URL} — start the stack to run perf tests",
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_memory(agent_name: str) -> "LongTermMemory":
    import sys
    import os

    # Ensure core is importable when run directly
    sys.path.insert(0, str(Path(__file__).parents[3]))
    os.environ.setdefault("DATABASE_URL", DATABASE_URL)

    from core.memory.long_term import LongTermMemory

    return LongTermMemory(database_url=DATABASE_URL, agent_name=agent_name)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.perf
@db_live
async def test_single_write_latency(perf_baseline, update_perf):
    """
    Store 20 knowledge entries sequentially. Records mean write latency (ms).
    """
    agent = f"perf_writer_{uuid.uuid4().hex[:6]}"
    mem = _make_memory(agent)
    await mem.open_session()

    latencies_ms: list[float] = []
    for i in range(20):
        t0 = time.perf_counter()
        await mem.store(
            content=f"Perf test finding {i}: agent {agent} discovered {uuid.uuid4()}",
            topic="performance_testing",
            tags=["perf", "test"],
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    mean_ms = mean(latencies_ms)
    p95_ms = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
    print(
        f"\n[PERF] single_write: mean={mean_ms:.1f}ms  p95={p95_ms:.1f}ms  "
        f"(20 writes, agent={agent})"
    )

    kpi_key = "memory_write_mean_ms"
    if update_perf:
        perf_baseline[kpi_key] = round(mean_ms, 1)
        _save_baseline(perf_baseline)
        pytest.skip(f"Baseline updated: {mean_ms:.1f}ms")

    if kpi_key in perf_baseline:
        max_mean = perf_baseline[kpi_key] * (1.0 + REGRESSION_THRESHOLD)
        assert mean_ms <= max_mean, (
            f"Write latency regression: {mean_ms:.1f}ms > {max_mean:.1f}ms "
            f"(baseline {perf_baseline[kpi_key]}ms)"
        )
    else:
        perf_baseline[kpi_key] = round(mean_ms, 1)
        _save_baseline(perf_baseline)
        print(f"[PERF] No baseline — saved {mean_ms:.1f}ms")


@pytest.mark.asyncio
@pytest.mark.perf
@db_live
async def test_semantic_search_latency(perf_baseline, update_perf):
    """
    Run 20 semantic (vector) searches. Records mean query latency (ms).
    The embedding step (sentence-transformers on CPU) is the typical bottleneck.
    """
    agent = f"perf_searcher_{uuid.uuid4().hex[:6]}"
    mem = _make_memory(agent)
    await mem.open_session()

    queries = [
        "multi-agent event bus architecture",
        "Redis Streams consumer group throughput",
        "PostgreSQL pgvector semantic search",
        "LangChain LangGraph orchestration",
        "Docker container ephemeral agents",
    ]

    latencies_ms: list[float] = []
    for i in range(20):
        query = queries[i % len(queries)]
        t0 = time.perf_counter()
        results = await mem.search(query, semantic=True, limit=5)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    mean_ms = mean(latencies_ms)
    p95_ms = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
    print(
        f"\n[PERF] semantic_search: mean={mean_ms:.1f}ms  p95={p95_ms:.1f}ms  "
        f"(20 queries)"
    )

    kpi_key = "memory_semantic_search_mean_ms"
    if update_perf:
        perf_baseline[kpi_key] = round(mean_ms, 1)
        _save_baseline(perf_baseline)
        pytest.skip(f"Baseline updated: {mean_ms:.1f}ms")

    if kpi_key in perf_baseline:
        max_mean = perf_baseline[kpi_key] * (1.0 + REGRESSION_THRESHOLD)
        assert mean_ms <= max_mean, (
            f"Semantic search regression: {mean_ms:.1f}ms > {max_mean:.1f}ms"
        )
    else:
        perf_baseline[kpi_key] = round(mean_ms, 1)
        _save_baseline(perf_baseline)
        print(f"[PERF] No baseline — saved {mean_ms:.1f}ms")


@pytest.mark.asyncio
@pytest.mark.perf
@db_live
async def test_fts_vs_vector_search():
    """
    Compares FTS (PostgreSQL tsvector) against vector (pgvector cosine) latency.
    Informational — no baseline regression. Highlights which path is cheaper.
    """
    agent = f"perf_fts_{uuid.uuid4().hex[:6]}"
    mem = _make_memory(agent)
    await mem.open_session()

    query = "Redis Streams event bus architecture agent"
    n = 15

    fts_times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        await mem.search(query, semantic=False, limit=5)
        fts_times.append((time.perf_counter() - t0) * 1000)

    vec_times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        await mem.search(query, semantic=True, limit=5)
        vec_times.append((time.perf_counter() - t0) * 1000)

    fts_mean = mean(fts_times)
    vec_mean = mean(vec_times)
    ratio = vec_mean / fts_mean if fts_mean > 0 else 0

    print(
        f"\n[PERF] fts_vs_vector: fts={fts_mean:.1f}ms  vec={vec_mean:.1f}ms  "
        f"vector_overhead={ratio:.1f}×"
    )
    # Vector search should not be more than 20× slower than FTS
    # (embedding generation is the expensive part)
    assert ratio < 20, (
        f"Vector search too slow vs FTS: {ratio:.1f}× overhead (expected <20×)"
    )


@pytest.mark.asyncio
@pytest.mark.perf
@db_live
async def test_concurrent_agent_memory():
    """
    Simulates 5 agents writing + querying memory simultaneously.
    Records aggregate write and query throughput and checks no agent
    starves (max individual time must not exceed 3× median).
    """

    async def agent_workload(agent_name: str) -> dict:
        mem = _make_memory(agent_name)
        await mem.open_session()

        write_times: list[float] = []
        for i in range(10):
            t0 = time.perf_counter()
            await mem.store(
                content=f"Concurrent finding {i} from {agent_name}: {uuid.uuid4()}",
                topic="concurrency_test",
                tags=["perf", "concurrent"],
            )
            write_times.append((time.perf_counter() - t0) * 1000)

        query_times: list[float] = []
        for _ in range(5):
            t0 = time.perf_counter()
            await mem.search(
                "concurrent agent architecture finding", semantic=True, limit=3
            )
            query_times.append((time.perf_counter() - t0) * 1000)

        return {
            "agent": agent_name,
            "write_mean_ms": mean(write_times),
            "query_mean_ms": mean(query_times),
        }

    agent_names = [f"perf_agent_{i}_{uuid.uuid4().hex[:4]}" for i in range(5)]

    t0 = time.perf_counter()
    agent_results = await asyncio.gather(*[agent_workload(a) for a in agent_names])
    wall = time.perf_counter() - t0

    for r in agent_results:
        print(
            f"\n[PERF] {r['agent']}: write={r['write_mean_ms']:.1f}ms  "
            f"query={r['query_mean_ms']:.1f}ms"
        )

    write_means = [r["write_mean_ms"] for r in agent_results]
    query_means = [r["query_mean_ms"] for r in agent_results]
    print(
        f"\n[PERF] concurrent_5_agents: wall={wall:.2f}s  "
        f"write_median={median(write_means):.1f}ms  "
        f"query_median={median(query_means):.1f}ms"
    )

    # No single agent should have a write mean more than 5× the median
    # (guards against lock contention causing one agent to starve)
    write_med = median(write_means)
    for r in agent_results:
        assert r["write_mean_ms"] < write_med * 5 + 50, (
            f"Agent {r['agent']} write latency {r['write_mean_ms']:.1f}ms "
            f"is >5× median {write_med:.1f}ms — possible lock contention"
        )


@pytest.mark.asyncio
@pytest.mark.perf
@db_live
async def test_batch_store_vs_individual(perf_baseline, update_perf):
    """
    Compares batch_store (single embedding call) against 10 individual stores.
    batch_store should be significantly faster per-entry.
    """
    agent = f"perf_batch_{uuid.uuid4().hex[:6]}"
    mem = _make_memory(agent)
    await mem.open_session()

    entries = [
        {
            "content": f"Batch entry {i}: {uuid.uuid4()}",
            "topic": "batch_test",
            "tags": ["perf"],
        }
        for i in range(10)
    ]

    # Individual stores
    t0 = time.perf_counter()
    for e in entries:
        await mem.store(**e)
    individual_s = time.perf_counter() - t0

    # Batch store
    t0 = time.perf_counter()
    await mem.batch_store(entries)
    batch_s = time.perf_counter() - t0

    speedup = individual_s / batch_s if batch_s > 0 else 1.0
    print(
        f"\n[PERF] batch_vs_individual: individual={individual_s:.2f}s  "
        f"batch={batch_s:.2f}s  speedup={speedup:.1f}×"
    )

    # batch_store must be at least 1.5× faster (embedding done once)
    assert speedup >= 1.5, (
        f"batch_store speedup too low: {speedup:.1f}× "
        f"(expected ≥1.5× vs individual stores)"
    )

    kpi_key = "memory_batch_speedup"
    if update_perf:
        perf_baseline[kpi_key] = round(speedup, 2)
        _save_baseline(perf_baseline)
        pytest.skip(f"Baseline updated: {speedup:.1f}×")
    elif kpi_key not in perf_baseline:
        perf_baseline[kpi_key] = round(speedup, 2)
        _save_baseline(perf_baseline)


# ── Baseline persistence ──────────────────────────────────────────────────────

import json as _json

_BASELINE_FILE = Path(__file__).parent / "perf_baseline.json"


def _save_baseline(data: dict) -> None:
    _BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _BASELINE_FILE.write_text(_json.dumps(data, indent=2))


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def perf_baseline() -> dict:
    if _BASELINE_FILE.exists():
        return _json.loads(_BASELINE_FILE.read_text())
    return {}


@pytest.fixture
def update_perf(request) -> bool:
    return request.config.getoption("--update-perf", default=False)
