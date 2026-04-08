"""
tests/perf/test_event_bus_throughput.py
────────────────────────────────────────
Redis Streams publish/consume throughput.

Measures:
  - Publish rate (events/s) for burst of N events
  - Consume rate (events/s) for a consumer group draining the same burst
  - Round-trip latency (publish → consume) for a single event

Marks:
  @pytest.mark.perf        — all tests here
  @pytest.mark.redis_live  — skipped unless Redis is reachable

Run:
  pytest tests/perf/test_event_bus_throughput.py -v -m perf
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import os

import pytest

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PERF_STREAM = "perf:event_bus_test"
PERF_GROUP = "perf_consumer_group"
PERF_CONSUMER = "perf_consumer_1"

BURST_SIZE = 500
REGRESSION_THRESHOLD = 0.20  # 20% drop triggers failure

BASELINE_KEY_PUB = "event_bus_publish_eps"
BASELINE_KEY_RTT = "event_bus_rtt_ms"


# ── Reachability ──────────────────────────────────────────────────────────────


def _redis_available() -> bool:
    try:
        import redis

        r = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2)
        r.ping()
        r.close()
        return True
    except Exception:
        return False


redis_live = pytest.mark.skipif(
    not _redis_available(),
    reason=f"Redis not reachable at {REDIS_URL} — start the stack to run perf tests",
)


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _get_redis():
    import redis.asyncio as aioredis

    return await aioredis.from_url(REDIS_URL, decode_responses=True)


async def _cleanup(r, stream: str = PERF_STREAM) -> None:
    """Delete the test stream so runs don't interfere."""
    try:
        await r.delete(stream)
    except Exception:
        pass


async def _ensure_group(r, stream: str, group: str) -> None:
    try:
        await r.xgroup_create(stream, group, id="0", mkstream=True)
    except Exception:
        pass  # group already exists


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.perf
@redis_live
async def test_publish_throughput(perf_baseline, update_perf):
    """
    Publish BURST_SIZE events as fast as possible. Records events/second.
    Fails if throughput drops >20% below stored baseline.
    """
    r = await _get_redis()
    await _cleanup(r)

    payload = json.dumps(
        {
            "type": "task.created",
            "source": "perf_test",
            "payload": {"seq": 0, "data": "x" * 64},
        }
    )

    t0 = time.perf_counter()
    pipe = r.pipeline(transaction=False)
    for i in range(BURST_SIZE):
        pipe.xadd(PERF_STREAM, {"event": payload, "seq": str(i)})
    await pipe.execute()
    elapsed = time.perf_counter() - t0

    eps = BURST_SIZE / elapsed
    print(
        f"\n[PERF] publish_throughput: {eps:.0f} evt/s  "
        f"({BURST_SIZE} events in {elapsed:.3f}s)"
    )

    await _cleanup(r)
    await r.aclose()

    kpi_key = BASELINE_KEY_PUB
    if update_perf:
        perf_baseline[kpi_key] = round(eps, 1)
        _save_baseline(perf_baseline)
        pytest.skip(f"Baseline updated: {eps:.0f} evt/s")

    if kpi_key in perf_baseline:
        baseline_eps = perf_baseline[kpi_key]
        min_eps = baseline_eps * (1.0 - REGRESSION_THRESHOLD)
        assert eps >= min_eps, (
            f"Publish throughput regression: {eps:.0f} evt/s < {min_eps:.0f} evt/s "
            f"(baseline {baseline_eps:.0f} evt/s)"
        )
    else:
        perf_baseline[kpi_key] = round(eps, 1)
        _save_baseline(perf_baseline)
        print(f"[PERF] No baseline — saved {eps:.0f} evt/s")


@pytest.mark.asyncio
@pytest.mark.perf
@redis_live
async def test_consume_throughput(perf_baseline, update_perf):
    """
    Pre-fill a stream with BURST_SIZE events, then consume them all via a
    consumer group. Records consume rate (events/s).
    """
    r = await _get_redis()
    await _cleanup(r)
    await _ensure_group(r, PERF_STREAM, PERF_GROUP)

    # Pre-fill
    payload = json.dumps({"type": "task.created", "source": "perf_test"})
    pipe = r.pipeline(transaction=False)
    for i in range(BURST_SIZE):
        pipe.xadd(PERF_STREAM, {"event": payload, "seq": str(i)})
    await pipe.execute()

    # Consume
    consumed = 0
    t0 = time.perf_counter()
    while consumed < BURST_SIZE:
        entries = await r.xreadgroup(
            PERF_GROUP,
            PERF_CONSUMER,
            {PERF_STREAM: ">"},
            count=100,
            block=200,
        )
        if not entries:
            break
        for _stream, messages in entries:
            ids = [msg_id for msg_id, _ in messages]
            if ids:
                await r.xack(PERF_STREAM, PERF_GROUP, *ids)
            consumed += len(messages)

    elapsed = time.perf_counter() - t0
    eps = consumed / elapsed if elapsed > 0 else 0

    print(
        f"\n[PERF] consume_throughput: {eps:.0f} evt/s  "
        f"({consumed} events in {elapsed:.3f}s)"
    )

    await _cleanup(r)
    await r.aclose()

    kpi_key = "event_bus_consume_eps"
    if update_perf:
        perf_baseline[kpi_key] = round(eps, 1)
        _save_baseline(perf_baseline)
        pytest.skip(f"Baseline updated: {eps:.0f} evt/s")

    if kpi_key in perf_baseline:
        baseline_eps = perf_baseline[kpi_key]
        min_eps = baseline_eps * (1.0 - REGRESSION_THRESHOLD)
        assert eps >= min_eps, (
            f"Consume throughput regression: {eps:.0f} evt/s < {min_eps:.0f} evt/s"
        )
    else:
        perf_baseline[kpi_key] = round(eps, 1)
        _save_baseline(perf_baseline)
        print(f"[PERF] No baseline — saved {eps:.0f} evt/s")


@pytest.mark.asyncio
@pytest.mark.perf
@redis_live
async def test_round_trip_latency():
    """
    Single event round-trip: publish to stream → consume via group.
    P50 and P95 over 50 iterations. No baseline regression — informational.
    """
    r = await _get_redis()
    stream = f"perf:rtt_{uuid.uuid4().hex[:8]}"
    group = "perf_rtt_group"
    await _ensure_group(r, stream, group)

    latencies_ms: list[float] = []

    for _ in range(50):
        payload = json.dumps({"type": "task.created", "ts": time.time()})
        t0 = time.perf_counter()
        await r.xadd(stream, {"event": payload})

        while True:
            entries = await r.xreadgroup(
                group, "rtt_consumer", {stream: ">"}, count=1, block=100
            )
            if entries:
                for _s, messages in entries:
                    ids = [mid for mid, _ in messages]
                    await r.xack(stream, group, *ids)
                break

        latencies_ms.append((time.perf_counter() - t0) * 1000)

    await r.delete(stream)
    await r.aclose()

    latencies_ms.sort()
    p50 = latencies_ms[len(latencies_ms) // 2]
    p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
    print(f"\n[PERF] round_trip_latency: p50={p50:.2f}ms  p95={p95:.2f}ms")

    assert p95 < 100, f"RTT p95 too high: {p95:.2f}ms (expected <100ms)"


# ── Baseline persistence ──────────────────────────────────────────────────────

import json as _json
from pathlib import Path

_BASELINE_FILE = Path(__file__).parent / "perf_baseline.json"


def _save_baseline(data: dict) -> None:
    _BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _BASELINE_FILE.write_text(_json.dumps(data, indent=2))


# ── Fixtures (local to this module — shared ones in conftest) ─────────────────


@pytest.fixture
def perf_baseline() -> dict:
    if _BASELINE_FILE.exists():
        return _json.loads(_BASELINE_FILE.read_text())
    return {}


@pytest.fixture
def update_perf(request) -> bool:
    return request.config.getoption("--update-perf", default=False)
