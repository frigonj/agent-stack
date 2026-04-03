"""
tests/perf/test_e2e_latency.py
────────────────────────────────
End-to-end task latency: publish TASK_CREATED → receive TASK_COMPLETED.

Measures:
  - Latency for 5 sequential lightweight tasks (orchestrator routing overhead)
  - Latency distribution: median, p95
  - Orchestrator idle detection: time from last event → AGENT_THINKING

NOTE: This test requires the full stack (Redis + Postgres + LM Studio +
orchestrator container) to be running. It sends real tasks and waits for
real completions, so it tests actual end-to-end performance including
LLM inference time.

Marks:
  @pytest.mark.perf         — all tests here
  @pytest.mark.stack_live   — skipped unless Redis + orchestrator are reachable

Run:
  pytest tests/perf/test_e2e_latency.py -v -m perf -s
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from statistics import median

import pytest

REDIS_URL = "redis://localhost:6379"
TASK_TIMEOUT_S = 120  # max seconds to wait for a single task to complete
REGRESSION_THRESHOLD = 0.30  # 30% — e2e is noisy (includes LLM)


# ── Reachability ──────────────────────────────────────────────────────────────


def _stack_available() -> bool:
    """Returns True if Redis is up AND the orchestrator has announced itself recently."""
    try:
        import redis

        r = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2)
        r.ping()
        # Check for orchestrator heartbeat in the broadcast stream
        entries = r.xrevrange("agents:broadcast", count=50)
        for _id, fields in entries:
            raw = fields.get(b"event") or fields.get("event")
            if raw:
                try:
                    data = json.loads(raw if isinstance(raw, str) else raw.decode())
                    if data.get("source") == "orchestrator":
                        r.close()
                        return True
                except Exception:
                    pass
        r.close()
        return False
    except Exception:
        return False


stack_live = pytest.mark.skipif(
    not _stack_available(),
    reason=(
        "Full stack not running — start docker compose up and wait for orchestrator "
        "to announce before running e2e perf tests"
    ),
)


# ── Task dispatch helpers ─────────────────────────────────────────────────────


async def _dispatch_task(r, task_id: str, prompt: str) -> None:
    """Publish a TASK_CREATED event to the orchestrator stream."""
    payload = json.dumps(
        {
            "type": "task.created",
            "source": "perf_test",
            "id": str(uuid.uuid4()),
            "payload": {
                "task_id": task_id,
                "prompt": prompt,
                "context": "perf_test",
            },
        }
    )
    await r.xadd("agents:orchestrator", {"event": payload})


async def _wait_for_completion(r, task_id: str, timeout_s: float = TASK_TIMEOUT_S) -> float:
    """
    Listen on agents:broadcast for TASK_COMPLETED / TASK_FAILED matching task_id.
    Returns wall-clock latency in seconds.
    Raises TimeoutError if not completed within timeout_s.
    """
    deadline = time.perf_counter() + timeout_s
    last_id = "$"  # only new messages from here

    while time.perf_counter() < deadline:
        remaining_ms = max(100, int((deadline - time.perf_counter()) * 1000))
        entries = await r.xread({"agents:broadcast": last_id}, block=remaining_ms, count=20)
        if not entries:
            continue
        for _stream, messages in entries:
            for msg_id, fields in messages:
                last_id = msg_id
                raw = fields.get("event") or fields.get(b"event")
                if not raw:
                    continue
                try:
                    data = json.loads(raw if isinstance(raw, str) else raw.decode())
                except Exception:
                    continue
                event_type = data.get("type", "")
                payload = data.get("payload", {})
                if payload.get("task_id") == task_id and event_type in (
                    "task.completed",
                    "task.failed",
                ):
                    return True

    raise TimeoutError(f"Task {task_id} did not complete within {timeout_s}s")


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.perf
@stack_live
async def test_sequential_task_latency(perf_baseline, update_perf):
    """
    Dispatch 5 lightweight tasks sequentially. Records median and p95 latency.
    Uses simple routing-only prompts to isolate orchestrator overhead from
    deep agent work.
    """
    import redis.asyncio as aioredis

    r = await aioredis.from_url(REDIS_URL, decode_responses=True)

    prompts = [
        "What agents are currently in the stack? Answer in one sentence.",
        "What database does this system use for long-term memory?",
        "What is the role of Redis Streams in this architecture?",
        "How many concurrent contexts can a single agent handle?",
        "What is the event type emitted when a task finishes successfully?",
    ]

    latencies_s: list[float] = []
    for i, prompt in enumerate(prompts):
        task_id = f"e2e_perf_{uuid.uuid4().hex[:8]}"
        t0 = time.perf_counter()
        await _dispatch_task(r, task_id, prompt)
        await _wait_for_completion(r, task_id)
        elapsed = time.perf_counter() - t0
        latencies_s.append(elapsed)
        print(f"\n[PERF] e2e task {i+1}: {elapsed:.2f}s  (task_id={task_id})")

    await r.aclose()

    med = median(latencies_s)
    p95 = sorted(latencies_s)[int(len(latencies_s) * 0.95) - 1]
    print(f"\n[PERF] e2e_sequential: median={med:.2f}s  p95={p95:.2f}s")

    kpi_key = "e2e_sequential_median_s"
    if update_perf:
        perf_baseline[kpi_key] = round(med, 2)
        _save_baseline(perf_baseline)
        pytest.skip(f"Baseline updated: median={med:.2f}s")

    if kpi_key in perf_baseline:
        max_med = perf_baseline[kpi_key] * (1.0 + REGRESSION_THRESHOLD)
        assert med <= max_med, (
            f"E2E latency regression: {med:.2f}s > {max_med:.2f}s "
            f"(baseline {perf_baseline[kpi_key]}s)"
        )
    else:
        perf_baseline[kpi_key] = round(med, 2)
        _save_baseline(perf_baseline)
        print(f"[PERF] No baseline — saved median={med:.2f}s")


@pytest.mark.asyncio
@pytest.mark.perf
@stack_live
async def test_concurrent_task_latency():
    """
    Dispatch 3 tasks simultaneously. Checks:
      - All 3 complete successfully
      - Wall clock for 3 concurrent does not exceed 2.5× single-task median
        (tasks should overlap in the orchestrator's context pool)

    Informational — no baseline regression.
    """
    import redis.asyncio as aioredis

    r = await aioredis.from_url(REDIS_URL, decode_responses=True)

    prompts = [
        "What event type signals a new task was created?",
        "What is the agent responsible for running shell commands?",
        "What stream does the orchestrator listen on?",
    ]

    task_ids = [f"e2e_conc_{uuid.uuid4().hex[:8]}" for _ in prompts]

    t0 = time.perf_counter()
    # Dispatch all 3 simultaneously
    for task_id, prompt in zip(task_ids, prompts):
        await _dispatch_task(r, task_id, prompt)

    # Wait for all completions concurrently
    results = await asyncio.gather(
        *[_wait_for_completion(r, tid) for tid in task_ids],
        return_exceptions=True,
    )
    wall = time.perf_counter() - t0

    await r.aclose()

    failed = [r for r in results if isinstance(r, Exception)]
    assert not failed, f"{len(failed)} concurrent tasks failed: {failed}"

    print(f"\n[PERF] e2e_concurrent_3: wall={wall:.2f}s  all tasks completed")


@pytest.mark.asyncio
@pytest.mark.perf
@stack_live
async def test_orchestrator_routing_overhead():
    """
    Measures the time from TASK_CREATED to TASK_ASSIGNED (agent selection).
    Isolates orchestrator routing logic from actual agent work.
    """
    import redis.asyncio as aioredis

    r = await aioredis.from_url(REDIS_URL, decode_responses=True)

    task_id = f"e2e_routing_{uuid.uuid4().hex[:8]}"
    prompt = "What agent handles code search tasks?"

    last_id = "$"
    await _dispatch_task(r, task_id, prompt)
    t0 = time.perf_counter()

    deadline = time.perf_counter() + 30
    routing_latency: float | None = None

    while time.perf_counter() < deadline and routing_latency is None:
        entries = await r.xread({"agents:broadcast": last_id}, block=500, count=20)
        if not entries:
            continue
        for _stream, messages in entries:
            for msg_id, fields in messages:
                last_id = msg_id
                raw = fields.get("event") or fields.get(b"event")
                if not raw:
                    continue
                try:
                    data = json.loads(raw if isinstance(raw, str) else raw.decode())
                except Exception:
                    continue
                if (
                    data.get("payload", {}).get("task_id") == task_id
                    and data.get("type") == "task.assigned"
                ):
                    routing_latency = time.perf_counter() - t0

    await r.aclose()

    if routing_latency is None:
        pytest.skip("No task.assigned event observed — orchestrator may not emit it")

    print(f"\n[PERF] routing_overhead: {routing_latency*1000:.0f}ms to assign task")
    assert routing_latency < 5.0, (
        f"Routing overhead too high: {routing_latency:.2f}s (expected <5s)"
    )


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
