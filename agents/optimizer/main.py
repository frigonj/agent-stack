"""
agents/optimizer/main.py
─────────────────────────
Performance optimizer agent — runs the perf test suite periodically,
analyses results against stored baselines, and proposes concrete
improvements to the agent stack.

Lifecycle:
  - On startup: run a full perf suite pass, store results in Postgres
  - Every think_interval (default 6 h): re-run, compare against prior run,
    generate improvement suggestions if regressions or new bottlenecks found
  - On OPTIMIZE_REQUEST event: run targeted tests + produce a suggestion report

The agent does NOT apply changes itself. It publishes suggestions as
AGENT_RESPONSE events and stores them in long-term memory so the
orchestrator or user can review and act on them. Destructive changes
(code edits, restarts) go through the executor's approval gate.

Suggestion categories:
  - redis_throughput   — event bus bottlenecks
  - llm_latency        — inference speed / model swap recommendation
  - memory_ops         — pgvector / embedding bottlenecks
  - e2e_latency        — end-to-end task pipeline
  - concurrency        — context pool sizing, agent parallelism
  - architecture       — structural changes (new caches, batching, etc.)
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from core.base_agent import BaseAgent, run_agent
from core.config import Settings
from core.events.bus import Event, EventType

log = structlog.get_logger()

# ── Constants ─────────────────────────────────────────────────────────────────

# How often to run the full perf suite proactively (seconds). Default 6 h.
_THINK_INTERVAL = int(os.getenv("OPTIMIZER_INTERVAL", str(6 * 3600)))

# How many historical perf snapshots to keep in long-term memory.
_MAX_HISTORY = 20

# Regression threshold above which the optimizer generates a suggestion (20 %).
_REGRESSION_WARN = 0.20

# Path to the perf suite inside the container.
_PERF_DIR = Path(os.getenv("PERF_DIR", "/workspace/tests/perf"))
_BASELINE_FILE = _PERF_DIR / "perf_baseline.json"

# Redis / Postgres / LM Studio URLs passed through from env.
_REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
_DB_URL = os.getenv("DATABASE_URL", "postgresql://agent:agent@postgres:5432/agentmem")
_LM_URL = os.getenv("LM_STUDIO_URL", "http://host.docker.internal:1234")

SYSTEM_PROMPT = """You are a performance optimisation specialist for a multi-agent \
AI system running on local hardware.

Your job is to:
1. Read perf test results (JSON dictionaries with timing metrics).
2. Compare them against prior baselines and identify regressions or bottlenecks.
3. Produce actionable, concrete improvement suggestions.

## Output format
Always produce a JSON object with this shape:
{
  "summary": "one-sentence overall assessment",
  "regressions": [
    {"metric": "...", "current": ..., "baseline": ..., "delta_pct": ...}
  ],
  "suggestions": [
    {
      "category": "redis_throughput|llm_latency|memory_ops|e2e_latency|concurrency|architecture",
      "priority": "high|medium|low",
      "title": "short title",
      "detail": "concrete what-to-change and why",
      "estimated_impact": "free-text estimate of improvement"
    }
  ]
}

## Suggestion rules
- redis_throughput: if publish < 500 evt/s or RTT p95 > 50 ms → suggest pipeline batching, connection pooling, or stream trimming.
- llm_latency: if TTFT > 3 s or tok/s < 10 → suggest model quantisation change, context length reduction, or GPU memory pressure mitigation.
- memory_ops: if semantic search > 200 ms → suggest sentence-transformers GPU offload (CUDA), index tuning, or HNSW parameter changes.
- e2e_latency: if median > 30 s → suggest agent pre-warming, context pool size increase, or LLM prompt shortening.
- concurrency: if 3-concurrent wall > 3× single → suggest queueing strategy, max_concurrent_contexts adjustment.
- architecture: structural suggestions (caching layers, batching, async improvements).

Always include at least one suggestion even if no regressions are detected — there is always room to improve.
Be specific: name the file, config key, or env var to change. Do not be vague.
Output only the JSON object — no prose before or after it.
"""


# ── Result runner ─────────────────────────────────────────────────────────────


def _run_perf_suite(marks: list[str] | None = None) -> dict[str, Any]:
    """
    Run the perf test suite via pytest in a subprocess.
    Returns a dict of {kpi_key: value} extracted from stdout JSON lines,
    plus meta fields: run_ts, passed, failed, errors.

    marks: list of pytest marks to filter (e.g. ["redis_live", "db_live"]).
           If None, runs all perf tests that don't require external services.
    """
    mark_expr = "perf"
    if marks:
        mark_expr = "perf and (" + " or ".join(marks) + ")"

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(_PERF_DIR),
        "-m",
        mark_expr,
        "--tb=short",
        "-q",
        "--no-header",
        f"--rootdir={_PERF_DIR.parents[1]}",
        # Pass service URLs so tests connect to container services, not localhost
        f"--perf-url-redis={_REDIS_URL}",
        f"--perf-url-db={_DB_URL}",
        f"--perf-url-lm={_LM_URL}",
    ]

    env = os.environ.copy()
    env.update(
        {
            "REDIS_URL": _REDIS_URL,
            "DATABASE_URL": _DB_URL,
            "LM_STUDIO_URL": _LM_URL,
            "PYTHONPATH": str(_PERF_DIR.parents[1]),
        }
    )

    log.info("optimizer.running_perf_suite", cmd=" ".join(cmd))
    t0 = time.monotonic()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            cwd=str(_PERF_DIR.parents[1]),
        )
    except subprocess.TimeoutExpired:
        return {
            "run_ts": datetime.now(timezone.utc).isoformat(),
            "passed": 0,
            "failed": 0,
            "errors": ["Perf suite timed out after 600s"],
            "duration_s": 600,
        }

    duration = time.monotonic() - t0

    # Parse pytest summary: "N passed, M failed"
    passed = failed = 0
    for line in result.stdout.splitlines():
        if "passed" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "passed," or p == "passed":
                    try:
                        passed = int(parts[i - 1])
                    except (ValueError, IndexError):
                        pass
                if p == "failed," or p == "failed":
                    try:
                        failed = int(parts[i - 1])
                    except (ValueError, IndexError):
                        pass

    # Load the baseline file written by the tests (they update it in-place)
    metrics: dict[str, Any] = {}
    if _BASELINE_FILE.exists():
        try:
            metrics = json.loads(_BASELINE_FILE.read_text())
        except Exception as exc:
            log.warning("optimizer.baseline_parse_error", error=str(exc))

    return {
        "run_ts": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "failed": failed,
        "duration_s": round(duration, 1),
        "errors": [result.stderr[-2000:]] if result.returncode not in (0, 1) else [],
        "metrics": metrics,
        "stdout_tail": result.stdout[-3000:],
    }


def _compare_runs(current: dict, prior: dict) -> list[dict]:
    """
    Compare current metrics dict against a prior run's metrics.
    Returns a list of regression dicts.
    """
    regressions = []
    cur_m = current.get("metrics", {})
    pri_m = prior.get("metrics", {})

    # Higher-is-better metrics (throughput, tok/s, speedup)
    higher_better = {
        "event_bus_publish_eps",
        "event_bus_consume_eps",
        "llm_agent_tps",
        "memory_batch_speedup",
    }
    # Lower-is-better metrics (latency)
    lower_better = {
        "llm_short_ttft_s",
        "memory_write_mean_ms",
        "memory_semantic_search_mean_ms",
        "e2e_sequential_median_s",
    }

    for key in set(cur_m) | set(pri_m):
        cur_val = cur_m.get(key)
        pri_val = pri_m.get(key)
        if cur_val is None or pri_val is None or not isinstance(cur_val, (int, float)):
            continue
        if pri_val == 0:
            continue

        delta = (cur_val - pri_val) / abs(pri_val)

        regressed = False
        if key in higher_better and delta < -_REGRESSION_WARN:
            regressed = True
        elif key in lower_better and delta > _REGRESSION_WARN:
            regressed = True

        if regressed:
            regressions.append(
                {
                    "metric": key,
                    "current": cur_val,
                    "baseline": pri_val,
                    "delta_pct": round(delta * 100, 1),
                }
            )

    return regressions


# ── Optimizer agent ────────────────────────────────────────────────────────────


class OptimizerAgent(BaseAgent):
    think_interval: int = _THINK_INTERVAL
    idle_timeout: int = 0  # always-on

    # ── Watchdog config ───────────────────────────────────────────────────────
    # How long a plan step can be "running" before we emit STALE_PLAN (seconds).
    _STALE_STEP_TIMEOUT_S: int = int(os.getenv("STALE_STEP_TIMEOUT", "300"))  # 5 min
    # How often the stale-plan and container-health watchdogs run (seconds).
    _WATCHDOG_INTERVAL_S: int = 60
    # LLM lock key and heartbeat key (mirrors base_agent constants).
    _LLM_LOCK_KEY = "llm:lock"
    _LLM_LOCK_HEARTBEAT_KEY = "llm:lock:heartbeat"
    _LLM_LOCK_HEARTBEAT_TTL = 15

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._prior_run: Optional[dict] = None
        self._last_run_ts: float = 0.0

    async def on_startup(self) -> None:
        log.info("optimizer.startup", interval_h=_THINK_INTERVAL // 3600)
        # Restore prior run from memory if available
        try:
            recalled = await self.memory.search(
                "optimizer perf suite run results metrics",
                semantic=False,
                limit=1,
            )
            if recalled:
                raw = recalled[0].get("content", "")
                # Content is stored as JSON string
                start = raw.find("{")
                if start != -1:
                    self._prior_run = json.loads(raw[start:])
                    log.info(
                        "optimizer.prior_run_restored", ts=self._prior_run.get("run_ts")
                    )
        except Exception as exc:
            log.warning("optimizer.restore_failed", error=str(exc))

        # Launch watchdog loops
        asyncio.create_task(self._stale_plan_watchdog_loop())
        asyncio.create_task(self._container_health_watchdog_loop())
        asyncio.create_task(self._llm_lock_watchdog_loop())

        # Run an initial perf pass shortly after startup (5 s delay so infra is ready)
        asyncio.get_event_loop().call_later(
            5, lambda: asyncio.create_task(self._run_and_analyse())
        )

    async def on_shutdown(self) -> None:
        log.info("optimizer.shutdown")

    async def think(self) -> None:
        """Periodic proactive perf run — called every think_interval seconds."""
        log.info("optimizer.think_cycle")
        await self._run_and_analyse()

    async def handle_event(self, event: Event) -> None:
        """
        Handles:
          OPTIMIZE_REQUEST  — run targeted tests + return suggestion report
          TASK_CREATED      — if payload.target_agent == "optimizer", handle as request
        """
        payload = event.payload or {}

        # Accept direct task routing
        if (
            event.type in (EventType.TASK_CREATED,)
            and payload.get("target_agent") == "optimizer"
        ):
            await self._handle_optimize_request(event, payload)
            return

        # Named event type for direct optimizer requests
        if event.type.value == "optimize.request":
            await self._handle_optimize_request(event, payload)

    # ── Watchdog loops ────────────────────────────────────────────────────────

    async def _stale_plan_watchdog_loop(self) -> None:
        """
        Every _WATCHDOG_INTERVAL_S seconds, query active_plans for any plan
        whose updated_at is older than _STALE_STEP_TIMEOUT_S and whose status
        is still 'running'.  Emit STALE_PLAN to the orchestrator for each one
        so it can reset and re-dispatch the stuck steps.
        """
        while self._running:
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=float(self._WATCHDOG_INTERVAL_S)
                )
                return
            except asyncio.TimeoutError:
                pass
            if not self._running:
                return
            try:
                pool = await self.memory._get_pool()
                async with pool.connection() as conn:
                    cur = await conn.execute(
                        """
                        SELECT task_id, plan_id, original_task, plan_json
                        FROM active_plans
                        WHERE status = 'running'
                          AND updated_at < NOW() - INTERVAL '%s seconds'
                        """,
                        (self._STALE_STEP_TIMEOUT_S,),
                    )
                    rows = await cur.fetchall()

                for row in rows:
                    task_id, plan_id, original_task, plan_json = row
                    if isinstance(plan_json, str):
                        plan_json = json.loads(plan_json)

                    # Find which step_ids are "running"
                    stale_steps = [
                        s["step_id"]
                        for s in plan_json.get("steps", [])
                        if s.get("status") == "running"
                    ]
                    if not stale_steps:
                        continue

                    log.warning(
                        "optimizer.stale_plan_detected",
                        task_id=task_id,
                        stale_steps=stale_steps,
                        task_preview=str(original_task)[:60],
                    )
                    await self.bus.publish(
                        Event(
                            type=EventType.STALE_PLAN,
                            source=self.role,
                            payload={
                                "task_id": task_id,
                                "plan_id": plan_id,
                                "stale_step_ids": stale_steps,
                            },
                        ),
                        target="orchestrator",
                    )
            except Exception as exc:
                log.warning("optimizer.stale_watchdog_error", error=str(exc))

    async def _container_health_watchdog_loop(self) -> None:
        """
        Every _WATCHDOG_INTERVAL_S seconds, check agent container health via
        docker inspect.  Any container marked 'unhealthy' that has a restart
        policy of 'always' or 'unless-stopped' is restarted automatically.
        Ephemeral containers (restart: no) are reported but not force-restarted
        — the orchestrator manages their lifecycle.
        """
        # Containers managed by the orchestrator (ephemeral — don't auto-restart)
        _EPHEMERAL = {
            "agent_executor",
            "agent_developer",
            "agent_code_search",
            "agent_document_qa",
            "agent_research",
            "agent_claude_code",
        }
        # Always-on containers we should restart if unhealthy
        _ALWAYS_ON = {"agent_orchestrator", "agent_discord_bridge", "agent_optimizer"}

        while self._running:
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=float(self._WATCHDOG_INTERVAL_S)
                )
                return
            except asyncio.TimeoutError:
                pass
            if not self._running:
                return
            try:
                proc = await asyncio.create_subprocess_exec(
                    "docker",
                    "ps",
                    "-a",
                    "--filter",
                    "name=agent_",
                    "--format",
                    "{{.Names}}\t{{.Status}}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
                lines = stdout.decode().strip().splitlines()

                for line in lines:
                    parts = line.split("\t", 1)
                    if len(parts) != 2:
                        continue
                    name, status = parts[0].strip(), parts[1].strip()
                    if "(unhealthy)" not in status:
                        continue
                    if name in _ALWAYS_ON:
                        log.warning(
                            "optimizer.unhealthy_container_restart",
                            container=name,
                            status=status,
                        )
                        await asyncio.create_subprocess_exec(
                            "docker",
                            "restart",
                            name,
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL,
                        )
                    elif name in _EPHEMERAL:
                        log.info(
                            "optimizer.unhealthy_ephemeral_skipped",
                            container=name,
                            status=status,
                        )
            except Exception as exc:
                log.warning("optimizer.container_health_error", error=str(exc))

    async def _llm_lock_watchdog_loop(self) -> None:
        """
        Central LLM lock stale-lock watchdog.  Runs in the optimizer so a
        single process owns this responsibility rather than every agent.
        Clears the lock if the heartbeat has expired (holder crashed).
        """
        check_interval = float(self._LLM_LOCK_HEARTBEAT_TTL)
        while self._running:
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=check_interval)
                return
            except asyncio.TimeoutError:
                pass
            if not self._running:
                return
            try:
                lock_val = await self.bus._client.get(self._LLM_LOCK_KEY)
                if lock_val is None:
                    continue
                heartbeat = await self.bus._client.get(self._LLM_LOCK_HEARTBEAT_KEY)
                if heartbeat is None:
                    log.warning(
                        "optimizer.llm_lock_stale",
                        holder=lock_val,
                        action="force_expiring",
                    )
                    await self.bus._client.delete(self._LLM_LOCK_KEY)
                    await self.bus._client.publish(
                        f"{self._LLM_LOCK_KEY}:released", "released"
                    )
            except Exception as exc:
                log.debug("optimizer.llm_lock_watchdog_error", error=str(exc))

    async def _handle_optimize_request(self, event: Event, payload: dict) -> None:
        task_id = payload.get("task_id", str(uuid.uuid4()))
        focus = payload.get("focus")  # optional: "redis"|"llm"|"memory"|"e2e"

        log.info("optimizer.request_received", task_id=task_id, focus=focus)

        marks: list[str] | None = None
        if focus == "redis":
            marks = ["redis_live"]
        elif focus == "llm":
            marks = ["lm_live"]
        elif focus == "memory":
            marks = ["db_live"]
        elif focus == "e2e":
            marks = ["stack_live"]

        run_result = await asyncio.get_event_loop().run_in_executor(
            None, _run_perf_suite, marks
        )

        suggestions = await self._analyse(run_result)

        await self.bus.publish(
            Event(
                type=EventType.AGENT_RESPONSE,
                source=self.role,
                payload={
                    "task_id": task_id,
                    "result": suggestions,
                    "run_summary": {
                        "passed": run_result.get("passed"),
                        "failed": run_result.get("failed"),
                        "duration_s": run_result.get("duration_s"),
                        "run_ts": run_result.get("run_ts"),
                    },
                },
            ),
            target="broadcast",
        )

    async def _run_and_analyse(self) -> None:
        """Run full perf suite, analyse, store results, publish suggestions."""
        log.info("optimizer.full_run_start")

        run_result = await asyncio.get_event_loop().run_in_executor(
            None, _run_perf_suite, None
        )
        log.info(
            "optimizer.full_run_complete",
            passed=run_result.get("passed"),
            failed=run_result.get("failed"),
            duration_s=run_result.get("duration_s"),
        )

        suggestions = await self._analyse(run_result)

        # Only promote to long-term memory when suggestions were actually produced —
        # an empty run has no recall value.
        if suggestions:
            await self.promote_now(
                content=f"Optimizer perf suite run {run_result['run_ts']}: "
                f"passed={run_result.get('passed')} failed={run_result.get('failed')} "
                f"suggestions={json.dumps(suggestions)}",
                topic="optimizer_results",
                tags=["optimizer", "perf", "suggestions"],
            )

        self._prior_run = run_result
        self._last_run_ts = time.monotonic()

        # Broadcast suggestion report to the orchestrator / Discord
        await self.bus.publish(
            Event(
                type=EventType.AGENT_RESPONSE,
                source=self.role,
                payload={
                    "task_id": f"optimizer_think_{uuid.uuid4().hex[:8]}",
                    "result": suggestions,
                    "run_summary": {
                        "passed": run_result.get("passed"),
                        "failed": run_result.get("failed"),
                        "duration_s": run_result.get("duration_s"),
                        "run_ts": run_result.get("run_ts"),
                    },
                },
            ),
            target="broadcast",
        )

        log.info(
            "optimizer.suggestions_published",
            n_suggestions=len(suggestions.get("suggestions", [])),
        )

    async def _analyse(self, run_result: dict) -> dict:
        """
        Use the LLM to produce structured improvement suggestions from the run result.
        Falls back to a rule-based analysis if the LLM is unavailable.
        """
        regressions = _compare_runs(run_result, self._prior_run or {})
        metrics = run_result.get("metrics", {})

        analysis_input = {
            "run_ts": run_result.get("run_ts"),
            "passed": run_result.get("passed"),
            "failed": run_result.get("failed"),
            "duration_s": run_result.get("duration_s"),
            "metrics": metrics,
            "regressions": regressions,
            "prior_metrics": (self._prior_run or {}).get("metrics", {}),
            "errors": run_result.get("errors", []),
        }

        prompt = (
            "Here are the perf test results from the agent stack. "
            "Analyse them and produce improvement suggestions.\n\n"
            f"```json\n{json.dumps(analysis_input, indent=2)}\n```"
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm_invoke(messages)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Extract JSON from response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
            else:
                log.warning("optimizer.llm_no_json", content_preview=content[:200])
                return _fallback_analysis(metrics, regressions)

        except Exception as exc:
            log.warning("optimizer.llm_error", error=str(exc))
            return _fallback_analysis(metrics, regressions)


# ── Rule-based fallback analysis ─────────────────────────────────────────────


def _fallback_analysis(metrics: dict, regressions: list[dict]) -> dict:
    """
    Simple rule-based suggestions when the LLM is unavailable.
    Covers the most common bottlenecks seen in this stack.
    """
    suggestions = []

    pub_eps = metrics.get("event_bus_publish_eps")
    if pub_eps is not None and pub_eps < 500:
        suggestions.append(
            {
                "category": "redis_throughput",
                "priority": "high",
                "title": "Low Redis publish throughput",
                "detail": (
                    f"Measured {pub_eps:.0f} evt/s (target ≥500). "
                    "Try using pipeline batching in EventBus.publish(): accumulate events "
                    "and flush with a single XADD pipeline every 10 ms. "
                    "Also check if maxmemory-policy allkeys-lru is causing eviction pressure — "
                    "increase Redis maxmemory in docker-compose.yml from 256mb to 512mb."
                ),
                "estimated_impact": "2–5× publish throughput",
            }
        )

    ttft = metrics.get("llm_short_ttft_s")
    if ttft is not None and ttft > 3.0:
        suggestions.append(
            {
                "category": "llm_latency",
                "priority": "high",
                "title": "High TTFT — model may be swapped out of GPU",
                "detail": (
                    f"Time-to-first-token is {ttft:.1f}s (target ≤3s). "
                    "Ensure no other processes are consuming VRAM (check with nvidia-smi). "
                    "If using a large model, switch to Qwen3 VL 8B Q8_0 which fits across RTX 3070 + P4000 (16 GB split) "
                    "and typically halves TTFT. "
                    "Set LM_STUDIO_MODEL=qwen3-vl-8b in .env and restart the stack."
                ),
                "estimated_impact": "40–60% TTFT reduction",
            }
        )

    tps = metrics.get("llm_agent_tps")
    if tps is not None and tps < 10:
        suggestions.append(
            {
                "category": "llm_latency",
                "priority": "medium",
                "title": "Low token throughput — consider smaller model",
                "detail": (
                    f"Generating {tps:.1f} tok/s (target ≥10). "
                    "For orchestration tasks (routing, planning) Qwen3 VL 8B Q8_0 is the recommended model. "
                    "Ensure it is split across RTX 3070 (primary) and P4000 (overflow) in LM Studio. "
                    "If throughput is still low, try Q6_K quantisation."
                ),
                "estimated_impact": "2× throughput for routine agent tasks",
            }
        )

    sem_ms = metrics.get("memory_semantic_search_mean_ms")
    if sem_ms is not None and sem_ms > 200:
        suggestions.append(
            {
                "category": "memory_ops",
                "priority": "medium",
                "title": "Slow semantic search — embedding on CPU",
                "detail": (
                    f"Semantic search mean {sem_ms:.0f}ms (target ≤200ms). "
                    "sentence-transformers (all-MiniLM-L6-v2) is likely running on CPU. "
                    "Set SENTENCE_TRANSFORMERS_USE_CUDA=1 in the Postgres container environment "
                    "and ensure the model_cache volume is mounted. "
                    "Alternatively, pre-embed at write time and cache in a Redis hash "
                    "keyed by content hash to avoid re-embedding repeated queries."
                ),
                "estimated_impact": "5–10× search speedup with GPU offload",
            }
        )

    write_ms = metrics.get("memory_write_mean_ms")
    if write_ms is not None and write_ms > 500:
        suggestions.append(
            {
                "category": "memory_ops",
                "priority": "medium",
                "title": "Slow knowledge writes — consider async queue",
                "detail": (
                    f"Mean write latency {write_ms:.0f}ms (target ≤500ms). "
                    "Writes block agent task handling. Introduce an asyncio.Queue in "
                    "LongTermMemory to buffer stores and flush in a background coroutine. "
                    "This decouples agent task latency from Postgres I/O. "
                    "File: core/memory/long_term.py — add a _write_queue and _flush_task."
                ),
                "estimated_impact": "Near-zero write latency from agent perspective",
            }
        )

    e2e_s = metrics.get("e2e_sequential_median_s")
    if e2e_s is not None and e2e_s > 30:
        suggestions.append(
            {
                "category": "e2e_latency",
                "priority": "high",
                "title": "High end-to-end latency — agent cold-start overhead",
                "detail": (
                    f"Median task latency {e2e_s:.0f}s (target ≤30s). "
                    "Check if ephemeral agents (executor, code_search) are being spun up fresh "
                    "for each task — Docker cold-start adds 5–15s. "
                    "Pre-warm them: set restart: unless-stopped for executor and code_search "
                    "in docker-compose.yml, and increase IDLE_TIMEOUT to 1800s."
                ),
                "estimated_impact": "10–20s reduction in first-task latency",
            }
        )

    batch_sp = metrics.get("memory_batch_speedup")
    if batch_sp is not None and batch_sp < 1.5:
        suggestions.append(
            {
                "category": "architecture",
                "priority": "low",
                "title": "batch_store not faster than individual stores",
                "detail": (
                    f"batch_store speedup is only {batch_sp:.1f}× (expected ≥1.5×). "
                    "This suggests embedding generation is not the bottleneck — "
                    "Postgres INSERT latency is. Enable connection pooling: "
                    "increase min_size in AsyncConnectionPool in core/memory/long_term.py "
                    "from 1 to 3 to reduce connection setup overhead."
                ),
                "estimated_impact": "20–30% write throughput improvement",
            }
        )

    if regressions:
        for reg in regressions:
            suggestions.append(
                {
                    "category": "architecture",
                    "priority": "high",
                    "title": f"Regression in {reg['metric']}",
                    "detail": (
                        f"Metric '{reg['metric']}' changed {reg['delta_pct']:+.1f}% "
                        f"(current={reg['current']}, prior={reg['baseline']}). "
                        "Investigate recent code changes with: "
                        "CMD: git -C /workspace/src log --oneline -20"
                    ),
                    "estimated_impact": "Restore to baseline",
                }
            )

    if not suggestions:
        suggestions.append(
            {
                "category": "architecture",
                "priority": "low",
                "title": "All metrics within baseline — consider HNSW tuning",
                "detail": (
                    "No regressions detected. For future gains: tune the pgvector HNSW index "
                    "parameters (ef_construction, m) in core/memory/long_term.py. "
                    "Current default (m=16, ef_construction=64) is conservative. "
                    "Increasing to m=32, ef_construction=128 improves recall at ~20% higher "
                    "build time — worthwhile once the knowledge table exceeds 10k entries."
                ),
                "estimated_impact": "Better semantic recall at scale",
            }
        )

    regression_summary = (
        f"{len(regressions)} regression(s) detected"
        if regressions
        else "no regressions"
    )
    return {
        "summary": (
            f"Perf suite: {metrics.get('passed', '?')} passed, "
            f"{metrics.get('failed', '?')} failed. {regression_summary}."
        ),
        "regressions": regressions,
        "suggestions": suggestions,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    settings = Settings()
    agent = OptimizerAgent(settings)
    run_agent(agent)
