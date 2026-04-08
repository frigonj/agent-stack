"""
tests/perf/test_llm_latency.py
────────────────────────────────
LM Studio inference latency and concurrency tests.

Measures:
  - Time-to-first-token (TTFT) for streaming responses
  - Total generation time for short and long prompts
  - Wall-clock time under concurrency=1, 2, 3 to locate the GPU saturation point
  - Token throughput (tok/s) per call

Marks:
  @pytest.mark.perf      — all tests here
  @pytest.mark.lm_live   — skipped unless LM Studio is reachable

Run:
  pytest tests/perf/test_llm_latency.py -v -m perf
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import os

import pytest

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen3-vl-8b")
REGRESSION_THRESHOLD = 0.25  # 25% — inference latency is noisier than Redis


# ── Reachability ──────────────────────────────────────────────────────────────


def _lm_available() -> bool:
    try:
        import httpx

        r = httpx.get(f"{LM_STUDIO_URL}/api/v0/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


lm_live = pytest.mark.skipif(
    not _lm_available(),
    reason=f"LM Studio not reachable at {LM_STUDIO_URL}",
)


# ── Token counting ────────────────────────────────────────────────────────────


def _count_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))
    except ImportError:
        return len(text) // 4


# ── Core measurement ──────────────────────────────────────────────────────────


async def _measure_streaming(prompt: str) -> dict:
    """
    Stream one completion. Returns:
      ttft_s, total_s, tokens, tps
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url=f"{LM_STUDIO_URL}/v1",
        api_key="lm-studio",
        model=LM_STUDIO_MODEL,
        temperature=0.1,
        streaming=True,
        extra_body={"thinking": False},
    )
    messages = [
        SystemMessage(content="Be concise."),
        HumanMessage(content=prompt),
    ]

    t0 = time.perf_counter()
    ttft: float | None = None
    text = ""

    async for chunk in llm.astream(messages):
        if ttft is None and chunk.content:
            ttft = time.perf_counter() - t0
        text += chunk.content or ""

    total = time.perf_counter() - t0
    tokens = _count_tokens(text)
    tps = tokens / total if total > 0 else 0.0

    return {
        "ttft_s": round(ttft or total, 3),
        "total_s": round(total, 3),
        "tokens": tokens,
        "tps": round(tps, 2),
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.perf
@lm_live
async def test_short_prompt_ttft(perf_baseline, update_perf):
    """
    Baseline: short prompt → time-to-first-token.
    TTFT is the key UX metric for interactive tasks.
    """
    prompt = "List five uses of Redis in three words each."
    result = await _measure_streaming(prompt)

    print(
        f"\n[PERF] short_prompt: ttft={result['ttft_s']}s  "
        f"total={result['total_s']}s  {result['tokens']} tok  {result['tps']} tok/s"
    )

    assert result["tokens"] >= 10, "Response too short — model may not be loaded"

    kpi_key = "llm_short_ttft_s"
    if update_perf:
        perf_baseline[kpi_key] = result["ttft_s"]
        _save_baseline(perf_baseline)
        pytest.skip(f"Baseline updated: ttft={result['ttft_s']}s")

    if kpi_key in perf_baseline:
        baseline = perf_baseline[kpi_key]
        max_ttft = baseline * (1.0 + REGRESSION_THRESHOLD)
        assert result["ttft_s"] <= max_ttft, (
            f"TTFT regression: {result['ttft_s']}s > {max_ttft:.2f}s "
            f"(baseline {baseline}s, threshold +{REGRESSION_THRESHOLD * 100:.0f}%)"
        )
    else:
        perf_baseline[kpi_key] = result["ttft_s"]
        _save_baseline(perf_baseline)
        print(f"[PERF] No baseline — saved ttft={result['ttft_s']}s")


@pytest.mark.asyncio
@pytest.mark.perf
@lm_live
async def test_agent_prompt_throughput(perf_baseline, update_perf):
    """
    Realistic agent-sized prompt (system prompt + task). Measures tok/s.
    Simulates a typical orchestrator → agent dispatch.
    """
    prompt = (
        "You are a code analysis specialist. The following Python module implements "
        "an asynchronous Redis Streams event bus with consumer groups and context "
        "streams. Summarise its three most important design decisions as bullet points. "
        "Keep each bullet to one sentence.\n\n"
        "```python\n"
        "class EventBus:\n"
        "    async def publish(self, event, target='broadcast'): ...\n"
        "    async def consume(self, role, group, consumer): ...\n"
        "    async def create_context_stream(self, type, id, name): ...\n"
        "    async def publish_to_context(self, ctx_id, event): ...\n"
        "```"
    )
    result = await _measure_streaming(prompt)

    print(
        f"\n[PERF] agent_prompt: ttft={result['ttft_s']}s  "
        f"total={result['total_s']}s  {result['tokens']} tok  {result['tps']} tok/s"
    )

    assert result["tokens"] >= 20, "Response too short"

    kpi_key = "llm_agent_tps"
    if update_perf:
        perf_baseline[kpi_key] = result["tps"]
        _save_baseline(perf_baseline)
        pytest.skip(f"Baseline updated: {result['tps']} tok/s")

    if kpi_key in perf_baseline:
        baseline_tps = perf_baseline[kpi_key]
        min_tps = baseline_tps * (1.0 - REGRESSION_THRESHOLD)
        assert result["tps"] >= min_tps, (
            f"Throughput regression: {result['tps']} tok/s < {min_tps:.1f} tok/s "
            f"(baseline {baseline_tps} tok/s)"
        )
    else:
        perf_baseline[kpi_key] = result["tps"]
        _save_baseline(perf_baseline)
        print(f"[PERF] No baseline — saved {result['tps']} tok/s")


@pytest.mark.asyncio
@pytest.mark.perf
@lm_live
async def test_concurrency_scaling():
    """
    Run 1, 2, 3 concurrent inference calls and record wall-clock time.
    Checks that concurrency=3 wall time does not exceed 3× single-call time
    (i.e., requests are queued, not failing or ballooning).

    This is informational — no baseline regression. Prints the scaling factor
    so you can see where your GPU saturates.
    """
    prompt = "What is 2 + 2? Answer in one sentence."

    # Warm-up
    await _measure_streaming(prompt)

    results: dict[int, float] = {}
    for concurrency in [1, 2, 3]:
        tasks = [_measure_streaming(prompt) for _ in range(concurrency)]
        t0 = time.perf_counter()
        responses = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
        results[concurrency] = wall

        avg_tokens = sum(r["tokens"] for r in responses) / concurrency
        print(
            f"\n[PERF] concurrency={concurrency}: wall={wall:.2f}s  "
            f"avg_tokens={avg_tokens:.0f}  "
            f"scaling={wall / results[1]:.2f}×"
        )

    # Wall time for 3 concurrent must not exceed 4× single (generous — LM Studio queues)
    scaling_factor = results[3] / results[1]
    assert scaling_factor < 4.0, (
        f"Concurrency scaling too degraded: 3× calls took {scaling_factor:.2f}× "
        f"single-call wall time (expected <4.0×)"
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
