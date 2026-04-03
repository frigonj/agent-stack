"""
agents/research/main.py
────────────────────────
Research agent — multi-source internet research with staged evidence accumulation.

Loop per task:
  1. Decompose the question into N sub-queries (LLM)
  2. For each sub-query: search SearXNG → store raw snippets in ctx:research:{id}
  3. Extract single-sentence facts from each snippet (LLM)
  4. Cross-source consensus check: fact is reliable if ≥ MIN_SOURCES_FOR_CONSENSUS
     independent domains agree
  5. Gap / conflict detection → generate follow-up sub-queries
  6. Repeat up to max_search_iterations
  7. Synthesise final answer (LLM) → commit confident sources to Postgres
  8. Return result to orchestrator

Zero Claude API calls — all LLM work uses local LM Studio (Qwen via LangChain).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from typing import Optional
from urllib.parse import urlparse

import httpx
import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from core.base_agent import BaseAgent, run_agent
from core.config import Settings
from core.events.bus import Event, EventType

log = structlog.get_logger()

# ── Constants ─────────────────────────────────────────────────────────────────

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")
MAX_SEARCH_ITERATIONS = int(os.getenv("MAX_SEARCH_ITERATIONS", "3"))
MAX_RESULTS_PER_QUERY = int(os.getenv("MAX_RESULTS_PER_QUERY", "8"))
MIN_SOURCES_FOR_CONSENSUS = 2  # ≥2 independent domains → fact is reliable
CONFIDENCE_COMMIT_THRESHOLD = 0.75  # facts above this go to Postgres

SYSTEM_PROMPT = """You are a research specialist. You receive internet search results and
reason across multiple sources to produce accurate, well-sourced answers.

Your workflow:
1. Decompose the question into targeted sub-queries
2. Evaluate snippets for relevance, freshness, and source credibility
3. Identify agreements and contradictions across sources
4. Synthesise a clear, concise answer — cite sources inline as [domain.com]
5. Flag gaps where you could not find reliable information

Rules:
- Never fabricate facts; only state what sources confirm
- Prefer primary sources (official docs, papers) over secondary
- Shorter, sharper sub-queries return better search results than natural language questions
- If two sources contradict, note the conflict and give both viewpoints

## Other agents in this stack
- executor: shell commands, file I/O, Docker
- code_search: codebase search, grep patterns
- document_qa: PDF reading, LaTeX generation, architecture review
Route non-research tasks back via the orchestrator.
"""

# ── SearXNG helpers ───────────────────────────────────────────────────────────


async def _search(
    client: httpx.AsyncClient,
    query: str,
    num_results: int = MAX_RESULTS_PER_QUERY,
) -> list[dict]:
    """
    Hit SearXNG JSON API and return a list of result dicts.
    Each result has: url, title, content (snippet), engine.
    Returns [] on any error so the loop can continue gracefully.
    """
    try:
        resp = await client.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "engines": "google,bing,duckduckgo",
                "language": "en",
                "safesearch": "0",
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])[:num_results]
        log.info("research.search_done", query=query[:60], hits=len(results))
        return results
    except Exception as exc:
        log.warning("research.search_failed", query=query[:60], error=str(exc))
        return []


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lstrip("www.")
    except Exception:
        return url


# ── LLM helpers (all local — LM Studio / Qwen) ────────────────────────────────


def _decompose_prompt(question: str) -> str:
    return (
        "Break this research question into 2-4 short, targeted web search queries.\n"
        "Return ONLY a JSON array of strings, nothing else.\n\n"
        f"Question: {question}"
    )


def _extract_fact_prompt(query: str, snippet: str) -> str:
    return (
        "Extract ONE concise factual sentence (≤40 words) relevant to the query, or "
        "return null if the snippet is not relevant.\n"
        "Return ONLY a JSON string or JSON null, nothing else.\n\n"
        f"Query: {query}\n"
        f"Snippet: {snippet[:800]}"
    )


def _synthesise_prompt(question: str, facts: list[dict]) -> str:
    facts_text = "\n".join(
        f"- [{f['domain']}] {f['fact']}" for f in facts if f.get("fact")
    )
    return (
        "Synthesise a clear, accurate answer to the question using only the provided facts.\n"
        "Cite sources inline as [domain]. Flag any gaps.\n\n"
        f"Question: {question}\n\nFacts:\n{facts_text}"
    )


def _gap_prompt(question: str, facts: list[dict], iteration: int) -> str:
    covered = "; ".join(f["fact"][:80] for f in facts if f.get("fact"))[:600]
    return (
        "Given what we know so far and the original question, "
        "generate 1-2 follow-up search queries to fill remaining gaps.\n"
        "Return ONLY a JSON array of strings, or [] if the question is fully answered.\n\n"
        f"Question: {question}\n"
        f"Already found (iteration {iteration}): {covered}"
    )


def _parse_json_from_llm(text: str) -> object:
    """Extract the first JSON value (array or string or null) from LLM output."""
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.M)
    text = re.sub(r"\s*```$", "", text, flags=re.M)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON array or string
        m = re.search(r'(\[.*?\]|".*?")', text, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return None


# ── Research Agent ────────────────────────────────────────────────────────────


class ResearchAgent(BaseAgent):
    _OWN_TOOLS = [
        (
            "research-question",
            "Research a factual question across multiple internet sources and return a sourced answer",
            "event:task.assigned:research",
            ["research", "search", "web", "facts", "sources"],
        ),
        (
            "lookup-current-info",
            "Find current information about a topic (news, documentation, prices, status)",
            "event:task.assigned:research",
            ["research", "current", "lookup", "news"],
        ),
    ]

    async def handle_event(self, event: Event) -> None:
        if event.type not in (EventType.TASK_ASSIGNED, EventType.TASK_CREATED):
            return

        task = event.payload.get("task", "")
        task_id = event.task_id or str(uuid.uuid4())
        subtask_id = event.payload.get("subtask_id", str(uuid.uuid4()))
        parent_id = event.payload.get("parent_task_id", task_id)
        discord_mid = event.payload.get("discord_message_id")

        log.info("research.task_received", task=task[:80])

        try:
            result = await self._research(task, task_id)
        except Exception as exc:
            log.exception("research.task_failed", error=str(exc))
            result = f"Research failed: {exc}"

        await self.bus.publish(
            Event(
                type=EventType.TASK_COMPLETED,
                source=self.role,
                payload={
                    "result": result,
                    "task_id": task_id,
                    "subtask_id": subtask_id,
                    "parent_task_id": parent_id,
                    "discord_message_id": discord_mid,
                },
                task_id=task_id,
            ),
            target="orchestrator",
        )

    async def _research(self, question: str, task_id: str) -> str:
        """
        Full staged evidence loop.
        Returns the synthesised answer as a string.
        """
        query_id = task_id

        # Create a context stream for this research session
        await self.bus.create_context_stream(
            "research",
            query_id,
            question[:60],
            metadata={"status": "active", "question": question},
        )

        staging_key = f"research:staging:{query_id}"
        all_facts: list[dict] = []  # {fact, domain, url, confidence}

        async with httpx.AsyncClient() as http:
            # ── Step 1: initial sub-query decomposition ────────────────────────
            sub_queries = await self._decompose(question)
            log.info("research.sub_queries", queries=sub_queries)

            for iteration in range(1, MAX_SEARCH_ITERATIONS + 1):
                log.info(
                    "research.iteration_start",
                    iteration=iteration,
                    queries=len(sub_queries),
                )

                new_facts: list[dict] = []

                for sub_q in sub_queries:
                    results = await _search(http, sub_q)
                    for hit in results:
                        url = hit.get("url", "")
                        snippet = hit.get("content", hit.get("snippet", ""))
                        title = hit.get("title", "")
                        domain = _domain(url)

                        if not snippet or not url:
                            continue

                        # Store raw snippet in context stream
                        await self.bus.publish_to_context(
                            query_id,
                            Event(
                                type=EventType.TASK_CREATED,
                                source="research",
                                payload={
                                    "kind": "snippet",
                                    "sub_q": sub_q,
                                    "url": url,
                                    "title": title,
                                    "domain": domain,
                                    "snippet": snippet[:400],
                                },
                            ),
                        )

                        # Extract single fact from snippet
                        fact = await self._extract_fact(sub_q, snippet)
                        if not fact:
                            continue

                        new_facts.append(
                            {
                                "fact": fact,
                                "domain": domain,
                                "url": url,
                                "title": title,
                                "sub_q": sub_q,
                            }
                        )

                # Store new facts in Redis staging hash
                if new_facts:
                    pipe = self.bus._client.pipeline()
                    for i, f in enumerate(new_facts):
                        key = f"{len(all_facts) + i}"
                        pipe.hset(staging_key, key, json.dumps(f))
                    pipe.expire(staging_key, 3600)  # TTL 1 h
                    await pipe.execute()

                all_facts.extend(new_facts)
                log.info(
                    "research.facts_accumulated",
                    total=len(all_facts),
                    new=len(new_facts),
                    iteration=iteration,
                )

                # ── Consensus check ────────────────────────────────────────────
                confident = _compute_confidence(all_facts)
                if iteration < MAX_SEARCH_ITERATIONS:
                    follow_ups = await self._find_gaps(question, confident, iteration)
                    if not follow_ups:
                        log.info("research.gaps_closed", iteration=iteration)
                        break
                    sub_queries = follow_ups

        # ── Synthesise ────────────────────────────────────────────────────────
        confident_facts = _compute_confidence(all_facts)
        answer = await self._synthesise(question, confident_facts)

        # ── Commit to Postgres ────────────────────────────────────────────────
        await self._commit_sources(query_id, question, confident_facts)

        # Clean up staging hash
        await self.bus._client.delete(staging_key)

        return answer

    # ── LLM calls (local) ─────────────────────────────────────────────────────

    async def _decompose(self, question: str) -> list[str]:
        try:
            resp = await self.llm_invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=_decompose_prompt(question)),
                ]
            )
            parsed = _parse_json_from_llm(resp.content)
            if isinstance(parsed, list) and parsed:
                return [str(q) for q in parsed[:4]]
        except Exception as exc:
            log.warning("research.decompose_failed", error=str(exc))
        # Fallback: use the original question as-is
        return [question]

    async def _extract_fact(self, query: str, snippet: str) -> Optional[str]:
        try:
            # Use llm.ainvoke directly — no SystemMessage, content is tiny (< 300 tokens),
            # and this fires once per search snippet so lock contention would stall the pipeline.
            resp = await self.llm.ainvoke(
                [
                    HumanMessage(content=_extract_fact_prompt(query, snippet)),
                ]
            )
            parsed = _parse_json_from_llm(resp.content)
            if isinstance(parsed, str) and parsed.strip():
                return parsed.strip()
        except Exception as exc:
            log.warning("research.extract_fact_failed", error=str(exc))
        return None

    async def _find_gaps(
        self,
        question: str,
        facts: list[dict],
        iteration: int,
    ) -> list[str]:
        try:
            # Use llm.ainvoke directly — no SystemMessage and called once per iteration,
            # so not worth holding the distributed lock.
            resp = await self.llm.ainvoke(
                [
                    HumanMessage(content=_gap_prompt(question, facts, iteration)),
                ]
            )
            parsed = _parse_json_from_llm(resp.content)
            if isinstance(parsed, list):
                return [str(q) for q in parsed[:2]]
        except Exception as exc:
            log.warning("research.gap_failed", error=str(exc))
        return []

    async def _synthesise(self, question: str, facts: list[dict]) -> str:
        if not facts:
            return (
                "I was unable to find reliable information on this topic. "
                "The search returned results but none reached consensus across "
                "multiple independent sources. Please try a more specific query."
            )
        try:
            resp = await self.llm_invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=_synthesise_prompt(question, facts)),
                ]
            )
            return resp.content.strip()
        except Exception as exc:
            log.warning("research.synthesise_failed", error=str(exc))
            # Fallback: bullet list of facts
            lines = [f"- [{f['domain']}] {f['fact']}" for f in facts]
            return "Here is what I found:\n" + "\n".join(lines)

    # ── Postgres persistence ──────────────────────────────────────────────────

    async def _commit_sources(
        self,
        query_id: str,
        question: str,
        facts: list[dict],
    ) -> None:
        if not facts:
            return
        try:
            pool = await self.memory._get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    for f in facts:
                        await cur.execute(
                            """
                            INSERT INTO research_sources
                                (query_id, query_text, url, domain, title,
                                 snippet, fact, confidence, committed)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, TRUE)
                            ON CONFLICT DO NOTHING
                            """,
                            (
                                query_id,
                                question[:500],
                                f.get("url", ""),
                                f.get("domain", ""),
                                f.get("title", "")[:200],
                                f.get("snippet", "")[:500],
                                f.get("fact", "")[:300],
                                f.get("confidence", 0.0),
                            ),
                        )
                await conn.commit()
            log.info("research.sources_committed", count=len(facts), query_id=query_id)
        except Exception as exc:
            log.warning("research.commit_failed", error=str(exc))


# ── Consensus / confidence scoring ────────────────────────────────────────────


def _compute_confidence(facts: list[dict]) -> list[dict]:
    """
    Group facts by semantic similarity (simple: same sub_q bucket).
    A fact earns confidence proportional to how many independent domains confirm it.
    Returns only facts meeting CONFIDENCE_COMMIT_THRESHOLD, de-duplicated by domain+fact.
    """
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for f in facts:
        key = (f.get("domain", ""), f.get("fact", "")[:60])
        if key not in seen:
            seen.add(key)
            deduped.append(f)

    # For each fact, count distinct domains that share the same sub_q
    sub_q_domains: dict[str, set[str]] = {}
    for f in deduped:
        sq = f.get("sub_q", "")
        sub_q_domains.setdefault(sq, set()).add(f.get("domain", ""))

    result = []
    for f in deduped:
        sq = f.get("sub_q", "")
        n_agree = len(sub_q_domains.get(sq, set()))
        conf = min(1.0, n_agree / MIN_SOURCES_FOR_CONSENSUS)
        if conf >= CONFIDENCE_COMMIT_THRESHOLD:
            result.append({**f, "confidence": round(conf, 2)})

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    from core.config import Settings

    asyncio.run(run_agent(ResearchAgent(Settings())))
