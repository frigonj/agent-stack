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

import html.parser
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
from core.context import truncate_task
from core.events.bus import Event, EventType

log = structlog.get_logger()

# ── Constants ─────────────────────────────────────────────────────────────────

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")
MAX_SEARCH_ITERATIONS = int(os.getenv("MAX_SEARCH_ITERATIONS", "3"))
MAX_RESULTS_PER_QUERY = int(os.getenv("MAX_RESULTS_PER_QUERY", "8"))
MIN_SOURCES_FOR_CONSENSUS = 3  # ≥3 independent domains → fact is reliable
CONFIDENCE_COMMIT_THRESHOLD = 0.75  # facts above this go to Postgres

# Engine sets tried in order. If the first set returns 0 results (bot-blocked),
# fall through to the next set so we always get something back.
_SEARCH_ENGINE_SETS = [
    "bing,duckduckgo,brave",  # primary: Google disabled (HTML parser broken)
    "wikipedia,mojeek,qwant",  # fallback: no Brave overlap, bot-detection-resistant
]

SYSTEM_PROMPT = """You are a research specialist in a multi-agent AI stack. You receive \
internet search results and reason across multiple sources to produce accurate, well-sourced \
answers that are immediately actionable by the other agents.

## Workflow
1. Decompose the question into 2–4 targeted sub-queries (short keyword phrases, not questions).
2. Evaluate each snippet for relevance, freshness, and source credibility.
3. Identify agreements and contradictions across sources.
4. Synthesise a concise, direct answer — cite inline as [domain.com].
5. Where the answer directly informs an engineering decision (version numbers, API changes,
   known bugs, correct approaches), make that conclusion explicit and prominent.
6. Flag gaps only when they are significant; don't pad with caveats.

## Quality rules
- Never fabricate. Only state what sources confirm.
- Prefer primary sources: official docs, GitHub releases, RFCs, papers.
- Shorter sub-queries return sharper results: "redis stream consumer group lag" not
  "how do I check the lag of a redis stream consumer group in python asyncio".
- If sources contradict, give both views with their source dates.
- Lead with the most actionable fact. Put caveats last.

## Output format for engineering questions
When the question is about a library, tool, or version:
  ANSWER: <direct one-sentence answer>
  VERSION/DETAILS: <specifics — version number, API name, config key, etc.>
  SOURCE: [domain.com] [other.com]
  CAVEATS: <only if genuinely important>

## Prior knowledge injection
If "Prior knowledge" is injected above this prompt: cross-check it against search results.
If search results contradict prior memory, trust the search results (they are more current)
and note the discrepancy so the orchestrator can update memory.

## Other agents in this stack
- executor: shell commands, file I/O, Docker operations
- code_search: codebase search, grep patterns, understanding local code
- document_qa: PDF reading, LaTeX generation, architecture review
Route non-research tasks back via the orchestrator. Do not attempt to run commands.
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
    # Primary engines: broad coverage.  If SearXNG blocks or returns 0 hits,
    # retry with a different engine set (Wikipedia + Brave + Mojeek) as fallback.
    for engines in _SEARCH_ENGINE_SETS:
        try:
            resp = await client.get(
                f"{SEARXNG_URL}/search",
                params={
                    "q": query,
                    "format": "json",
                    "engines": engines,
                    "language": "en",
                    "safesearch": "0",
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])[:num_results]
            if results:
                log.info(
                    "research.search_done",
                    query=query[:60],
                    hits=len(results),
                    engines=engines,
                )
                return results
            log.debug("research.search_empty", query=query[:60], engines=engines)
        except Exception as exc:
            log.warning(
                "research.search_failed",
                query=query[:60],
                engines=engines,
                error=str(exc),
            )
    return []


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lstrip("www.")
    except Exception:
        return url


# ── Direct URL fetch ──────────────────────────────────────────────────────────

_FETCH_MAX_CHARS = 12_000  # ~3k tokens — enough for a doc page without blowing context
_FETCH_TIMEOUT = 20.0


class _TagStripper(html.parser.HTMLParser):
    """Minimal HTML → plain-text converter using stdlib only."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = False
        if tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "br", "tr"):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse runs of whitespace / blank lines
        lines = [ln.strip() for ln in raw.splitlines()]
        lines = [ln for ln in lines if ln]
        return "\n".join(lines)


async def _fetch_url(url: str) -> str:
    """
    Fetch a URL and return readable plain text, truncated to _FETCH_MAX_CHARS.
    Returns an error string (never raises) so callers can continue gracefully.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; agent-stack-research/1.0; "
            "+https://github.com/agent-stack)"
        ),
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
    }
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=_FETCH_TIMEOUT
        ) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "html" in content_type:
                stripper = _TagStripper()
                stripper.feed(resp.text)
                text = stripper.get_text()
            else:
                text = resp.text
            return text[:_FETCH_MAX_CHARS]
    except Exception as exc:
        return f"[fetch error: {exc}]"


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
        "return null if the snippet is not relevant, promotional, SEO filler, or lacks a verifiable claim.\n"
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
    # Strip Qwen3 <think>...</think> reasoning blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.M)
    text = re.sub(r"\s*```$", "", text, flags=re.M)
    text = text.strip()
    try:
        parsed = json.loads(text)
        # Qwen3 wraps string answers in {"text": "..."} — unwrap it
        if isinstance(parsed, dict):
            for key in ("text", "sentence", "fact", "answer", "result"):
                if key in parsed and isinstance(parsed[key], str):
                    return parsed[key]
            return None
        return parsed
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


def _cluster_prompt(topic: str, facts: list[dict]) -> str:
    facts_text = "\n".join(
        f"{i + 1}. {f['fact']} [source: {f['domain']}]" for i, f in enumerate(facts)
    )
    return (
        "You are organising research facts about a topic into a structured knowledge database.\n"
        "Group the facts below into 3-6 sub-topics. For each sub-topic:\n"
        "  - Give it a short, specific name (3-6 words)\n"
        "  - List the fact numbers that belong to it\n"
        "Return ONLY a JSON array of objects with keys 'subtopic' and 'indices' (array of 1-based ints).\n\n"
        f"Topic: {topic}\n\nFacts:\n{facts_text}"
    )


def _knowledge_summary_prompt(subtopic: str, facts: list[dict]) -> str:
    facts_text = "\n".join(f"- [{f['domain']}] {f['fact']}" for f in facts)
    return (
        "Synthesise the following facts about a subtopic into a concise knowledge entry (2-5 sentences).\n"
        "Cite domains inline as [domain]. Be factual and precise.\n\n"
        f"Subtopic: {subtopic}\n\nFacts:\n{facts_text}"
    )


_BUILD_KNOWLEDGE_PHRASES = (
    "build knowledge",
    "knowledge database",
    "knowledge base",
    "knowledge db",
    "research and store",
    "deep research",
    "compile information",
    "build a database",
)

_FETCH_LEARN_PHRASES = (
    "fetch and learn",
    "fetch and store",
    "read url",
    "ingest url",
    "learn from url",
    "learn from link",
    "learn from source",
)


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
        (
            "build-knowledge-base",
            "Build a structured knowledge database on a topic by running deep multi-angle research and storing organised facts by subtopic",
            "event:task.assigned:research",
            ["knowledge", "database", "deep-research", "compile", "store", "topic"],
        ),
        (
            "fetch-and-learn",
            "Fetch a URL directly, extract its content, and store the key facts to the knowledge base",
            "event:task.assigned:research",
            ["fetch", "url", "ingest", "learn", "source", "link"],
        ),
    ]

    async def on_plan_proposed(
        self, plan_id: str, steps: list[dict], payload: dict, request_clarification=None
    ) -> tuple[bool | None, str, float]:
        """
        Use the LLM to evaluate research steps. Ask for clarification if the
        topic or scope is ambiguous before voting.
        """
        my_steps = [s for s in steps if s.get("agent") == "research"]
        if not my_steps:
            return None, "", 0.0

        steps_txt = "\n".join(
            f"  Phase {s.get('phase', 1)}: {s.get('task', '')}" for s in my_steps
        )
        original_task = payload.get("original_task", "")

        try:
            response = await self.llm_invoke(
                [
                    SystemMessage(
                        content=(
                            "You are the research agent. You search the internet and synthesise information. "
                            "You are reviewing steps assigned to you in a proposed execution plan. "
                            "Decide if the research topic is clear enough for you to execute effectively.\n\n"
                            "Reply in exactly this format:\n"
                            "UNDERSTOOD: yes/no\n"
                            "CLARIFICATION_NEEDED: <one focused question if UNDERSTOOD=no, else 'none'>\n"
                            "APPROVE: yes/no\n"
                            "REASON: <one sentence>"
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Overall task: {original_task}\nMy steps:\n{steps_txt}"
                        )
                    ),
                ]
            )
            lines = {
                k.strip(): v.strip()
                for line in response.content.splitlines()
                if ":" in line
                for k, v in [line.split(":", 1)]
            }
        except Exception as exc:
            log.warning("research.vote_llm_error", error=str(exc))
            return True, "could not evaluate — defaulting to approve", 0.5

        understood = lines.get("UNDERSTOOD", "yes").lower() == "yes"
        clarification_q = lines.get("CLARIFICATION_NEEDED", "none")
        approve_str = lines.get("APPROVE", "yes").lower()
        reason = lines.get("REASON", "")

        if (
            not understood
            and clarification_q
            and clarification_q.lower() != "none"
            and request_clarification
        ):
            answer = await request_clarification(clarification_q)
            if answer:
                try:
                    response2 = await self.llm_invoke(
                        [
                            SystemMessage(
                                content=(
                                    "You are the research agent re-evaluating a plan step after receiving clarification. "
                                    "Reply in exactly this format:\n"
                                    "APPROVE: yes/no\n"
                                    "REASON: <one sentence>"
                                )
                            ),
                            HumanMessage(
                                content=(
                                    f"Overall task: {original_task}\n"
                                    f"My steps:\n{steps_txt}\n\n"
                                    f"Clarification received: {answer}"
                                )
                            ),
                        ]
                    )
                    lines2 = {
                        k.strip(): v.strip()
                        for line in response2.content.splitlines()
                        if ":" in line
                        for k, v in [line.split(":", 1)]
                    }
                    approve_str = lines2.get("APPROVE", approve_str).lower()
                    reason = lines2.get("REASON", reason)
                except Exception:
                    pass

        approve = approve_str == "yes"
        confidence = 0.85 if approve else 0.8
        return approve, reason[:200], confidence

    async def handle_event(self, event: Event) -> None:
        if event.type not in (EventType.TASK_ASSIGNED, EventType.TASK_CREATED):
            return

        task = truncate_task(event.payload.get("task", ""))
        task_id = event.task_id or str(uuid.uuid4())
        subtask_id = event.payload.get("subtask_id", str(uuid.uuid4()))
        parent_id = event.payload.get("parent_task_id", task_id)
        discord_mid = event.payload.get("discord_message_id")

        log.info("research.task_received", task=task[:80])

        is_knowledge_build = any(p in task.lower() for p in _BUILD_KNOWLEDGE_PHRASES)
        is_fetch_learn = any(p in task.lower() for p in _FETCH_LEARN_PHRASES)

        # fetch-and-learn tasks carry the URL in the payload; fall back to
        # extracting a URL-shaped token from the task string itself.
        source_url = event.payload.get("source_url", "")
        if not source_url and is_fetch_learn:
            m = re.search(r"https?://\S+", task)
            source_url = m.group(0) if m else ""

        try:
            if is_fetch_learn and source_url:
                result = await self._fetch_and_learn(source_url, task_id)
            elif is_knowledge_build:
                result = await self._build_knowledge_db(task, task_id)
            else:
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
        Full staged evidence loop with Postgres checkpointing.

        On first run: decomposes question, iterates, saves facts to Postgres
        after each iteration, snapshots checkpoint, checks for pause signal.
        On resume: reloads staged facts from Postgres, continues from iteration N+1.
        Returns the synthesised answer as a string, or a pause notice if paused.
        """
        query_id = task_id

        # ── Resume check: reload any previously staged facts ──────────────────
        prior_facts = await self.memory.load_research_staging(task_id)
        if prior_facts:
            resume_iter = max(f.get("_iteration", 1) for f in prior_facts)
            all_facts: list[dict] = [
                {k: v for k, v in f.items() if k != "_iteration"} for f in prior_facts
            ]
            start_iteration = resume_iter + 1
            log.info(
                "research.resuming",
                task_id=task_id,
                facts_loaded=len(all_facts),
                resuming_from_iteration=start_iteration,
            )
        else:
            all_facts = []
            start_iteration = 1

        # Create (or rejoin) context stream for this research session
        await self.bus.create_context_stream(
            "research",
            query_id,
            question[:60],
            metadata={"status": "active", "question": question},
        )

        async with httpx.AsyncClient() as http:
            # ── Step 1: initial sub-query decomposition (skip on resume) ──────
            if start_iteration == 1:
                sub_queries = await self._decompose(question)
                log.info("research.sub_queries", queries=sub_queries)
            else:
                # Re-derive follow-up queries from what we already know
                confident_so_far = _compute_confidence(all_facts)
                sub_queries = await self._find_gaps(
                    question, confident_so_far, start_iteration - 1
                )
                if not sub_queries:
                    sub_queries = [question]

            for iteration in range(start_iteration, MAX_SEARCH_ITERATIONS + 1):
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

                        if not snippet or not url or len(snippet) < 40:
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

                all_facts.extend(new_facts)
                log.info(
                    "research.facts_accumulated",
                    total=len(all_facts),
                    new=len(new_facts),
                    iteration=iteration,
                )

                # ── Checkpoint: persist to Postgres after each iteration ───────
                if new_facts:
                    stamped = [{**f, "_iteration": iteration} for f in new_facts]
                    await self.memory.save_research_staging(task_id, iteration, stamped)

                await self.memory.save_context_snapshot(
                    task_id,
                    "research",
                    question[:60],
                    checkpoint_label=f"iteration_{iteration}",
                    snapshot_seq=iteration,
                    snapshot_json={
                        "iteration": iteration,
                        "total_facts": len(all_facts),
                        "question": question[:200],
                    },
                )

                # ── Pause signal check ─────────────────────────────────────────
                pause_key = f"research:pause:{task_id}"
                if await self.bus._client.getdel(pause_key):
                    log.info("research.paused", task_id=task_id, iteration=iteration)
                    await self.bus.publish(
                        Event(
                            type=EventType.TASK_PAUSED,
                            source=self.role,
                            payload={
                                "task_id": task_id,
                                "paused_at_iteration": iteration,
                                "facts_staged": len(all_facts),
                            },
                            task_id=task_id,
                        ),
                        target="orchestrator",
                    )
                    return (
                        f"Research paused after iteration {iteration} "
                        f"({len(all_facts)} facts staged). "
                        f"Use /resume {task_id} to continue."
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

        # ── Commit to Postgres + clean up staging ─────────────────────────────
        await self._commit_sources(query_id, question, confident_facts)
        await self.memory.delete_research_staging(task_id)

        return answer

    async def _build_knowledge_db(self, task: str, task_id: str) -> str:
        """
        Multi-angle deep research → cluster facts by subtopic → store in knowledge table.

        1. Run _research() to collect high-confidence facts on the topic
        2. Ask LLM to cluster facts into 3-6 named subtopics
        3. For each subtopic: synthesise a knowledge entry and batch-store in `knowledge`
        4. Return a summary of what was stored
        """
        # Extract the core topic: strip common prefixes like "build knowledge base on ..."
        topic_raw = (
            re.sub(
                r"(?i)^(build|compile|create|research and store|deep research|"
                r"build a knowledge (database|base|db) (on|about|for)|"
                r"knowledge (database|base|db) (on|about|for)|"
                r"build knowledge (on|about|for))\s*",
                "",
                task,
            ).strip()
            or task
        )

        log.info(
            "research.build_knowledge_start", topic=topic_raw[:80], task_id=task_id
        )

        # Run normal research loop to gather facts
        answer = await self._research(topic_raw, task_id)

        # Reload facts committed in that research pass
        pool = await self.memory._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT fact, domain, url, title
                FROM research_sources
                WHERE query_id = %s AND committed = TRUE
                ORDER BY confidence DESC
                """,
                (task_id,),
            )
            rows = await cur.fetchall()

        if not rows:
            return (
                f"Research completed but no high-confidence facts were found for '{topic_raw}'. "
                f"The synthesised answer: {answer}"
            )

        facts = [
            {"fact": r[0], "domain": r[1], "url": r[2], "title": r[3]} for r in rows
        ]

        # Cluster facts into subtopics
        clusters = await self._cluster_facts(topic_raw, facts)
        if not clusters:
            # Fallback: store everything as one entry
            clusters = [
                {"subtopic": topic_raw, "indices": list(range(1, len(facts) + 1))}
            ]

        # Build knowledge entries per subtopic
        entries: list[dict] = []
        subtopics_built: list[str] = []
        for cluster in clusters:
            subtopic = cluster.get("subtopic", "general")
            indices = [
                i - 1 for i in cluster.get("indices", []) if 1 <= i <= len(facts)
            ]
            cluster_facts = [facts[i] for i in indices]
            if not cluster_facts:
                continue
            summary = await self._summarise_subtopic(subtopic, cluster_facts)
            tags = ["research", "knowledge-db", topic_raw[:50].lower()]
            entries.append(
                {
                    "topic": f"{topic_raw}/{subtopic}",
                    "content": summary,
                    "tags": tags,
                }
            )
            subtopics_built.append(subtopic)

        if entries:
            # Use batch_store which handles embeddings automatically
            await self.memory.batch_store(entries)
            log.info(
                "research.knowledge_stored",
                topic=topic_raw[:60],
                subtopics=len(entries),
                task_id=task_id,
            )

        subtopic_list = ", ".join(subtopics_built) if subtopics_built else "none"
        return (
            f"Knowledge database built for '{topic_raw}': "
            f"{len(entries)} subtopic entries stored ({subtopic_list}). "
            f"Summary: {answer[:400]}"
        )

    # ── LLM calls (local) ─────────────────────────────────────────────────────

    async def _cluster_facts(self, topic: str, facts: list[dict]) -> list[dict]:
        """Ask LLM to group facts into named subtopics. Returns list of {subtopic, indices}."""
        try:
            resp = await self.llm_invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=_cluster_prompt(topic, facts)),
                ]
            )
            parsed = _parse_json_from_llm(resp.content)
            if isinstance(parsed, list) and parsed:
                return [
                    c
                    for c in parsed
                    if isinstance(c, dict) and "subtopic" in c and "indices" in c
                ]
        except Exception as exc:
            log.warning("research.cluster_failed", error=str(exc))
        return []

    async def _summarise_subtopic(self, subtopic: str, facts: list[dict]) -> str:
        """Synthesise a knowledge paragraph for a single subtopic cluster."""
        try:
            resp = await self.llm_invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=_knowledge_summary_prompt(subtopic, facts)),
                ]
            )
            return resp.content.strip()
        except Exception as exc:
            log.warning("research.subtopic_summary_failed", error=str(exc))
            return " ".join(f["fact"] for f in facts)

    async def _fetch_and_learn(self, url: str, task_id: str) -> str:
        """
        Fetch a URL directly, extract plain text, ask the LLM to pull out key
        facts, and store them in the knowledge base under the URL's domain.

        Returns a summary of what was stored.
        """
        log.info("research.fetch_and_learn", url=url[:120])
        text = await _fetch_url(url)

        if text.startswith("[fetch error:"):
            log.warning("research.fetch_failed", url=url[:120], error=text)
            return f"Could not fetch {url}: {text}"

        domain = _domain(url)

        # Ask LLM to extract 3-8 key facts from the page content
        extract_prompt = (
            "You are extracting key facts from a web page for a knowledge base.\n"
            "Read the content below and return a JSON array of concise fact strings "
            "(3-8 facts, each under 120 characters). Include only information that is "
            "specific, verifiable, and useful for answering future questions.\n"
            "Return ONLY the JSON array, nothing else.\n\n"
            f"Source: {url}\n\n"
            f"Content:\n{text[:6000]}"
        )
        try:
            resp = await self.llm_invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=extract_prompt),
                ]
            )
            facts = _parse_json_from_llm(resp.content)
            if not isinstance(facts, list):
                facts = []
        except Exception as exc:
            log.warning("research.fetch_learn_extract_failed", error=str(exc))
            facts = []

        if not facts:
            # Fall back: store the truncated raw text as a single knowledge entry
            await self.promote_memory(
                f"[{domain}] Page content: {text[:500]}",
                topic=domain,
                tags=["fetched", domain],
            )
            return f"Fetched {url} but could not extract structured facts. Raw content stored."

        # Store each fact individually
        entries = [
            {
                "topic": domain,
                "content": f"[{domain}] {fact}",
                "tags": ["fetched", domain],
            }
            for fact in facts
            if isinstance(fact, str) and fact.strip()
        ]
        for entry in entries:
            await self.promote_memory(
                entry["content"],
                topic=entry["topic"],
                tags=entry["tags"],
            )

        log.info("research.fetch_learn_stored", url=url[:80], facts=len(entries))
        summary_lines = "\n".join(f"- {e['content']}" for e in entries[:5])
        return f"Fetched {url} and stored {len(entries)} facts:\n{summary_lines}" + (
            "\n…" if len(entries) > 5 else ""
        )

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
            # Inject prior memory so the LLM can cross-check and flag discrepancies.
            mem_ctx = await self.build_task_context(
                question, tools_limit=0, memory_limit=4
            )
            resp = await self.llm_invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT + mem_ctx),
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
    A fact earns confidence proportional to how many independent domains extracted
    the same fact (matched by 60-char prefix), regardless of which sub-query found it.
    Returns only facts meeting CONFIDENCE_COMMIT_THRESHOLD, de-duplicated by domain+fact.
    """
    # Deduplicate: one entry per (domain, fact-prefix) pair
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for f in facts:
        key = (f.get("domain", ""), f.get("fact", "")[:60])
        if key not in seen:
            seen.add(key)
            deduped.append(f)

    # Count distinct domains that extracted the same fact prefix (cross-sub-query)
    fact_domains: dict[str, set[str]] = {}
    for f in deduped:
        fp = f.get("fact", "")[:60]
        fact_domains.setdefault(fp, set()).add(f.get("domain", ""))

    result = []
    for f in deduped:
        fp = f.get("fact", "")[:60]
        n_agree = len(fact_domains.get(fp, set()))
        conf = min(1.0, n_agree / MIN_SOURCES_FOR_CONSENSUS)
        if conf >= CONFIDENCE_COMMIT_THRESHOLD:
            result.append({**f, "confidence": round(conf, 2)})

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_agent(ResearchAgent(Settings()))
