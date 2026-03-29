"""
agents/code_search/main.py
───────────────────────────
Code search agent — indexes and searches repositories,
finds patterns, explains code, promotes findings to long-term memory.
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from core.base_agent import BaseAgent, run_agent
from core.config import Settings
from core.events.bus import Event, EventType

log = structlog.get_logger()

SYSTEM_PROMPT = """You are a code analysis specialist.
You search codebases for patterns, bugs, and architectural decisions.
Be precise about file paths and line references when possible.
When you find a bug or important pattern, flag it clearly so it can be stored in long-term memory.
"""

# File extensions to index
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".cs",
    ".cpp", ".c", ".h", ".rb", ".php", ".swift", ".kt",
    ".sh", ".yaml", ".yml", ".toml", ".json", ".md",
}

# Dirs to skip
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", "target",
}


class CodeSearchAgent(BaseAgent):

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.repo_path = Path(os.getenv("REPO_PATH", "/workspace/repos"))

    async def on_startup(self) -> None:
        self.repo_path.mkdir(parents=True, exist_ok=True)
        log.info("code_search.startup", repo_path=str(self.repo_path))

    async def handle_event(self, event: Event) -> None:
        if event.type == EventType.TASK_ASSIGNED:
            if event.payload.get("assigned_to") == "code_search":
                await self._handle_task(event)

    async def _handle_task(self, event: Event) -> None:
        task = event.payload.get("task", "")
        task_id = event.task_id
        subtask_id = event.payload.get("subtask_id")
        parent_task_id = event.payload.get("parent_task_id")
        log.info("code_search.task", task=task[:80])

        def _reply(result: str) -> dict:
            return {
                "result": result,
                "task_id": task_id,
                "subtask_id": subtask_id,
                "parent_task_id": parent_task_id,
            }

        # Check long-term memory first
        prior = await self.recall(task)
        if prior:
            log.info("code_search.cache_hit", count=len(prior))
            result_text = "\n\n".join(
                f"[{e['topic']}] {e['content']}" for e in prior[:3]
            )
            await self.emit(
                EventType.TASK_COMPLETED,
                payload=_reply(result_text),
                target="orchestrator",
            )
            return

        # Search codebase
        snippets = self._search_code(task)
        if not snippets:
            await self.emit(
                EventType.TASK_COMPLETED,
                payload=_reply("No relevant code found in /workspace/repos."),
                target="orchestrator",
            )
            return

        # Analyze via LLM
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Code snippets:\n{snippets[:8000]}\n\nTask: {task}"),
        ]
        response = await self.llm.ainvoke(messages)
        analysis = response.content

        # Stage findings for long-term promotion
        self.stage_finding(
            content=f"Task: {task}\nAnalysis: {analysis}",
            topic="code_analysis",
            tags=["code", "search"],
        )

        await self.emit(
            EventType.TASK_COMPLETED,
            payload=_reply(analysis),
            target="orchestrator",
        )

    def _search_code(self, query: str, max_chars: int = 12000) -> str:
        """Simple keyword search across repo files."""
        keywords = query.lower().split()
        results = []
        total = 0

        for path in self._iter_code_files():
            try:
                text = path.read_text(errors="ignore")
            except Exception:
                continue

            text_lower = text.lower()
            if any(kw in text_lower for kw in keywords):
                # Extract relevant lines
                lines = text.splitlines()
                relevant = [
                    f"  {i+1}: {line}"
                    for i, line in enumerate(lines)
                    if any(kw in line.lower() for kw in keywords)
                ]
                if relevant:
                    snippet = f"### {path.relative_to(self.repo_path)}\n" + "\n".join(relevant[:20])
                    results.append(snippet)
                    total += len(snippet)
                    if total >= max_chars:
                        break

        return "\n\n".join(results)

    def _iter_code_files(self):
        for path in self.repo_path.rglob("*"):
            if any(skip in path.parts for skip in SKIP_DIRS):
                continue
            if path.is_file() and path.suffix in CODE_EXTENSIONS:
                yield path

    async def on_shutdown(self) -> None:
        log.info("code_search.shutdown")


if __name__ == "__main__":
    settings = Settings()
    agent = CodeSearchAgent(settings)
    run_agent(agent)
