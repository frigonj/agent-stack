"""
agents/document_qa/main.py
───────────────────────────
Document Q&A agent — reads PDFs and text files, answers questions,
promotes findings to long-term memory.
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

SYSTEM_PROMPT = """You are a document analysis specialist.
You receive document content and answer questions about it accurately and concisely.
Cite the relevant section when possible.
If the answer is not in the document, say so clearly — do not hallucinate.
"""


class DocumentQAAgent(BaseAgent):

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.docs_path = Path(os.getenv("DOCS_PATH", "/workspace/docs"))

    async def on_startup(self) -> None:
        self.docs_path.mkdir(parents=True, exist_ok=True)
        log.info("document_qa.startup", docs_path=str(self.docs_path))

    async def handle_event(self, event: Event) -> None:
        if event.type == EventType.TASK_ASSIGNED:
            if event.payload.get("assigned_to") == "document_qa":
                await self._handle_task(event)

    async def _handle_task(self, event: Event) -> None:
        task = event.payload.get("task", "")
        task_id = event.task_id
        subtask_id = event.payload.get("subtask_id")
        parent_task_id = event.payload.get("parent_task_id")
        log.info("document_qa.task", task=task[:80])

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
            log.info("document_qa.cache_hit", count=len(prior))
            best = prior[0]
            await self.emit(
                EventType.TASK_COMPLETED,
                payload=_reply(best["content"]),
                target="orchestrator",
            )
            return

        # Load documents from workspace
        doc_content = self._load_docs()
        if not doc_content:
            await self.emit(
                EventType.TASK_COMPLETED,
                payload=_reply("No documents found in /workspace/docs."),
                target="orchestrator",
            )
            return

        # Answer via LLM
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Document content:\n{doc_content[:8000]}\n\nQuestion: {task}"),
        ]
        response = await self.llm.ainvoke(messages)
        answer = response.content

        # Stage for promotion — document findings persist across sessions
        self.stage_finding(
            content=f"Q: {task}\nA: {answer}",
            topic="document_qa",
            tags=["qa", "document"],
        )

        await self.emit(
            EventType.TASK_COMPLETED,
            payload=_reply(answer),
            target="orchestrator",
        )

    def _load_docs(self) -> str:
        """Load all .txt and .md files from docs workspace."""
        chunks = []
        for ext in ("*.txt", "*.md"):
            for path in self.docs_path.glob(ext):
                try:
                    chunks.append(f"--- {path.name} ---\n{path.read_text()}")
                except Exception as e:
                    log.warning("document_qa.read_error", path=str(path), error=str(e))
        return "\n\n".join(chunks)

    async def on_shutdown(self) -> None:
        log.info("document_qa.shutdown")


if __name__ == "__main__":
    settings = Settings()
    agent = DocumentQAAgent(settings)
    run_agent(agent)
