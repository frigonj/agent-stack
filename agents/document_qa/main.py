"""
agents/document_qa/main.py
───────────────────────────
Document Q&A agent — reads PDFs and text files, answers questions,
generates LaTeX documentation, reviews agent-stack architecture,
and promotes findings to long-term memory.

Capabilities:
  • Q&A over documents in /workspace/docs (PDF, markdown, text)
  • PDF extraction via pypdf
  • Architecture review of the agent stack source at /workspace/src
  • LaTeX document generation via pylatex + compilation via latexmk
  • Output (generated PDFs and .tex files) written to /workspace/docs/generated/
  • Google Drive sync — uploads architecture.pdf to a shared Drive folder
    after every successful review (requires GOOGLE_DRIVE_FOLDER_ID +
    GOOGLE_APPLICATION_CREDENTIALS in the environment)

Architecture change detection lives in the orchestrator's think() loop so
this agent stays ephemeral and is only spun up when work is needed.
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from core.base_agent import BaseAgent, run_agent
from core.config import Settings
from core.events.bus import Event, EventType
from core.context import truncate_file, truncate_task

log = structlog.get_logger()

# ── Path constants ─────────────────────────────────────────────────────────────
_DOCS_PATH = Path(os.getenv("DOCS_PATH", "/workspace/docs"))
_REPO_PATH = Path(os.getenv("REPO_PATH", "/workspace/src"))
_GENERATED_PATH = Path(os.getenv("GENERATED_PATH", "/workspace/docs/generated"))

# Source extensions included in architecture review
_SRC_EXTENSIONS = {".py", ".yml", ".yaml", ".toml", ".json", ".md", ".txt", ".sh"}
_SRC_SKIP_DIRS = {
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
}

# Hard cap on source-tree chars loaded for a single review.
# The real cap is calculated dynamically from the model's context limit in
# _budget_content_chars(); this is the fallback for very small context windows.
_SRC_MAX_CHARS = 80_000

SYSTEM_PROMPT = """You are a documentation and architecture specialist with LaTeX typesetting skills.

## Capabilities

### Document Q&A
Answer questions accurately from documents in /workspace/docs (PDF, markdown, text).
Cite the relevant section when possible. Do not hallucinate.

### Architecture review (/workspace/src)
Read the agent-stack source tree and produce structured analysis:
- Component inventory (agents, core modules, Docker services)
- Data-flow descriptions (event bus, Redis Streams, Postgres memory)
- Dependency map
- Recent changes or design patterns worth documenting

### LaTeX document generation
Produce well-structured LaTeX source for architecture documents, reports, or summaries.
Use \\section, \\subsection, \\begin{itemize}, \\begin{lstlisting} as appropriate.
Output .tex files to /workspace/docs/generated/ and compile to PDF with latexmk.

### PDF reading
Extract and analyse content from existing PDF files in /workspace/docs.

## Output location
All generated .tex and compiled .pdf files go to /workspace/docs/generated/.
Reference them in your response with the full path.

## Multi-step document retrieval (ReAct loop)
If the initially provided content is insufficient, request additional files:
  READ: /workspace/docs/<filename>

After each read you will receive:
  OBSERVATION: <file content>

Read as many files as needed. When you have a complete answer:
  DONE: <your answer>

Only read files under /workspace/docs. If initial content is sufficient, respond
with DONE: directly.

## Tool-building
If you develop a reusable extraction or generation pattern, request executor to
save it as a named script in /workspace/tools/ for future reuse.

## research agent (delegate via orchestrator)
The research agent searches the internet via Wikipedia (offline) + Brave Search API:
- Decomposes questions → multi-query search → fact extraction → cross-source consensus
- Commits sourced facts to Postgres (table: research_sources)
Route "what is X", "latest version of Y", "current news/status" queries to research.

## developer agent (delegate via orchestrator)
The developer agent writes, edits, refactors, and fixes code:
- Implements features, fixes bugs, scaffolds new agents, writes tests, reviews code
- Works across /workspace/src (agent stack) and /workspace/projects (user projects)
Route tasks that require *writing or modifying* source code to developer.
"""


# ── Routing keywords ──────────────────────────────────────────────────────────

_ARCH_RE = re.compile(
    r"\b(architecture|arch\s*doc|document\s+the\s+stack|generate\s+arch|"
    r"review\s+(the\s+)?(source|stack|agents?|codebase)|"
    r"how\s+does\s+the\s+stack\s+work|agent\s+stack\s+overview|"
    r"system\s+design|component\s+diagram)\b",
    re.I,
)

_LATEX_RE = re.compile(
    r"\b(latex|generate\s+(?:a\s+)?(?:pdf|doc(?:ument)?|report)|"
    r"create\s+(?:a\s+)?(?:pdf|report|latex)|"
    r"write\s+(?:a\s+)?(?:report|document|doc)|compile\s+(?:latex|tex))\b",
    re.I,
)

_PDF_RE = re.compile(
    r"\b(read\s+(?:the\s+)?pdf|extract\s+from\s+(?:the\s+)?pdf|"
    r"what(\'?s|\s+is)\s+in\s+(?:the\s+)?pdf|"
    r"parse\s+(?:the\s+)?pdf|open\s+(?:the\s+)?pdf)\b",
    re.I,
)


class DocumentQAAgent(BaseAgent):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.docs_path = _DOCS_PATH
        self.repo_path = _REPO_PATH
        self.generated_path = _GENERATED_PATH

    _OWN_TOOLS = [
        (
            "answer-from-documents",
            "Answer questions by reading and analyzing documents in /workspace/docs (text, markdown, PDF)",
            "event:task.assigned:document_qa",
            ["documents", "qa", "answer", "summarize"],
        ),
        (
            "read-pdf-document",
            "Extract and analyze content from PDF files in /workspace/docs using pypdf",
            "event:task.assigned:document_qa",
            ["pdf", "read", "extract", "documents"],
        ),
        (
            "generate-architecture-docs",
            "Review the agent-stack source at /workspace/src and generate a LaTeX architecture document",
            "event:task.assigned:document_qa",
            ["architecture", "docs", "latex", "agent-stack", "review"],
        ),
        (
            "compile-latex-document",
            "Compile a .tex file in /workspace/docs/generated/ to PDF using latexmk",
            "event:task.assigned:document_qa",
            ["latex", "compile", "pdf", "texlive"],
        ),
        (
            "list-generated-documents",
            "List all documents generated by document_qa in /workspace/docs/generated/",
            "shell:ls /workspace/docs/generated/",
            ["documents", "list", "generated"],
        ),
        (
            "list-workspace-documents",
            "List documents available in /workspace/docs",
            "shell:ls /workspace/docs/",
            ["documents", "list"],
        ),
    ]

    async def on_startup(self) -> None:
        self.docs_path.mkdir(parents=True, exist_ok=True)
        self.generated_path.mkdir(parents=True, exist_ok=True)
        for name, desc, inv, tags in self._OWN_TOOLS:
            await self.memory.register_tool(
                name, desc, "document_qa", inv, tags, "document_qa"
            )
        log.info(
            "document_qa.startup",
            docs_path=str(self.docs_path),
            repo_path=str(self.repo_path),
            generated_path=str(self.generated_path),
            tools_seeded=len(self._OWN_TOOLS),
        )

    async def handle_event(self, event: Event) -> None:
        if event.type == EventType.TASK_ASSIGNED:
            if event.payload.get("assigned_to") == "document_qa":
                await self._handle_task(event)

    async def _handle_task(self, event: Event) -> None:
        task = truncate_task(event.payload.get("task", ""))
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

        # ── Route by keyword ──────────────────────────────────────────────────
        if _ARCH_RE.search(task):
            result = await self._handle_architecture_review(task)
        elif _LATEX_RE.search(task):
            result = await self._handle_latex_generate(task)
        elif _PDF_RE.search(task):
            result = await self._handle_pdf_read(task)
        else:
            result = await self._handle_qa(task)

        await self.emit(
            EventType.TASK_COMPLETED,
            payload=_reply(result),
            target="orchestrator",
        )

    # ── Q&A handler (default) ─────────────────────────────────────────────────

    async def _handle_qa(self, task: str) -> str:
        # Embed once — check memory and load tools in parallel
        prior, tool_hits = await self.recall_and_search_tools(task)
        if prior:
            log.info("document_qa.cache_hit", count=len(prior))
            return prior[0]["content"]

        # Try PDFs first, then text/markdown docs as initial context seed
        doc_content = self._load_pdfs_as_text() or self._load_text_docs()
        if not doc_content:
            return "No documents found in /workspace/docs."

        tools_ctx = self.format_tools_context(tool_hits)
        system_msg = SYSTEM_PROMPT + tools_ctx
        budget = await self._budget_content_chars(system_msg, task)
        capped_content = doc_content[:budget]

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(
                content=(
                    f"Documents loaded:\n{capped_content}\n\n"
                    f"Question: {task}\n\n"
                    f"Answer from the documents. Use READ: /workspace/docs/<file> to load "
                    f"additional files if needed. When done, respond with DONE: <answer>."
                )
            ),
        ]

        async def _read_action(action_type: str, payload: str) -> str:
            if action_type == "READ":
                path = Path(payload.strip())
                if not str(path).startswith("/workspace/docs"):
                    return "Access denied: can only read files under /workspace/docs."
                try:
                    text = path.read_text(errors="ignore")
                    return truncate_file(text)
                except FileNotFoundError:
                    return f"File not found: {path}"
                except Exception as exc:
                    return f"Error reading {path}: {exc}"
            return f"Unknown action: {action_type}"

        answer = await self.agent_loop(
            messages,
            action_handler=_read_action,
            max_steps=3,
            subtask_id=subtask_id or "",
            parent_task_id=parent_task_id or "",
        )

        return answer

    # ── PDF reading handler ───────────────────────────────────────────────────

    async def _handle_pdf_read(self, task: str) -> str:
        pdf_content = self._load_pdfs_as_text()
        if not pdf_content:
            return "No PDF files found in /workspace/docs."

        tool_hits = await self.search_tools(task)
        tools_ctx = self.format_tools_context(tool_hits)
        system_msg = SYSTEM_PROMPT + tools_ctx
        budget = await self._budget_content_chars(system_msg, task)
        capped_content = pdf_content[:budget]

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(
                content=f"Extracted PDF content:\n{capped_content}\n\nTask: {task}"
            ),
        ]
        response = await self.llm_invoke(messages)
        return response.content

    # ── Architecture review + LaTeX generation ────────────────────────────────

    async def _handle_architecture_review(self, task: str) -> str:
        tool_hits = await self.search_tools(task)
        tools_ctx = self.format_tools_context(tool_hits)
        system_msg = SYSTEM_PROMPT + tools_ctx

        # Calculate how many chars of source tree fit within the context window
        # Reserve ~300 chars for the fixed instructions appended to the human message
        instructions = (
            "\n\nProduce a comprehensive architecture summary covering:\n"
            "1. Component inventory (agents, core modules, Docker services)\n"
            "2. Event flow (Redis Streams, context streams, routing)\n"
            "3. Memory architecture (Redis short-term, Postgres long-term)\n"
            "4. Key design patterns (voting, fix subtasks, topic classification)\n"
            "5. Configuration surface (env vars, runtime Redis config)\n\n"
            "Then produce a complete LaTeX document source for this architecture.\n"
            "I will compile and save it."
        )
        budget = await self._budget_content_chars(system_msg, task + instructions)
        source_tree = self._load_source_tree(max_chars=min(budget, _SRC_MAX_CHARS))
        if not source_tree:
            return (
                "Agent-stack source not found at /workspace/src. "
                "Ensure the volume is mounted correctly."
            )

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(
                content=(
                    f"Agent-stack source tree:\n{source_tree}\n\n"
                    f"Task: {task}" + instructions
                )
            ),
        ]
        response = await self.llm_invoke(messages)
        arch_text = response.content

        # Extract LaTeX block if present and compile
        compile_result = await self._extract_and_compile_latex(
            arch_text, "architecture"
        )

        # Sync to Google Drive whenever a review produces a fresh PDF
        pdf_path = self.generated_path / "architecture.pdf"
        if pdf_path.exists():
            await self._upload_to_drive(pdf_path)

        return arch_text + compile_result

    async def _handle_latex_generate(self, task: str) -> str:
        tool_hits = await self.search_tools(task)
        tools_ctx = self.format_tools_context(tool_hits)
        system_msg = SYSTEM_PROMPT + tools_ctx

        # Provide source context if the task references the stack
        ctx = ""
        if any(
            kw in task.lower() for kw in ("stack", "agent", "system", "architecture")
        ):
            latex_instructions = (
                "\n\nProduce a complete, compilable LaTeX document. "
                "Wrap the LaTeX source in \\begin{document}...\\end{document}. "
                "I will extract and compile it."
            )
            budget = await self._budget_content_chars(
                system_msg, task + latex_instructions
            )
            ctx = self._load_source_tree(max_chars=min(budget, _SRC_MAX_CHARS))

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(
                content=(
                    (f"Context (agent-stack source):\n{ctx}\n\n" if ctx else "")
                    + f"Task: {task}\n\n"
                    "Produce a complete, compilable LaTeX document. "
                    "Wrap the LaTeX source in \\begin{document}...\\end{document}. "
                    "I will extract and compile it."
                )
            ),
        ]
        response = await self.llm_invoke(messages)
        latex_text = response.content

        compile_result = await self._extract_and_compile_latex(
            latex_text, "generated_doc"
        )
        return latex_text + compile_result

    # ── LaTeX helpers ─────────────────────────────────────────────────────────

    async def _extract_and_compile_latex(self, llm_output: str, basename: str) -> str:
        """
        Pull a LaTeX document out of *llm_output*, write it to the generated
        directory, and compile it with latexmk. Returns a status suffix.
        Falls back to writing raw markdown when no LaTeX block is found.
        """
        tex_source = self._extract_latex_block(llm_output)
        if not tex_source:
            log.warning(
                "document_qa.no_latex_found",
                basename=basename,
                output_preview=llm_output[:200],
            )
            # Fallback: write the raw output as a markdown file
            safe_name = re.sub(r"[^a-z0-9_\-]", "_", basename.lower())[:50]
            md_path = self.generated_path / f"{safe_name}.md"
            try:
                md_path.write_text(llm_output, encoding="utf-8")
                log.info("document_qa.md_written", path=str(md_path))
            except Exception as exc:
                log.warning(
                    "document_qa.md_write_error", path=str(md_path), error=str(exc)
                )
            return f"\n\n[No LaTeX block found — saved as {md_path}]"

        # Sanitise basename
        safe_name = re.sub(r"[^a-z0-9_\-]", "_", basename.lower())[:50]
        tex_path = self.generated_path / f"{safe_name}.tex"

        try:
            tex_path.write_text(tex_source, encoding="utf-8")
            log.info("document_qa.tex_written", path=str(tex_path))
        except Exception as exc:
            return f"\n\n[LaTeX write error: {exc}]"

        compile_msg = await self._compile_latex(tex_path)
        return f"\n\n---\n{compile_msg}"

    def _extract_latex_block(self, text: str) -> Optional[str]:
        """
        Return the LaTeX source from *text*. Accepts three forms:
        1. ```latex ... ``` code fence
        2. Raw text that contains \\documentclass
        3. Nothing found → None
        """
        # Fenced code block
        fence_match = re.search(r"```(?:latex|tex)\s*\n(.*?)```", text, re.S)
        if fence_match:
            return fence_match.group(1).strip()

        # Raw LaTeX (starts with \documentclass or contains \begin{document})
        if r"\documentclass" in text or r"\begin{document}" in text:
            # Find from \documentclass to \end{document}
            start = text.find(r"\documentclass")
            end = text.find(r"\end{document}")
            if start != -1 and end != -1:
                return text[start : end + len(r"\end{document}")]

        return None

    async def _compile_latex(self, tex_path: Path) -> str:
        """
        Run latexmk on *tex_path* inside its parent directory.
        Returns a human-readable status string.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                tex_path.name,
                cwd=str(tex_path.parent),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            code = proc.returncode

            pdf_path = tex_path.with_suffix(".pdf")
            if code == 0 and pdf_path.exists():
                log.info("document_qa.pdf_compiled", pdf=str(pdf_path))
                return f"LaTeX compiled successfully → {pdf_path}"
            else:
                err_snippet = stderr.decode(errors="replace")[-800:]
                log.warning(
                    "document_qa.compile_failed", code=code, stderr=err_snippet[:200]
                )
                return f"LaTeX compilation failed (exit {code}):\n{err_snippet}"

        except asyncio.TimeoutError:
            return "LaTeX compilation timed out after 120 s."
        except Exception as exc:
            return f"LaTeX compilation error: {exc}"

    # ── Document loading helpers ──────────────────────────────────────────────

    _CHUNK_SIZE = 1_000
    _CHUNK_OVERLAP = 200

    def _chunk_text(self, text: str, filename: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self._CHUNK_SIZE
            chunk = text[start:end]
            chunks.append(f"[{filename} chunk {idx}]\n{chunk}")
            start += self._CHUNK_SIZE - self._CHUNK_OVERLAP
            idx += 1
        return chunks

    def _load_text_docs(self) -> str:
        """Load .txt and .md files from /workspace/docs."""
        all_chunks: list[str] = []
        for ext in ("*.txt", "*.md"):
            for path in self.docs_path.glob(ext):
                try:
                    content = path.read_text()
                    all_chunks.extend(self._chunk_text(content, path.name))
                    log.debug("document_qa.loaded_text", file=path.name)
                except Exception as e:
                    log.warning("document_qa.read_error", path=str(path), error=str(e))
        return "\n\n".join(all_chunks)

    def _load_pdfs_as_text(self) -> str:
        """Extract text from all PDFs in /workspace/docs using pypdf."""
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            log.warning("document_qa.pypdf_missing")
            return ""

        all_chunks: list[str] = []
        for path in self.docs_path.glob("*.pdf"):
            try:
                reader = PdfReader(str(path))
                pages_text: list[str] = []
                for page in reader.pages:
                    text = page.extract_text() or ""
                    if text.strip():
                        pages_text.append(text)
                combined = "\n".join(pages_text)
                all_chunks.extend(self._chunk_text(combined, path.name))
                log.debug(
                    "document_qa.loaded_pdf", file=path.name, pages=len(reader.pages)
                )
            except Exception as exc:
                log.warning("document_qa.pdf_error", path=str(path), error=str(exc))
        return "\n\n".join(all_chunks)

    def _load_source_tree(self, max_chars: int = _SRC_MAX_CHARS) -> str:
        """
        Read the agent-stack source tree at /workspace/src.
        Returns a concatenated, labelled view of key source files.
        Prioritises agent main.py files, core modules, and config files.
        """
        if not self.repo_path.exists():
            return ""

        sections: list[str] = []
        total = 0

        # Priority order: agents first, then core, then top-level configs
        def _priority(p: Path) -> int:
            parts = p.relative_to(self.repo_path).parts
            if parts[0] == "agents":
                return 0
            if parts[0] == "core":
                return 1
            if parts[0] in ("config", "docker", "scripts"):
                return 2
            return 3

        candidates = sorted(
            (
                p
                for p in self.repo_path.rglob("*")
                if p.is_file()
                and p.suffix in _SRC_EXTENSIONS
                and not any(skip in p.parts for skip in _SRC_SKIP_DIRS)
            ),
            key=_priority,
        )

        for path in candidates:
            if total >= max_chars:
                break
            try:
                content = path.read_text(errors="ignore")
                rel = str(path.relative_to(self.repo_path))
                header = f"### {rel}"
                snippet = f"{header}\n{content[:3_000]}"  # cap per-file at 3k chars
                sections.append(snippet)
                total += len(snippet)
            except Exception:
                pass

        return "\n\n".join(sections)

    # ── Google Drive sync ─────────────────────────────────────────────────────

    async def _upload_to_drive(self, pdf_path: Path) -> None:
        """
        Upload *pdf_path* to Google Drive.

        Creates a dated snapshot (architecture_YYYY-MM-DD_HHMM.pdf) AND
        overwrites architecture_latest.pdf in the configured folder.

        Silently skips if GOOGLE_DRIVE_FOLDER_ID or GOOGLE_APPLICATION_CREDENTIALS
        are not set, so local-only deployments need no changes.
        """
        folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        if not folder_id or not creds_path or not Path(creds_path).exists():
            log.debug(
                "document_qa.drive_skip",
                reason="GOOGLE_DRIVE_FOLDER_ID or credentials not configured",
            )
            return

        try:
            from google.oauth2 import service_account  # type: ignore
            from googleapiclient.discovery import build  # type: ignore
            from googleapiclient.http import MediaFileUpload  # type: ignore
        except ImportError:
            log.warning(
                "document_qa.drive_libs_missing",
                hint="pip install google-api-python-client google-auth",
            )
            return

        try:
            creds = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=["https://www.googleapis.com/auth/drive.file"],
            )
            loop = asyncio.get_event_loop()
            service = await loop.run_in_executor(
                None, lambda: build("drive", "v3", credentials=creds)
            )

            # Dated snapshot
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
            dated_name = f"architecture_{stamp}.pdf"
            media = MediaFileUpload(str(pdf_path), mimetype="application/pdf")
            await loop.run_in_executor(
                None,
                lambda: (
                    service.files()
                    .create(
                        body={
                            "name": dated_name,
                            "parents": [folder_id],
                            "mimeType": "application/pdf",
                        },
                        media_body=media,
                        fields="id",
                    )
                    .execute()
                ),
            )
            log.info("document_qa.drive_snapshot_uploaded", name=dated_name)

            # Overwrite architecture_latest.pdf
            await self._drive_upsert(
                service, folder_id, "architecture_latest.pdf", pdf_path
            )
            log.info("document_qa.drive_latest_updated", folder=folder_id)

        except Exception as exc:
            log.warning("document_qa.drive_upload_failed", error=str(exc))

    async def _drive_upsert(
        self,
        service: object,
        folder_id: str,
        filename: str,
        pdf_path: Path,
    ) -> None:
        """Create or overwrite *filename* in *folder_id* on Google Drive."""
        from googleapiclient.http import MediaFileUpload  # type: ignore

        loop = asyncio.get_event_loop()
        media = MediaFileUpload(
            str(pdf_path), mimetype="application/pdf", resumable=False
        )

        results = await loop.run_in_executor(
            None,
            lambda: (
                service.files()
                .list(
                    q=f"name='{filename}' and '{folder_id}' in parents and trashed=false",
                    fields="files(id)",
                    spaces="drive",
                )
                .execute()
            ),
        )
        existing = results.get("files", [])

        if existing:
            fid = existing[0]["id"]
            await loop.run_in_executor(
                None,
                lambda: service.files().update(fileId=fid, media_body=media).execute(),
            )
        else:
            await loop.run_in_executor(
                None,
                lambda: (
                    service.files()
                    .create(
                        body={
                            "name": filename,
                            "parents": [folder_id],
                            "mimeType": "application/pdf",
                        },
                        media_body=media,
                        fields="id",
                    )
                    .execute()
                ),
            )

    async def on_shutdown(self) -> None:
        log.info("document_qa.shutdown")


if __name__ == "__main__":
    settings = Settings()
    agent = DocumentQAAgent(settings)
    run_agent(agent)
