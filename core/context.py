"""
core/context.py
───────────────
Context window budget management and truncation utilities.

All agents share these constants. Inputs are truncated before passing to the
LLM to stay within the model's context window.

Rule of thumb: 1 token ≈ 4 characters.
Target: ≤ 24,000 tokens of input per LLM call (≈ 96,000 chars).
The system prompt + self-modify context consume ~5,000 chars, leaving ~91,000
chars usable. These conservative budgets leave plenty of headroom and keep
individual inputs sane even as task complexity grows.
"""

from __future__ import annotations

# ── Character budgets per input category ──────────────────────────────────────

BUDGET: dict[str, int] = {
    "task":          4_000,   # incoming task description from Discord / think loop
    "file":          8_000,   # file content read by executor or document_qa
    "code":         10_000,   # code search snippets
    "memory_entry":    600,   # single recalled memory entry content
    "memory_total":  3_000,   # all recall results combined
    "result":        3_000,   # specialist result relayed back to orchestrator
    "payload":       2_000,   # event payload stored in Redis streams
    "command_out":   3_000,   # shell command stdout / stderr
    "context":       4_000,   # assembled prior-knowledge context block
}

# ── Memory pruning thresholds ─────────────────────────────────────────────────

MEMORY_WARN_THRESHOLD  = 500    # emit MEMORY_PRUNED warning when count exceeds this
MEMORY_HARD_LIMIT      = 1_000  # prune when count exceeds this
MEMORY_PRUNE_TARGET    =   800  # prune oldest entries down to this count


# ── Core truncation helper ────────────────────────────────────────────────────

def truncate(text: str, max_chars: int, label: str = "content") -> str:
    """
    Truncate *text* to *max_chars*, appending a concise notice if truncated.
    The notice is intentionally short so it doesn't eat into the budget.
    """
    if not text or len(text) <= max_chars:
        return text
    removed = len(text) - max_chars
    return text[:max_chars] + f"\n[…{removed:,} chars omitted from {label}]"


# ── Typed helpers (used throughout the agent codebase) ───────────────────────

def truncate_task(task: str) -> str:
    return truncate(task, BUDGET["task"], "task")


def truncate_file(content: str, path: str = "file") -> str:
    return truncate(content, BUDGET["file"], path)


def truncate_code(snippets: str) -> str:
    return truncate(snippets, BUDGET["code"], "code search results")


def truncate_result(result: str) -> str:
    return truncate(result, BUDGET["result"], "result")


def truncate_payload(text: str) -> str:
    return truncate(text, BUDGET["payload"], "payload")


def truncate_command_output(output: str) -> str:
    return truncate(output, BUDGET["command_out"], "command output")


def truncate_context(context: str) -> str:
    return truncate(context, BUDGET["context"], "prior knowledge")


def truncate_memory_entries(entries: list[dict]) -> list[dict]:
    """
    Truncate each memory entry's content field and stop accumulating once the
    combined length exceeds BUDGET["memory_total"].
    Returns a (possibly shorter) list of entries safe to include in an LLM prompt.
    """
    out: list[dict] = []
    total = 0
    for e in entries:
        content = truncate(
            e.get("content", ""),
            BUDGET["memory_entry"],
            f"memory:{e.get('topic', '')}",
        )
        total += len(content)
        if total > BUDGET["memory_total"]:
            break
        out.append({**e, "content": content})
    return out
