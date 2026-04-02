"""
core/errors.py
──────────────
Structured error codes for inter-agent communication.

Using typed error codes instead of raw strings makes retry logic, escalation
decisions, and debugging far more reliable across the stack.
"""

from __future__ import annotations

from enum import Enum


class AgentError(str, Enum):
    # Command execution errors
    COMMAND_TIMEOUT   = "command_timeout"    # Command exceeded time budget
    COMMAND_DENIED    = "command_denied"     # User denied approval gate
    COMMAND_BLOCKED   = "command_blocked"    # Command not on allowlist
    COMMAND_FAILED    = "command_failed"     # Non-zero exit code, no recovery

    # Approval gate errors
    APPROVAL_TIMEOUT  = "approval_timeout"   # No decision within timeout window
    APPROVAL_DENIED   = "approval_denied"    # User explicitly denied

    # Parsing / protocol errors
    PARSE_FAILURE     = "parse_failure"      # Could not extract command/intent
    SCHEMA_VIOLATION  = "schema_violation"   # Event payload missing required field

    # LLM / inference errors
    LLM_UNAVAILABLE   = "llm_unavailable"    # LM Studio unreachable
    LLM_TIMEOUT       = "llm_timeout"        # LLM call exceeded budget
    LLM_CIRCUIT_OPEN  = "llm_circuit_open"   # Circuit breaker open; LM Studio down
    LLM_CONTEXT_EXCEEDED = "llm_context_exceeded"  # Input too large even after truncation

    # Memory errors
    MEMORY_FAILURE    = "memory_failure"     # PostgreSQL unreachable or query failed
    EMBED_FAILURE     = "embed_failure"      # Sentence-transformer encoding failed

    # Task lifecycle errors
    TASK_TIMEOUT      = "task_timeout"       # Plan exceeded PLAN_TIMEOUT
    TASK_NOT_FOUND    = "task_not_found"     # Referenced task_id unknown
    MAX_RETRIES       = "max_retries"        # Phase exhausted retry budget

    # Generic
    UNKNOWN           = "unknown"            # Catch-all for unexpected errors


def error_payload(
    code: AgentError,
    message: str,
    *,
    task_id: str = "",
    triggering_event_id: str = "",
    details: dict | None = None,
) -> dict:
    """
    Build a structured error payload for EventType.ERROR events.
    Consumers can branch on payload["error_code"] instead of parsing strings.
    """
    p: dict = {
        "error_code": code.value,
        "error": message,
    }
    if task_id:
        p["task_id"] = task_id
    if triggering_event_id:
        p["triggering_event_id"] = triggering_event_id
    if details:
        p["details"] = details
    return p
