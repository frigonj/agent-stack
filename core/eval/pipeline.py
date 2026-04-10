"""
core/eval/pipeline.py
──────────────────────
Orchestrates the full 3-tier eval pipeline for a completed plan.

Called from orchestrator._close_task_context() after every plan completion.
Runs asynchronously — does NOT block the orchestrator's main loop.

Flow:
  1. Classify task type
  2. Tier 1: structural checks (always)
  3. Tier 2: local LLM judge (only if T1 passed)
  4. Tier 3: Gemini judge (only if T2 score is borderline)
  5. Save eval_result to Postgres
  6. Emit review event if human review needed

Usage:
    from core.eval.pipeline import EvalPipeline
    pipeline = EvalPipeline(memory=self.memory, bus=self.bus)
    asyncio.create_task(pipeline.run(plan, synthesised_reply))
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import structlog

from core.eval.structural import StructuralChecker, classify_task
from core.eval.llm_judge import LLMJudge, needs_tier3, is_auto_approve

if TYPE_CHECKING:
    from core.memory.long_term import LongTermMemory
    from core.events.bus import EventBus

log = structlog.get_logger(__name__)

_EVAL_DIR = Path(os.getenv("EVAL_DIR", "/workspace/eval"))
_REVIEW_PENDING = _EVAL_DIR / "review_queue" / "pending"
_REVIEW_APPROVED = _EVAL_DIR / "review_queue" / "approved"
_REVIEW_REJECTED = _EVAL_DIR / "review_queue" / "rejected"


class EvalPipeline:
    def __init__(self, memory: "LongTermMemory", bus: "EventBus"):
        self.memory = memory
        self.bus = bus
        self._checker = StructuralChecker()
        self._judge = LLMJudge()

    async def run(self, plan, synthesised_reply: str = "") -> None:
        """
        Run the full eval pipeline for a completed plan.
        All exceptions are caught — eval failures must never affect the orchestrator.
        """
        try:
            await self._run(plan, synthesised_reply)
        except Exception as exc:
            log.warning("eval.pipeline.unhandled_error", error=str(exc))

    async def _run(self, plan, synthesised_reply: str) -> None:
        plan_dict = plan.to_dict()
        original_task = plan.original_task
        task_id = plan.task_id
        plan_id = plan.plan_id

        # Derive plan metadata
        steps = plan_dict.get("steps", [])
        agents_used = list(dict.fromkeys(
            s.get("agent") for s in steps if s.get("status") == "done"
        ))
        plan_retries = sum(s.get("retry_count", 0) for s in steps)
        approval_requested = plan_dict.get("approval_requested", False)

        # ── Tier 1: structural ────────────────────────────────────────────
        task_type = classify_task(original_task)
        t1 = self._checker.check(
            original_task=original_task,
            plan_dict=plan_dict,
            result_text=synthesised_reply,
            task_type=task_type,
        )

        eval_record: dict = {
            "task_id": task_id,
            "plan_id": plan_id,
            "task_type": t1.task_type or "unknown",
            "original_task": original_task,
            "artifact_path": t1.artifact_path,
            "tier1_passed": t1.passed,
            "tier1_reasons": t1.reasons,
            "tier2_score": None,
            "tier2_breakdown": None,
            "tier2_flags": [],
            "tier3_score": None,
            "tier3_model": None,
            "final_score": None,
            "review_status": "pending",
            "plan_steps": len(steps),
            "plan_retries": plan_retries,
            "approval_requested": approval_requested,
            "agents_used": agents_used,
        }

        if not t1.passed:
            log.info(
                "eval.pipeline.tier1_failed",
                task_id=task_id[:8],
                reasons=t1.reasons[:3],
            )
            eval_record["review_status"] = "pending"
            eval_record["final_score"] = 0.0
            await self._save_and_notify(eval_record, synthesised_reply)
            return

        if t1.task_type is None:
            # No spec — skip LLM judge, record as no_eval
            eval_record["review_status"] = "no_eval"
            await self.memory.save_eval_result(eval_record)
            return

        # ── Tier 2: local LLM judge ───────────────────────────────────────
        content = synthesised_reply
        if t1.artifact_path:
            try:
                content = Path(t1.artifact_path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass

        t2_score, t2_breakdown, t2_flags = await self._judge.score_tier2(
            t1.task_type, content
        )
        eval_record["tier2_score"] = t2_score
        eval_record["tier2_breakdown"] = t2_breakdown
        eval_record["tier2_flags"] = t2_flags

        final_score = t2_score

        # ── Tier 3: Gemini judge (borderline only) ────────────────────────
        if needs_tier3(t2_score):
            t3_score, t3_model = await self._judge.score_tier3(t1.task_type, content)
            eval_record["tier3_score"] = t3_score
            eval_record["tier3_model"] = t3_model
            # Average T2 and T3 for final score
            if t3_score > 0:
                final_score = round((t2_score + t3_score) / 2, 2)

        eval_record["final_score"] = round(final_score, 2)

        # ── Determine review_status ───────────────────────────────────────
        if is_auto_approve(final_score) and not t2_flags:
            eval_record["review_status"] = "auto_approved"
        else:
            eval_record["review_status"] = "pending"

        await self._save_and_notify(eval_record, synthesised_reply)

    async def _save_and_notify(self, record: dict, content: str) -> None:
        """Save to DB and emit review event if human review is needed."""
        eval_id = await self.memory.save_eval_result(record)

        review_status = record.get("review_status")
        task_type = record.get("task_type", "unknown")
        final_score = record.get("final_score")

        log.info(
            "eval.pipeline.saved",
            eval_id=eval_id,
            task_type=task_type,
            tier1=record["tier1_passed"],
            final_score=final_score,
            status=review_status,
        )

        # Write a review file for the pending queue
        if review_status == "pending":
            await asyncio.to_thread(
                self._write_review_file, eval_id, record, content
            )
            # Emit event for Discord bridge to post to #eval-queue
            await self.bus.publish(
                "agents:broadcast",
                {
                    "type": "eval.review_needed",
                    "eval_id": eval_id,
                    "task_type": task_type,
                    "original_task": record["original_task"][:200],
                    "final_score": final_score,
                    "tier1_passed": record["tier1_passed"],
                    "tier1_reasons": record["tier1_reasons"][:5],
                    "tier2_flags": record["tier2_flags"][:5],
                    "artifact_path": record.get("artifact_path"),
                    "approval_requested": record.get("approval_requested", False),
                },
            )

    def _write_review_file(self, eval_id: int, record: dict, content: str) -> None:
        """Write a JSON summary to the review_queue/pending/ directory."""
        _REVIEW_PENDING.mkdir(parents=True, exist_ok=True)
        path = _REVIEW_PENDING / f"eval_{eval_id}_{record['task_type']}.json"
        summary = {
            "eval_id": eval_id,
            "task_id": record["task_id"],
            "task_type": record["task_type"],
            "original_task": record["original_task"],
            "artifact_path": record.get("artifact_path"),
            "final_score": record.get("final_score"),
            "tier1_passed": record["tier1_passed"],
            "tier1_reasons": record["tier1_reasons"],
            "tier2_score": record.get("tier2_score"),
            "tier2_flags": record.get("tier2_flags"),
            "tier3_score": record.get("tier3_score"),
            "approval_requested": record.get("approval_requested"),
            "agents_used": record.get("agents_used"),
            "plan_retries": record.get("plan_retries"),
            "content_preview": content[:500],
        }
        path.write_text(json.dumps(summary, indent=2, default=str))
