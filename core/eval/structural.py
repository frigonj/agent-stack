"""
core/eval/structural.py
────────────────────────
Tier 1 evaluation: deterministic structural checks against golden reference specs.

No LLM calls. No network. Runs in-process immediately after a plan completes.

Usage:
    from core.eval.structural import StructuralChecker
    checker = StructuralChecker()
    passed, reasons, artifact_path = await checker.check(plan, result_text)
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# Path to golden reference directory (inside the container at /workspace/eval)
_EVAL_DIR = Path(os.getenv("EVAL_DIR", "/workspace/eval"))
_GOLDEN_DIR = _EVAL_DIR / "golden"

# Maps task_type strings to their golden reference directory name
TASK_TYPE_MAP: dict[str, str] = {
    "arch_doc": "arch_doc",
    "architecture": "arch_doc",
    "document_qa": "arch_doc",
    "research": "research",
    "code_analysis": "code_analysis",
    "code_search": "code_analysis",
}

# Classify a plan's original_task into a task_type
_TASK_TYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(architecture|arch[\s_-]?doc|architectural)\b", re.I), "arch_doc"),
    (re.compile(r"\b(research|look up|find out|what are the latest|web search)\b", re.I), "research"),
    (re.compile(r"\b(find all|where is|what calls|usages of|code search|explain how)\b", re.I), "code_analysis"),
]


def classify_task(original_task: str) -> Optional[str]:
    """Return a task_type string for the given task, or None if unknown."""
    for pattern, task_type in _TASK_TYPE_PATTERNS:
        if pattern.search(original_task):
            return task_type
    return None


@dataclass
class CheckResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    artifact_path: Optional[str] = None
    task_type: Optional[str] = None


class StructuralChecker:
    """
    Runs deterministic checks against golden reference specs.
    One instance is safe to share across plans.
    """

    def _load_spec(self, task_type: str) -> Optional[dict]:
        spec_key = TASK_TYPE_MAP.get(task_type, task_type)
        spec_path = _GOLDEN_DIR / spec_key / "expected_structure.json"
        if not spec_path.exists():
            log.debug("eval.structural.no_spec", task_type=task_type, path=str(spec_path))
            return None
        try:
            return json.loads(spec_path.read_text())
        except Exception as exc:
            log.warning("eval.structural.spec_load_failed", error=str(exc))
            return None

    def _find_artifact(self, spec: dict) -> Optional[str]:
        """Glob for the most recently modified artifact matching the spec pattern."""
        pattern = spec.get("artifact", {}).get("path_pattern")
        if not pattern:
            return None
        matches = glob.glob(pattern)
        if not matches:
            return None
        return max(matches, key=os.path.getmtime)

    def check(
        self,
        original_task: str,
        plan_dict: dict,
        result_text: str = "",
        task_type: Optional[str] = None,
    ) -> CheckResult:
        """
        Run all Tier 1 checks. Returns a CheckResult synchronously.

        Args:
            original_task: The plan's original_task string.
            plan_dict:     plan.to_dict() — used for plan-level checks.
            result_text:   The synthesised final reply text (may be empty).
            task_type:     Override auto-detection if already known.
        """
        if task_type is None:
            task_type = classify_task(original_task)
        if task_type is None:
            return CheckResult(passed=True, reasons=["no_spec_for_task_type"], task_type=None)

        spec = self._load_spec(task_type)
        if spec is None:
            return CheckResult(passed=True, reasons=["no_spec_found"], task_type=task_type)

        failures: list[str] = []
        artifact_path: Optional[str] = None

        # ── Artifact checks ───────────────────────────────────────────────
        artifact_spec = spec.get("artifact", {})
        if artifact_spec.get("path_pattern"):
            artifact_path = self._find_artifact(spec)
            if artifact_path is None:
                failures.append(f"artifact_missing: no file matching {artifact_spec['path_pattern']}")
            else:
                # Size check
                min_bytes = artifact_spec.get("min_size_bytes")
                if min_bytes:
                    size = os.path.getsize(artifact_path)
                    if size < min_bytes:
                        failures.append(f"artifact_too_small: {size}B < {min_bytes}B minimum")

                # Extension check
                required_ext = artifact_spec.get("required_extension")
                if required_ext and not artifact_path.endswith(required_ext):
                    failures.append(f"artifact_wrong_extension: expected {required_ext}")

        # ── Structural content checks (against artifact or result_text) ───
        structural = spec.get("structural_checks", {})
        content_to_check = result_text

        # Prefer artifact content for file-based checks
        if artifact_path and os.path.exists(artifact_path):
            try:
                content_to_check = Path(artifact_path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass

        if content_to_check:
            # Required LaTeX commands
            for cmd in structural.get("required_latex_commands", []):
                if cmd not in content_to_check:
                    failures.append(f"missing_latex_cmd: {cmd}")

            # Required sections
            for section in structural.get("required_sections", []):
                if section.lower() not in content_to_check.lower():
                    failures.append(f"missing_section: {section}")

            # Required agents mentioned
            for agent in structural.get("required_agents_mentioned", []):
                if agent not in content_to_check:
                    failures.append(f"agent_not_mentioned: {agent}")

            # Required terms
            for term in structural.get("required_terms", []):
                if term.lower() not in content_to_check.lower():
                    failures.append(f"missing_term: {term}")

            # Forbidden content
            for bad in structural.get("must_not_contain", []):
                if bad.lower() in content_to_check.lower():
                    failures.append(f"forbidden_content: {bad}")

            # must_contain_one_of (for code_analysis)
            one_of = structural.get("must_contain_one_of", [])
            if one_of and not any(kw in content_to_check for kw in one_of):
                failures.append(f"must_contain_one_of: none of {one_of[:3]}... found")

            # Min length
            min_len = structural.get("min_response_length_chars")
            if min_len and len(content_to_check) < min_len:
                failures.append(f"response_too_short: {len(content_to_check)} < {min_len} chars")

            # Min sources
            min_sources = structural.get("min_sources_cited")
            if min_sources:
                source_patterns = structural.get("source_patterns", ["http://", "https://"])
                found = sum(1 for p in source_patterns if p in content_to_check)
                if found < min_sources:
                    failures.append(f"insufficient_sources: found ~{found}, need {min_sources}")

        # ── PDF existence check ───────────────────────────────────────────
        if structural.get("pdf_must_exist") and artifact_path:
            pdf_path = re.sub(r"\.tex$", ".pdf", artifact_path)
            if not os.path.exists(pdf_path):
                failures.append(f"pdf_missing: {pdf_path} not found")

        # ── Plan-level checks ─────────────────────────────────────────────
        plan_checks = spec.get("plan_checks", {})
        steps = plan_dict.get("steps", [])

        if plan_checks.get("must_not_request_approval"):
            if plan_dict.get("approval_requested", False):
                failures.append("plan_requested_user_approval")

        max_steps = plan_checks.get("max_steps")
        if max_steps and len(steps) > max_steps:
            failures.append(f"plan_too_many_steps: {len(steps)} > {max_steps}")

        expected_agents = plan_checks.get("expected_agents_used", [])
        agents_used = {s.get("agent") for s in steps if s.get("status") == "done"}
        for expected in expected_agents:
            if expected not in agents_used:
                failures.append(f"expected_agent_not_used: {expected}")

        passed = len(failures) == 0
        log.info(
            "eval.structural.result",
            task_type=task_type,
            passed=passed,
            failure_count=len(failures),
        )
        return CheckResult(
            passed=passed,
            reasons=failures,
            artifact_path=artifact_path,
            task_type=task_type,
        )
