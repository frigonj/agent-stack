"""
core/eval/llm_judge.py
───────────────────────
Tier 2 evaluation: local Qwen LLM judge using structured rubrics.
Tier 3 evaluation: Gemini Flash free-tier judge for borderline scores.

The judge is called only when Tier 1 passes.
Tier 3 is called only when Tier 2 score is in the borderline range [4, 7].

Usage:
    from core.eval.llm_judge import LLMJudge
    judge = LLMJudge()
    score, breakdown, flags = await judge.score_tier2(task_type, content)
    if needs_tier3(score):
        score3, model = await judge.score_tier3(task_type, content)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import httpx
import structlog

log = structlog.get_logger(__name__)

_EVAL_DIR = Path(os.getenv("EVAL_DIR", "/workspace/eval"))
_RUBRICS_DIR = _EVAL_DIR / "rubrics"

# LM Studio (local Qwen) endpoint — same as agents use
_LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://host.docker.internal:1234")
_LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen3-vl-8b")

# Gemini Flash free tier
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
_GEMINI_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.0-flash")
_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    f"/{_GEMINI_MODEL}:generateContent"
)

# Score band that triggers Tier 3
_TIER3_LOW = float(os.getenv("EVAL_TIER3_LOW", "4"))
_TIER3_HIGH = float(os.getenv("EVAL_TIER3_HIGH", "7"))

# Auto-approve threshold (no human review needed)
_AUTO_APPROVE_ABOVE = float(os.getenv("EVAL_AUTO_APPROVE", "8"))


def needs_tier3(tier2_score: float) -> bool:
    return _TIER3_LOW <= tier2_score <= _TIER3_HIGH and bool(_GEMINI_API_KEY)


def is_auto_approve(score: float) -> bool:
    return score > _AUTO_APPROVE_ABOVE


def _build_judge_prompt(rubric: dict, content: str) -> str:
    criteria_text = "\n".join(
        f"- {c['id']} (weight {c['weight']}): {c['description']}"
        for c in rubric.get("criteria", [])
    )
    fmt = rubric.get("response_format", {})
    return (
        f"{rubric['instructions']}\n\n"
        f"## Scoring criteria\n{criteria_text}\n\n"
        f"## Output to evaluate\n```\n{content[:6000]}\n```\n\n"
        f"## Response format (JSON only, no prose)\n"
        f"{json.dumps(fmt, indent=2)}"
    )


def _parse_judge_response(raw: str, rubric: dict) -> tuple[float, dict, list[str]]:
    """
    Parse the judge's JSON response. Returns (overall_score, breakdown, flags).
    Falls back gracefully on parse errors.
    """
    # Strip markdown fences if present
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        # Try to extract JSON object from the text
        import re

        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return 0.0, {}, ["parse_failed"]
        else:
            return 0.0, {}, ["parse_failed"]

    # Compute weighted overall if not provided or zero
    scores = data.get("scores", {})
    overall = float(data.get("overall", 0))
    if overall == 0 and scores:
        criteria = {c["id"]: c["weight"] for c in rubric.get("criteria", [])}
        overall = sum(float(scores.get(cid, 0)) * w for cid, w in criteria.items())

    # Clamp to [0, 10]
    overall = max(0.0, min(10.0, overall))
    flags = data.get("flags", [])
    return round(overall, 2), scores, flags


class LLMJudge:
    """Stateless judge. Safe to instantiate once and reuse."""

    def _load_rubric(self, task_type: str) -> Optional[dict]:
        path = _RUBRICS_DIR / f"{task_type}.json"
        if not path.exists():
            # Fall back to arch_doc rubric for unknown doc tasks
            path = _RUBRICS_DIR / "arch_doc.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            log.warning("eval.llm_judge.rubric_load_failed", error=str(exc))
            return None

    async def score_tier2(
        self, task_type: str, content: str
    ) -> tuple[float, dict, list[str]]:
        """
        Call local Qwen to score the output.
        Returns (score 0-10, per-criterion breakdown, flags).
        Returns (0.0, {}, ["llm_unavailable"]) on failure.
        """
        rubric = self._load_rubric(task_type)
        if rubric is None:
            return 0.0, {}, ["no_rubric"]

        prompt = _build_judge_prompt(rubric, content)
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{_LM_STUDIO_URL}/v1/chat/completions",
                    json={
                        "model": _LM_STUDIO_MODEL,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a strict quality evaluator. "
                                    "Output ONLY valid JSON matching the format provided. "
                                    "No preamble, no markdown, no explanation outside the JSON."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.0,
                        "max_tokens": 512,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            log.warning("eval.llm_judge.tier2_failed", error=str(exc))
            return 0.0, {}, ["llm_unavailable"]

        score, breakdown, flags = _parse_judge_response(raw, rubric)
        log.info("eval.llm_judge.tier2", task_type=task_type, score=score, flags=flags)
        return score, breakdown, flags

    async def score_tier3(self, task_type: str, content: str) -> tuple[float, str]:
        """
        Call Gemini Flash to score the output.
        Returns (score 0-10, model_name).
        Returns (0.0, "gemini_unavailable") on failure.
        """
        if not _GEMINI_API_KEY:
            return 0.0, "gemini_no_api_key"

        rubric = self._load_rubric(task_type)
        if rubric is None:
            return 0.0, "no_rubric"

        prompt = _build_judge_prompt(rubric, content)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    _GEMINI_URL,
                    params={"key": _GEMINI_API_KEY},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.0,
                            "maxOutputTokens": 512,
                            "responseMimeType": "application/json",
                        },
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:
            log.warning("eval.llm_judge.tier3_failed", error=str(exc))
            return 0.0, "gemini_unavailable"

        score, _, _ = _parse_judge_response(raw, rubric)
        log.info(
            "eval.llm_judge.tier3",
            task_type=task_type,
            score=score,
            model=_GEMINI_MODEL,
        )
        return score, _GEMINI_MODEL
