"""Shared eval data types.

Ported from the TypeScript evals/types.ts. Kept separate from the runner so the
judge, thresholds, and report modules can all import them without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.sse import ResponseKind

ExpectedPath = ResponseKind  # "encouragement" | "support"


@dataclass
class EvalCase:
    id: str
    feeling: str
    expected_path: ExpectedPath
    category: str
    free_text: str | None = None


@dataclass
class JudgeScores:
    empathy: int  # 1-5
    empathy_reason: str
    tone: int  # 1-5
    tone_reason: str
    safety: Literal["pass", "fail"]
    safety_reason: str


@dataclass
class CaseResult:
    id: str
    category: str
    feeling: str
    free_text: str | None
    expected_path: ExpectedPath
    actual_path: ExpectedPath
    path_correct: bool
    text: str
    word_count: int
    # None on the support path (word limit only applies to generated text).
    word_limit_ok: bool | None
    non_empty: bool
    # None on the support path (the static message is not judged).
    judge: JudgeScores | None
    # Set when the pipeline or judge errored irrecoverably.
    error: str | None = None
