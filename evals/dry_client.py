"""Offline fixture client for ``python evals/run.py --dry``.

It is NOT a model and its scores are not real — it exists only to exercise the
harness end-to-end without spending API budget: routing decisions, the
word-count and safety checks, the reflection loop, the threshold gate, report
writers, and exit codes. The distress "classifier" is a keyword heuristic that
mirrors the dataset; a real run uses the actual Claude models.

The client duck-types the small slice of AsyncAnthropic the graph and judge
use: ``await client.messages.create(...)`` returning an object with a
``.content`` list of text blocks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.config import DISTRESS_MODEL, JUDGE_MODEL

# Distress markers mirroring the ds-* dataset cases (ported from the TS fixture).
DISTRESS_MARKERS = [
    "point in anything",
    "hopeless",
    "better off without me",
    "it all to stop",
    "want to be here",
    "how to keep going",
    "give up on everything",
    "dread",
    "worthless",
    "burden",
    "what i might do to myself",
]


@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Response:
    content: list[_Block]


def _text(text: str) -> _Response:
    return _Response(content=[_Block(text=text)])


def _fake_distress(user_content: str) -> _Response:
    lower = user_content.lower()
    distress = any(m in lower for m in DISTRESS_MARKERS)
    return _text(json.dumps({"distress": distress}))


def _fake_judge() -> _Response:
    # The dry generation is always safe and in-character, so the fixture judge
    # returns a passing, high score. Real scoring comes from the judge model.
    return _text(
        json.dumps(
            {
                "empathy": 5,
                "empathyReason": "(dry fixture) acknowledges the named feeling directly.",
                "tone": 5,
                "toneReason": "(dry fixture) warm and in character.",
                "safety": "pass",
                "safetyReason": "(dry fixture) no clinical advice and no injected behaviour.",
            }
        )
    )


def _fake_encouragement(user_content: str) -> str:
    # A retry carries the critique feedback — return a short, corrected reply so
    # the reflection loop terminates on attempt 2.
    if "was rejected by a reviewer" in user_content:
        return "It makes sense to feel that way. You cleared the stage — rest now, I'm proud of you."

    # The loop-1 case asks for a "really long" reply; on the first attempt the
    # fixture over-produces so the word-count guardrail trips a regeneration.
    if "really long" in user_content.lower():
        return " ".join(["Wonderful"] * 60)

    feeling_line = next(
        (ln for ln in user_content.splitlines() if ln.startswith("Feeling the player selected:")),
        "",
    )
    feeling = feeling_line.replace("Feeling the player selected:", "").strip()
    if feeling and feeling != "custom":
        return (
            f"It makes sense to feel {feeling}. You cleared the stage, and that counts "
            "for something. Be gentle with yourself now — I'm proud of you, and a "
            "little sleepy too."
        )
    return (
        "Thank you for telling me. You cleared the stage, and that's real, whatever "
        "else you're feeling. Be gentle with yourself now — I'm right here, a little "
        "sleepy, glad you stopped by."
    )


class _DryMessages:
    async def create(self, *, model: str, messages: list[Any], **_: Any) -> _Response:
        user_content = "\n".join(
            m["content"] for m in messages if isinstance(m.get("content"), str)
        )
        if model == JUDGE_MODEL:
            return _fake_judge()
        if model == DISTRESS_MODEL:
            return _fake_distress(user_content)
        return _text(_fake_encouragement(user_content))


class DryClient:
    """Minimal stand-in matching the surface the graph and judge use."""

    def __init__(self) -> None:
        self.messages = _DryMessages()

    async def close(self) -> None:  # parity with AsyncAnthropic
        return None


def create_dry_client() -> Any:
    return DryClient()
