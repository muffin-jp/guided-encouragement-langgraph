"""A configurable fake AsyncAnthropic client for graph tests.

Records which models were called so tests can assert e.g. that the distress
classifier is skipped on chip-only requests, and lets each test script the
generation drafts, distress verdict, and judge scores it needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.config import DISTRESS_MODEL, GENERATION_MODEL, JUDGE_MODEL
from app.graph.state import Passage


@dataclass
class FakeRetriever:
    """Duck-types the ``Retriever.retrieve`` surface for graph tests.

    Records call count so a test can assert the distress path never retrieves and
    that the reflection loop retrieves only once. Set ``raises=True`` to exercise
    the retrieve node's fail-open path.
    """

    passages: list[Passage] = field(
        default_factory=lambda: [
            Passage(id="p1", kind="technique", text="honour the effort", source="test"),
            Passage(id="p2", kind="phrasing", text="rest now, you did enough", source="test"),
        ]
    )
    raises: bool = False
    calls: int = 0

    async def retrieve(self, feeling: str, free_text: str | None, *, k: int = 3) -> list[Passage]:
        self.calls += 1
        if self.raises:
            raise RuntimeError("retriever boom")
        return list(self.passages[:k])


@dataclass
class Block:
    text: str
    type: str = "text"


@dataclass
class Response:
    content: list[Block]


@dataclass
class FakeClient:
    """Duck-types the ``client.messages.create`` surface the graph + judge use."""

    distress: bool = False
    judge_safety: str = "pass"
    judge_empathy: int = 5
    judge_tone: int = 5
    # Successive generation drafts; the last is reused if the loop asks for more.
    drafts: list[str] = field(default_factory=lambda: ["It makes sense to feel proud."])
    _draft_i: int = 0
    calls: list[str] = field(init=False)
    messages: _Messages = field(init=False)

    def __post_init__(self) -> None:
        self.calls = []
        self.messages = _Messages(self)

    def next_draft(self) -> str:
        draft = self.drafts[min(self._draft_i, len(self.drafts) - 1)]
        self._draft_i += 1
        return draft

    def calls_to(self, model: str) -> int:
        return self.calls.count(model)

    async def close(self) -> None:
        return None


class _Messages:
    def __init__(self, outer: FakeClient) -> None:
        self._outer = outer

    async def create(self, *, model: str, **_: Any) -> Response:
        o = self._outer
        o.calls.append(model)
        if model == DISTRESS_MODEL:
            return Response([Block(json.dumps({"distress": o.distress}))])
        if model == JUDGE_MODEL:
            return Response(
                [
                    Block(
                        json.dumps(
                            {
                                "empathy": o.judge_empathy,
                                "empathyReason": "r",
                                "tone": o.judge_tone,
                                "toneReason": "r",
                                "safety": o.judge_safety,
                                "safetyReason": "r",
                            }
                        )
                    )
                ]
            )
        if model == GENERATION_MODEL:
            return Response([Block(o.next_draft())])
        raise AssertionError(f"unexpected model {model}")
