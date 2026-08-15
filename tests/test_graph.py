"""Graph routing and bounded-loop termination, with the Anthropic client mocked.

These are the safety-critical invariants: the classifier is skipped for
chip-only requests, distress routes to support, ordinary frustration does not,
and the reflection loop always terminates at MAX_ATTEMPTS — it can never loop
forever.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

import pytest
from anthropic import AsyncAnthropic
from langgraph.types import Command

from app.config import DISTRESS_MODEL, GENERATION_MODEL, MAX_ATTEMPTS
from app.graph.build import build_graph
from app.graph.state import GraphContext
from tests.fakes import FakeClient

pytestmark = pytest.mark.asyncio


async def _run(
    client: FakeClient,
    feeling: str,
    free_text: str | None,
    *,
    enable_moderation: bool = False,
) -> dict[str, Any]:
    """Invoke the graph once, auto-resuming a moderation interrupt if it pauses."""
    graph: Any = build_graph(enable_moderation=enable_moderation)
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    context = GraphContext(client=cast(AsyncAnthropic, client))
    result = await graph.ainvoke(
        {"stage_id": "s1", "feeling": feeling, "free_text": free_text, "attempts": 0},
        config=config,
        context=context,
    )
    result["_interrupted"] = "__interrupt__" in result
    if result["_interrupted"]:
        result = {
            **await graph.ainvoke(
                Command(resume={"approve": True, "note": None}),
                config=config,
                context=context,
            ),
            "_interrupted": True,
        }
    return result


async def test_chip_only_skips_the_classifier() -> None:
    client = FakeClient(distress=False)
    result = await _run(client, "proud", None)
    assert client.calls_to(DISTRESS_MODEL) == 0  # no free text -> no classifier call
    assert result["path"] == "encouragement"
    assert result["distress"] is False


async def test_free_text_runs_the_classifier() -> None:
    client = FakeClient(distress=False, drafts=["It makes sense to feel frustrated. Rest now."])
    await _run(client, "frustrated", "took me forever but I got there")
    assert client.calls_to(DISTRESS_MODEL) == 1


async def test_distress_streams_support_immediately_by_default() -> None:
    # Default: a distressed player never waits on a human — support streams
    # straight away, with no interrupt.
    client = FakeClient(distress=True)
    result = await _run(client, "custom", "I feel hopeless and alone")
    assert result["_interrupted"] is False  # no pause on the live path
    assert result["path"] == "support"
    assert result["final_text"].startswith("Thank you for telling me")
    # The static support message is never model-generated.
    assert client.calls_to(GENERATION_MODEL) == 0


async def test_distress_pauses_for_moderation_when_enabled() -> None:
    # With the HITL gate enabled, distress pauses at the interrupt and is
    # continued via resume; the delivered words are still the static support ones.
    client = FakeClient(distress=True)
    result = await _run(client, "custom", "I feel hopeless and alone", enable_moderation=True)
    assert result["_interrupted"] is True  # paused for human moderation
    assert result["path"] == "support"
    assert result["final_text"].startswith("Thank you for telling me")
    assert client.calls_to(GENERATION_MODEL) == 0


async def test_ordinary_frustration_is_not_distress() -> None:
    client = FakeClient(distress=False, drafts=["It makes sense to feel frustrated. Rest now."])
    result = await _run(client, "frustrated", "this stage is so annoying, I almost quit")
    assert result["path"] == "encouragement"
    assert result["_interrupted"] is False


async def test_passing_critique_emits_the_draft() -> None:
    draft = "It makes sense to feel proud. You cleared the stage — rest now, I'm proud of you."
    client = FakeClient(distress=False, drafts=[draft])
    result = await _run(client, "proud", None)
    assert result["path"] == "encouragement"
    assert result["final_text"] == draft
    assert result["attempts"] == 1  # passed on the first attempt, no loop


async def test_over_length_draft_triggers_one_regeneration() -> None:
    long_draft = " ".join(["word"] * 50)  # over the 40-word ceiling
    good_draft = "It makes sense to feel tired. Rest now, I'm proud of you."
    client = FakeClient(distress=False, drafts=[long_draft, good_draft])
    result = await _run(client, "tired", None)
    assert client.calls_to(GENERATION_MODEL) == 2  # looped exactly once
    assert result["attempts"] == 2
    assert result["final_text"] == good_draft


async def test_persistent_safety_failure_terminates_at_max_attempts() -> None:
    # The judge fails safety on every draft; the loop must stop and fall back to
    # the static support message rather than looping forever.
    client = FakeClient(distress=False, judge_safety="fail", drafts=["A plausible-looking reply."])
    result = await _run(client, "proud", None)
    assert client.calls_to(GENERATION_MODEL) == MAX_ATTEMPTS  # bounded — cannot loop forever
    assert result["attempts"] == MAX_ATTEMPTS
    assert result["path"] == "support"  # safety-fail exhaustion -> safe fallback


async def test_persistent_quality_failure_emits_best_effort() -> None:
    # Non-safety (low empathy/tone) exhaustion emits the best-effort draft,
    # truncated to the word limit — still bounded.
    client = FakeClient(
        distress=False, judge_empathy=1, judge_tone=1, drafts=["A dull generic reply."]
    )
    result = await _run(client, "proud", None)
    assert client.calls_to(GENERATION_MODEL) == MAX_ATTEMPTS
    assert result["path"] == "encouragement"
    assert result["final_text"]  # non-empty best effort


async def test_unparseable_distress_fails_safe_to_support() -> None:
    # A classifier reply that cannot be parsed must route to support (the safe
    # failure mode), not to generation.
    client = FakeClient(distress=False)

    async def bad_create(*, model: str, **_: Any) -> Any:
        from tests.fakes import Block, Response

        client.calls.append(model)
        if model == DISTRESS_MODEL:
            return Response([Block("not json at all")])
        raise AssertionError("should not reach generation on a fail-safe")

    client.messages.create = bad_create  # type: ignore[assignment]
    result = await _run(client, "custom", "ambiguous note here")
    assert result["path"] == "support"
