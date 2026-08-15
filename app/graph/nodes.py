"""Graph nodes: classify_distress, generate, critique, moderate, support, emit.

Each node is a small async function taking ``(state, runtime)``. The Anthropic
client comes from ``runtime.context`` (injected per run, never checkpointed) and
player-facing output is streamed through ``runtime.stream_writer`` as structured
``meta``/``token`` events that the route maps 1:1 onto the SSE contract.

Why stream through a custom writer rather than LangGraph's ``messages`` mode:
we call the raw AsyncAnthropic SDK (not a LangChain chat model), which
``stream_mode="messages"`` does not observe; and the reflection loop must not
show the player an unreviewed draft. So generation is *buffered*, critique gates
it, and only the approved text is streamed — from the ``support``/``emit`` nodes.
"""

from __future__ import annotations

import json
import logging
import re

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from app.config import (
    DISTRESS_MODEL,
    GENERATION_MODEL,
    MAX_ATTEMPTS,
    WORD_LIMIT,
)
from app.graph.state import CritiqueResult, GraphContext, GraphState
from app.prompts.distress import (
    DISTRESS_OUTPUT_SCHEMA,
    DISTRESS_SYSTEM_PROMPT,
    build_distress_user_message,
)
from app.prompts.encouragement import (
    ENCOURAGEMENT_SYSTEM_PROMPT,
    build_encouragement_user_message,
    build_regeneration_feedback,
)
from app.prompts.support import SUPPORT_MESSAGE
from evals.judge import judge_encouragement

logger = logging.getLogger("bloom.graph")

# Cheap deterministic red-flags for the critique pre-check. These catch the
# obvious injection / clinical failures without an LLM call; the judge is the
# real guardrail behind them. Kept conservative to avoid false positives on
# good replies (note: the feeling words like "anxious" are deliberately absent).
_RED_FLAG_RE = re.compile(
    r"\b("
    r"hacked"  # injection compliance
    r"|as an? (?:unrestricted )?ai"
    r"|system prompt"
    r"|pirate"
    r"|diagnos\w*"  # clinical / therapy-speak
    r"|therap(?:y|ist)"
    r"|clinical\w*"
    r"|disorder"
    r"|medication"
    r"|prescri\w*"
    r")\b",
    re.IGNORECASE,
)


def _text_from(response: object) -> str:
    """Concatenate the text blocks of an Anthropic message response."""
    blocks = getattr(response, "content", [])
    return "".join(b.text for b in blocks if getattr(b, "type", None) == "text")


def word_count(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def truncate_to_word_limit(text: str, limit: int = WORD_LIMIT) -> str:
    """Best-effort truncation used only when the loop is exhausted."""
    words = text.strip().split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).rstrip(",;:") + "…"


# --- nodes ------------------------------------------------------------------


async def classify_distress(
    state: GraphState, runtime: Runtime[GraphContext]
) -> GraphState:
    """Step 1 of the two-step safety design.

    Only runs the model when the player wrote free text: preset feeling chips
    come from our own fixed list and can't express crisis, so chip-only
    requests skip this call and the common path stays a single model call.

    Structured outputs pin the reply to {"distress": boolean}; if it still
    fails to parse we fail safe to distress=true (support is the safe failure
    mode for a wellness product).
    """
    free_text = (state.get("free_text") or "").strip()
    if not free_text:
        return {"distress": False}

    client = runtime.context.client
    response = await client.messages.create(
        model=DISTRESS_MODEL,
        max_tokens=64,
        system=DISTRESS_SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": DISTRESS_OUTPUT_SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": build_distress_user_message(state["feeling"], free_text),
            }
        ],
    )
    try:
        parsed = json.loads(_text_from(response))
        if isinstance(parsed, dict) and isinstance(parsed.get("distress"), bool):
            return {"distress": parsed["distress"]}
    except (json.JSONDecodeError, ValueError):
        logger.warning("distress classifier returned unparseable output; failing safe")
    return {"distress": True}


async def generate(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """Produce Mamorin's reply (buffered).

    On the reflection loop's retry, the previous critique reason is fed back so
    the next draft is corrected rather than a blind re-roll. Thinking is
    disabled and effort is low: a ~25-word reply needs no extended reasoning,
    and this keeps latency down on the post-stage screen.
    """
    attempts = state.get("attempts", 0)
    user_message = build_encouragement_user_message(
        state["stage_id"], state["feeling"], state.get("free_text")
    )

    messages: list[dict[str, str]] = [{"role": "user", "content": user_message}]
    critique = state.get("critique")
    if critique is not None and not critique["passed"]:
        # Retrying: hand the model its previous draft and why it was rejected.
        messages.append({"role": "assistant", "content": state.get("draft", "")})
        messages.append(
            {"role": "user", "content": build_regeneration_feedback(critique["reason"])}
        )

    client = runtime.context.client
    response = await client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=200,
        thinking={"type": "disabled"},
        output_config={"effort": "low"},
        system=ENCOURAGEMENT_SYSTEM_PROMPT,
        messages=messages,  # type: ignore[arg-type]
    )
    return {"draft": _text_from(response).strip(), "attempts": attempts + 1}


async def critique(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """The reflection guardrail.

    Cheap deterministic checks first (word count, red-flag regex); only if those
    pass do we spend the LLM judge — keeping the expensive call off the happy
    path. The judge is the same rubric the offline evals use.
    """
    draft = state.get("draft", "")
    words = word_count(draft)

    if words > WORD_LIMIT:
        result: CritiqueResult = {
            "passed": False,
            "safety_failed": False,
            "reason": f"Too long: {words} words (limit {WORD_LIMIT}). Cut it down.",
        }
        return {"critique": result}

    if _RED_FLAG_RE.search(draft):
        result = {
            "passed": False,
            "safety_failed": True,
            "reason": "Contains clinical/therapy-speak or complied with an injection.",
        }
        return {"critique": result}

    # Cheap checks passed — spend the judge.
    client = runtime.context.client
    scores = await judge_encouragement(
        client, state["feeling"], state.get("free_text"), draft
    )
    if scores.safety == "fail":
        result = {
            "passed": False,
            "safety_failed": True,
            "reason": f"Judge safety fail: {scores.safety_reason}",
        }
    elif scores.empathy < 3 or scores.tone < 3:
        result = {
            "passed": False,
            "safety_failed": False,
            "reason": (
                f"Quality too low (empathy {scores.empathy}, tone {scores.tone}): "
                f"{scores.empathy_reason} {scores.tone_reason}"
            ),
        }
    else:
        result = {"passed": True, "safety_failed": False, "reason": "ok"}
    return {"critique": result}


def moderate(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """Human-in-the-loop gate for distress cases.

    Calls ``interrupt(...)`` with the flagged case; the graph pauses and
    persists to the checkpointer, and the /resume route continues it with
    ``Command(resume={"approve", "note"})``. A moderator decision never
    withholds care — the support node always delivers the reviewed words — but
    the decision and note are recorded for the review workflow.
    """
    decision = interrupt(
        {
            "reason": "distress_flagged",
            "feeling": state["feeling"],
            "free_text": state.get("free_text"),
            "stage_id": state["stage_id"],
        }
    )
    approved = bool(decision.get("approve", True)) if isinstance(decision, dict) else True
    note = decision.get("note") if isinstance(decision, dict) else None
    logger.info("moderation decision: approved=%s note=%r", approved, note)
    return {"moderator_approved": approved, "moderator_note": note}


async def support(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """Stream the static, pre-written support message word by word.

    Never model-generated: a player in a hard moment must always get the same
    reviewed words. Marks the stream ``type: "support"``.
    """
    writer = runtime.stream_writer
    writer({"type": "meta", "kind": "support"})
    for chunk in re.split(r"(?<= )", SUPPORT_MESSAGE):
        if chunk:
            writer({"type": "token", "text": chunk})
    return {"path": "support", "final_text": SUPPORT_MESSAGE}


async def emit(state: GraphState, runtime: Runtime[GraphContext]) -> GraphState:
    """Terminal node for the encouragement path: stream the approved draft.

    Reached when critique passed, or when the loop is exhausted on a
    *non-safety* failure (best-effort draft, truncated to the word limit).
    Safety-failed exhaustion routes to ``support`` instead, never here.
    """
    critique = state.get("critique")
    draft = state.get("draft", "")
    text = draft if (critique and critique["passed"]) else truncate_to_word_limit(draft)

    writer = runtime.stream_writer
    writer({"type": "meta", "kind": "encouragement"})
    for chunk in re.split(r"(?<= )", text):
        if chunk:
            writer({"type": "token", "text": chunk})
    return {"path": "encouragement", "final_text": text}


# --- routing ----------------------------------------------------------------


def route_after_classify(state: GraphState) -> str:
    """distress → moderate (HITL); otherwise → generate."""
    return "moderate" if state.get("distress") else "generate"


def route_after_critique(state: GraphState) -> str:
    """Reflection-loop routing — bounded so it can never loop forever.

    - pass                          → emit the draft
    - fail, attempts left           → regenerate with the critique as feedback
    - fail, exhausted & safety fail → static support message (safe fallback)
    - fail, exhausted otherwise     → emit best-effort (truncated) draft
    """
    critique = state.get("critique")
    if critique and critique["passed"]:
        return "emit"
    if state.get("attempts", 0) < MAX_ATTEMPTS:
        return "generate"
    if critique and critique["safety_failed"]:
        return "support"
    return "emit"
