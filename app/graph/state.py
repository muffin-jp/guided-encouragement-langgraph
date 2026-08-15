"""Graph state and runtime context.

State is an explicit TypedDict rather than a bag of loose values — this code is
meant to be read in review, so every field the nodes touch is named and typed
here. The Anthropic client is *not* state; it is a runtime dependency injected
through LangGraph's typed context (see GraphContext) so it never gets
checkpointed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Required, TypedDict

from anthropic import AsyncAnthropic

from app.sse import ResponseKind


@dataclass
class GraphContext:
    """Per-run dependencies, injected via ``astream(..., context=...)``.

    Passing the client here (not in state) keeps credentials out of the
    checkpointer and lets tests inject a mock without patching module globals.
    """

    client: AsyncAnthropic


class CritiqueResult(TypedDict):
    """Output of the critique node: the reflection guardrail's verdict."""

    passed: bool
    # A safety failure is special: on exhaustion it falls back to the static
    # support message rather than shipping a truncated draft.
    safety_failed: bool
    reason: str


class GraphState(TypedDict, total=False):
    """The full graph state.

    Marked ``total=False`` because nodes fill it in incrementally; the initial
    input only carries the request fields.
    """

    # --- request (always set at invocation; the rest is filled in by nodes) ---
    stage_id: Required[str]
    feeling: Required[str]
    free_text: Required[str | None]

    # --- classification ---
    distress: bool | None

    # --- generation / reflection loop ---
    draft: str
    critique: CritiqueResult | None
    attempts: int

    # --- moderation (HITL) ---
    moderator_approved: bool | None
    moderator_note: str | None

    # --- outcome ---
    path: ResponseKind
    # The reviewed text actually streamed to the player (approved draft,
    # truncated best-effort, or the static support message).
    final_text: str
