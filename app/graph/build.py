"""StateGraph wiring and compilation.

The graph is compiled once at app startup (FastAPI lifespan) with a checkpointer
and reused for every request. The checkpointer is what makes the moderation
interrupt durable — the graph pauses, persists, and resumes on the /resume call.

    START
      → classify_distress
          ├─ distress → support ─(stream reviewed words immediately)
          │              ├─ moderate ─(interrupt/resume, non-blocking)→ END   [moderation on]
          │              └─ END                                                [moderation off]
          └─ else     → generate → critique
                                     ├─ pass                     → emit  → END
                                     ├─ fail & attempts left     → generate  (loop)
                                     ├─ fail, spent, safety fail → support → …
                                     └─ fail, spent, otherwise   → emit  → END

Support always streams *before* any human review, so a distressed player never
waits on a moderator. When ``MODERATION_ENABLED`` is set, the ``moderate``
interrupt runs after support and only records a decision — it never withholds
care. With moderation off, the distress path is simply ``support → END``.
"""

# LangGraph 1.x ships incomplete type info for the StateGraph builder — its
# add_node / compile overloads resolve to `_Node[Unknown]`, so pyright reports
# their member types as partially unknown at this construction boundary only.
# Our own code stays fully strict; this narrowly relaxes that one library gap.
# pyright: reportUnknownMemberType=false
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import MODERATION_ENABLED
from app.graph.nodes import (
    classify_distress,
    critique,
    emit,
    generate,
    moderate,
    route_after_classify,
    route_after_critique,
    route_after_support,
    support,
)
from app.graph.state import GraphContext, GraphState


def build_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    *,
    enable_moderation: bool = MODERATION_ENABLED,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Wire and compile the graph.

    ``enable_moderation`` appends the non-blocking human-in-the-loop review
    *after* the support node on the distress path (default from config). Tests
    pass it explicitly to exercise the interrupt/resume machinery; evals can
    leave it off, since routing and quality never depend on it.

    For the demo a MemorySaver checkpointer is fine. In production this is the
    single seam to swap for an async Postgres checkpointer
    (``langgraph.checkpoint.postgres.aio.AsyncPostgresSaver``) so paused
    moderation cases survive restarts and span instances — nothing else in the
    graph changes.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(GraphState, context_schema=GraphContext)

    builder.add_node("classify_distress", classify_distress)
    builder.add_node("generate", generate)
    builder.add_node("critique", critique)
    builder.add_node("support", support)
    builder.add_node("emit", emit)

    builder.add_edge(START, "classify_distress")

    # Distress always streams the reviewed support words first.
    builder.add_conditional_edges(
        "classify_distress",
        route_after_classify,
        {"distress": "support", "generate": "generate"},
    )

    builder.add_edge("generate", "critique")
    builder.add_conditional_edges(
        "critique",
        route_after_critique,
        {"generate": "generate", "emit": "emit", "support": "support"},
    )
    builder.add_edge("emit", END)

    if enable_moderation:
        # Human review runs *after* support has been delivered — non-blocking.
        # Only genuine distress cases are routed into it; a support message
        # reached via the encouragement safety-fallback just ends.
        builder.add_node("moderate", moderate)
        builder.add_conditional_edges(
            "support",
            route_after_support,
            {"moderate": "moderate", "end": END},
        )
        builder.add_edge("moderate", END)
    else:
        builder.add_edge("support", END)

    return builder.compile(checkpointer=checkpointer)
