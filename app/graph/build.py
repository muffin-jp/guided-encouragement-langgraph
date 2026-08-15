"""StateGraph wiring and compilation.

The graph is compiled once at app startup (FastAPI lifespan) with a checkpointer
and reused for every request. The checkpointer is what makes the moderation
interrupt durable — the graph pauses, persists, and resumes on the /resume call.

    START
      → classify_distress
          ├─ distress → moderate ─(interrupt / resume)→ support → END
          └─ else     → generate → critique
                                      ├─ pass                     → emit  → END
                                      ├─ fail & attempts left     → generate  (loop)
                                      ├─ fail, spent, safety fail → support → END
                                      └─ fail, spent, otherwise   → emit  → END
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

from app.graph.nodes import (
    classify_distress,
    critique,
    emit,
    generate,
    moderate,
    route_after_classify,
    route_after_critique,
    support,
)
from app.graph.state import GraphContext, GraphState


def build_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Wire and compile the graph.

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
    builder.add_node("moderate", moderate)
    builder.add_node("support", support)
    builder.add_node("emit", emit)

    builder.add_edge(START, "classify_distress")
    builder.add_conditional_edges(
        "classify_distress",
        route_after_classify,
        {"moderate": "moderate", "generate": "generate"},
    )
    builder.add_edge("moderate", "support")
    builder.add_edge("generate", "critique")
    builder.add_conditional_edges(
        "critique",
        route_after_critique,
        {"generate": "generate", "emit": "emit", "support": "support"},
    )
    builder.add_edge("support", END)
    builder.add_edge("emit", END)

    return builder.compile(checkpointer=checkpointer)
