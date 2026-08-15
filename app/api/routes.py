"""HTTP routes: POST /api/encourage and its /resume continuation.

The route is a thin translation layer. It drives the compiled graph with
``astream(stream_mode=["custom", "updates"])`` and maps what the graph emits
onto the existing SSE wire contract (see app/sse.py):

- custom ``meta``/``token`` events  → ``event: meta`` / ``event: token``
- a graph pause (``__interrupt__``)  → an ``awaiting_moderation`` meta + done,
  with the thread id so the client can call /resume
- end of run                        → ``event: done``
- any exception                     → logged server-side; friendly ``error`` + done

The graph, checkpointer, and Anthropic client are built once in the lifespan
(main.py) and read from ``app.state`` here.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from langgraph.types import Command
from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse

from app.config import RATE_LIMIT
from app.graph.state import GraphContext
from app.ratelimit import limiter
from app.schemas import EncourageRequest, ResumeRequest
from app.sse import (
    SSE_HEADERS,
    done_frame,
    error_frame,
    meta_frame,
    sse_frame,
    token_frame,
)

logger = logging.getLogger("bloom.api")

router = APIRouter(prefix="/api")


def _translate(mode: str, data: Any, thread_id: str) -> dict[str, str] | None:
    """Map one graph stream item to an SSE frame, or None to ignore it."""
    if mode == "custom" and isinstance(data, dict):
        if data.get("type") == "meta":
            return meta_frame(data["kind"])
        if data.get("type") == "token":
            return token_frame(data["text"])
    elif mode == "updates" and isinstance(data, dict) and "__interrupt__" in data:
        # Distress case paused for human moderation: tell the client which
        # thread to resume. Contract-compatible (clients still read `.type`).
        return sse_frame(
            "meta",
            {"type": "support", "status": "awaiting_moderation", "threadId": thread_id},
        )
    return None


async def _stream_graph(
    request: Request, graph_input: Any, thread_id: str
) -> AsyncIterator[dict[str, str]]:
    """Drive the graph and yield SSE frames. Never lets an exception escape as a
    stack trace — logs it and emits the friendly in-world error instead."""
    graph = request.app.state.graph
    client = request.app.state.anthropic_client
    config = {"configurable": {"thread_id": thread_id}}
    context = GraphContext(client=client)

    try:
        async for mode, data in graph.astream(
            graph_input, config=config, context=context, stream_mode=["custom", "updates"]
        ):
            frame = _translate(mode, data, thread_id)
            if frame is not None:
                yield frame
        yield done_frame()
    except Exception:
        logger.exception("[/api/encourage] graph run failed for thread %s", thread_id)
        yield error_frame()
        yield done_frame()


@router.post("/encourage")
@limiter.limit(RATE_LIMIT)
async def encourage(request: Request) -> Any:
    """Start a run. Streams SSE for both the encouragement and support paths."""
    if request.app.state.anthropic_client is None:
        return JSONResponse(
            status_code=503,
            content={"error": "The encouragement service is not configured yet."},
        )

    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse(status_code=400, content={"error": "Request body must be JSON."})

    try:
        parsed = EncourageRequest.model_validate(body)
    except ValidationError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid request.", "details": exc.errors(include_url=False)},
        )

    thread_id = uuid.uuid4().hex
    graph_input = {
        "stage_id": parsed.stage_id,
        "feeling": parsed.feeling.value,
        "free_text": parsed.free_text,
        "attempts": 0,
    }
    return EventSourceResponse(
        _stream_graph(request, graph_input, thread_id),
        headers={**SSE_HEADERS, "X-Thread-Id": thread_id},
    )


@router.post("/encourage/{thread_id}/resume")
@limiter.limit(RATE_LIMIT)
async def resume(request: Request, thread_id: str) -> Any:
    """Resume a graph paused at the moderation interrupt with the moderator
    decision. Streams the continuation using the same SSE format."""
    if request.app.state.anthropic_client is None:
        return JSONResponse(
            status_code=503,
            content={"error": "The encouragement service is not configured yet."},
        )

    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse(status_code=400, content={"error": "Request body must be JSON."})

    try:
        decision = ResumeRequest.model_validate(body)
    except ValidationError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid request.", "details": exc.errors(include_url=False)},
        )

    # Only resume a thread that is actually paused at an interrupt.
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await request.app.state.graph.aget_state(config)
    if not snapshot.next:
        return JSONResponse(
            status_code=404,
            content={"error": "No paused run for this thread."},
        )

    resume_value = {"approve": decision.approve, "note": decision.note}
    return EventSourceResponse(
        _stream_graph(request, Command(resume=resume_value), thread_id),
        headers={**SSE_HEADERS, "X-Thread-Id": thread_id},
    )
