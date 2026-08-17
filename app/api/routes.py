"""HTTP routes: POST /api/encourage and its /resume continuation.

The route is a thin translation layer. ``/api/encourage`` drives the compiled
graph with ``astream(stream_mode="custom")`` and maps the structured
``meta``/``token`` events the nodes emit onto the existing SSE wire contract
(see app/sse.py):

- custom ``meta`` / ``token`` events → ``event: meta`` / ``event: token``
- end of run                        → ``event: done``
- any exception                     → logged server-side; friendly ``error`` + done

On the distress path the support message is streamed first; when moderation is
enabled the graph then *pauses* at the ``moderate`` interrupt. The player's
stream simply ends (they already have their support words); the run stays
checkpointed under its ``thread_id`` (returned in the ``X-Thread-Id`` header) for
out-of-band review. ``/resume`` is called by a moderator tool — not the player —
to record the decision; it returns a small JSON acknowledgement and streams
nothing.

The graph, checkpointer, and Anthropic client are built once in the lifespan
(main.py) and read from ``app.state`` here.
"""

# slowapi's @limiter.limit decorator is untyped and its `.limit` member resolves
# to an unknown return, so pyright flags the two decorated routes below. This
# narrowly relaxes those two library-boundary rules; the rest stays strict.
# pyright: reportUntypedFunctionDecorator=false, reportUnknownMemberType=false
from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

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
    token_frame,
)

logger = logging.getLogger("bloom.api")

router = APIRouter(prefix="/api")


def _translate(data: Any) -> dict[str, str] | None:
    """Map one graph custom-stream item to an SSE frame, or None to ignore it."""
    if isinstance(data, dict):
        event = cast("dict[str, Any]", data)
        if event.get("type") == "meta":
            return meta_frame(event["kind"])
        if event.get("type") == "token":
            return token_frame(event["text"])
    return None


async def _stream_graph(
    request: Request, graph_input: Any, thread_id: str
) -> AsyncIterator[dict[str, str]]:
    """Drive the graph and yield SSE frames. Never lets an exception escape as a
    stack trace — logs it and emits the friendly in-world error instead.

    If the run pauses at the moderation interrupt, ``astream`` simply stops
    yielding and the run is left checkpointed; the player has already received
    their message, so we close the stream cleanly with ``done``.
    """
    graph = request.app.state.graph
    client = request.app.state.anthropic_client
    config = {"configurable": {"thread_id": thread_id}}
    context = GraphContext(client=client)

    try:
        async for data in graph.astream(
            graph_input, config=config, context=context, stream_mode="custom"
        ):
            frame = _translate(data)
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
    """Record a moderator decision for a paused distress case.

    Called by a moderator tool, not the player — the support message was already
    delivered on the original request. Resumes the checkpointed run past the
    ``moderate`` interrupt with ``Command(resume=...)`` to log the decision, then
    returns a JSON acknowledgement. It does **not** re-stream anything.
    """
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

    context = GraphContext(client=request.app.state.anthropic_client)
    resume_value = {"approve": decision.approve, "note": decision.note}
    final_state = cast(
        "dict[str, Any]",
        await request.app.state.graph.ainvoke(
            Command(resume=resume_value), config=config, context=context
        ),
    )
    return JSONResponse(
        status_code=200,
        content={
            "status": "recorded",
            "threadId": thread_id,
            "approve": bool(final_state.get("moderator_approved", decision.approve)),
            "note": final_state.get("moderator_note"),
        },
    )
