"""Server-Sent Events framing for the /api/encourage stream.

Wire format (identical for the encouragement and support paths, so the Unity
client and the web UI handle both the same way):

    event: meta   data: {"type":"encouragement" | "support"}
    event: token  data: {"text":"..."}          (repeated)
    event: error  data: {"message":"..."}       (only on failure, friendly text)
    event: done   data: {}

This module owns the framing only; sse-starlette's EventSourceResponse in the
route does the disconnect / keep-alive handling.
"""

from __future__ import annotations

import json
from typing import Any, Literal

ResponseKind = Literal["encouragement", "support"]

# A mid-stream failure never leaks a stack trace to the player: we log the real
# error server-side and send this friendly, in-world message instead.
FRIENDLY_ERROR_MESSAGE = (
    "Mamorin is having a little nap and couldn't hear you just now. Your stage "
    "clear still counts — please try again in a moment."
)

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # disable proxy buffering so tokens flush live
}


def sse_frame(event: str, data: Any) -> dict[str, str]:
    """Build one EventSourceResponse frame.

    sse-starlette takes {"event", "data"} dicts and serialises the wire bytes
    (``event: <event>\\ndata: <data>\\n\\n``) itself, adding keep-alive pings.
    """
    return {"event": event, "data": json.dumps(data, separators=(",", ":"))}


def meta_frame(kind: ResponseKind) -> dict[str, str]:
    return sse_frame("meta", {"type": kind})


def token_frame(text: str) -> dict[str, str]:
    return sse_frame("token", {"text": text})


def error_frame(message: str = FRIENDLY_ERROR_MESSAGE) -> dict[str, str]:
    return sse_frame("error", {"message": message})


def done_frame() -> dict[str, str]:
    return sse_frame("done", {})
