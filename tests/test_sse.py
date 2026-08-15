"""SSE frame serialization."""

from __future__ import annotations

import json

from app.sse import (
    FRIENDLY_ERROR_MESSAGE,
    done_frame,
    error_frame,
    meta_frame,
    sse_frame,
    token_frame,
)


def test_meta_frame_shapes() -> None:
    assert meta_frame("encouragement") == {"event": "meta", "data": '{"type":"encouragement"}'}
    assert meta_frame("support") == {"event": "meta", "data": '{"type":"support"}'}


def test_token_frame_preserves_text_and_spacing() -> None:
    frame = token_frame("It makes sense ")
    assert frame["event"] == "token"
    assert json.loads(frame["data"]) == {"text": "It makes sense "}


def test_done_frame_is_empty_object() -> None:
    assert done_frame() == {"event": "done", "data": "{}"}


def test_error_frame_defaults_to_friendly_message() -> None:
    frame = error_frame()
    assert frame["event"] == "error"
    assert json.loads(frame["data"])["message"] == FRIENDLY_ERROR_MESSAGE
    assert "nap" in frame["data"]


def test_error_frame_never_leaks_custom_internals_by_default() -> None:
    # Callers pass no argument on failure, so a stack trace can never leak.
    assert "Traceback" not in error_frame()["data"]


def test_sse_frame_is_compact_json() -> None:
    frame = sse_frame("token", {"text": "hi", "n": 1})
    # No spaces after separators — compact wire format.
    assert frame["data"] == '{"text":"hi","n":1}'
