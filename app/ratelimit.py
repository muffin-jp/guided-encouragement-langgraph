"""slowapi rate limiter.

In-memory sliding window keyed by client IP — good enough for a public demo.
In production this backing store would be Redis (pass ``storage_uri=...`` to the
Limiter) so limits hold across instances and restarts. The limit and the
Retry-After hint live in config.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import RATE_LIMIT_RETRY_AFTER

limiter = Limiter(key_func=get_remote_address)


def rate_limit_exceeded_handler(request: Request, exc: Exception) -> JSONResponse:
    """Friendly 429 with a Retry-After header (no internals leaked)."""
    return JSONResponse(
        status_code=429,
        content={"error": "Too many requests. Please take a short breather and retry."},
        headers={"Retry-After": RATE_LIMIT_RETRY_AFTER},
    )
