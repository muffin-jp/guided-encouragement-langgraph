"""FastAPI application.

The graph is built and compiled exactly once, in the lifespan, with a
checkpointer, and reused for every request. The Anthropic client is created once
too (or left None when the API key is absent, so routes return a friendly 503
rather than crashing at startup). ANTHROPIC_API_KEY stays server-side.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded

from app.api.routes import router
from app.config import langsmith_enabled
from app.graph.build import build_graph
from app.llm import build_anthropic_client
from app.ratelimit import limiter, rate_limit_exceeded_handler

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bloom")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if langsmith_enabled():
        # LangSmith reads its own env (LANGSMITH_API_KEY/PROJECT). We just flip
        # the tracing switch LangChain/LangGraph look for.
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        logger.info("LangSmith tracing enabled")

    # Compile the graph once (MemorySaver for the demo — see build_graph for the
    # production Postgres seam).
    app.state.graph = build_graph()

    try:
        app.state.anthropic_client = build_anthropic_client()
    except RuntimeError:
        # Missing key: start anyway; routes surface a friendly 503.
        app.state.anthropic_client = None
        logger.warning("ANTHROPIC_API_KEY not set — /api/encourage will return 503")

    yield

    client = app.state.anthropic_client
    if client is not None:
        await client.close()


app = FastAPI(title="Bloom — Guided Encouragement", lifespan=lifespan)

# Wire slowapi: the limiter needs to live on app.state and its 429s need the
# friendly handler with Retry-After.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

app.include_router(router)


@app.get("/healthz")
async def healthz() -> dict[str, bool]:
    return {"ok": True, "configured": app.state.anthropic_client is not None}
