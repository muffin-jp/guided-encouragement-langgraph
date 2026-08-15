"""AsyncAnthropic client factory.

A single shared async client is created at app startup (see main.py lifespan)
and injected into the graph via LangGraph's runtime context, so nodes never
read env or construct clients themselves. Keeping ANTHROPIC_API_KEY here — read
from server env, never returned to any client — is the one place credentials
live.
"""

from __future__ import annotations

from anthropic import AsyncAnthropic

from app.config import get_api_key


def build_anthropic_client() -> AsyncAnthropic:
    """Construct the shared AsyncAnthropic client.

    Raises RuntimeError (via get_api_key) when the key is missing so the caller
    can surface a friendly 503 without leaking configuration.
    """
    return AsyncAnthropic(api_key=get_api_key())
