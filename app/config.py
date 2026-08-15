"""Central configuration: model IDs, thresholds, the feeling enum, and env.

Everything tunable about the graph lives here so the routes, nodes, and evals
share one source of truth. Model IDs mirror the original TypeScript service
(verified against the Claude docs, 2026-08) — the whole point of the port is
that evals exercise exactly what production ships.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache

# --- Models -----------------------------------------------------------------
# Generation: current recommended Sonnet ("best combination of speed and
# intelligence"). Distress pre-check: cheap/fast Haiku. Judge: sonnet-4-6 at
# temperature 0 — deliberately an older Sonnet because it still accepts
# sampling params (sonnet-5 rejects them), giving a reproducible grader.
GENERATION_MODEL = "claude-sonnet-5"
DISTRESS_MODEL = "claude-haiku-4-5"
JUDGE_MODEL = "claude-sonnet-4-6"

# --- Graph thresholds -------------------------------------------------------
# Hard ceiling on a Mamorin reply. The prompt aims for ~25 words; anything
# over this is a critique failure that triggers a regeneration.
WORD_LIMIT = 40

# Reflection loop bound. attempts counts generations; with MAX_ATTEMPTS = 2 the
# graph generates at most twice before falling back, so it can never loop
# unbounded. Kept in config so the bound is a single, testable knob.
MAX_ATTEMPTS = 2

# Human-in-the-loop moderation gate on the distress path. Off by default: a
# distressed player should never wait on a human, so distress streams the static
# support message immediately. When enabled, distress cases pause at an
# interrupt() for a moderator to review and are continued via the /resume route.
# The HITL machinery is always present (node, interrupt, /resume); this flag only
# decides whether it sits on the live request path.
MODERATION_ENABLED = os.environ.get("MODERATION_ENABLED", "").lower() in {"1", "true", "yes"}

# --- Rate limiting ----------------------------------------------------------
RATE_LIMIT = "10/minute"
RATE_LIMIT_RETRY_AFTER = "60"

# --- Free text --------------------------------------------------------------
MAX_FREE_TEXT_LENGTH = 200


class Feeling(StrEnum):
    """The preset feelings a player can pick on the post-stage screen.

    "custom" means the player skipped the chips and only wrote free text.
    Shared by request validation and the encouragement prompt so the contract
    can't drift between client and server.
    """

    PROUD = "proud"
    RELIEVED = "relieved"
    FRUSTRATED = "frustrated"
    DISAPPOINTED = "disappointed"
    ANXIOUS = "anxious"
    TIRED = "tired"
    CUSTOM = "custom"


def get_api_key() -> str:
    """Read ANTHROPIC_API_KEY from server env.

    Raises so the route can turn a missing key into a friendly 503 without ever
    leaking configuration details to the client. Kept server-side only.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")
    return api_key


@lru_cache(maxsize=1)
def langsmith_enabled() -> bool:
    """LangSmith tracing is opt-in behind a single env flag, off by default."""
    return os.environ.get("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}
