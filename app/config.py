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

# Human-in-the-loop moderation gate on the distress path. On by default, and
# safe to be: it is non-blocking. The support node always streams the reviewed
# words to the player first; only then does the moderate interrupt fire, pausing
# the run for out-of-band review via the /resume route. So a distressed player
# never waits on a human. Set MODERATION_ENABLED=0 to drop the gate entirely
# (distress becomes simply support -> END). The HITL machinery is always present.
MODERATION_ENABLED = os.environ.get("MODERATION_ENABLED", "true").lower() in {"1", "true", "yes"}

# --- Retrieval grounding ----------------------------------------------------
# The retrieve node fetches a few pre-approved passages (CBT-informed coping
# techniques, tone-approved phrasing, reviewed static responses) filtered by the
# player's feeling and injects them into the generation prompt as *grounding, not
# a script*. The critique/judge guardrail is unchanged and remains the real gate.
#
# RAG_ENABLED is the kill switch (default on), mirroring MODERATION_ENABLED. When
# off, build_graph wires classify_distress -> generate directly — byte-for-byte
# today's graph — and the retriever/index/embedder are never loaded at startup.
# Flipping it off is an instant rollback to the pre-RAG behaviour, no redeploy.
RAG_ENABLED = os.environ.get("RAG_ENABLED", "true").lower() in {"1", "true", "yes"}
RAG_K = int(os.environ.get("RAG_K", "3"))  # passages injected as grounding
RAG_MIN_K = int(os.environ.get("RAG_MIN_K", "2"))  # preferred floor when the corpus allows

# Pinned local sentence-transformer. No third-party embedding API and no runtime
# network to embed: the weights are vendored at build time (huggingface.co is not
# on the runtime allow-list) and loaded with local_files_only=True, and the query
# is embedded in-process. Pinning by revision freezes one snapshot so the index is
# reproducible — that reproducibility *is* the defensibility story. The model's
# weights have not changed since 2021; later commits only add file formats, so
# embeddings are identical across revisions. Verify/refresh the full SHA with:
#   git ls-remote https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 refs/heads/main
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_REVISION = (
    "ea78891063587eb050ed4166b20062eaf978037c"  # pinned HF commit (verify at build)
)

# --- Rate limiting ----------------------------------------------------------
RATE_LIMIT = "10/minute"
RATE_LIMIT_RETRY_AFTER = "60"

# --- CORS -------------------------------------------------------------------
# Browser clients on a different origin (e.g. the Next.js demo UI) must be
# allow-listed or the browser blocks the fetch. Server-to-server callers (curl,
# the Unity client) are unaffected by CORS.
#
# For convenience the default matches localhost / 127.0.0.1 on ANY port via a
# regex, so the demo works whether Next.js lands on :3000, :3001, etc. Set an
# explicit comma-separated CORS_ALLOW_ORIGINS list in production instead.
CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
]
CORS_ALLOW_ORIGIN_REGEX = os.environ.get(
    "CORS_ALLOW_ORIGIN_REGEX", r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
)

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
