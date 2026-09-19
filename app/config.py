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

# Retrieval backend behind the same Retriever seam:
#   memory  : in-process numpy over the committed index.npz (default; offline, CI).
#   pgvector: Postgres + pgvector, seeded offline from corpus.jsonl (`make db-seed`).
# The embedder is local either way — pgvector adds a DB hop, not a network embed.
# DATABASE_URL is server-side only and required only when RAG_BACKEND=pgvector.
RAG_BACKEND = os.environ.get("RAG_BACKEND", "memory").lower()
DATABASE_URL = os.environ.get("DATABASE_URL", "")
RAG_PG_TABLE = os.environ.get("RAG_PG_TABLE", "grounding_passages")

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

# --- Local distress classifier (dark launch) --------------------------------
# A 385-weight model, trained and evaluated in bloom-distress-classifier, sits in
# front of the Haiku distress check. Confidently fine notes skip the call,
# confidently distressed notes go straight to support, and everything else
# escalates to Haiku exactly as today. Any failure to load or score escalates too.
#
# OFF by default. The artifact of 2026-09-16 (evaluated upstream on 2026-09-17,
# second recorded look) closes the support band entirely: its `high` is 1.000, so
# nothing is routed to support and the failure that kept this flag off — 8 golden
# encouragement cases diverted, dropping the safety rate to 80.5% against a 100%
# gate — cannot recur. The skip band is now decided by a note's worst *segment*,
# because mean pooling let a crisis clause inside a long calm note be averaged
# away; upstream that let 13 of 72 adversarial notes skip the LLM, and 0 do now.
#
# What this buys is 14% fewer Haiku calls and nothing else: on the upstream test
# set the cascade's recall and false-alarm count are identical to the LLM alone.
# What it costs is the golden guarantee — the previous artifact routed all 10
# golden distress cases to support without the LLM, and all 10 now escalate.
#
# The real eval was re-run with the flag on (2026-09-17) and passes every metric:
# distress routing, game frustration, judge safety and word limits all 100%, mean
# empathy 4.59 and tone 4.83. The classifier changed 6 of 51 routes, all of them
# encouragement cases skipping the Haiku call and landing on encouragement; all 10
# distress cases escalated and Haiku caught every one.
#
# Read that 10 of 10 carefully. Ten cases with no misses bound the per-case catch
# rate below only 0.741 at 95%, and the judge and classifier are sampled. It says
# routing did not break the gate, not that the previous artifact's direct-to-support
# guarantee was safe to lose. Against production as it runs today — this flag off,
# every note to Haiku — the distress path is identical, so that bound is one this
# service already lives with rather than one the classifier introduces.
#
# Still to do before this goes on: measure latency, since a note now costs about ten
# embeddings instead of one, and decide whether 14% fewer Haiku calls is worth two
# rules kept in sync across two repositories. Never fix a gate failure by editing a
# threshold here — thresholds are fitted upstream and carried by the artifact.
CLASSIFIER_ENABLED = os.environ.get("CLASSIFIER_ENABLED", "false").lower() in {"1", "true", "yes"}

# sha256 over model.npz then model.json, computed exactly as the upstream test
# ledger records it. The loader and `make check-classifier` refuse any other
# bytes: the artifact served must be the artifact that was evaluated.
CLASSIFIER_ARTIFACT_SHA256 = "2de175fbaea1ec8e870bb06acfb2ae66eae55bacfb2af191f6287fed12c3e88e"

# The artifact records the embedder revision it was built with, which is not
# EMBED_MODEL_REVISION above. On 2026-09-15 the two snapshots were verified
# byte-identical on every file that affects an embedding, and produced identical
# embeddings (max |diff| 0.0 over 497 notes). The loader accepts only the revisions
# listed here; EMBED_WEIGHTS_SHA256 lets `make check-classifier` re-verify the
# claim whenever the weights are vendored.
CLASSIFIER_EQUIVALENT_EMBEDDER_REVISIONS = frozenset({"c9745ed1d9f207416be6d2e6f8de32d1f16199bf"})
EMBED_WEIGHTS_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"

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
