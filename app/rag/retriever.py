"""Retrieval grounding: the ``Retriever`` seam and an in-memory reference.

``Retriever`` is a Protocol — one async ``retrieve`` method. The graph
(``GraphContext.retriever``, the ``retrieve`` node) depends only on it. The one
runtime implementation is :class:`app.rag.pg_retriever.PgVectorRetriever`.

``MemoryRetriever`` is **not a runtime backend**. It is the in-memory reference
for the same filter-then-rank semantics — filter the vetted corpus by the
player's feeling, embed the query, rank by cosine similarity — built from vectors
and records handed to it directly. The unit tests, the pgvector parity test (as
the ranking oracle), and ``evals/run.py --dry`` use it, so retrieval stays
testable with no DB, no weights, and no network.
"""

# NumPy's overloaded ufunc/linalg stubs resolve to partially-unknown types under
# pyright strict, so member/argument/variable types leak as Unknown at this
# numeric boundary. Our own logic stays typed; this narrowly relaxes the three
# "unknown" rules for the numpy calls in this module only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from app.config import RAG_K, RAG_MIN_K
from app.graph.state import Passage

if TYPE_CHECKING:
    from app.rag.embedder import Embedder

logger = logging.getLogger("bloom.rag")


def build_query(feeling: str, free_text: str | None) -> str:
    """The exact string embedded for retrieval — feeling plus any free text.

    Kept as one function so the query built at request time is identical to the
    one an offline analysis or test would build.
    """
    return f"{feeling}. {free_text or ''}"


def _passage_matches(feelings: list[str], feeling: str) -> bool:
    """A passage is eligible if it is tagged for this feeling or for ``"*"``."""
    return feeling in feelings or "*" in feelings


class Retriever(Protocol):
    """The graph's view of retrieval grounding — one async method.

    Both ``MemoryRetriever`` and a future ``PgVectorRetriever`` satisfy this, so
    ``GraphContext.retriever`` and the ``retrieve`` node bind to the Protocol,
    never a concrete backend.
    """

    async def retrieve(
        self, feeling: str, free_text: str | None, *, k: int = RAG_K
    ) -> list[Passage]: ...


class MemoryRetriever:
    """In-memory reference filter-then-rank retrieval (numpy) — tests and dry eval only."""

    def __init__(self, vectors: np.ndarray, records: list[dict[str, Any]], embedder: Embedder):
        # ``records`` carry the filter tags (``feelings``) alongside the public
        # Passage fields; the returned Passage strips the tags back out.
        self._vectors = np.asarray(vectors, dtype=np.float32)
        self._records = records
        self._embedder = embedder

    async def retrieve(
        self, feeling: str, free_text: str | None, *, k: int = RAG_K
    ) -> list[Passage]:
        """Filter by feeling, rank by cosine similarity, return the top ``k``.

        Prefers at least ``RAG_MIN_K`` passages when the filter yields that many;
        if fewer pass it returns what there is — including an empty list.
        """
        idx = [i for i, r in enumerate(self._records) if _passage_matches(r["feelings"], feeling)]
        if not idx:
            return []

        query = build_query(feeling, free_text)
        qvec = self._embedder.embed([query])[0]  # unit vector -> cosine == dot
        candidate_vectors = self._vectors[idx]
        scores = candidate_vectors @ qvec

        # Top-k by score, ties broken by corpus order for a stable, reviewable result.
        order = sorted(range(len(idx)), key=lambda j: (-float(scores[j]), idx[j]))
        top = order[: max(k, 0)]

        if 0 < len(idx) < RAG_MIN_K:
            logger.info(
                "retrieval below preferred floor: %d passage(s) for feeling=%r (min %d)",
                len(idx),
                feeling,
                RAG_MIN_K,
            )
        return [_to_passage(self._records[idx[j]]) for j in top]


def _to_passage(record: dict[str, Any]) -> Passage:
    """Project a stored record down to the public Passage fields."""
    return Passage(
        id=record["id"],
        kind=record["kind"],
        text=record["text"],
        source=record["source"],
    )
