"""In-process retriever over the offline-built index.

Load once at startup (``Retriever.from_files``) and inject via ``GraphContext``,
mirroring how the Anthropic client is injected — never checkpointed. At request
time it does no network: it filters the vetted corpus by the player's feeling,
embeds the query with the pinned local model in-process, and ranks by cosine
similarity.

The index (``index.npz``) is built offline and deterministically by
:mod:`app.rag.build_index`; this module only reads it.
"""

# NumPy's overloaded ufunc/linalg stubs resolve to partially-unknown types under
# pyright strict, so member/argument/variable types leak as Unknown at this
# numeric boundary. Our own logic stays typed; this narrowly relaxes the three
# "unknown" rules for the numpy calls in this module only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from app.config import RAG_K, RAG_MIN_K
from app.graph.state import Passage

if TYPE_CHECKING:
    from app.rag.embedder import Embedder

logger = logging.getLogger("bloom.rag")

RAG_DIR = Path(__file__).resolve().parent
INDEX_PATH = RAG_DIR / "index.npz"


def build_query(feeling: str, free_text: str | None) -> str:
    """The exact string embedded for retrieval — feeling plus any free text.

    Kept as one function so the query built at request time is identical to the
    one an offline analysis or test would build.
    """
    return f"{feeling}. {free_text or ''}"


def _passage_matches(feelings: list[str], feeling: str) -> bool:
    """A passage is eligible if it is tagged for this feeling or for ``"*"``."""
    return feeling in feelings or "*" in feelings


class Retriever:
    """Filter-then-rank retrieval over the vetted corpus."""

    def __init__(self, vectors: np.ndarray, records: list[dict[str, Any]], embedder: Embedder):
        # ``records`` carry the filter tags (``feelings``) alongside the public
        # Passage fields; the returned Passage strips the tags back out.
        self._vectors = np.asarray(vectors, dtype=np.float32)
        self._records = records
        self._embedder = embedder

    @classmethod
    def from_files(cls, index_path: Path, embedder: Embedder) -> Retriever:
        """Load a committed ``index.npz`` and pair it with an embedder.

        ``allow_pickle=False``: the vectors are a plain float array and the
        metadata is a JSON string, so the index never needs to unpickle
        arbitrary objects to load.
        """
        with np.load(index_path, allow_pickle=False) as data:
            vectors = data["vectors"].astype(np.float32)
            meta = json.loads(str(data["meta"].item()))
        records: list[dict[str, Any]] = meta["passages"]
        if vectors.shape[0] != len(records):
            raise ValueError(
                f"index.npz is inconsistent: {vectors.shape[0]} vectors vs "
                f"{len(records)} passages — rebuild it with `make build-index`."
            )
        if embedder.dim and vectors.shape[1] != embedder.dim:
            raise ValueError(
                f"embedder dim {embedder.dim} != index dim {vectors.shape[1]}: the "
                "index was built with a different model. Rebuild with `make build-index`."
            )
        return cls(vectors, records, embedder)

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
