"""Postgres + pgvector backend for retrieval grounding.

Same ``Retriever`` Protocol as ``MemoryRetriever`` — filter by feeling, rank by
cosine, take top-k — but the filter + rank + limit run as **one SQL query** in
Postgres (``WHERE feeling-filter ORDER BY embedding <=> query LIMIT k``). The
query is still embedded **locally, in-process**; pgvector only stores and
searches vectors, so the "no runtime network to embed" story holds — the one new
hop is to Postgres, which is internal infrastructure, not a third-party embed API.

The table is seeded offline from the reviewed corpus by :mod:`app.rag.seed_db`;
this module only reads it. ``asyncpg`` / ``pgvector`` are imported lazily inside
``connect`` so the default ``memory`` backend never pulls a DB driver.
"""

# asyncpg and pgvector ship no type stubs; relax the unknowns at this boundary.
# Our own logic stays typed.
# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from app.config import RAG_K, RAG_MIN_K
from app.graph.state import Passage
from app.rag.retriever import build_query

if TYPE_CHECKING:
    from app.rag.embedder import Embedder

logger = logging.getLogger("bloom.rag")

_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def safe_table(table: str) -> str:
    """Validate the table identifier — it can't be a bound parameter.

    The name comes from ``RAG_PG_TABLE`` (operator config, not user input), but a
    strict identifier check keeps it impossible to smuggle SQL through it.
    """
    if not _TABLE_RE.match(table):
        raise ValueError(f"unsafe pg table name: {table!r}")
    return table


class PgVectorRetriever:
    """Filter-then-rank retrieval as a single pgvector SQL query."""

    def __init__(self, pool: Any, embedder: Embedder, table: str) -> None:
        self._pool = pool
        self._embedder = embedder
        self._table = safe_table(table)

    @classmethod
    async def connect(cls, dsn: str, table: str, embedder: Embedder) -> PgVectorRetriever:
        """Open a pooled connection, register the pgvector codec, and probe once.

        The probe (``SELECT 1 FROM <table>``) fails fast at startup if the table
        or extension is missing, so the app's startup fail-open leaves the
        retriever ``None`` rather than erroring on every request.
        """
        import asyncpg
        from pgvector.asyncpg import register_vector

        safe = safe_table(table)

        async def _init(conn: Any) -> None:
            await register_vector(conn)

        pool = await asyncpg.create_pool(dsn, init=_init, min_size=1, max_size=5)
        try:
            async with pool.acquire() as conn:
                await conn.fetchval(f"SELECT 1 FROM {safe} LIMIT 1")
        except BaseException:
            await pool.close()
            raise
        return cls(pool, embedder, safe)

    async def aclose(self) -> None:
        await self._pool.close()

    async def retrieve(
        self, feeling: str, free_text: str | None, *, k: int = RAG_K
    ) -> list[Passage]:
        """Embed the query locally, then one round trip: filter, cosine, top-k."""
        qvec = self._embedder.embed([build_query(feeling, free_text)])[0]
        # `, id` is a deterministic tiebreak so equal-distance rows order stably
        # (the memory backend breaks ties by corpus order; exact ties don't occur
        # with real float embeddings, so the two agree in practice).
        sql = (
            f"SELECT id, kind, text, source FROM {self._table} "
            "WHERE $1 = ANY(feelings) OR '*' = ANY(feelings) "
            "ORDER BY embedding <=> $2, id LIMIT $3"
        )
        rows = await self._pool.fetch(sql, feeling, qvec, max(k, 0))
        if 0 < len(rows) < RAG_MIN_K:
            logger.info(
                "retrieval below preferred floor: %d row(s) for feeling=%r (min %d)",
                len(rows),
                feeling,
                RAG_MIN_K,
            )
        return [
            Passage(id=r["id"], kind=r["kind"], text=r["text"], source=r["source"]) for r in rows
        ]
