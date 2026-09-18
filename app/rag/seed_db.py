"""Offline seed for the pgvector retrieval table.

    uv run python -m app.rag.seed_db           # (re)seed the grounding table
    uv run python -m app.rag.seed_db --check    # CI: does the table match corpus.jsonl?

The pgvector twin of :mod:`app.rag.build_index`. It reads the reviewed
``corpus.jsonl``, embeds each ``text`` once with the pinned local model, and
upserts the rows into Postgres (deleting rows whose id has left the corpus). Same
corpus + same model → same rows; this runs offline, never at request time.

``--check`` reads the live table and compares it against a fresh build of the
current corpus — metadata (id/feelings/kind/text/source, and the row set) exactly,
vectors within a small tolerance — and exits non-zero on drift, so a corpus edit
can't land without a re-seed. The corpus file remains the audit surface; the table
is derived from it and guarded here.
"""

# asyncpg/pgvector ship no stubs and numpy's stubs leak Unknown under pyright
# strict; relax those at this boundary. Our own logic stays typed.
# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

import numpy as np

from app.config import DATABASE_URL, RAG_PG_TABLE
from app.rag.build_index import load_corpus
from app.rag.pg_retriever import safe_table

_UPSERT = """
INSERT INTO {table} (id, feelings, kind, text, source, embedding)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (id) DO UPDATE SET
    feelings = EXCLUDED.feelings, kind = EXCLUDED.kind, text = EXCLUDED.text,
    source = EXCLUDED.source, embedding = EXCLUDED.embedding
"""


def _embed(records: list[dict[str, Any]]) -> np.ndarray:
    """Embed every passage with the pinned local model (vendor weights if absent)."""
    from app.rag.embedder import MODEL_DIR, SentenceTransformerEmbedder, fetch_model

    if not MODEL_DIR.exists():
        print(f"Vendoring pinned weights into {MODEL_DIR} (build-time network) ...")
        fetch_model()
    return SentenceTransformerEmbedder().embed([r["text"] for r in records]).astype(np.float32)


async def _connect(dsn: str) -> Any:
    import asyncpg
    from pgvector.asyncpg import register_vector

    conn = await asyncpg.connect(dsn)
    await register_vector(conn)
    return conn


async def seed(records: list[dict[str, Any]], vectors: np.ndarray, table: str, dsn: str) -> None:
    safe = safe_table(table)
    conn = await _connect(dsn)
    try:
        await conn.executemany(
            _UPSERT.format(table=safe),
            [
                (r["id"], r["feelings"], r["kind"], r["text"], r["source"], vectors[i].tolist())
                for i, r in enumerate(records)
            ],
        )
        # Drop rows whose id has left the corpus, so the table can't drift stale.
        ids = [r["id"] for r in records]
        await conn.execute(f"DELETE FROM {safe} WHERE id <> ALL($1::text[])", ids)
    finally:
        await conn.close()
    print(f"Seeded {len(records)} passage(s) into {safe}.")


async def check(records: list[dict[str, Any]], vectors: np.ndarray, table: str, dsn: str) -> int:
    """Return 0 if the live table matches a fresh build of the corpus, else 1."""
    safe = safe_table(table)
    conn = await _connect(dsn)
    try:
        rows = await conn.fetch(f"SELECT id, feelings, kind, text, source, embedding FROM {safe}")
    finally:
        await conn.close()

    fresh_meta = {
        r["id"]: (list(r["feelings"]), r["kind"], r["text"], r["source"]) for r in records
    }
    fresh_vec = {r["id"]: vectors[i] for i, r in enumerate(records)}
    live_meta = {r["id"]: (list(r["feelings"]), r["kind"], r["text"], r["source"]) for r in rows}

    if set(live_meta) != set(fresh_meta):
        print(
            "pgvector table drifted from corpus.jsonl: row set differs. Run `make db-seed`.",
            file=sys.stderr,
        )
        return 1
    if live_meta != fresh_meta:
        print(
            "pgvector table metadata is stale vs corpus.jsonl. Run `make db-seed`.",
            file=sys.stderr,
        )
        return 1
    for r in rows:
        if not np.allclose(np.asarray(r["embedding"]), fresh_vec[r["id"]], atol=1e-4, rtol=1e-4):
            print(
                f"embedding for {r['id']!r} differs from a fresh build; run `make db-seed`.",
                file=sys.stderr,
            )
            return 1
    print(f"pgvector table matches corpus.jsonl ({len(rows)} rows).")
    return 0


async def _amain(do_check: bool) -> int:
    if not DATABASE_URL:
        print("DATABASE_URL is not set (required for the pgvector backend).", file=sys.stderr)
        return 1
    records = load_corpus()
    print(f"Loaded {len(records)} reviewed passage(s).")
    vectors = _embed(records)
    if do_check:
        return await check(records, vectors, RAG_PG_TABLE, DATABASE_URL)
    await seed(records, vectors, RAG_PG_TABLE, DATABASE_URL)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the live table matches a fresh build of corpus.jsonl (CI); do not write",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_amain(args.check)))


if __name__ == "__main__":
    main()
