"""pgvector backend: integration + parity with MemoryRetriever.

Requires a real Postgres+pgvector at DATABASE_URL; skipped otherwise (local dev
with no DB, and the fast CI lane). Uses the offline StubEmbedder over a per-test
temp table — no model download, no network — so the tests exercise the SQL
filter-then-rank and its parity with the memory backend, not the embedder.
"""

# asyncpg/pgvector ship no stubs and numpy leaks Unknown under pyright strict;
# relax those at this boundary. Our own logic stays typed.
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio

from app.rag.pg_retriever import PgVectorRetriever
from app.rag.retriever import MemoryRetriever
from tests.test_retriever import StubEmbedder

DATABASE_URL = os.environ.get("DATABASE_URL", "")

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATABASE_URL, reason="no DATABASE_URL; pgvector tests skipped"),
]


def _record(pid: str, feelings: list[str], text: str, kind: str = "technique") -> dict[str, Any]:
    return {"id": pid, "feelings": feelings, "kind": kind, "text": text, "source": "test"}


async def _open() -> Any:
    import asyncpg
    from pgvector.asyncpg import register_vector

    conn = await asyncpg.connect(DATABASE_URL)
    # register_vector introspects the `vector` type, so the extension must exist
    # first — create it here so the tests are self-contained on a bare Postgres
    # (the CI service provides one with no migration run).
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await register_vector(conn)
    return conn


@pytest_asyncio.fixture
async def table() -> AsyncIterator[str]:
    """A per-test pgvector table sized to the stub embedder, dropped on teardown."""
    conn = await _open()
    name = "test_grounding_" + uuid.uuid4().hex
    await conn.execute(
        f"CREATE TABLE {name} (id text PRIMARY KEY, feelings text[] NOT NULL, "
        f"kind text NOT NULL, text text NOT NULL, source text NOT NULL, "
        f"embedding vector({StubEmbedder.dim}) NOT NULL)"
    )
    try:
        yield name
    finally:
        await conn.execute(f"DROP TABLE IF EXISTS {name}")
        await conn.close()


async def _seed(
    table: str, records: list[dict[str, Any]]
) -> tuple[PgVectorRetriever, MemoryRetriever]:
    """Seed the table with stub vectors; return both backends over the same data."""
    embedder = StubEmbedder()
    vectors = embedder.embed([r["text"] for r in records])
    conn = await _open()
    try:
        await conn.executemany(
            f"INSERT INTO {table} (id, feelings, kind, text, source, embedding) "
            "VALUES ($1, $2, $3, $4, $5, $6)",
            [
                (r["id"], r["feelings"], r["kind"], r["text"], r["source"], vectors[i].tolist())
                for i, r in enumerate(records)
            ],
        )
    finally:
        await conn.close()
    pg = await PgVectorRetriever.connect(DATABASE_URL, table, embedder)
    memory = MemoryRetriever(vectors, records, embedder)
    return pg, memory


async def test_pg_filters_by_feeling(table: str) -> None:
    records = [
        _record("proud-1", ["proud"], "you earned this proud moment"),
        _record("frustrated-1", ["frustrated"], "that was frustrating and hard"),
    ]
    pg, _ = await _seed(table, records)
    try:
        got = await pg.retrieve("proud", None, k=3)
        assert [p["id"] for p in got] == ["proud-1"]
    finally:
        await pg.aclose()


async def test_pg_wildcard_matches_any_feeling(table: str) -> None:
    records = [
        _record("uni-1", ["*"], "it makes sense to feel this way"),
        _record("tired-1", ["tired"], "rest now you did enough"),
    ]
    pg, _ = await _seed(table, records)
    try:
        for feeling in ("proud", "anxious", "custom"):
            ids = {p["id"] for p in await pg.retrieve(feeling, None, k=3)}
            assert "uni-1" in ids, f"wildcard should match {feeling}"
    finally:
        await pg.aclose()


async def test_pg_no_match_returns_empty(table: str) -> None:
    pg, _ = await _seed(table, [_record("proud-1", ["proud"], "you earned this")])
    try:
        assert await pg.retrieve("tired", None, k=3) == []
    finally:
        await pg.aclose()


async def test_pg_returned_passages_have_public_fields_only(table: str) -> None:
    pg, _ = await _seed(table, [_record("proud-1", ["proud"], "you earned this", kind="phrasing")])
    try:
        got = await pg.retrieve("proud", None, k=3)
        assert got[0] == {
            "id": "proud-1",
            "kind": "phrasing",
            "text": "you earned this",
            "source": "test",
        }
    finally:
        await pg.aclose()


async def test_pg_parity_with_memory_backend(table: str) -> None:
    # Distinct vocabularies keep every cosine score distinct, so both backends
    # rank identically regardless of tie-break rule.
    records = [
        _record("proud-1", ["proud", "*"], "gentle sleepy proud rest kindness earned"),
        _record("proud-2", ["proud"], "effort patience persistence counted today"),
        _record("uni-1", ["*"], "warm quiet welcome however feeling"),
        _record("tired-1", ["tired"], "long stretch put down look after"),
        _record("custom-1", ["custom"], "thank you telling whatever carrying real"),
    ]
    pg, memory = await _seed(table, records)
    try:
        queries = [
            ("proud", "gentle sleepy proud rest kindness earned"),
            ("proud", None),
            ("custom", "thank you telling"),
            ("tired", "long day"),
            ("anxious", "worried about next"),
        ]
        for feeling, note in queries:
            pg_ids = [p["id"] for p in await pg.retrieve(feeling, note, k=3)]
            mem_ids = [p["id"] for p in await memory.retrieve(feeling, note, k=3)]
            assert pg_ids == mem_ids, f"backend parity broke for {feeling!r}/{note!r}"
    finally:
        await pg.aclose()
