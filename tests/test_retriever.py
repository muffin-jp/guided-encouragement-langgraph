"""Retriever unit tests: filtering, ordering, floors, and offline loading.

No model and no network — a deterministic ``StubEmbedder`` (bag-of-words over a
hashed vocabulary) stands in for the pinned sentence-transformer, exactly the
seam the real ``Embedder`` protocol exists for.
"""

# numpy's stubs leak Unknown under pyright strict; relax the "unknown" rules here.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.rag.retriever import Retriever

pytestmark = pytest.mark.asyncio

_DIM = 32


class StubEmbedder:
    """Deterministic offline embedder: normalized hashed bag-of-words.

    Shared vocabulary between a passage and the query yields a higher cosine, so
    ordering is controllable from the fixture text — with no real model.
    """

    dim = _DIM

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                out[i, bucket] += 1.0
            norm = float(np.linalg.norm(out[i]))
            if norm:
                out[i] /= norm
        return out


def _record(pid: str, feelings: list[str], text: str, kind: str = "technique") -> dict[str, Any]:
    return {"id": pid, "feelings": feelings, "kind": kind, "text": text, "source": "test"}


def _retriever(records: list[dict[str, Any]]) -> Retriever:
    embedder = StubEmbedder()
    vectors = embedder.embed([r["text"] for r in records])
    return Retriever(vectors, records, embedder)


async def test_filters_by_feeling() -> None:
    records = [
        _record("proud-1", ["proud"], "you earned this proud moment"),
        _record("frustrated-1", ["frustrated"], "that was frustrating and hard"),
    ]
    r = _retriever(records)
    got = await r.retrieve("proud", None, k=3)
    assert [p["id"] for p in got] == ["proud-1"]  # frustrated passage filtered out


async def test_wildcard_matches_any_feeling() -> None:
    records = [
        _record("uni-1", ["*"], "it makes sense to feel this way"),
        _record("tired-1", ["tired"], "rest now you did enough"),
    ]
    r = _retriever(records)
    for feeling in ("proud", "anxious", "custom"):
        ids = {p["id"] for p in await r.retrieve(feeling, None, k=3)}
        assert "uni-1" in ids, f"wildcard should match {feeling}"


async def test_custom_feeling_gets_custom_and_wildcard() -> None:
    records = [
        _record("custom-1", ["custom"], "thank you for telling me"),
        _record("uni-1", ["*"], "you showed up and finished"),
        _record("proud-1", ["proud"], "you earned this"),
    ]
    r = _retriever(records)
    ids = {p["id"] for p in await r.retrieve("custom", "some free note", k=5)}
    assert ids == {"custom-1", "uni-1"}  # proud-only passage excluded


async def test_top_k_ordering_by_similarity() -> None:
    # The query shares the most words with target-1, fewer with target-2.
    records = [
        _record("target-1", ["proud"], "gentle sleepy proud rest kindness"),
        _record("target-2", ["proud"], "proud effort only"),
        _record("target-3", ["proud"], "completely unrelated vocabulary tokens"),
    ]
    r = _retriever(records)
    got = await r.retrieve("proud", "gentle sleepy proud rest kindness", k=2)
    assert [p["id"] for p in got] == ["target-1", "target-2"]
    assert len(got) == 2  # k respected


async def test_fewer_than_min_k_returns_what_there_is() -> None:
    records = [
        _record("anx-1", ["anxious"], "a slow breath, you are here now"),
        _record("proud-1", ["proud"], "you earned this"),
    ]
    r = _retriever(records)
    got = await r.retrieve("anxious", None, k=3)  # only one anxious passage
    assert [p["id"] for p in got] == ["anx-1"]


async def test_no_match_returns_empty() -> None:
    records = [_record("proud-1", ["proud"], "you earned this")]
    r = _retriever(records)
    assert await r.retrieve("tired", None, k=3) == []


async def test_empty_corpus_returns_empty() -> None:
    r = _retriever([])
    assert await r.retrieve("proud", None, k=3) == []


async def test_returned_passages_drop_internal_tags() -> None:
    records = [_record("proud-1", ["proud"], "you earned this", kind="phrasing")]
    r = _retriever(records)
    got = await r.retrieve("proud", None, k=3)
    assert got[0] == {
        "id": "proud-1",
        "kind": "phrasing",
        "text": "you earned this",
        "source": "test",
    }
    assert "feelings" not in got[0]  # filter tag never leaks to the prompt


async def test_from_files_round_trip(tmp_path: Path) -> None:
    records = [
        _record("proud-1", ["proud"], "you earned this"),
        _record("uni-1", ["*"], "it makes sense"),
    ]
    embedder = StubEmbedder()
    vectors = embedder.embed([r["text"] for r in records])
    meta = json.dumps({"model": "stub", "revision": "0", "dim": _DIM, "passages": records})
    index_path = tmp_path / "index.npz"
    np.savez(index_path, vectors=vectors, meta=np.array(meta))

    r = Retriever.from_files(index_path, embedder)
    got = await r.retrieve("proud", None, k=3)
    assert {p["id"] for p in got} == {"proud-1", "uni-1"}


async def test_from_files_rejects_dim_mismatch(tmp_path: Path) -> None:
    records = [_record("proud-1", ["proud"], "you earned this")]
    vectors = np.zeros((1, 8), dtype=np.float32)  # 8 != stub's 32
    meta = json.dumps({"model": "stub", "revision": "0", "dim": 8, "passages": records})
    index_path = tmp_path / "index.npz"
    np.savez(index_path, vectors=vectors, meta=np.array(meta))
    with pytest.raises(ValueError, match="dim"):
        Retriever.from_files(index_path, StubEmbedder())
