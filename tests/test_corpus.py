"""Corpus loader tests: the committed corpus loads, and malformed passages fail loudly.

``corpus.jsonl`` is the audit surface every retrieval store is seeded from, so a
mis-tagged or malformed passage must fail the seed rather than slip through.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.config import Feeling
from app.rag.corpus import coverage, load_corpus


def _write(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _row(pid: str = "p-1", **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": pid,
        "feelings": ["proud"],
        "kind": "technique",
        "text": "some reviewed text",
        "source": "test",
    }
    row.update(overrides)
    return row


def test_committed_corpus_loads_and_covers_every_feeling() -> None:
    records = load_corpus()
    assert records
    cov = coverage(records)
    assert set(cov) == {f.value for f in Feeling}
    assert all(n > 0 for n in cov.values())


def test_valid_rows_round_trip(tmp_path: Path) -> None:
    records = load_corpus(_write(tmp_path, [_row("a"), _row("b", feelings=["*"])]))
    assert [r["id"] for r in records] == ["a", "b"]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_row("a"), _row("a")], "duplicate id"),
        ([_row(feelings=["ecstatic"])], "unknown feeling"),
        ([_row(feelings=[])], "non-empty list"),
        ([_row(kind="essay")], "unknown kind"),
        ([_row(text="   ")], "empty text"),
        ([{k: v for k, v in _row().items() if k != "source"}], "missing required field"),
    ],
)
def test_malformed_rows_are_rejected(
    tmp_path: Path, rows: list[dict[str, Any]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_corpus(_write(tmp_path, rows))


def test_invalid_json_and_empty_corpus_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_corpus(bad)
    with pytest.raises(ValueError, match="corpus is empty"):
        load_corpus(_write(tmp_path, []))
