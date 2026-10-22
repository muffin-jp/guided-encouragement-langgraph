"""The reviewed corpus: loading and validation.

``corpus.jsonl`` is the audit surface — every retrieval store is derived from it
offline (the pgvector table via :mod:`app.rag.seed_db`). This module parses and
strictly validates it, and reports per-feeling coverage. It imports nothing heavy
(no numpy, torch, or DB driver), so the seed, the evals, and the tests can all
share it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from app.config import Feeling

RAG_DIR = Path(__file__).resolve().parent
CORPUS_PATH = RAG_DIR / "corpus.jsonl"

_VALID_FEELINGS = {f.value for f in Feeling} | {"*"}
_VALID_KINDS = {"technique", "phrasing", "static_response"}
_REQUIRED_FIELDS = ("id", "feelings", "kind", "text", "source")


def load_corpus(path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    """Parse and validate ``corpus.jsonl`` into a list of passage records.

    Validation is strict on purpose — the corpus is the audit surface, so a
    malformed or mis-tagged passage should fail the build, not slip through.
    """
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{lineno}: invalid JSON: {exc}") from exc
        for field in _REQUIRED_FIELDS:
            if field not in raw:
                raise ValueError(f"{path.name}:{lineno}: missing required field {field!r}")
        pid = raw["id"]
        if pid in seen_ids:
            raise ValueError(f"{path.name}:{lineno}: duplicate id {pid!r}")
        seen_ids.add(pid)
        feelings = raw["feelings"]
        if not isinstance(feelings, list) or not feelings:
            raise ValueError(f"{path.name}:{lineno}: {pid}: 'feelings' must be a non-empty list")
        tags = cast("list[Any]", feelings)
        bad = [f for f in tags if f not in _VALID_FEELINGS]
        if bad:
            raise ValueError(f"{path.name}:{lineno}: {pid}: unknown feeling(s) {bad}")
        if raw["kind"] not in _VALID_KINDS:
            raise ValueError(f"{path.name}:{lineno}: {pid}: unknown kind {raw['kind']!r}")
        if not str(raw["text"]).strip():
            raise ValueError(f"{path.name}:{lineno}: {pid}: empty text")
        records.append(
            {
                "id": pid,
                "feelings": list(tags),
                "kind": raw["kind"],
                "text": raw["text"],
                "source": raw["source"],
            }
        )
    if not records:
        raise ValueError(f"{path.name}: corpus is empty")
    return records


def coverage(records: list[dict[str, Any]]) -> dict[str, int]:
    """Passage count reachable per feeling (incl. ``"*"`` wildcards) — a corpus
    gap check surfaced at build time."""
    counts: dict[str, int] = {}
    for feeling in Feeling:
        counts[feeling.value] = sum(
            1 for r in records if feeling.value in r["feelings"] or "*" in r["feelings"]
        )
    return counts
