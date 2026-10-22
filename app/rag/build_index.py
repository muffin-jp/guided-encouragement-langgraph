"""Offline, deterministic index builder for the vetted corpus.

Reads ``corpus.jsonl``, embeds each passage's ``text`` once with the pinned local
model, and writes ``index.npz`` (vectors) plus a parallel metadata list. Same
corpus + same pinned model → same index. This runs at build time / in CI, never
at request time.

Usage::

    uv run python -m app.rag.build_index          # (re)build index.npz
    uv run python -m app.rag.build_index --check   # CI: committed index up to date?

``--check`` rebuilds into memory and compares against the committed ``index.npz``
so a corpus edit can't land without its index. The passage metadata (text,
feelings, order) must match exactly; the vectors must match within a small
tolerance (the model is deterministic, the tolerance only absorbs cross-platform
float noise).
"""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary; our
# own logic stays typed. Narrowly relax the "unknown" rules for this module only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from app.config import EMBED_MODEL, EMBED_MODEL_REVISION
from app.rag.corpus import CORPUS_PATH, coverage, load_corpus

RAG_DIR = Path(__file__).resolve().parent
INDEX_PATH = RAG_DIR / "index.npz"


def build(records: list[dict[str, Any]]) -> tuple[np.ndarray, str]:
    """Embed every passage and return (vectors, metadata-JSON)."""
    # Real model, loaded from vendored weights (fetch first if absent).
    from app.rag.embedder import MODEL_DIR, SentenceTransformerEmbedder, fetch_model

    if not MODEL_DIR.exists():
        print(f"Vendoring pinned weights into {MODEL_DIR} (build-time network) ...")
        fetch_model()
    embedder = SentenceTransformerEmbedder()
    vectors = embedder.embed([r["text"] for r in records])
    meta = json.dumps(
        {
            "model": EMBED_MODEL,
            "revision": EMBED_MODEL_REVISION,
            "dim": int(vectors.shape[1]),
            "passages": records,
        }
    )
    return vectors.astype(np.float32), meta


def write_index(vectors: np.ndarray, meta: str, path: Path = INDEX_PATH) -> None:
    np.savez(path, vectors=vectors, meta=np.array(meta))


def _check(records: list[dict[str, Any]]) -> int:
    """Return 0 if the committed index matches a fresh build, else 1."""
    if not INDEX_PATH.exists():
        print("index.npz is missing — run `make build-index`.", file=sys.stderr)
        return 1
    fresh_vectors, fresh_meta = build(records)
    with np.load(INDEX_PATH, allow_pickle=False) as data:
        committed_vectors = data["vectors"].astype(np.float32)
        committed_meta = json.loads(str(data["meta"].item()))
    fresh_passages = json.loads(fresh_meta)["passages"]
    if committed_meta.get("passages") != fresh_passages:
        print(
            "index.npz metadata is stale: corpus.jsonl changed but the index was "
            "not rebuilt. Run `make build-index` and commit index.npz.",
            file=sys.stderr,
        )
        return 1
    if committed_vectors.shape != fresh_vectors.shape or not np.allclose(
        committed_vectors, fresh_vectors, atol=1e-4, rtol=1e-4
    ):
        print(
            "index.npz vectors do not match a fresh build of corpus.jsonl. Run "
            "`make build-index` and commit index.npz.",
            file=sys.stderr,
        )
        return 1
    print("index.npz is up to date with corpus.jsonl.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed index matches a fresh build (CI); do not write",
    )
    args = parser.parse_args()

    records = load_corpus()
    print(f"Loaded {len(records)} reviewed passage(s) from {CORPUS_PATH.name}.")
    cov = coverage(records)
    print("Coverage by feeling: " + ", ".join(f"{k}={v}" for k, v in cov.items()))
    gaps = [k for k, v in cov.items() if v == 0]
    if gaps:
        print(f"WARNING: no passages reachable for feeling(s): {gaps}", file=sys.stderr)

    if args.check:
        sys.exit(_check(records))

    vectors, meta = build(records)
    write_index(vectors, meta)
    print(f"Wrote {INDEX_PATH.name}: {vectors.shape[0]} vectors x {vectors.shape[1]} dims.")


if __name__ == "__main__":
    main()
