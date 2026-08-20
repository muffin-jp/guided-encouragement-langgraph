"""The embedding seam.

``Embedder`` is a tiny protocol — ``embed(texts) -> (n, dim) float32``, unit
vectors so cosine is a dot product. The retriever depends only on this protocol,
so a stub embedder drives the unit tests with no model and no network, and the
real model (or a ``fastembed``/onnx swap) drops in behind the same interface.

The one concrete implementation here is a **pinned local sentence-transformer**.
It loads *vendored* weights with ``local_files_only=True`` and never reaches the
network to embed — ``huggingface.co`` is not on the runtime allow-list. The
weights are fetched once at build time by :func:`fetch_model` (network open
there) into ``app/rag/model/`` and loaded from that path at startup.

The heavy ``sentence_transformers`` / ``torch`` import is deliberately lazy
(inside the methods), so importing this module — or running with
``RAG_ENABLED=0`` — costs nothing.
"""

# sentence_transformers is untyped and numpy's stubs leak Unknown under pyright
# strict, so this numeric/library boundary trips the "unknown" rules; our logic
# stays typed. Narrowly relax them for this module only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingImports=false
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from app.config import EMBED_MODEL, EMBED_MODEL_REVISION

RAG_DIR = Path(__file__).resolve().parent
# Vendored weights (git-ignored). Populated at build time by fetch_model / the
# `make build-index` target; the runtime loads from here with local_files_only.
MODEL_DIR = RAG_DIR / "model"


class Embedder(Protocol):
    """The retriever's only view of an embedding model.

    ``embed`` returns an ``(n, dim)`` float32 array of **L2-normalized** row
    vectors, so a cosine similarity is a plain dot product.
    """

    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Pinned local sentence-transformer, loaded from vendored weights.

    ``local_files_only=True`` guarantees the offline story: if the weights are
    not vendored the constructor fails loudly at startup rather than silently
    reaching for the network mid-request.
    """

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        # Lazy heavy import: keeps torch out of the process when RAG is off.
        from sentence_transformers import SentenceTransformer

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Embedding model not vendored at {model_dir}. Run `make build-index` "
                "at build time (network is open there); the runtime never downloads "
                "weights (huggingface.co is off the runtime allow-list)."
            )
        self._model = SentenceTransformer(str(model_dir), local_files_only=True)
        # get_embedding_dimension is the current name; fall back to the older
        # (now-deprecated) name via getattr so neither is a static reference.
        get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        self.dim = int(get_dim() or 0) if get_dim is not None else 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


def fetch_model(model_dir: Path = MODEL_DIR) -> Path:
    """Build-time only: vendor the pinned weights locally (network open here).

    Downloads the safetensors snapshot at the pinned revision into ``model_dir``
    so the runtime can load it with ``local_files_only=True``. Idempotent — a
    second call re-uses the already-vendored files.
    """
    from huggingface_hub import snapshot_download

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=EMBED_MODEL,
        revision=EMBED_MODEL_REVISION,
        local_dir=str(model_dir),
        # safetensors is safer to load than pickle; skip the redundant .bin.
        ignore_patterns=["*.bin", "*.h5", "*.ot", "*.msgpack"],
    )
    return model_dir


def load_embedder(model_dir: Path = MODEL_DIR) -> Embedder:
    """Startup factory for the real embedder (vendored weights, offline)."""
    return SentenceTransformerEmbedder(model_dir)
