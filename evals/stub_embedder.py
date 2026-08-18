"""Offline stub embedder for the ``--dry`` eval.

Like the dry client, this is NOT a model: it is a deterministic hashed
bag-of-words that lets the dry eval exercise the retrieve node and the
retrieval-count report over the real corpus with no weights and no network. A
real eval run uses the pinned local model via ``app.rag.embedder.load_embedder``.
"""

# numpy's stubs leak Unknown under pyright strict; relax the "unknown" rules here.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

_DIM = 64


class StubEmbedder:
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
