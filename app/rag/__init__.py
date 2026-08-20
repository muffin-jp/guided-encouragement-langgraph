"""Retrieval grounding for the encouragement path.

A small, in-process retriever over a vetted, human-reviewed corpus
(``corpus.jsonl``). It fetches a few pre-approved passages filtered by the
player's feeling and hands them to ``generate`` as *grounding, not a script* —
the critique/judge guardrail remains the enforced gate.

This package is imported only when ``RAG_ENABLED`` is on; keep the module import
light (no torch/onnx at import time) so ``RAG_ENABLED=0`` pays nothing. The heavy
embedding model lives behind a lazy import in :mod:`app.rag.embedder`.
"""
