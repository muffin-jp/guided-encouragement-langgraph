"""Retrieval grounding for the encouragement path.

Retrieval over a vetted, human-reviewed corpus (``corpus.jsonl``), stored in
Postgres + pgvector and seeded offline from that file (:mod:`app.rag.seed_db`).
It fetches a few pre-approved passages filtered by the player's feeling and hands
them to ``generate`` as *grounding, not a script* — the critique/judge guardrail
remains the enforced gate. The query is embedded locally, in-process.

This package is imported only when ``RAG_ENABLED`` is on; keep the module import
light (no torch/onnx at import time) so ``RAG_ENABLED=0`` pays nothing. The heavy
embedding model lives behind a lazy import in :mod:`app.rag.embedder`.
"""
