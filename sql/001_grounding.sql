-- Retrieval-grounding schema for the pgvector backend (RAG_BACKEND=pgvector).
-- Apply once per database: `make db-migrate` (psql "$DATABASE_URL" -f this file),
-- then seed the rows from the reviewed corpus with `make db-seed`.
--
-- vector(384) is pinned to the embedder's dimension (all-MiniLM-L6-v2); the seed
-- asserts the embedder dim matches before writing. The table name matches the
-- RAG_PG_TABLE default; change both together if you rename it.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS grounding_passages (
    id        text        PRIMARY KEY,
    feelings  text[]      NOT NULL,      -- Feeling values; may contain '*'
    kind      text        NOT NULL,      -- technique | phrasing | static_response
    text      text        NOT NULL,      -- reviewed product content (verbatim)
    source    text        NOT NULL,
    embedding vector(384) NOT NULL       -- L2-normalized, so <=> is cosine distance
);

-- Filter index for the `feeling = ANY(feelings) OR '*' = ANY(feelings)` predicate.
CREATE INDEX IF NOT EXISTS grounding_feelings_gin
    ON grounding_passages USING gin (feelings);

-- ANN index for cosine. Demonstrative at the current corpus size — the planner
-- scans exactly for a few rows, which is correct; the index earns its keep as the
-- corpus grows.
CREATE INDEX IF NOT EXISTS grounding_embedding_hnsw
    ON grounding_passages USING hnsw (embedding vector_cosine_ops);
