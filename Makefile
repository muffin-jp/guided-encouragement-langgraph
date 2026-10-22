.PHONY: fetch-model build-index check-index check-classifier sync-classifier test lint types eval eval-dry \
	db-up db-down db-migrate db-migrate-url db-seed db-check db-setup

# Local Postgres+pgvector for retrieval grounding. Override for other envs.
DATABASE_URL ?= postgresql://bloom:bloom@localhost:5432/bloom

# Vendor the pinned embedding weights into app/rag/model/ (build-time network;
# git-ignored). The seed, the real eval, and the classifier parity tests need them;
# the runtime loads them with local_files_only and never downloads.
fetch-model:
	uv run python -m app.rag.embedder

# Vendor the pinned embedding weights (build-time network) and (re)build the
# committed retrieval index from the reviewed corpus. Run this after editing
# app/rag/corpus.jsonl, then commit app/rag/index.npz.
build-index:
	uv run python -m app.rag.build_index

# CI guard: fail if the committed index.npz is out of date with corpus.jsonl —
# a corpus edit can't land without its rebuilt index.
check-index:
	uv run python -m app.rag.build_index --check

# CI guard: the committed classifier artifact is byte-identical to the one whose
# test-set evaluation is recorded upstream, and loadable. No weights, no network.
check-classifier:
	uv run python -m app.classifier.check

# Copy the artifact from a sibling bloom-distress-classifier checkout, then guard
# it. A new artifact fails the guard until CLASSIFIER_ARTIFACT_SHA256 is updated —
# which should only happen after it has been evaluated upstream.
sync-classifier:
	cp ../bloom-distress-classifier/artifacts/model.npz ../bloom-distress-classifier/artifacts/model.json ../bloom-distress-classifier/artifacts/routes.json app/classifier/
	uv run python -m app.classifier.check

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

types:
	uv run pyright

# Real eval suite (needs ANTHROPIC_API_KEY); eval-dry is the offline harness check.
eval:
	uv run python evals/run.py

eval-dry:
	uv run python evals/run.py --dry

# --- pgvector retrieval store ------------------------------------------------
# db-up starts local Postgres and waits for its healthcheck; db-migrate applies the
# schema (psql runs inside the container, so no host psql needed); db-seed loads the
# corpus rows; db-check is the guard that the live table matches corpus.jsonl.
db-up:
	docker compose up -d --wait db

db-down:
	docker compose down

db-migrate:
	docker compose exec -T db psql -U bloom -d bloom -f - < sql/001_grounding.sql

# Same schema, applied with a host psql straight to DATABASE_URL — for CI service
# containers and any environment that isn't the local compose DB.
db-migrate-url:
	psql "$(DATABASE_URL)" -v ON_ERROR_STOP=1 -f sql/001_grounding.sql

db-seed:
	DATABASE_URL="$(DATABASE_URL)" uv run python -m app.rag.seed_db

db-check:
	DATABASE_URL="$(DATABASE_URL)" uv run python -m app.rag.seed_db --check

# One-shot local setup: start Postgres, apply the schema, seed from the corpus.
db-setup: db-up db-migrate db-seed
