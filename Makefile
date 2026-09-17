.PHONY: build-index check-index check-classifier sync-classifier test lint types eval eval-dry

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
