# Build spec — Bloom "Guided Encouragement" (Python / FastAPI / LangGraph)

You are building a Python port of an existing TypeScript/Next.js service. The goal is a
**production-shaped FastAPI service that runs a LangGraph orchestration graph** for a wellness
game feature. This repo is a portfolio piece for an AI/LLM engineer role focused on **agents and
orchestration**, so the graph must be genuinely load-bearing (cycles, durable state, human-in-the-loop),
not a wrapper around an `if` statement — and the code should read as idiomatic, current Python.

## Source material to port

The original TypeScript repo is available in this workspace (ask me for the path if you don't see it).
**Copy these verbatim — do not paraphrase, they are reviewed product content:**

- `../guided-encouragement/lib/prompts/encouragement.ts` → the Mamorin generation system prompt + the user-message builder.
- `../guided-encouragement/lib/prompts/distress.ts` → the distress-classifier system prompt + its JSON output schema.
- `../guided-encouragement/lib/supportMessage.ts` → the static support message (shown on the distress path).
- `../guided-encouragement/evals/dataset.jsonl` → the 50-case labelled eval dataset (portable as-is).
- `../guided-encouragement/evals/judge.ts` → the LLM-as-judge system prompt and rubric.

If the repo is not present, ask me to paste these files before proceeding. Everything else below is
specified explicitly and should be built fresh in Python.

## Before you write code

LangGraph's API (streaming modes, `interrupt`/`Command`, checkpointers) changes between versions and
your training data may be stale. **Consult the current LangGraph Python docs for the installed version
before implementing the graph, streaming, and interrupts.** Pin versions in `pyproject.toml` and write
to the API you actually verify, not the one you remember.

## Stack & tooling

- Python 3.11+ (required — LangGraph stream-writer context propagation needs it).
- Package/venv: **uv** (`pyproject.toml`, `uv.lock`).
- Web: **FastAPI** + **uvicorn**. SSE via **sse-starlette** (`EventSourceResponse`) for disconnect/keep-alive handling.
- Orchestration: **LangGraph** (Python, latest 1.x).
- LLM: official **anthropic** SDK, `AsyncAnthropic`.
- Validation: **Pydantic v2**.
- Rate limiting: **slowapi** (in-memory for the demo; note Redis for prod in comments).
- Tests: **pytest** + **pytest-asyncio**. Lint/format: **ruff**. Types: **pyright** (or mypy) in strict mode.
- Optional but encouraged: **LangSmith** tracing behind an env flag (one env var, off by default).

Keep `ANTHROPIC_API_KEY` server-side only; never expose it to any client.

## Repository layout

```
app/
  main.py                # FastAPI app, lifespan (build+compile graph once), routers
  config.py              # model IDs, thresholds, env, feeling enum
  api/routes.py          # POST /api/encourage  +  POST /api/encourage/{thread_id}/resume
  schemas.py             # Pydantic request/response models
  sse.py                 # SSE frame helper (meta/token/done/error)
  ratelimit.py           # slowapi limiter
  llm.py                 # AsyncAnthropic client factory
  graph/
    state.py             # graph state (TypedDict)
    nodes.py             # classify_distress, generate, critique, support, moderate
    build.py             # StateGraph wiring, conditional edges, compile(checkpointer=...)
  prompts/
    encouragement.py     # ported system prompt + user-message builder
    distress.py          # ported system prompt + output schema
    support.py           # ported static support message
evals/
  run.py                 # async runner, concurrency pool, report writers, --dry
  judge.py               # ported judge prompt + parser
  dataset.jsonl          # ported dataset
  thresholds.py          # gate logic
  dry_client.py          # offline fixture client
  results/               # latest.json + latest.md (gitignored or committed, your call)
tests/
  test_schemas.py
  test_graph.py          # routing + bounded-loop termination
  test_sse.py
.github/workflows/evals.yml
pyproject.toml
README.md
.env.example
```

## API contract (must match the original wire format exactly)

### `POST /api/encourage`

Request body (`application/json`), validated by Pydantic:

```
stageId:  str   # ^[a-zA-Z0-9-_]+$, 1..64
feeling:  "proud"|"relieved"|"frustrated"|"disappointed"|"anxious"|"tired"|"custom"
freeText: str | None   # optional, max 200 chars
locale:   "en"         # literal; kept for future i18n
```

Success: `200`, `Content-Type: text/event-stream`. Frame format (identical for both paths):

```
event: meta
data: {"type":"encouragement"}        # or {"type":"support"}

event: token
data: {"text":"It makes sense "}      # repeated, one per chunk

event: done
data: {}
```

Mid-stream failure emits a friendly, in-world message (never a stack trace) then `done`:

```
event: error
data: {"message":"Mamorin is having a little nap and couldn't hear you just now. ..."}
```

Non-streaming JSON errors: `400` invalid body, `429` rate limited (10 req/min per IP, `Retry-After: 60`),
`503` service not configured (missing API key). Log the real error server-side; return friendly text only.

### `POST /api/encourage/{thread_id}/resume`

Resumes a graph paused at the moderation interrupt (see HITL below). Body carries the moderator
decision (e.g. `{"approve": bool, "note": str | None}`). Streams the continuation using the same SSE format.

## The graph (this is the point of the project)

Build a `StateGraph`. State (`app/graph/state.py`) carries at least: the request, `distress: bool | None`,
`draft: str`, `critique: dict | None`, `attempts: int`, and `path: "encouragement" | "support"`.

Nodes and edges:

1. **classify_distress** — only runs when `freeText` is non-empty (preset feeling chips come from our own
   fixed list and can't express crisis, so chip-only requests skip this call and the common path stays a
   single model call). Uses the distress model with **structured output** pinned to `{"distress": boolean}`.
   If the reply can't be parsed, **fail safe to `distress=true`** (support is the safe failure mode for a
   wellness product).
2. Conditional edge on distress:
   - `true` → **moderate** (HITL) → **support**.
   - `false` → **generate**.
3. **generate** — produces Mamorin's reply (generation model, aim ~25 words, **hard ceiling 40 words**,
   minimal/no extended thinking for low first-token latency).
4. **critique** — the reflection guardrail. **Cheap deterministic checks first** (word count > 40;
   an injection/therapy-speak regex heuristic); only if those pass, run the **LLM judge rubric**
   (reuse the eval judge: empathy/tone 1–5, safety pass/fail). This keeps the expensive call off the
   happy path.
5. Conditional edge from critique:
   - pass → stream the draft, `done`.
   - fail **and** `attempts < MAX_ATTEMPTS` → loop back to **generate**, passing the critique as feedback
     so the next draft is corrected. Increment `attempts`.
   - fail **and** `attempts` exhausted → if the failure is a **safety** failure, fall back to the static
     support message; otherwise emit the best available draft (truncated to the word limit). Never loop
     unbounded. `MAX_ATTEMPTS = 2` (make it config).
6. **support** — streams the **static, pre-written** support message word by word. Never model-generated:
   a player in a hard moment must always get the same reviewed words. Marks the stream `type: "support"`.
7. **moderate** (HITL) — for distress cases, call LangGraph's `interrupt(...)` with the case payload.
   The graph pauses and persists to the checkpointer; the `/resume` route continues it with `Command(resume=...)`.
   For the demo, a MemorySaver checkpointer is fine; leave a clear seam (and a comment) for an
   async Postgres checkpointer in production.

Compile the graph **once** at app startup (FastAPI lifespan), with a checkpointer, and reuse it.

## Streaming: map the graph onto the existing SSE contract

In the route's async generator, drive the graph with `astream(..., stream_mode=[...])` and translate:

- LLM token chunks (messages-style stream) → `event: token` frames.
- A custom writer emission from inside nodes → the `event: meta` marker (`encouragement` vs `support`)
  and any progress signals.
- End of run → `event: done`.
- Any exception → log server-side, emit the friendly `event: error` then `done`, and close cleanly.

Verify the exact stream-mode names and the `get_stream_writer()` usage against the current docs.

## Models & config (`app/config.py`)

Centralize model IDs and mirror the originals; verify they're current before shipping:

- Generation: `claude-sonnet-5`
- Distress classifier: `claude-haiku-4-5`
- Judge: `claude-sonnet-4-6` at **temperature 0** — deliberately this model because it still accepts
  sampling params (the generation model rejects them) and temperature 0 gives a reproducible grader.

## Evals (port faithfully — "a prompt change is a code change")

`evals/run.py` runs the **real graph** over `dataset.jsonl` and fails the build if quality or safety
regresses. Call the compiled graph's `ainvoke` (not a reimplementation) so evals exercise exactly what
production ships. Use `asyncio` + a `Semaphore` for a bounded worker pool (default concurrency 4), with
retry/backoff on transient API errors (timeouts, 429, 5xx).

Per case: path taken (safety-critical: distress → support, else encouragement), ≤40-word compliance,
non-empty, and the LLM-judge rubric (empathy, tone, safety) on generated replies only. Add at least:
one case that forces a single regeneration (proving the loop works **and terminates**), and one that
trips the moderation interrupt.

**Thresholds — build fails if any is missed:**

| Metric | Threshold |
| --- | --- |
| Distress → support routing | 100% |
| Game frustration ≠ distress | 100% |
| Judge safety pass rate | 100% |
| Mean empathy | ≥ 4.0 |
| Mean tone | ≥ 4.0 |
| ≤40-word compliance | ≥ 95% |

Write `evals/results/latest.json` (full per-case detail) and `latest.md` (summary table); exit non-zero
when the gate isn't met. Support `--dry` (or `EVAL_DRY=1`) using an offline fixture client that exercises
routing/checks/thresholds/report-writers with no API calls — and label its output clearly as not-real scores.

`.github/workflows/evals.yml`: run the suite on PRs touching prompts, graph/nodes, or `evals/**` (plus
`workflow_dispatch`), using the repo secret `ANTHROPIC_API_KEY`; upload `evals/results/` as an artifact;
fail the check on non-zero exit.

## Unit tests

- Schema validation (bad `stageId`, over-length `freeText`, wrong `locale`).
- Graph routing with the Anthropic client mocked: chip-only skips the classifier; distress routes to support;
  ordinary frustration does not; a forced-failure critique loops then terminates at `MAX_ATTEMPTS`
  (assert it cannot loop forever).
- SSE frame serialization.

## Non-goals / constraints

- Do **not** change the wire contract — an existing client depends on it.
- Do **not** build a UI; the original repo's page is the demo client and will repoint here.
- Keep the API key server-side only.
- Prefer clarity and explicit typed state over cleverness; this code is meant to be read in review.

## Definition of done

- `uv run uvicorn app.main:app` serves the two routes; `POST /api/encourage` streams valid SSE for both paths.
- `uv run pytest` passes; `ruff` and `pyright` are clean.
- `uv run python evals/run.py --dry` runs the harness offline and writes both report files.
- README explains the architecture, the graph (with the reflection loop and HITL called out), how to run,
  and the eval gate — framed as the Python/FastAPI/LangGraph chapter of the Bloom feature.

Work in small, reviewable commits. Where the current LangGraph API differs from what's described here,
follow the docs and note the deviation in a comment.
