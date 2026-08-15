"""Eval runner: runs the REAL compiled graph over the dataset and gates the build.

Ported from the TypeScript evals/run.ts. It calls the compiled graph's
``ainvoke`` (not a reimplementation) so the evals exercise exactly what
production ships — routing, the reflection loop, and the moderation interrupt
included. Distress cases pause at the interrupt and are auto-resumed here, which
is what makes the HITL path testable end-to-end.

Usage:
    uv run python evals/run.py            # real run (needs ANTHROPIC_API_KEY)
    uv run python evals/run.py --dry      # offline fixture, no API calls
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from app.graph.build import build_graph
from app.graph.state import GraphContext
from app.llm import build_anthropic_client
from evals.dry_client import create_dry_client
from evals.judge import judge_encouragement
from evals.report import print_console, render_markdown
from evals.thresholds import evaluate
from evals.types import CaseResult, EvalCase

load_dotenv()

CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "4"))
RETRIES = 3
STAGE_ID = "stage-eval"
DRY = "--dry" in sys.argv or os.environ.get("EVAL_DRY") == "1"

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evals" / "dataset.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"


def load_dataset() -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        cases.append(
            EvalCase(
                id=raw["id"],
                feeling=raw["feeling"],
                expected_path=raw["expectedPath"],
                category=raw["category"],
                free_text=raw.get("freeText"),
            )
        )
    return cases


def word_count(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def _is_transient(err: BaseException) -> bool:
    if isinstance(err, anthropic.APIConnectionError):
        return True
    if isinstance(err, anthropic.APIStatusError):
        return err.status_code in (408, 429) or err.status_code >= 500
    return False


async def _with_retry(coro_factory: Any) -> Any:
    last: BaseException | None = None
    for attempt in range(RETRIES):
        try:
            return await coro_factory()
        except BaseException as err:  # noqa: BLE001 - re-raised below
            last = err
            if not _is_transient(err) or attempt == RETRIES - 1:
                raise
            await asyncio.sleep(0.5 * 2**attempt)
    assert last is not None
    raise last


async def _run_graph(graph: Any, client: Any, case: EvalCase) -> tuple[str, str]:
    """Invoke the graph for one case, auto-resuming the moderation interrupt.

    Returns (path, text). Each case gets its own thread id so runs never share
    checkpointed state.
    """
    config = {"configurable": {"thread_id": f"eval-{case.id}-{uuid.uuid4().hex}"}}
    context = GraphContext(client=client)
    graph_input: dict[str, Any] = {
        "stage_id": STAGE_ID,
        "feeling": case.feeling,
        "free_text": case.free_text,
        "attempts": 0,
    }

    result = await graph.ainvoke(graph_input, config=config, context=context)
    if "__interrupt__" in result:
        # Distress path paused for moderation — the eval acts as the moderator
        # and approves so the support message is delivered (never withheld).
        from langgraph.types import Command

        result = await graph.ainvoke(
            Command(resume={"approve": True, "note": "auto-approved (eval)"}),
            config=config,
            context=context,
        )
    return result["path"], result.get("final_text", "")


async def run_case(graph: Any, client: Any, case: EvalCase) -> CaseResult:
    try:
        actual_path, text = await _with_retry(lambda: _run_graph(graph, client, case))
        words = word_count(text)
        is_encouragement = actual_path == "encouragement"

        judge = None
        if is_encouragement and text:
            scores = await _with_retry(
                lambda: judge_encouragement(client, case.feeling, case.free_text, text)
            )
            judge = scores

        return CaseResult(
            id=case.id,
            category=case.category,
            feeling=case.feeling,
            free_text=case.free_text,
            expected_path=case.expected_path,
            actual_path=actual_path,  # type: ignore[arg-type]
            path_correct=actual_path == case.expected_path,
            text=text,
            word_count=words,
            word_limit_ok=(words <= 40) if is_encouragement else None,
            non_empty=bool(text.strip()),
            judge=judge,
        )
    except BaseException as err:  # noqa: BLE001 - recorded as a case error
        return CaseResult(
            id=case.id,
            category=case.category,
            feeling=case.feeling,
            free_text=case.free_text,
            expected_path=case.expected_path,
            actual_path=case.expected_path,
            path_correct=False,
            text="",
            word_count=0,
            word_limit_ok=None,
            non_empty=False,
            judge=None,
            error=str(err),
        )


async def _map_pool(cases: list[EvalCase], graph: Any, client: Any) -> list[CaseResult]:
    """Bounded worker pool; preserves input order in the results list."""
    results: list[CaseResult | None] = [None] * len(cases)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    done = 0

    async def worker(i: int, case: EvalCase) -> None:
        nonlocal done
        async with semaphore:
            results[i] = await run_case(graph, client, case)
            done += 1
            print(f"\r  {done}/{len(cases)} cases complete", end="", flush=True)

    await asyncio.gather(*(worker(i, c) for i, c in enumerate(cases)))
    print()
    return [r for r in results if r is not None]


def build_client() -> Any:
    if DRY:
        print("Running in --dry mode: fixture client, no API calls.\n")
        return create_dry_client()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set. Set it to run the real eval, or pass "
            "--dry for an offline harness check.",
            file=sys.stderr,
        )
        sys.exit(1)
    return build_anthropic_client()


async def main() -> None:
    client = build_client()
    graph = build_graph()
    cases = load_dataset()
    print(f"Running {len(cases)} eval cases (concurrency {CONCURRENCY})...")

    results = await _map_pool(cases, graph, client)
    summary = evaluate(results)
    generated_at = datetime.now(timezone.utc).isoformat()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "latest.json").write_text(
        json.dumps(
            {
                "generatedAt": generated_at,
                "dry": DRY,
                "summary": asdict(summary),
                "results": [asdict(r) for r in results],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    label = f"{generated_at} (dry fixture — not real scores)" if DRY else generated_at
    (RESULTS_DIR / "latest.md").write_text(render_markdown(summary, label), encoding="utf-8")

    print_console(summary)
    print("Wrote evals/results/latest.json and latest.md")

    if hasattr(client, "close"):
        await client.close()

    if not summary.passed:
        print("\nThresholds not met — see failures above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
