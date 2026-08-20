"""Console + Markdown report writers.

Ported from the TypeScript evals/report.ts. Produces the compact console gate
and the human-readable evals/results/latest.md summary.
"""

from __future__ import annotations

from app.config import DISTRESS_MODEL, GENERATION_MODEL, JUDGE_MODEL
from evals.thresholds import Metric, Summary
from evals.types import CaseResult


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _render_metric(m: Metric) -> str:
    return _pct(m.value) if m.fmt == "percent" else f"{m.value:.2f}"


def _render_threshold(m: Metric) -> str:
    return _pct(m.threshold) if m.fmt == "percent" else f"{m.threshold:.2f}"


def _describe_failure(f: CaseResult) -> str:
    if f.error:
        return f"error: {f.error}"
    parts: list[str] = []
    if not f.path_correct:
        parts.append(f'routed to "{f.actual_path}", expected "{f.expected_path}"')
    if f.word_limit_ok is False:
        parts.append(f"over word limit ({f.word_count})")
    if f.judge is not None and f.judge.safety == "fail":
        parts.append(f"safety FAIL — {f.judge.safety_reason}")
    if f.expected_path == "encouragement" and f.judge is None and not f.error:
        parts.append("no judge score")
    return "; ".join(parts) or "flagged"


def print_console(summary: Summary) -> None:
    """Compact console output: metric gate, per-category accuracy, failures."""
    print("\n=== Guided Encouragement — Eval Results ===\n")

    label_w = max(len(m.label) for m in summary.metrics)
    print(f"  {'metric'.ljust(label_w)}  {'value':>8}  {'thresh':>8}  {'n':>3}  status")
    for m in summary.metrics:
        status = "PASS" if m.pass_ else "FAIL"
        print(
            f"  {m.label.ljust(label_w)}  {_render_metric(m):>8}  "
            f"{_render_threshold(m):>8}  {m.sample_size:>3}  {status}"
        )

    print()
    cat_w = max(len(c.category) for c in summary.by_category)
    for c in summary.by_category:
        print(f"  {c.category.ljust(cat_w)}  {c.count:>3} cases  {_pct(c.path_accuracy)}")

    if summary.failures:
        print(f"\n{len(summary.failures)} case(s) need attention:")
        for f in summary.failures:
            print(f"  - [{f.id}] {_describe_failure(f)}")

    result = "PASS ✅" if summary.passed else "FAIL ❌"
    print(f"\nOverall path accuracy: {_pct(summary.overall_path_accuracy)}  |  Result: {result}\n")


def retrieval_report(results: list[CaseResult]) -> str:
    """How often 0 / 2 / 3 passages were retrieved, by feeling.

    A quality aid, not a gate: it makes corpus gaps visible (feelings that keep
    retrieving 0 want more passages). Returns a Markdown block that reads fine in
    the console too. Only encouragement cases carry a grounding count.
    """
    rows = [r for r in results if r.grounding_count is not None]
    if not rows:
        return (
            "## Retrieval grounding\n\n"
            "RAG disabled (or no grounded cases) — no retrieval-count report.\n"
        )
    counts_seen = sorted({r.grounding_count for r in rows if r.grounding_count is not None})
    headers = ["Feeling", "Cases", *[f"{c} passages" for c in counts_seen], "Mean"]

    def _row(label: str, subset: list[CaseResult]) -> list[str]:
        cells = [label, str(len(subset))]
        cells += [str(sum(1 for r in subset if r.grounding_count == c)) for c in counts_seen]
        mean = sum(r.grounding_count or 0 for r in subset) / len(subset)
        cells.append(f"{mean:.1f}")
        return cells

    body = [_row(f, [r for r in rows if r.feeling == f]) for f in sorted({r.feeling for r in rows})]
    body.append(_row("all", rows))
    return "## Retrieval grounding\n\n" + _md_table(headers, body) + "\n"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = f"| {' | '.join(headers)} |"
    sep = f"| {' | '.join('---' for _ in headers)} |"
    body = "\n".join(f"| {' | '.join(r)} |" for r in rows)
    return f"{head}\n{sep}\n{body}"


def render_markdown(summary: Summary, generated_at: str) -> str:
    """The human-readable summary written to evals/results/latest.md."""
    lines: list[str] = [
        "# Guided Encouragement — Eval Results",
        "",
        f"- Generated: {generated_at}",
        f"- Cases: {summary.total}",
        f"- Generation model: `{GENERATION_MODEL}`",
        f"- Distress classifier: `{DISTRESS_MODEL}`",
        f"- Judge: `{JUDGE_MODEL}` (temperature 0)",
        f"- **Result: {'PASS ✅' if summary.passed else 'FAIL ❌'}**",
        "",
        "## Thresholds",
        "",
        _md_table(
            ["Metric", "Value", "Threshold", "n", "Status"],
            [
                [
                    m.label,
                    _render_metric(m),
                    _render_threshold(m),
                    str(m.sample_size),
                    "✅" if m.pass_ else "❌",
                ]
                for m in summary.metrics
            ],
        ),
        "",
        "## By category",
        "",
        _md_table(
            ["Category", "Cases", "Path accuracy"],
            [[c.category, str(c.count), _pct(c.path_accuracy)] for c in summary.by_category],
        ),
        "",
        f"Overall path accuracy: **{_pct(summary.overall_path_accuracy)}**",
        "",
        "## Failures",
        "",
    ]
    if not summary.failures:
        lines.append("None — every case met the bar. 🌸")
    else:
        lines.append(
            _md_table(
                ["Case", "Category", "Detail"],
                [[f.id, f.category, _describe_failure(f)] for f in summary.failures],
            )
        )
    lines.append("")
    return "\n".join(lines)
