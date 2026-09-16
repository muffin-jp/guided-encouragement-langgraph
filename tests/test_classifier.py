"""The local distress classifier: what it refuses to load, and what it may decide.

Two properties carry the whole integration. With no classifier injected the graph
is exactly today's. With one injected, only a *confident* answer skips the LLM —
anything else, including the classifier failing, still reaches it.
"""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from anthropic import AsyncAnthropic

from app.classifier.check import problems
from app.classifier.local import (
    ARTIFACT_DIR,
    ArtifactSpec,
    ClassifierLoadError,
    LocalDistressClassifier,
    read_artifact,
    route_for,
)
from app.config import (
    CLASSIFIER_ARTIFACT_SHA256,
    CLASSIFIER_EQUIVALENT_EMBEDDER_REVISIONS,
    DISTRESS_MODEL,
)
from app.graph.build import build_graph
from app.graph.state import GraphContext
from tests.fakes import FakeClassifier, FakeClient

REPO = Path(__file__).resolve().parents[1]


def copy_artifact(tmp_path: Path) -> Path:
    for name in ("model.npz", "model.json"):
        shutil.copy(ARTIFACT_DIR / name, tmp_path / name)
    return tmp_path


def edit_json(directory: Path, **changes: Any) -> None:
    document = json.loads((directory / "model.json").read_text())
    for dotted, value in changes.items():
        target = document
        *parents, leaf = dotted.split("__")
        for key in parents:
            target = target[key]
        target[leaf] = value
    (directory / "model.json").write_text(json.dumps(document))


# --- the committed artifact ------------------------------------------------------------


def test_the_committed_artifact_is_the_evaluated_one() -> None:
    spec = read_artifact()
    assert spec.sha256 == CLASSIFIER_ARTIFACT_SHA256
    assert 0.0 <= spec.low <= spec.high <= 1.0
    assert spec.coef.shape == (spec.embed_dim,)


def test_the_committed_artifact_uses_a_verified_equivalent_embedder() -> None:
    assert read_artifact().embed_revision in CLASSIFIER_EQUIVALENT_EMBEDDER_REVISIONS


def test_the_ci_guard_passes_on_the_committed_artifact() -> None:
    assert problems() == []


def test_the_classifier_is_off_by_default() -> None:
    """Checked in a fresh interpreter, so no test's environment leaks in."""
    env = {k: v for k, v in os.environ.items() if k != "CLASSIFIER_ENABLED"}
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from app.config import CLASSIFIER_ENABLED; print(CLASSIFIER_ENABLED)",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


# --- what the loader refuses -----------------------------------------------------------


def test_a_modified_artifact_is_refused(tmp_path: Path) -> None:
    directory = copy_artifact(tmp_path)
    (directory / "model.json").write_text((directory / "model.json").read_text() + " ")
    with pytest.raises(ClassifierLoadError, match="not the evaluated artifact"):
        read_artifact(directory)


def test_an_unverified_embedder_revision_is_refused(tmp_path: Path) -> None:
    directory = copy_artifact(tmp_path)
    edit_json(directory, embedder__revision="0" * 40)
    with pytest.raises(ClassifierLoadError, match="verified-equivalent"):
        read_artifact(directory, expected_sha256=None)


def test_an_artifact_that_reads_the_feeling_chip_is_refused(tmp_path: Path) -> None:
    directory = copy_artifact(tmp_path)
    edit_json(directory, features__include_feeling=True)
    with pytest.raises(ClassifierLoadError, match="feeling chip"):
        read_artifact(directory, expected_sha256=None)


def test_an_artifact_without_thresholds_is_refused(tmp_path: Path) -> None:
    directory = copy_artifact(tmp_path)
    edit_json(directory, thresholds=None)
    with pytest.raises(ClassifierLoadError, match="no fitted thresholds"):
        read_artifact(directory, expected_sha256=None)


def test_inverted_thresholds_are_refused(tmp_path: Path) -> None:
    directory = copy_artifact(tmp_path)
    edit_json(directory, thresholds__low=0.9, thresholds__high=0.1)
    with pytest.raises(ClassifierLoadError, match="invalid thresholds"):
        read_artifact(directory, expected_sha256=None)


def test_weights_that_do_not_fit_the_embedder_are_refused(tmp_path: Path) -> None:
    directory = copy_artifact(tmp_path)
    np.savez(directory / "model.npz", coef=np.zeros(8), intercept=np.zeros(1))
    with pytest.raises(ClassifierLoadError, match="coef shape"):
        read_artifact(directory, expected_sha256=None)


def test_missing_files_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ClassifierLoadError, match="missing"):
        read_artifact(tmp_path, expected_sha256=None)


# --- routing -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.05, "skip-llm"),
        (0.1, "escalate"),
        (0.5, "escalate"),
        (0.8, "escalate"),
        (0.95, "support"),
    ],
)
def test_routes_by_band_and_boundaries_escalate(score: float, expected: str) -> None:
    assert route_for(score, 0.1, 0.8) == expected


@pytest.mark.parametrize("bad", [math.nan, math.inf, -0.1, 1.1, None, "0.5", True])
def test_anything_that_is_not_a_probability_escalates(bad: object) -> None:
    assert route_for(bad, 0.1, 0.8) == "escalate"


class AxisEmbedder:
    """Returns a unit vector along axis 0 scaled by a word count — fully offline."""

    def __init__(self, dim: int = 4, shape: tuple[int, ...] | None = None) -> None:
        self.dim = dim
        self._shape = shape

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._shape is not None:
            return np.zeros(self._shape, dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i, 0] = float(text.count("sad"))
        return out


def spec(dim: int = 4) -> ArtifactSpec:
    coef = np.zeros(dim)
    coef[0] = 4.0
    return ArtifactSpec(coef, -2.0, 0.1, 0.8, dim, "x" * 40, "s" * 64)


def test_decide_scores_and_routes() -> None:
    classifier = LocalDistressClassifier(spec(), AxisEmbedder())
    calm, heavy = classifier.decide("fine"), classifier.decide("sad sad sad")
    assert calm.route == "escalate" and calm.score == pytest.approx(1 / (1 + math.exp(2.0)))
    assert heavy.route == "support"


def test_an_embedding_of_the_wrong_shape_escalates() -> None:
    classifier = LocalDistressClassifier(spec(), AxisEmbedder(shape=(2, 4)))
    decision = classifier.decide("anything")
    assert (decision.route, decision.score) == ("escalate", None)


def test_an_embedder_of_the_wrong_width_is_refused() -> None:
    with pytest.raises(ClassifierLoadError, match="expects 4"):
        LocalDistressClassifier(spec(4), AxisEmbedder(dim=8))


# --- the graph ---------------------------------------------------------------------------


async def run(
    client: FakeClient, classifier: FakeClassifier | None, free_text: str | None
) -> dict[str, Any]:
    graph: Any = build_graph(enable_moderation=False, enable_rag=False)
    context = GraphContext(client=cast(AsyncAnthropic, client), classifier=cast("Any", classifier))
    return await graph.ainvoke(
        {"stage_id": "s1", "feeling": "tired", "free_text": free_text, "attempts": 0},
        config={"configurable": {"thread_id": uuid.uuid4().hex}},
        context=context,
    )


@pytest.mark.asyncio
async def test_without_a_classifier_the_llm_decides_as_today() -> None:
    client = FakeClient()
    result = await run(client, None, "a note")
    assert client.calls_to(DISTRESS_MODEL) == 1
    assert result["distress_source"] == "llm"


@pytest.mark.asyncio
async def test_a_confident_negative_skips_the_llm() -> None:
    client, classifier = FakeClient(), FakeClassifier(route="skip-llm", score=0.01)
    result = await run(client, classifier, "a note")
    assert client.calls_to(DISTRESS_MODEL) == 0
    assert (result["path"], result["distress_source"]) == ("encouragement", "local")


@pytest.mark.asyncio
async def test_a_confident_positive_goes_straight_to_support() -> None:
    client, classifier = FakeClient(), FakeClassifier(route="support", score=0.99)
    result = await run(client, classifier, "a note")
    assert client.calls_to(DISTRESS_MODEL) == 0
    assert (result["path"], result["distress_source"]) == ("support", "local")


@pytest.mark.asyncio
async def test_an_uncertain_note_still_reaches_the_llm() -> None:
    client, classifier = FakeClient(distress=True), FakeClassifier(route="escalate")
    result = await run(client, classifier, "a note")
    assert client.calls_to(DISTRESS_MODEL) == 1
    assert (result["path"], result["distress_source"]) == ("support", "llm")


@pytest.mark.asyncio
async def test_a_failing_classifier_falls_back_to_the_llm() -> None:
    client, classifier = FakeClient(), FakeClassifier(raises=True)
    result = await run(client, classifier, "a note")
    assert classifier.calls == 1
    assert client.calls_to(DISTRESS_MODEL) == 1
    assert result["path"] == "encouragement"


@pytest.mark.asyncio
async def test_a_chip_only_request_reaches_neither_classifier() -> None:
    client, classifier = FakeClient(), FakeClassifier(route="support")
    result = await run(client, classifier, None)
    assert classifier.calls == 0
    assert client.calls_to(DISTRESS_MODEL) == 0
    assert result["path"] == "encouragement"
