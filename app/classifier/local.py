"""The local distress classifier: a front door to the LLM check, not a replacement.

Trained and evaluated in ``bloom-distress-classifier``; this module only serves it.
Each free-text note gets a score and one of three routes:

* ``skip-llm`` — confidently not distress. The note goes to encouragement without
  calling the classifier model.
* ``escalate`` — anything else. ``claude-haiku-4-5`` decides, exactly as it does
  today for every note.
* ``support`` — confidently distress. Straight to the reviewed support message.

Serving is one embedding, one dot product and a sigmoid, in numpy. Nothing is
unpickled.

Loading is strict; failing is not
---------------------------------
The artifact is refused unless it is byte-identical to the one whose single
test-set evaluation is recorded upstream, was built on this service's embedder
weights, reads no feeling features, and carries fitted thresholds. A refused or
failed load leaves the classifier ``None``, and every note escalates — today's
behaviour. A score that is not a finite probability escalates as well.
"""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary; our
# own logic stays typed. The same narrow relaxation app/rag/retriever.py uses.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np

from app.config import (
    CLASSIFIER_ARTIFACT_SHA256,
    CLASSIFIER_EQUIVALENT_EMBEDDER_REVISIONS,
    EMBED_MODEL,
    EMBED_MODEL_REVISION,
)

if TYPE_CHECKING:
    from app.rag.embedder import Embedder

__all__ = [
    "ARTIFACT_DIR",
    "ArtifactSpec",
    "ClassifierLoadError",
    "Decision",
    "DistressClassifier",
    "LocalDistressClassifier",
    "Route",
    "artifact_sha256",
    "load_classifier",
    "read_artifact",
    "route_for",
]

ARTIFACT_DIR = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
_FILES = ("model.npz", "model.json")

Route = Literal["skip-llm", "escalate", "support"]


class ClassifierLoadError(RuntimeError):
    """The artifact may not be served. The caller escalates every note instead."""


@dataclass(frozen=True)
class Decision:
    route: Route
    #: None when the note was escalated without a usable score.
    score: float | None


class DistressClassifier(Protocol):
    """The graph's only view of the classifier."""

    artifact_sha256: str

    def decide(self, free_text: str) -> Decision: ...


def artifact_sha256(directory: Path = ARTIFACT_DIR) -> str:
    """The artifact's hash, computed exactly as the upstream test ledger records it."""
    digest = hashlib.sha256()
    for name in _FILES:
        path = directory / name
        digest.update(name.encode())
        digest.update(path.read_bytes() if path.exists() else b"<missing>")
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class ArtifactSpec:
    coef: np.ndarray
    intercept: float
    low: float
    high: float
    embed_dim: int
    embed_revision: str
    sha256: str


def read_artifact(
    directory: Path = ARTIFACT_DIR, *, expected_sha256: str | None = CLASSIFIER_ARTIFACT_SHA256
) -> ArtifactSpec:
    """Validate the committed artifact without loading any model weights."""
    if not all((directory / name).exists() for name in _FILES):
        raise ClassifierLoadError(f"missing model.npz or model.json in {directory}")

    sha256 = artifact_sha256(directory)
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ClassifierLoadError(
            f"artifact sha256 {sha256[:12]}… is not the evaluated artifact "
            f"({expected_sha256[:12]}…). Re-sync it from bloom-distress-classifier, or — for a "
            "new artifact — evaluate it there first and update CLASSIFIER_ARTIFACT_SHA256."
        )

    raw: Any = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ClassifierLoadError("model.json is not a JSON object")
    document = cast("dict[str, Any]", raw)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ClassifierLoadError(f"unsupported schema_version {document.get('schema_version')!r}")

    embedder = cast("dict[str, Any]", document.get("embedder") or {})
    revision = str(embedder.get("revision"))
    accepted = {EMBED_MODEL_REVISION, *CLASSIFIER_EQUIVALENT_EMBEDDER_REVISIONS}
    if embedder.get("model") != EMBED_MODEL or revision not in accepted:
        raise ClassifierLoadError(
            f"artifact was built on {embedder.get('model')}@{revision}, which is not a "
            "verified-equivalent snapshot of this service's embedder. Scores over different "
            "vectors look normal and mean nothing."
        )

    features = cast("dict[str, Any]", document.get("features") or {})
    if features.get("include_feeling"):
        raise ClassifierLoadError(
            "artifact reads the feeling chip; this service does not supply it"
        )

    bands = document.get("thresholds")
    if not isinstance(bands, dict):
        raise ClassifierLoadError("artifact has no fitted thresholds")
    thresholds = cast("dict[str, Any]", bands)
    low, high = thresholds.get("low"), thresholds.get("high")
    if not (
        isinstance(low, (int, float))
        and isinstance(high, (int, float))
        and math.isfinite(low)
        and math.isfinite(high)
        and 0.0 <= low <= high <= 1.0
    ):
        raise ClassifierLoadError(f"invalid thresholds low={low!r} high={high!r}")

    dim = int(embedder.get("dim") or 0)
    with np.load(directory / "model.npz", allow_pickle=False) as data:
        coef = np.asarray(data["coef"], dtype=np.float64)
        intercept = float(np.asarray(data["intercept"], dtype=np.float64).ravel()[0])
    if coef.ndim != 1 or coef.shape[0] != dim:
        raise ClassifierLoadError(f"coef shape {coef.shape} does not match embedder dim {dim}")
    if not (np.all(np.isfinite(coef)) and math.isfinite(intercept)):
        raise ClassifierLoadError("model weights are not finite")

    return ArtifactSpec(coef, intercept, float(low), float(high), dim, revision, sha256)


def route_for(score: object, low: float, high: float) -> Route:
    """Boundaries escalate, and so does anything that is not a finite probability."""
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return "escalate"
    value = float(score)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        return "escalate"
    if value < low:
        return "skip-llm"
    if value > high:
        return "support"
    return "escalate"


class LocalDistressClassifier:
    def __init__(self, spec: ArtifactSpec, embedder: Embedder) -> None:
        if embedder.dim and embedder.dim != spec.embed_dim:
            raise ClassifierLoadError(
                f"embedder produces {embedder.dim}-d vectors; the artifact expects {spec.embed_dim}"
            )
        self._spec = spec
        self._embedder = embedder
        self.artifact_sha256 = spec.sha256

    def decide(self, free_text: str) -> Decision:
        vector = np.asarray(self._embedder.embed([free_text]), dtype=np.float64)
        if vector.shape != (1, self._spec.embed_dim):
            return Decision("escalate", None)
        logit = float(vector[0] @ self._spec.coef + self._spec.intercept)
        score = float(np.exp(-np.logaddexp(0.0, -logit)))
        return Decision(route_for(score, self._spec.low, self._spec.high), score)


def load_classifier(embedder: Embedder, directory: Path = ARTIFACT_DIR) -> LocalDistressClassifier:
    return LocalDistressClassifier(read_artifact(directory), embedder)
