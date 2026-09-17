"""CI guard: the committed classifier artifact is the one that was evaluated.

    uv run python -m app.classifier.check

Needs no weights and no network, so it runs on every push. It checks that the
committed files hash to CLASSIFIER_ARTIFACT_SHA256 — the hash recorded at the
upstream test-set look — and that the artifact is loadable: built on a
verified-equivalent embedder snapshot, feeling-free, with fitted thresholds, and
splitting notes exactly the way this service does.

It also checks that ``routes.json`` is present and belongs to this artifact. That
file is what ``tests/test_classifier_parity.py`` compares against; a stale one
would let every parity test pass while proving nothing.

When the embedder weights are vendored (after `make build-index`), it also checks
they are the snapshot the equivalence was verified against, so that claim is
re-tested rather than trusted.
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

from app.classifier.local import ARTIFACT_DIR, ArtifactSpec, ClassifierLoadError, read_artifact
from app.config import CLASSIFIER_ENABLED, EMBED_WEIGHTS_SHA256
from app.rag.embedder import MODEL_DIR


def _routes_problems(spec: ArtifactSpec) -> list[str]:
    """The frozen-route fixture must exist and describe this artifact."""
    path = ARTIFACT_DIR / "routes.json"
    if not path.exists():
        return [
            "app/classifier/routes.json is missing, so the parity tests cannot check this "
            "service's splitting rule against the one the model was fitted with. Run "
            "`make sync-classifier`."
        ]
    try:
        document: Any = json.loads(path.read_text(encoding="utf-8"))
        bands = document["thresholds"]
        low, high = float(bands["low"]), float(bands["high"])
        cases = len(document["cases"])
    except (ValueError, KeyError, TypeError) as exc:
        return [f"routes.json is malformed: {exc}"]
    if (low, high) != (spec.low, spec.high):
        return [
            f"routes.json was frozen at low {low:.4f} / high {high:.3f}, but the artifact "
            f"serves low {spec.low:.4f} / high {spec.high:.3f}. Re-run `make export-routes` "
            "upstream and `make sync-classifier` here."
        ]
    print(f"routes.json freezes {cases} routes at these thresholds.")
    return []


def problems() -> list[str]:
    found: list[str] = []
    try:
        spec = read_artifact()
    except ClassifierLoadError as exc:
        found.append(str(exc))
    else:
        print(
            f"artifact {spec.sha256[:12]}… matches the evaluated artifact; "
            f"low {spec.low:.4f}, high {spec.high:.3f}, embedder @{spec.embed_revision[:8]}."
        )

        found.extend(_routes_problems(spec))

    weights = MODEL_DIR / "model.safetensors"
    if weights.exists():
        actual = hashlib.sha256(weights.read_bytes()).hexdigest()
        if actual != EMBED_WEIGHTS_SHA256:
            found.append(
                f"vendored embedder weights hash {actual[:12]}…, not the verified snapshot "
                f"{EMBED_WEIGHTS_SHA256[:12]}…; the revision-equivalence claim no longer holds."
            )
        else:
            print("vendored embedder weights match the verified snapshot.")
    else:
        print("embedder weights not vendored here; skipped the weights check.")

    print(f"CLASSIFIER_ENABLED={CLASSIFIER_ENABLED}")
    return found


def main() -> None:
    found = problems()
    for problem in found:
        print(f"ERROR: {problem}", file=sys.stderr)
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
