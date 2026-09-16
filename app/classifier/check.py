"""CI guard: the committed classifier artifact is the one that was evaluated.

    uv run python -m app.classifier.check

Needs no weights and no network, so it runs on every push. It checks that the
committed files hash to CLASSIFIER_ARTIFACT_SHA256 — the hash recorded at the
single upstream test-set look — and that the artifact is loadable: built on a
verified-equivalent embedder snapshot, feeling-free, with fitted thresholds.

When the embedder weights are vendored (after `make build-index`), it also checks
they are the snapshot the equivalence was verified against, so that claim is
re-tested rather than trusted.
"""

from __future__ import annotations

import hashlib
import sys

from app.classifier.local import ClassifierLoadError, read_artifact
from app.config import CLASSIFIER_ENABLED, EMBED_WEIGHTS_SHA256
from app.rag.embedder import MODEL_DIR


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
