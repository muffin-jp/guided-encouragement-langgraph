"""This service must route notes exactly as the repository that fitted the model.

The splitting and scope rules exist twice — once in ``bloom-distress-classifier``,
where they were fitted and red-teamed, and once in ``app/classifier/rules.py``,
which serves them. Nothing about that arrangement is self-correcting: a regex
tweak on either side changes production routing with no error, no exception and
no obviously wrong score. The only symptom is different decisions.

``app/classifier/routes.json`` is the guard. Upstream scores 91 cases with the
shipped artifact and freezes each one's whole-note score, worst-segment score and
route: the adversarial probe, which is written to exploit exactly this kind of
gap, plus edge cases for packed markup, window boundaries, accented Latin, emoji
and non-Latin scripts. Here every route is re-derived with this service's own
code and compared.

Two of these tests need the vendored embedder weights and are skipped without
them; the rest — the segmentation itself, which is where drift actually happens —
run everywhere, because they compare *segments*, not scores.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.classifier.local import ARTIFACT_DIR, SCOPE, SEGMENTATION, read_artifact, route_for
from app.rag.embedder import MODEL_DIR

ROUTES_PATH = ARTIFACT_DIR / "routes.json"
pytestmark = pytest.mark.skipif(not ROUTES_PATH.exists(), reason="routes.json not synced")


def frozen() -> dict[str, Any]:
    return json.loads(ROUTES_PATH.read_text(encoding="utf-8"))


def cases() -> list[dict[str, Any]]:
    return frozen()["cases"]


def case_ids() -> list[str]:
    return [case["id"] for case in cases()]


# --- the parameters themselves --------------------------------------------------------


def test_the_frozen_parameters_are_the_ones_this_service_implements() -> None:
    document = frozen()
    assert document["segmentation"] == {
        "window": SEGMENTATION.window,
        "stride": SEGMENTATION.stride,
        "boundary": SEGMENTATION.boundary,
    }
    assert document["scope"] == {"escalate_non_latin_letters": SCOPE.escalate_non_latin_letters}


def test_the_frozen_routes_belong_to_the_committed_artifact() -> None:
    """A stale fixture would pass every comparison below and mean nothing."""
    spec = read_artifact()
    document = frozen()
    assert document["thresholds"]["low"] == pytest.approx(spec.low)
    assert document["thresholds"]["high"] == pytest.approx(spec.high)


# --- the split, compared segment by segment -------------------------------------------


@pytest.mark.parametrize("case", cases(), ids=case_ids())
def test_every_note_splits_into_the_same_segments(case: dict[str, Any]) -> None:
    """Needs no weights, and is where drift would actually show up first."""
    assert SEGMENTATION.split(case["text"]) == case["segments"]


@pytest.mark.parametrize("case", cases(), ids=case_ids())
def test_the_scope_rule_agrees_on_every_note(case: dict[str, Any]) -> None:
    assert bool(SCOPE.out_of_scope(case["text"])) == bool(case["out_of_scope"])


# --- the routes, which need the model --------------------------------------------------


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="embedder weights not vendored")
def test_every_frozen_route_is_reproduced() -> None:
    """The whole contract, end to end, on the committed artifact and weights."""
    from app.classifier.local import load_classifier
    from app.rag.embedder import SentenceTransformerEmbedder

    classifier = load_classifier(SentenceTransformerEmbedder())
    mismatches = [
        f"{case['id']}: expected {case['route']}, got {decision.route} "
        f"(whole {case['whole']} vs {decision.score}, worst {case['worst']} vs {decision.worst})"
        for case in cases()
        if (decision := classifier.decide(case["text"])).route != case["route"]
    ]
    assert not mismatches, "routes differ from the repository that fitted them:\n  " + "\n  ".join(
        mismatches
    )


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="embedder weights not vendored")
def test_scores_match_to_within_float32_noise() -> None:
    """Routes can agree while the scores drift; that would be luck, not agreement.

    The two repositories embed in different batch shapes, which moves the last bit
    of a float32 sum. Anything larger means a different computation, not noise.
    """
    from app.classifier.local import load_classifier
    from app.rag.embedder import SentenceTransformerEmbedder

    classifier = load_classifier(SentenceTransformerEmbedder())
    for case in cases():
        decision = classifier.decide(case["text"])
        if case["whole"] is None or case["out_of_scope"]:
            continue
        assert decision.score == pytest.approx(case["whole"], abs=1e-6), case["id"]
        assert decision.worst == pytest.approx(case["worst"], abs=1e-6), case["id"]


# --- the routes, re-derived without the model ------------------------------------------


@pytest.mark.parametrize("case", cases(), ids=case_ids())
def test_the_routing_rule_maps_the_frozen_scores_to_the_frozen_route(
    case: dict[str, Any],
) -> None:
    """Separates the two ways parity can break: the scores, or the rule over them.

    This one needs no weights. It takes the scores upstream computed and checks
    that this service's banding turns them into the same route — so a failure here
    is a rule difference, and a failure in the test above with this one passing is
    a scoring difference.
    """
    spec_low, spec_high = frozen()["thresholds"]["low"], frozen()["thresholds"]["high"]
    if case["out_of_scope"]:
        assert case["route"] == "escalate"
        return
    if case["whole"] is None:
        assert case["route"] == "escalate"
        return
    assert route_for(case["whole"], case["worst"], spec_low, spec_high) == case["route"]


def test_the_fixture_exercises_every_route_and_both_scope_outcomes() -> None:
    """A fixture that never skips would pass while the skip band was broken."""
    routes = {case["route"] for case in cases()}
    assert {"skip-llm", "escalate"} <= routes
    assert any(case["out_of_scope"] for case in cases())
    assert any(not case["out_of_scope"] for case in cases())
    assert any(case["whole"] is None for case in cases())  # unscorable notes
