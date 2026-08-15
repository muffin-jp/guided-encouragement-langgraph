"""Request schema validation: the wire contract must reject bad input."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Feeling
from app.schemas import EncourageRequest


def _valid() -> dict[str, object]:
    return {"stageId": "stage-1", "feeling": "proud", "locale": "en"}


def test_accepts_valid_minimal_request() -> None:
    req = EncourageRequest.model_validate(_valid())
    assert req.stage_id == "stage-1"
    assert req.feeling is Feeling.PROUD
    assert req.free_text is None


def test_accepts_custom_feeling_with_free_text() -> None:
    req = EncourageRequest.model_validate(
        {"stageId": "s1", "feeling": "custom", "freeText": "happy but tired", "locale": "en"}
    )
    assert req.feeling is Feeling.CUSTOM
    assert req.free_text == "happy but tired"


@pytest.mark.parametrize("bad_stage", ["bad id!", "has space", "", "x" * 65, "emoji🌸"])
def test_rejects_bad_stage_id(bad_stage: str) -> None:
    with pytest.raises(ValidationError):
        EncourageRequest.model_validate({**_valid(), "stageId": bad_stage})


def test_rejects_over_length_free_text() -> None:
    with pytest.raises(ValidationError):
        EncourageRequest.model_validate({**_valid(), "freeText": "x" * 201})


def test_accepts_free_text_at_limit() -> None:
    req = EncourageRequest.model_validate({**_valid(), "freeText": "x" * 200})
    assert req.free_text is not None and len(req.free_text) == 200


def test_rejects_unknown_feeling() -> None:
    with pytest.raises(ValidationError):
        EncourageRequest.model_validate({**_valid(), "feeling": "ecstatic"})


@pytest.mark.parametrize("bad_locale", ["fr", "ja", "EN", ""])
def test_rejects_wrong_locale(bad_locale: str) -> None:
    with pytest.raises(ValidationError):
        EncourageRequest.model_validate({**_valid(), "locale": bad_locale})


def test_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        EncourageRequest.model_validate({**_valid(), "isAdmin": True})
