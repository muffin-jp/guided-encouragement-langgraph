"""Pydantic request/response models.

The request contract is the same one the Bloom Unity client calls in
production — it must match the original TypeScript wire format exactly, so it is
validated strictly here (bad stageId, over-length freeText, wrong locale all
become 400s).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import MAX_FREE_TEXT_LENGTH, Feeling


class EncourageRequest(BaseModel):
    """Body for POST /api/encourage."""

    model_config = ConfigDict(extra="forbid")

    stage_id: str = Field(
        alias="stageId",
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9-_]+$",
    )
    feeling: Feeling
    free_text: str | None = Field(
        default=None, alias="freeText", max_length=MAX_FREE_TEXT_LENGTH
    )
    # English only for now; the field exists so the Unity client contract does
    # not need to change when Japanese ships.
    locale: Literal["en"]


class ResumeRequest(BaseModel):
    """Body for POST /api/encourage/{thread_id}/resume — the moderator decision
    that releases a graph paused at the moderation interrupt."""

    model_config = ConfigDict(extra="forbid")

    approve: bool
    note: str | None = Field(default=None, max_length=500)
