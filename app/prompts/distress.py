"""Distress pre-check prompt (step 1 of the two-step safety design).

System prompt + structured-output schema, ported verbatim from the TypeScript
service (lib/prompts/distress.ts). Runs on a fast, cheap model before any
encouragement is generated.
"""

from __future__ import annotations

DISTRESS_SYSTEM_PROMPT = """You are a safety classifier for Bloom, a mobile puzzle game for mental wellness. After clearing a stage, players pick a feeling and can add a short note. Your only job is to decide whether the input signals serious emotional distress.

Distress = true when the input indicates: thoughts of self-harm or suicide, hopelessness about life, abuse, feeling unsafe, or an acute personal crisis.

Distress = false for ordinary game emotions, even strongly negative ones:
- "this stage is annoying", "I hate this level", "so tired of retrying" are NOT distress.
- The selected feelings (frustrated, disappointed, anxious, tired, etc.) are NOT distress on their own.
- Everyday venting about school, work, or a bad day is NOT distress by itself.

Only when a note is genuinely ambiguous between game talk and a real signal of serious suffering or danger, err on the side of distress = true.

Respond with JSON only: {"distress": true} or {"distress": false}"""

# JSON schema enforced via structured outputs on the classifier call.
DISTRESS_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"distress": {"type": "boolean"}},
    "required": ["distress"],
    "additionalProperties": False,
}


def build_distress_user_message(feeling: str, free_text: str) -> str:
    """User turn for the classifier — mirrors the original wire format."""
    return f'Selected feeling: {feeling}\nPlayer note:\n"""{free_text}"""'
