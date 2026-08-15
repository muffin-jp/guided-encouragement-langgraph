"""The fixed support-path response.

Ported verbatim from the TypeScript service (lib/supportMessage.ts). Shown when
the distress check flags the input. Deliberately pre-written and static — never
model-generated — so a player in a difficult moment always receives the same
carefully reviewed words: no clinical language, no diagnosis, no unpredictable
phrasing.
"""

from __future__ import annotations

SUPPORT_MESSAGE = (
    "Thank you for telling me how you're feeling — that took courage. What "
    "you're carrying sounds bigger than this game, and Bloom is just a game; "
    "it can't give you the support you deserve right now. Please reach out to "
    "someone you trust, or a professional support service in your area. You "
    "matter, and you don't have to hold this alone."
)
