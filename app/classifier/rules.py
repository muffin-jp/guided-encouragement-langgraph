"""How a note is cut up, and which notes the model declines to judge.

This is the second implementation of a rule that was fitted somewhere else, which
is the arrangement most likely to go quietly wrong. A regex tweak here, or there,
changes production routing with no error anywhere: both copies keep returning
perfectly ordinary-looking scores, and only the decisions differ.

Three things guard against that, in order of how much they are worth:

1. **The artifact carries the parameters.** ``app/classifier/local.py`` refuses to
   load an artifact whose ``segmentation`` or ``scope`` block is not the one this
   file implements, so drift is a startup failure rather than a silent change.
2. **The routes are frozen as data.** ``routes.json`` holds 91 cases scored
   upstream — the adversarial probe plus edge cases for markup, window
   boundaries, accents, emoji and non-Latin scripts — and
   ``tests/test_classifier_parity.py`` re-derives every one of them here.
3. **The code is kept boring**, so a reader can compare it to the other copy.

Why a note is cut up at all
---------------------------
``all-MiniLM-L6-v2`` mean-pools a note, so a crisis clause inside a longer note
about a puzzle is averaged away: *"cleared the level finally, tries and tries.
honestly i've been thinking about ending it"* scores 0.008 as one string and 0.479
as its final clause. Skipping the LLM is the only route that can add a missed
crisis, so it is decided by the *highest-scoring* segment, never by the note.

Why some notes are not judged at all
------------------------------------
The embedder is an English model. On Japanese or Korean text it still returns a
confident-looking number, and that number means nothing — *"もう生きていたくない"*
scores low because the model cannot read it, not because the note is calm. A low
score from a model that cannot read the input is the one thing the skip band must
never act on, so those notes go to the LLM whatever they score.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import cached_property

__all__ = ["BOUNDARY", "STRIDE", "WINDOW", "ScopeRule", "Segmentation", "has_non_latin_letters"]

#: The fitted parameters. These are not tunable here: they are checked against the
#: artifact at load time, and changing one without retraining upstream makes every
#: artifact refuse to load.
WINDOW = 8
STRIDE = 4
#: Sentence terminators, newlines, and the end of a JSON object, bracket or tag.
#: The last alternative is zero-width on purpose: markup packed tight against its
#: text ("</user><note>i want to end it") has no whitespace to split on.
BOUNDARY = r"(?<=[.!?;])\s+|\n+|(?<=[}\]>])\s*"


@dataclass(frozen=True)
class Segmentation:
    window: int = WINDOW
    stride: int = STRIDE
    boundary: str = BOUNDARY

    def split(self, text: str) -> list[str]:
        """The note, its sentences, and sliding word windows over it.

        The whole note is always first, so segment scoring sees everything
        whole-note scoring sees. Order is stable and duplicates are dropped.
        """
        cleaned = text.strip()
        if not cleaned:
            return []

        segments = [cleaned]
        segments.extend(part.strip() for part in self.pattern.split(cleaned) if part.strip())

        words = cleaned.split()
        if len(words) > self.window:
            for start in range(0, len(words) - self.window + 1, self.stride):
                segments.append(" ".join(words[start : start + self.window]))
            segments.append(" ".join(words[-self.window :]))

        seen: dict[str, None] = {}
        for segment in segments:
            seen.setdefault(segment, None)
        return list(seen)

    @cached_property
    def pattern(self) -> re.Pattern[str]:
        return re.compile(self.boundary)


def has_non_latin_letters(text: str) -> bool:
    """True if any *letter* belongs to a script other than Latin.

    Only letters are examined, so emoji, digits, curly quotes and em dashes never
    trigger it — an ordinary English note typed on a phone must not be treated as
    foreign. ``café`` and ``naïve`` are Latin and stay in scope.
    """
    return any(
        character.isalpha() and not unicodedata.name(character, "").startswith("LATIN")
        for character in text
    )


@dataclass(frozen=True)
class ScopeRule:
    escalate_non_latin_letters: bool = True

    def out_of_scope(self, text: str) -> str:
        """Why this note may not be judged locally, or ``""`` if it may."""
        if self.escalate_non_latin_letters and has_non_latin_letters(text):
            return "non-Latin script: the embedder is English-only"
        return ""
