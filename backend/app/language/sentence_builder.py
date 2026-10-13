"""Accumulates committed sign predictions into a gloss sequence and renders
it as natural-language text.

This sits downstream of ``PredictionSmoother`` — it only ever receives
already-debounced "committed" labels, but still guards against duplicates
itself so it is safe to use independently (e.g. from tests or a future
continuous-recognition segmenter that commits glosses on its own schedule).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from app.config import settings
from app.language.gloss_to_text import gloss_sequence_to_text


@dataclass
class SentenceBuilder:
    unknown_label: str = field(default_factory=lambda: settings.unknown_label)
    max_gloss_length: int = 20

    _gloss_sequence: List[str] = field(default_factory=list)

    def add_gloss(self, token: str) -> bool:
        """Append a committed gloss token. Returns True if it was appended
        (False if suppressed as UNKNOWN or an immediate duplicate)."""
        if not token:
            return False
        token = token.upper()
        if token == self.unknown_label.upper():
            return False
        if self._gloss_sequence and self._gloss_sequence[-1] == token:
            return False  # repeated-prediction suppression
        self._gloss_sequence.append(token)
        if len(self._gloss_sequence) > self.max_gloss_length:
            self._gloss_sequence.pop(0)
        return True

    @property
    def gloss_sequence(self) -> List[str]:
        return list(self._gloss_sequence)

    def get_text(self) -> str:
        return gloss_sequence_to_text(self._gloss_sequence, unknown_label=self.unknown_label)

    def clear(self) -> None:
        self._gloss_sequence.clear()
