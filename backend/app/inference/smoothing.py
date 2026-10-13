"""Prediction smoothing: confidence thresholding, majority voting, debounce
and repeated-prediction suppression.

Raw per-window model predictions fluctuate frame to frame. This module
turns that noisy stream into a stable "currently displayed sign" plus a
much less frequent stream of "committed" signs — the ones that should
actually be appended to the sentence being built. Without this, "HELLO"
held for a second would flood the sentence with repeated "HELLO HELLO
HELLO HELLO" instead of a single "HELLO".
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

from app.config import settings


@dataclass
class SmoothedPrediction:
    display_label: str          # what to show as "current sign" right now
    display_confidence: float   # confidence backing that display label
    is_unknown: bool
    committed_label: Optional[str]  # non-None only on the frame a new sign is confirmed


class PredictionSmoother:
    def __init__(
        self,
        window: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        unknown_label: Optional[str] = None,
        min_frames_between_commits: Optional[int] = None,
        debounce_repeats: Optional[int] = None,
    ):
        self.window = window if window is not None else settings.smoothing_window
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else settings.confidence_threshold
        )
        self.unknown_label = unknown_label if unknown_label is not None else settings.unknown_label
        self.min_frames_between_commits = (
            min_frames_between_commits
            if min_frames_between_commits is not None
            else settings.min_frames_between_commits
        )
        # Votes are taken from the smoothing window, so it caps how many can agree.
        self.debounce_repeats = min(
            debounce_repeats if debounce_repeats is not None else settings.debounce_repeats, self.window
        )

        self._history: Deque[Tuple[str, float]] = deque(maxlen=self.window)
        self._last_committed_label: Optional[str] = None
        self._frames_since_commit = self.min_frames_between_commits  # allow immediate first commit

    def _majority_vote(self) -> Tuple[str, float]:
        labels = [label for label, _ in self._history]
        counts = Counter(labels)
        top_label, _ = counts.most_common(1)[0]
        confidences = [c for label, c in self._history if label == top_label]
        mean_conf = sum(confidences) / len(confidences)
        return top_label, mean_conf

    def _is_stable(self, label: str) -> bool:
        """True when the last ``debounce_repeats`` raw predictions all agree
        on ``label`` with enough confidence. A live window that only partly
        covers a sign can produce one or two confident votes for the wrong
        sign; requiring agreement keeps those from being committed."""
        if self.debounce_repeats <= 1:
            return True
        recent = list(self._history)[-self.debounce_repeats:]
        return len(recent) == self.debounce_repeats and all(
            lbl == label and conf >= self.confidence_threshold for lbl, conf in recent
        )

    def update(self, label: str, confidence: float) -> SmoothedPrediction:
        """Feed one raw (label, confidence) prediction from the model and
        get back the smoothed, debounced result to display/commit."""
        self._history.append((label, confidence))
        majority_label, mean_conf = self._majority_vote()

        is_unknown = mean_conf < self.confidence_threshold
        display_label = self.unknown_label if is_unknown else majority_label

        committed_label = None
        self._frames_since_commit += 1
        if not is_unknown and self._is_stable(majority_label):
            changed = majority_label != self._last_committed_label
            cooldown_elapsed = self._frames_since_commit >= self.min_frames_between_commits
            if changed or cooldown_elapsed:
                if changed or self._last_committed_label is None:
                    committed_label = majority_label
                    self._last_committed_label = majority_label
                    self._frames_since_commit = 0
                # Note: a *repeated* sign (same label, cooldown elapsed) is
                # intentionally NOT auto-committed again here — continuous
                # segmentation (Stage 7) decides whether a deliberate repeat
                # occurred. This prevents holding a sign from spamming the
                # sentence with duplicates, per the debounce requirement.

        return SmoothedPrediction(
            display_label=display_label,
            display_confidence=round(mean_conf, 4),
            is_unknown=is_unknown,
            committed_label=committed_label,
        )

    def clear_pending(self) -> None:
        """Drop the voting history (any pending candidate) but keep the
        last-committed label, so repeat suppression still applies. Used when
        hands leave the view: votes cast before that must not be able to
        carry a candidate over into a commit or a displayed sign."""
        self._history.clear()

    def reset(self) -> None:
        self._history.clear()
        self._last_committed_label = None
        self._frames_since_commit = self.min_frames_between_commits
