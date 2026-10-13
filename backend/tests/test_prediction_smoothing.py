"""Tests for app.inference.smoothing.PredictionSmoother — confidence
thresholding, majority voting, debounce, repeated-prediction suppression."""

from __future__ import annotations

from app.inference.smoothing import PredictionSmoother


def test_low_confidence_becomes_unknown():
    smoother = PredictionSmoother(window=3, confidence_threshold=0.6, unknown_label="UNKNOWN")
    result = smoother.update("HELLO", 0.2)
    assert result.is_unknown
    assert result.display_label == "UNKNOWN"


def test_high_confidence_majority_is_displayed():
    smoother = PredictionSmoother(window=3, confidence_threshold=0.5, min_frames_between_commits=5)
    smoother.update("HELLO", 0.9)
    smoother.update("HELLO", 0.9)
    result = smoother.update("HELLO", 0.9)
    assert not result.is_unknown
    assert result.display_label == "HELLO"


def test_repeated_stable_sign_commits_once_not_every_frame():
    smoother = PredictionSmoother(window=1, confidence_threshold=0.5, min_frames_between_commits=100)
    commits = []
    for _ in range(10):
        result = smoother.update("HELLO", 0.95)
        commits.append(result.committed_label)
    # Only the first stable detection should commit; holding the same sign
    # must not spam the sentence with repeats.
    assert commits[0] == "HELLO"
    assert all(c is None for c in commits[1:])


def test_new_sign_commits_immediately_after_a_different_sign():
    smoother = PredictionSmoother(window=1, confidence_threshold=0.5, min_frames_between_commits=100)
    first = smoother.update("HELLO", 0.9)
    second = smoother.update("THANK_YOU", 0.9)
    assert first.committed_label == "HELLO"
    assert second.committed_label == "THANK_YOU"


def test_majority_vote_ignores_single_flicker():
    smoother = PredictionSmoother(window=5, confidence_threshold=0.5, min_frames_between_commits=100)
    for _ in range(4):
        smoother.update("HELLO", 0.9)
    # one noisy flicker frame shouldn't flip the majority away from HELLO
    result = smoother.update("BAD", 0.6)
    assert result.display_label == "HELLO"


def test_reset_clears_history_and_commit_state():
    smoother = PredictionSmoother(window=3, confidence_threshold=0.5, min_frames_between_commits=100, debounce_repeats=1)
    smoother.update("HELLO", 0.9)
    smoother.reset()
    result = smoother.update("HELLO", 0.9)
    assert result.committed_label == "HELLO"  # commits again as if it were the first time


def test_clear_pending_drops_votes_so_unknown_is_shown_immediately():
    """Regression: an UNKNOWN vote after hands left was outvoted by the
    previous sign, which stayed on screen at its old confidence."""
    smoother = PredictionSmoother(window=5, confidence_threshold=0.6, min_frames_between_commits=100)
    for _ in range(4):
        smoother.update("BLACK", 0.69)
    smoother.clear_pending()
    result = smoother.update("UNKNOWN", 0.0)
    assert result.is_unknown and result.display_label == "UNKNOWN"
    assert result.display_confidence == 0.0
    assert result.committed_label is None


def test_clear_pending_keeps_repeat_suppression():
    smoother = PredictionSmoother(window=1, confidence_threshold=0.5, min_frames_between_commits=100)
    assert smoother.update("HELLO", 0.9).committed_label == "HELLO"
    smoother.clear_pending()
    assert smoother.update("HELLO", 0.9).committed_label is None


def test_debounce_requires_agreeing_confident_votes_before_commit():
    """A live window that only partly covers a sign can give one or two
    confident votes for the wrong sign; those must not be committed."""
    smoother = PredictionSmoother(window=5, confidence_threshold=0.6, min_frames_between_commits=100, debounce_repeats=3)
    assert smoother.update("HELLO", 0.9).committed_label is None
    assert smoother.update("HELLO", 0.9).committed_label is None
    assert smoother.update("HELLO", 0.9).committed_label == "HELLO"


def test_debounce_streak_is_broken_by_a_different_or_unconfident_vote():
    smoother = PredictionSmoother(window=5, confidence_threshold=0.6, min_frames_between_commits=100, debounce_repeats=3)
    smoother.update("HELLO", 0.9)
    smoother.update("HELLO", 0.9)
    assert smoother.update("HELLO", 0.5).committed_label is None  # low confidence
    smoother.update("HELLO", 0.9)
    assert smoother.update("BAD", 0.9).committed_label is None  # flicker
    smoother.update("HELLO", 0.9)
    smoother.update("HELLO", 0.9)
    assert smoother.update("HELLO", 0.9).committed_label == "HELLO"


def test_debounce_is_capped_by_smoothing_window():
    smoother = PredictionSmoother(window=2, confidence_threshold=0.5, debounce_repeats=4)
    assert smoother.debounce_repeats == 2
    smoother.update("HELLO", 0.9)
    assert smoother.update("HELLO", 0.9).committed_label == "HELLO"
