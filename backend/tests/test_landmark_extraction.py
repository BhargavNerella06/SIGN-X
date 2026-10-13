"""Tests for app.vision.landmarks — no physical webcam needed; MediaPipe
runs directly on synthetic in-memory frames."""

from __future__ import annotations

from app.vision.landmarks import FrameLandmarks, LandmarkExtractor


def test_extractor_loads_when_model_present():
    extractor = LandmarkExtractor()
    try:
        assert extractor.available, (
            "MediaPipe model not found — run scripts/download_models.ps1 "
            "(or .sh) first."
        )
    finally:
        extractor.close()


def test_process_returns_frame_landmarks_on_random_frame(random_frame):
    extractor = LandmarkExtractor()
    try:
        result = extractor.process(random_frame)
        assert isinstance(result, FrameLandmarks)
        # Random noise should not contain a detectable hand/pose/face, but
        # the call must still return a well-formed (all-None) result
        # rather than raising.
        assert result.left_hand is None or result.left_hand.shape == (21, 3)
        assert result.right_hand is None or result.right_hand.shape == (21, 3)
    finally:
        extractor.close()


def test_process_handles_none_gracefully_when_unavailable(monkeypatch, random_frame):
    extractor = LandmarkExtractor()
    extractor._available = False  # simulate "model missing" (Section 25)
    result = extractor.process(random_frame)
    assert result.left_hand is None
    assert result.right_hand is None
    assert result.pose is None
    assert result.face is None
