"""Edge cases at the pipeline's boundaries: degenerate frames through real
MediaPipe, partially missing landmark groups, and sentence/smoothing limits."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.inference.smoothing import PredictionSmoother
from app.inference.temporal_buffer import TemporalBuffer
from app.language.gloss_to_text import gloss_sequence_to_text
from app.language.sentence_builder import SentenceBuilder
from app.vision.feature_extractor import FeatureExtractor, landmarks_to_vector
from app.vision.landmarks import FrameLandmarks
from app.vision.preprocessing import normalize_frame

_N_FACE = len(settings.face_landmark_indices)


@pytest.fixture(scope="module")
def extractor():
    fe = FeatureExtractor()
    if not fe.available:
        pytest.skip("MediaPipe model not downloaded")
    yield fe
    fe.close()


@pytest.mark.parametrize("frame", [
    np.zeros((1, 1, 3), dtype=np.uint8),             # tiny
    np.zeros((480, 640), dtype=np.uint8),            # grayscale (2-D)
    np.zeros((480, 640, 4), dtype=np.uint8),         # BGRA
    np.full((720, 1280, 3), 255, dtype=np.uint8),    # large, saturated
    None,
], ids=["1x1", "gray", "bgra", "720p-white", "none"])
def test_degenerate_frames_yield_an_empty_well_formed_vector(extractor, frame):
    vector, raw = extractor.extract(frame)
    assert vector.shape == (settings.feature_dim,) and vector.dtype == np.float32
    assert np.isfinite(vector).all()
    assert not raw.has_any_hand


def _points(n, seed):
    return np.random.default_rng(seed).random((n, 3)).astype(np.float32)


@pytest.mark.parametrize("missing", ["left_hand", "right_hand", "pose", "face", "all"])
def test_missing_groups_are_zero_filled_in_their_own_slots(missing):
    groups = {"left_hand": _points(21, 1), "right_hand": _points(21, 2), "pose": _points(33, 3),
              "face": _points(_N_FACE, 4)}
    if missing == "all":
        groups = dict.fromkeys(groups)
    else:
        groups[missing] = None
    vec = landmarks_to_vector(normalize_frame(FrameLandmarks(**groups)))
    assert vec.shape == (330,) and vec.dtype == np.float32 and np.isfinite(vec).all()

    slots = {"left_hand": slice(0, 63), "right_hand": slice(63, 126), "pose": slice(126, 225),
             "face": slice(225, 330)}
    for name, sl in slots.items():
        absent = missing in (name, "all")
        assert (np.all(vec[sl] == 0)) == absent, name


def test_hand_without_pose_is_still_normalized():
    hand = _points(21, 5) * 0.2 + 0.4
    vec = landmarks_to_vector(normalize_frame(FrameLandmarks(None, hand, None, None)))
    right = vec[63:126].reshape(21, 3)
    assert np.allclose(right.mean(axis=0), 0, atol=1e-6)  # centered on itself
    assert np.isclose(np.abs(right).max(), 1.0)


def test_buffer_rejects_malformed_vectors():
    buf = TemporalBuffer(window_size=4, feature_dim=330)
    with pytest.raises(ValueError):
        buf.push(np.zeros(329, dtype=np.float32))
    with pytest.raises((ValueError, IndexError)):
        buf.push(np.zeros(0, dtype=np.float32))
    seq, mask = buf.get_sequence()  # nothing was buffered
    assert seq.shape == (4, 330) and not mask.any()


# --- sentence construction ---------------------------------------------------

def test_sign_can_repeat_after_a_different_sign():
    sb = SentenceBuilder()
    for token in ["HELLO", "HELLO", "YOU", "HELLO"]:
        sb.add_gloss(token)
    assert sb.gloss_sequence == ["HELLO", "YOU", "HELLO"]


def test_sentence_keeps_only_the_most_recent_glosses():
    sb = SentenceBuilder(max_gloss_length=3)
    for token in ["RED", "BLUE", "GREEN", "PINK"]:
        sb.add_gloss(token)
    assert sb.gloss_sequence == ["BLUE", "GREEN", "PINK"]


@pytest.mark.parametrize("tokens, text", [
    (["hello"], "Hello."),                        # case-insensitive
    (["GOOD_MORNING"], "Good morning."),          # underscores become spaces
    (["UNKNOWN", "UNKNOWN"], ""),
    (["HOW", "YOU"], "How are you?"),
    (["I", "WATER", "WANT"], "I want water."),
])
def test_gloss_to_text_punctuation(tokens, text):
    assert gloss_sequence_to_text(tokens) == text


def test_empty_and_unknown_tokens_are_never_added():
    sb = SentenceBuilder()
    assert sb.add_gloss("") is False and sb.add_gloss("unknown") is False
    assert sb.get_text() == ""


# --- smoothing thresholds ----------------------------------------------------

def test_confidence_exactly_at_threshold_counts_as_confident():
    s = PredictionSmoother(window=3, confidence_threshold=0.5, debounce_repeats=3)
    results = [s.update("RED", 0.5) for _ in range(3)]
    assert not results[-1].is_unknown
    assert [r.committed_label for r in results] == [None, None, "RED"]


def test_just_below_threshold_never_commits():
    s = PredictionSmoother(window=3, confidence_threshold=0.5, debounce_repeats=3)
    results = [s.update("RED", 0.4999) for _ in range(10)]
    assert all(r.is_unknown and r.committed_label is None for r in results)
