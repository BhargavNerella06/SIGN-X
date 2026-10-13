"""Integration tests for the real-time path: RealtimePredictor (buffer ->
model -> smoothing -> sentence) and the /ws/predict speech controls.

MediaPipe is replaced with a fake extractor so each test controls exactly
which frames contain hands; no webcam or model checkpoint is needed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import app.inference.predictor as predictor_module
from app.api import websocket as websocket_module
from app.config import settings
from app.inference.predictor import RealtimePredictor
from app.models.classifier import ClassificationResult
from app.models.model_loader import get_model_bundle
from app.vision.landmarks import FrameLandmarks, empty_hand

_FRAME = np.zeros((48, 64, 3), dtype=np.uint8)
# Frames until a steady, confident sign is first committed in a fresh
# session: the window must fill, then debounce_repeats votes must agree.
_FIRST_COMMIT = settings.realtime_window_frames + (settings.debounce_repeats - 1) * settings.sequence_stride


class _FakeExtractor:
    """Stands in for FeatureExtractor; ``hands`` toggles hand detection."""

    def __init__(self, *args, **kwargs):
        self.hands = True
        self.value = 0.1  # feature value of hand frames, to tell signs apart
        self.fail = False

    available = True

    def extract(self, frame_bgr):
        if self.fail:
            raise RuntimeError("simulated extraction failure")
        hand = empty_hand() + 0.1 if self.hands else None
        raw = FrameLandmarks(left_hand=None, right_hand=hand, pose=None, face=None)
        vector = np.full(settings.feature_dim, self.value if self.hands else 0.0, dtype=np.float32)
        return vector, raw

    def reset(self):
        pass

    def close(self):
        pass


class _FakeClassifier:
    def __init__(self, label="HELLO", prob=0.9):
        self.label, self.prob = label, prob
        self.calls = []
        self.inputs = []

    def predict(self, x, mask=None, top_k=3):
        self.calls.append((tuple(x.shape), mask.clone() if mask is not None else None))
        self.inputs.append(x.clone())
        return ClassificationResult(top_indices=[0], top_labels=[self.label], top_probs=[self.prob])


class _FakeBundle:
    def __init__(self, classifier):
        self.classifier = classifier

    def info(self):
        return {"is_trained": True}


@pytest.fixture
def make_predictor(monkeypatch):
    monkeypatch.setattr(predictor_module, "FeatureExtractor", _FakeExtractor)

    def _make(label="HELLO", prob=0.9):
        clf = _FakeClassifier(label, prob)
        return RealtimePredictor(model_bundle=_FakeBundle(clf)), clf

    return _make


def _feed(predictor, n):
    return [predictor.process_frame(_FRAME).to_dict() for _ in range(n)]


def test_no_inference_until_window_is_full(make_predictor):
    predictor, clf = make_predictor()
    msgs = _feed(predictor, settings.realtime_window_frames - 1)
    assert clf.calls == []
    assert all(m["sign"] == settings.unknown_label for m in msgs)


def test_model_receives_training_shaped_sequence(make_predictor):
    predictor, clf = make_predictor()
    _feed(predictor, settings.realtime_window_frames)
    assert len(clf.calls) == 1
    shape, mask = clf.calls[0]
    assert shape == (1, settings.sequence_length, settings.feature_dim)
    assert mask.shape == (1, settings.sequence_length) and bool(mask.all())


def test_inference_runs_once_per_stride(make_predictor):
    predictor, clf = make_predictor()
    _feed(predictor, settings.realtime_window_frames + 3 * settings.sequence_stride)
    assert len(clf.calls) == 4


def test_commit_flag_is_sent_on_exactly_one_message(make_predictor):
    """Regression: the committed flag used to be repeated on every message
    until the next stride, so the UI history logged one sign many times."""
    predictor, _ = make_predictor()
    msgs = _feed(predictor, settings.realtime_window_frames + 3 * settings.sequence_stride)
    assert sum(m["committed"] for m in msgs) == 1
    assert msgs[-1]["sentence"] != ""


def test_held_sign_is_committed_to_sentence_once(make_predictor):
    predictor, _ = make_predictor("HELLO", 0.95)
    _feed(predictor, settings.realtime_window_frames + 10 * settings.sequence_stride)
    assert predictor.sentence_builder.gloss_sequence == ["HELLO"]


def test_low_confidence_is_unknown_and_not_committed(make_predictor):
    predictor, _ = make_predictor("HELLO", settings.confidence_threshold - 0.1)
    msgs = _feed(predictor, settings.realtime_window_frames + 2 * settings.sequence_stride)
    assert msgs[-1]["is_unknown"] and msgs[-1]["sign"] == settings.unknown_label
    assert not any(m["committed"] for m in msgs)
    assert predictor.sentence_builder.gloss_sequence == []


def test_window_without_hands_skips_model_and_commits_nothing(make_predictor):
    """Regression: an empty scene (all-zero landmarks) was classified by the
    trained model as a confident sign and written into the sentence."""
    predictor, clf = make_predictor("IT", 0.99)
    predictor.feature_extractor.hands = False
    msgs = _feed(predictor, settings.realtime_window_frames + 4 * settings.sequence_stride)
    assert clf.calls == []
    assert msgs[-1]["sign"] == settings.unknown_label and msgs[-1]["is_unknown"]
    assert msgs[-1]["hands_detected"] is False
    assert predictor.sentence_builder.gloss_sequence == []


def test_sparse_hand_detections_still_run_the_model(make_predictor):
    """Brief landmark dropouts must not suppress inference: half the window
    with hands is above the min_hand_frame_ratio gate."""
    predictor, clf = make_predictor()
    for i in range(settings.realtime_window_frames):
        predictor.feature_extractor.hands = i % 2 == 0
        predictor.process_frame(_FRAME)
    assert len(clf.calls) == 1


def test_sign_resumes_after_hands_return(make_predictor):
    predictor, clf = make_predictor()
    predictor.feature_extractor.hands = False
    _feed(predictor, settings.realtime_window_frames)
    predictor.feature_extractor.hands = True
    msgs = _feed(predictor, _FIRST_COMMIT)
    assert clf.calls
    assert predictor.sentence_builder.gloss_sequence == ["HELLO"]
    assert sum(m["committed"] for m in msgs) == 1


# --- Hands leaving the view / camera switch (stale prediction regressions) ---

_W = settings.realtime_window_frames
_S = settings.sequence_stride


def test_valid_sign_then_no_hands_reports_unknown_and_keeps_sentence(make_predictor):
    """Regression: after the hands left, the last sign (e.g. BLACK 69%) kept
    being displayed next to "no hands detected" for several seconds."""
    predictor, _ = make_predictor("BLACK", 0.69)
    msgs = _feed(predictor, _W + 3 * _S)
    assert msgs[-1]["sign"] == "BLACK"
    sentence, commit_id = msgs[-1]["sentence"], msgs[-1]["commit_id"]

    predictor.feature_extractor.hands = False
    msgs = _feed(predictor, _S)
    for m in msgs[-1:] + _feed(predictor, 2 * _W):
        assert m["sign"] == settings.unknown_label and m["is_unknown"]
        assert m["confidence"] == 0.0 and m["top_predictions"] == []
        assert m["sentence"] == sentence and m["commit_id"] == commit_id
        assert m["type"] == "prediction" and m["message"] is None
    assert predictor.sentence_builder.gloss_sequence == ["BLACK"]


def test_no_hands_never_commits_or_classifies_the_stale_window(make_predictor):
    predictor, clf = make_predictor("HELLO", 0.95)
    _feed(predictor, _W + 3 * _S)
    calls_before = len(clf.calls)

    predictor.feature_extractor.hands = False
    clf.label = "WATER"  # whatever the model would say about the old window
    msgs = _feed(predictor, 4 * _W)
    assert not any(m["committed"] for m in msgs)
    assert len(clf.calls) == calls_before
    assert predictor.sentence_builder.gloss_sequence == ["HELLO"]


def test_pending_candidate_cannot_commit_after_hands_leave(make_predictor):
    predictor, clf = make_predictor("HELLO", 0.95)
    _feed(predictor, _W + 3 * _S)
    # WATER gets two votes: not yet the majority, so still pending.
    clf.label = "WATER"
    msgs = _feed(predictor, 2 * _S)
    assert not any(m["committed"] for m in msgs)

    predictor.feature_extractor.hands = False
    msgs = _feed(predictor, 4 * _W)
    assert not any(m["committed"] for m in msgs)
    assert predictor.sentence_builder.gloss_sequence == ["HELLO"]

    # Fresh signing starts from an empty vote history.
    clf.label = "GO"
    predictor.feature_extractor.hands = True
    _feed(predictor, _W)
    assert predictor.sentence_builder.gloss_sequence == ["HELLO", "GO"]


def test_recognition_resumes_after_idle_period(make_predictor):
    predictor, clf = make_predictor("HELLO", 0.95)
    _feed(predictor, _W + 3 * _S)
    predictor.feature_extractor.hands = False
    _feed(predictor, 2 * _W)

    clf.label = "WATER"
    predictor.feature_extractor.hands = True
    msgs = _feed(predictor, _W)
    assert sum(m["committed"] for m in msgs) == 1
    assert msgs[-1]["sign"] == "WATER" and not msgs[-1]["is_unknown"]
    assert predictor.sentence_builder.gloss_sequence == ["HELLO", "WATER"]


def test_old_sign_frames_never_reach_the_model_after_idle(make_predictor):
    predictor, clf = make_predictor("HELLO", 0.95)
    predictor.feature_extractor.value = 0.5  # old sign
    _feed(predictor, _W + 3 * _S + 4)
    predictor.feature_extractor.hands = False
    _feed(predictor, _W)
    calls_before = len(clf.inputs)

    predictor.feature_extractor.value = 0.1  # new sign
    predictor.feature_extractor.hands = True
    _feed(predictor, _W + _S)
    new_inputs = clf.inputs[calls_before:]
    assert new_inputs
    assert all(not torch.isclose(x, torch.tensor(0.5)).any() for x in new_inputs)


def test_brief_dropout_does_not_block_resumption(make_predictor):
    """Hands missing for one stride (below the idle gate) only blank the
    display; recognition continues as soon as they are back."""
    predictor, clf = make_predictor("HELLO", 0.95)
    _feed(predictor, _W + 3 * _S)
    predictor.feature_extractor.hands = False
    assert _feed(predictor, _S)[-1]["sign"] == settings.unknown_label

    predictor.feature_extractor.hands = True
    calls_before = len(clf.calls)
    assert _feed(predictor, _S)[-1]["sign"] == "HELLO"
    assert len(clf.calls) == calls_before + 1
    assert predictor.sentence_builder.gloss_sequence == ["HELLO"]  # no duplicate


def test_commit_id_counts_commits_only(make_predictor):
    predictor, clf = make_predictor("HELLO", 0.95)
    msgs = _feed(predictor, _W + 3 * _S)
    assert [m["commit_id"] for m in msgs if m["committed"]] == [1]
    assert msgs[-1]["commit_id"] == 1
    clf.label = "WATER"
    msgs = _feed(predictor, settings.debounce_repeats * _S)
    assert [m["commit_id"] for m in msgs if m["committed"]] == [2]


def test_error_frame_after_commit_does_not_repeat_commit(make_predictor):
    predictor, _ = make_predictor("HELLO", 0.95)
    msgs = _feed(predictor, _FIRST_COMMIT)
    assert msgs[-1]["committed"]
    predictor.feature_extractor.fail = True
    err = predictor.process_frame(_FRAME).to_dict()
    assert err["type"] == "error" and err["committed"] is False
    predictor.feature_extractor.fail = False
    ok = predictor.process_frame(_FRAME).to_dict()
    assert ok["type"] == "prediction" and ok["message"] is None and not ok["committed"]


def test_reset_stream_drops_temporal_state_but_keeps_sentence(make_predictor):
    predictor, clf = make_predictor("HELLO", 0.95)
    predictor.feature_extractor.value = 0.5  # old camera
    _feed(predictor, _W + 3 * _S)
    sentence = predictor.sentence_builder.get_text()

    predictor.reset_stream()
    predictor.feature_extractor.value = 0.1  # new camera
    calls_before = len(clf.calls)
    msgs = _feed(predictor, _W - 1)
    assert len(clf.calls) == calls_before  # needs a full fresh window
    for m in msgs:
        assert m["sign"] == settings.unknown_label and m["confidence"] == 0.0
        assert m["sentence"] == sentence and m["commit_id"] == 1 and not m["committed"]

    clf.label = "WATER"
    msgs = _feed(predictor, 1 + (settings.debounce_repeats - 1) * _S)
    assert len(clf.calls) == calls_before + settings.debounce_repeats
    assert all(not torch.isclose(x, torch.tensor(0.5)).any() for x in clf.inputs[calls_before:])
    assert [m["commit_id"] for m in msgs if m["committed"]] == [2] and msgs[-1]["committed"]
    assert predictor.sentence_builder.gloss_sequence == ["HELLO", "WATER"]


def test_trained_checkpoint_is_loaded_when_present():
    bundle = get_model_bundle()
    checkpoint = settings.models_dir / (
        settings.transformer_checkpoint if settings.model_type == "transformer" else settings.bilstm_checkpoint
    )
    if not checkpoint.exists():
        pytest.skip("no trained checkpoint in backend/models/")
    assert bundle.is_trained
    assert bundle.input_dim == settings.feature_dim
    classes = json.loads((settings.models_dir / settings.classes_file).read_text(encoding="utf-8"))["classes"]
    assert bundle.class_names == classes


class _RecordingTTS:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)
        return True

    def stop(self):
        pass


def test_speech_only_on_explicit_speak_message(monkeypatch):
    tts = _RecordingTTS()
    monkeypatch.setattr(websocket_module, "get_tts_service", lambda: tts)
    import cv2

    jpeg = cv2.imencode(".jpg", np.zeros((120, 160, 3), dtype=np.uint8))[1].tobytes()
    client = TestClient(websocket_module.router)
    with client.websocket_connect("/ws/predict") as ws:
        ws.receive_json()
        for _ in range(settings.realtime_window_frames + settings.sequence_stride):
            ws.send_bytes(jpeg)
            ws.receive_json()
        assert tts.spoken == []

        ws.send_text(json.dumps({"type": "speak"}))
        assert ws.receive_json()["message"] == "nothing_to_speak"  # empty sentence is never spoken
        assert tts.spoken == []


def test_speak_message_speaks_current_sentence_once(make_predictor):
    import asyncio

    predictor, _ = make_predictor("HELLO", 0.95)
    _feed(predictor, settings.realtime_window_frames + 5 * settings.sequence_stride)
    tts = _RecordingTTS()

    class _WS:
        def __init__(self):
            self.sent = []

        async def send_json(self, data):
            self.sent.append(data)

    ws = _WS()
    asyncio.run(websocket_module._handle_control_message(ws, predictor, tts, json.dumps({"type": "speak"})))
    assert tts.spoken == [predictor.sentence_builder.get_text()] and tts.spoken[0]
    assert ws.sent == [{"type": "status", "message": "speaking"}]


def test_reset_message_resets_stream_and_keeps_sentence(monkeypatch):
    """The client sends {"type": "reset"} on a camera switch; frames before
    it may still be in flight, so the ack marks where the new source starts."""
    monkeypatch.setattr(predictor_module, "FeatureExtractor", _FakeExtractor)
    monkeypatch.setattr(predictor_module, "get_model_bundle", lambda: _FakeBundle(_FakeClassifier("HELLO", 0.95)))
    monkeypatch.setattr(websocket_module, "get_tts_service", lambda: _RecordingTTS())
    import cv2

    jpeg = cv2.imencode(".jpg", np.zeros((120, 160, 3), dtype=np.uint8))[1].tobytes()
    client = TestClient(websocket_module.router)
    with client.websocket_connect("/ws/predict") as ws:
        ws.receive_json()
        for _ in range(_FIRST_COMMIT):
            ws.send_bytes(jpeg)
            last = ws.receive_json()
        assert last["committed"] and last["sentence"]

        ws.send_text(json.dumps({"type": "reset"}))
        ack = ws.receive_json()
        assert ack == {"type": "status", "message": "stream_reset", "sentence": last["sentence"]}

        ws.send_bytes(jpeg)
        after = ws.receive_json()
        assert after["sign"] == settings.unknown_label and after["confidence"] == 0.0
        assert after["sentence"] == last["sentence"] and after["commit_id"] == last["commit_id"]
        assert not after["committed"]
