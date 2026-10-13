"""Robustness of the real-time path: malformed control messages, processing
errors, non-finite features, connection lifecycle and isolation.

MediaPipe is replaced with a scripted fake extractor (no webcam needed);
tests marked "real model" run the shipped trained ensemble.
"""

from __future__ import annotations

import asyncio
import json
import math

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.inference.predictor as predictor_module
from app.api import websocket as websocket_module
from app.config import settings
from app.inference.predictor import RealtimePredictor
from app.models.classifier import ClassificationResult
from app.models.model_loader import get_model_bundle
from app.vision.landmarks import FrameLandmarks, empty_hand

_FRAME = np.zeros((48, 64, 3), dtype=np.uint8)
_JPEG = cv2.imencode(".jpg", np.zeros((120, 160, 3), dtype=np.uint8))[1].tobytes()
_FIRST_COMMIT = settings.realtime_window_frames + (settings.debounce_repeats - 1) * settings.sequence_stride


class _ScriptedExtractor:
    """Fake FeatureExtractor. ``vector`` overrides the next frame's features."""

    available = True

    def __init__(self, *args, **kwargs):
        self.hands = True
        self.value = 0.1
        self.vector = None
        self.closed = False

    def extract(self, frame_bgr):
        hand = empty_hand() + 0.1 if self.hands else None
        raw = FrameLandmarks(left_hand=None, right_hand=hand, pose=None, face=None)
        if self.vector is not None:
            vector, self.vector = self.vector, None
        else:
            vector = np.full(settings.feature_dim, self.value if self.hands else 0.0, dtype=np.float32)
        return vector, raw

    def reset(self):
        pass

    def close(self):
        self.closed = True


class _Classifier:
    def __init__(self, label="HELLO", prob=0.95):
        self.label, self.prob, self.fail_next = label, prob, 0

    def predict(self, x, mask=None, top_k=3):
        if self.fail_next:
            self.fail_next -= 1
            raise RuntimeError("simulated model failure")
        return ClassificationResult(top_indices=[0], top_labels=[self.label], top_probs=[self.prob])


class _Bundle:
    def __init__(self, classifier):
        self.classifier = classifier

    def info(self):
        return {"is_trained": True}


class _NullTTS:
    def __init__(self):
        self.spoken, self.stopped = [], 0

    def speak(self, text):
        self.spoken.append(text)
        return True

    def stop(self):
        self.stopped += 1


class _FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, data):
        self.sent.append(data)


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Patch the server so /ws/predict uses fake vision + a scripted model."""
    monkeypatch.setattr(predictor_module, "FeatureExtractor", _ScriptedExtractor)
    clf = _Classifier()
    monkeypatch.setattr(predictor_module, "get_model_bundle", lambda: _Bundle(clf))
    tts = _NullTTS()
    monkeypatch.setattr(websocket_module, "get_tts_service", lambda: tts)
    return clf, tts


def _strict_json(msg: dict) -> str:
    """What the browser's JSON.parse would accept (no NaN/Infinity)."""
    return json.dumps(msg, allow_nan=False)


# --- control messages --------------------------------------------------------

@pytest.mark.parametrize("payload", ["[]", "null", "5", '"clear"', "true", '[{"type": "clear"}]'])
def test_non_object_json_control_message_is_rejected_without_raising(payload):
    """Regression: valid JSON that is not an object used to raise
    AttributeError, which ended the whole WebSocket session."""
    predictor = RealtimePredictor(model_bundle=_Bundle(_Classifier()))
    ws = _FakeSocket()
    asyncio.run(websocket_module._handle_control_message(ws, predictor, _NullTTS(), payload))
    assert ws.sent == [{"type": "error", "message": "Malformed control message"}]


def test_session_survives_non_object_control_messages(fake_pipeline):
    client = TestClient(websocket_module.router)
    with client.websocket_connect("/ws/predict") as ws:
        ws.receive_json()
        for payload in ["[]", "null", "5", '"x"']:
            ws.send_text(payload)
            assert ws.receive_json() == {"type": "error", "message": "Malformed control message"}
        ws.send_text(json.dumps({"type": "ping"}))
        assert ws.receive_json() == {"type": "status", "message": "pong"}
        ws.send_bytes(_JPEG)
        assert ws.receive_json()["type"] == "prediction"


@pytest.mark.parametrize("payload", ["{}", '{"type": 5}', '{"type": null}', '{"type": "CLEAR"}'])
def test_object_without_a_known_type_is_reported(payload):
    predictor = RealtimePredictor(model_bundle=_Bundle(_Classifier()))
    ws = _FakeSocket()
    asyncio.run(websocket_module._handle_control_message(ws, predictor, _NullTTS(), payload))
    assert ws.sent[0]["type"] == "error" and ws.sent[0]["message"].startswith("Unknown control message type")


def test_stop_speech_and_ping_messages(fake_pipeline):
    _, tts = fake_pipeline
    client = TestClient(websocket_module.router)
    with client.websocket_connect("/ws/predict") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "stop_speech"}))
        assert ws.receive_json() == {"type": "status", "message": "speech_stopped"}
        assert tts.stopped == 1
        ws.send_text(json.dumps({"type": "ping"}))
        assert ws.receive_json() == {"type": "status", "message": "pong"}


def test_every_binary_frame_gets_exactly_one_reply(fake_pipeline):
    """The frontend's in-flight limiter relies on one prediction/error reply per frame."""
    client = TestClient(websocket_module.router)
    with client.websocket_connect("/ws/predict") as ws:
        ws.receive_json()
        for data in [_JPEG, b"", b"not an image", _JPEG, b"\x00" * 64]:
            ws.send_bytes(data)
            reply = ws.receive_json()
            assert reply["type"] in ("prediction", "error")
        ws.send_text(json.dumps({"type": "ping"}))
        assert ws.receive_json()["message"] == "pong"  # nothing extra was queued


# --- processing errors -------------------------------------------------------

def test_model_failure_degrades_to_an_error_message_and_recovers(monkeypatch):
    """process_frame promises never to raise; an exception in the model must
    not propagate (it would end the WebSocket session)."""
    monkeypatch.setattr(predictor_module, "FeatureExtractor", _ScriptedExtractor)
    clf = _Classifier("HELLO", 0.95)
    predictor = RealtimePredictor(model_bundle=_Bundle(clf))
    for _ in range(settings.realtime_window_frames - 1):
        predictor.process_frame(_FRAME)

    clf.fail_next = 1
    msg = predictor.process_frame(_FRAME).to_dict()  # this frame triggers inference
    assert msg["type"] == "error" and msg["committed"] is False

    msgs = [predictor.process_frame(_FRAME).to_dict() for _ in range(_FIRST_COMMIT)]
    assert msgs[-1]["type"] == "prediction"
    assert sum(m["committed"] for m in msgs) == 1  # recovered, committed once
    assert predictor.sentence_builder.gloss_sequence == ["HELLO"]


def test_model_failure_does_not_end_the_session(fake_pipeline):
    clf, _ = fake_pipeline
    client = TestClient(websocket_module.router)
    with client.websocket_connect("/ws/predict") as ws:
        ws.receive_json()
        clf.fail_next = 10**6  # every inference fails
        replies = []
        for _ in range(settings.realtime_window_frames + 2 * settings.sequence_stride):
            ws.send_bytes(_JPEG)
            replies.append(ws.receive_json())
        assert any(r["type"] == "error" for r in replies)
        clf.fail_next = 0
        ws.send_text(json.dumps({"type": "ping"}))
        assert ws.receive_json()["message"] == "pong"


def test_error_after_commit_never_repeats_the_commit(monkeypatch):
    monkeypatch.setattr(predictor_module, "FeatureExtractor", _ScriptedExtractor)
    clf = _Classifier("HELLO", 0.95)
    predictor = RealtimePredictor(model_bundle=_Bundle(clf))
    msgs = [predictor.process_frame(_FRAME).to_dict() for _ in range(_FIRST_COMMIT)]
    assert msgs[-1]["committed"]
    clf.fail_next = 10**6
    later = [predictor.process_frame(_FRAME).to_dict() for _ in range(3 * settings.sequence_stride)]
    assert not any(m["committed"] for m in later)
    assert {m["commit_id"] for m in later} == {msgs[-1]["commit_id"]}


# --- non-finite input (real model) -------------------------------------------

def test_non_finite_features_never_reach_the_client_as_nan(monkeypatch):
    """One corrupt frame (NaN/inf features) must not poison the following
    window of predictions: the trained models would turn it into NaN
    probabilities, which are not valid JSON for the browser."""
    monkeypatch.setattr(predictor_module, "FeatureExtractor", _ScriptedExtractor)
    predictor = RealtimePredictor(model_bundle=get_model_bundle())  # real trained ensemble
    extractor = predictor.feature_extractor

    msgs = []
    for i in range(settings.realtime_window_frames + 6 * settings.sequence_stride):
        if i == settings.realtime_window_frames // 2:
            bad = np.full(settings.feature_dim, 0.1, dtype=np.float32)
            bad[:5] = [np.nan, np.inf, -np.inf, np.nan, np.nan]
            extractor.vector = bad
        msgs.append(predictor.process_frame(_FRAME).to_dict())

    for m in msgs:
        _strict_json(m)  # raises ValueError on NaN/inf
        assert math.isfinite(m["confidence"])
        assert all(math.isfinite(p["confidence"]) for p in m["top_predictions"])


# --- lifecycle / isolation ---------------------------------------------------

def test_disconnect_closes_the_predictor(monkeypatch, fake_pipeline):
    created = []
    real_cls = websocket_module.RealtimePredictor

    def factory(*args, **kwargs):
        p = real_cls(*args, **kwargs)
        created.append(p)
        return p

    monkeypatch.setattr(websocket_module, "RealtimePredictor", factory)
    client = TestClient(websocket_module.router)
    for _ in range(3):
        with client.websocket_connect("/ws/predict") as ws:
            ws.receive_json()
            ws.send_bytes(_JPEG)
            ws.receive_json()
    assert len(created) == 3
    assert all(p.feature_extractor.closed for p in created)


def test_sequential_connections_have_independent_state(fake_pipeline):
    client = TestClient(websocket_module.router)
    for _ in range(3):
        with client.websocket_connect("/ws/predict") as ws:
            assert ws.receive_json()["type"] == "connection"
            for _ in range(_FIRST_COMMIT):
                ws.send_bytes(_JPEG)
                last = ws.receive_json()
            # Each connection starts from scratch: first commit is id 1, one sign.
            assert last["committed"] and last["commit_id"] == 1
            assert last["sentence"] == "Hello."


def test_clear_then_same_sign_can_be_committed_again(fake_pipeline):
    client = TestClient(websocket_module.router)
    with client.websocket_connect("/ws/predict") as ws:
        ws.receive_json()
        for _ in range(_FIRST_COMMIT):
            ws.send_bytes(_JPEG)
            ws.receive_json()
        ws.send_text(json.dumps({"type": "clear"}))
        assert ws.receive_json() == {"type": "status", "message": "Sentence cleared", "sentence": ""}
        commits = []
        for _ in range(settings.debounce_repeats * settings.sequence_stride + 1):
            ws.send_bytes(_JPEG)
            r = ws.receive_json()
            commits += [r["commit_id"]] if r["committed"] else []
        assert commits == [2]  # recommitted once, with a fresh id
