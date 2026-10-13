"""Tests for the /ws/predict WebSocket protocol (Section 18). Uses
FastAPI's TestClient, which drives the ASGI app in-process — no real
network socket or webcam is needed."""

from __future__ import annotations

import json

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

_REQUIRED_PREDICTION_KEYS = {
    "type", "sign", "confidence", "sentence", "fps", "latency_ms",
}


def _make_jpeg_bytes() -> bytes:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def test_connection_message_on_connect():
    client = TestClient(app)
    with client.websocket_connect("/ws/predict") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "connection"
        assert msg["status"] == "connected"
        assert "model_info" in msg
        assert "vision_available" in msg


def test_prediction_message_has_required_fields():
    client = TestClient(app)
    with client.websocket_connect("/ws/predict") as ws:
        ws.receive_json()  # connection message
        ws.send_bytes(_make_jpeg_bytes())
        msg = ws.receive_json()
        assert _REQUIRED_PREDICTION_KEYS.issubset(msg.keys())
        assert msg["type"] in ("prediction", "error")
        assert isinstance(msg["confidence"], float)
        assert isinstance(msg["fps"], float)


def test_invalid_binary_frame_returns_error_not_crash():
    client = TestClient(app)
    with client.websocket_connect("/ws/predict") as ws:
        ws.receive_json()
        ws.send_bytes(b"not a real image")
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_clear_control_message():
    client = TestClient(app)
    with client.websocket_connect("/ws/predict") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "clear"}))
        msg = ws.receive_json()
        assert msg["type"] == "status"
        assert msg["sentence"] == ""


def test_unknown_control_message_type_returns_error():
    client = TestClient(app)
    with client.websocket_connect("/ws/predict") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "does_not_exist"}))
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_malformed_json_control_message_returns_error():
    client = TestClient(app)
    with client.websocket_connect("/ws/predict") as ws:
        ws.receive_json()
        ws.send_text("{not valid json")
        msg = ws.receive_json()
        assert msg["type"] == "error"
