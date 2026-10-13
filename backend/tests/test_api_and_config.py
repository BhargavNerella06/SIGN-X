"""REST endpoints, CORS, configuration validation and the TTS service's
failure modes. Uses the real app (and the real trained checkpoints)."""

from __future__ import annotations

import sys
import threading
import types

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings, settings
from app.main import app
from app.speech import tts as tts_module


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the lifespan (model load) like uvicorn does
        yield c


# --- REST endpoints ----------------------------------------------------------

def test_root_lists_entry_points(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"app": "SIGN-X", "docs": "/docs", "websocket": "/ws/predict"}


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["app"] == "SIGN-X"
    assert body["device"] in ("cpu", "cuda")


def test_model_info_reports_the_trained_live_ensemble(client):
    info = client.get("/api/model/info").json()
    assert info["is_trained"] is True
    assert info["checkpoint_path"].endswith("isl_transformer.pt")
    assert info["input_dim"] == 330 and info["num_classes"] == 28 == len(info["class_names"])
    assert info["ensemble_members"] == ["transformer", "bilstm"]
    assert info["num_parameters"] > 1_000_000
    assert info["dataset_name"] == "isl_include_subset28"
    assert set(info["metrics"]) >= {"val_accuracy", "epoch"}


def test_dataset_info_shape(client):
    body = client.get("/api/dataset/info").json()
    assert body["status"] in ("ok", "not_prepared", "not_found")
    if body["status"] == "ok":
        ds = {d["name"]: d for d in body["datasets"]}
        if "isl_include_subset28" in ds:
            assert ds["isl_include_subset28"]["num_classes"] == 28
            assert ds["isl_include_subset28"]["splits"] == {"train": 370, "validation": 60, "test": 150}


@pytest.mark.parametrize("body, status", [
    ({"model_type": "cnn"}, 400),
    ({"model_type": 5}, 422),
    (None, 422),
])
def test_model_reload_rejects_invalid_requests(client, body, status):
    r = client.post("/api/model/reload", json=body) if body is not None else client.post(
        "/api/model/reload", content=b"not json", headers={"Content-Type": "application/json"})
    assert r.status_code == status


def test_model_reload_can_be_disabled(client, monkeypatch):
    monkeypatch.setattr(settings, "allow_model_reload", False)
    r = client.post("/api/model/reload", json={})
    assert r.status_code == 403
    # ...and the served model is untouched
    assert client.get("/api/model/info").json()["is_trained"] is True


def test_model_reload_keeps_trained_weights(client):
    r = client.post("/api/model/reload", json={})
    assert r.status_code == 200
    info = r.json()
    assert info["is_trained"] is True and info["model_type"] == settings.model_type


def test_unknown_route_and_wrong_method(client):
    assert client.get("/api/does-not-exist").status_code == 404
    assert client.post("/api/health").status_code == 405


def test_cors_preflight_is_answered(client):
    r = client.options("/api/model/info", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] in ("*", "http://localhost:5173")


# --- configuration -----------------------------------------------------------

def test_default_configuration_is_consistent():
    s = Settings(_env_file=None)
    assert s.feature_dim == 330
    assert s.sequence_length == 32 and s.realtime_window_frames >= s.sequence_length
    assert 0 < s.confidence_threshold < 1
    assert 1 <= s.debounce_repeats <= s.smoothing_window
    assert 0 <= s.min_hand_frame_ratio <= 1
    assert s.inference_threads == 1
    assert s.allow_model_reload is True


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("SIGNX_MODEL_TYPE", "bilstm")
    monkeypatch.setenv("SIGNX_CORS_ORIGINS", '["https://signx.example.com"]')
    monkeypatch.setenv("SIGNX_ALLOW_MODEL_RELOAD", "false")
    monkeypatch.setenv("SIGNX_TTS_ENGINE", "none")
    s = Settings(_env_file=None)
    assert s.model_type == "bilstm"
    assert s.cors_origins == ["https://signx.example.com"]
    assert s.allow_model_reload is False
    assert s.tts_engine == "none"


@pytest.mark.parametrize("var, value", [
    ("SIGNX_MODEL_TYPE", "cnn"),
    ("SIGNX_DEVICE", "tpu"),
    ("SIGNX_TTS_ENGINE", "espeak"),
    ("SIGNX_SEQUENCE_LENGTH", "thirty-two"),
    ("SIGNX_ENVIRONMENT", "staging"),
])
def test_invalid_configuration_is_rejected(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_feature_dim_tracks_enabled_groups():
    assert Settings(_env_file=None, use_face=False).feature_dim == 126 + 99
    assert Settings(_env_file=None, use_pose=False, use_face=False).feature_dim == 126
    assert Settings(_env_file=None, use_velocity_features=True).feature_dim == 660


def test_explicit_device_is_respected():
    assert Settings(_env_file=None, device="cpu").resolved_device == "cpu"


# --- TTS failure modes -------------------------------------------------------

@pytest.fixture
def fresh_tts(monkeypatch):
    monkeypatch.setattr(tts_module, "_service", None)
    yield
    tts_module._service = None


def _fake_pyttsx3(init):
    mod = types.ModuleType("pyttsx3")
    mod.init = init
    return mod


def test_tts_falls_back_to_null_when_engine_cannot_start(monkeypatch, fresh_tts):
    def broken_init():
        raise OSError("no audio device")

    monkeypatch.setitem(sys.modules, "pyttsx3", _fake_pyttsx3(broken_init))
    monkeypatch.setattr(settings, "tts_engine", "pyttsx3")
    service = tts_module.get_tts_service()
    assert isinstance(service, tts_module.NullTTSService)
    assert service.available is False
    assert service.speak("Hello.") is False
    service.stop()  # no-op, must not raise
    assert service.is_speaking is False


def test_tts_disabled_by_configuration(monkeypatch, fresh_tts):
    monkeypatch.setattr(settings, "tts_engine", "none")
    assert isinstance(tts_module.get_tts_service(), tts_module.NullTTSService)


def test_tts_playback_failure_is_contained(monkeypatch):
    done = threading.Event()

    class Engine:
        def stop(self):
            pass

        def setProperty(self, *a):
            pass

        def say(self, text):
            pass

        def runAndWait(self):
            done.set()
            raise RuntimeError("audio device vanished")

    monkeypatch.setitem(sys.modules, "pyttsx3", _fake_pyttsx3(lambda: Engine()))
    service = tts_module.Pyttsx3TTSService()
    assert service.available
    assert service.speak("") is False and service.speak("   ") is False
    assert service.speak("Hello.") is True
    assert done.wait(5)
    service._thread.join(5)
    assert service.is_speaking is False
    service.stop()  # engine already released; must not raise
