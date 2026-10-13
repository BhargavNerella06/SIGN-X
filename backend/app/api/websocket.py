"""Real-time prediction WebSocket endpoint.

Protocol (Section 18):

Client -> Server
    - binary frame: one JPEG/PNG-encoded image (a single webcam frame)
    - text frame: JSON control message, e.g. {"type": "clear"},
      {"type": "speak"}, {"type": "reset"} (camera switched) or {"type": "ping"}

Server -> Client (JSON text frames)
    - {"type": "connection", "status": "connected", "model_info": {...}}
    - {"type": "prediction", "sign": ..., "confidence": ..., "sentence": ...,
       "fps": ..., "latency_ms": ..., ...}
    - {"type": "status", ...}
    - {"type": "error", "message": ...}

Each connection gets its own ``RealtimePredictor`` instance so temporal
state (the sliding window buffer, the sentence being built) is never mixed
between two simultaneous users.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.inference.predictor import RealtimePredictor
from app.speech.tts import get_tts_service
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Reject absurdly large payloads outright (Section 38: validate input).
_MAX_FRAME_BYTES = 5 * 1024 * 1024  # 5 MB


def _decode_frame(data: bytes) -> np.ndarray | None:
    if not data or len(data) > _MAX_FRAME_BYTES:
        return None
    array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return frame


@router.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket) -> None:
    await websocket.accept()
    predictor = RealtimePredictor()
    tts = get_tts_service()

    await websocket.send_json({
        "type": "connection",
        "status": "connected",
        "model_info": predictor.bundle.info(),
        "vision_available": predictor.vision_available,
    })

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if (data := message.get("bytes")) is not None:
                frame = _decode_frame(data)
                if frame is None:
                    await websocket.send_json({"type": "error", "message": "Invalid or oversized frame"})
                    continue
                result = predictor.process_frame(frame)
                await websocket.send_json(result.to_dict())

            elif (text := message.get("text")) is not None:
                await _handle_control_message(websocket, predictor, tts, text)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.exception("Unexpected WebSocket error")
        try:
            await websocket.send_json({"type": "error", "message": "Internal server error"})
        except Exception:
            pass
    finally:
        predictor.close()


async def _handle_control_message(websocket: WebSocket, predictor: RealtimePredictor, tts, text: str) -> None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if not isinstance(payload, dict):
        # Valid JSON that is not an object ([], null, 5, ...) is just as malformed.
        await websocket.send_json({"type": "error", "message": "Malformed control message"})
        return

    msg_type = payload.get("type")

    if msg_type == "clear":
        predictor.clear_sentence()
        await websocket.send_json({"type": "status", "message": "Sentence cleared", "sentence": ""})

    elif msg_type == "reset":
        # Video source changed (camera switch): drop temporal state, keep the sentence.
        predictor.reset_stream()
        await websocket.send_json({
            "type": "status",
            "message": "stream_reset",
            "sentence": predictor.sentence_builder.get_text(),
        })

    elif msg_type == "speak":
        sentence = predictor.sentence_builder.get_text()
        spoken = tts.speak(sentence) if sentence else False
        await websocket.send_json({"type": "status", "message": "speaking" if spoken else "nothing_to_speak"})

    elif msg_type == "stop_speech":
        tts.stop()
        await websocket.send_json({"type": "status", "message": "speech_stopped"})

    elif msg_type == "ping":
        await websocket.send_json({"type": "status", "message": "pong"})

    else:
        await websocket.send_json({"type": "error", "message": f"Unknown control message type: {msg_type}"})
