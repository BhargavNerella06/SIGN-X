"""Text-to-speech service, abstracted behind a small interface so the
concrete engine (currently the offline/local ``pyttsx3`` engine) can later
be swapped for another engine without touching any caller code.

Design notes:
- Speech is triggered explicitly (SPEAK button / a completed phrase being
  committed) — never per-frame.
- ``speak()`` stops any in-flight speech first so rapid re-triggers don't
  queue up a backlog of stale sentences.
- Runs on a background thread so it never blocks the FastAPI event loop or
  the WebSocket frame-processing loop.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Optional

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class TTSService(ABC):
    """Abstract text-to-speech service interface."""

    @abstractmethod
    def speak(self, text: str) -> bool:
        """Speak ``text`` asynchronously. Returns False if the engine is
        unavailable or the text is empty; never raises."""

    @abstractmethod
    def stop(self) -> None:
        """Stop any speech currently in progress."""

    @property
    @abstractmethod
    def is_speaking(self) -> bool:
        ...

    @property
    @abstractmethod
    def available(self) -> bool:
        ...


class NullTTSService(TTSService):
    """No-op engine used when TTS is disabled or no working engine is
    available — keeps the rest of the app from having to special-case it."""

    def speak(self, text: str) -> bool:
        logger.debug("TTS disabled/unavailable — not speaking: %r", text)
        return False

    def stop(self) -> None:
        return None

    @property
    def is_speaking(self) -> bool:
        return False

    @property
    def available(self) -> bool:
        return False


class Pyttsx3TTSService(TTSService):
    """Local/offline TTS via ``pyttsx3`` (uses SAPI5 on Windows, NSSpeechSynthesizer
    on macOS, espeak on Linux — no network calls, no audio leaves the machine)."""

    def __init__(self, rate: Optional[int] = None, volume: Optional[float] = None,
                 voice_id: Optional[str] = None):
        self._rate = rate if rate is not None else settings.tts_rate
        self._volume = volume if volume is not None else settings.tts_volume
        self._voice_id = voice_id if voice_id is not None else settings.tts_voice_id

        self._lock = threading.Lock()
        self._engine = None
        self._thread: Optional[threading.Thread] = None
        self._available = self._probe()

    def _probe(self) -> bool:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.stop()
            return True
        except Exception:
            logger.warning("pyttsx3 TTS engine unavailable on this system", exc_info=True)
            return False

    def speak(self, text: str) -> bool:
        if not self._available or not text or not text.strip():
            return False
        self.stop()  # cancel any in-flight utterance before starting a new one

        thread = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
        self._thread = thread
        thread.start()
        return True

    def _speak_worker(self, text: str) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            if self._voice_id:
                engine.setProperty("voice", self._voice_id)
            with self._lock:
                self._engine = engine
            engine.say(text)
            engine.runAndWait()
        except Exception:
            logger.exception("TTS playback failed")
        finally:
            with self._lock:
                self._engine = None

    def stop(self) -> None:
        with self._lock:
            engine = self._engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                logger.exception("Failed to stop TTS engine")

    @property
    def is_speaking(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def available(self) -> bool:
        return self._available


_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Process-wide TTS service singleton, chosen by ``settings.tts_engine``."""
    global _service
    if _service is not None:
        return _service

    if settings.tts_engine == "pyttsx3":
        service = Pyttsx3TTSService()
        _service = service if service.available else NullTTSService()
    else:
        _service = NullTTSService()

    if not _service.available:
        logger.warning("No working TTS engine — speech output is disabled.")
    return _service
