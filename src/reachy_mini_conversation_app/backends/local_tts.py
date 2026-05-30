"""Local text-to-speech adapters."""

from __future__ import annotations
import shutil
import asyncio
import logging
import tempfile
import threading
import subprocess
from typing import Protocol
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from numpy.typing import NDArray

from reachy_mini_conversation_app.runtime.config import config


logger = logging.getLogger(__name__)


class LocalTTSAdapter(Protocol):
    """Interface for local speech synthesis."""

    output_sample_rate: int

    async def synthesize(self, text: str) -> tuple[int, NDArray[np.int16]]:
        """Return synthesized mono PCM."""
        ...


class PiperTTSAdapter:
    """Local Piper TTS adapter using the Python API, with CLI fallback."""

    output_sample_rate = 22050

    def __init__(self, *, voice_model: str | None = None) -> None:
        """Initialize with an optional Piper voice model path."""
        self.voice_model = voice_model or config.PIPER_VOICE
        self._voice: object | None = None
        self._voice_lock = threading.Lock()

    async def synthesize(self, text: str) -> tuple[int, NDArray[np.int16]]:
        """Synthesize text using Piper when available."""
        return await asyncio.to_thread(self._synthesize_sync, text)

    async def warm(self) -> None:
        """Load the Piper voice into memory before the first spoken turn."""
        await asyncio.to_thread(self._load_python_voice)

    def _synthesize_sync(self, text: str) -> tuple[int, NDArray[np.int16]]:
        status = piper_tts_status(self.voice_model)
        voice_model = self.voice_model
        if not status["ready"] or not voice_model:
            logger.error("Piper TTS unavailable: %s", status.get("error") or "not_ready")
            return self.output_sample_rate, np.zeros(0, dtype=np.int16)

        if status.get("python_available"):
            try:
                return self._synthesize_python_sync(text)
            except Exception as exc:
                logger.warning("Python Piper failed, falling back to CLI: %s", exc)

        piper_bin = status.get("piper_bin")
        if not isinstance(piper_bin, str):
            logger.error("Piper CLI unavailable: %s", status.get("error") or "missing_piper")
            return self.output_sample_rate, np.zeros(0, dtype=np.int16)

        return self._synthesize_cli_sync(text, piper_bin, voice_model)

    def _load_python_voice(self) -> object:
        """Load and cache a PiperVoice instance."""
        voice_model = self.voice_model
        if not voice_model:
            raise RuntimeError("missing_piper_voice")
        voice_path = Path(voice_model).expanduser()
        if not voice_path.is_file():
            raise RuntimeError("invalid_piper_voice")
        with self._voice_lock:
            if self._voice is None:
                from piper import PiperVoice

                self._voice = PiperVoice.load(voice_path)
            return self._voice

    def _synthesize_python_sync(self, text: str) -> tuple[int, NDArray[np.int16]]:
        """Synthesize text through the in-process Piper API."""
        voice = self._load_python_voice()
        chunks: list[NDArray[np.int16]] = []
        sample_rate = self.output_sample_rate
        with self._voice_lock:
            synthesize = getattr(voice, "synthesize")
            for chunk in synthesize(text):
                sample_rate = int(getattr(chunk, "sample_rate", sample_rate))
                audio = np.asarray(getattr(chunk, "audio_int16_array"))
                if audio.ndim == 2:
                    audio = audio[:, 0]
                chunks.append(audio.astype(np.int16, copy=False).reshape(-1))
        self.output_sample_rate = int(sample_rate)
        if not chunks:
            return int(sample_rate), np.zeros(0, dtype=np.int16)
        return int(sample_rate), np.concatenate(chunks).astype(np.int16, copy=False)

    def _synthesize_cli_sync(self, text: str, piper_bin: str, voice_model: str) -> tuple[int, NDArray[np.int16]]:
        """Synthesize text through the Piper CLI fallback."""
        with tempfile.TemporaryDirectory(prefix="reachy-piper-") as tmp:
            out_path = Path(tmp) / "speech.wav"
            proc = subprocess.run(
                [piper_bin, "--model", voice_model, "--output_file", str(out_path)],
                input=text,
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                logger.warning("Piper failed: %s", proc.stderr.strip() or proc.stdout.strip())
                return self.output_sample_rate, np.zeros(0, dtype=np.int16)
            sample_rate, data = wavfile.read(out_path)

        audio = np.asarray(data)
        if audio.ndim == 2:
            audio = audio[:, 0]
        if audio.dtype != np.int16:
            if np.issubdtype(audio.dtype, np.floating):
                audio = np.clip(audio, -1.0, 1.0) * 32767
            audio = audio.astype(np.int16)
        self.output_sample_rate = int(sample_rate)
        return int(sample_rate), audio


class NoopTTSAdapter:
    """Testing/null adapter that returns silence."""

    output_sample_rate = 24000

    async def synthesize(self, text: str) -> tuple[int, NDArray[np.int16]]:
        """Return empty audio."""
        return self.output_sample_rate, np.zeros(0, dtype=np.int16)


def piper_tts_status(voice_model: str | None = None) -> dict[str, object]:
    """Return strict local Piper readiness information."""
    piper_bin = shutil.which("piper")
    python_available = _python_piper_available()
    selected_voice = (voice_model or config.PIPER_VOICE or "").strip()
    voice_path = Path(selected_voice).expanduser() if selected_voice else None
    voice_exists = bool(voice_path and voice_path.is_file())
    ready = bool((python_available or piper_bin) and selected_voice and voice_exists)
    error = None
    if not (python_available or piper_bin):
        error = "missing_piper"
    elif not selected_voice:
        error = "missing_piper_voice"
    elif not voice_exists:
        error = "invalid_piper_voice"
    return {
        "provider": "piper",
        "ready": ready,
        "python_available": python_available,
        "piper_available": bool(piper_bin),
        "piper_bin": piper_bin,
        "voice_configured": bool(selected_voice),
        "voice_model": selected_voice or None,
        "voice_exists": voice_exists,
        "error": error,
    }


def _python_piper_available() -> bool:
    """Return whether the in-process Piper API can be imported."""
    try:
        from piper import PiperVoice  # noqa: F401
    except Exception:
        return False
    return True
