"""Local text-to-speech adapters."""

from __future__ import annotations
import shutil
import asyncio
import logging
import tempfile
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
    """Local Piper TTS adapter using the `piper` command when configured."""

    output_sample_rate = 22050

    def __init__(self, *, voice_model: str | None = None) -> None:
        """Initialize with an optional Piper voice model path."""
        self.voice_model = voice_model or config.PIPER_VOICE

    async def synthesize(self, text: str) -> tuple[int, NDArray[np.int16]]:
        """Synthesize text using Piper when available."""
        return await asyncio.to_thread(self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> tuple[int, NDArray[np.int16]]:
        status = piper_tts_status(self.voice_model)
        piper_bin = status.get("piper_bin")
        voice_model = self.voice_model
        if not status["ready"] or not isinstance(piper_bin, str) or not voice_model:
            logger.error("Piper TTS unavailable: %s", status.get("error") or "not_ready")
            return self.output_sample_rate, np.zeros(0, dtype=np.int16)

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
    selected_voice = (voice_model or config.PIPER_VOICE or "").strip()
    voice_path = Path(selected_voice).expanduser() if selected_voice else None
    voice_exists = bool(voice_path and voice_path.is_file())
    ready = bool(piper_bin and selected_voice and voice_exists)
    error = None
    if not piper_bin:
        error = "missing_piper"
    elif not selected_voice:
        error = "missing_piper_voice"
    elif not voice_exists:
        error = "invalid_piper_voice"
    return {
        "provider": "piper",
        "ready": ready,
        "piper_available": bool(piper_bin),
        "piper_bin": piper_bin,
        "voice_configured": bool(selected_voice),
        "voice_model": selected_voice or None,
        "voice_exists": voice_exists,
        "error": error,
    }
