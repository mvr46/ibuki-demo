"""Local speech-to-text adapters."""

from __future__ import annotations
import asyncio
import logging
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)


class LocalSTTAdapter(Protocol):
    """Interface for local speech-to-text."""

    async def transcribe(self, audio: NDArray[np.int16], sample_rate: int) -> str:
        """Return text for one complete utterance."""
        ...


class MLXWhisperSTTAdapter:
    """Apple Silicon Whisper adapter using mlx-whisper when installed."""

    def __init__(self, *, model: str | None = None) -> None:
        """Initialize the MLX Whisper model selector."""
        self.model = model or config.LOCAL_STT_MODEL

    async def transcribe(self, audio: NDArray[np.int16], sample_rate: int) -> str:
        """Transcribe one complete utterance."""
        return await asyncio.to_thread(self._transcribe_sync, audio, sample_rate)

    def _transcribe_sync(self, audio: NDArray[np.int16], sample_rate: int) -> str:
        try:
            import mlx_whisper
        except Exception as exc:
            logger.warning("mlx-whisper is unavailable: %s", exc)
            return ""

        if sample_rate != 16000:
            logger.warning("MLXWhisperSTTAdapter expected 16 kHz audio, got %s Hz", sample_rate)
        audio_float = audio.astype(np.float32) / 32768.0
        result = mlx_whisper.transcribe(audio_float, path_or_hf_repo=self.model)
        if isinstance(result, dict):
            return str(result.get("text") or "").strip()
        return str(result or "").strip()


class NoopSTTAdapter:
    """Testing/null adapter that never emits text."""

    async def transcribe(self, audio: NDArray[np.int16], sample_rate: int) -> str:
        """Return no transcription."""
        return ""
