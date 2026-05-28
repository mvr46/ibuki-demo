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

from reachy_mini_conversation_app.config import config


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
        piper_bin = shutil.which("piper")
        if not piper_bin or not self.voice_model:
            logger.warning("Piper TTS unavailable; falling back to macOS say.")
            return self._synthesize_with_macos_say(text)

        with tempfile.TemporaryDirectory(prefix="reachy-piper-") as tmp:
            out_path = Path(tmp) / "speech.wav"
            proc = subprocess.run(
                [piper_bin, "--model", self.voice_model, "--output_file", str(out_path)],
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

    def _synthesize_with_macos_say(self, text: str) -> tuple[int, NDArray[np.int16]]:
        """Synthesize with macOS `say` when Piper is not configured."""
        say_bin = shutil.which("say")
        afconvert_bin = shutil.which("afconvert")
        if not say_bin or not afconvert_bin:
            logger.warning("No local TTS available; install piper or use macOS say.")
            return self.output_sample_rate, np.zeros(0, dtype=np.int16)

        with tempfile.TemporaryDirectory(prefix="reachy-say-") as tmp:
            aiff_path = Path(tmp) / "speech.aiff"
            wav_path = Path(tmp) / "speech.wav"
            say_proc = subprocess.run(
                [say_bin, "-o", str(aiff_path), text],
                text=True,
                capture_output=True,
                check=False,
            )
            if say_proc.returncode != 0:
                logger.warning("macOS say failed: %s", say_proc.stderr.strip() or say_proc.stdout.strip())
                return self.output_sample_rate, np.zeros(0, dtype=np.int16)

            convert_proc = subprocess.run(
                [afconvert_bin, "-f", "WAVE", "-d", "LEI16@24000", str(aiff_path), str(wav_path)],
                text=True,
                capture_output=True,
                check=False,
            )
            if convert_proc.returncode != 0:
                logger.warning("afconvert failed: %s", convert_proc.stderr.strip() or convert_proc.stdout.strip())
                return self.output_sample_rate, np.zeros(0, dtype=np.int16)
            sample_rate, data = wavfile.read(wav_path)

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
