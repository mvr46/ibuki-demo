"""Local speech-to-text adapters."""

from __future__ import annotations
import re
import asyncio
import logging
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.runtime.config import config


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
        self.last_reject_reason: str | None = None

    async def transcribe(self, audio: NDArray[np.int16], sample_rate: int) -> str:
        """Transcribe one complete utterance."""
        return await asyncio.to_thread(self._transcribe_sync, audio, sample_rate)

    def _transcribe_sync(self, audio: NDArray[np.int16], sample_rate: int) -> str:
        try:
            import mlx_whisper
        except Exception as exc:
            logger.warning("mlx-whisper is unavailable: %s", exc)
            self.last_reject_reason = "mlx_whisper_unavailable"
            return ""

        if sample_rate != 16000:
            logger.warning("MLXWhisperSTTAdapter expected 16 kHz audio, got %s Hz", sample_rate)
        audio_float = audio.astype(np.float32) / 32768.0
        self.last_reject_reason = None
        result = mlx_whisper.transcribe(
            audio_float,
            path_or_hf_repo=self.model,
            language="en",
            condition_on_previous_text=False,
            no_speech_threshold=0.45,
            logprob_threshold=-0.7,
            compression_ratio_threshold=1.8,
            hallucination_silence_threshold=0.3,
            temperature=0,
        )
        if isinstance(result, dict):
            text = str(result.get("text") or "").strip()
            if _result_indicates_no_speech(result):
                self.last_reject_reason = "whisper_no_speech"
                return ""
        else:
            text = str(result or "").strip()
        reject_reason = reject_transcript_reason(text)
        if reject_reason is not None:
            self.last_reject_reason = reject_reason
            logger.info("Rejected local STT transcript reason=%s text=%r", reject_reason, text[:80])
            return ""
        return text


class NoopSTTAdapter:
    """Testing/null adapter that never emits text."""

    async def transcribe(self, audio: NDArray[np.int16], sample_rate: int) -> str:
        """Return no transcription."""
        return ""


def _result_indicates_no_speech(result: dict[object, object]) -> bool:
    """Return whether an mlx-whisper result carries strong no-speech metadata."""
    segments = result.get("segments")
    if not isinstance(segments, list):
        return False
    no_speech_probs = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        value = segment.get("no_speech_prob")
        if isinstance(value, int | float):
            no_speech_probs.append(float(value))
    return bool(no_speech_probs and min(no_speech_probs) >= 0.8)


def reject_transcript_reason(text: str) -> str | None:
    """Return a reject reason for hallucinated/noise-like transcripts."""
    candidate = text.strip()
    if not candidate:
        return "empty_transcript"

    compact = re.sub(r"\s+", "", candidate.casefold())
    alnum = "".join(ch for ch in compact if ch.isalnum())
    letters = [ch for ch in alnum if ch.isalpha()]
    ascii_letters = [ch for ch in letters if "a" <= ch <= "z"]

    if letters and not ascii_letters:
        return "non_english_glyphs"
    if len(alnum) >= 3 and len(set(alnum)) == 1:
        return "repeated_character"
    if _is_repeated_short_pattern(alnum):
        return "repeated_pattern"
    if len(alnum) >= 8 and len(set(alnum)) / len(alnum) < 0.25:
        return "low_diversity_transcript"

    words = re.findall(r"[a-zA-Z']+", candidate.casefold())
    if len(words) >= 4 and len(set(words)) == 1:
        return "repeated_word"
    if len(words) >= 6:
        bigrams = list(zip(words, words[1:]))
        if bigrams and max(bigrams.count(item) for item in set(bigrams)) >= 3:
            return "repeated_phrase"

    return None


def _is_repeated_short_pattern(value: str) -> bool:
    """Return whether a string is made of one short unit repeated."""
    if len(value) < 6:
        return False
    for size in range(1, 4):
        if len(value) % size:
            continue
        unit = value[:size]
        if unit and unit * (len(value) // size) == value and len(value) // size >= 3:
            return True
    return False
