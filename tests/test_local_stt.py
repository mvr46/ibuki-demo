"""Tests for local MLX Whisper STT hardening."""

from __future__ import annotations
import sys
from types import SimpleNamespace

import numpy as np

from reachy_mini_conversation_app.backends.local_stt import MLXWhisperSTTAdapter, reject_transcript_reason


def test_mlx_whisper_uses_strict_english_noise_hardening_kwargs(monkeypatch) -> None:
    """MLX Whisper should receive the strict local transcription options."""
    captured = {}

    def fake_transcribe(audio, **kwargs):
        captured.update(kwargs)
        return {"text": "hello there", "segments": [{"no_speech_prob": 0.01}]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=fake_transcribe))

    text = MLXWhisperSTTAdapter(model="test-model")._transcribe_sync(np.ones(1600, dtype=np.int16), 16000)

    assert text == "hello there"
    assert captured["path_or_hf_repo"] == "test-model"
    assert captured["language"] == "en"
    assert captured["condition_on_previous_text"] is False
    assert captured["no_speech_threshold"] == 0.45
    assert captured["logprob_threshold"] == -0.7
    assert captured["compression_ratio_threshold"] == 1.8
    assert captured["hallucination_silence_threshold"] == 0.3
    assert captured["temperature"] == 0


def test_pathological_transcript_rejector() -> None:
    """Repeated/glyph hallucinations should be filtered before chat."""
    assert reject_transcript_reason("sss") == "repeated_character"
    assert reject_transcript_reason("vvv") == "repeated_character"
    assert reject_transcript_reason("වවව") == "non_english_glyphs"
    assert reject_transcript_reason("la la la la la la") in {"repeated_pattern", "repeated_word"}
    assert reject_transcript_reason("look at matt") is None


def test_mlx_whisper_rejects_pathological_result(monkeypatch) -> None:
    """Adapter should return empty text and store the reject reason."""

    def fake_transcribe(audio, **kwargs):
        return {"text": "ssssss", "segments": [{"no_speech_prob": 0.01}]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=fake_transcribe))
    adapter = MLXWhisperSTTAdapter(model="test-model")

    text = adapter._transcribe_sync(np.ones(1600, dtype=np.int16), 16000)

    assert text == ""
    assert adapter.last_reject_reason == "repeated_character"
