"""Tests for local text-to-speech adapters."""

from __future__ import annotations
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from reachy_mini_conversation_app.backends.local_tts import PiperTTSAdapter, piper_tts_status


def test_piper_tts_does_not_fall_back_to_macos_say(monkeypatch) -> None:
    """Missing Piper or PIPER_VOICE should return silence without invoking say."""
    calls = []

    def fake_which(name: str) -> str | None:
        return {
            "piper": None,
            "say": "/usr/bin/say",
            "afconvert": "/usr/bin/afconvert",
        }.get(name)

    def fake_run(args, **kwargs):
        calls.append(args)
        out_path = Path(args[-1])
        if out_path.suffix == ".aiff":
            out_path.write_bytes(b"AIFF")
        elif out_path.suffix == ".wav":
            out_path.write_bytes(b"WAV")

        class Proc:
            returncode = 0
            stderr = ""
            stdout = ""

        return Proc()

    monkeypatch.setattr("reachy_mini_conversation_app.backends.local_tts.shutil.which", fake_which)
    monkeypatch.setattr("reachy_mini_conversation_app.backends.local_tts.subprocess.run", fake_run)
    monkeypatch.setattr("reachy_mini_conversation_app.backends.local_tts.config.PIPER_VOICE", None)

    sample_rate, audio = PiperTTSAdapter(voice_model=None)._synthesize_sync("hello")

    assert sample_rate == 22050
    assert audio.size == 0
    assert calls == []


def test_piper_status_requires_voice_file(monkeypatch, tmp_path: Path) -> None:
    """Piper readiness should require both binary and a real voice model path."""
    voice = tmp_path / "voice.onnx"
    voice.write_bytes(b"voice")
    monkeypatch.setattr("reachy_mini_conversation_app.backends.local_tts.shutil.which", lambda name: "/usr/bin/piper")

    ready = piper_tts_status(str(voice))
    missing = piper_tts_status(str(tmp_path / "missing.onnx"))

    assert ready["ready"] is True
    assert ready["error"] is None
    assert missing["ready"] is False
    assert missing["error"] == "invalid_piper_voice"


def test_piper_python_api_path_loads_once_and_returns_int16(monkeypatch, tmp_path: Path) -> None:
    """Warmed in-process Piper should synthesize int16 PCM without requiring the CLI."""
    voice = tmp_path / "voice.onnx"
    voice.write_bytes(b"voice")
    loads = []

    class FakeVoice:
        def synthesize(self, text: str):
            assert text == "hello"
            yield SimpleNamespace(sample_rate=16000, audio_int16_array=np.array([1, -2, 3], dtype=np.int16))

    class FakePiperVoice:
        @staticmethod
        def load(path: Path) -> FakeVoice:
            loads.append(path)
            return FakeVoice()

    monkeypatch.setitem(sys.modules, "piper", SimpleNamespace(PiperVoice=FakePiperVoice))
    monkeypatch.setattr("reachy_mini_conversation_app.backends.local_tts.shutil.which", lambda name: None)

    adapter = PiperTTSAdapter(voice_model=str(voice))
    sample_rate, audio = adapter._synthesize_sync("hello")
    sample_rate_2, audio_2 = adapter._synthesize_sync("hello")

    assert sample_rate == 16000
    assert sample_rate_2 == 16000
    assert audio.dtype == np.int16
    assert audio.tolist() == [1, -2, 3]
    assert audio_2.tolist() == [1, -2, 3]
    assert loads == [voice]
