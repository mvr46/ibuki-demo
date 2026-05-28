"""Tests for local text-to-speech adapters."""

from __future__ import annotations
from pathlib import Path

import numpy as np

from reachy_mini_conversation_app.local_tts import PiperTTSAdapter


def test_piper_tts_falls_back_to_macos_say(monkeypatch) -> None:
    """Missing Piper should use macOS say and return converted PCM audio."""
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

    def fake_read(path: Path) -> tuple[int, np.ndarray]:
        assert path.name == "speech.wav"
        return 24000, np.ones(8, dtype=np.int16)

    monkeypatch.setattr("reachy_mini_conversation_app.local_tts.shutil.which", fake_which)
    monkeypatch.setattr("reachy_mini_conversation_app.local_tts.subprocess.run", fake_run)
    monkeypatch.setattr("reachy_mini_conversation_app.local_tts.wavfile.read", fake_read)

    sample_rate, audio = PiperTTSAdapter(voice_model=None)._synthesize_sync("hello")

    assert sample_rate == 24000
    assert audio.shape == (8,)
    assert calls[0][0] == "/usr/bin/say"
    assert calls[1][0] == "/usr/bin/afconvert"
