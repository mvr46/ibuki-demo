"""Tests for Ollama vision analyzer payloads."""

import json
from unittest.mock import patch

import numpy as np

from reachy_mini_conversation_app.vision.analyzers import OllamaVisionAnalyzer


class _Response:
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps({"message": {"content": "A red cup."}}).encode("utf-8")


def test_ollama_vision_sends_image_chat_payload() -> None:
    """Ollama vision adapter should send model, question, and base64 image."""
    captured = {}

    def fake_urlopen(req: object, timeout: float) -> _Response:
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response()

    analyzer = OllamaVisionAnalyzer(base_url="http://ollama.test", model="gemma3:test", timeout_seconds=7)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    with patch("reachy_mini_conversation_app.vision.analyzers.urlopen", fake_urlopen):
        result = analyzer._analyze_sync(frame, "What is visible?")

    assert result == "A red cup."
    assert captured["url"] == "http://ollama.test/api/chat"
    assert captured["timeout"] == 7
    assert captured["payload"]["model"] == "gemma3:test"
    message = captured["payload"]["messages"][0]
    assert message["role"] == "user"
    assert "What is visible?" in message["content"]
    assert len(message["images"]) == 1

