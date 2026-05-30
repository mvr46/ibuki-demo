"""Tests for local vision analyzer payloads."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from reachy_mini_conversation_app.runtime.config import config
from reachy_mini_conversation_app.vision.analyzers import (
    OpenAICompatibleVisionAnalyzer,
    build_default_vision_analyzer,
)


class _OpenAIResponse:
    def __enter__(self) -> "_OpenAIResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps({"choices": [{"message": {"content": "A keyboard is visible."}}]}).encode("utf-8")


def test_openai_compatible_vision_sends_image_url_payload() -> None:
    """llama.cpp vision adapter should use OpenAI chat completions with a data URL image."""
    captured = {}

    def fake_urlopen(req: object, timeout: float) -> _OpenAIResponse:
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _OpenAIResponse()

    analyzer = OpenAICompatibleVisionAnalyzer(
        base_url="http://llama.test/v1",
        model="ggml-org/SmolVLM2-500M-Video-Instruct-GGUF",
        timeout_seconds=9,
        num_predict=32,
        max_image_side=32,
    )
    frame = np.zeros((96, 64, 3), dtype=np.uint8)

    with patch("reachy_mini_conversation_app.vision.analyzers.urlopen", fake_urlopen):
        result = analyzer._analyze_sync(frame, "What is on the desk?")

    assert result == "A keyboard is visible."
    assert captured["url"] == "http://llama.test/v1/chat/completions"
    assert captured["timeout"] == 9
    payload = captured["payload"]
    assert payload["model"] == "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF"
    assert payload["stream"] is False
    assert payload["max_tokens"] == 32
    assert "tools" not in payload
    assert "format" not in payload
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert "What is on the desk?" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_default_vision_analyzer_uses_llama_cpp_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local backend should use the fast OpenAI-compatible vision path."""
    monkeypatch.setattr(config, "LOCAL_VISION_BASE_URL", "http://127.0.0.1:8081/v1")
    monkeypatch.setattr(config, "LOCAL_VISION_SERVER_MODEL", "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF")

    analyzer = build_default_vision_analyzer()

    assert isinstance(analyzer, OpenAICompatibleVisionAnalyzer)
