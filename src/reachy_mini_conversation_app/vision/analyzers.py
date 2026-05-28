"""Vision analyzer adapters for realtime, SmolVLM, and Ollama image answers."""

from __future__ import annotations
import json
import base64
import asyncio
import logging
from typing import Any, Protocol
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.camera_frame_encoding import encode_bgr_frame_as_jpeg


logger = logging.getLogger(__name__)


class VisionAnalyzer(Protocol):
    """Interface for camera question answering adapters."""

    async def analyze(self, frame: NDArray[np.uint8], question: str) -> dict[str, Any]:
        """Analyze one BGR image frame and return a tool result payload."""
        ...


class RealtimeVisionAnalyzer:
    """Return an encoded image for the active realtime backend to analyze."""

    async def analyze(self, frame: NDArray[np.uint8], question: str) -> dict[str, Any]:
        """Return a base64 JPEG payload for realtime backends."""
        jpeg_bytes = encode_bgr_frame_as_jpeg(frame)
        return {"b64_im": base64.b64encode(jpeg_bytes).decode("utf-8")}


class SmolVLMVisionAnalyzer:
    """Adapter around the existing local SmolVLM VisionProcessor."""

    def __init__(self, vision_processor: Any) -> None:
        """Initialize with an existing processor."""
        self.vision_processor = vision_processor

    async def analyze(self, frame: NDArray[np.uint8], question: str) -> dict[str, Any]:
        """Analyze a frame using the local vision processor."""
        result = await asyncio.to_thread(self.vision_processor.process_image, frame, question)
        return {"image_description": result} if isinstance(result, str) else {"error": "vision returned non-string"}


class OllamaVisionAnalyzer:
    """Vision adapter using Ollama's local chat API with image input."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        """Initialize the Ollama target."""
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout_seconds = timeout_seconds

    async def analyze(self, frame: NDArray[np.uint8], question: str) -> dict[str, Any]:
        """Analyze a frame using Ollama."""
        try:
            answer = await asyncio.to_thread(self._analyze_sync, frame, question)
        except Exception as exc:
            logger.warning("Ollama vision analysis failed: %s", exc)
            return {"error": f"Ollama vision analysis failed: {exc}"}
        return {"image_description": answer}

    def _analyze_sync(self, frame: NDArray[np.uint8], question: str) -> str:
        jpeg_bytes = encode_bgr_frame_as_jpeg(frame)
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{question}\n\n"
                        "Answer only from visible image evidence. If uncertain, say you cannot tell."
                    ),
                    "images": [base64.b64encode(jpeg_bytes).decode("utf-8")],
                }
            ],
        }
        req = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        return str(content or "").strip() or "I can't tell from this image."


def build_default_vision_analyzer(vision_processor: Any | None = None) -> VisionAnalyzer:
    """Return the preferred local-first vision analyzer."""
    if vision_processor is not None:
        return SmolVLMVisionAnalyzer(vision_processor)
    return OllamaVisionAnalyzer()

