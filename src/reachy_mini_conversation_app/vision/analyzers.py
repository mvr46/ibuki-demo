"""Vision analyzer adapters for realtime, SmolVLM, and llama.cpp image answers."""

from __future__ import annotations
import json
import base64
import asyncio
import logging
from typing import Any, Protocol
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.runtime.config import config
from reachy_mini_conversation_app.vision.camera_frame_encoding import encode_bgr_frame_as_jpeg


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


class OpenAICompatibleVisionAnalyzer:
    """Vision adapter using a local OpenAI-compatible chat completions server."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 30.0,
        num_predict: int | None = None,
        max_image_side: int | None = None,
        diagnostics: Any | None = None,
    ) -> None:
        """Initialize the OpenAI-compatible vision target."""
        self.base_url = (base_url or config.LOCAL_VISION_BASE_URL).rstrip("/")
        self.model = model or config.LOCAL_VISION_SERVER_MODEL
        self.timeout_seconds = timeout_seconds
        self.num_predict = int(num_predict if num_predict is not None else config.LOCAL_VISION_NUM_PREDICT)
        self.max_image_side = int(
            max_image_side if max_image_side is not None else config.LOCAL_VISION_MAX_IMAGE_SIDE
        )
        self.diagnostics = diagnostics

    async def warm(self) -> None:
        """Prime the local vision server with a tiny image request."""
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        await self.analyze(frame, "Say ready.")

    async def analyze(self, frame: NDArray[np.uint8], question: str) -> dict[str, Any]:
        """Analyze a frame using a local OpenAI-compatible server."""
        try:
            answer = await asyncio.to_thread(self._analyze_sync, frame, question)
        except Exception as exc:
            logger.warning("Local OpenAI-compatible vision analysis failed: %s", exc)
            _record_local_model(
                self.diagnostics,
                vision_model=self.model,
                vision_provider="openai_compatible",
                last_vision_status="error",
                last_vision_error=str(exc),
            )
            return {"error": f"Local vision analysis failed: {exc}"}
        _record_local_model(
            self.diagnostics,
            vision_model=self.model,
            vision_provider="openai_compatible",
            last_vision_status="ok",
            last_vision_error=None,
        )
        return {"image_description": answer}

    def _analyze_sync(self, frame: NDArray[np.uint8], question: str) -> str:
        jpeg_bytes = encode_bgr_frame_as_jpeg(_downsample_frame(frame, self.max_image_side))
        image_data_url = f"data:image/jpeg;base64,{base64.b64encode(jpeg_bytes).decode('utf-8')}"
        payload = {
            "model": self.model,
            "stream": False,
            "temperature": 0.2,
            "max_tokens": self.num_predict,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _vision_prompt(question)},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ],
                }
            ],
        }
        req = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        return _openai_answer_text(data) or "I can't tell from this image."


def build_default_vision_analyzer(
    vision_processor: Any | None = None,
    diagnostics: Any | None = None,
) -> VisionAnalyzer:
    """Return the preferred local-first vision analyzer."""
    if vision_processor is not None:
        return SmolVLMVisionAnalyzer(vision_processor)
    return OpenAICompatibleVisionAnalyzer(diagnostics=diagnostics)


def _record_local_model(diagnostics: Any | None, **payload: object) -> None:
    """Best-effort local model diagnostics update."""
    set_local_model = getattr(diagnostics, "set_local_model", None)
    if callable(set_local_model):
        set_local_model(**payload)


def _downsample_frame(frame: NDArray[np.uint8], max_image_side: int) -> NDArray[np.uint8]:
    """Cheaply shrink large camera frames before local VLM upload."""
    if max_image_side <= 0:
        return frame
    height, width = frame.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_image_side:
        return frame
    step = max(1, (longest_side + max_image_side - 1) // max_image_side)
    return frame[::step, ::step]


def _vision_prompt(question: str) -> str:
    """Return a compact prompt for one-frame camera questions."""
    cleaned = (question or "").strip() or "What is visible?"
    return f"{cleaned}\n\nAnswer only from visible image evidence. Keep the answer under 25 words."


def _openai_answer_text(data: object) -> str:
    """Extract assistant content from an OpenAI-compatible chat response."""
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""
    return _content_text(message.get("content")).strip()


def _content_text(content: object) -> str:
    """Return text from string or OpenAI content-part payloads."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "".join(parts)
