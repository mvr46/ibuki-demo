"""Conversation backend factory."""

from __future__ import annotations
import logging

from reachy_mini_conversation_app.runtime.config import (
    HF_BACKEND,
    LOCAL_BACKEND,
    GEMINI_BACKEND,
    OPENAI_BACKEND,
    HF_LOCAL_CONNECTION_MODE,
    config,
    is_gemini_model,
    get_backend_label,
    get_hf_connection_selection,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.backends.interface import ConversationHandler


logger = logging.getLogger(__name__)


def create_conversation_handler(
    deps: ToolDependencies,
    *,
    instance_path: str | None,
    startup_voice: str | None,
) -> ConversationHandler:
    """Create the configured conversation handler Adapter."""
    if config.BACKEND_PROVIDER == LOCAL_BACKEND:
        from reachy_mini_conversation_app.backends.local_conversation import LocalConversationHandler

        logger.info("Using %s via LocalConversationHandler", get_backend_label(config.BACKEND_PROVIDER))
        return LocalConversationHandler(
            deps,
            instance_path=instance_path,
            startup_voice=startup_voice,
        )

    if config.BACKEND_PROVIDER == HF_BACKEND:
        from reachy_mini_conversation_app.backends.huggingface_realtime import HuggingFaceRealtimeHandler

        hf_connection_selection = get_hf_connection_selection()
        transport_label = (
            "Hugging Face direct websocket"
            if hf_connection_selection.mode == HF_LOCAL_CONNECTION_MODE and hf_connection_selection.has_target
            else "Hugging Face session proxy"
        )
        logger.info(
            "Using %s via Hugging Face realtime handler (%s)",
            get_backend_label(config.BACKEND_PROVIDER),
            transport_label,
        )
        return HuggingFaceRealtimeHandler(
            deps,
            instance_path=instance_path,
            startup_voice=startup_voice,
        )

    if config.BACKEND_PROVIDER == GEMINI_BACKEND or is_gemini_model():
        from reachy_mini_conversation_app.backends.gemini_live import GeminiLiveHandler

        logger.warning("Gemini backend is legacy-only; prefer local or Hugging Face for production.")
        return GeminiLiveHandler(
            deps,
            instance_path=instance_path,
            startup_voice=startup_voice,
        )

    if config.BACKEND_PROVIDER == OPENAI_BACKEND:
        from reachy_mini_conversation_app.backends.openai_realtime import OpenaiRealtimeHandler

        logger.warning("OpenAI backend is legacy-only; prefer local or Hugging Face for production.")
        return OpenaiRealtimeHandler(
            deps,
            instance_path=instance_path,
            startup_voice=startup_voice,
        )

    raise RuntimeError(f"Unsupported backend provider: {config.BACKEND_PROVIDER!r}")
