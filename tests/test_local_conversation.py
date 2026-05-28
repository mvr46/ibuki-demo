"""Tests for the local-first conversation handler."""

from __future__ import annotations
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastrtc import AdditionalOutputs

from reachy_mini_conversation_app.local_llm import LocalLLMResponse
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.local_conversation import LocalConversationHandler


class _FakeSTT:
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        return "hello reachy"


class _FakeLLM:
    async def chat(self, messages: list[dict[str, object]], tools: list[dict[str, object]]) -> LocalLLMResponse:
        return LocalLLMResponse(content="hello human", tool_calls=[])


class _FakeTTS:
    output_sample_rate = 24000

    async def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        return 24000, np.ones(2400, dtype=np.int16)


@pytest.mark.asyncio
async def test_local_conversation_processes_one_audio_turn() -> None:
    """Local handler should segment speech, transcribe, answer, and synthesize."""
    diagnostics = MagicMock()
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        performance_diagnostics=diagnostics,
    )
    handler = LocalConversationHandler(
        deps,
        stt_adapter=_FakeSTT(),
        llm_adapter=_FakeLLM(),
        tts_adapter=_FakeTTS(),
        silence_seconds=0.01,
        min_speech_seconds=0.0,
        speech_rms_threshold=100.0,
    )

    speech = np.full(3200, 1000, dtype=np.int16)
    await handler._process_turn(speech)

    user_output = await handler.output_queue.get()
    assistant_output = await handler.output_queue.get()
    audio_output = await handler.output_queue.get()

    assert isinstance(user_output, AdditionalOutputs)
    assert user_output.args[0]["content"] == "hello reachy"
    assert isinstance(assistant_output, AdditionalOutputs)
    assert assistant_output.args[0]["content"] == "hello human"
    assert isinstance(audio_output, tuple)
    assert audio_output[0] == 24000
    assert audio_output[1].shape == (1, 2400)
    diagnostics.record_turn_metrics.assert_called()
