"""Tests for the local-first conversation handler."""

from __future__ import annotations
from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.runtime.streaming import AdditionalOutputs
from reachy_mini_conversation_app.backends.local_llm import LocalLLMResponse, LocalToolRoutingResult
from reachy_mini_conversation_app.backends.local_conversation import LocalConversationHandler


class _FakeSTT:
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        return "hello reachy"


class _FakeLLM:
    def __init__(self) -> None:
        self.tool_counts: list[int] = []

    async def chat(self, messages: list[dict[str, object]], tools: list[dict[str, object]]) -> LocalLLMResponse:
        self.tool_counts.append(len(tools))
        return LocalLLMResponse(content="hello human", tool_calls=[])


class _FakeRouter:
    def __init__(self, tool_calls: list[dict[str, object]] | None = None) -> None:
        self.tool_calls = tool_calls or []
        self.tool_counts: list[int] = []

    async def route(self, user_text: str, tools: list[dict[str, object]]) -> LocalToolRoutingResult:
        self.tool_counts.append(len(tools))
        return LocalToolRoutingResult(tool_calls=self.tool_calls)


class _FakeTTS:
    output_sample_rate = 24000

    async def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        return 24000, np.ones(2400, dtype=np.int16)


class _CapturingTTS:
    output_sample_rate = 24000

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        self.texts.append(text)
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
        llm_adapter=(llm := _FakeLLM()),
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
    assert llm.tool_counts == [0]
    diagnostics.record_turn_metrics.assert_called()


@pytest.mark.asyncio
async def test_local_conversation_attaches_tools_for_robot_action_turn() -> None:
    """Local handler should route robot actions through the compact router only."""
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        performance_diagnostics=MagicMock(),
    )
    llm = _FakeLLM()
    router = _FakeRouter([{"name": "look_at_person", "arguments": {"name": "Matt"}}])
    handler = LocalConversationHandler(
        deps,
        stt_adapter=_FakeSTT(),
        llm_adapter=llm,
        tts_adapter=_FakeTTS(),
        tool_router=router,
    )
    handler._messages.append({"role": "user", "content": "look at Matt"})

    async def fake_dispatch(tool_name: str, args_json: str, deps: object) -> dict[str, object]:
        return {"status": "looking_at", "name": "Matt"}

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("reachy_mini_conversation_app.backends.local_conversation.dispatch_tool_call", fake_dispatch)
        await handler._respond_to_current_messages()

    tool_output = await handler.output_queue.get()
    assistant_output = await handler.output_queue.get()
    audio_output = await handler.output_queue.get()

    assert router.tool_counts and router.tool_counts[-1] > 0
    assert llm.tool_counts == []
    assert isinstance(tool_output, AdditionalOutputs)
    assert tool_output.args[0]["metadata"]["title"] == "Used tool look_at_person"
    assert isinstance(assistant_output, AdditionalOutputs)
    assert assistant_output.args[0]["content"] == "Okay, looking at Matt."
    assert isinstance(audio_output, tuple)


@pytest.mark.asyncio
async def test_rejected_stt_transcript_never_becomes_user_message() -> None:
    """STT-rejected noise text should not enter chat history or UI output."""
    diagnostics = MagicMock()

    class RejectingSTT:
        last_reject_reason = "repeated_character"

        async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
            return ""

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        performance_diagnostics=diagnostics,
    )
    handler = LocalConversationHandler(
        deps,
        stt_adapter=RejectingSTT(),
        llm_adapter=_FakeLLM(),
        tts_adapter=_FakeTTS(),
    )

    await handler._process_turn(np.ones(1600, dtype=np.int16))

    assert handler.output_queue.empty()
    assert not any(message.get("role") == "user" for message in handler._messages)
    diagnostics.record_rejected_segment.assert_called_with(reason="repeated_character", source="stt")


@pytest.mark.asyncio
async def test_environment_messages_are_not_stored_as_user_turns() -> None:
    """Ambient vision context should not look like user speech to routing/history."""
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
    )
    handler = LocalConversationHandler(
        deps,
        stt_adapter=_FakeSTT(),
        llm_adapter=_FakeLLM(),
        tts_adapter=_FakeTTS(),
    )

    await handler.inject_environment_message("[Vision: Alice entered the frame (center)]")

    assert handler._messages[-1] == {"role": "system", "content": "[Vision: Alice entered the frame (center)]"}
    assert not any(message["role"] == "user" for message in handler._messages)


@pytest.mark.asyncio
async def test_local_conversation_strips_echoed_ambient_context_before_speaking() -> None:
    """The robot should not read model-echoed Vision/Speech tags aloud."""

    class EchoingLLM:
        async def chat(self, messages: list[dict[str, object]], tools: list[dict[str, object]]) -> LocalLLMResponse:
            return LocalLLMResponse(
                content="[Vision: Alice entered the frame (center)] Hello Alice, nice to see you.",
                tool_calls=[],
            )

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
    )
    tts = _CapturingTTS()
    handler = LocalConversationHandler(
        deps,
        stt_adapter=_FakeSTT(),
        llm_adapter=EchoingLLM(),
        tts_adapter=tts,
    )
    handler._messages.append({"role": "system", "content": "[Vision: Alice entered the frame (center)]"})
    handler._messages.append({"role": "user", "content": "hi"})

    await handler._respond_to_current_messages()

    assistant_output = await handler.output_queue.get()
    audio_output = await handler.output_queue.get()

    assert isinstance(assistant_output, AdditionalOutputs)
    assert assistant_output.args[0]["content"] == "Hello Alice, nice to see you."
    assert tts.texts == ["Hello Alice, nice to see you."]
    assert handler._messages[-1]["content"] == "Hello Alice, nice to see you."
    assert isinstance(audio_output, tuple)
