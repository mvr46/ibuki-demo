"""Tests for the local-first conversation handler."""

from __future__ import annotations
import asyncio
import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.runtime.streaming import AdditionalOutputs
from reachy_mini_conversation_app.backends.local_llm import LocalLLMResponse, LocalToolRoutingResult
from reachy_mini_conversation_app.backends.local_conversation import LocalConversationHandler, _normalize_spoken_text


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
        self.user_texts: list[str] = []

    async def route(self, user_text: str, tools: list[dict[str, object]]) -> LocalToolRoutingResult:
        self.user_texts.append(user_text)
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


class _StreamingLLM:
    def __init__(self) -> None:
        self.chat_calls = 0
        self.stream_calls = 0
        self.first_chunk_spoken = asyncio.Event()
        self.allow_finish = asyncio.Event()

    async def chat(self, messages: list[dict[str, object]], tools: list[dict[str, object]]) -> LocalLLMResponse:
        self.chat_calls += 1
        return LocalLLMResponse(content="non-stream fallback", tool_calls=[])

    async def stream_chat(self, messages: list[dict[str, object]], tools: list[dict[str, object]]):
        self.stream_calls += 1
        yield "Hello there. "
        self.first_chunk_spoken.set()
        await self.allow_finish.wait()
        yield "This is the rest."


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
async def test_local_conversation_logs_stt_and_tts_phrases(caplog: pytest.LogCaptureFixture) -> None:
    """Local voice turns should log incoming STT and outgoing TTS phrases."""
    caplog.set_level(logging.INFO, logger="reachy_mini_conversation_app.backends.local_conversation")
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        performance_diagnostics=MagicMock(),
    )
    handler = LocalConversationHandler(
        deps,
        stt_adapter=_FakeSTT(),
        llm_adapter=_FakeLLM(),
        tts_adapter=_FakeTTS(),
    )

    await handler._process_turn(np.ones(1600, dtype=np.int16))

    log_messages = [record.getMessage() for record in caplog.records]
    assert any("STT start" in message for message in log_messages)
    assert any("STT transcript" in message and "hello reachy" in message for message in log_messages)
    assert any("TTS synth start" in message and "hello human" in message for message in log_messages)
    assert any("TTS synth done" in message for message in log_messages)


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
async def test_local_conversation_falls_back_to_llm_when_qwen_selects_no_tool() -> None:
    """Action-like utterances should not use deterministic shortcuts when Qwen returns no tool."""
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        performance_diagnostics=MagicMock(),
    )
    llm = _FakeLLM()
    router = _FakeRouter([])
    handler = LocalConversationHandler(
        deps,
        stt_adapter=_FakeSTT(),
        llm_adapter=llm,
        tts_adapter=_FakeTTS(),
        tool_router=router,
    )
    handler._messages.append({"role": "user", "content": "move your head left"})

    await handler._respond_to_current_messages()

    assistant_output = await handler.output_queue.get()
    audio_output = await handler.output_queue.get()

    assert router.user_texts == ["move your head left"]
    assert router.tool_counts and router.tool_counts[-1] > 0
    assert llm.tool_counts == [0]
    assert isinstance(assistant_output, AdditionalOutputs)
    assert assistant_output.args[0]["content"] == "hello human"
    assert isinstance(audio_output, tuple)


@pytest.mark.asyncio
async def test_local_conversation_streams_gemma_and_queues_first_tts_chunk_before_done() -> None:
    """No-tool turns should stream Gemma and queue audio before the full response finishes."""
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        performance_diagnostics=MagicMock(),
    )
    llm = _StreamingLLM()
    tts = _CapturingTTS()
    router = _FakeRouter([])
    handler = LocalConversationHandler(
        deps,
        stt_adapter=_FakeSTT(),
        llm_adapter=llm,
        tts_adapter=tts,
        tool_router=router,
    )
    handler._messages.append({"role": "user", "content": "move your head left"})

    task = asyncio.create_task(handler._respond_to_current_messages())
    await asyncio.wait_for(llm.first_chunk_spoken.wait(), timeout=1)

    first_output = await asyncio.wait_for(handler.output_queue.get(), timeout=1)
    assert isinstance(first_output, tuple)
    assert tts.texts == ["Hello there."]
    assert llm.chat_calls == 0
    assert llm.stream_calls == 1
    assert router.user_texts == ["move your head left"]
    assert not task.done()

    llm.allow_finish.set()
    await task

    remaining = []
    while not handler.output_queue.empty():
        remaining.append(handler.output_queue.get_nowait())
    assistant_outputs = [item for item in remaining if isinstance(item, AdditionalOutputs)]
    assert assistant_outputs[-1].args[0]["content"] == "Hello there. This is the rest."


@pytest.mark.asyncio
async def test_local_barge_in_clears_audio_and_stops_stale_stream_chunks() -> None:
    """User speech start should clear queued assistant audio and cancel stale generation."""
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        head_wobbler=MagicMock(),
    )
    llm = _StreamingLLM()
    tts = _CapturingTTS()
    handler = LocalConversationHandler(
        deps,
        stt_adapter=_FakeSTT(),
        llm_adapter=llm,
        tts_adapter=tts,
        tool_router=_FakeRouter([]),
    )
    handler._messages.append({"role": "user", "content": "tell me something"})

    task = asyncio.create_task(handler._respond_to_current_messages())
    handler._processing_task = task
    await asyncio.wait_for(llm.first_chunk_spoken.wait(), timeout=1)
    assert not handler.output_queue.empty()

    handler._handle_user_speech_started()
    llm.allow_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert handler.output_queue.empty()
    assert tts.texts == ["Hello there."]
    deps.head_wobbler.reset.assert_called_once()


@pytest.mark.asyncio
async def test_robot_activity_delays_barge_in_clear_until_stt_confirms_speech() -> None:
    """Playback echo should not flush queued audio unless STT confirms real speech."""
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        head_wobbler=MagicMock(),
    )
    handler = LocalConversationHandler(
        deps,
        stt_adapter=_FakeSTT(),
        llm_adapter=_FakeLLM(),
        tts_adapter=_FakeTTS(),
        tool_router=_FakeRouter([]),
    )
    handler.output_queue.put_nowait((24000, np.ones((1, 2400), dtype=np.int16)))
    clear_queue = MagicMock(side_effect=handler._drain_output_queue)
    handler._clear_queue = clear_queue

    handler._handle_user_speech_started(robot_activity=True)

    clear_queue.assert_not_called()
    assert not handler.output_queue.empty()

    await handler._process_turn(
        np.ones(1600, dtype=np.int16),
        robot_activity=True,
        speech_ratio=0.9,
        avg_snr_db=20.0,
    )

    clear_queue.assert_called_once()
    deps.head_wobbler.reset.assert_called_once()
    user_output = await handler.output_queue.get()
    assert isinstance(user_output, AdditionalOutputs)
    assert user_output.args[0]["content"] == "hello reachy"


@pytest.mark.asyncio
async def test_robot_activity_rejects_low_confidence_stt_barge_in() -> None:
    """Robot playback transcriptions should not become fake user turns when confidence is weak."""
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        head_wobbler=MagicMock(),
        performance_diagnostics=MagicMock(),
    )
    handler = LocalConversationHandler(
        deps,
        stt_adapter=_FakeSTT(),
        llm_adapter=_FakeLLM(),
        tts_adapter=_FakeTTS(),
        tool_router=_FakeRouter([]),
    )
    handler.output_queue.put_nowait((24000, np.ones((1, 2400), dtype=np.int16)))
    clear_queue = MagicMock(side_effect=handler._drain_output_queue)
    handler._clear_queue = clear_queue

    await handler._process_turn(
        np.ones(1600, dtype=np.int16),
        robot_activity=True,
        speech_ratio=0.14,
        avg_snr_db=-1.3,
    )

    clear_queue.assert_not_called()
    deps.head_wobbler.reset.assert_not_called()
    assert handler.output_queue.qsize() == 1
    deps.performance_diagnostics.record_rejected_segment.assert_called()


@pytest.mark.asyncio
async def test_local_conversation_speaks_fallback_for_empty_llm_response() -> None:
    """Empty local LLM responses should still produce useful speech."""
    class EmptyLLM:
        async def chat(self, messages: list[dict[str, object]], tools: list[dict[str, object]]) -> LocalLLMResponse:
            return LocalLLMResponse(content="", tool_calls=[])

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
    )
    tts = _CapturingTTS()
    handler = LocalConversationHandler(
        deps,
        stt_adapter=_FakeSTT(),
        llm_adapter=EmptyLLM(),
        tts_adapter=tts,
    )
    handler._messages.append({"role": "user", "content": "say something"})

    await handler._respond_to_current_messages()

    assistant_output = await handler.output_queue.get()

    assert isinstance(assistant_output, AdditionalOutputs)
    assert assistant_output.args[0]["content"] == "I heard you, but my local model came back empty. Try that once more?"
    assert tts.texts == ["I heard you, but my local model came back empty. Try that once more?"]


@pytest.mark.asyncio
async def test_local_conversation_strips_markdown_bullets_before_tts() -> None:
    """Piper should not receive Markdown bullets that it reads aloud as punctuation names."""
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
    )
    tts = _CapturingTTS()
    handler = LocalConversationHandler(
        deps,
        stt_adapter=_FakeSTT(),
        llm_adapter=_FakeLLM(),
        tts_adapter=tts,
    )

    await handler._speak_response(
        "Here's what I see in the image: * A young man with dark hair. * A white shirt. * A bookshelf with books."
    )

    assistant_output = await handler.output_queue.get()

    assert isinstance(assistant_output, AdditionalOutputs)
    expected = "Here's what I see in the image: A young man with dark hair. A white shirt. A bookshelf with books."
    assert assistant_output.args[0]["content"] == expected
    assert tts.texts == [expected]
    assert "*" not in tts.texts[0]


@pytest.mark.asyncio
async def test_local_conversation_uses_useful_informational_tool_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Informational tools should not be reduced to generic acknowledgements."""
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
    results = {
        "who_am_i": {"status": "identified", "name": "Matteo"},
        "who_is_here": {"people": [{"name": "Alice"}, {"name": None}]},
        "camera": {"image_description": "I see a desk and a keyboard."},
    }

    async def fake_dispatch(tool_name: str, args_json: str, deps: object) -> dict[str, object]:
        return results[tool_name]

    monkeypatch.setattr("reachy_mini_conversation_app.backends.local_conversation.dispatch_tool_call", fake_dispatch)

    assert await handler._execute_routed_tool_call({"name": "who_am_i", "arguments": {}}) == "You look like Matteo."
    assert (
        await handler._execute_routed_tool_call({"name": "who_is_here", "arguments": {}})
        == "I see Alice and 1 unknown person."
    )
    assert (
        await handler._execute_routed_tool_call({"name": "camera", "arguments": {}})
        == "I see a desk and a keyboard."
    )


@pytest.mark.asyncio
async def test_local_conversation_dispatches_task_status_with_background_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local system tools should receive the handler's BackgroundToolManager."""
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
    captured = {}

    async def fake_dispatch_with_manager(
        tool_name: str,
        args_json: str,
        deps_arg: object,
        tool_manager: object,
    ) -> dict[str, object]:
        captured["tool_name"] = tool_name
        captured["deps"] = deps_arg
        captured["manager"] = tool_manager
        return {"status": "idle", "message": "No tools running in the background."}

    monkeypatch.setattr(
        "reachy_mini_conversation_app.backends.local_conversation.dispatch_tool_call_with_manager",
        fake_dispatch_with_manager,
    )

    spoken = await handler._execute_routed_tool_call({"name": "task_status", "arguments": {}})

    assert captured == {"tool_name": "task_status", "deps": deps, "manager": handler.tool_manager}
    assert spoken == "No tools running in the background."


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
async def test_environment_messages_are_ignored_in_local_chat_history() -> None:
    """Passive environment updates should not enter the latency-critical local LLM context."""
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
    before = list(handler._messages)

    await handler.inject_environment_message("Alice entered the frame.")

    assert handler._messages == before


def test_normalize_spoken_text_drops_emoji_only_tail() -> None:
    """Emoji-only chunks should not burn a TTS call or be spoken aloud."""
    assert _normalize_spoken_text("Hello there. \U0001f60a") == "Hello there."
    assert _normalize_spoken_text("\U0001f60a") == ""
