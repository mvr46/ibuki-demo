"""Local-first conversation handler for Mac-hosted STT, Ollama, and TTS."""

from __future__ import annotations
import json
import time
import asyncio
import logging
from typing import Any, Tuple, Optional

import numpy as np
from fastrtc import AdditionalOutputs, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.config import LOCAL_BACKEND, get_default_voice_for_backend
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.local_llm import (
    LocalLLMAdapter,
    OllamaLLMAdapter,
    ollama_tool_call_messages,
)
from reachy_mini_conversation_app.local_stt import LocalSTTAdapter, MLXWhisperSTTAdapter
from reachy_mini_conversation_app.local_tts import LocalTTSAdapter, PiperTTSAdapter
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    dispatch_tool_call,
    get_active_tool_specs,
)
from reachy_mini_conversation_app.conversation_handler import ConversationHandler


logger = logging.getLogger(__name__)

LOCAL_INPUT_SAMPLE_RATE = 16000
LOCAL_OUTPUT_SAMPLE_RATE = 24000


class LocalConversationHandler(ConversationHandler):
    """Turn-based local handler using robot media but Mac-side AI compute."""

    BACKEND_PROVIDER = LOCAL_BACKEND

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
        *,
        stt_adapter: LocalSTTAdapter | None = None,
        llm_adapter: LocalLLMAdapter | None = None,
        tts_adapter: LocalTTSAdapter | None = None,
        silence_seconds: float = 0.85,
        min_speech_seconds: float = 0.35,
        speech_rms_threshold: float = 420.0,
    ) -> None:
        """Initialize the local turn-based conversation pipeline."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=LOCAL_OUTPUT_SAMPLE_RATE,
            input_sample_rate=LOCAL_INPUT_SAMPLE_RATE,
        )
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        self._voice_override = startup_voice
        self.stt_adapter = stt_adapter or MLXWhisperSTTAdapter()
        self.llm_adapter = llm_adapter or OllamaLLMAdapter()
        self.tts_adapter = tts_adapter or PiperTTSAdapter()
        self.output_queue: asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue()
        self._clear_queue = None
        self._stop_event = asyncio.Event()
        self._processing_task: asyncio.Task[None] | None = None
        self._audio_chunks: list[NDArray[np.int16]] = []
        self._speech_started_at: float | None = None
        self._last_voice_at: float | None = None
        self._silence_seconds = silence_seconds
        self._min_speech_seconds = min_speech_seconds
        self._speech_rms_threshold = speech_rms_threshold
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": get_session_instructions()},
        ]

    def copy(self) -> "LocalConversationHandler":
        """Create a copy of the handler."""
        return type(self)(
            self.deps,
            self.gradio_mode,
            self.instance_path,
            self._voice_override,
            stt_adapter=self.stt_adapter,
            llm_adapter=self.llm_adapter,
            tts_adapter=self.tts_adapter,
            silence_seconds=self._silence_seconds,
            min_speech_seconds=self._min_speech_seconds,
            speech_rms_threshold=self._speech_rms_threshold,
        )

    async def start_up(self) -> None:
        """Prepare local conversation state."""
        logger.info("Local conversation backend ready.")

    async def shutdown(self) -> None:
        """Stop local processing."""
        self._stop_event.set()
        if self._processing_task is not None and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Buffer incoming microphone audio and process complete utterances."""
        input_sample_rate, audio_frame = frame
        audio = _prepare_audio_frame(audio_frame, input_sample_rate, LOCAL_INPUT_SAMPLE_RATE)
        if audio.size == 0:
            return

        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        now = time.monotonic()
        if rms >= self._speech_rms_threshold:
            if self._speech_started_at is None:
                self._speech_started_at = now
                self.deps.movement_manager.set_listening(True)
                self._notify("notify_user_speech_started")
            self._last_voice_at = now
            self._audio_chunks.append(audio)
            return

        if self._speech_started_at is not None:
            self._audio_chunks.append(audio)
            last_voice_at = self._last_voice_at or self._speech_started_at
            if now - last_voice_at >= self._silence_seconds:
                duration = now - self._speech_started_at
                chunks = self._audio_chunks
                self._audio_chunks = []
                self._speech_started_at = None
                self._last_voice_at = None
                self.deps.movement_manager.set_listening(False)
                self._notify("notify_user_speech_stopped")
                if duration >= self._min_speech_seconds and self._processing_task is None:
                    audio_turn = np.concatenate(chunks)
                    self._processing_task = asyncio.create_task(self._process_turn(audio_turn), name="local-turn")

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit the next local output item."""
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a personality for subsequent local turns."""
        from reachy_mini_conversation_app.config import set_custom_profile

        set_custom_profile(profile)
        self._messages = [{"role": "system", "content": get_session_instructions()}]
        return "Applied personality to local session."

    async def get_available_voices(self) -> list[str]:
        """Return local voices."""
        return [get_default_voice_for_backend(LOCAL_BACKEND)]

    def get_current_voice(self) -> str:
        """Return current local voice label."""
        return self._voice_override or get_session_voice(get_default_voice_for_backend(LOCAL_BACKEND))

    async def change_voice(self, voice: str) -> str:
        """Store a local voice label; Piper uses PIPER_VOICE env for actual model path."""
        self._voice_override = voice.strip() or None
        return "Local voice changed. Set PIPER_VOICE to change the Piper model file."

    async def inject_environment_message(self, text: str, *, trigger_response: bool = False) -> None:
        """Inject context into local chat history."""
        self._messages.append({"role": "user", "content": text})
        if trigger_response:
            self._processing_task = asyncio.create_task(self._respond_to_current_messages(), name="local-env-response")

    async def _process_turn(self, audio: NDArray[np.int16]) -> None:
        started = time.perf_counter()
        diagnostics = getattr(self.deps, "performance_diagnostics", None)
        try:
            stt_start = time.perf_counter()
            transcript = await self.stt_adapter.transcribe(audio, LOCAL_INPUT_SAMPLE_RATE)
            stt_ms = (time.perf_counter() - stt_start) * 1000
            _record_turn_metrics(diagnostics, stt_ms=stt_ms)
            if not transcript:
                return
            self._notify("notify_user_transcript", transcript)
            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
            self._messages.append({"role": "user", "content": transcript})
            await self._respond_to_current_messages(turn_started=started)
        finally:
            self._processing_task = None

    async def _respond_to_current_messages(self, *, turn_started: float | None = None) -> None:
        tools = get_active_tool_specs(self.deps)
        llm_start = time.perf_counter()
        response_text = ""
        for _ in range(4):
            try:
                response = await self.llm_adapter.chat(self._messages, tools)
            except Exception:
                logger.exception("Local LLM failed while responding.")
                response_text = "I had trouble with my local language model just now."
                break
            response_text = response.content
            if not response.tool_calls:
                break
            self._messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": ollama_tool_call_messages(response.tool_calls),
                }
            )
            for tool_call in response.tool_calls:
                name = str(tool_call.get("name") or "")
                arguments = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
                result = await dispatch_tool_call(name, json.dumps(arguments), self.deps)
                await self.output_queue.put(
                    AdditionalOutputs(
                        {
                            "role": "assistant",
                            "content": json.dumps(result),
                            "metadata": {"title": f"Used tool {name}", "status": "done"},
                        }
                    )
                )
                self._messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})

        llm_total_ms = (time.perf_counter() - llm_start) * 1000
        diagnostics = getattr(self.deps, "performance_diagnostics", None)
        _record_turn_metrics(diagnostics, llm_first_token_ms=llm_total_ms, llm_total_ms=llm_total_ms)
        if not response_text:
            return

        self._messages.append({"role": "assistant", "content": response_text})
        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": response_text}))
        tts_start = time.perf_counter()
        sample_rate, audio = await self.tts_adapter.synthesize(response_text)
        tts_ms = (time.perf_counter() - tts_start) * 1000
        first_audio_ms = (time.perf_counter() - turn_started) * 1000 if turn_started is not None else None
        _record_turn_metrics(diagnostics, tts_ms=tts_ms, first_audio_ms=first_audio_ms)
        if audio.size:
            self._notify("notify_assistant_audio_started")
            await self.output_queue.put((sample_rate, audio.reshape(1, -1)))
            self._notify("notify_assistant_audio_done")

    def _notify(self, method_name: str, *args: Any) -> None:
        """Notify camera and speaker attribution helpers."""
        camera_worker = getattr(self.deps, "camera_worker", None)
        camera_method = getattr(camera_worker, method_name, None)
        if callable(camera_method):
            camera_method()

        speaker_worker = getattr(self.deps, "speaker_attribution_worker", None)
        speaker_method = getattr(speaker_worker, method_name, None)
        if callable(speaker_method):
            speaker_method(*args)


def _prepare_audio_frame(
    audio_frame: NDArray[np.int16],
    input_sample_rate: int,
    output_sample_rate: int,
) -> NDArray[np.int16]:
    """Normalize an audio frame to mono int16 at the requested sample rate."""
    audio = audio_frame
    if audio.ndim == 2:
        if audio.shape[1] > audio.shape[0]:
            audio = audio.T
        if audio.shape[1] > 1:
            audio = audio[:, 0]
    if input_sample_rate != output_sample_rate:
        target_len = int(len(audio) * output_sample_rate / input_sample_rate)
        if target_len <= 0:
            return np.zeros(0, dtype=np.int16)
        audio = resample(audio, target_len)
    return audio_to_int16(audio)


def _record_turn_metrics(
    diagnostics: object | None,
    *,
    stt_ms: float | None = None,
    llm_first_token_ms: float | None = None,
    llm_total_ms: float | None = None,
    tts_ms: float | None = None,
    first_audio_ms: float | None = None,
) -> None:
    """Record turn metrics if diagnostics are available."""
    record_metrics = getattr(diagnostics, "record_turn_metrics", None)
    if callable(record_metrics):
        record_metrics(
            stt_ms=stt_ms,
            llm_first_token_ms=llm_first_token_ms,
            llm_total_ms=llm_total_ms,
            tts_ms=tts_ms,
            first_audio_ms=first_audio_ms,
        )
