"""Local-first conversation handler for Mac-hosted STT, LLM/router, and TTS."""

from __future__ import annotations
import re
import json
import time
import asyncio
import logging
from typing import Any, Tuple, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.runtime.config import LOCAL_BACKEND, config, get_default_voice_for_backend
from reachy_mini_conversation_app.profiles.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    dispatch_tool_call,
    get_active_tool_specs,
    dispatch_tool_call_with_manager,
)
from reachy_mini_conversation_app.runtime.streaming import AdditionalOutputs, wait_for_item, audio_to_int16
from reachy_mini_conversation_app.backends.interface import ConversationHandler
from reachy_mini_conversation_app.backends.local_llm import (
    LocalLLMAdapter,
    LocalToolRouter,
    create_local_llm_adapter,
    create_local_tool_router,
)
from reachy_mini_conversation_app.backends.local_stt import LocalSTTAdapter, MLXWhisperSTTAdapter
from reachy_mini_conversation_app.backends.local_tts import LocalTTSAdapter, PiperTTSAdapter
from reachy_mini_conversation_app.tools.tool_constants import SystemTool
from reachy_mini_conversation_app.backends.local_turn_detector import (
    LocalRejectedTurn,
    LocalTurnDetector,
    LocalCompletedTurn,
    LocalTurnDetectorConfig,
)
from reachy_mini_conversation_app.tools.background_tool_manager import (
    ToolNotification,
    BackgroundToolManager,
)


logger = logging.getLogger(__name__)

LOCAL_INPUT_SAMPLE_RATE = 16000
LOCAL_OUTPUT_SAMPLE_RATE = 24000


class LocalConversationHandler(ConversationHandler):
    """Turn-based local handler using robot media but Mac-side AI compute."""

    BACKEND_PROVIDER = LOCAL_BACKEND

    def __init__(
        self,
        deps: ToolDependencies,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
        *,
        stt_adapter: LocalSTTAdapter | None = None,
        llm_adapter: LocalLLMAdapter | None = None,
        tts_adapter: LocalTTSAdapter | None = None,
        tool_router: LocalToolRouter | None = None,
        turn_detector: LocalTurnDetector | None = None,
        silence_seconds: float | None = None,
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
        self.instance_path = instance_path
        self._voice_override = startup_voice
        resolved_silence_seconds = config.LOCAL_VAD_SILENCE_SECONDS if silence_seconds is None else silence_seconds
        self.stt_adapter = stt_adapter or MLXWhisperSTTAdapter()
        diagnostics = getattr(deps, "performance_diagnostics", None)
        self.llm_adapter = llm_adapter or create_local_llm_adapter(diagnostics=diagnostics)
        self.tool_router = tool_router or create_local_tool_router(diagnostics=diagnostics)
        self.tts_adapter = tts_adapter or PiperTTSAdapter()
        self.turn_detector = turn_detector or LocalTurnDetector(
            LocalTurnDetectorConfig(
                silence_seconds=resolved_silence_seconds,
                min_speech_seconds=min_speech_seconds,
                min_frame_rms=max(40.0, speech_rms_threshold * 0.35),
            )
        )
        self.output_queue: asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue()
        self._clear_queue = None
        self._stop_event = asyncio.Event()
        self._processing_task: asyncio.Task[None] | None = None
        self.tool_manager = BackgroundToolManager()
        self._generation = 0
        self._robot_noise_until = 0.0
        self._silence_seconds = resolved_silence_seconds
        self._min_speech_seconds = min_speech_seconds
        self._speech_rms_threshold = speech_rms_threshold
        self._max_messages = 40
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": get_session_instructions()},
        ]

    def copy(self) -> "LocalConversationHandler":
        """Create a copy of the handler."""
        return type(self)(
            self.deps,
            self.instance_path,
            self._voice_override,
            stt_adapter=self.stt_adapter,
            llm_adapter=self.llm_adapter,
            tts_adapter=self.tts_adapter,
            tool_router=self.tool_router,
            turn_detector=LocalTurnDetector(self.turn_detector.config),
            silence_seconds=self._silence_seconds,
            min_speech_seconds=self._min_speech_seconds,
            speech_rms_threshold=self._speech_rms_threshold,
        )

    async def start_up(self) -> None:
        """Prepare local conversation state."""
        logger.info("Local conversation backend ready.")
        self.tool_manager.start_up(tool_callbacks=[self._handle_tool_notification])
        asyncio.create_task(self._warm_local_components(), name="local-component-warmup")

    async def _warm_local_components(self) -> None:
        """Best-effort warmup for local latency-sensitive components."""
        for label, component in (
            ("MLX Whisper STT", self.stt_adapter),
            ("Local chat model", self.llm_adapter),
            ("Local router model", self.tool_router),
            ("Local vision model", self.deps.vision_analyzer),
            ("Piper voice", self.tts_adapter),
        ):
            if component is None:
                continue
            await self._warm_component(label, component)

    async def _warm_component(self, label: str, component: object) -> None:
        """Best-effort warmup for one local latency-sensitive component."""
        warm = getattr(component, "warm", None)
        if not callable(warm):
            return
        try:
            await warm()
            logger.info("Warmed %s.", label)
        except Exception as exc:
            logger.warning("Could not warm %s: %s", label, exc)

    def _schedule_component_warm(self, label: str, component: object) -> None:
        """Warm one component in the background without delaying first audio."""
        warm = getattr(component, "warm", None)
        if callable(warm) and not self._stop_event.is_set():
            asyncio.create_task(self._warm_component(label, component), name=f"local-warm-{label.casefold().replace(' ', '-')}")

    async def shutdown(self) -> None:
        """Stop local processing."""
        self._stop_event.set()
        self._generation += 1
        if self._processing_task is not None and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        await self.tool_manager.shutdown()
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

        now = time.monotonic()
        activity = self._robot_activity_state(now)
        update = self.turn_detector.process(audio, robot_activity=bool(activity["active"]))
        self._record_voice_activity(activity)

        if update.speech_started:
            self._handle_user_speech_started(robot_activity=bool(activity["active"]))

        for rejected_turn in update.rejected_turns:
            self._handle_rejected_turn(rejected_turn)

        if update.speech_stopped:
            self.deps.movement_manager.set_listening(False)
            self._notify("notify_user_speech_stopped")

        for completed_turn in update.completed_turns:
            self._handle_completed_turn(completed_turn)

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit the next local output item."""
        return await wait_for_item(self.output_queue)

    def _handle_user_speech_started(self, *, robot_activity: bool = False) -> None:
        """Prepare for barge-in by stopping queued assistant output."""
        if robot_activity:
            logger.debug("Speech-like audio started during robot activity; delaying barge-in until STT confirms speech.")
            return
        if self._is_turn_processing() and not self._assistant_output_active():
            logger.debug("Speech-like audio started while a local turn is still processing; preserving generation.")
            self.deps.movement_manager.set_listening(True)
            self._notify("notify_user_speech_started")
            return
        self._interrupt_assistant_output(cancel_processing=True)
        self.deps.movement_manager.set_listening(True)
        self._notify("notify_user_speech_started")

    def _is_turn_processing(self) -> bool:
        """Return whether a local turn is still producing a response."""
        return self._processing_task is not None and not self._processing_task.done()

    def _assistant_output_active(self) -> bool:
        """Return whether assistant audio is queued or likely still playing."""
        return time.monotonic() < self._robot_noise_until

    def _interrupt_assistant_output(self, *, cancel_processing: bool) -> None:
        """Clear assistant output and guard any stale generation."""
        self._generation += 1
        if callable(self._clear_queue):
            self._clear_queue()
        else:
            self._drain_output_queue()
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()
        if cancel_processing and self._processing_task is not None and not self._processing_task.done():
            self._processing_task.cancel()

    def _drain_output_queue(self) -> None:
        """Drop any locally queued audio/transcript output."""
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _is_current_generation(self, generation: int) -> bool:
        """Return whether work belongs to the current assistant generation."""
        return generation == self._generation and not self._stop_event.is_set()

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a profile for subsequent local turns."""
        from reachy_mini_conversation_app.runtime.config import set_custom_profile

        set_custom_profile(profile)
        self._messages = [{"role": "system", "content": get_session_instructions()}]
        return "Applied profile to local session."

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
        """Ignore passive environment context in the latency-critical local chat path."""
        logger.debug("Ignoring local environment message: %r", _truncate_for_log(text))

    async def _process_turn(
        self,
        audio: NDArray[np.int16],
        *,
        robot_activity: bool = False,
        speech_ratio: float | None = None,
        avg_snr_db: float | None = None,
    ) -> None:
        started = time.perf_counter()
        diagnostics = getattr(self.deps, "performance_diagnostics", None)
        try:
            stt_start = time.perf_counter()
            logger.info("STT start duration=%.2fs samples=%d", audio.size / LOCAL_INPUT_SAMPLE_RATE, audio.size)
            transcript = await self.stt_adapter.transcribe(audio, LOCAL_INPUT_SAMPLE_RATE)
            stt_ms = (time.perf_counter() - stt_start) * 1000
            _record_turn_metrics(diagnostics, stt_ms=stt_ms)
            if not transcript:
                reason = str(getattr(self.stt_adapter, "last_reject_reason", None) or "empty_transcript")
                _record_rejected_segment(diagnostics, reason=reason, source="stt")
                logger.info("STT rejected reason=%s latency=%.0fms", reason, stt_ms)
                return
            if robot_activity and not _is_confident_robot_activity_barge_in(speech_ratio, avg_snr_db):
                reason = "robot_activity_low_confidence_transcript"
                _record_rejected_segment(
                    diagnostics,
                    reason=reason,
                    source="stt",
                    speech_confidence_ratio=speech_ratio,
                    avg_snr_db=avg_snr_db,
                )
                logger.info(
                    "STT rejected reason=%s latency=%.0fms text=%r speech_ratio=%s snr=%s",
                    reason,
                    stt_ms,
                    _truncate_for_log(transcript),
                    speech_ratio,
                    avg_snr_db,
                )
                return
            if robot_activity:
                logger.info("Validated barge-in transcript during robot activity; clearing assistant output.")
                self._interrupt_assistant_output(cancel_processing=False)
            logger.info("STT transcript latency=%.0fms text=%r", stt_ms, _truncate_for_log(transcript))
            self._notify("notify_user_transcript", transcript)
            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
            self._append_message({"role": "user", "content": transcript})
            await self._respond_to_current_messages(turn_started=started)
        finally:
            self._processing_task = None

    async def _respond_to_current_messages(self, *, turn_started: float | None = None) -> None:
        llm_start = time.perf_counter()
        generation = self._generation
        latest_user_text = _latest_user_message_content(self._messages)
        if _latest_user_message_needs_tools(self._messages):
            tool_calls = []
            try:
                tool_calls = (await self.tool_router.route(latest_user_text, get_active_tool_specs(self.deps))).tool_calls
            except Exception:
                logger.exception("Local tool router failed while responding.")
            if tool_calls:
                response_text = await self._execute_routed_tool_call(tool_calls[0])
                llm_total_ms = (time.perf_counter() - llm_start) * 1000
                diagnostics = getattr(self.deps, "performance_diagnostics", None)
                _record_turn_metrics(diagnostics, llm_first_token_ms=llm_total_ms, llm_total_ms=llm_total_ms)
                await self._speak_response(response_text, turn_started=turn_started, generation=generation)
                self._schedule_component_warm("Local chat model", self.llm_adapter)
                return

        if await self._stream_response_if_available(
            llm_start=llm_start,
            turn_started=turn_started,
            generation=generation,
            latest_user_text=latest_user_text,
        ):
            return

        try:
            response = await self.llm_adapter.chat(self._messages, [])
            response_text = response.content
        except Exception:
            logger.exception("Local LLM failed while responding.")
            response_text = "I had trouble with my local language model just now."

        llm_total_ms = (time.perf_counter() - llm_start) * 1000
        diagnostics = getattr(self.deps, "performance_diagnostics", None)
        _record_turn_metrics(diagnostics, llm_first_token_ms=llm_total_ms, llm_total_ms=llm_total_ms)
        if not response_text:
            logger.info("Local LLM returned empty response for user_text=%r", _truncate_for_log(latest_user_text))
            response_text = "I heard you, but my local model came back empty. Try that once more?"

        await self._speak_response(response_text, turn_started=turn_started, generation=generation)

    async def _stream_response_if_available(
        self,
        *,
        llm_start: float,
        turn_started: float | None,
        generation: int,
        latest_user_text: str,
    ) -> bool:
        """Stream Gemma output into sentence-level TTS chunks when supported."""
        stream_chat = getattr(self.llm_adapter, "stream_chat", None)
        if not callable(stream_chat):
            return False

        raw_parts: list[str] = []
        chunker = _StreamingTextChunker()
        audio_started = False
        first_audio_recorded = False
        first_token_recorded = False
        diagnostics = getattr(self.deps, "performance_diagnostics", None)
        try:
            async for delta in stream_chat(self._messages, []):
                if not self._is_current_generation(generation):
                    return True
                if not delta:
                    continue
                if not first_token_recorded:
                    first_token_recorded = True
                    first_token_ms = (time.perf_counter() - llm_start) * 1000
                    _record_turn_metrics(diagnostics, llm_first_token_ms=first_token_ms)
                    logger.info("Local LLM stream first token latency=%.0fms", first_token_ms)
                raw_parts.append(delta)
                for chunk in chunker.feed(delta):
                    logger.info(
                        "TTS chunk ready latency=%.0fms text=%r",
                        (time.perf_counter() - llm_start) * 1000,
                        _truncate_for_log(chunk),
                    )
                    audio_started, first_audio_recorded = await self._speak_chunk(
                        chunk,
                        generation=generation,
                        turn_started=turn_started,
                        audio_started=audio_started,
                        first_audio_recorded=first_audio_recorded,
                    )

            response_text = "".join(raw_parts)
            for chunk in chunker.flush():
                logger.info(
                    "TTS final chunk ready latency=%.0fms text=%r",
                    (time.perf_counter() - llm_start) * 1000,
                    _truncate_for_log(chunk),
                )
                audio_started, first_audio_recorded = await self._speak_chunk(
                    chunk,
                    generation=generation,
                    turn_started=turn_started,
                    audio_started=audio_started,
                    first_audio_recorded=first_audio_recorded,
                )
        except Exception:
            logger.exception("Local LLM streaming failed while responding.")
            if audio_started:
                return True
            return False
        finally:
            if audio_started and self._is_current_generation(generation):
                self._notify("notify_assistant_audio_done")

        llm_total_ms = (time.perf_counter() - llm_start) * 1000
        _record_turn_metrics(diagnostics, llm_total_ms=llm_total_ms)
        logger.info("Local LLM stream done latency=%.0fms chars=%d", llm_total_ms, len(response_text))
        spoken_text = _normalize_spoken_text(response_text)
        if not spoken_text:
            logger.info("Local LLM returned empty response for user_text=%r", _truncate_for_log(latest_user_text))
            spoken_text = "I heard you, but my local model came back empty. Try that once more?"
            await self._speak_response(spoken_text, turn_started=turn_started, generation=generation)
            return True
        if not self._is_current_generation(generation):
            return True
        self._append_message({"role": "assistant", "content": spoken_text})
        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": spoken_text}))
        return True

    async def _speak_response(
        self,
        response_text: str,
        *,
        turn_started: float | None = None,
        generation: int | None = None,
    ) -> None:
        """Emit assistant text and synthesize local Piper audio."""
        generation = self._generation if generation is None else generation
        spoken_text = _normalize_spoken_text(response_text)
        if not spoken_text:
            logger.info("Skipping empty local response.")
            return
        if spoken_text != response_text:
            logger.debug("Normalized local assistant response before speech.")
        if not self._is_current_generation(generation):
            return
        self._append_message({"role": "assistant", "content": spoken_text})
        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": spoken_text}))
        audio_started, _ = await self._speak_chunk(
            spoken_text,
            generation=generation,
            turn_started=turn_started,
            audio_started=False,
            first_audio_recorded=False,
        )
        if audio_started:
            self._notify("notify_assistant_audio_done")

    async def _speak_chunk(
        self,
        text: str,
        *,
        generation: int,
        turn_started: float | None,
        audio_started: bool,
        first_audio_recorded: bool,
    ) -> tuple[bool, bool]:
        """Synthesize and queue one spoken text chunk."""
        spoken_text = _normalize_spoken_text(text)
        if not spoken_text or not self._is_current_generation(generation):
            return audio_started, first_audio_recorded
        logger.info("TTS synth start text=%r", _truncate_for_log(spoken_text))
        tts_start = time.perf_counter()
        sample_rate, audio = await self.tts_adapter.synthesize(spoken_text)
        tts_ms = (time.perf_counter() - tts_start) * 1000
        first_audio_ms = (
            (time.perf_counter() - turn_started) * 1000
            if turn_started is not None and audio.size and not first_audio_recorded
            else None
        )
        diagnostics = getattr(self.deps, "performance_diagnostics", None)
        _record_turn_metrics(diagnostics, tts_ms=tts_ms, first_audio_ms=first_audio_ms)
        logger.info(
            "TTS synth done latency=%.0fms first_audio=%s samples=%d",
            tts_ms,
            "n/a" if first_audio_ms is None else f"{first_audio_ms:.0f}ms",
            audio.size,
        )
        if audio.size and self._is_current_generation(generation):
            self._extend_robot_noise_window((audio.size / max(1, sample_rate)) + 0.3)
            if not audio_started:
                self._notify("notify_assistant_audio_started")
                audio_started = True
            await self.output_queue.put((sample_rate, audio.reshape(1, -1)))
            first_audio_recorded = True
        return audio_started, first_audio_recorded

    def _handle_completed_turn(self, completed_turn: LocalCompletedTurn) -> None:
        """Start STT for a completed detector turn."""
        if self._processing_task is not None:
            diagnostics = getattr(self.deps, "performance_diagnostics", None)
            _record_rejected_segment(
                diagnostics,
                reason="processing_busy",
                source="vad",
                speech_confidence_ratio=completed_turn.speech_ratio,
                noise_floor_rms=completed_turn.noise_floor_rms,
            )
            logger.info("Rejected local turn reason=processing_busy")
            return
        logger.info(
            "VAD completed turn duration=%.2fs speech_ratio=%.2f snr=%.1fdB robot_activity=%s",
            completed_turn.duration_s,
            completed_turn.speech_ratio,
            completed_turn.avg_snr_db,
            completed_turn.robot_activity,
        )
        self._processing_task = asyncio.create_task(
            self._process_turn(
                completed_turn.audio,
                robot_activity=completed_turn.robot_activity,
                speech_ratio=completed_turn.speech_ratio,
                avg_snr_db=completed_turn.avg_snr_db,
            ),
            name="local-turn",
        )

    def _handle_rejected_turn(self, rejected_turn: LocalRejectedTurn) -> None:
        """Record a detector-rejected turn without forwarding it to STT."""
        diagnostics = getattr(self.deps, "performance_diagnostics", None)
        _record_rejected_segment(
            diagnostics,
            reason=rejected_turn.reason,
            source="vad",
            speech_confidence_ratio=rejected_turn.speech_ratio,
            noise_floor_rms=rejected_turn.noise_floor_rms,
            robot_activity=rejected_turn.robot_activity,
            avg_snr_db=rejected_turn.avg_snr_db,
        )
        logger.info(
            "VAD rejected local audio segment reason=%s duration=%.2fs speech_ratio=%.2f snr=%.1fdB robot_activity=%s",
            rejected_turn.reason,
            rejected_turn.duration_s,
            rejected_turn.speech_ratio,
            rejected_turn.avg_snr_db,
            rejected_turn.robot_activity,
        )

    async def _execute_routed_tool_call(self, tool_call: dict[str, Any]) -> str:
        """Execute one routed tool call and return a templated spoken acknowledgement."""
        name = str(tool_call.get("name") or "")
        raw_arguments = tool_call.get("arguments")
        arguments: dict[str, Any] = (
            {str(key): value for key, value in raw_arguments.items()} if isinstance(raw_arguments, dict) else {}
        )
        logger.info("Tool call executing name=%s args=%s", name, arguments)
        if name in {tool.value for tool in SystemTool}:
            result = await dispatch_tool_call_with_manager(name, json.dumps(arguments), self.deps, self.tool_manager)
        else:
            result = await dispatch_tool_call(name, json.dumps(arguments), self.deps)
        logger.info("Tool call result name=%s summary=%s", name, _truncate_for_log(json.dumps(result, default=str)))
        self._extend_robot_noise_window(_tool_noise_window_s(name, arguments, result))
        await self.output_queue.put(
            AdditionalOutputs(
                {
                    "role": "assistant",
                    "content": json.dumps(result),
                    "metadata": {"title": f"Used tool {name}", "status": "done"},
                }
            )
        )
        return _tool_ack_text(name, arguments, result)

    async def _handle_tool_notification(self, notification: ToolNotification) -> None:
        """Log completed background-tool notifications for local manager visibility."""
        logger.info(
            "Background tool notification name=%s status=%s result=%s error=%s",
            notification.tool_name,
            notification.status.value,
            _truncate_for_log(json.dumps(notification.result, default=str)) if notification.result else None,
            notification.error,
        )

    def _append_message(self, message: dict[str, Any]) -> None:
        """Append chat context while bounding local history size."""
        self._messages.append(message)
        self._trim_context_history()

    def _trim_context_history(self) -> None:
        """Keep local context small."""
        if not self._messages:
            return
        first = self._messages[0]
        rest = self._messages[1:]
        if len(rest) > self._max_messages - 1:
            rest = rest[-(self._max_messages - 1) :]
        self._messages = [first, *rest]

    def _extend_robot_noise_window(self, duration_s: float) -> None:
        """Suppress robot-generated mic artifacts for at least the given duration."""
        if duration_s <= 0:
            return
        self._robot_noise_until = max(self._robot_noise_until, time.monotonic() + duration_s)
        self._record_voice_activity(self._robot_activity_state(time.monotonic()))

    def _robot_activity_state(self, now: float) -> dict[str, object]:
        """Return whether robot motion/playback should strengthen speech gating."""
        window_ms = max(0.0, (self._robot_noise_until - now) * 1000)
        active = window_ms > 0.0

        movement_manager = getattr(self.deps, "movement_manager", None)
        get_status = getattr(movement_manager, "get_status", None)
        if callable(get_status):
            try:
                status = get_status()
            except Exception:
                status = None
            if isinstance(status, dict):
                active = active or bool(status.get("active_motion"))
                active = active or int(status.get("queue_size") or 0) > 0

        camera_worker = getattr(self.deps, "camera_worker", None)
        if bool(getattr(camera_worker, "is_head_tracking_enabled", False)):
            offsets = _camera_tracking_offsets(camera_worker)
            active = active or any(abs(value) > 0.002 for value in offsets)

        return {
            "active": active,
            "active_motion_playback": active,
            "robot_noise_suppression_window_ms": round(window_ms, 1),
        }

    def _record_voice_activity(self, activity: dict[str, object]) -> None:
        """Push current detector/activity state into diagnostics."""
        diagnostics = getattr(self.deps, "performance_diagnostics", None)
        set_voice_activity = getattr(diagnostics, "set_voice_activity", None)
        if callable(set_voice_activity):
            payload = self.turn_detector.snapshot()
            set_voice_activity(**payload, **activity)

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


class _StreamingTextChunker:
    """Split streamed model text into TTS-friendly sentence/phrase chunks."""

    def __init__(self, *, first_words: int = 14, later_words: int = 24) -> None:
        self._buffer = ""
        self._chunks_emitted = 0
        self._first_words = first_words
        self._later_words = later_words

    def feed(self, text: str) -> list[str]:
        """Add text and return chunks ready for speech."""
        self._buffer += text
        chunks: list[str] = []
        while True:
            chunk = self._pop_ready_chunk()
            if chunk is None:
                break
            chunks.append(chunk)
        return chunks

    def flush(self) -> list[str]:
        """Return any remaining text as a final chunk."""
        text = self._buffer.strip()
        self._buffer = ""
        if not text:
            return []
        self._chunks_emitted += 1
        return [text]

    def _pop_ready_chunk(self) -> str | None:
        boundary = _sentence_boundary_index(self._buffer)
        if boundary is not None:
            chunk = self._buffer[:boundary].strip()
            self._buffer = self._buffer[boundary:].lstrip()
            self._chunks_emitted += 1
            return chunk

        limit = self._first_words if self._chunks_emitted == 0 else self._later_words
        words = re.findall(r"\S+", self._buffer)
        if len(words) < limit:
            return None
        chunk_words = words[:limit]
        chunk = " ".join(chunk_words).strip()
        remainder = self._buffer
        for word in chunk_words:
            index = remainder.find(word)
            if index < 0:
                continue
            remainder = remainder[index + len(word) :]
        self._buffer = remainder.lstrip()
        self._chunks_emitted += 1
        return chunk


def _sentence_boundary_index(text: str) -> int | None:
    """Return the end index for the first sentence-like boundary."""
    match = re.search(r"(?<=[.!?])(?:\s+|$)", text)
    return match.end() if match else None


def _latest_user_message_needs_tools(messages: list[dict[str, Any]]) -> bool:
    """Return whether the latest user turn appears to need robot tools."""
    latest = next((m for m in reversed(messages) if m.get("role") == "user"), {})
    content = str(latest.get("content") or "").casefold()
    if not content:
        return False
    tool_keywords = {
        "camera",
        "see",
        "look",
        "who",
        "face",
        "person",
        "remember",
        "dance",
        "emotion",
        "move",
        "head",
        "track",
        "tracking",
        "stop",
        "cancel",
        "task",
        "wave",
        "turn",
        "nod",
        "shake",
    }
    return any(keyword in content for keyword in tool_keywords)


def _latest_user_message_content(messages: list[dict[str, Any]]) -> str:
    """Return latest user content from local chat history."""
    latest = next((m for m in reversed(messages) if m.get("role") == "user"), {})
    return str(latest.get("content") or "")


def _is_confident_robot_activity_barge_in(speech_ratio: float | None, avg_snr_db: float | None) -> bool:
    """Return whether robot-activity audio is strong enough to treat as real barge-in speech."""
    if speech_ratio is None or avg_snr_db is None:
        return True
    return speech_ratio >= 0.45 and avg_snr_db >= 8.0


def _normalize_spoken_text(text: str) -> str:
    """Remove lightweight Markdown that speech engines read aloud awkwardly."""
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^(?:[*+-]|\d+[.)])\s+", "", line)
        if line:
            lines.append(line)
    cleaned = " ".join(lines) if lines else text.strip()
    cleaned = re.sub(r"\s+(?:[*+-]|\d+[.)])\s+", " ", cleaned)
    cleaned = re.sub(r"(?<!\w)[*_]{1,3}(?=\w)", "", cleaned)
    cleaned = re.sub(r"(?<=\w)[*_]{1,3}(?!\w)", "", cleaned)
    cleaned = re.sub(r"[\U00002600-\U000027BF\U0001F300-\U0001FAFF]+", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _camera_tracking_offsets(camera_worker: object | None) -> tuple[float, ...]:
    """Return current tracking offsets if a camera worker exposes them."""
    if camera_worker is None:
        return ()
    offsets: list[float] = []
    get_offsets = getattr(camera_worker, "get_face_tracking_offsets", None)
    if callable(get_offsets):
        try:
            raw_offsets = get_offsets()
        except Exception:
            raw_offsets = ()
        if isinstance(raw_offsets, (list, tuple)):
            offsets.extend(float(value) for value in raw_offsets if isinstance(value, (int, float)))
    get_body_yaw = getattr(camera_worker, "get_tracking_body_yaw_offset", None)
    if callable(get_body_yaw):
        try:
            body_yaw = get_body_yaw()
        except Exception:
            body_yaw = 0.0
        if isinstance(body_yaw, (int, float)):
            offsets.append(float(body_yaw))
    return tuple(offsets)


def _tool_noise_window_s(name: str, arguments: dict[str, Any], result: dict[str, Any]) -> float:
    """Return a conservative robot-noise suppression window for a tool call."""
    if result.get("error"):
        return 0.3
    if name == "move_head":
        return 1.4
    if name in {"look_at_person", "head_tracking"}:
        return 1.0
    if name == "dance":
        repeat = arguments.get("repeat")
        try:
            repeat_count = max(1, min(5, int(repeat or result.get("repeat") or 1)))
        except (TypeError, ValueError):
            repeat_count = 1
        return 4.0 * repeat_count
    if name == "play_emotion":
        return 3.0
    if name in {"stop_dance", "stop_emotion", "task_cancel"}:
        return 0.5
    return 0.8


def _tool_ack_text(name: str, arguments: dict[str, Any], result: dict[str, Any]) -> str:
    """Return a short templated acknowledgement for a routed local tool."""
    if result.get("error"):
        return f"I couldn't complete that: {result['error']}"
    if name == "who_am_i":
        identity = str(result.get("name") or "").strip()
        if identity:
            return f"You look like {identity}."
        message = str(result.get("message") or "").strip()
        return message or "I can't tell who you are yet."
    if name == "who_is_here":
        people = result.get("people")
        if not isinstance(people, list) or not people:
            return "I don't see anyone I can identify right now."
        named = [str(person.get("name")) for person in people if isinstance(person, dict) and person.get("name")]
        unknown_count = sum(1 for person in people if isinstance(person, dict) and not person.get("name"))
        unknown_label = "unknown person" if unknown_count == 1 else "unknown people"
        if named and unknown_count:
            return f"I see {', '.join(named)} and {unknown_count} {unknown_label}."
        if named:
            return f"I see {', '.join(named)}."
        return f"I see {unknown_count} {unknown_label}."
    if name == "camera":
        description = str(result.get("image_description") or "").strip()
        if description:
            return description
        if result.get("b64_im"):
            return "I took a picture, but I need a vision answer to describe it."
    if name == "task_status":
        status = str(result.get("status") or "").strip()
        message = str(result.get("message") or "").strip()
        if message:
            return message
        if status:
            return f"Task status: {status}."
    if name == "look_at_person":
        person = str(result.get("name") or arguments.get("name") or "").strip()
        return f"Okay, looking at {person}." if person else "Okay, looking there."
    if name == "move_head":
        status = str(result.get("status") or "").strip()
        return f"Okay, {status}." if status else "Okay, moving my head."
    if name == "head_tracking":
        return str(result.get("status") or "Okay, updated head tracking.").capitalize() + "."
    if name == "dance":
        move = str(result.get("move") or arguments.get("move") or "").strip()
        return f"Okay, starting {move}." if move else "Okay, starting a dance."
    if name == "play_emotion":
        emotion = str(result.get("emotion") or arguments.get("emotion") or "").strip()
        return f"Okay, playing {emotion}." if emotion else "Okay, playing an emotion."
    if name.startswith("stop_"):
        return "Okay, stopping that."
    return "Okay, done."


def _truncate_for_log(value: str, limit: int = 220) -> str:
    """Return compact text for log events."""
    cleaned = " ".join(str(value).split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "..."


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


def _record_rejected_segment(
    diagnostics: object | None,
    *,
    reason: str,
    source: str,
    **payload: object,
) -> None:
    """Record a rejected local speech/STT segment if diagnostics are available."""
    record_rejected = getattr(diagnostics, "record_rejected_segment", None)
    if callable(record_rejected):
        record_rejected(reason=reason, source=source, **payload)
