"""Async stream that injects situational awareness into conversation context."""

from __future__ import annotations
import time
import asyncio
import logging

from reachy_mini_conversation_app.conversation_handler import ConversationHandler
from reachy_mini_conversation_app.face_identity_worker import VisionEvent, PerceptionSnapshot, FaceIdentifierWorker
from reachy_mini_conversation_app.speaker_attribution import format_attributed_speech


logger = logging.getLogger(__name__)


async def run_perception_stream(
    worker: FaceIdentifierWorker | None,
    handler: ConversationHandler,
    *,
    speaker_attribution_worker: object | None = None,
    snapshot_interval_s: float = 12.0,
    event_debounce_s: float = 1.5,
) -> None:
    """Drain face and speech-attribution events and emit debounced scene snapshots."""
    last_snapshot_at = 0.0
    last_snapshot_set: frozenset[str] = frozenset()

    while True:
        if _assistant_is_speaking(worker, speaker_attribution_worker):
            await asyncio.sleep(0.5)
            continue

        if worker is not None:
            for event in worker.drain_events():
                await handler.inject_environment_message(_format_event(event), trigger_response=False)
                await asyncio.sleep(event_debounce_s)

        if speaker_attribution_worker is not None:
            drain_speech_events = getattr(speaker_attribution_worker, "drain_events", None)
            if callable(drain_speech_events):
                for segment in drain_speech_events():
                    await handler.inject_environment_message(format_attributed_speech(segment), trigger_response=False)

        if worker is not None:
            now = time.monotonic()
            snapshot = worker.snapshot()
            current_set = frozenset(_visible_key(target.name) for target in snapshot.visible)
            if current_set != last_snapshot_set and now - last_snapshot_at >= snapshot_interval_s:
                await handler.inject_environment_message(_format_snapshot(snapshot, now=now), trigger_response=False)
                last_snapshot_at = now
                last_snapshot_set = current_set

        await asyncio.sleep(0.5)


def _format_event(event: VisionEvent) -> str:
    if event.kind == "entered":
        subject = event.name if event.name else "An unknown person"
        return f"[Vision: {subject} entered the frame ({event.position})]"
    if event.kind == "left":
        subject = event.name if event.name else "An unknown person"
        if event.last_seen_at is None:
            return f"[Vision: {subject} left]"
        elapsed = max(0.0, event.timestamp - event.last_seen_at)
        return f"[Vision: {subject} left, last seen {_format_elapsed(elapsed)} ago]"
    if event.kind == "named":
        subject = event.name if event.name else "An unknown person"
        return f"[Vision: {subject} is now recognized ({event.position})]"
    return "[Vision: Face visibility changed]"


def _format_snapshot(snapshot: PerceptionSnapshot, *, now: float | None = None) -> str:
    current_time = time.monotonic() if now is None else now
    visible_parts = [_format_visible_target(target) for target in snapshot.visible]
    if not visible_parts:
        visible_text = "no one"
    else:
        visible_text = ", ".join(visible_parts)

    visible_names = {target.name for target in snapshot.visible if target.name}
    recent_parts: list[str] = []
    for name, seen_at in sorted(snapshot.last_seen.items()):
        if name in visible_names:
            continue
        position = snapshot.last_positions.get(name, "center")
        recent_parts.append(f"{name} ({position}, left {_format_elapsed(max(0.0, current_time - seen_at))} ago)")

    if recent_parts:
        return f"[Vision: Visible now - {visible_text}. Last seen recently: {', '.join(recent_parts)}]"
    return f"[Vision: Visible now - {visible_text}]"


def _format_visible_target(target: object) -> str:
    name = getattr(target, "name", None) or "unknown"
    head_target = getattr(target, "target", None)
    x_offset = float(getattr(head_target, "x_offset", 0.0))
    if x_offset <= -0.33:
        position = "left"
    elif x_offset >= 0.33:
        position = "right"
    else:
        position = "center"
    return f"{name} ({position})"


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{int(round(seconds))}s"
    minutes = int(round(seconds / 60.0))
    return f"{minutes} min"


def _visible_key(name: str | None) -> str:
    return name if name else "unknown"


run_situational_awareness_stream = run_perception_stream


def _assistant_is_speaking(
    worker: FaceIdentifierWorker | None,
    speaker_attribution_worker: object | None = None,
) -> bool:
    camera_worker = getattr(worker, "camera_worker", None) if worker is not None else None
    if camera_worker is None and speaker_attribution_worker is not None:
        camera_worker = getattr(speaker_attribution_worker, "assistant_state_source", None)
    if camera_worker is None:
        return False
    lock = getattr(camera_worker, "_speech_state_lock", None)
    if lock is None:
        return bool(getattr(camera_worker, "_assistant_speaking", False))
    with lock:
        return bool(getattr(camera_worker, "_assistant_speaking", False))
