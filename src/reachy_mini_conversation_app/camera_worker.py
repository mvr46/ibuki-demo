"""Camera worker thread with frame buffering and optional head tracking."""

import time
import logging
import threading
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import linear_pose_interpolation
from reachy_mini_conversation_app.vision.head_tracking import HeadTracker, HeadTrackerTarget
from reachy_mini_conversation_app.vision.head_tracking.speaker import (
    SoundTarget,
    DaemonDoAPoller,
    SpeakerSelectionState,
    SoundOrientationController,
    select_speaker,
)


logger = logging.getLogger(__name__)

DOA_FRESH_SECONDS = 1.0
USER_SPEECH_HOLD_SECONDS = 1.0
USER_TRANSCRIPT_HOLD_SECONDS = 1.0
SOUND_SEARCH_HOLD_SECONDS = 3.0
SOUND_SEARCH_THRESHOLD = 0.14
SOUND_FACE_MATCH_TOLERANCE = 0.42
SOUND_FRESH_FACE_DIRECTION_THRESHOLD = 0.08
BODY_YAW_VISUAL_SETTLE_RATE = 0.35
BODY_YAW_VISUAL_ZERO_EPSILON = 0.02
SOUND_DEBUG_LOG_INTERVAL_SECONDS = 0.5
SPEAKER_FOCUS_SMOOTHING = 0.25
SPEAKER_FOCUS_YAW_GAIN = 0.55
SPEAKER_FOCUS_PITCH_GAIN = 0.25


class CameraWorker:
    """Thread-safe camera worker with frame buffering and optional head tracking."""

    def __init__(
        self,
        reachy_mini: ReachyMini,
        head_tracker: HeadTracker | None = None,
        doa_poller: DaemonDoAPoller | None = None,
    ) -> None:
        """Initialize."""
        self.reachy_mini = reachy_mini
        self.head_tracker = head_tracker

        self.latest_frame: NDArray[np.uint8] | None = None
        self.frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.is_head_tracking_enabled = True
        self.face_tracking_offsets: List[float] = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]  # x, y, z, roll, pitch, yaw
        self.tracking_body_yaw_offset = 0.0
        self.face_tracking_lock = threading.Lock()

        self.last_face_detected_time: float | None = None
        self.interpolation_start_time: float | None = None
        self.interpolation_start_pose: NDArray[np.float32] | None = None
        self.interpolation_start_body_yaw = 0.0
        self.face_lost_delay = 2.0
        self.interpolation_duration = 1.0

        self.previous_head_tracking_state = self.is_head_tracking_enabled
        self._speaker_selection_state = SpeakerSelectionState()
        self._sound_controller = SoundOrientationController()
        self._smoothed_target_x: float | None = None
        self._smoothed_target_y: float | None = None
        self._sound_search_target: SoundTarget | None = None
        self._sound_search_started_at: float | None = None
        self._sound_search_last_seen_at: float | None = None
        self._last_sound_debug_at = 0.0
        self._last_sound_debug_key: tuple[str, str] | None = None
        self._doa_poller = doa_poller if doa_poller is not None else self._build_doa_poller()

        self._speech_state_lock = threading.Lock()
        self._user_speech_active = False
        self._assistant_speaking = False
        self._last_user_speech_at: float | None = None
        self._last_user_transcript_at: float | None = None

    def get_latest_frame(self) -> NDArray[np.uint8] | None:
        """Get the latest frame (thread-safe)."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_face_tracking_offsets(
        self,
    ) -> Tuple[float, float, float, float, float, float]:
        """Get current face tracking offsets (thread-safe)."""
        with self.face_tracking_lock:
            offsets = self.face_tracking_offsets
            return (offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], offsets[5])

    def get_tracking_body_yaw_offset(self) -> float:
        """Get current body-yaw speaker-tracking offset (thread-safe)."""
        with self.face_tracking_lock:
            return float(self.tracking_body_yaw_offset)

    def notify_user_speech_started(self) -> None:
        """Notify speaker tracking that user speech is active."""
        now = time.monotonic()
        with self._speech_state_lock:
            self._user_speech_active = True
            self._assistant_speaking = False
            self._last_user_speech_at = now

    def notify_user_speech_stopped(self) -> None:
        """Notify speaker tracking that user speech stopped."""
        now = time.monotonic()
        with self._speech_state_lock:
            self._user_speech_active = False
            self._last_user_speech_at = now

    def notify_user_transcript(self) -> None:
        """Notify speaker tracking that a user utterance was recognized."""
        now = time.monotonic()
        with self._speech_state_lock:
            self._user_speech_active = False
            self._last_user_speech_at = now
            self._last_user_transcript_at = now

    def notify_user_partial(self) -> None:
        """Notify speaker tracking that partial user speech is flowing."""
        now = time.monotonic()
        with self._speech_state_lock:
            self._last_user_speech_at = now
            self._last_user_transcript_at = now

    def notify_assistant_audio_started(self) -> None:
        """Pause spatial-audio speaker tracking while assistant audio is active."""
        with self._speech_state_lock:
            self._assistant_speaking = True
        self._clear_sound_search()

    def notify_assistant_audio_done(self) -> None:
        """Resume spatial-audio speaker tracking after assistant audio completes."""
        with self._speech_state_lock:
            self._assistant_speaking = False

    def set_head_tracking_enabled(self, enabled: bool) -> None:
        """Enable/disable head tracking."""
        self.is_head_tracking_enabled = enabled
        logger.info(f"Head tracking {'enabled' if enabled else 'disabled'}")

    def start(self) -> None:
        """Start the camera worker loop in a thread."""
        self._stop_event.clear()
        if self._doa_poller is not None:
            doa_start = getattr(self._doa_poller, "start", None)
            if callable(doa_start):
                doa_start()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.debug("Camera worker started")

    def stop(self) -> None:
        """Stop the camera worker loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        if self._doa_poller is not None:
            doa_stop = getattr(self._doa_poller, "stop", None)
            if callable(doa_stop):
                doa_stop()
        head_tracker_close = getattr(self.head_tracker, "close", None)
        if callable(head_tracker_close):
            head_tracker_close()

        logger.debug("Camera worker stopped")

    def _build_doa_poller(self) -> DaemonDoAPoller | None:
        """Create the robot-side DoA poller when the tracker can use target lists."""
        if not self._supports_spatial_speaker_tracking():
            return None
        client = getattr(self.reachy_mini, "client", None)
        host = getattr(client, "host", None) or getattr(self.reachy_mini, "host", None)
        port = getattr(client, "port", None) or getattr(self.reachy_mini, "port", None)
        if not host or port is None:
            return None
        try:
            return DaemonDoAPoller(str(host), int(port))
        except Exception as exc:
            logger.debug("Skipping robot DoA poller setup: %s", exc)
            return None

    def _supports_spatial_speaker_tracking(self) -> bool:
        """Return whether the active tracker can provide all visible targets."""
        return callable(getattr(self.head_tracker, "get_head_targets", None))

    def _fresh_sound_target(self, current_time: float) -> SoundTarget | None:
        """Return a fresh DoA target gated by speech state."""
        if self._doa_poller is None:
            return None
        target, target_at = self._doa_poller.get_latest()
        if target is None or target_at is None or current_time - target_at > DOA_FRESH_SECONDS:
            return None

        with self._speech_state_lock:
            assistant_speaking = self._assistant_speaking
            user_speech_active = self._user_speech_active
            last_user_speech_at = self._last_user_speech_at
            last_user_transcript_at = self._last_user_transcript_at

        if assistant_speaking:
            self._clear_sound_search()
            return None

        recent_user_speech = (
            last_user_speech_at is not None and current_time - last_user_speech_at <= USER_SPEECH_HOLD_SECONDS
        )
        recent_user_transcript = (
            last_user_transcript_at is not None
            and current_time - last_user_transcript_at <= USER_TRANSCRIPT_HOLD_SECONDS
        )
        if not (target.speech_detected or user_speech_active or recent_user_speech or recent_user_transcript):
            return None
        return target

    def _is_off_front_sound(self, target: SoundTarget) -> bool:
        """Return whether a sound cue should initiate directional search."""
        return abs(target.x_offset) >= SOUND_SEARCH_THRESHOLD

    def _sound_direction_label(self, target: SoundTarget) -> str:
        """Return a human-readable side label for debug logs."""
        if target.x_offset >= SOUND_SEARCH_THRESHOLD:
            return "right"
        if target.x_offset <= -SOUND_SEARCH_THRESHOLD:
            return "left"
        return "front"

    def _log_off_front_sound_debug(
        self,
        event: str,
        target: SoundTarget,
        current_time: float,
        *,
        visible_faces: int,
        matching_faces: int | None = None,
        selected: HeadTrackerTarget | None = None,
    ) -> None:
        """Log a throttled trace for side-sound speaker-search decisions."""
        direction = self._sound_direction_label(target)
        key = (event, direction)
        if key == self._last_sound_debug_key and current_time - self._last_sound_debug_at < SOUND_DEBUG_LOG_INTERVAL_SECONDS:
            return

        self._last_sound_debug_key = key
        self._last_sound_debug_at = current_time
        match_text = "" if matching_faces is None else f" matching_faces={matching_faces}"
        selected_text = "" if selected is None else f" selected_x={selected.x_offset:.2f}"
        logger.debug(
            "Spatial audio: sound not in front heard event=%s direction=%s x_offset=%.2f angle=%.2f "
            "speech_detected=%s visible_faces=%d%s%s",
            event,
            direction,
            target.x_offset,
            target.angle,
            target.speech_detected,
            visible_faces,
            match_text,
            selected_text,
        )

    def _clear_sound_search(self) -> None:
        """Clear any held off-front sound-search intent."""
        self._sound_search_target = None
        self._sound_search_started_at = None
        self._sound_search_last_seen_at = None

    def _remember_sound_search(self, target: SoundTarget, current_time: float) -> None:
        """Hold an off-front sound cue long enough for the robot to turn."""
        previous_direction = (
            1.0 if self._sound_search_target is not None and self._sound_search_target.x_offset >= 0.0 else -1.0
        )
        next_direction = 1.0 if target.x_offset >= 0.0 else -1.0
        if self._sound_search_target is None or previous_direction != next_direction:
            self._sound_search_started_at = current_time
        self._sound_search_target = target
        self._sound_search_last_seen_at = current_time

    def _refresh_sound_search(self, current_time: float) -> None:
        """Keep an active search alive while related speech remains fresh."""
        if self._sound_search_target is not None:
            self._sound_search_last_seen_at = current_time

    def _current_sound_search_target(self, current_time: float) -> SoundTarget | None:
        """Return the held off-front sound cue while it is still useful."""
        if self._sound_search_target is None or self._sound_search_last_seen_at is None:
            return None

        with self._speech_state_lock:
            assistant_speaking = self._assistant_speaking

        if assistant_speaking or current_time - self._sound_search_last_seen_at > SOUND_SEARCH_HOLD_SECONDS:
            self._clear_sound_search()
            return None
        return self._sound_search_target

    def _update_sound_orientation(self, target: SoundTarget | None, current_time: float) -> None:
        """Update body-yaw tracking from spatial audio."""
        if target is None:
            self._sound_controller.update(None)
            return

        self.last_face_detected_time = current_time
        command = self._sound_controller.update(target)
        if command is None:
            return
        with self.face_tracking_lock:
            self.tracking_body_yaw_offset = command.body_yaw

    def _settle_body_yaw_toward_neutral(self) -> None:
        """Blend audio-search body yaw away once visual speaker focus is available."""
        with self.face_tracking_lock:
            body_yaw = self.tracking_body_yaw_offset * (1.0 - BODY_YAW_VISUAL_SETTLE_RATE)
            if abs(body_yaw) <= BODY_YAW_VISUAL_ZERO_EPSILON:
                body_yaw = 0.0
            self.tracking_body_yaw_offset = body_yaw
        self._sound_controller.body_yaw = body_yaw

    def _get_head_targets(self, frame: NDArray[np.uint8]) -> list[HeadTrackerTarget]:
        """Return target-list detections from the tracker."""
        get_targets = getattr(self.head_tracker, "get_head_targets", None)
        if not callable(get_targets):
            return []
        try:
            targets = get_targets(frame)
        except Exception as exc:
            logger.error("Head target detection failed: %s", exc)
            return []
        return [target for target in targets if isinstance(target, HeadTrackerTarget)]

    def _speaker_focus_pixels(self, target: HeadTrackerTarget) -> tuple[int, int]:
        """Return a subtle image point for speaker focus."""
        smoothing = min(1.0, max(0.0, SPEAKER_FOCUS_SMOOTHING))
        if self._smoothed_target_x is None or self._smoothed_target_y is None:
            self._smoothed_target_x = target.x_offset
            self._smoothed_target_y = target.y_offset
        else:
            self._smoothed_target_x = (1.0 - smoothing) * self._smoothed_target_x + smoothing * target.x_offset
            self._smoothed_target_y = (1.0 - smoothing) * self._smoothed_target_y + smoothing * target.y_offset

        width, height = target.frame_size
        width = max(1, int(width))
        height = max(1, int(height))
        effective_x = max(-0.40, min(0.40, self._smoothed_target_x * SPEAKER_FOCUS_YAW_GAIN))
        effective_y = max(-0.22, min(0.22, self._smoothed_target_y * SPEAKER_FOCUS_PITCH_GAIN))
        pixel_x = int(round((0.5 + effective_x / 2.0) * width))
        pixel_y = int(round((0.5 + effective_y / 2.0) * height))
        return min(max(pixel_x, 1), max(1, width - 1)), min(max(pixel_y, 1), max(1, height - 1))

    def _apply_tracking_pixels(self, pixel_x: float, pixel_y: float) -> None:
        """Convert image-space focus pixels into additive head offsets."""
        target_pose = self.reachy_mini.look_at_image(
            pixel_x,
            pixel_y,
            duration=0.0,
            perform_movement=False,
        )

        translation = target_pose[:3, 3]
        rotation = R.from_matrix(target_pose[:3, :3]).as_euler("xyz", degrees=False)

        # The camera FOV is tighter than the motion model expects.
        translation *= 0.6
        rotation *= 0.6

        with self.face_tracking_lock:
            self.face_tracking_offsets = [
                translation[0],
                translation[1],
                translation[2],
                rotation[0],
                rotation[1],
                rotation[2],
            ]

    def _apply_eye_center(self, frame: NDArray[np.uint8], eye_center: NDArray[np.float32]) -> None:
        """Apply a classic normalized tracker point."""
        h, w = frame.shape[:2]
        eye_center_norm = (eye_center + 1) / 2
        self._apply_tracking_pixels(eye_center_norm[0] * w, eye_center_norm[1] * h)

    def _reset_head_tracking_offsets(self) -> None:
        """Return head-only tracking offsets to neutral during audio search."""
        self._smoothed_target_x = None
        self._smoothed_target_y = None
        with self.face_tracking_lock:
            self.face_tracking_offsets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def _matching_sound_search_targets(
        self,
        targets: list[HeadTrackerTarget],
        *,
        search_target: SoundTarget,
        fresh_target: SoundTarget | None,
        current_time: float,
    ) -> list[HeadTrackerTarget]:
        """Return visible faces that are plausible matches for an active audio search."""
        if not targets:
            return []

        if fresh_target is not None:
            if self._is_off_front_sound(fresh_target):
                direction = 1.0 if fresh_target.x_offset >= 0.0 else -1.0
                return [
                    target
                    for target in targets
                    if abs(target.x_offset - fresh_target.x_offset) <= SOUND_FACE_MATCH_TOLERANCE
                    and direction * target.x_offset >= SOUND_FRESH_FACE_DIRECTION_THRESHOLD
                ]
            return [target for target in targets if abs(target.x_offset) <= SOUND_FACE_MATCH_TOLERANCE]

        direction = 1.0 if search_target.x_offset >= 0.0 else -1.0
        return [target for target in targets if direction * target.x_offset >= SOUND_FRESH_FACE_DIRECTION_THRESHOLD]

    def _select_audio_aligned_target(
        self,
        targets: list[HeadTrackerTarget],
        *,
        sound_target: SoundTarget,
        fresh: bool,
    ) -> HeadTrackerTarget | None:
        """Select the visible face nearest a directional audio cue."""
        if fresh and self._is_off_front_sound(sound_target):
            direction = 1.0 if sound_target.x_offset >= 0.0 else -1.0
            candidates = [
                target
                for target in targets
                if abs(target.x_offset - sound_target.x_offset) <= SOUND_FACE_MATCH_TOLERANCE
                and direction * target.x_offset >= SOUND_FRESH_FACE_DIRECTION_THRESHOLD
            ]
        else:
            candidates = targets

        if not candidates:
            return None
        return max(
            candidates,
            key=lambda target: (
                1.0 - min(2.0, abs(target.x_offset - sound_target.x_offset)) / 2.0,
                target.confidence,
            ),
        )

    def _lock_visual_speaker(self, target: HeadTrackerTarget, current_time: float) -> None:
        """Apply visual speaker focus and let body yaw settle behind it."""
        self.last_face_detected_time = current_time
        self.interpolation_start_time = None
        self._speaker_selection_state.remember(target)
        pixel_x, pixel_y = self._speaker_focus_pixels(target)
        self._apply_tracking_pixels(pixel_x, pixel_y)
        self._settle_body_yaw_toward_neutral()

    def _hold_sound_search_focus(self, current_time: float) -> None:
        """Let audio-driven body yaw lead until a matching face appears."""
        self.last_face_detected_time = current_time
        self.interpolation_start_time = None
        self._reset_head_tracking_offsets()

    def _update_tracking_from_frame(self, frame: NDArray[np.uint8], current_time: float) -> None:
        """Update face/body tracking offsets from one camera frame."""
        sound_target = self._fresh_sound_target(current_time)

        if self._supports_spatial_speaker_tracking():
            targets = self._get_head_targets(frame)
            if sound_target is not None and self._is_off_front_sound(sound_target):
                audio_selected = self._select_audio_aligned_target(targets, sound_target=sound_target, fresh=True)
                if audio_selected is not None:
                    self._log_off_front_sound_debug(
                        "matched_visible_face",
                        sound_target,
                        current_time,
                        visible_faces=len(targets),
                        selected=audio_selected,
                    )
                    self._clear_sound_search()
                    self._sound_controller.reset_observation()
                    self._lock_visual_speaker(audio_selected, current_time)
                    return
                self._log_off_front_sound_debug(
                    "starting_search",
                    sound_target,
                    current_time,
                    visible_faces=len(targets),
                    matching_faces=0,
                )
                self._remember_sound_search(sound_target, current_time)
            elif sound_target is not None:
                self._clear_sound_search()
                self._settle_body_yaw_toward_neutral()

            search_target = self._current_sound_search_target(current_time)
            self._update_sound_orientation(search_target, current_time)
            if search_target is not None:
                matching_targets = self._matching_sound_search_targets(
                    targets,
                    search_target=search_target,
                    fresh_target=None,
                    current_time=current_time,
                )
                selected = self._select_audio_aligned_target(matching_targets, sound_target=search_target, fresh=False)
                if selected is not None:
                    self._log_off_front_sound_debug(
                        "search_found_face",
                        search_target,
                        current_time,
                        visible_faces=len(targets),
                        matching_faces=len(matching_targets),
                        selected=selected,
                    )
                    self._clear_sound_search()
                    self._sound_controller.reset_observation()
                    self._lock_visual_speaker(selected, current_time)
                else:
                    self._log_off_front_sound_debug(
                        "searching_no_face",
                        search_target,
                        current_time,
                        visible_faces=len(targets),
                        matching_faces=0,
                    )
                    self._hold_sound_search_focus(current_time)
                return

            selected = select_speaker(
                targets,
                audio_x_offset=(
                    sound_target.x_offset
                    if sound_target is not None and self._is_off_front_sound(sound_target)
                    else None
                ),
                state=self._speaker_selection_state,
            ).target
            if selected is not None:
                self._lock_visual_speaker(selected, current_time)
            return

        if self.head_tracker is None:
            return

        eye_center, _ = self.head_tracker.get_head_position(frame)
        if eye_center is not None:
            self.last_face_detected_time = current_time
            self.interpolation_start_time = None
            self._smoothed_target_x = None
            self._smoothed_target_y = None
            self._apply_eye_center(frame, eye_center)

    def _update_neutral_interpolation(self, current_time: float, neutral_pose: NDArray[np.float32]) -> None:
        """Smooth tracking offsets back to neutral after the target is lost."""
        if self.last_face_detected_time is None:
            return

        time_since_face_lost = current_time - self.last_face_detected_time
        if time_since_face_lost < self.face_lost_delay:
            return

        if self.interpolation_start_time is None:
            self.interpolation_start_time = current_time
            with self.face_tracking_lock:
                current_translation = self.face_tracking_offsets[:3]
                current_rotation_euler = self.face_tracking_offsets[3:]
                self.interpolation_start_body_yaw = self.tracking_body_yaw_offset
                pose_matrix = np.eye(4, dtype=np.float32)
                pose_matrix[:3, 3] = current_translation
                pose_matrix[:3, :3] = R.from_euler(
                    "xyz",
                    current_rotation_euler,
                ).as_matrix()
                self.interpolation_start_pose = pose_matrix

        elapsed_interpolation = current_time - self.interpolation_start_time
        t = min(1.0, elapsed_interpolation / self.interpolation_duration)

        interpolated_pose = linear_pose_interpolation(
            self.interpolation_start_pose,
            neutral_pose,
            t,
        )

        translation = interpolated_pose[:3, 3]
        rotation = R.from_matrix(interpolated_pose[:3, :3]).as_euler("xyz", degrees=False)

        with self.face_tracking_lock:
            self.face_tracking_offsets = [
                translation[0],
                translation[1],
                translation[2],
                rotation[0],
                rotation[1],
                rotation[2],
            ]
            self.tracking_body_yaw_offset = self.interpolation_start_body_yaw * (1.0 - t)

        if t >= 1.0:
            self.last_face_detected_time = None
            self.interpolation_start_time = None
            self.interpolation_start_pose = None
            self.interpolation_start_body_yaw = 0.0
            self._smoothed_target_x = None
            self._smoothed_target_y = None
            self._sound_controller.reset()

    def working_loop(self) -> None:
        """Run the camera worker loop."""
        logger.debug("Starting camera working loop")

        neutral_pose = np.eye(4, dtype=np.float32)
        self.previous_head_tracking_state = self.is_head_tracking_enabled

        while not self._stop_event.is_set():
            try:
                current_time = time.monotonic()
                frame = self.reachy_mini.media.get_frame()

                if frame is not None:
                    # Keep the latest frame available for tools and UI consumers
                    with self.frame_lock:
                        self.latest_frame = frame

                    if self.previous_head_tracking_state and not self.is_head_tracking_enabled:
                        # Reuse the face-lost interpolation path to return smoothly to neutral
                        self.last_face_detected_time = current_time
                        self.interpolation_start_time = None
                        self.interpolation_start_pose = None

                    self.previous_head_tracking_state = self.is_head_tracking_enabled

                    if self.is_head_tracking_enabled and self.head_tracker is not None:
                        self._update_tracking_from_frame(frame, current_time)

                    self._update_neutral_interpolation(current_time, neutral_pose)

                time.sleep(0.04)

            except Exception as e:
                logger.error(f"Camera worker error: {e}")
                time.sleep(0.1)

        logger.debug("Camera worker thread exited")
