"""Speech-quality local turn detection for noisy robot microphones."""

from __future__ import annotations
import math
from typing import Literal, Iterable
from collections import deque
from dataclasses import field, dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LocalTurnDetectorConfig:
    """Tunable parameters for local speech turn detection."""

    sample_rate: int = 16000
    frame_ms: int = 20
    pre_roll_ms: int = 250
    silence_seconds: float = 0.85
    min_speech_seconds: float = 0.35
    min_speech_ratio: float = 0.45
    min_snr_db: float = 8.0
    min_frame_rms: float = 120.0
    initial_noise_floor_rms: float = 80.0
    noise_floor_alpha: float = 0.05

    @property
    def frame_samples(self) -> int:
        """Return samples per analysis frame."""
        return max(1, int(self.sample_rate * self.frame_ms / 1000))

    @property
    def pre_roll_frames(self) -> int:
        """Return number of frames retained before speech onset."""
        return max(1, int(math.ceil(self.pre_roll_ms / self.frame_ms)))

    @property
    def silence_frames(self) -> int:
        """Return number of non-speech frames that ends a turn."""
        return max(1, int(math.ceil(self.silence_seconds * 1000 / self.frame_ms)))

    @property
    def min_turn_frames(self) -> int:
        """Return minimum frames in a valid user turn."""
        return max(1, int(math.ceil(self.min_speech_seconds * 1000 / self.frame_ms)))


@dataclass(frozen=True)
class LocalFrameStats:
    """Speech/noise metrics for one detector frame."""

    rms: float
    noise_floor_rms: float
    snr_db: float
    speech_band_ratio: float
    spectral_flatness: float
    spectral_centroid_hz: float
    zero_crossing_rate: float
    peak_dominance: float
    speech_like: bool
    robot_activity: bool
    noise_class: Literal["speech_like", "narrowband", "broadband", "low_band", "quiet"]


@dataclass(frozen=True)
class LocalCompletedTurn:
    """A completed speech-quality turn ready for STT."""

    audio: NDArray[np.int16]
    duration_s: float
    speech_ratio: float
    avg_snr_db: float
    noise_floor_rms: float
    robot_activity: bool


@dataclass(frozen=True)
class LocalRejectedTurn:
    """A detector-rejected segment that must not reach STT or the LLM."""

    reason: str
    duration_s: float
    speech_ratio: float
    avg_snr_db: float
    noise_floor_rms: float
    robot_activity: bool


@dataclass(frozen=True)
class LocalTurnDetectorUpdate:
    """Events produced after feeding one audio chunk."""

    speech_started: bool = False
    speech_stopped: bool = False
    completed_turns: list[LocalCompletedTurn] = field(default_factory=list)
    rejected_turns: list[LocalRejectedTurn] = field(default_factory=list)


@dataclass(frozen=True)
class _BufferedFrame:
    audio: NDArray[np.int16]
    stats: LocalFrameStats


class LocalTurnDetector:
    """Frame-based local turn detector that rejects robot/mechanical noise."""

    def __init__(self, config: LocalTurnDetectorConfig | None = None) -> None:
        """Initialize detector state."""
        self.config = config or LocalTurnDetectorConfig()
        self._carry: NDArray[np.int16] = np.zeros(0, dtype=np.int16)
        self._pre_roll: deque[_BufferedFrame] = deque(maxlen=self.config.pre_roll_frames)
        self._segment: list[_BufferedFrame] = []
        self._speech_frame_indexes: list[int] = []
        self._silence_frames = 0
        self._noise_floor_rms = max(1.0, float(self.config.initial_noise_floor_rms))
        self._in_turn = False
        self._last_frame_stats: LocalFrameStats | None = None
        self._last_speech_ratio = 0.0

    @property
    def in_turn(self) -> bool:
        """Return whether a candidate speech segment is open."""
        return self._in_turn

    @property
    def noise_floor_rms(self) -> float:
        """Return the current adaptive noise floor in int16 RMS units."""
        return self._noise_floor_rms

    @property
    def last_frame_stats(self) -> LocalFrameStats | None:
        """Return the most recent frame metrics."""
        return self._last_frame_stats

    @property
    def last_speech_ratio(self) -> float:
        """Return speech-like ratio from the most recent completed/rejected segment."""
        return self._last_speech_ratio

    def snapshot(self) -> dict[str, object]:
        """Return JSON-friendly detector state for diagnostics."""
        stats = self._last_frame_stats
        return {
            "vad_state": "speech" if self._in_turn else "idle",
            "noise_floor_rms": round(self._noise_floor_rms, 2),
            "speech_confidence_ratio": round(self._last_speech_ratio, 3),
            "last_frame_snr_db": round(stats.snr_db, 2) if stats is not None else None,
            "last_frame_speech_band_ratio": round(stats.speech_band_ratio, 3) if stats is not None else None,
            "last_frame_noise_class": stats.noise_class if stats is not None else None,
        }

    def process(self, audio: NDArray[np.int16], *, robot_activity: bool = False) -> LocalTurnDetectorUpdate:
        """Feed audio and return any detector events."""
        if audio.size == 0:
            return LocalTurnDetectorUpdate()

        samples: NDArray[np.int16] = np.asarray(audio, dtype=np.int16).reshape(-1)
        if self._carry.size:
            samples = np.concatenate([self._carry, samples])

        frame_samples = self.config.frame_samples
        usable = (samples.size // frame_samples) * frame_samples
        self._carry = samples[usable:].copy()

        speech_started = False
        speech_stopped = False
        completed: list[LocalCompletedTurn] = []
        rejected: list[LocalRejectedTurn] = []

        for start in range(0, usable, frame_samples):
            frame = samples[start : start + frame_samples].copy()
            stats = self._analyze_frame(frame, robot_activity=robot_activity)
            self._last_frame_stats = stats

            if stats.speech_like:
                if not self._in_turn:
                    self._start_segment(frame, stats)
                    speech_started = True
                else:
                    self._append_segment(frame, stats)
                self._speech_frame_indexes.append(len(self._segment) - 1)
                self._silence_frames = 0
                continue

            self._update_noise_floor(stats.rms)
            if self._in_turn:
                self._append_segment(frame, stats)
                self._silence_frames += 1
                if self._silence_frames >= self.config.silence_frames:
                    result = self._finish_segment()
                    speech_stopped = True
                    if isinstance(result, LocalCompletedTurn):
                        completed.append(result)
                    else:
                        rejected.append(result)
            else:
                self._pre_roll.append(_BufferedFrame(frame, stats))

        return LocalTurnDetectorUpdate(
            speech_started=speech_started,
            speech_stopped=speech_stopped,
            completed_turns=completed,
            rejected_turns=rejected,
        )

    def _start_segment(self, frame: NDArray[np.int16], stats: LocalFrameStats) -> None:
        """Open a new candidate segment including pre-roll frames."""
        self._segment = list(self._pre_roll)
        self._pre_roll.clear()
        self._speech_frame_indexes = []
        self._silence_frames = 0
        self._in_turn = True
        self._append_segment(frame, stats)

    def _append_segment(self, frame: NDArray[np.int16], stats: LocalFrameStats) -> None:
        """Append one frame to the active candidate segment."""
        self._segment.append(_BufferedFrame(frame, stats))

    def _finish_segment(self) -> LocalCompletedTurn | LocalRejectedTurn:
        """Close and classify the active candidate segment."""
        segment = self._segment
        speech_indexes = self._speech_frame_indexes
        self._segment = []
        self._speech_frame_indexes = []
        self._silence_frames = 0
        self._in_turn = False

        for item in segment[-self.config.pre_roll_frames :]:
            self._pre_roll.append(item)

        if not segment:
            return self._reject("empty_segment", 0.0, 0.0, 0.0, False)

        duration_s = len(segment) * self.config.frame_ms / 1000.0
        robot_activity = any(item.stats.robot_activity for item in segment)
        if len(segment) < self.config.min_turn_frames:
            return self._reject_from_segment("too_short", segment, duration_s, robot_activity)
        if not speech_indexes:
            return self._reject_from_segment("no_speech_like_frames", segment, duration_s, robot_activity)

        first_speech = min(speech_indexes)
        last_speech = max(speech_indexes)
        voiced_region = segment[first_speech : last_speech + 1]
        speech_frames = sum(1 for item in voiced_region if item.stats.speech_like)
        speech_ratio = speech_frames / max(1, len(voiced_region))
        self._last_speech_ratio = speech_ratio
        avg_snr = _mean(item.stats.snr_db for item in voiced_region)

        narrowband_ratio = _ratio(item.stats.noise_class == "narrowband" for item in voiced_region)
        broadband_ratio = _ratio(item.stats.noise_class == "broadband" for item in voiced_region)
        low_band_ratio = _ratio(item.stats.noise_class == "low_band" for item in voiced_region)
        if narrowband_ratio >= 0.35:
            return self._reject_from_segment("narrowband_noise", segment, duration_s, robot_activity, speech_ratio, avg_snr)
        if broadband_ratio >= 0.45:
            return self._reject_from_segment("broadband_noise", segment, duration_s, robot_activity, speech_ratio, avg_snr)
        if low_band_ratio >= 0.5:
            return self._reject_from_segment("mechanical_noise", segment, duration_s, robot_activity, speech_ratio, avg_snr)

        audio = np.concatenate([item.audio for item in segment]).astype(np.int16, copy=False)
        return LocalCompletedTurn(
            audio=audio,
            duration_s=duration_s,
            speech_ratio=speech_ratio,
            avg_snr_db=avg_snr,
            noise_floor_rms=self._noise_floor_rms,
            robot_activity=robot_activity,
        )

    def _reject_from_segment(
        self,
        reason: str,
        segment: list[_BufferedFrame],
        duration_s: float,
        robot_activity: bool,
        speech_ratio: float | None = None,
        avg_snr: float | None = None,
    ) -> LocalRejectedTurn:
        """Build a rejected-turn payload from a segment."""
        if speech_ratio is None:
            speech_ratio = _ratio(item.stats.speech_like for item in segment)
        if avg_snr is None:
            avg_snr = _mean(item.stats.snr_db for item in segment)
        self._last_speech_ratio = speech_ratio
        return self._reject(reason, duration_s, speech_ratio, avg_snr, robot_activity)

    def _reject(
        self,
        reason: str,
        duration_s: float,
        speech_ratio: float,
        avg_snr: float,
        robot_activity: bool,
    ) -> LocalRejectedTurn:
        """Build a rejected-turn payload."""
        return LocalRejectedTurn(
            reason=reason,
            duration_s=duration_s,
            speech_ratio=speech_ratio,
            avg_snr_db=avg_snr,
            noise_floor_rms=self._noise_floor_rms,
            robot_activity=robot_activity,
        )

    def _analyze_frame(self, frame: NDArray[np.int16], *, robot_activity: bool) -> LocalFrameStats:
        """Return speech/noise metrics for one frame."""
        x = frame.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2) + 1e-6))
        noise_floor = max(1.0, self._noise_floor_rms)
        snr_db = float(20.0 * math.log10(max(rms, 1.0) / noise_floor))

        if x.size <= 1:
            zcr = 0.0
        else:
            signs = np.signbit(x)
            zcr = float(np.mean(signs[1:] != signs[:-1]))

        windowed = x * np.hanning(x.size)
        power = np.abs(np.fft.rfft(windowed)) ** 2
        freqs = np.fft.rfftfreq(x.size, d=1.0 / self.config.sample_rate)
        total_power = float(np.sum(power) + 1e-12)
        speech_mask = (freqs >= 300.0) & (freqs <= 3400.0)
        speech_band_ratio = float(np.sum(power[speech_mask]) / total_power)
        centroid = float(np.sum(freqs * power) / total_power)
        peak_dominance = float(np.max(power) / total_power) if power.size else 0.0
        flatness = float(np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power) + 1e-12))

        noise_class = self._classify_noise(
            rms=rms,
            speech_band_ratio=speech_band_ratio,
            spectral_flatness=flatness,
            spectral_centroid_hz=centroid,
            zero_crossing_rate=zcr,
            peak_dominance=peak_dominance,
        )
        speech_like = (
            rms >= self.config.min_frame_rms
            and snr_db >= self.config.min_snr_db
            and speech_band_ratio >= 0.35
            and 0.005 <= zcr <= 0.35
            and noise_class == "speech_like"
        )
        if speech_like:
            noise_class = "speech_like"

        return LocalFrameStats(
            rms=rms,
            noise_floor_rms=noise_floor,
            snr_db=snr_db,
            speech_band_ratio=speech_band_ratio,
            spectral_flatness=flatness,
            spectral_centroid_hz=centroid,
            zero_crossing_rate=zcr,
            peak_dominance=peak_dominance,
            speech_like=speech_like,
            robot_activity=robot_activity,
            noise_class=noise_class,
        )

    def _classify_noise(
        self,
        *,
        rms: float,
        speech_band_ratio: float,
        spectral_flatness: float,
        spectral_centroid_hz: float,
        zero_crossing_rate: float,
        peak_dominance: float,
    ) -> Literal["speech_like", "narrowband", "broadband", "low_band", "quiet"]:
        """Classify the dominant non-speech shape of a frame."""
        if rms < self.config.min_frame_rms:
            return "quiet"
        if peak_dominance >= 0.55 and spectral_flatness <= 0.2:
            return "narrowband"
        if spectral_flatness >= 0.68 and (speech_band_ratio < 0.72 or zero_crossing_rate >= 0.28):
            return "broadband"
        if speech_band_ratio < 0.3 or spectral_centroid_hz < 180.0 or spectral_centroid_hz > 4200.0:
            return "low_band"
        return "speech_like"

    def _update_noise_floor(self, rms: float) -> None:
        """Update adaptive noise floor from frames that did not pass speech gating."""
        alpha = min(1.0, max(0.0, self.config.noise_floor_alpha))
        clamped = max(1.0, min(float(rms), self._noise_floor_rms * 3.0))
        self._noise_floor_rms = (1.0 - alpha) * self._noise_floor_rms + alpha * clamped


def _ratio(values: Iterable[object]) -> float:
    """Return the true ratio for an iterable of booleans."""
    items = list(values)
    if not items:
        return 0.0
    return sum(1 for item in items if item) / len(items)


def _mean(values: Iterable[float]) -> float:
    """Return the mean for an iterable of floats."""
    items = [float(item) for item in values]
    return float(sum(items) / len(items)) if items else 0.0
