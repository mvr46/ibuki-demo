"""Tests for shared spatial-audio history."""

from __future__ import annotations
import math

import pytest

from reachy_mini_conversation_app.vision.head_tracking.speaker import DaemonDoAPoller, target_from_doa


def test_spatial_audio_source_latest_and_window_signs() -> None:
    """Spatial audio history should expose latest and segment-window samples."""
    source = DaemonDoAPoller("localhost", 1, history_seconds=10.0)

    source._record_target(target_from_doa(math.pi, speech_detected=True), 10.0)
    source._record_target(target_from_doa(math.pi / 2.0, speech_detected=False), 11.0)
    source._record_target(target_from_doa(0.0, speech_detected=True), 12.0)

    latest, latest_at = source.get_latest()
    assert latest is not None
    assert latest_at == 12.0
    assert latest.x_offset == pytest.approx(1.0)

    samples = source.window(9.5, 11.5)
    assert [sample.timestamp for sample in samples] == [10.0, 11.0]
    assert samples[0].azimuth_deg == pytest.approx(-90.0)
    assert samples[1].azimuth_deg == pytest.approx(0.0)


def test_spatial_audio_source_prunes_stale_samples() -> None:
    """Old samples should fall out of the bounded history while latest remains fresh."""
    source = DaemonDoAPoller("localhost", 1, history_seconds=1.0)

    assert source.get_latest() == (None, None)
    assert source.window(0.0, 1.0) == ()

    source._record_target(target_from_doa(math.pi), 1.0)
    source._record_target(target_from_doa(0.0), 3.0)

    assert source.window(0.5, 1.5) == ()
    latest, latest_at = source.get_latest()
    assert latest is not None
    assert latest_at == 3.0
