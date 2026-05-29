"""Tests for Reachy transport host resolution."""

from reachy_mini_conversation_app.runtime.transport import (
    HARDWARE_PROFILE_AUTO,
    HARDWARE_PROFILE_LEGACY,
    HARDWARE_PROFILE_MAC_MINI_WIRED,
    resolve_transport,
    default_robot_host_for_profile,
)


def test_media_override_wins_over_control_and_wlan() -> None:
    """REACHY_MEDIA_HOST should be the strongest media-routing knob."""
    result = resolve_transport(
        control_host="10.42.0.2",
        daemon_wlan_ip="192.168.0.149",
        media_host_override="10.42.0.99",
        hardware_profile=HARDWARE_PROFILE_AUTO,
    )

    assert result.media_host == "10.42.0.99"
    assert result.media_host_source == "REACHY_MEDIA_HOST"


def test_explicit_control_host_wins_over_wlan_for_optimized_profiles() -> None:
    """The wired control host should also drive media in optimized profiles."""
    result = resolve_transport(
        control_host="10.42.0.2",
        daemon_wlan_ip="192.168.0.149",
        media_host_override=None,
        hardware_profile=HARDWARE_PROFILE_AUTO,
    )

    assert result.media_host == "10.42.0.2"
    assert result.media_host_source == "control_host"


def test_legacy_profile_keeps_daemon_wlan_preference() -> None:
    """Legacy mode preserves the previous daemon wlan_ip behavior."""
    result = resolve_transport(
        control_host="10.42.0.2",
        daemon_wlan_ip="192.168.0.149",
        media_host_override=None,
        hardware_profile=HARDWARE_PROFILE_LEGACY,
    )

    assert result.media_host == "192.168.0.149"
    assert result.media_host_source == "daemon_wlan_ip"


def test_hardware_profile_defaults_to_wired_host() -> None:
    """Optimized profiles should use the wired Reachy host by default."""
    assert default_robot_host_for_profile(HARDWARE_PROFILE_AUTO) == "10.42.0.2"
    assert default_robot_host_for_profile(HARDWARE_PROFILE_MAC_MINI_WIRED) == "10.42.0.2"
    assert default_robot_host_for_profile(HARDWARE_PROFILE_LEGACY) is None
