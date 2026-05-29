"""Transport resolution for wired and legacy Reachy Mini network paths."""

from __future__ import annotations
import time
import socket
import logging
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import urlopen

from reachy_mini_conversation_app.runtime.config import DEFAULT_WIRED_REACHY_HOST


logger = logging.getLogger(__name__)

HARDWARE_PROFILE_AUTO = "auto"
HARDWARE_PROFILE_MAC_MINI_WIRED = "mac-mini-wired"
HARDWARE_PROFILE_LEGACY = "legacy"
HARDWARE_PROFILES = (
    HARDWARE_PROFILE_AUTO,
    HARDWARE_PROFILE_MAC_MINI_WIRED,
    HARDWARE_PROFILE_LEGACY,
)


@dataclass(frozen=True)
class TransportSelection:
    """Resolved hosts used for daemon control and media signaling."""

    control_host: str | None
    media_host: str | None
    daemon_wlan_ip: str | None
    media_host_source: str
    hardware_profile: str


def probe_host(host: str, port: int = 8000, *, timeout_seconds: float = 0.2) -> bool:
    """Return whether a TCP host:port can be reached quickly."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def measure_http_rtt_ms(url: str, *, timeout_seconds: float = 1.0) -> float | None:
    """Return a best-effort HTTP round-trip time in milliseconds."""
    start = time.perf_counter()
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            response.read(1)
    except (OSError, URLError, TimeoutError):
        return None
    return (time.perf_counter() - start) * 1000


def default_robot_host_for_profile(
    hardware_profile: str,
    *,
    port: int = 8000,
    wired_host: str = DEFAULT_WIRED_REACHY_HOST,
) -> str | None:
    """Return an auto-selected robot host for an optimized hardware profile."""
    if hardware_profile == HARDWARE_PROFILE_LEGACY:
        return None
    if hardware_profile == HARDWARE_PROFILE_MAC_MINI_WIRED:
        return wired_host
    if hardware_profile == HARDWARE_PROFILE_AUTO and probe_host(wired_host, port):
        return wired_host
    return None


def resolve_transport(
    *,
    control_host: str | None,
    daemon_wlan_ip: str | None,
    media_host_override: str | None,
    hardware_profile: str,
) -> TransportSelection:
    """Resolve the media host without letting daemon wlan_ip override explicit wired control."""
    cleaned_override = (media_host_override or "").strip() or None
    cleaned_control = (control_host or "").strip() or None
    cleaned_wlan = (daemon_wlan_ip or "").strip() or None

    if cleaned_override:
        media_host = cleaned_override
        media_host_source = "REACHY_MEDIA_HOST"
    elif cleaned_control and hardware_profile != HARDWARE_PROFILE_LEGACY:
        media_host = cleaned_control
        media_host_source = "control_host"
    elif cleaned_wlan:
        media_host = cleaned_wlan
        media_host_source = "daemon_wlan_ip"
    elif cleaned_control:
        media_host = cleaned_control
        media_host_source = "control_host"
    else:
        media_host = "localhost"
        media_host_source = "fallback"

    return TransportSelection(
        control_host=cleaned_control,
        media_host=media_host,
        daemon_wlan_ip=cleaned_wlan,
        media_host_source=media_host_source,
        hardware_profile=hardware_profile,
    )

