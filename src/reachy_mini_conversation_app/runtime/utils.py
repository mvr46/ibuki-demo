from __future__ import annotations
import sys
import logging
import argparse
import warnings
import threading
import subprocess
from typing import TYPE_CHECKING, Optional

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.runtime.transport import HARDWARE_PROFILES
from reachy_mini_conversation_app.vision.camera_worker import CameraWorker
from reachy_mini_conversation_app.vision.head_tracking import HeadTracker


if TYPE_CHECKING:
    from reachy_mini_conversation_app.vision.local_vision import VisionProcessor
    from reachy_mini_conversation_app.vision.head_tracking.speaker import SpatialAudioSource


class CameraVisionInitializationError(Exception):
    """Raised when camera or vision setup fails in an expected way."""


class LazyVisionProcessor:
    """Initialize the local VLM only when a camera tool actually needs it."""

    def __init__(self) -> None:
        """Create an unloaded local vision processor wrapper."""
        self._processor: VisionProcessor | None = None
        self._lock = threading.Lock()

    def _get_processor(self) -> VisionProcessor:
        if self._processor is not None:
            return self._processor
        with self._lock:
            if self._processor is None:
                from reachy_mini_conversation_app.vision.local_vision import initialize_vision_processor

                self._processor = initialize_vision_processor()
            return self._processor

    def process_image(self, image: object, prompt: str) -> str:
        """Analyze an image with the lazily initialized local VLM."""
        return self._get_processor().process_image(image, prompt)


def parse_args() -> tuple[argparse.Namespace, list]:  # type: ignore
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Reachy Mini Conversation App")
    parser.add_argument(
        "--head-tracker",
        choices=["yolo", "mediapipe"],
        default=None,
        help=(
            "Optional head-tracking backend: yolo uses a local face detector in a subprocess, "
            "mediapipe uses reachy_mini_toolbox in process. Disabled by default."
        ),
    )
    parser.add_argument("--no-camera", default=False, action="store_true", help="Disable camera usage")
    parser.add_argument(
        "--media-backend",
        choices=["auto", "default", "local", "webrtc", "no_media"],
        default="webrtc",
        help="Reachy Mini SDK media backend. Defaults to WebRTC over the direct wired-LAN setup; use no_media for headless runs without camera/audio hardware.",
    )
    parser.add_argument(
        "--local-vision",
        default=False,
        action="store_true",
        help="Use local vision model instead of the selected realtime backend vision",
    )
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--connection-mode",
        choices=["auto", "localhost_only", "network"],
        default="network",
        help=(
            "Reachy Mini SDK daemon connection mode. Defaults to network so media streams come from the robot daemon. "
            "Use localhost_only for local development daemons."
        ),
    )
    parser.add_argument(
        "--robot-host",
        type=str,
        default=None,
        help="Reachy Mini daemon hostname or IP address when using --connection-mode network.",
    )
    parser.add_argument(
        "--robot-port",
        type=int,
        default=None,
        help="Reachy Mini daemon port. Defaults to the SDK default when omitted.",
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default=None,
        help="[Optional] Robot name to target. Must match the daemon's --robot-name when connecting to a specific robot, mainly useful for development with multiple robots.",
    )
    parser.add_argument(
        "--hardware-profile",
        choices=list(HARDWARE_PROFILES),
        default="mac-mini-wired",
        help="Hardware optimization profile. Defaults to the direct Mac/Reachy wired LAN link at 10.42.0.2.",
    )
    return parser.parse_known_args()


def initialize_camera_and_vision(
    args: argparse.Namespace,
    current_robot: ReachyMini,
    spatial_audio_source: SpatialAudioSource | None = None,
    performance_diagnostics: object | None = None,
) -> tuple[CameraWorker | None, VisionProcessor | None]:
    """Initialize camera capture, optional head tracking, and optional local vision."""
    camera_worker: Optional[CameraWorker] = None
    head_tracker: HeadTracker | None = None
    vision_processor: Optional[VisionProcessor] = None

    if not args.no_camera:
        if args.head_tracker is not None:
            try:
                if args.head_tracker == "yolo":
                    from reachy_mini_conversation_app.vision.head_tracking.yolo_process import (
                        PROCESS_START_TIMEOUT,
                        YoloHeadTrackerProcess,
                    )

                    logging.getLogger(__name__).info(
                        "Starting yolo head tracker subprocess. First run can take up to %.0fs while the model loads.",
                        PROCESS_START_TIMEOUT,
                    )
                    head_tracker = YoloHeadTrackerProcess()
                    logging.getLogger(__name__).info("Yolo head tracker subprocess ready")
                else:
                    from reachy_mini_conversation_app.vision.head_tracking.mediapipe import (
                        MediapipeHeadTracker,
                    )

                    head_tracker = MediapipeHeadTracker()
                    logging.getLogger(__name__).info("Using mediapipe head tracker in process")
            except Exception as e:
                raise CameraVisionInitializationError(
                    f"Failed to initialize {args.head_tracker} head tracker: {e}",
                ) from e

        if spatial_audio_source is None and performance_diagnostics is None:
            camera_worker = CameraWorker(current_robot, head_tracker)
        else:
            camera_worker = CameraWorker(
                current_robot,
                head_tracker,
                spatial_audio_source=spatial_audio_source,
                performance_diagnostics=performance_diagnostics,
            )

        if args.local_vision:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from reachy_mini_conversation_app.vision.local_vision import VisionProcessor",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode < 0:
                raise CameraVisionInitializationError(
                    "Local vision import crashed on this machine. "
                    "Run without --local-vision or install compatible dependencies.",
                )
            try:
                __import__("reachy_mini_conversation_app.vision.local_vision", fromlist=["initialize_vision_processor"])
            except ImportError as e:
                raise CameraVisionInitializationError(
                    "To use --local-vision, please install the extra dependencies: pip install '.[local_vision]'",
                ) from e

            vision_processor = LazyVisionProcessor()
            logging.getLogger(__name__).info("Local vision enabled; model loading is deferred until first camera-tool use.")
        else:
            logging.getLogger(__name__).info(
                "Using the configured backend vision analyzer. Use --local-vision for Transformers SmolVLM processing.",
            )

    return camera_worker, vision_processor


def setup_logger(debug: bool) -> logging.Logger:
    """Setups the logger."""
    log_level = "DEBUG" if debug else "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Suppress WebRTC warnings
    warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

    # Tame third-party noise (looser in DEBUG)
    if log_level == "DEBUG":
        logging.getLogger("aiortc").setLevel(logging.INFO)
        logging.getLogger("aioice").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("websockets").setLevel(logging.INFO)
    else:
        logging.getLogger("aiortc").setLevel(logging.ERROR)
        logging.getLogger("aioice").setLevel(logging.WARNING)
    return logger


def log_connection_troubleshooting(logger: logging.Logger, robot_name: Optional[str]) -> None:
    """Log troubleshooting steps for connection issues."""
    logger.error("Troubleshooting steps:")
    logger.error("  1. Verify reachy-mini-daemon is running")

    if robot_name is not None:
        logger.error(f"  2. Daemon must be started with: --robot-name '{robot_name}'")
    else:
        logger.error("  2. If daemon uses --robot-name, add the same flag here: --robot-name <name>")

    logger.error("  3. For wireless: check network connectivity")
    logger.error("  4. Review daemon logs")
    logger.error("  5. Restart the daemon")
