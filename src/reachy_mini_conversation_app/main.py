"""Entrypoint for the Reachy Mini conversation app."""

import os
import sys
import time
import asyncio
import argparse
import threading
from typing import Optional
from pathlib import Path

from fastapi import FastAPI

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini_conversation_app.runtime.utils import (
    CameraVisionInitializationError,
    parse_args,
    setup_logger,
    initialize_camera_and_vision,
    log_connection_troubleshooting,
)


def _disable_broken_gstreamer_python_plugin() -> None:
    """Remove the optional libgstpython plugin from GStreamer scan paths on macOS uv Python."""
    plugin_suffix = os.path.join("gstreamer_python", "lib", "gstreamer-1.0")
    for env_name in ("GST_PLUGIN_PATH_1_0", "GST_PLUGIN_SYSTEM_PATH_1_0", "GST_PLUGIN_PATH", "GST_PLUGIN_SYSTEM_PATH"):
        value = os.environ.get(env_name)
        if not value:
            continue
        paths = [path for path in value.split(os.pathsep) if not path.endswith(plugin_suffix)]
        os.environ[env_name] = os.pathsep.join(paths)


def _install_network_media_host_fallback() -> None:
    """Use the connected robot host for WebRTC media when daemon status has no wlan_ip."""
    if getattr(ReachyMini, "_conversation_app_media_host_fallback", False):
        return

    from reachy_mini.daemon.utils import is_local_camera_available
    from reachy_mini.media.media_manager import MediaBackend, MediaManager
    from reachy_mini.media.camera_constants import get_camera_specs_by_name
    from reachy_mini_conversation_app.runtime.config import config
    from reachy_mini_conversation_app.runtime.transport import resolve_transport

    def _configure_mediamanager_with_host_fallback(
        self: ReachyMini,
        media_backend: str,
        log_level: str,
    ) -> MediaManager:
        daemon_status = self.client.get_status()
        self._warn_if_daemon_version_mismatch(daemon_status)

        if getattr(daemon_status, "media_released", False) and media_backend.lower() != "no_media":
            self.logger.info("Daemon media is released; asking daemon to re-acquire camera/audio hardware.")
            if self.client.acquire_media():
                daemon_status = self.client.get_status()
            else:
                self.logger.error("Failed to re-acquire media on daemon.")

        specs_name = getattr(daemon_status, "camera_specs_name", "")
        camera_specs = get_camera_specs_by_name(specs_name) if specs_name else None

        if media_backend.lower() == "no_media":
            self.logger.info("No media backend requested by user.")
            if not getattr(daemon_status, "no_media", False) and not self._media_released:
                self.release_media()
            media_backend_enum = MediaBackend.NO_MEDIA
        elif getattr(daemon_status, "no_media", False):
            self.logger.info("Daemon reports no_media=True; skipping media initialisation.")
            media_backend_enum = MediaBackend.NO_MEDIA
        elif media_backend.lower() in ("default", "auto"):
            if self.connection_mode == "localhost_only" and is_local_camera_available():
                self.logger.info("Auto-detected local IPC endpoint. Using LOCAL backend.")
                media_backend_enum = MediaBackend.LOCAL
            else:
                self.logger.info("No local IPC endpoint. Using WebRTC backend for streaming.")
                media_backend_enum = MediaBackend.WEBRTC
        else:
            try:
                media_backend_enum = MediaBackend(media_backend.lower())
            except ValueError:
                self.logger.warning("Unknown media backend %r, falling back to auto-detect.", media_backend)
                if self.connection_mode == "localhost_only" and is_local_camera_available():
                    media_backend_enum = MediaBackend.LOCAL
                else:
                    media_backend_enum = MediaBackend.WEBRTC

        transport_selection = resolve_transport(
            control_host=getattr(self.client, "host", None),
            daemon_wlan_ip=getattr(daemon_status, "wlan_ip", None),
            media_host_override=getattr(config, "REACHY_MEDIA_HOST", None),
            hardware_profile=getattr(ReachyMini, "_conversation_app_hardware_profile", "auto"),
        )
        self._conversation_app_transport_selection = transport_selection
        signalling_host = transport_selection.media_host or "localhost"
        if self.connection_mode == "network" and signalling_host == "localhost":
            self.logger.warning(
                "Daemon status did not provide wlan_ip; falling back to %s for media signaling.",
                signalling_host,
            )
        else:
            self.logger.info(
                "Reachy media signaling host resolved to %s (source=%s, control=%s, wlan=%s)",
                signalling_host,
                transport_selection.media_host_source,
                transport_selection.control_host,
                transport_selection.daemon_wlan_ip,
            )

        if media_backend_enum == MediaBackend.WEBRTC:
            from reachy_mini.media import audio_base
            from reachy_mini.media.webrtc_utils import get_producer_list

            if self.connection_mode == "network":

                class RemoteWebRTCAudioDoA:
                    def get_DoA(self) -> None:
                        return None

                    def close(self) -> None:
                        return None

                audio_base.AudioDoA = RemoteWebRTCAudioDoA

            producer_ready = False
            for _ in range(10):
                try:
                    producers = get_producer_list(signalling_host, 8443)
                    producer_ready = any(meta.get("name") == "reachymini" for meta in producers.values())
                    if producer_ready:
                        break
                except Exception:
                    pass
                time.sleep(0.5)
            if not producer_ready:
                self.logger.warning("WebRTC producer 'reachymini' was not visible before media startup.")

        return MediaManager(
            backend=media_backend_enum,
            log_level=log_level,
            signalling_host=signalling_host,
            camera_specs=camera_specs,
            daemon_url=self._daemon_http_url,
        )

    ReachyMini._configure_mediamanager = _configure_mediamanager_with_host_fallback
    ReachyMini._conversation_app_media_host_fallback = True


def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args, _ = parse_args()
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    # Putting these dependencies here makes the dashboard faster to load when the conversation app is installed
    from reachy_mini_conversation_app.motion.moves import MovementManager
    from reachy_mini_conversation_app.runtime.config import (
        HF_BACKEND,
        LOCAL_BACKEND,
        config,
        get_backend_label,
        get_hf_connection_selection,
        refresh_runtime_config_from_env,
    )
    from reachy_mini_conversation_app.runtime.transport import measure_http_rtt_ms, default_robot_host_for_profile
    from reachy_mini_conversation_app.runtime.diagnostics import PerformanceDiagnostics
    from reachy_mini_conversation_app.runtime.startup_settings import (
        StartupSettings,
        load_startup_settings_into_runtime,
    )

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App")
    diagnostics = PerformanceDiagnostics()
    startup_settings = StartupSettings()

    if instance_path is not None:
        try:
            from dotenv import load_dotenv

            env_path = Path(instance_path) / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=str(env_path), override=True)
                refresh_runtime_config_from_env()
                logger.info("Loaded instance configuration from %s", env_path)
        except Exception as e:
            logger.warning("Failed to load instance configuration: %s", e)

        try:
            startup_settings = load_startup_settings_into_runtime(instance_path)
        except Exception as e:
            logger.warning("Failed to load startup settings: %s", e)

    if config.BACKEND_PROVIDER == HF_BACKEND:
        logger.info(
            "Configured backend provider: %s (%s), connection mode: %s",
            config.BACKEND_PROVIDER,
            get_backend_label(config.BACKEND_PROVIDER),
            get_hf_connection_selection().mode,
        )
    else:
        logger.info(
            "Configured backend provider: %s (%s), model: %s",
            config.BACKEND_PROVIDER,
            get_backend_label(config.BACKEND_PROVIDER),
            config.MODEL_NAME,
        )

    from reachy_mini_conversation_app.runtime.console import LocalStream
    from reachy_mini_conversation_app.tools.core_tools import ToolRegistry, ToolDependencies
    from reachy_mini_conversation_app.audio.head_wobbler import HeadWobbler
    from reachy_mini_conversation_app.backends.interface import ConversationHandler

    if args.media_backend == "no_media":
        if not args.no_camera:
            logger.warning("Media backend no_media selected; disabling camera capture.")
            args.no_camera = True
        if args.head_tracker is not None:
            logger.warning("Head tracking disabled: --media-backend no_media was selected.")
            args.head_tracker = None
        if args.local_vision:
            logger.warning("Local vision disabled: --media-backend no_media was selected.")
            args.local_vision = False

    if args.no_camera and args.head_tracker is not None:
        logger.warning("Head tracking disabled: --no-camera flag is set. Remove --no-camera to enable head tracking.")

    if robot is None:
        try:
            _disable_broken_gstreamer_python_plugin()
            ReachyMini._conversation_app_hardware_profile = args.hardware_profile
            _install_network_media_host_fallback()
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["robot_name"] = args.robot_name
            robot_kwargs["connection_mode"] = args.connection_mode
            if args.robot_host is None:
                selected_host = default_robot_host_for_profile(
                    args.hardware_profile,
                    port=args.robot_port or 8000,
                )
                if selected_host is not None:
                    args.robot_host = selected_host
                    logger.info("Hardware profile %s selected robot host %s", args.hardware_profile, selected_host)
            if args.robot_host is not None:
                robot_kwargs["host"] = args.robot_host
            if args.robot_port is not None:
                robot_kwargs["port"] = args.robot_port
            if args.media_backend != "auto":
                robot_kwargs["media_backend"] = args.media_backend

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)
            transport_selection = getattr(robot, "_conversation_app_transport_selection", None)
            if transport_selection is not None:
                diagnostics.set_transport(
                    control_host=transport_selection.control_host,
                    media_host=transport_selection.media_host,
                    daemon_wlan_ip=transport_selection.daemon_wlan_ip,
                    media_host_source=transport_selection.media_host_source,
                    hardware_profile=transport_selection.hardware_profile,
                )

        except TimeoutError as e:
            logger.error(f"Connection timeout: Failed to connect to Reachy Mini daemon. Details: {e}")
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(f"Connection failed: Unable to establish connection to Reachy Mini. Details: {e}")
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(f"Unexpected error during robot initialization: {type(e).__name__}: {e}")
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    status = robot.client.get_status()
    diagnostics.set_daemon(
        rtt_ms=measure_http_rtt_ms(f"http://{getattr(robot.client, 'host', 'localhost')}:{getattr(robot.client, 'port', 8000)}/api/daemon/status"),
        state=(status.get("state") if isinstance(status, dict) else getattr(status, "state", None)),
    )
    from reachy_mini_conversation_app.vision.speaker_attribution import SpeakerAttributionWorker

    spatial_audio_source = None

    try:
        camera_worker, vision_processor = initialize_camera_and_vision(
            args,
            robot,
            spatial_audio_source=spatial_audio_source,
            performance_diagnostics=diagnostics,
        )
    except CameraVisionInitializationError as e:
        logger.error("Failed to initialize camera/vision: %s", e)
        sys.exit(1)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )

    head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)

    face_identity_worker = None
    if camera_worker is not None and getattr(camera_worker, "head_tracker", None) is not None:
        try:
            from reachy_mini_conversation_app.vision.face_identity import (
                build_default_face_identity_service,
                build_detection_only_face_identity_service,
            )
            from reachy_mini_conversation_app.vision.face_identity_worker import FaceIdentifierWorker

            try:
                face_identity_service = build_default_face_identity_service()
                require_embedding_to_confirm = True
                logger.info("Face recognition worker initialized")
            except Exception as e:
                logger.warning("Face recognition unavailable: %s", e)
                logger.warning("Falling back to detection-only face boxes; naming faces requires InsightFace models.")
                face_identity_service = build_detection_only_face_identity_service()
                require_embedding_to_confirm = False

            face_identity_worker = FaceIdentifierWorker(
                camera_worker,
                face_identity_service.identifier,
                require_embedding_to_confirm=require_embedding_to_confirm,
            )
            set_face_identity_worker = getattr(camera_worker, "set_face_identity_worker", None)
            if callable(set_face_identity_worker):
                set_face_identity_worker(face_identity_worker)
            if not require_embedding_to_confirm:
                logger.info("Face detection-only worker initialized")
        except Exception as e:
            logger.warning("Face identity worker unavailable: %s", e)

    speaker_attribution_worker = SpeakerAttributionWorker(
        spatial_audio_source=spatial_audio_source,
        face_identity_worker=face_identity_worker,
        assistant_state_source=camera_worker,
    )

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        face_identity_worker=face_identity_worker,
        spatial_audio_source=spatial_audio_source,
        speaker_attribution_worker=speaker_attribution_worker,
        vision_processor=vision_processor,
        vision_analyzer=None,
        head_wobbler=head_wobbler,
        performance_diagnostics=diagnostics,
    )
    deps.tool_registry = ToolRegistry.from_active_profile()
    if config.BACKEND_PROVIDER == LOCAL_BACKEND:
        from reachy_mini_conversation_app.vision.analyzers import build_default_vision_analyzer

        deps.vision_analyzer = build_default_vision_analyzer(
            vision_processor,
            diagnostics=diagnostics,
        )
    from reachy_mini_conversation_app.backends.factory import create_conversation_handler

    handler: ConversationHandler = create_conversation_handler(
        deps,
        instance_path=instance_path,
        startup_voice=startup_settings.voice,
    )

    stream_manager = LocalStream(
        handler,
        robot,
        settings_app=settings_app,
        instance_path=instance_path,
    )

    # Each async service → its own thread/loop
    movement_manager.start()
    head_wobbler.start()
    if camera_worker:
        camera_worker.start()
    if face_identity_worker:
        face_identity_worker.start()

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        movement_manager.stop()
        head_wobbler.stop()
        if face_identity_worker:
            face_identity_worker.stop()
        if camera_worker:
            camera_worker.stop()

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")


class ReachyMiniConversationApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini conversation app."""
        asyncio.set_event_loop(asyncio.new_event_loop())

        args, _ = parse_args()

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = ReachyMiniConversationApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
