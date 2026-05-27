"""Entrypoint for the Reachy Mini conversation app."""

import os
import sys
import time
import asyncio
import argparse
import threading
from typing import Any, Dict, List, Optional
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream
from gradio.utils import get_space

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini_conversation_app.utils import (
    CameraVisionInitializationError,
    parse_args,
    setup_logger,
    initialize_camera_and_vision,
    log_connection_troubleshooting,
)


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


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

        signalling_host = getattr(daemon_status, "wlan_ip", None) or getattr(self.client, "host", None) or "localhost"
        if self.connection_mode == "network" and signalling_host == "localhost":
            self.logger.warning(
                "Daemon status did not provide wlan_ip; falling back to %s for media signaling.",
                signalling_host,
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

    ReachyMini._configure_mediamanager = _configure_mediamanager_with_host_fallback  # type: ignore[method-assign]
    ReachyMini._conversation_app_media_host_fallback = True  # type: ignore[attr-defined]


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
    from reachy_mini_conversation_app.moves import MovementManager
    from reachy_mini_conversation_app.config import (
        HF_BACKEND,
        GEMINI_BACKEND,
        OPENAI_BACKEND,
        HF_LOCAL_CONNECTION_MODE,
        config,
        is_gemini_model,
        get_backend_label,
        get_hf_connection_selection,
        refresh_runtime_config_from_env,
    )
    from reachy_mini_conversation_app.startup_settings import (
        StartupSettings,
        load_startup_settings_into_runtime,
    )

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App")
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

    from reachy_mini_conversation_app.console import LocalStream
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
    from reachy_mini_conversation_app.audio.head_wobbler import HeadWobbler

    if args.media_backend == "no_media":
        if not args.no_camera:
            logger.warning("Media backend no_media selected; disabling camera capture.")
            args.no_camera = True
        if not args.gradio:
            logger.info("Media backend no_media selected; enabling Gradio for browser audio.")
            args.gradio = True
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
            _install_network_media_host_fallback()
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["robot_name"] = args.robot_name
            robot_kwargs["connection_mode"] = args.connection_mode
            if args.robot_host is not None:
                robot_kwargs["host"] = args.robot_host
            if args.robot_port is not None:
                robot_kwargs["port"] = args.robot_port
            if args.media_backend != "auto":
                robot_kwargs["media_backend"] = args.media_backend

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)

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

    # Auto-enable Gradio in simulation mode (both MuJoCo for daemon and mockup-sim for desktop app)
    status = robot.client.get_status()
    if isinstance(status, dict):
        simulation_enabled = status.get("simulation_enabled", False)
        mockup_sim_enabled = status.get("mockup_sim_enabled", False)
    else:
        simulation_enabled = getattr(status, "simulation_enabled", False)
        mockup_sim_enabled = getattr(status, "mockup_sim_enabled", False)

    is_simulation = simulation_enabled or mockup_sim_enabled

    if is_simulation and not args.gradio:
        logger.info("Simulation mode detected. Automatically enabling gradio flag.")
        args.gradio = True

    from reachy_mini_conversation_app.speaker_attribution import SpeakerAttributionWorker
    from reachy_mini_conversation_app.vision.head_tracking.speaker import build_daemon_spatial_audio_source

    spatial_audio_source = build_daemon_spatial_audio_source(robot)

    try:
        camera_worker, vision_processor = initialize_camera_and_vision(
            args,
            robot,
            spatial_audio_source=spatial_audio_source,
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
            from reachy_mini_conversation_app.face_identity_worker import FaceIdentifierWorker
            from reachy_mini_conversation_app.vision.face_identity import build_default_face_identity_service

            face_identity_service = build_default_face_identity_service()
            face_identity_worker = FaceIdentifierWorker(camera_worker, face_identity_service.identifier)
            set_face_identity_worker = getattr(camera_worker, "set_face_identity_worker", None)
            if callable(set_face_identity_worker):
                set_face_identity_worker(face_identity_worker)
            logger.info("Face recognition worker initialized")
        except Exception as e:
            logger.warning("Face recognition worker unavailable: %s", e)

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
        head_wobbler=head_wobbler,
    )
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"Current file absolute path: {current_file_path}")
    chatbot = gr.Chatbot(
        type="messages",
        resizable=True,
        avatar_images=(
            os.path.join(current_file_path, "images", "user_avatar.png"),
            os.path.join(current_file_path, "images", "reachymini_avatar.png"),
        ),
    )
    logger.debug(f"Chatbot avatar images: {chatbot.avatar_images}")

    if is_gemini_model():
        from reachy_mini_conversation_app.gemini_live import GeminiLiveHandler

        logger.info(
            "Using %s via GeminiLiveHandler",
            get_backend_label(config.BACKEND_PROVIDER),
        )
        handler = GeminiLiveHandler(
            deps,
            gradio_mode=args.gradio,
            instance_path=instance_path,
            startup_voice=startup_settings.voice,
        )
    elif config.BACKEND_PROVIDER == HF_BACKEND:
        from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler

        hf_connection_selection = get_hf_connection_selection()
        transport_label = (
            "Hugging Face direct websocket"
            if hf_connection_selection.mode == HF_LOCAL_CONNECTION_MODE and hf_connection_selection.has_target
            else "Hugging Face session proxy"
        )
        logger.info(
            "Using %s via Hugging Face realtime handler (%s)",
            get_backend_label(config.BACKEND_PROVIDER),
            transport_label,
        )
        handler = HuggingFaceRealtimeHandler(
            deps,
            gradio_mode=args.gradio,
            instance_path=instance_path,
            startup_voice=startup_settings.voice,
        )  # type: ignore[assignment]
    else:
        from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler

        logger.info(
            "Using %s via OpenAI realtime handler (OpenAI Realtime API)",
            get_backend_label(config.BACKEND_PROVIDER),
        )
        handler = OpenaiRealtimeHandler(
            deps,
            gradio_mode=args.gradio,
            instance_path=instance_path,
            startup_voice=startup_settings.voice,
        )  # type: ignore[assignment]

    stream_manager: gr.Blocks | LocalStream | None = None

    if args.gradio:
        from reachy_mini_conversation_app.gradio_personality import PersonalityUI

        personality_ui = PersonalityUI()
        personality_ui.create_components()
        additional_inputs: list[Any] = [chatbot, *personality_ui.additional_inputs_ordered()]

        if config.BACKEND_PROVIDER in {OPENAI_BACKEND, GEMINI_BACKEND}:
            uses_gemini_backend = is_gemini_model()
            api_key_textbox = gr.Textbox(
                label="GEMINI_API_KEY" if uses_gemini_backend else "OPENAI API Key",
                type="password",
                value=(os.getenv("GEMINI_API_KEY") if uses_gemini_backend else os.getenv("OPENAI_API_KEY"))
                if not get_space()
                else "",
            )
            additional_inputs.insert(1, api_key_textbox)

        stream = Stream(
            handler=handler,
            mode="send-receive",
            modality="audio",
            additional_inputs=additional_inputs,
            additional_outputs=[chatbot],
            additional_outputs_handler=update_chatbot,
            ui_args={"title": "Talk with Reachy Mini"},
        )
        stream_manager = stream.ui
        if not settings_app:
            app = FastAPI()
        else:
            app = settings_app

        personality_ui.wire_events(handler, stream_manager)

        app = gr.mount_gradio_app(app, stream.ui, path="/")
    else:
        # In headless mode, wire settings_app + instance_path to console LocalStream
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
    elif spatial_audio_source:
        spatial_audio_start = getattr(spatial_audio_source, "start", None)
        if callable(spatial_audio_start):
            spatial_audio_start()
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
        elif spatial_audio_source:
            spatial_audio_stop = getattr(spatial_audio_source, "stop", None)
            if callable(spatial_audio_stop):
                spatial_audio_stop()

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
