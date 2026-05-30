"""Tests for the headless console stream."""

import asyncio
import threading
from types import SimpleNamespace
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from numpy.typing import NDArray
from fastapi.testclient import TestClient

from reachy_mini.media.media_manager import MediaBackend
from reachy_mini_conversation_app.profiles.store import ProfileStore
from reachy_mini_conversation_app.runtime.config import LOCAL_BACKEND, config
from reachy_mini_conversation_app.profiles.routes import mount_personality_routes
from reachy_mini_conversation_app.runtime.console import LOCAL_PLAYER_BACKEND, LocalStream
from reachy_mini_conversation_app.runtime.diagnostics import PerformanceDiagnostics
from reachy_mini_conversation_app.runtime.startup_settings import (
    StartupSettings,
    load_startup_settings_into_runtime,
)


def test_clear_audio_queue_prefers_clear_player_when_available() -> None:
    """Local GStreamer audio should use the lower-level player flush when available."""
    handler = MagicMock()
    audio = SimpleNamespace(
        clear_player=MagicMock(),
        clear_output_buffer=MagicMock(),
    )
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=LOCAL_PLAYER_BACKEND))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_player.assert_called_once()
    audio.clear_output_buffer.assert_not_called()
    assert isinstance(handler.output_queue, asyncio.Queue)
    assert handler.output_queue.empty()


def test_clear_audio_queue_uses_output_buffer_for_webrtc() -> None:
    """WebRTC audio should flush queued playback via the output buffer API."""
    handler = MagicMock()
    audio = SimpleNamespace(
        clear_player=MagicMock(),
        clear_output_buffer=MagicMock(),
    )
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=MediaBackend.WEBRTC))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_output_buffer.assert_called_once()
    audio.clear_player.assert_not_called()
    assert isinstance(handler.output_queue, asyncio.Queue)
    assert handler.output_queue.empty()


def test_clear_audio_queue_falls_back_when_backend_is_unknown() -> None:
    """Unknown backends should still best-effort flush pending playback."""
    handler = MagicMock()
    audio = SimpleNamespace(clear_output_buffer=MagicMock())
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=None))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_output_buffer.assert_called_once()
    assert isinstance(handler.output_queue, asyncio.Queue)
    assert handler.output_queue.empty()


def test_refresh_performance_health_marks_doa_deprecated_without_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dashboard health should not poll /api/state/doa after DoA deprecation."""
    diagnostics = PerformanceDiagnostics()
    handler = MagicMock()
    handler.deps = SimpleNamespace(performance_diagnostics=diagnostics)
    robot = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1", port=8000), media=SimpleNamespace())
    stream = LocalStream(handler, robot)
    fetched_urls: list[str] = []

    def fake_fetch(url: str, *, timeout: float) -> dict[str, object]:
        fetched_urls.append(url)
        if url.endswith("/api/daemon/status"):
            return {"state": "running"}
        if url.endswith("/api/media/status"):
            return {"available": True, "released": False}
        if url.endswith("/api/state/full"):
            return {"timestamp": 1}
        return {}

    monkeypatch.setattr("reachy_mini_conversation_app.runtime.console.measure_http_rtt_ms", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr("reachy_mini_conversation_app.runtime.console.probe_host", lambda *args, **kwargs: True)
    monkeypatch.setattr("reachy_mini_conversation_app.runtime.console._fetch_json_dict", fake_fetch)

    stream._refresh_performance_health()

    snapshot = diagnostics.snapshot()
    assert all("/api/state/doa" not in url for url in fetched_urls)
    assert snapshot["health_checks"]["doa_status"] == "disabled/deprecated"


def test_local_backend_requires_ready_piper_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local backend readiness should fail when Piper or PIPER_VOICE is not ready."""
    handler = MagicMock()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(handler, robot)

    monkeypatch.setattr("reachy_mini_conversation_app.backends.local_tts.shutil.which", lambda name: "/usr/bin/piper")
    monkeypatch.setattr(config, "PIPER_VOICE", None)

    assert stream._has_required_key(LOCAL_BACKEND) is False
    assert stream._requirement_name(LOCAL_BACKEND) == "PIPER_VOICE"


def test_ready_uses_explicit_profile_tool_registry(tmp_path: Path) -> None:
    """The dashboard ready probe should not depend on the legacy global tool flag."""
    app = FastAPI()
    handler = MagicMock()
    handler.deps = SimpleNamespace(tool_registry=SimpleNamespace(tools={"dance": object()}))
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(handler, robot, settings_app=app, instance_path=str(tmp_path))

    stream._init_settings_ui_if_needed()

    assert TestClient(app).get("/ready").json() == {"ready": True}


@pytest.mark.asyncio
async def test_play_loop_feeds_head_wobbler_with_local_playback_delay() -> None:
    """Local playback should drive speech wobble using the queued player delay."""
    head_wobbler = MagicMock()
    chunk = np.array([1, -2, 3, -4], dtype=np.int16)

    class Handler:
        def __init__(self) -> None:
            self.deps = SimpleNamespace(head_wobbler=head_wobbler)
            self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
            self._emitted = False

        async def emit(self) -> tuple[int, NDArray[np.int16]] | None:
            if not self._emitted:
                self._emitted = True
                return (24000, chunk.copy())
            return None

    audio = SimpleNamespace(
        _playback_next_pts_ns=1_500_000_000,
        _get_playback_running_time_ns=lambda: 500_000_000,
    )
    media = SimpleNamespace(
        audio=audio,
        backend=LOCAL_PLAYER_BACKEND,
        get_output_audio_samplerate=lambda: 24000,
        push_audio_sample=MagicMock(),
    )
    robot = SimpleNamespace(media=media)
    handler = Handler()
    stream = LocalStream(handler, robot)

    async def stop_soon() -> None:
        await asyncio.sleep(0.01)
        stream._stop_event.set()

    stopper = asyncio.create_task(stop_soon())
    try:
        await asyncio.wait_for(stream.play_loop(), timeout=1.0)
    finally:
        await stopper

    head_wobbler.feed_pcm.assert_called_once()
    args, kwargs = head_wobbler.feed_pcm.call_args
    assert np.array_equal(args[0], chunk.reshape(1, -1))
    assert args[1] == 24000
    assert kwargs["start_delay_s"] == pytest.approx(1.0)
    media.push_audio_sample.assert_called_once()


def test_backend_config_rejects_legacy_cloud_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should expose only local and Hugging Face production backends."""
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "local")
    monkeypatch.setenv("BACKEND_PROVIDER", "local")

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream._init_settings_ui_if_needed()

    client = TestClient(app)

    assert client.post("/backend_config", json={"backend": "gemini"}).status_code == 400
    assert client.post("/backend_config", json={"backend": "openai"}).status_code == 400
    assert not (tmp_path / ".env").exists()


def test_backend_config_persists_local_hf_selection_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should persist a direct Hugging Face websocket target."""
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "openai")
    monkeypatch.setattr(config, "MODEL_NAME", "gpt-realtime-2")
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", None)
    monkeypatch.setenv("BACKEND_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_NAME", "gpt-realtime-2")
    monkeypatch.delenv("HF_REALTIME_CONNECTION_MODE", raising=False)
    monkeypatch.delenv("HF_REALTIME_SESSION_URL", raising=False)
    monkeypatch.delenv("HF_REALTIME_WS_URL", raising=False)

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream._init_settings_ui_if_needed()

    client = TestClient(app)
    response = client.post(
        "/backend_config",
        json={
            "backend": "huggingface",
            "hf_mode": "local",
            "hf_host": "localhost",
            "hf_port": 8765,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["backend_provider"] == "huggingface"
    assert data["active_backend"] == "openai"
    assert data["has_hf_ws_url"] is True
    assert data["has_hf_connection"] is True
    assert data["hf_connection_mode"] == "local"
    assert data["hf_direct_host"] == "localhost"
    assert data["hf_direct_port"] == 8765
    assert data["requires_restart"] is True

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    env_lines = env_text.splitlines()
    assert "BACKEND_PROVIDER=huggingface" in env_text
    assert "HF_REALTIME_CONNECTION_MODE=local" in env_text
    assert "HF_REALTIME_WS_URL=ws://localhost:8765/v1/realtime" in env_text
    assert not any(line.startswith("MODEL_NAME=") for line in env_lines)


def test_backend_config_persists_deployed_mode_without_clearing_local_hf_ws_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving deployed mode should make env selection explicit and remove stale allocator URLs."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "BACKEND_PROVIDER=huggingface\n"
        "HF_REALTIME_SESSION_URL=https://lb.example.test/session\n"
        "HF_REALTIME_WS_URL=ws://localhost:8765/v1/realtime\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "BACKEND_PROVIDER", "huggingface")
    monkeypatch.setattr(config, "MODEL_NAME", "gpt-realtime-2")
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://localhost:8765/v1/realtime")
    monkeypatch.setenv("BACKEND_PROVIDER", "huggingface")
    monkeypatch.setenv("MODEL_NAME", "gpt-realtime-2")
    monkeypatch.delenv("HF_REALTIME_CONNECTION_MODE", raising=False)
    monkeypatch.setenv("HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    monkeypatch.setenv("HF_REALTIME_WS_URL", "ws://localhost:8765/v1/realtime")

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream._init_settings_ui_if_needed()

    client = TestClient(app)
    response = client.post(
        "/backend_config",
        json={
            "backend": "huggingface",
            "hf_mode": "deployed",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["has_hf_session_url"] is True
    assert data["has_hf_ws_url"] is True
    assert data["hf_connection_mode"] == "deployed"

    env_text = env_path.read_text(encoding="utf-8")
    assert "HF_REALTIME_CONNECTION_MODE=deployed" in env_text
    assert "HF_REALTIME_SESSION_URL=" not in env_text
    assert "HF_REALTIME_WS_URL=ws://localhost:8765/v1/realtime" in env_text


def test_backend_config_switches_to_saved_local_hf_connection_without_payload_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Switching back to a saved local Hugging Face backend should reuse the persisted target."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "BACKEND_PROVIDER=openai\n"
        "MODEL_NAME=gpt-realtime-2\n"
        "HF_REALTIME_CONNECTION_MODE=local\n"
        "HF_REALTIME_WS_URL=ws://192.168.1.42:8766/v1/realtime\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "BACKEND_PROVIDER", "openai")
    monkeypatch.setattr(config, "MODEL_NAME", "gpt-realtime-2")
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://192.168.1.42:8766/v1/realtime")
    monkeypatch.setenv("BACKEND_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_NAME", "gpt-realtime-2")
    monkeypatch.setenv("HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setenv("HF_REALTIME_WS_URL", "ws://192.168.1.42:8766/v1/realtime")

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream._init_settings_ui_if_needed()

    client = TestClient(app)
    response = client.post(
        "/backend_config",
        json={"backend": "huggingface"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["backend_provider"] == "huggingface"
    assert data["hf_connection_mode"] == "local"
    assert data["hf_direct_host"] == "192.168.1.42"
    assert data["hf_direct_port"] == 8766

    env_text = env_path.read_text(encoding="utf-8")
    assert "BACKEND_PROVIDER=huggingface" in env_text
    assert "HF_REALTIME_CONNECTION_MODE=local" in env_text
    assert "HF_REALTIME_WS_URL=ws://192.168.1.42:8766/v1/realtime" in env_text


def test_backend_config_rejects_invalid_hf_port_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should reject invalid local Hugging Face ports from direct callers."""
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "huggingface")
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", None)

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream._init_settings_ui_if_needed()

    client = TestClient(app)
    response = client.post(
        "/backend_config",
        json={
            "backend": "huggingface",
            "hf_mode": "local",
            "hf_host": "localhost",
            "hf_port": 0,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_hf_port"


def test_status_reports_direct_hf_ws_url_as_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should treat a direct Hugging Face websocket as a valid configuration."""
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "huggingface")
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream._init_settings_ui_if_needed()

    client = TestClient(app)
    response = client.get("/status")

    assert response.status_code == 200
    data = response.json()
    assert data["backend_provider"] == "huggingface"
    assert data["has_hf_session_url"] is False
    assert data["has_hf_ws_url"] is True
    assert data["has_hf_connection"] is True
    assert data["hf_connection_mode"] == "local"
    assert data["can_proceed_with_hf"] is True


def _profile_store_with_default(tmp_path: Path) -> ProfileStore:
    store = ProfileStore(tmp_path)
    store.save_new("default", instructions="[default_prompt]", tools_text="dance\ncamera\n", voice="local")
    return store


def test_headless_profile_routes_return_local_voice_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Headless profile UI should expose the local production voice by default."""
    store = _profile_store_with_default(tmp_path)
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "local")

    app = FastAPI()
    handler = MagicMock()
    mount_personality_routes(app, handler, lambda: None, profile_store=store)

    client = TestClient(app)
    response = client.get("/voices")

    assert response.status_code == 200
    assert response.json() == ["local"]


def test_headless_profile_routes_load_default_tools(tmp_path: Path) -> None:
    """Headless profile UI should expose repo-backed default tools on initial load."""
    store = _profile_store_with_default(tmp_path)
    app = FastAPI()
    handler = MagicMock()
    mount_personality_routes(app, handler, lambda: None, profile_store=store)

    client = TestClient(app)
    response = client.get("/profiles/load", params={"name": "default"})

    assert response.status_code == 200
    data = response.json()
    assert data["tools_text"]
    assert "dance" in data["enabled_tools"]
    assert "camera" in data["enabled_tools"]


def test_headless_profile_routes_save_new_and_overwrite(tmp_path: Path) -> None:
    """Dashboard profile routes should create and overwrite repo-backed profiles."""
    store = _profile_store_with_default(tmp_path)
    app = FastAPI()
    handler = MagicMock()
    mount_personality_routes(app, handler, lambda: None, profile_store=store)

    client = TestClient(app)
    created = client.post(
        "/profiles/save",
        json={
            "name": "Demo Profile",
            "instructions": "Be concise.",
            "tools_text": "dance\n",
            "voice": "local",
        },
    )
    overwritten = client.post(
        "/profiles/save",
        json={
            "name": "Demo_Profile",
            "instructions": "Be very concise.",
            "tools_text": "camera\n",
            "voice": "local",
            "overwrite": True,
        },
    )

    assert created.status_code == 200
    assert created.json()["profile"] == "Demo_Profile"
    assert overwritten.status_code == 200
    loaded = client.get("/profiles/load", params={"name": "Demo_Profile"}).json()
    assert loaded["instructions"] == "Be very concise."
    assert loaded["enabled_tools"] == ["camera"]


def test_headless_profile_routes_apply_voice_accepts_query_param(tmp_path: Path) -> None:
    """Headless profile UI should apply and persist a voice change from a POST query param."""
    store = _profile_store_with_default(tmp_path)
    app = FastAPI()
    handler = MagicMock()
    handler.change_voice = AsyncMock(return_value="Voice changed to cedar.")

    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run_loop() -> None:
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    started.wait(timeout=1.0)

    try:
        mount_personality_routes(app, handler, lambda: loop, profile_store=store)

        client = TestClient(app)
        response = client.post("/voices/apply?voice=cedar")

        assert response.status_code == 200
        assert response.json() == {"ok": True, "status": "Voice changed to cedar."}
        handler.change_voice.assert_awaited_once_with("cedar")
        assert store.load("default").voice == "cedar"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()


def test_headless_profile_routes_persist_startup_with_voice_override(tmp_path: Path) -> None:
    """Saving a startup profile should persist the active manual voice override."""
    store = _profile_store_with_default(tmp_path)
    store.save_new("demo", instructions="hello", tools_text="dance\n", voice="local")
    app = FastAPI()
    handler = MagicMock()
    handler.apply_personality = AsyncMock(return_value="Applied profile and restarted realtime session.")
    handler.get_current_voice = MagicMock(return_value="shimmer")
    persist_personality = MagicMock()

    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run_loop() -> None:
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    started.wait(timeout=1.0)

    try:
        mount_personality_routes(
            app,
            handler,
            lambda: loop,
            persist_personality=persist_personality,
            profile_store=store,
        )

        client = TestClient(app)
        response = client.post("/profiles/apply?name=demo&persist=1")

        assert response.status_code == 200
        assert response.json()["ok"] is True
        handler.apply_personality.assert_awaited_once_with("demo")
        persist_personality.assert_called_once_with("demo", "shimmer")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()


def test_local_stream_persist_personality_stores_voice_override(tmp_path) -> None:
    """Persisting startup settings should write both profile and voice override."""
    stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

    stream._persist_personality("default", "shimmer")

    settings_path = tmp_path / "startup_settings.json"
    assert settings_path.exists()
    assert settings_path.read_text(encoding="utf-8") == '{\n  "profile": "default",\n  "voice": "shimmer"\n}\n'
    assert stream._read_persisted_personality() == "default"


def test_local_stream_persist_personality_clears_legacy_startup_env_overrides(tmp_path, monkeypatch) -> None:
    """Saving startup settings should remove legacy `.env` profile and voice overrides."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=test-key\n"
        "REACHY_MINI_CUSTOM_PROFILE=mad_scientist_assistant\n"
        "REACHY_MINI_VOICE_OVERRIDE=shimmer\n",
        encoding="utf-8",
    )
    stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

    stream._persist_personality(None, "Aiden")

    env_text = env_path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=test-key" in env_text
    assert "REACHY_MINI_CUSTOM_PROFILE=" not in env_text
    assert "REACHY_MINI_VOICE_OVERRIDE=" not in env_text

    applied_profiles: list[str | None] = []
    monkeypatch.delenv("REACHY_MINI_CUSTOM_PROFILE", raising=False)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.runtime.config.set_custom_profile",
        lambda profile: applied_profiles.append(profile),
    )

    settings = load_startup_settings_into_runtime(tmp_path)

    assert settings == StartupSettings(voice="Aiden")
    assert applied_profiles == [None]


def test_local_stream_launch_waits_for_manual_openai_key_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI startup should wait for settings input instead of claiming a bundled key."""
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "openai")
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    monkeypatch.setenv("BACKEND_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    media = SimpleNamespace(
        start_recording=MagicMock(),
        start_playing=MagicMock(),
    )
    robot = SimpleNamespace(media=media)
    stream = LocalStream(MagicMock(), robot, settings_app=FastAPI(), instance_path=str(tmp_path))
    stream._active_backend_name = "openai"

    init_settings_ui = MagicMock()
    monkeypatch.setattr(stream, "_init_settings_ui_if_needed", init_settings_ui)
    monkeypatch.setattr(stream, "_has_required_key", MagicMock(side_effect=[False, False]))
    monkeypatch.setattr("reachy_mini_conversation_app.runtime.console.time.sleep", MagicMock(side_effect=KeyboardInterrupt))

    stream.launch()

    init_settings_ui.assert_called_once()
    media.start_recording.assert_not_called()
    media.start_playing.assert_not_called()
