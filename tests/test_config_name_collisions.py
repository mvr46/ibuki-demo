import pytest

import reachy_mini_conversation_app.runtime.config as config_mod


def test_config_ignores_external_profile_and_tool_directories(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production config should ignore external profile/tool roots."""
    monkeypatch.setenv("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY", "/tmp/external_profiles")
    monkeypatch.setenv("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY", "/tmp/external_tools")

    created = config_mod.Config()

    assert created.PROFILES_DIRECTORY == config_mod.DEFAULT_PROFILES_DIRECTORY
    assert created.TOOLS_DIRECTORY is None
    assert created.AUTOLOAD_EXTERNAL_TOOLS is False


def test_backend_provider_defaults_to_local_when_unset() -> None:
    """Production should default to the local Gemma/Qwen backend."""
    assert config_mod._normalize_backend_provider(None, None) == config_mod.LOCAL_BACKEND
    assert config_mod._normalize_backend_provider("", None) == config_mod.LOCAL_BACKEND
    assert config_mod._normalize_backend_provider(None, "gpt-realtime-2") == config_mod.LOCAL_BACKEND
    assert config_mod._normalize_backend_provider(None, "gemini-3.1-flash-live-preview") == config_mod.LOCAL_BACKEND


def test_backend_provider_rejects_explicit_unknown_backend() -> None:
    """An explicit backend typo should fail instead of falling through to the default backend."""
    with pytest.raises(ValueError, match="Invalid BACKEND_PROVIDER='openia'"):
        config_mod._normalize_backend_provider("openia", None)


def test_huggingface_backend_does_not_resolve_model_name() -> None:
    """Hugging Face should rely on the server's model selection."""
    assert config_mod._resolve_model_name(config_mod.HF_BACKEND, None) == ""
    assert config_mod._resolve_model_name(config_mod.HF_BACKEND, "gpt-realtime-2") == ""


def test_local_backend_resolves_ollama_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local backend should use the Ollama model selector."""
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:test")

    assert config_mod._normalize_backend_provider("local", None) == config_mod.LOCAL_BACKEND
    assert config_mod._resolve_model_name(config_mod.LOCAL_BACKEND, None) == "qwen3.5:test"


def test_local_backend_defaults_to_gemma3_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local backend should default to Gemma for normal local chat and vision."""
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    assert config_mod._resolve_model_name(config_mod.LOCAL_BACKEND, None) == "gemma3:latest"


def test_hf_default_session_url_uses_stable_space_proxy() -> None:
    """The app should not embed the raw, replaceable Inference Endpoint allocator URL."""
    assert config_mod.HF_DEFAULTS.session_url == "https://pollen-robotics-reachy-mini-realtime-url.hf.space/session"
    assert ".aws.endpoints.huggingface.cloud" not in config_mod.HF_DEFAULTS.session_url


def test_refresh_runtime_config_reloads_hf_runtime_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instance-local .env reloads should update every env-backed Hugging Face runtime field."""
    monkeypatch.setenv("HF_TOKEN", "hf-runtime-token")
    monkeypatch.setenv("HF_HOME", "/tmp/reachy-hf-cache")
    monkeypatch.setenv("LOCAL_VISION_MODEL", "test/local-vision-model")

    monkeypatch.setattr(config_mod.config, "HF_TOKEN", None)
    monkeypatch.setattr(config_mod.config, "HF_HOME", "./old-cache")
    monkeypatch.setattr(config_mod.config, "LOCAL_VISION_MODEL", "old/model")

    config_mod.refresh_runtime_config_from_env()

    assert config_mod.config.HF_TOKEN == "hf-runtime-token"
    assert config_mod.config.HF_HOME == "/tmp/reachy-hf-cache"
    assert config_mod.config.LOCAL_VISION_MODEL == "test/local-vision-model"


@pytest.mark.parametrize(
    ("configured_mode", "session_url", "direct_ws_url", "expected_mode", "expected_has_target"),
    [
        ("local", "https://hf.example.test/session", None, "local", False),
        ("deployed", "https://hf.example.test/session", "ws://127.0.0.1:8765/v1/realtime", "deployed", True),
        ("local", None, "ws://127.0.0.1:8765/v1/realtime", "local", True),
        ("deployed", None, "ws://127.0.0.1:8765/v1/realtime", "deployed", False),
    ],
)
def test_hf_connection_selection_uses_explicit_mode_for_target(
    monkeypatch: pytest.MonkeyPatch,
    configured_mode: str | None,
    session_url: str | None,
    direct_ws_url: str | None,
    expected_mode: str,
    expected_has_target: bool,
) -> None:
    """Hugging Face selection should use the configured mode without inferring from URLs."""
    monkeypatch.setattr(config_mod.config, "HF_REALTIME_CONNECTION_MODE", configured_mode)
    monkeypatch.setattr(config_mod.config, "HF_REALTIME_SESSION_URL", session_url)
    monkeypatch.setattr(config_mod.config, "HF_REALTIME_WS_URL", direct_ws_url)

    selection = config_mod.get_hf_connection_selection()

    assert selection.mode == expected_mode
    assert selection.has_target is expected_has_target
    assert selection.session_url == session_url
    assert selection.direct_ws_url == direct_ws_url


def test_hf_connection_selection_requires_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hugging Face selection should fail instead of inferring a missing mode."""
    monkeypatch.setattr(config_mod.config, "HF_REALTIME_CONNECTION_MODE", None)
    monkeypatch.setattr(config_mod.config, "HF_REALTIME_SESSION_URL", "https://hf.example.test/session")
    monkeypatch.setattr(config_mod.config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")

    with pytest.raises(RuntimeError, match="HF_REALTIME_CONNECTION_MODE must be set"):
        config_mod.get_hf_connection_selection()
