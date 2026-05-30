import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urlsplit, parse_qsl, urlunsplit

from dotenv import find_dotenv, load_dotenv

from reachy_mini_conversation_app.profiles.store import default_profiles_root


# Locked profile: set to a profile name (e.g., "astronomer") to lock the app
# to that profile and disable all profile switching. Leave as None for normal behavior.
LOCKED_PROFILE: str | None = None
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _packaged_profiles_directory() -> Path | None:
    """Return the package-data profiles directory when available."""
    try:
        return default_profiles_root()
    except Exception:
        return None


def _resolve_default_profiles_directory() -> Path:
    """Resolve production profiles from package data."""
    packaged_profiles = _packaged_profiles_directory()
    if packaged_profiles is not None and packaged_profiles.is_dir():
        return packaged_profiles
    return PROJECT_ROOT / "src" / "reachy_mini_conversation_app" / "profiles"


DEFAULT_PROFILES_DIRECTORY = _resolve_default_profiles_directory()
DEFAULT_PIPER_VOICE_PATH = PROJECT_ROOT / "cache" / "piper-voices" / "en_US-lessac-medium.onnx"


def _default_piper_voice() -> str | None:
    """Return the bundled/downloaded default Piper voice path when available."""
    return str(DEFAULT_PIPER_VOICE_PATH) if DEFAULT_PIPER_VOICE_PATH.is_file() else None

# Full list of voices supported by the OpenAI Realtime / TTS API.
# Source: https://developers.openai.com/api/docs/guides/text-to-speech/#voice-options
# "marin" and "cedar" are recommended for gpt-realtime-2.
AVAILABLE_VOICES: list[str] = [
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "marin",
    "sage",
    "shimmer",
    "verse",
]
OPENAI_DEFAULT_VOICE = "cedar"

# Qwen3-TTS CustomVoice speaker catalog from the deployed Hugging Face backend.
HF_AVAILABLE_VOICES: list[str] = [
    "Aiden",
    "Ryan",
    "Dylan",
    "Eric",
    "Ono_Anna",
    "Serena",
    "Sohee",
    "Uncle_Fu",
    "Vivian",
]

# Voices supported by the Gemini Live API
GEMINI_AVAILABLE_VOICES: list[str] = [
    "Aoede",
    "Charon",
    "Fenrir",
    "Kore",
    "Leda",
    "Orus",
    "Puck",
    "Zephyr",
]

OPENAI_BACKEND = "openai"
GEMINI_BACKEND = "gemini"
HF_BACKEND = "huggingface"
LOCAL_BACKEND = "local"
PRODUCTION_BACKENDS = {LOCAL_BACKEND, HF_BACKEND}
LEGACY_BACKENDS = {OPENAI_BACKEND, GEMINI_BACKEND}
DEFAULT_BACKEND_PROVIDER = LOCAL_BACKEND
DEFAULT_WIRED_REACHY_HOST = "10.42.0.2"
REACHY_MEDIA_HOST_ENV = "REACHY_MEDIA_HOST"
HF_REALTIME_CONNECTION_MODE_ENV = "HF_REALTIME_CONNECTION_MODE"
HF_REALTIME_WS_URL_ENV = "HF_REALTIME_WS_URL"
HF_LOCAL_CONNECTION_MODE = "local"
HF_DEPLOYED_CONNECTION_MODE = "deployed"
HF_REALTIME_SESSION_PROXY_URL = "https://pollen-robotics-reachy-mini-realtime-url.hf.space/session"
DEFAULT_LOCAL_CHAT_SERVER_MODEL = "ggml-org/gemma-3-1b-it-GGUF"
DEFAULT_LOCAL_CHAT_SERVER_HF = "ggml-org/gemma-3-1b-it-GGUF:Q4_K_M"
DEFAULT_LOCAL_ROUTER_SERVER_HF = "Qwen/Qwen3-0.6B-GGUF:Q8_0"
DEFAULT_LOCAL_VISION_SERVER_HF = "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF:Q8_0"


@dataclass(frozen=True)
class HFBackendDefaults:
    """Defaults for the Hugging Face realtime backend."""

    connection_mode: str = HF_DEPLOYED_CONNECTION_MODE
    # App-managed Hugging Face Space proxy. The Space forwards to the current
    # session allocator, so allocator changes do not require app releases.
    # Users who need a custom target should use HF_REALTIME_CONNECTION_MODE=local
    # with HF_REALTIME_WS_URL.
    session_url: str = HF_REALTIME_SESSION_PROXY_URL
    voice: str = "Aiden"
    model_name: str = ""
    direct_port: int = 8765


HF_DEFAULTS = HFBackendDefaults()
DEFAULT_MODEL_NAME_BY_BACKEND = {
    OPENAI_BACKEND: "gpt-realtime-2",
    GEMINI_BACKEND: "gemini-3.1-flash-live-preview",
    HF_BACKEND: HF_DEFAULTS.model_name,
    LOCAL_BACKEND: DEFAULT_LOCAL_CHAT_SERVER_MODEL,
}
DEFAULT_LOCAL_ROUTER_SERVER_MODEL = "Qwen/Qwen3-0.6B-GGUF"
DEFAULT_LOCAL_VISION_SERVER_MODEL = "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF"
BACKEND_LABEL_BY_PROVIDER = {
    OPENAI_BACKEND: "OpenAI Realtime",
    GEMINI_BACKEND: "Gemini Live",
    HF_BACKEND: "Hugging Face",
    LOCAL_BACKEND: "Local Mac",
}
DEFAULT_VOICE_BY_BACKEND = {
    OPENAI_BACKEND: OPENAI_DEFAULT_VOICE,
    GEMINI_BACKEND: "Kore",
    HF_BACKEND: HF_DEFAULTS.voice,
    LOCAL_BACKEND: "local",
}

logger = logging.getLogger(__name__)


def _is_gemini_model_name(model_name: str | None) -> bool:
    """Return True when the provided model name targets Gemini."""
    candidate = (model_name or "").strip().lower()
    return candidate.startswith("gemini")


def _normalize_backend_provider(
    backend_provider: str | None = None,
    model_name: str | None = None,
) -> str:
    """Normalize the configured backend provider."""
    candidate = (backend_provider or "").strip().lower()
    if candidate in DEFAULT_MODEL_NAME_BY_BACKEND:
        if candidate in LEGACY_BACKENDS:
            logger.warning("%s backend is legacy-only in this production build.", candidate)
        return candidate
    if candidate:
        expected = ", ".join(sorted(DEFAULT_MODEL_NAME_BY_BACKEND))
        raise ValueError(f"Invalid BACKEND_PROVIDER={backend_provider!r}. Expected one of: {expected}.")
    if _is_gemini_model_name(model_name):
        logger.warning("MODEL_NAME looks like Gemini but BACKEND_PROVIDER is unset; using local production backend.")
    return DEFAULT_BACKEND_PROVIDER


def _resolve_model_name(
    backend_provider: str | None = None,
    model_name: str | None = None,
) -> str:
    """Return a model name that matches the selected backend provider."""
    normalized_backend = _normalize_backend_provider(backend_provider, model_name)
    if normalized_backend == HF_BACKEND:
        return DEFAULT_MODEL_NAME_BY_BACKEND[HF_BACKEND]
    if normalized_backend == LOCAL_BACKEND:
        return (model_name or DEFAULT_MODEL_NAME_BY_BACKEND[LOCAL_BACKEND]).strip()

    candidate = (model_name or "").strip()
    if candidate:
        if normalized_backend == GEMINI_BACKEND and _is_gemini_model_name(candidate):
            return candidate
        if normalized_backend != GEMINI_BACKEND and not _is_gemini_model_name(candidate):
            return candidate
        logger.warning(
            "MODEL_NAME=%r does not match BACKEND_PROVIDER=%r, using default %r",
            candidate,
            normalized_backend,
            DEFAULT_MODEL_NAME_BY_BACKEND[normalized_backend],
        )
    return DEFAULT_MODEL_NAME_BY_BACKEND[normalized_backend]


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment flag.

    Accepted truthy values: 1, true, yes, on
    Accepted falsy values: 0, false, no, off
    """
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    logger.warning("Invalid boolean value for %s=%r, using default=%s", name, raw, default)
    return default


def _env_float(name: str, default: float) -> float:
    """Parse a floating-point environment value."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float value for %s=%r, using default=%s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    """Parse an integer environment value."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer value for %s=%r, using default=%s", name, raw, default)
        return default


def _normalize_hf_connection_mode(value: str | None) -> str | None:
    """Normalize the Hugging Face connection mode, if explicitly configured."""
    candidate = (value or "").strip().lower()
    if not candidate:
        return None

    if candidate not in {HF_LOCAL_CONNECTION_MODE, HF_DEPLOYED_CONNECTION_MODE}:
        logger.warning(
            "Invalid %s=%r. Expected local or deployed.",
            HF_REALTIME_CONNECTION_MODE_ENV,
            value,
        )
        return None
    return candidate


@dataclass(frozen=True)
class HFConnectionSelection:
    """Resolved Hugging Face connection mode and target availability."""

    mode: str
    has_target: bool
    session_url: str | None = None
    direct_ws_url: str | None = None


@dataclass(frozen=True)
class HFRealtimeURLParts:
    """Parsed Hugging Face realtime URL components used by UI and client setup."""

    base_url: str
    websocket_base_url: str
    connect_query: dict[str, str]
    host: str | None
    port: int | None
    has_realtime_path: bool


def parse_hf_realtime_url(realtime_url: str) -> HFRealtimeURLParts:
    """Parse a Hugging Face realtime URL into OpenAI-compatible client endpoints."""
    parsed = urlsplit(realtime_url)
    scheme = parsed.scheme.lower()
    if scheme not in {"ws", "wss", "http", "https"}:
        raise ValueError(
            "Expected Hugging Face realtime URL to start with ws://, wss://, http://, or https://, "
            f"got: {realtime_url}"
        )

    path = parsed.path.rstrip("/")
    has_realtime_path = path.endswith("/realtime")
    if has_realtime_path:
        base_path = path[: -len("/realtime")]
    else:
        base_path = path

    connect_query = {key: value for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key != "model"}
    http_scheme = "https" if scheme in {"wss", "https"} else "http"
    websocket_scheme = "wss" if scheme in {"wss", "https"} else "ws"
    base_url = urlunsplit((http_scheme, parsed.netloc, base_path, "", ""))
    websocket_base_url = urlunsplit((websocket_scheme, parsed.netloc, base_path, "", ""))
    return HFRealtimeURLParts(
        base_url=base_url,
        websocket_base_url=websocket_base_url,
        connect_query=connect_query,
        host=parsed.hostname,
        port=parsed.port or HF_DEFAULTS.direct_port,
        has_realtime_path=has_realtime_path,
    )


def parse_hf_direct_target(ws_url: str | None) -> tuple[str | None, int | None]:
    """Extract host and port from a direct Hugging Face realtime URL."""
    if not ws_url:
        return None, None
    try:
        parsed = parse_hf_realtime_url(ws_url)
        return parsed.host, parsed.port
    except Exception:
        return None, None


def build_hf_direct_ws_url(host: str, port: int) -> str:
    """Build the direct Hugging Face realtime websocket URL used by the app."""
    return f"ws://{host}:{port}/v1/realtime"


# Validate LOCKED_PROFILE at startup
if LOCKED_PROFILE is not None:
    _profiles_dir = DEFAULT_PROFILES_DIRECTORY
    _profile_path = _profiles_dir / LOCKED_PROFILE
    _instructions_file = _profile_path / "instructions.txt"
    if not _profile_path.is_dir():
        print(f"Error: LOCKED_PROFILE '{LOCKED_PROFILE}' does not exist in {_profiles_dir}", file=sys.stderr)
        sys.exit(1)
    if not _instructions_file.is_file():
        print(f"Error: LOCKED_PROFILE '{LOCKED_PROFILE}' has no instructions.txt", file=sys.stderr)
        sys.exit(1)

_skip_dotenv = _env_flag("REACHY_MINI_SKIP_DOTENV", default=False)

if _skip_dotenv:
    logger.info("Skipping .env loading because REACHY_MINI_SKIP_DOTENV is set")
else:
    # Locate .env file (search upward from current working directory)
    dotenv_path = find_dotenv(usecwd=True)

    if dotenv_path:
        # Load .env and override environment variables
        load_dotenv(dotenv_path=dotenv_path, override=True)
        logger.info(f"Configuration loaded from {dotenv_path}")
    else:
        logger.warning("No .env file found, using environment variables")


class Config:
    """Configuration class for the conversation app."""

    # Required (one of these depending on BACKEND_PROVIDER)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # The key is downloaded in console.py if needed
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    # Optional
    BACKEND_PROVIDER = _normalize_backend_provider(
        os.getenv("BACKEND_PROVIDER"),
        os.getenv("MODEL_NAME"),
    )
    MODEL_NAME = _resolve_model_name(BACKEND_PROVIDER, os.getenv("MODEL_NAME"))
    HF_REALTIME_CONNECTION_MODE = (
        _normalize_hf_connection_mode(os.getenv(HF_REALTIME_CONNECTION_MODE_ENV)) or HF_DEFAULTS.connection_mode
    )
    # Deliberately ignore HF_REALTIME_SESSION_URL from the environment; the app-managed proxy is HF_DEFAULTS.session_url.
    HF_REALTIME_SESSION_URL = HF_DEFAULTS.session_url
    HF_REALTIME_WS_URL = os.getenv(HF_REALTIME_WS_URL_ENV)
    REACHY_MEDIA_HOST = os.getenv(REACHY_MEDIA_HOST_ENV)
    LOCAL_MODEL_SERVER_AUTOSTART = _env_flag("LOCAL_MODEL_SERVER_AUTOSTART", True)
    LOCAL_MODEL_SERVER_START_TIMEOUT_SECONDS = _env_float("LOCAL_MODEL_SERVER_START_TIMEOUT_SECONDS", 240.0)
    LOCAL_LLAMA_SERVER_BIN = os.getenv("LOCAL_LLAMA_SERVER_BIN", "llama-server")
    LOCAL_CHAT_SERVER_HF = os.getenv("LOCAL_CHAT_SERVER_HF", DEFAULT_LOCAL_CHAT_SERVER_HF)
    LOCAL_CHAT_BASE_URL = os.getenv("LOCAL_CHAT_BASE_URL", "http://127.0.0.1:8080/v1")
    LOCAL_CHAT_MODEL = os.getenv("LOCAL_CHAT_MODEL") or DEFAULT_LOCAL_CHAT_SERVER_MODEL
    LOCAL_CHAT_NUM_PREDICT = _env_int("LOCAL_CHAT_NUM_PREDICT", 96)
    LOCAL_ROUTER_SERVER_HF = os.getenv("LOCAL_ROUTER_SERVER_HF", DEFAULT_LOCAL_ROUTER_SERVER_HF)
    LOCAL_ROUTER_BASE_URL = os.getenv("LOCAL_ROUTER_BASE_URL", "http://127.0.0.1:8082/v1")
    LOCAL_ROUTER_MODEL = os.getenv("LOCAL_ROUTER_MODEL") or DEFAULT_LOCAL_ROUTER_SERVER_MODEL
    LOCAL_ROUTER_NUM_CTX = _env_int("LOCAL_ROUTER_NUM_CTX", 448)
    LOCAL_ROUTER_NUM_PREDICT = _env_int("LOCAL_ROUTER_NUM_PREDICT", 18)
    LOCAL_STT_PROVIDER = os.getenv("LOCAL_STT_PROVIDER", "mlx-whisper")
    LOCAL_STT_MODEL = os.getenv("LOCAL_STT_MODEL", "mlx-community/whisper-small-mlx")
    LOCAL_TTS_PROVIDER = os.getenv("LOCAL_TTS_PROVIDER", "piper")
    PIPER_VOICE = os.getenv("PIPER_VOICE") or _default_piper_voice()
    LOCAL_VAD_SILENCE_SECONDS = _env_float("LOCAL_VAD_SILENCE_SECONDS", 0.45)
    HF_HOME = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_SERVER_HF = os.getenv("LOCAL_VISION_SERVER_HF", DEFAULT_LOCAL_VISION_SERVER_HF)
    LOCAL_VISION_BASE_URL = os.getenv("LOCAL_VISION_BASE_URL", "http://127.0.0.1:8081/v1")
    LOCAL_VISION_SERVER_MODEL = os.getenv("LOCAL_VISION_SERVER_MODEL", DEFAULT_LOCAL_VISION_SERVER_MODEL)
    LOCAL_VISION_NUM_PREDICT = _env_int("LOCAL_VISION_NUM_PREDICT", 48)
    LOCAL_VISION_MAX_IMAGE_SIDE = _env_int("LOCAL_VISION_MAX_IMAGE_SIDE", 512)
    LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, falls back to hf auth login if not set
    REACHY_CAMERA_HORIZONTAL_FOV_DEG = _env_float("REACHY_CAMERA_HORIZONTAL_FOV_DEG", 60.0)

    logger.debug(
        "Backend provider: %s, Model: %s, HF mode: %s, HF session URL set: %s, HF direct URL set: %s, HF_HOME: %s, Vision Model: %s",
        BACKEND_PROVIDER,
        MODEL_NAME,
        HF_REALTIME_CONNECTION_MODE,
        bool(HF_REALTIME_SESSION_URL and HF_REALTIME_SESSION_URL.strip()),
        bool(HF_REALTIME_WS_URL and HF_REALTIME_WS_URL.strip()),
        HF_HOME,
        LOCAL_VISION_MODEL,
    )

    # Filesystem root containing repo-backed production profiles.
    PROFILES_DIRECTORY = DEFAULT_PROFILES_DIRECTORY
    TOOLS_DIRECTORY = None
    AUTOLOAD_EXTERNAL_TOOLS = False
    REACHY_MINI_CUSTOM_PROFILE = LOCKED_PROFILE or os.getenv("REACHY_MINI_CUSTOM_PROFILE")

    logger.debug(f"Custom Profile: {REACHY_MINI_CUSTOM_PROFILE}")

    def __init__(self) -> None:
        """Initialize the configuration."""
        logger.info("Using repo-backed production profiles from %s.", DEFAULT_PROFILES_DIRECTORY)
        if os.getenv("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY") or os.getenv("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY"):
            logger.warning("External profile/tool directories are ignored in this production build.")


config = Config()


def refresh_runtime_config_from_env() -> None:
    """Refresh mutable runtime config fields from the current environment."""
    config.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    config.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    config.BACKEND_PROVIDER = _normalize_backend_provider(
        os.getenv("BACKEND_PROVIDER"),
        os.getenv("MODEL_NAME"),
    )
    config.MODEL_NAME = _resolve_model_name(config.BACKEND_PROVIDER, os.getenv("MODEL_NAME"))
    config.HF_REALTIME_CONNECTION_MODE = (
        _normalize_hf_connection_mode(os.getenv(HF_REALTIME_CONNECTION_MODE_ENV)) or HF_DEFAULTS.connection_mode
    )
    # Deliberately ignore HF_REALTIME_SESSION_URL from the environment; the app-managed proxy is HF_DEFAULTS.session_url.
    config.HF_REALTIME_SESSION_URL = HF_DEFAULTS.session_url
    config.HF_REALTIME_WS_URL = os.getenv(HF_REALTIME_WS_URL_ENV)
    config.REACHY_MEDIA_HOST = os.getenv(REACHY_MEDIA_HOST_ENV)
    config.LOCAL_MODEL_SERVER_AUTOSTART = _env_flag("LOCAL_MODEL_SERVER_AUTOSTART", True)
    config.LOCAL_MODEL_SERVER_START_TIMEOUT_SECONDS = _env_float("LOCAL_MODEL_SERVER_START_TIMEOUT_SECONDS", 240.0)
    config.LOCAL_LLAMA_SERVER_BIN = os.getenv("LOCAL_LLAMA_SERVER_BIN", "llama-server")
    config.LOCAL_CHAT_SERVER_HF = os.getenv("LOCAL_CHAT_SERVER_HF", DEFAULT_LOCAL_CHAT_SERVER_HF)
    config.LOCAL_CHAT_BASE_URL = os.getenv("LOCAL_CHAT_BASE_URL", "http://127.0.0.1:8080/v1")
    config.LOCAL_CHAT_MODEL = os.getenv("LOCAL_CHAT_MODEL") or DEFAULT_LOCAL_CHAT_SERVER_MODEL
    config.LOCAL_CHAT_NUM_PREDICT = _env_int("LOCAL_CHAT_NUM_PREDICT", 96)
    config.LOCAL_ROUTER_SERVER_HF = os.getenv("LOCAL_ROUTER_SERVER_HF", DEFAULT_LOCAL_ROUTER_SERVER_HF)
    config.LOCAL_ROUTER_BASE_URL = os.getenv("LOCAL_ROUTER_BASE_URL", "http://127.0.0.1:8082/v1")
    config.LOCAL_ROUTER_MODEL = os.getenv("LOCAL_ROUTER_MODEL") or DEFAULT_LOCAL_ROUTER_SERVER_MODEL
    config.LOCAL_ROUTER_NUM_CTX = _env_int("LOCAL_ROUTER_NUM_CTX", 448)
    config.LOCAL_ROUTER_NUM_PREDICT = _env_int("LOCAL_ROUTER_NUM_PREDICT", 18)
    config.LOCAL_STT_PROVIDER = os.getenv("LOCAL_STT_PROVIDER", "mlx-whisper")
    config.LOCAL_STT_MODEL = os.getenv("LOCAL_STT_MODEL", "mlx-community/whisper-small-mlx")
    config.LOCAL_TTS_PROVIDER = os.getenv("LOCAL_TTS_PROVIDER", "piper")
    config.PIPER_VOICE = os.getenv("PIPER_VOICE") or _default_piper_voice()
    config.LOCAL_VAD_SILENCE_SECONDS = _env_float("LOCAL_VAD_SILENCE_SECONDS", 0.45)
    config.HF_HOME = os.getenv("HF_HOME", "./cache")
    config.LOCAL_VISION_SERVER_HF = os.getenv("LOCAL_VISION_SERVER_HF", DEFAULT_LOCAL_VISION_SERVER_HF)
    config.LOCAL_VISION_BASE_URL = os.getenv("LOCAL_VISION_BASE_URL", "http://127.0.0.1:8081/v1")
    config.LOCAL_VISION_SERVER_MODEL = os.getenv("LOCAL_VISION_SERVER_MODEL", DEFAULT_LOCAL_VISION_SERVER_MODEL)
    config.LOCAL_VISION_NUM_PREDICT = _env_int("LOCAL_VISION_NUM_PREDICT", 48)
    config.LOCAL_VISION_MAX_IMAGE_SIDE = _env_int("LOCAL_VISION_MAX_IMAGE_SIDE", 512)
    config.LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    config.HF_TOKEN = os.getenv("HF_TOKEN")
    config.REACHY_CAMERA_HORIZONTAL_FOV_DEG = _env_float("REACHY_CAMERA_HORIZONTAL_FOV_DEG", 60.0)
    config.REACHY_MINI_CUSTOM_PROFILE = LOCKED_PROFILE or os.getenv("REACHY_MINI_CUSTOM_PROFILE")


def get_backend_choice(model_name: str | None = None) -> str:
    """Return the configured backend family."""
    if model_name is not None:
        return _normalize_backend_provider(model_name=model_name)
    return _normalize_backend_provider(config.BACKEND_PROVIDER, config.MODEL_NAME)


def get_model_name_for_backend(backend: str) -> str:
    """Return the default model name for a backend selector value."""
    return DEFAULT_MODEL_NAME_BY_BACKEND[_normalize_backend_provider(backend)]


def get_backend_label(backend: str | None = None) -> str:
    """Return a human-readable label for a backend selector value."""
    normalized_backend = get_backend_choice() if backend is None else _normalize_backend_provider(backend)
    return BACKEND_LABEL_BY_PROVIDER[normalized_backend]


def get_available_voices_for_backend(backend: str | None = None) -> list[str]:
    """Return the curated voice list for a backend selector value."""
    normalized_backend = get_backend_choice() if backend is None else _normalize_backend_provider(backend)
    if normalized_backend == GEMINI_BACKEND:
        return list(GEMINI_AVAILABLE_VOICES)
    if normalized_backend == HF_BACKEND:
        return list(HF_AVAILABLE_VOICES)
    if normalized_backend == LOCAL_BACKEND:
        return [DEFAULT_VOICE_BY_BACKEND[LOCAL_BACKEND]]
    return list(AVAILABLE_VOICES)


def get_default_voice_for_backend(backend: str | None = None) -> str:
    """Return the default voice for a backend selector value."""
    normalized_backend = get_backend_choice() if backend is None else _normalize_backend_provider(backend)
    return DEFAULT_VOICE_BY_BACKEND[normalized_backend]


def get_hf_session_url() -> str | None:
    """Return the built-in Hugging Face session proxy URL, if any."""
    value = (getattr(config, "HF_REALTIME_SESSION_URL", None) or "").strip()
    return value or None


def get_hf_direct_ws_url() -> str | None:
    """Return the configured direct Hugging Face realtime URL, if any."""
    value = (getattr(config, "HF_REALTIME_WS_URL", None) or "").strip()
    return value or None


def get_hf_connection_selection() -> HFConnectionSelection:
    """Resolve the selected Hugging Face connection mode and whether it is usable."""
    session_url = get_hf_session_url()
    direct_ws_url = get_hf_direct_ws_url()
    mode = _normalize_hf_connection_mode(getattr(config, "HF_REALTIME_CONNECTION_MODE", None))
    if mode is None:
        raise RuntimeError(f"{HF_REALTIME_CONNECTION_MODE_ENV} must be set to local or deployed.")

    target = direct_ws_url if mode == HF_LOCAL_CONNECTION_MODE else session_url

    return HFConnectionSelection(
        mode=mode,
        has_target=bool(target),
        session_url=session_url,
        direct_ws_url=direct_ws_url,
    )


def has_hf_realtime_target() -> bool:
    """Return whether Hugging Face has a target for the selected mode."""
    return get_hf_connection_selection().has_target


def is_gemini_model() -> bool:
    """Return True if the configured MODEL_NAME is a Gemini Live model."""
    return get_backend_choice() == GEMINI_BACKEND


def set_custom_profile(profile: str | None) -> None:
    """Update the selected custom profile at runtime and expose it via env.

    This ensures modules that read `config` and code that inspects the
    environment see a consistent value.
    """
    if LOCKED_PROFILE is not None:
        return
    try:
        config.REACHY_MINI_CUSTOM_PROFILE = profile
    except Exception:
        pass
    try:
        import os as _os

        if profile:
            _os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile
        else:
            # Remove to reflect default
            _os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
    except Exception:
        pass
