# Development Guide

## Environment

Use Python 3.12 when possible. The project supports Python 3.10+, but the repo configuration and type checking are tuned for 3.12.

Recommended setup:

```bash
uv venv --python python3.12 .venv
source .venv/bin/activate
uv sync --group dev
```

Install optional vision stacks only when you need them:

```bash
uv sync --extra local_vision --group dev
uv sync --extra yolo_vision --group dev
uv sync --extra mediapipe_vision --group dev
uv sync --extra all_vision --group dev
```

## Common commands

Run the app:

```bash
uv run reachy-mini-conversation-app
```

Run with the web UI:

```bash
uv run reachy-mini-conversation-app --gradio
```

Run without camera hardware:

```bash
uv run reachy-mini-conversation-app --no-camera
```

Run without Reachy Mini camera/audio media hardware:

```bash
uv run reachy-mini-conversation-app --media-backend no_media
```

The app defaults to the robot daemon over the network. For local development daemons, override it:

```bash
uv run reachy-mini-conversation-app --connection-mode localhost_only
```

Run tests:

```bash
uv run pytest tests/ -v
```

Run focused tests:

```bash
uv run pytest tests/test_openai_realtime.py -v
uv run pytest tests/tools/test_camera.py -v
```

Format and lint:

```bash
uv run ruff check . --fix
uv run ruff format .
```

Type check:

```bash
uv run mypy --pretty --show-error-codes
```

## Test map

| Area | Tests |
| --- | --- |
| Config and URL parsing | `tests/test_utils.py`, `tests/test_startup_settings.py`, `tests/test_profile_paths.py` |
| Backend adapters | `tests/test_huggingface_realtime.py`, `tests/test_openai_realtime.py`, `tests/test_gemini_live.py` |
| Console mode | `tests/test_console.py` |
| Tool loading and external content | `tests/test_external_loading.py`, `tests/test_config_name_collisions.py` |
| Tool behavior | `tests/tools/` |
| Audio helpers | `tests/audio/` |
| Vision helpers | `tests/vision/` |

## Adding or changing a backend

For an OpenAI-compatible realtime backend, prefer subclassing `BaseRealtimeHandler` and implementing:

- provider constants such as `BACKEND_PROVIDER` and `SAMPLE_RATE`
- `_get_session_instructions()`
- `_get_session_voice()`
- `_get_active_tool_specs()`
- `_get_session_config()`
- `_build_realtime_client()`

For a backend with a different SDK/event shape, implement `ConversationHandler` directly, using `GeminiLiveHandler` as the closest example.

Also update config defaults, backend labels, voice lists, README configuration docs, and backend-specific tests.

## Changing tools

When adding a core tool:

1. Add a module under `src/reachy_mini_conversation_app/tools/`.
2. Implement a concrete `Tool` subclass.
3. Add the tool name to the relevant profile `tools.txt`.
4. Add or update tests under `tests/tools/` or `tests/test_external_loading.py`.

Keep tool results small. If a result contains transport-only data such as base64 images, sanitize it before it goes back into model context.

## Changing motion or camera behavior

Motion and camera code are threaded. Check these points before shipping changes:

- Startup and shutdown are symmetrical.
- Worker threads have stop events and join paths.
- Shared mutable state is protected by locks or owned by one thread.
- Robot movement still goes through `MovementManager` unless there is a strong reason.
- Long-running tool work is cancellable or routed through `BackgroundToolManager`.

## Configuration notes

The main environment variables are documented in `README.md`. In code, runtime config lives in `config.py` and can be refreshed with `refresh_runtime_config_from_env()` after loading an instance `.env`.

Backend selection is controlled by `BACKEND_PROVIDER`:

- `huggingface`
- `openai`
- `gemini`

For Hugging Face, `HF_REALTIME_CONNECTION_MODE=deployed` uses the app-managed session proxy and `local` uses `HF_REALTIME_WS_URL`.
