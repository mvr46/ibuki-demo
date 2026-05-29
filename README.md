---
title: Reachy Mini Conversation App
emoji: 🎤
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Talk with Reachy Mini!
suggested_storage: large
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Reachy Mini conversation app

Conversational app for the Reachy Mini robot combining realtime voice backends, vision pipelines, and choreographed motion libraries.

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## Table of contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the app](#running-the-app)
- [LLM tools](#llm-tools-exposed-to-the-assistant)
- [Advanced features](#advanced-features)
- [Contributing](#contributing)
- [License](#license)

## Overview
- Real-time audio conversation loop over the Reachy Mini media pipeline. Supported production backends:
  - **Local Mac** - default local-first STT/LLM/TTS path using MLX Whisper, Ollama Gemma, Qwen tool routing, and required Piper voice output.
  - **Hugging Face** - fallback backend, using the built-in Hugging Face server or your own local endpoint.
- Vision processing uses the selected realtime backend by default (when the camera tool is used), with optional on-device local vision using SmolVLM2 (CPU/GPU/MPS) via `--local-vision`.
- Layered motion system queues primary moves (dances, emotions, goto poses, breathing) while blending speech-reactive wobble and head-tracking.
- Static production dashboard handles backend selection, profile editing, face naming, diagnostics, and logs.

OpenAI Realtime and Gemini Live remain in the codebase as legacy adapters with deprecation warnings. They are hidden from the production dashboard and docs path.

## Architecture

The app follows a layered architecture connecting the user, local/Hugging Face AI services, tool registry, and robot hardware. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the current diagram and startup notes.

## Installation

> [!IMPORTANT]
> Before using this app, you need to install [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/).<br>
> Windows support is currently experimental and has not been extensively tested. Use with caution.

<details open>
<summary><b>Using uv (recommended)</b></summary>

Set up the project quickly using [uv](https://docs.astral.sh/uv/):

```bash
# macOS (Homebrew)
uv venv --python /opt/homebrew/bin/python3.12 .venv

# Linux / Windows (Python in PATH)
uv venv --python python3.12 .venv

source .venv/bin/activate
uv sync
```

> **Note:** To reproduce the exact dependency set from this repo's `uv.lock`, run `uv sync --frozen`. This ensures `uv` installs directly from the lockfile without re-resolving or updating any versions.

**Install optional features:**
```bash
uv sync --extra local_vision         # Local PyTorch/Transformers vision
uv sync --extra local_voice          # Local MLX Whisper STT + Piper TTS
uv sync --extra yolo_vision          # YOLO face-detection backend for head tracking
uv sync --extra mediapipe_vision     # MediaPipe-based head-tracking
uv sync --extra all_vision           # All vision features
```

Combine extras or include dev dependencies:
```bash
uv sync --extra all_vision --group dev
```

</details>

<details>
<summary><b>Using pip</b></summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Install optional features:**
```bash
pip install -e .[local_vision]          # Local vision stack
pip install -e .[local_voice]           # Local MLX Whisper STT + Piper TTS
pip install -e .[yolo_vision]           # YOLO face-detection backend for head tracking
pip install -e .[mediapipe_vision]      # MediaPipe-based vision
pip install -e .[all_vision]            # All vision features
pip install -e .[dev]                   # Development tools
```

Some wheels (like PyTorch) are large and require compatible CUDA or CPU builds—make sure your platform matches the binaries pulled in by each extra.

</details>

### Optional dependency groups

| Extra | Purpose | Notes |
|-------|---------|-------|
| `local_vision` | Run the local VLM (SmolVLM2) through PyTorch/Transformers | GPU recommended. Ensure compatible PyTorch builds for your platform. |
| `local_voice` | Run local STT/TTS with MLX Whisper and Piper | Apple Silicon recommended for MLX. Set `PIPER_VOICE` to a Piper `.onnx` voice file. |
| `yolo_vision` | YOLOv11n face detection via `ultralytics` and `supervision` | Used as the `yolo` speaker-focus backend. Runs on CPU (default). GPU improves performance. |
| `mediapipe_vision` | Lightweight landmark tracking with MediaPipe | Works on CPU. Enables `--head-tracker mediapipe`. |
| `all_vision` | Convenience alias installing every vision extra | Install when you want the flexibility to experiment with every provider. |
| `dev` | Developer tooling (`pytest`, `ruff`, `mypy`) | Development-only dependencies. Use `--group dev` with uv or `[dev]` with pip. |

**Note:** `dev` is a dependency group (not an optional dependency). With uv, use `--group dev`. With pip, use `[dev]`.

## Configuration

The default setup uses the local backend. For the optimized Mac mini wired setup, install a Piper voice and pull the local Ollama models:

```bash
ollama pull gemma3
ollama pull qwen3.5:4b
uv run python -m piper.download_voices --download-dir ./cache/piper-voices en_US-lessac-medium
```

Copy `.env.example` to `.env` when you want to switch to Hugging Face fallback, point Hugging Face at your own endpoint, or customize local model paths.

| Variable | Description |
|----------|-------------|
| `BACKEND_PROVIDER` | Production backend to use: `local` (default) or `huggingface`. |
| `HF_REALTIME_CONNECTION_MODE` | Hugging Face connection selector: `deployed` uses the built-in Hugging Face server; `local` uses `HF_REALTIME_WS_URL`. Defaults to `deployed`. |
| `HF_REALTIME_WS_URL` | Direct websocket endpoint for your own Hugging Face backend. Accepts either a base URL like `ws://127.0.0.1:8765/v1` or the full websocket URL `ws://127.0.0.1:8765/v1/realtime`. Used when `HF_REALTIME_CONNECTION_MODE=local`. |
| `HF_HOME` | Cache directory for local Hugging Face downloads (only used with `--local-vision` flag, defaults to `./cache`). |
| `HF_TOKEN` | Optional token for Hugging Face access (for gated/private assets). |
| `LOCAL_VISION_MODEL` | Hugging Face model path for local vision processing (only used with `--local-vision` flag, defaults to `HuggingFaceTB/SmolVLM2-2.2B-Instruct`). |
| `REACHY_MEDIA_HOST` | Optional media signaling override. Set to `10.42.0.2` for the direct Mac mini ↔ Reachy Mini wired link. |
| `OLLAMA_BASE_URL` | Local Ollama URL for the `local` backend and Ollama vision, defaults to `http://127.0.0.1:11434`. |
| `OLLAMA_MODEL` | Ollama model for local chat/vision, defaults to `gemma3:latest`. Run `ollama pull gemma3` before using the optimized local backend. |
| `OLLAMA_ROUTER_MODEL` | Compact Ollama model for local tool routing, defaults to `qwen3.5:4b`. Run `ollama pull qwen3.5:4b` for local robot tools. |
| `LOCAL_STT_MODEL` | MLX Whisper model for local STT, defaults to `mlx-community/whisper-small-mlx`. |
| `PIPER_VOICE` | Required Piper `.onnx` voice model path for local TTS, for example `./cache/piper-voices/en_US-lessac-medium.onnx`. The local backend will report an error instead of falling back to macOS `say` when this is missing. |

### Hugging Face Connection Modes

Use the built-in Hugging Face server through the app-managed Space proxy:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=deployed
```

Run your own realtime voice backend using [speech-to-speech](https://github.com/huggingface/speech-to-speech) on the same machine as the conversation app:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
```

Run your own Hugging Face backend on your laptop and connect to it from Reachy Mini Wireless over the same Wi-Fi network:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://<your-laptop-lan-ip>:8765/v1/realtime
```

For that LAN setup, make sure the backend listens on an address reachable from the robot, not only on `127.0.0.1`.

If the backend stays bound to loopback on your laptop, you can forward it into the robot over SSH instead:

```bash
ssh -N -R 8765:127.0.0.1:8765 <robot-user>@<robot-host>
```

Then set this on the robot:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
```

When using the headless settings UI, selecting `Hugging Face` lets you choose either the built-in server or a local `host:port` target. The UI writes `HF_REALTIME_CONNECTION_MODE` for you, and the local path writes `HF_REALTIME_WS_URL` with a default of `localhost:8765`.

## Running the app

Activate your virtual environment, then launch:

```bash
reachy-mini-conversation-app
```

> [!TIP]
> Make sure the Reachy Mini daemon is running before launching the app. If you see a `TimeoutError`, it means the daemon isn't started. See [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/) for setup instructions.

The app runs through the local robot audio stream. When launched as a Reachy Mini app, the static dashboard is served by the app settings server. Vision and head-tracking options are described in the CLI table below.

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--head-tracker {yolo,mediapipe}` | `None` | Select a visual head-tracking backend when a camera is available. `yolo` uses local YOLO face detection; DoA/spatial-audio steering is deprecated and disabled. `mediapipe` comes from the `reachy_mini_toolbox` package. Requires the matching optional extra. |
| `--no-camera` | `False` | Run without camera capture or head tracking. |
| `--media-backend {auto,default,local,webrtc,no_media}` | `auto` | Select the Reachy Mini SDK media backend. Use `no_media` for headless runs when camera/audio hardware is unavailable. In this app, `no_media` also disables camera, head tracking, and local vision. |
| `--local-vision` | `False` | Use the local vision model (SmolVLM2) for camera-tool requests instead of the selected realtime backend. Requires `local_vision` extra to be installed. |
| `--connection-mode {auto,localhost_only,network}` | `network` | Select how the Reachy Mini SDK connects to the daemon. Defaults to `network` so camera/audio media streams come from the robot daemon. Use `localhost_only` for local development daemons. |
| `--robot-host HOST` | `None` | Reachy Mini daemon hostname or IP address when using `--connection-mode network`. |
| `--robot-port PORT` | `None` | Reachy Mini daemon port. Uses the SDK default when omitted. |
| `--robot-name` | `None` | Optional. Connect to a specific robot by name when running multiple daemons on the same subnet. See [Multiple robots on the same subnet](#advanced-features). |
| `--hardware-profile {auto,mac-mini-wired,legacy}` | `auto` | Select hardware transport behavior. `auto` prefers `10.42.0.2` when reachable; `legacy` preserves Wi-Fi media host behavior. |
| `--debug` | `False` | Enable verbose logging for troubleshooting. |

### Examples

```bash
# Run with MediaPipe head tracking
reachy-mini-conversation-app --head-tracker mediapipe

# Run with YOLO visual face tracking
reachy-mini-conversation-app --head-tracker yolo

# Override the default robot host if reachy-mini.local is not the right target
reachy-mini-conversation-app --robot-host <robot-ip-or-hostname> --head-tracker yolo

# Optimized wired Mac mini setup
ollama pull gemma3
ollama pull qwen3.5:4b
uv run python -m piper.download_voices --download-dir ./cache/piper-voices en_US-lessac-medium
BACKEND_PROVIDER=local PIPER_VOICE=./cache/piper-voices/en_US-lessac-medium.onnx reachy-mini-conversation-app --hardware-profile mac-mini-wired --head-tracker yolo

# Run with local vision processing (requires local_vision extra)
reachy-mini-conversation-app --local-vision

# Audio-only conversation (no camera)
reachy-mini-conversation-app --no-camera

# Headless run when camera/audio hardware is unavailable
reachy-mini-conversation-app --media-backend no_media

```

> [!WARNING]
> `--local-vision` is not supported when running the conversation app directly on Reachy Mini Wireless / the Raspberry Pi. For local vision, keep the daemon running on the robot and start the conversation app from your laptop or workstation instead.

## LLM tools exposed to the assistant

| Tool | Action | Dependencies |
|------|--------|--------------|
| `move_head` | Queue a head pose change (left/right/up/down/front). | Core install only. |
| `camera` | Capture the latest camera frame and analyze it with the selected realtime backend or the local vision model. | Requires camera worker. Uses local vision when `--local-vision` is enabled. |
| `head_tracking` | Enable or disable head-tracking offsets (not identity recognition - only detects and tracks head position). | Camera worker with configured head tracker (`--head-tracker`). |
| `dance` | Queue a dance from `reachy_mini_dances_library`. | Core install only. |
| `stop_dance` | Clear queued dances. | Core install only. |
| `play_emotion` | Play a recorded emotion clip via Hugging Face datasets. | Core install only. Uses the default open emotions dataset: [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library). |
| `stop_emotion` | Clear queued emotions. | Core install only. |
| `idle_do_nothing` | Explicitly remain idle during an idle turn. Not intended for normal conversation turns. | Core install only. |

## Advanced features

Built-in motion content is published as open Hugging Face datasets:
- Emotions: [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library)
- Dances: [`pollen-robotics/reachy-mini-dances-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-dances-library)

<details>
<summary><b>Custom profiles</b></summary>

Create custom profiles with dedicated instructions and enabled tools.

For normal usage, select a profile from the UI and save it for startup. That selection is persisted in `startup_settings.json`.

If no startup settings have been saved yet, you can still seed startup from the environment with `REACHY_MINI_CUSTOM_PROFILE=<name>` to load `src/reachy_mini_conversation_app/profiles/<name>/`. If neither is set, the `default` profile is used.

Each profile should include:

- `instructions.txt`: the production prompt text.
- `tools.txt`: enabled core tool names, one per line.
- `voice.txt`: optional backend voice preference.

Profiles do not load Python files and there is no external tool autoloading in the production path. Tool implementations live under `src/reachy_mini_conversation_app/tools/`; profiles only select which core tools are exposed.

**Enabling tools:**

List enabled tools in `tools.txt`, one per line. Prefix with `#` to comment out:
```
play_emotion
# move_head
```

**Edit profiles from the dashboard:**

The production dashboard can list profiles, load a profile, save a new profile, overwrite the selected profile, apply the current profile, and apply voice changes.

Prompt and voice changes apply live. Tool-list changes are saved to `tools.txt` and take effect after restart.

For the local backend, the dashboard voice is the logical `local` session voice. The audible speaker is the Piper model file selected with `PIPER_VOICE` when the app starts.

</details>

<details>
<summary><b>Locked profile mode</b></summary>

To create a locked variant of the app that cannot switch profiles, edit `src/reachy_mini_conversation_app/runtime/config.py` and set the `LOCKED_PROFILE` constant to the desired profile name:
```python
LOCKED_PROFILE: str | None = "mars_rover"  # Lock to this profile
```
When `LOCKED_PROFILE` is set, the app always uses that profile, ignoring saved startup settings, `REACHY_MINI_CUSTOM_PROFILE`, and dashboard profile switching. The UI shows "(locked)" and disables profile editing controls.
This is useful for creating dedicated clones of the app with a fixed profile. Clone scripts can simply edit this constant to lock the variant.

</details>

<details>
<summary><b>Multiple robots on the same subnet</b></summary>

If you run multiple Reachy Mini daemons on the same network, use:

```bash
reachy-mini-conversation-app --robot-name <name>
```

`<name>` must match the daemon's `--robot-name` value so the app connects to the correct robot.

</details>

## Contributing

We welcome bug fixes, features, profiles, and documentation improvements. Please review our
[contribution guide](CONTRIBUTING.md) for branch conventions, quality checks, and PR workflow.

Quick start:
- Fork and clone the repo
- Follow the [installation steps](#installation) (include the `dev` dependency group)
- Run contributor checks listed in [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache 2.0
