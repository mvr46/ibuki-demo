# Project Context

## What this project is

Reachy Mini Conversation App is a Python application for running realtime voice conversations on the Reachy Mini robot. It connects microphone/audio streaming, realtime AI backends, camera/vision features, and robot motion so the assistant can talk, see through the robot camera, move its head, dance, play emotions, and track faces.

The package is installed as `reachy_mini_conversation_app` and exposes the console command:

```bash
reachy-mini-conversation-app
```

The default backend is local (`BACKEND_PROVIDER=local`): MLX Whisper for STT, llama.cpp/OpenAI-compatible servers for chat, Qwen tool routing, and fast vision, plus Piper for TTS. Hugging Face remains the production fallback. OpenAI Realtime and Gemini Live remain legacy adapters with deprecation warnings.

## Important paths

| Path | Purpose |
| --- | --- |
| `src/reachy_mini_conversation_app/main.py` | Application entrypoint. Builds the robot, camera, motion manager, tool dependencies, selected conversation handler, and local stream/dashboard integration. |
| `src/reachy_mini_conversation_app/backends/` | Conversation backend interface, factory, local backend, Hugging Face adapter, and legacy OpenAI/Gemini adapters. |
| `src/reachy_mini_conversation_app/backends/base_realtime.py` | Shared handler for OpenAI-compatible realtime backends. Handles audio streaming, transcript updates, tool calls, reconnection, response queueing, and cost tracking. |
| `src/reachy_mini_conversation_app/backends/factory.py` | Backend factory. Keeps `main.py` focused on composition and hides legacy adapter selection behind one small interface. |
| `src/reachy_mini_conversation_app/backends/interface.py` | Backend interface expected by the local robot audio stream and dashboard code. |
| `src/reachy_mini_conversation_app/runtime/` | Runtime config, CLI helpers, local stream, dashboard routes, diagnostics, transport, and startup settings. |
| `src/reachy_mini_conversation_app/motion/` | Robot motion manager and queued dance/emotion moves. |
| `src/reachy_mini_conversation_app/tools/` | Function-call tools exposed to the realtime assistant. |
| `src/reachy_mini_conversation_app/vision/` | Camera capture, perception stream, face identity, local vision, and head-tracking implementations. |
| `src/reachy_mini_conversation_app/profiles/` | Profile store, prompt loading, profile dashboard routes, and curated production profile data. |
| `tests/` | Unit tests for backend adapters, config, tools, startup settings, camera, vision, and audio helpers. |

## Runtime flow

1. `main.run()` parses CLI/runtime settings and connects to `ReachyMini`.
2. Camera and optional local vision/head tracking are initialized from CLI flags.
3. `MovementManager` starts the robot motion control surface.
4. `ToolDependencies` collects robot, motion, camera, vision, and audio-reactive motion dependencies for tools.
5. The selected backend handler is created:
   - Local: `LocalConversationHandler`
   - Hugging Face: `HuggingFaceRealtimeHandler`
   - OpenAI/Gemini: legacy adapters only
6. `LocalStream` moves robot audio frames into the handler and plays returned audio/transcript outputs.
7. When the model calls a tool, `ToolRegistry` dispatches it against the injected dependencies.
8. Tools queue robot movements, capture/analyze camera frames, toggle head tracking, or manage background tasks.

## Design principles already visible in the code

- Backend handlers share a common `ConversationHandler` interface.
- OpenAI-compatible backends inherit most behavior from `BaseRealtimeHandler`.
- Tools are loaded through an explicit `ToolRegistry` for the active profile and dependency-injected through `ToolDependencies`.
- Motion has one robot output point: `MovementManager` calls `ReachyMini.set_target` from its own worker thread.
- Primary moves are sequential; speech wobble and face tracking are secondary offsets blended on top.
- Camera capture is isolated in a worker thread and exposes snapshots through thread-safe accessors.
- Profiles select instructions, voice, and enabled core tools without changing backend code.

## Working assumptions for contributors and agents

- Prefer `uv` for environment management and test commands.
- Keep changes scoped. The app coordinates realtime networking, hardware, audio, and threads, so small regressions can be hard to diagnose.
- Avoid platform-specific code unless it has a clear fallback. The project targets Linux, macOS, and Windows.
- Do not commit `.env` files or API keys.
- When changing realtime behavior, test at least the relevant backend unit tests and any shared `base_realtime` behavior.
- When changing movement or camera threading, look for lifecycle cleanup paths as carefully as happy paths.
