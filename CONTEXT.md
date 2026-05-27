# Project Context

## What this project is

Reachy Mini Conversation App is a Python application for running realtime voice conversations on the Reachy Mini robot. It connects microphone/audio streaming, realtime AI backends, camera/vision features, and robot motion so the assistant can talk, see through the robot camera, move its head, dance, play emotions, and track faces.

The package is installed as `reachy_mini_conversation_app` and exposes the console command:

```bash
reachy-mini-conversation-app
```

The default backend is Hugging Face. OpenAI Realtime and Gemini Live are also supported through environment configuration.

## Important paths

| Path | Purpose |
| --- | --- |
| `src/reachy_mini_conversation_app/main.py` | Application entrypoint. Builds the robot, camera, motion manager, tool dependencies, selected realtime handler, and console/Gradio UI. |
| `src/reachy_mini_conversation_app/base_realtime.py` | Shared handler for OpenAI-compatible realtime backends. Handles audio streaming, transcript updates, tool calls, reconnection, response queueing, and cost tracking. |
| `src/reachy_mini_conversation_app/openai_realtime.py` | OpenAI Realtime backend adapter. |
| `src/reachy_mini_conversation_app/huggingface_realtime.py` | Hugging Face OpenAI-compatible realtime backend adapter. |
| `src/reachy_mini_conversation_app/gemini_live.py` | Gemini Live backend adapter. It implements the common conversation handler contract directly because Gemini uses a different SDK shape. |
| `src/reachy_mini_conversation_app/conversation_handler.py` | Backend interface expected by FastRTC and app UI code. |
| `src/reachy_mini_conversation_app/moves.py` | Robot motion manager. Owns the motion thread and blends primary moves with secondary offsets. |
| `src/reachy_mini_conversation_app/camera_worker.py` | Camera capture loop and optional face-tracking offset producer. |
| `src/reachy_mini_conversation_app/tools/` | Function-call tools exposed to the realtime assistant. |
| `src/reachy_mini_conversation_app/vision/` | Local vision and head-tracking implementations. |
| `src/reachy_mini_conversation_app/prompts.py` | Prompt/profile loading and prompt include expansion. |
| `profiles/` | Built-in personality profiles. Each profile usually has `instructions.txt` and `tools.txt`. |
| `external_content/` | Starter examples for custom profiles and tools. |
| `tests/` | Unit tests for backend adapters, config, tools, startup settings, camera, vision, and audio helpers. |

## Runtime flow

1. `main.run()` parses CLI/runtime settings and connects to `ReachyMini`.
2. Camera and optional local vision/head tracking are initialized from CLI flags.
3. `MovementManager` starts the robot motion control surface.
4. `ToolDependencies` collects robot, motion, camera, vision, and audio-reactive motion dependencies for tools.
5. The selected backend handler is created:
   - Hugging Face: `HuggingFaceRealtimeHandler`
   - OpenAI: `OpenaiRealtimeHandler`
   - Gemini: `GeminiLiveHandler`
6. FastRTC streams audio into the handler and receives audio/transcript outputs.
7. When the model calls a tool, `tools.core_tools` dispatches it against the injected dependencies.
8. Tools queue robot movements, capture/analyze camera frames, toggle head tracking, or manage background tasks.

## Design principles already visible in the code

- Backend handlers share a common `ConversationHandler` interface.
- OpenAI-compatible backends inherit most behavior from `BaseRealtimeHandler`.
- Tools are dependency-injected through `ToolDependencies` instead of reaching for globals.
- Motion has one robot output point: `MovementManager` calls `ReachyMini.set_target` from its own worker thread.
- Primary moves are sequential; speech wobble and face tracking are secondary offsets blended on top.
- Camera capture is isolated in a worker thread and exposes snapshots through thread-safe accessors.
- Profiles select instructions and tools without changing backend code.

## Working assumptions for contributors and agents

- Prefer `uv` for environment management and test commands.
- Keep changes scoped. The app coordinates realtime networking, hardware, audio, and threads, so small regressions can be hard to diagnose.
- Avoid platform-specific code unless it has a clear fallback. The project targets Linux, macOS, and Windows.
- Do not commit `.env` files or API keys.
- When changing realtime behavior, test at least the relevant backend unit tests and any shared `base_realtime` behavior.
- When changing movement or camera threading, look for lifecycle cleanup paths as carefully as happy paths.

