# Architecture Notes

## High-level system

The app is a realtime conversation bridge between four domains:

- Audio/UI streaming through FastRTC and optional Gradio UI.
- AI realtime backends that produce speech, transcripts, and function calls.
- Robot capabilities through the Reachy Mini SDK.
- Local side systems such as camera capture, face tracking, local vision, motion blending, and background tool execution.

```mermaid
flowchart LR
    User["User audio"] --> FastRTC["FastRTC Stream"]
    FastRTC --> Handler["ConversationHandler"]
    Handler --> Backend["Realtime backend"]
    Backend --> Handler
    Handler --> ToolDispatch["Tool dispatch"]
    ToolDispatch --> Tools["App tools"]
    Tools --> Motion["MovementManager"]
    Tools --> Camera["CameraWorker"]
    Tools --> Vision["Vision processor or backend vision"]
    Motion --> Robot["Reachy Mini SDK"]
    Camera --> Robot
    Handler --> FastRTC
    FastRTC --> UserOut["Assistant audio + transcripts"]
```

## Startup composition

`src/reachy_mini_conversation_app/main.py` is intentionally the composition root. It wires together runtime dependencies and keeps backend/tool code from owning application startup.

Key startup decisions:

- `ReachyMini` is created unless a robot object is injected by the hosting app.
- Simulation mode automatically enables Gradio.
- `initialize_camera_and_vision()` creates `CameraWorker`, optional head tracker, and optional local vision processor.
- `MovementManager` receives the robot and camera worker.
- `HeadWobbler` feeds speech-reactive offsets into the movement manager.
- `ToolDependencies` is passed into the selected realtime handler.
- The selected handler is mounted in either console mode or Gradio/FastAPI mode.

## Backend layer

The shared contract is `ConversationHandler` in `conversation_handler.py`. All handlers must provide lifecycle, audio receive/emit, personality, and voice methods.

`BaseRealtimeHandler` implements the common OpenAI-compatible behavior:

- session startup and restart
- audio input/output queueing
- transcript output updates
- response serialization
- function-call dispatch
- background tool notifications
- reconnection handling
- cost tracking
- voice switching

OpenAI and Hugging Face mostly provide backend-specific session config and client creation. Gemini has a separate handler because the Gemini Live SDK has different session/event semantics.

## Tool system

Tools are subclasses of `Tool` in `tools/core_tools.py`. Each tool provides:

- `name`
- `description`
- `parameters_schema`
- async `__call__(deps, **kwargs)`

At import time, `core_tools` loads the selected profile's `tools.txt`, imports profile-local tools first, then shared tools from `reachy_mini_conversation_app.tools`, and finally optional external tools.

Tool dispatch accepts model-provided JSON arguments, parses them defensively, and invokes the registered tool with `ToolDependencies`. This keeps hardware and runtime services explicit:

- `reachy_mini`
- `movement_manager`
- `camera_worker`
- `vision_processor`
- `head_wobbler`
- default motion duration

`get_active_tool_specs()` filters out tools whose dependencies are unavailable. For example, `head_tracking` is hidden unless a camera worker and head tracker exist.

## Motion model

`MovementManager` owns robot movement. Other parts of the app should queue commands or set offsets through its public API instead of calling robot motion directly.

The model separates movement into:

- Primary moves: dances, emotions, goto poses, breathing. These are mutually exclusive and sequential.
- Secondary offsets: speech wobble and face tracking. These are additive and blended on top of the active primary pose.

The worker thread is the single control point that calls `ReachyMini.set_target`. This is important because tool calls, camera tracking, and audio callbacks can happen concurrently.

## Camera and vision

`CameraWorker` continuously reads frames from `reachy_mini.media.get_frame()` on a background thread. It stores the latest frame behind a lock so tools can grab a snapshot without owning camera capture.

If a head tracker is configured, the worker converts detected face/eye position into offsets using `reachy_mini.look_at_image(..., perform_movement=False)`. Those offsets are smoothed back to neutral after face loss.

Vision for the `camera` tool can run in two ways:

- Default: selected realtime backend handles image analysis.
- `--local-vision`: local SmolVLM2-based vision processor handles camera requests.

## Profiles and prompts

Profiles are personality/runtime bundles. A profile can define:

- `instructions.txt`: prompt content.
- `tools.txt`: tool names to expose.
- `voice.txt`: optional backend voice preference.
- `<tool_name>.py`: optional profile-local tool implementation.

Prompt files can include shared prompt fragments with lines like:

```text
[identities/basic_info]
```

`prompts.py` expands those includes from `src/reachy_mini_conversation_app/prompts/`.

