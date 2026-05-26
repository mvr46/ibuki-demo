# Face recognition integration spec

**Status:** Plan, ready for implementation.
**Author handoff:** Self-contained — implementer needs only this doc and the repo.
**Date:** 2026-05-26

## 1. Goal

Add face *identity* to the existing face-detection pipeline (YOLO `YoloHeadTracker` already finds faces and feeds spatial speaker tracking) so the LLM running the conversation always knows who it is looking at. Identities flow into the LLM's input stream as ambient `[Vision: …]` user-role messages — the model never has to call a tool to know who is in front of it, though it can. Naming new people happens conversationally via a tool, not via a terminal prompt.

## 2. What already exists

- **Reference CLI** at [`examples/face_recognition/`](../examples/face_recognition/). Two scripts (`run.sh`, `recognize.sh`) demonstrate the InsightFace pipeline this spec adopts: static-image detection + recognition + age/sex on `t1.jpg`, and a webcam loop with a multi-exemplar pickled face DB. The `FaceDB` class, IoU `Tracker`, and matching loop in [`recognize.py`](../examples/face_recognition/recognize.py) are the working code Phase 1 ports into the package.
- **InsightFace** is installed from PyPI (`insightface>=0.7`). No local checkout required. The package vendors its own ONNX models, downloading buffalo_l into `~/.insightface/models/` on first use.
- **`CameraWorker`** at [src/reachy_mini_conversation_app/camera_worker.py](../src/reachy_mini_conversation_app/camera_worker.py) — runs the camera thread, buffers BGR frames behind `get_latest_frame()`, calls `head_tracker.get_head_targets(frame) -> list[HeadTrackerTarget]` for the spatial speaker selector.
- **`HeadTrackerTarget`** at [vision/head_tracking/__init__.py:13](../src/reachy_mini_conversation_app/vision/head_tracking/__init__.py#L13) — frozen dataclass with `x_offset`, `y_offset`, `confidence`, `bbox` (normalized `(x, y, w, h)`), `frame_size`. **All we need to crop a face out of a frame for InsightFace recognition.**
- **`YoloHeadTracker`** at [vision/head_tracking/yolo.py:33](../src/reachy_mini_conversation_app/vision/head_tracking/yolo.py#L33) — uses `AdamCodd/YOLOv11n-face-detection`, returns `list[HeadTrackerTarget]`. Already loaded when `--head-tracker yolo` is in effect.
- **`ToolDependencies`** at [tools/core_tools.py:50](../src/reachy_mini_conversation_app/tools/core_tools.py#L50) — DI carrier passed to every tool. Tools are auto-registered as concrete `Tool` subclasses; `get_active_tool_specs(deps)` filters by which deps are present.
- **Environment-message injection precedent.** `send_idle_signal` at [base_realtime.py:1033](../src/reachy_mini_conversation_app/base_realtime.py#L1033) sends `connection.conversation.item.create(item={type:"message", role:"user", content:[{type:"input_text", text:"[Idle time update: …]"}]})` then `_safe_response_create(...)`. Gemini's equivalent is `session.send_realtime_input(text=…)` at [gemini_live.py:755](../src/reachy_mini_conversation_app/gemini_live.py#L755). These are the hooks we reuse for streaming perception.

## 3. Architecture

```
ReachyMini.media.get_frame()  ─► CameraWorker (25 Hz)
                                       │
                                       ├──► YoloHeadTracker.get_head_targets(frame)
                                       │      └─► list[HeadTrackerTarget]
                                       │
                                       ▼
                              FaceIdentifierWorker (2–3 Hz)
                                pulls latest frame + targets
                                for each target: crop bbox, embed, match FaceDB
                                        │
                                        ▼
                              PerceptionState (thread-safe)
                                visible: list[IdentifiedTarget]
                                last_seen: dict[name, monotonic_ts]
                                events:    deque[VisionEvent]
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
who_is_here tool             perception_stream task            speaker selector
remember_person tool         pushes [Vision: …]                (Phase 3) reads
                             events + scene snapshots          prefer_name hint
                             via handler.inject_environment_message()
```

## 4. File manifest

### New files in `src/reachy_mini_conversation_app/`

| File | Role |
|---|---|
| `vision/face_recognition_lib.py` | Library ported from [`examples/face_recognition/recognize.py`](../examples/face_recognition/recognize.py). Exports `FaceDB`, `Person`, `iou()`, `Tracker`, and a new `FaceRecognizer` class wrapping the **recognition arm only** (landmark + embed; no detection). |
| `vision/face_identity.py` | Adapter. `IdentifiedTarget = (target: HeadTrackerTarget, name: str \| None, similarity: float, embedding: np.ndarray)`. `FaceIdentifier.identify(frame, targets) -> list[IdentifiedTarget]`. |
| `face_identity_worker.py` | Background thread (Phase 2). Polls `camera_worker.get_latest_frame()` + `head_tracker.get_head_targets()` at 2–3 Hz, calls `FaceIdentifier.identify`, updates `PerceptionState`, emits events on changes. Mirrors `CameraWorker`'s `start`/`stop` shape. |
| `perception_stream.py` | Async injector (Phase 2). One coroutine that consumes `PerceptionState.events` and a debounced "scene-changed" timer; formats `[Vision: …]` strings; calls `handler.inject_environment_message(text)`. |
| `tools/who_is_here.py` | `Tool` returning `[{name, x_offset, y_offset, similarity, seconds_in_view}, …]`. |
| `tools/remember_person.py` | `Tool` taking `{name: str}`. Snapshots the largest currently-unknown `IdentifiedTarget` from the worker, calls `FaceDB.add(name, embedding)`. Returns exemplar count. |

### Edits to existing files

| File | Change |
|---|---|
| [tools/core_tools.py:50](../src/reachy_mini_conversation_app/tools/core_tools.py#L50) | Add `face_identity_worker: Any \| None = None` to `ToolDependencies`. In `get_active_tool_specs`, append `who_is_here` and `remember_person` to the exclusion list when worker is `None`. |
| [base_realtime.py:1033](../src/reachy_mini_conversation_app/base_realtime.py#L1033) | Extract the `connection.conversation.item.create(...) + _safe_response_create(...)` pattern into `async def inject_environment_message(self, text: str, *, trigger_response: bool = False)`. `send_idle_signal` becomes a one-liner over it. |
| [gemini_live.py:742](../src/reachy_mini_conversation_app/gemini_live.py#L742) | Add the matching `inject_environment_message` over `session.send_realtime_input(text=…)`. |
| [conversation_handler.py](../src/reachy_mini_conversation_app/conversation_handler.py) | Add `inject_environment_message` to the abstract `ConversationHandler` interface so `perception_stream` can call it polymorphically. |
| [main.py](../src/reachy_mini_conversation_app/main.py) | Instantiate `FaceIdentifierWorker` after `CameraWorker`, pass to `ToolDependencies`, start it. Launch `perception_stream.run(state, handler)` as an asyncio task in the same place handlers are spawned. Clean both up on shutdown. |
| `profiles/default/tools.txt` | Add `who_is_here` and `remember_person`. |
| `profiles/default/instructions.txt` (or new `prompts/face_awareness.txt` include) | Add a short paragraph (see §7). |
| `pyproject.toml` | New `[project.optional-dependencies]` extra `face_recognition` installing `insightface>=0.7` plus `onnxruntime`. |

## 5. Phased delivery

Each phase is independently shippable.

### Phase 1 — Tools-only, synchronous (≈200 LOC)

**Goal:** The LLM can answer "who do you see?" and learn names by being told.

1. Port the reference code from [`examples/face_recognition/recognize.py`](../examples/face_recognition/recognize.py) into a new module `src/reachy_mini_conversation_app/vision/face_recognition_lib.py`. Copy `FaceDB`, `Person`, `iou()`, `Tracker` verbatim, change the default DB path to `~/.reachy-mini/faces.db`. Add `FaceRecognizer`:

   ```python
   # vision/face_recognition_lib.py
   class FaceRecognizer:
       """Recognition arm only: landmark → align → embed. No detection."""
       def __init__(self, name: str = "buffalo_l"):
           from insightface import model_zoo
           models_dir = Path.home() / ".insightface" / "models" / name
           # buffalo_l auto-downloads on first FaceAnalysis() init; if absent, prompt
           # the user to run `insightface-cli model.download buffalo_l` or run the
           # example CLI once.
           self._landmarker = model_zoo.get_model(str(models_dir / "2d106det.onnx"))
           self._embedder   = model_zoo.get_model(str(models_dir / "w600k_r50.onnx"))
           self._landmarker.prepare(ctx_id=-1)
           self._embedder.prepare(ctx_id=-1)

       def embed(self, frame_bgr: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray | None:
           """Crop, align to 112x112 via 5pt landmarks, return 512-D normed embedding."""
           # Use insightface.utils.face_align.norm_crop with the landmarker's kps output.
           # Return None on failure (e.g. landmark detector returns no kps).
   ```

   The example CLI (`examples/face_recognition/recognize.sh`) keeps working unchanged — it uses the full `FaceAnalysis` pipeline. The package version uses recognition-only on top of YOLO bboxes.

2. Create `vision/face_identity.py`:

   ```python
   @dataclass(frozen=True)
   class IdentifiedTarget:
       target: HeadTrackerTarget
       name: str | None
       similarity: float
       embedding: np.ndarray  # for remember_person

   class FaceIdentifier:
       def __init__(self, recognizer: FaceRecognizer, db: FaceDB, threshold: float = 0.4):
           ...
       def identify(self, frame_bgr, targets: list[HeadTrackerTarget]) -> list[IdentifiedTarget]:
           # For each target: convert normalized bbox→pixel xyxy, call recognizer.embed,
           # FaceDB.match, package.
   ```

3. Implement the two tools:

   ```python
   # tools/who_is_here.py
   class WhoIsHere(Tool):
       name = "who_is_here"
       description = "List the people currently visible to the robot's camera."
       parameters_schema = {"type": "object", "properties": {}, "additionalProperties": False}
       async def __call__(self, deps, **_):
           cw, hw = deps.camera_worker, deps.face_identity_worker  # Phase 1: use deps.camera_worker only
           frame = cw.get_latest_frame()
           targets = cw.head_tracker.get_head_targets(frame) if cw.head_tracker else []
           identified = hw.identifier.identify(frame, targets)
           return {"people": [
               {"name": it.name, "x_offset": it.target.x_offset, "y_offset": it.target.y_offset,
                "similarity": round(it.similarity, 3)}
               for it in identified
           ]}
   ```

   ```python
   # tools/remember_person.py
   class RememberPerson(Tool):
       name = "remember_person"
       description = "Save the largest currently-unknown face under the given name."
       parameters_schema = {"type": "object",
           "properties": {"name": {"type": "string", "minLength": 1}},
           "required": ["name"], "additionalProperties": False}
       async def __call__(self, deps, *, name):
           # Pull latest frame, find largest unknown, db.add(name, embedding).
   ```

4. Add `face_identity_worker` field to `ToolDependencies`. In Phase 1 it holds a lightweight object that just owns `FaceIdentifier + FaceDB` (no background thread yet) so the tools work without the worker existing.

5. Update `profiles/default/tools.txt`, add the instructions paragraph (§7), add the `face_recognition` pyproject extra.

**Phase 1 acceptance:**
- `uv sync --extra face_recognition` installs without errors.
- Existing tests pass.
- New unit tests for `FaceRecognizer.embed` (mocked frame) and `FaceIdentifier.identify` (mocked targets + DB) pass.
- Manual: run the robot, ask "who do you see?", model calls `who_is_here`, returns names if any DB entries match; introduce yourself, model calls `remember_person`, recognition works on the next ask.

### Phase 2 — Background worker + streaming injection (≈250 LOC)

**Goal:** The robot says "Hi Alice!" when Alice walks in front of it, unprompted.

1. Promote `face_identity_worker` to a real `FaceIdentifierWorker` thread:

   ```python
   # face_identity_worker.py
   class FaceIdentifierWorker:
       def __init__(self, camera_worker, identifier: FaceIdentifier, *, rate_hz: float = 2.5):
           self.camera_worker = camera_worker
           self.identifier = identifier
           self._state = PerceptionState()
           self._stop = threading.Event()
           self._thread = None

       def start(self): ...
       def stop(self):  ...
       def snapshot(self) -> PerceptionSnapshot: ...  # thread-safe deep copy
       def drain_events(self) -> list[VisionEvent]: ...  # consume the queue
   ```

   `PerceptionState` is `(visible: list[IdentifiedTarget], last_seen: dict[str, float], events: deque[VisionEvent])` behind one lock. Events: `Entered(name|None)`, `Left(name|None, last_seen)`, `Named(name)`. Track presence by IoU tracking (reuse `Tracker` from `perception_lib.py`).

2. Add the abstraction to `ConversationHandler`:

   ```python
   # conversation_handler.py
   class ConversationHandler(Protocol):
       async def inject_environment_message(self, text: str, *, trigger_response: bool = False) -> None: ...
   ```

   Implement on `BaseRealtimeHandler` (extract from `send_idle_signal`) and on `GeminiLiveHandler` (over `session.send_realtime_input(text=…)`).

3. Create `perception_stream.py`:

   ```python
   async def run_perception_stream(
       worker: FaceIdentifierWorker,
       handler: ConversationHandler,
       *,
       snapshot_interval_s: float = 12.0,
       event_debounce_s: float = 1.5,
   ) -> None:
       """Drain perception events and emit periodic scene snapshots."""
       last_snapshot_at = 0.0
       last_snapshot_set: frozenset[str] = frozenset()
       while True:
           for event in worker.drain_events():
               text = _format_event(event, worker.snapshot())  # "[Vision: Alice entered the frame]"
               await handler.inject_environment_message(text)
               await asyncio.sleep(event_debounce_s)

           now = time.monotonic()
           snap = worker.snapshot()
           current_set = frozenset(p.name for p in snap.visible if p.name)
           if (now - last_snapshot_at >= snapshot_interval_s) and current_set != last_snapshot_set:
               await handler.inject_environment_message(_format_snapshot(snap))
               last_snapshot_at = now
               last_snapshot_set = current_set
           await asyncio.sleep(0.5)
   ```

   Message formats:
   - **Event:** `[Vision: Alice entered the frame (center)]` / `[Vision: Bob left, last seen 30s ago]` / `[Vision: An unknown person is now visible]`
   - **Snapshot:** `[Vision: Visible now — Alice (center), unknown (right). Last seen recently: Bob (left 4 min ago)]`
   - **Suppress while assistant is speaking** — check `camera_worker._assistant_speaking` (or wire a similar flag onto the handler) and skip injections until it clears. We don't want the model to interrupt itself.

4. Wire `main.py`: instantiate worker, start its thread, launch `run_perception_stream` task next to where handler tasks start, register graceful shutdown.

**Phase 2 acceptance:**
- The robot greets a known person by name within a few seconds of them entering the frame without anyone speaking first.
- The realtime log shows `[Vision: …]` items being added; the assistant references them in its replies.
- No injections fire while the assistant is talking.
- Existing tests pass; new tests cover `PerceptionState` event emission and `perception_stream` formatting and debouncing.

### Phase 3 — Identity-aware speaker selection + look_at_person (≈100 LOC)

**Goal:** "Look at Alice" actually turns the head to Alice. The active speaker picker uses identity as a hint.

1. Extend `select_speaker()` in [vision/head_tracking/speaker.py](../src/reachy_mini_conversation_app/vision/head_tracking/speaker.py) to accept an optional `prefer_name: str | None`. When present and any of the candidate `HeadTrackerTarget`s correspond (via IoU to the latest `PerceptionSnapshot.visible`) to that name, add a `name_match_bonus` to its score.
2. Plumb `prefer_name` from the worker into `CameraWorker._update_tracking_from_frame` via a setter on `CameraWorker` (`set_speaker_focus_name(name | None)`).
3. Add a new tool `look_at_person`:

   ```python
   class LookAtPerson(Tool):
       name = "look_at_person"
       description = "Turn the head toward the named visible person."
       parameters_schema = {"type": "object",
           "properties": {"name": {"type": "string"}}, "required": ["name"]}
       async def __call__(self, deps, *, name):
           # Validate the name is currently visible; call camera_worker.set_speaker_focus_name(name).
           # Return {"status": "looking_at", "name": name, "x_offset": ..., "y_offset": ...}
   ```

**Phase 3 acceptance:** "Look at Alice" turns the head; the speaker selector prefers Alice's bbox over a louder unknown when both are visible.

## 6. The injection mechanism (detail)

**Two cadences, one hook.**

| Trigger | Message | Cadence | Suppressed when |
|---|---|---|---|
| **Event** (new face enters / known leaves / face named) | `[Vision: Alice entered the frame (center)]` | On change, ≥1.5 s gap between events | Assistant is speaking |
| **Scene snapshot** (full visible list) | `[Vision: Visible now — Alice (center), unknown (right). Last seen recently: Bob (left 4 min ago)]` | Every 12 s **only if visible set changed since last snapshot** | Assistant is speaking |

Both flow through `handler.inject_environment_message(text)`. Internally:

- **OpenAI / HuggingFace realtime:** `connection.conversation.item.create({type:"message", role:"user", content:[{type:"input_text", text}]})`. Do not call `response.create` — the perception item just sits in context for the next turn.
- **Gemini Live:** `session.send_realtime_input(text=text)`. Gemini treats this as a user input augmenting the conversation.

**Why role=user, not system:** [base_realtime.py:1044](../src/reachy_mini_conversation_app/base_realtime.py#L1044) already does this for idle/timestamp signals. The realtime APIs have stronger support for user-role environment items mid-session than for ad-hoc system items. We follow the established convention; the model is trained to treat bracketed prefixes (`[Idle …]`, `[Vision: …]`) as environment, not user speech.

**DoA hint:** since the speaker selector already exposes which `HeadTrackerTarget` is "the active speaker" via audio+visual fusion, the snapshot formatter tags that target with `(talking)`. The model gets identity AND who's talking in one line: `[Vision: Alice (center, talking), unknown (right)]`.

## 7. System prompt addition

Append to `profiles/default/instructions.txt` (or include as `prompts/face_awareness.txt`):

```
## Face awareness

The robot has face recognition. You will receive ambient updates like
`[Vision: Alice entered the frame (center)]` or `[Vision: Visible now — Alice (center, talking), unknown (right)]`.
These are environment signals, not user speech. Use them to:
  - Address known people by name when you greet or reply to them.
  - Mention what you see when it's natural (don't narrate constantly).
  - When an unknown face is visible and the user introduces themselves
    (e.g. "Hi, I'm Bob"), call `remember_person(name="Bob")` to save
    their face, then continue the conversation. Don't ask permission first.
You can also call `who_is_here` at any time to get the current list with details.
```

## 8. Open decisions

1. **Phase 1+2 in one PR, or Phase 1 alone first.** Phase 2 is the streaming-injection feature the user asked for; Phase 1 is a worthwhile checkpoint. Default: ship 1+2 together.
2. **FaceDB location.** Default `~/.reachy-mini/faces.db` (per-user, survives repo deletes; both the example CLI and the package read/write the same path). The example CLI's local `./faces.db` is kept as a fallback when `--db` is passed explicitly.
3. **Model size.** Recognition-only on YOLO bboxes is fast even with `buffalo_l` (~25 ms per face on CPU). No need to switch to `buffalo_s` unless we discover latency pressure.
4. **What to do when YOLO is disabled** (e.g. `--head-tracker mediapipe` or `none`). Mediapipe returns one target; we still try identifying it. None → tools simply return empty / "no camera." Phase 1 handles this gracefully via the `get_active_tool_specs` filter.

## 9. Out of scope (intentionally)

- Multi-camera. The robot has one head camera.
- Recognizing people in side profile / heavy occlusion. The buffalo_l model is best on near-frontal faces; this matches Reachy's gaze use case.
- Identity confidence calibration / re-identification rerankers. The simple cosine threshold at 0.4 is good enough for a small known set; revisit if false matches become a problem.
- GUI for managing the FaceDB. The standalone `insightface-gui` works; if the user wants in-conversation management, that's a follow-up.
