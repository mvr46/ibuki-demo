"""Static dashboard APIs for camera-based face naming and live logs."""

from __future__ import annotations
import json
import time
import asyncio
import logging
import threading
from typing import Any, Callable
from datetime import datetime, timezone
from collections import deque
from dataclasses import field, dataclass

from fastapi import Request

from reachy_mini_conversation_app.vision.camera_frame_encoding import encode_bgr_frame_as_jpeg


logger = logging.getLogger(__name__)

DASHBOARD_RUNTIME_STATE_ATTR = "_reachy_mini_dashboard_runtime_state"
DASHBOARD_ROUTES_MOUNTED_ATTR = "_reachy_mini_dashboard_routes_mounted"


@dataclass(frozen=True)
class DashboardLogEvent:
    """One structured dashboard log event."""

    id: int
    created_at: str
    level: str
    category: str
    message: str
    logger_name: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable event payload."""
        return {
            "id": self.id,
            "type": "log",
            "createdAt": self.created_at,
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "logger": self.logger_name,
        }


class DashboardLogBuffer(logging.Handler):
    """Bounded logging handler used by the static dashboard."""

    def __init__(self, *, capacity: int = 500, logger_name: str = "reachy_mini_conversation_app") -> None:
        """Initialize a bounded event buffer."""
        super().__init__(level=logging.DEBUG)
        self.capacity = max(1, int(capacity))
        self.logger_name = logger_name
        self._events: deque[DashboardLogEvent] = deque(maxlen=self.capacity)
        self._condition = threading.Condition()
        self._next_id = 1
        self._installed = False

    def install(self) -> None:
        """Attach the buffer to the app logger once."""
        if self._installed:
            return
        app_logger = logging.getLogger(self.logger_name)
        app_logger.addHandler(self)
        app_logger.setLevel(min(app_logger.level or logging.INFO, logging.DEBUG))
        self._installed = True

    def emit(self, record: logging.LogRecord) -> None:
        """Store a log record as a structured dashboard event."""
        try:
            message = clean_log_message(record.getMessage())
            if not message:
                return
            self.add(
                message,
                level=record.levelname,
                category=classify_log(record.name, message),
                logger_name=record.name,
            )
        except Exception:
            self.handleError(record)

    def add(
        self,
        message: str,
        *,
        level: str = "INFO",
        category: str | None = None,
        logger_name: str = "dashboard",
    ) -> DashboardLogEvent:
        """Append a local dashboard event."""
        event = DashboardLogEvent(
            id=self._next_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            level=level.upper(),
            category=(category or classify_log(logger_name, message)).upper(),
            message=clean_log_message(message),
            logger_name=logger_name,
        )
        with self._condition:
            self._next_id += 1
            self._events.append(event)
            self._condition.notify_all()
        return event

    def snapshot(self, *, after_id: int = 0) -> list[DashboardLogEvent]:
        """Return buffered events newer than ``after_id``."""
        with self._condition:
            return [event for event in self._events if event.id > after_id]

    def wait_for_events(self, *, after_id: int, timeout: float = 15.0) -> list[DashboardLogEvent]:
        """Wait until new events are available, or return an empty list on timeout."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while True:
                events = [event for event in self._events if event.id > after_id]
                if events:
                    return events
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._condition.wait(remaining)


@dataclass
class DashboardRuntimeState:
    """Mutable route providers used while the app moves from startup to runtime."""

    logs: DashboardLogBuffer = field(default_factory=DashboardLogBuffer)
    deps_provider: Callable[[], Any] = lambda: None
    backend_status_provider: Callable[[], dict[str, object]] = lambda: {}

    def get_deps(self) -> Any:
        """Return current tool dependencies, if runtime has created them."""
        return self.deps_provider()

    def get_backend_status(self) -> dict[str, object]:
        """Return current backend status."""
        return self.backend_status_provider()


def ensure_dashboard_runtime_state(app: Any, *, logs: DashboardLogBuffer | None = None) -> DashboardRuntimeState:
    """Return the shared dashboard runtime state attached to a FastAPI app."""
    holder = getattr(app, "state", app)
    existing = getattr(holder, DASHBOARD_RUNTIME_STATE_ATTR, None)
    if isinstance(existing, DashboardRuntimeState):
        return existing
    state = DashboardRuntimeState(logs=logs or DashboardLogBuffer())
    setattr(holder, DASHBOARD_RUNTIME_STATE_ATTR, state)
    return state


def dashboard_routes_mounted(app: Any) -> bool:
    """Return whether dashboard routes have already been registered."""
    holder = getattr(app, "state", app)
    return bool(getattr(holder, DASHBOARD_ROUTES_MOUNTED_ATTR, False))


def mount_dashboard_routes(
    app: Any,
    *,
    get_deps: Callable[[], Any],
    get_backend_status: Callable[[], dict[str, object]],
    logs: DashboardLogBuffer,
) -> None:
    """Register static dashboard API routes on a FastAPI app."""
    if dashboard_routes_mounted(app):
        return
    try:
        from fastapi import Body
        from fastapi.responses import Response, JSONResponse, StreamingResponse
    except Exception:  # pragma: no cover - FastAPI is optional outside the app runtime
        return
    setattr(getattr(app, "state", app), DASHBOARD_ROUTES_MOUNTED_ATTR, True)

    def _camera_worker() -> Any | None:
        deps = get_deps()
        return getattr(deps, "camera_worker", None) if deps is not None else None

    def _face_identity_worker() -> Any | None:
        deps = get_deps()
        return getattr(deps, "face_identity_worker", None) if deps is not None else None

    def _performance_diagnostics() -> Any | None:
        deps = get_deps()
        return getattr(deps, "performance_diagnostics", None) if deps is not None else None

    @app.get("/api/dashboard/status")  # type: ignore[misc]
    def _dashboard_status() -> JSONResponse:
        camera_worker = _camera_worker()
        face_worker = _face_identity_worker()
        recognition_available = bool(getattr(face_worker, "recognition_available", face_worker is not None))
        db = getattr(getattr(face_worker, "identifier", None), "db", None)
        people = []
        if db is not None:
            try:
                people = [
                    {"name": person.name, "exemplar_count": len(person.embeddings)}
                    for person in getattr(db, "persons", lambda: [])()
                ]
            except Exception:
                people = []
        frame_available = False
        if camera_worker is not None:
            try:
                frame_available = camera_worker.get_latest_frame() is not None
            except Exception:
                frame_available = False
        visible_count = 0
        if face_worker is not None and callable(getattr(face_worker, "snapshot", None)):
            try:
                visible_count = len(face_worker.snapshot().visible)
            except Exception:
                visible_count = 0

        diagnostics = _performance_diagnostics()
        diagnostics_snapshot = getattr(diagnostics, "snapshot", None) if diagnostics is not None else None
        performance = diagnostics_snapshot() if callable(diagnostics_snapshot) else {}

        payload = {
            **get_backend_status(),
            "performance": performance,
            "camera": {
                "available": camera_worker is not None,
                "frame_available": frame_available,
                "head_tracker": type(getattr(camera_worker, "head_tracker", None)).__name__
                if getattr(camera_worker, "head_tracker", None) is not None
                else None,
            },
            "face_recognition": {
                "available": face_worker is not None,
                "recognition_available": recognition_available,
                "db_path": str(getattr(db, "path", "")) if db is not None else None,
                "people": people,
                "visible_count": visible_count,
            },
        }
        return JSONResponse(payload)

    @app.get("/api/face/frame.jpg")  # type: ignore[misc]
    def _face_frame() -> Response | JSONResponse:
        camera_worker = _camera_worker()
        if camera_worker is None:
            return JSONResponse({"ok": False, "error": "camera_unavailable"}, status_code=503)
        frame = camera_worker.get_latest_frame()
        if frame is None:
            return JSONResponse({"ok": False, "error": "frame_unavailable"}, status_code=503)
        try:
            jpeg = encode_bgr_frame_as_jpeg(frame)
        except Exception as exc:
            logger.warning("Dashboard frame encoding failed: %s", exc)
            return JSONResponse({"ok": False, "error": "frame_encode_failed"}, status_code=500)
        return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/face/state")  # type: ignore[misc]
    def _face_state() -> JSONResponse:
        camera_worker = _camera_worker()
        face_worker = _face_identity_worker()
        if face_worker is None or not callable(getattr(face_worker, "snapshot", None)):
            return JSONResponse({"ok": True, "available": False, "faces": []})
        recognition_available = bool(getattr(face_worker, "recognition_available", True))
        focus_name = None
        if camera_worker is not None and callable(getattr(camera_worker, "get_speaker_focus_name", None)):
            focus_name = camera_worker.get_speaker_focus_name()
        snapshot = face_worker.snapshot()
        return JSONResponse(
            {
                "ok": True,
                "available": True,
                "recognition_available": recognition_available,
                "focus_name": focus_name,
                "faces": [_face_payload(item, focus_name) for item in snapshot.visible],
            }
        )

    @app.post("/api/face/remember")  # type: ignore[misc]
    def _remember_face(payload: dict[str, Any] = Body(...)) -> JSONResponse:
        face_worker = _face_identity_worker()
        if face_worker is None or not callable(getattr(face_worker, "remember_visible", None)):
            return JSONResponse({"ok": False, "error": "face_recognition_unavailable"}, status_code=503)
        if not isinstance(payload, dict):
            return JSONResponse({"ok": False, "error": "invalid_payload"}, status_code=400)
        name = str(payload.get("name", "")).strip()
        if not name:
            return JSONResponse({"ok": False, "error": "name_required"}, status_code=400)
        try:
            raw_face_id = payload.get("face_id")
            if raw_face_id is None:
                raise ValueError
            face_id = int(str(raw_face_id))
        except (TypeError, ValueError):
            return JSONResponse({"ok": False, "error": "face_id_required"}, status_code=400)
        try:
            result = face_worker.remember_visible(face_id, name)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        except KeyError:
            return JSONResponse({"ok": False, "error": "face_not_visible"}, status_code=404)
        logs.add(
            f"Named visible face #{face_id} as {result['name']}",
            level="INFO",
            category="VISION",
        )
        return JSONResponse({"ok": True, **result})

    @app.get("/api/dashboard/events")  # type: ignore[misc]
    async def _dashboard_events(request: Request, last_id: str | None = None) -> StreamingResponse:
        try:
            initial_last_id = max(0, int(last_id or 0))
        except ValueError:
            initial_last_id = 0
        if initial_last_id == 0:
            try:
                initial_last_id = max(0, int(request.headers.get("last-event-id", "0")))
            except ValueError:
                initial_last_id = 0

        async def _stream() -> Any:
            latest_id = initial_last_id
            for event in logs.snapshot(after_id=latest_id):
                latest_id = max(latest_id, event.id)
                yield sse_event(event)
            while True:
                if await request.is_disconnected():
                    break
                events = await asyncio.to_thread(logs.wait_for_events, after_id=latest_id, timeout=15.0)
                if not events:
                    yield ": keepalive\n\n"
                    continue
                for event in events:
                    latest_id = max(latest_id, event.id)
                    yield sse_event(event)

        return StreamingResponse(_stream(), media_type="text/event-stream")


def _face_payload(item: Any, focus_name: str | None) -> dict[str, object]:
    target = item.target
    x, y, width, height = target.bbox
    name = item.name
    focused = bool(name and focus_name and name.casefold() == focus_name.casefold())
    observed = bool(getattr(item, "observed", True))
    held = bool(getattr(item, "held", False))
    stability = round(float(getattr(item, "stability", 1.0)), 3)
    can_remember = bool(getattr(item, "can_remember", getattr(item, "embedding", None) is not None))
    last_observed_at = getattr(item, "last_observed_at", None)
    return {
        "id": item.track_id,
        "track_id": item.track_id,
        "name": name,
        "label": name or "unknown",
        "similarity": round(float(item.similarity), 3),
        "x_offset": round(float(target.x_offset), 3),
        "y_offset": round(float(target.y_offset), 3),
        "confidence": round(float(target.confidence), 3),
        "bbox": {
            "x": round(float(x), 5),
            "y": round(float(y), 5),
            "width": round(float(width), 5),
            "height": round(float(height), 5),
        },
        "frame_size": {"width": int(target.frame_size[0]), "height": int(target.frame_size[1])},
        "focused": focused,
        "first_seen_at": item.first_seen_at,
        "last_seen_at": item.last_seen_at,
        "observed": observed,
        "held": held,
        "stability": stability,
        "can_remember": can_remember,
        "last_observed_at": last_observed_at if last_observed_at is not None else item.last_seen_at,
    }


def sse_event(event: DashboardLogEvent) -> str:
    """Format one log event for Server-Sent Events."""
    return f"id: {event.id}\nevent: log\ndata: {json.dumps(event.to_dict())}\n\n"


def clean_log_message(message: str) -> str:
    """Clean a log message for compact dashboard display."""
    return "\n".join(line.strip() for line in str(message).splitlines() if line.strip()).strip()


def classify_log(logger_name: str, message: str) -> str:
    """Map app loggers/messages to compact dashboard categories."""
    lowered = f"{logger_name} {message}".lower()
    if "face" in lowered or "vision" in lowered or "camera" in lowered or "yolo" in lowered:
        return "VISION"
    if "tool" in lowered:
        return "TOOL"
    if "openai" in lowered or "gemini" in lowered or "huggingface" in lowered or "realtime" in lowered:
        return "LLM"
    if (
        "audio" in lowered
        or "voice" in lowered
        or "speech" in lowered
        or "stt" in lowered
        or "tts" in lowered
        or "vad" in lowered
        or "transcript" in lowered
    ):
        return "VOICE"
    if "movement" in lowered or "motion" in lowered or "head" in lowered:
        return "MOTION"
    return "SYSTEM"
