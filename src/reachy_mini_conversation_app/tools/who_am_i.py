import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.vision.face_identity import IdentifiedTarget


logger = logging.getLogger(__name__)


class WhoAmI(Tool):
    """Identify the current user from speaker attribution and face recognition."""

    name = "who_am_i"
    description = "Identify the current user or speaker from current face recognition and speaker attribution."
    parameters_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def __call__(self, deps: ToolDependencies, **_kwargs: Any) -> Dict[str, Any]:
        """Return the best current identity estimate without guessing from prompt examples."""
        speaker_match = _latest_named_speaker(deps.speaker_attribution_worker)
        if speaker_match is not None:
            logger.info("Tool call: who_am_i source=speaker_attribution name=%s", speaker_match["name"])
            return speaker_match

        visible = _visible_targets(deps.face_identity_worker)
        focused = _focused_visible_name(deps.camera_worker, visible)
        if focused is not None:
            logger.info("Tool call: who_am_i source=focused_visible_face name=%s", focused["name"])
            return focused

        named = [target for target in visible if target.name]
        unknown_count = sum(1 for target in visible if not target.name)
        if len(named) == 1:
            target = named[0]
            logger.info("Tool call: who_am_i source=single_visible_face name=%s", target.name)
            return _identity_payload(
                target.name or "",
                source="single_visible_face",
                confidence=float(target.similarity),
                visible_names=[target.name] if target.name else [],
                unknown_count=unknown_count,
            )
        if len(named) > 1:
            visible_names = sorted({target.name for target in named if target.name})
            logger.info("Tool call: who_am_i source=ambiguous visible_names=%s", visible_names)
            return {
                "status": "ambiguous",
                "name": None,
                "source": "multiple_visible_faces",
                "visible_names": visible_names,
                "unknown_count": unknown_count,
                "message": f"I can see {', '.join(visible_names)}, so I can't tell which one is you yet.",
            }
        if unknown_count:
            logger.info("Tool call: who_am_i source=unknown_faces count=%d", unknown_count)
            return {
                "status": "unknown",
                "name": None,
                "source": "unknown_visible_faces",
                "visible_names": [],
                "unknown_count": unknown_count,
                "message": "I can see you, but I don't know your name yet.",
            }

        logger.info("Tool call: who_am_i source=no_identity_signal")
        return {
            "status": "unknown",
            "name": None,
            "source": "no_identity_signal",
            "visible_names": [],
            "unknown_count": 0,
            "message": "I can't tell who you are yet.",
        }


def _latest_named_speaker(speaker_worker: object | None) -> Dict[str, Any] | None:
    snapshot = getattr(speaker_worker, "snapshot", None)
    if not callable(snapshot):
        return None
    try:
        segments = list(snapshot())
    except Exception:
        return None
    for segment in reversed(segments):
        name = str(getattr(segment, "person_name", "") or "").strip()
        if not name:
            continue
        if bool(getattr(segment, "self_speech_suppressed", False)):
            continue
        return _identity_payload(
            name,
            source="speaker_attribution",
            confidence=float(getattr(segment, "confidence", 0.0) or 0.0),
            visible_names=[name],
            unknown_count=0,
        )
    return None


def _visible_targets(identity_worker: object | None) -> list[IdentifiedTarget]:
    snapshot = getattr(identity_worker, "snapshot", None)
    if not callable(snapshot):
        return []
    try:
        return list(snapshot().visible)
    except Exception:
        return []


def _focused_visible_name(camera_worker: object | None, visible: list[IdentifiedTarget]) -> Dict[str, Any] | None:
    get_focus_name = getattr(camera_worker, "get_speaker_focus_name", None)
    if not callable(get_focus_name):
        return None
    focus_name = str(get_focus_name() or "").strip()
    if not focus_name:
        return None
    for target in visible:
        if (target.name or "").casefold() == focus_name.casefold():
            return _identity_payload(
                target.name or focus_name,
                source="focused_visible_face",
                confidence=float(target.similarity),
                visible_names=sorted({item.name for item in visible if item.name}),
                unknown_count=sum(1 for item in visible if not item.name),
            )
    return None


def _identity_payload(
    name: str,
    *,
    source: str,
    confidence: float,
    visible_names: list[str],
    unknown_count: int,
) -> Dict[str, Any]:
    return {
        "status": "identified",
        "name": name,
        "source": source,
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "visible_names": visible_names,
        "unknown_count": unknown_count,
        "message": f"You look like {name}.",
    }
