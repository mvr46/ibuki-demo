import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.vision.face_identity import IdentifiedTarget, identify_from_camera


logger = logging.getLogger(__name__)


class LookAtPerson(Tool):
    """Turn head tracking toward a named visible person."""

    name = "look_at_person"
    description = "Turn the head toward the named visible person."
    parameters_schema = {
        "type": "object",
        "properties": {"name": {"type": "string", "minLength": 1}},
        "required": ["name"],
        "additionalProperties": False,
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Prefer the named visible person in the existing camera tracking loop."""
        requested_name = str(kwargs.get("name") or "").strip()
        if not requested_name:
            return {"error": "name must be a non-empty string"}
        if deps.camera_worker is None:
            return {"error": "Camera worker not available"}
        if deps.face_identity_worker is None:
            return {"error": "Face identity worker not available"}

        visible = _visible_targets(deps)
        match = _visible_name_match(visible, requested_name)
        if match is None:
            visible_names = sorted({target.name for target in visible if target.name})
            return {
                "error": f"{requested_name} is not currently visible",
                "visible_names": visible_names,
            }

        set_focus_name = getattr(deps.camera_worker, "set_speaker_focus_name", None)
        if not callable(set_focus_name):
            return {"error": "Camera worker cannot focus on named speakers"}

        set_focus_name(match.name)
        logger.info("Tool call: look_at_person name=%s", match.name)
        return {
            "status": "looking_at",
            "name": match.name,
            "x_offset": round(float(match.target.x_offset), 3),
            "y_offset": round(float(match.target.y_offset), 3),
            "similarity": round(float(match.similarity), 3),
        }


def _visible_targets(deps: ToolDependencies) -> list[IdentifiedTarget]:
    identity_worker = deps.face_identity_worker
    snapshot = getattr(identity_worker, "snapshot", None)
    if callable(snapshot):
        return list(snapshot().visible)
    return identify_from_camera(deps.camera_worker, identity_worker)


def _visible_name_match(visible: list[IdentifiedTarget], requested_name: str) -> IdentifiedTarget | None:
    folded_name = requested_name.casefold()
    matches = [target for target in visible if (target.name or "").casefold() == folded_name]
    if not matches:
        return None
    return max(matches, key=lambda target: target.target.area)
