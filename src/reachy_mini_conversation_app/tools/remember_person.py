import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.vision.face_identity import IdentifiedTarget, identify_from_camera


logger = logging.getLogger(__name__)


class RememberPerson(Tool):
    """Save the largest currently unknown face under a provided name."""

    name = "remember_person"
    description = "Save the largest currently-unknown face under the given name."
    parameters_schema = {
        "type": "object",
        "properties": {"name": {"type": "string", "minLength": 1}},
        "required": ["name"],
        "additionalProperties": False,
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Remember the largest unknown visible face as ``name``."""
        name = str(kwargs.get("name") or "").strip()
        if not name:
            return {"error": "name must be a non-empty string"}
        if deps.camera_worker is None:
            return {"error": "Camera worker not available"}
        if deps.face_identity_worker is None:
            return {"error": "Face identity worker not available"}

        identified = await asyncio.to_thread(identify_from_camera, deps.camera_worker, deps.face_identity_worker)
        unknown = _largest_unknown(identified)
        if unknown is None:
            return {"error": "No unknown face is currently visible"}

        db = deps.face_identity_worker.identifier.db
        await asyncio.to_thread(db.add, name, unknown.embedding)
        exemplar_count = db.exemplar_count(name)
        logger.info("Tool call: remember_person name=%s exemplars=%d", name, exemplar_count)
        return {
            "status": "remembered",
            "name": name,
            "exemplar_count": exemplar_count,
            "x_offset": round(float(unknown.target.x_offset), 3),
            "y_offset": round(float(unknown.target.y_offset), 3),
        }


def _largest_unknown(identified: list[IdentifiedTarget]) -> IdentifiedTarget | None:
    unknown = [target for target in identified if target.name is None]
    if not unknown:
        return None
    return max(unknown, key=lambda target: target.target.area)
