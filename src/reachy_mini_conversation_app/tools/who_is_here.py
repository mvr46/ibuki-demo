import time
import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.vision.face_identity import IdentifiedTarget, identify_from_camera


logger = logging.getLogger(__name__)


class WhoIsHere(Tool):
    """List the people currently visible to the robot's camera."""

    name = "who_is_here"
    description = "List the people currently visible to the robot's camera."
    parameters_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def __call__(self, deps: ToolDependencies, **_kwargs: Any) -> Dict[str, Any]:
        """Return the latest visible people known to face recognition."""
        if deps.camera_worker is None:
            return {"error": "Camera worker not available"}
        if deps.face_identity_worker is None:
            return {"error": "Face identity worker not available"}

        identified = await asyncio.to_thread(identify_from_camera, deps.camera_worker, deps.face_identity_worker)
        logger.info("Tool call: who_is_here visible=%d", len(identified))
        return {"people": [_target_payload(target) for target in identified]}


def _target_payload(target: IdentifiedTarget) -> dict[str, Any]:
    now = time.monotonic()
    seconds_in_view = 0.0 if target.first_seen_at is None else max(0.0, now - target.first_seen_at)
    return {
        "name": target.name,
        "x_offset": round(float(target.target.x_offset), 3),
        "y_offset": round(float(target.target.y_offset), 3),
        "similarity": round(float(target.similarity), 3),
        "seconds_in_view": round(seconds_in_view, 1),
    }
