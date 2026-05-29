import sys
import logging

from reachy_mini_conversation_app.profiles.store import ProfileStore
from reachy_mini_conversation_app.runtime.config import config, get_default_voice_for_backend


logger = logging.getLogger(__name__)


profile_store = ProfileStore()


def get_session_instructions() -> str:
    """Get session instructions from the active production profile."""
    try:
        profile = profile_store.load(config.REACHY_MINI_CUSTOM_PROFILE)
        logger.info("Loading prompt from production profile '%s'", profile.name)
        return profile.instructions
    except Exception as e:
        logger.error("Failed to load production profile instructions: %s", e)
        sys.exit(1)


def get_session_voice(default: str | None = None) -> str:
    """Resolve the voice to use for the session.

    If a custom profile is selected and contains a voice.txt, return its
    trimmed content; otherwise return the provided default or the active
    backend default voice.
    """
    fallback = get_default_voice_for_backend() if default is None else default
    try:
        profile = profile_store.load(config.REACHY_MINI_CUSTOM_PROFILE)
        return profile.voice or fallback
    except Exception:
        logger.debug("Falling back to default voice", exc_info=True)
    return fallback
