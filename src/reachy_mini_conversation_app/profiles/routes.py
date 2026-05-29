"""Settings routes for repo-backed production profiles."""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Callable, Optional

from fastapi import Query, FastAPI, Request

from reachy_mini_conversation_app.profiles.store import DEFAULT_PROFILE_NAME, ProfileStore
from reachy_mini_conversation_app.runtime.config import (
    LOCKED_PROFILE,
    config,
    set_custom_profile,
    get_default_voice_for_backend,
    get_available_voices_for_backend,
)
from reachy_mini_conversation_app.profiles.headless import available_tools_for
from reachy_mini_conversation_app.backends.interface import ConversationHandler


logger = logging.getLogger(__name__)


def mount_personality_routes(
    app: FastAPI,
    handler: ConversationHandler,
    get_loop: Callable[[], asyncio.AbstractEventLoop | None],
    *,
    persist_personality: Callable[[Optional[str], Optional[str]], None] | None = None,
    get_persisted_personality: Callable[[], Optional[str]] | None = None,
    profile_store: ProfileStore | None = None,
) -> None:
    """Register production profile endpoints on a FastAPI app."""
    try:
        from pydantic import BaseModel
        from fastapi.responses import JSONResponse
    except Exception:  # pragma: no cover
        return

    store = profile_store or ProfileStore()

    class ApplyPayload(BaseModel):
        name: str = DEFAULT_PROFILE_NAME
        persist: Optional[bool] = False

    class SavePayload(BaseModel):
        name: str
        instructions: str
        tools_text: str = ""
        voice: Optional[str] = None
        overwrite: Optional[bool] = False

    def _startup_choice() -> str:
        try:
            if get_persisted_personality is not None:
                stored = get_persisted_personality()
                if stored:
                    return store.resolve_startup_profile(stored)
            env_val = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
            if env_val:
                return store.resolve_startup_profile(env_val)
        except Exception:
            pass
        return DEFAULT_PROFILE_NAME

    def _current_choice() -> str:
        try:
            return store.resolve_startup_profile(getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None))
        except Exception:
            return DEFAULT_PROFILE_NAME

    def _profile_payload(name: str) -> dict[str, Any]:
        profile = store.load(name)
        voice = profile.voice or get_default_voice_for_backend()
        available_tools = available_tools_for(profile.name)
        enabled = list(profile.tools)
        return {
            "name": profile.name,
            "instructions": profile.instructions,
            "tools_text": profile.tools_text,
            "voice": voice,
            "uses_default_voice": profile.voice is None,
            "available_tools": available_tools,
            "enabled_tools": enabled,
        }

    def _list_payload() -> dict[str, Any]:
        summaries = store.list_profiles()
        choices = [summary.name for summary in summaries]
        return {
            "profiles": [{"name": summary.name, "is_default": summary.is_default} for summary in summaries],
            "choices": choices,
            "current": _current_choice(),
            "startup": _startup_choice(),
            "locked": LOCKED_PROFILE is not None,
            "locked_to": LOCKED_PROFILE,
        }

    async def _apply_profile(
        request: Request,
        payload: ApplyPayload | None,
        name: str | None,
        persist: bool | None,
    ) -> Any:
        if LOCKED_PROFILE is not None:
            return JSONResponse(
                {"ok": False, "error": "profile_locked", "locked_to": LOCKED_PROFILE},
                status_code=403,
            )
        loop = get_loop()
        if loop is None:
            return JSONResponse({"ok": False, "error": "loop_unavailable"}, status_code=503)

        selected = name or (payload.name if payload else None)
        persist_flag = bool(persist if persist is not None else (payload.persist if payload else False))
        if not selected:
            try:
                raw = await request.json()
                if isinstance(raw, dict):
                    selected = str(raw.get("name") or DEFAULT_PROFILE_NAME)
                    if "persist" in raw:
                        persist_flag = bool(raw.get("persist"))
            except Exception:
                selected = DEFAULT_PROFILE_NAME
        selected = store.resolve_startup_profile(selected)

        async def _do_apply() -> tuple[str, Optional[str]]:
            status = await handler.apply_personality(None if selected == DEFAULT_PROFILE_NAME else selected)
            get_current_voice = getattr(handler, "get_current_voice", None)
            voice_override = get_current_voice() if callable(get_current_voice) else None
            return status, voice_override

        try:
            logger.info("Applying production profile %r", selected)
            fut = asyncio.run_coroutine_threadsafe(_do_apply(), loop)
            status, voice_override = fut.result(timeout=10)
            if persist_flag and persist_personality is not None:
                persist_personality(None if selected == DEFAULT_PROFILE_NAME else selected, voice_override)
            return {"ok": True, "status": status, "startup": _startup_choice(), "profile": selected}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    def _save_profile(payload: SavePayload) -> Any:
        try:
            if payload.overwrite:
                profile = store.overwrite(
                    payload.name,
                    instructions=payload.instructions,
                    tools_text=payload.tools_text,
                    voice=payload.voice,
                )
            else:
                profile = store.save_new(
                    payload.name,
                    instructions=payload.instructions,
                    tools_text=payload.tools_text,
                    voice=payload.voice,
                )
            return {"ok": True, "profile": profile.name, **_list_payload()}
        except FileExistsError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
        except FileNotFoundError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    async def _save_payload_from_request(request: Request, *, overwrite: bool | None = None) -> Any:
        try:
            raw = await request.json()
            if not isinstance(raw, dict):
                raw = {}
            if overwrite is not None:
                raw["overwrite"] = overwrite
            return SavePayload(**raw)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    @app.get("/profiles")
    def _profiles() -> dict[str, Any]:
        return _list_payload()

    @app.get("/profiles/load")
    def _profiles_load(name: str = DEFAULT_PROFILE_NAME) -> dict[str, Any]:
        return _profile_payload(name)

    @app.post("/profiles/save")
    async def _profiles_save(request: Request) -> Any:
        payload = await _save_payload_from_request(request)
        if isinstance(payload, JSONResponse):
            return payload
        return _save_profile(payload)

    @app.post("/profiles/overwrite")
    async def _profiles_overwrite(request: Request) -> Any:
        payload = await _save_payload_from_request(request, overwrite=True)
        if isinstance(payload, JSONResponse):
            return payload
        return _save_profile(payload)

    @app.post("/profiles/apply")
    async def _profiles_apply(
        request: Request,
        payload: ApplyPayload | None = None,
        name: str | None = None,
        persist: Optional[bool] = None,
    ) -> Any:
        return await _apply_profile(request, payload, name, persist)

    @app.get("/personalities")
    def _legacy_personalities() -> dict[str, Any]:
        return _list_payload()

    @app.get("/personalities/load")
    def _legacy_personalities_load(name: str = DEFAULT_PROFILE_NAME) -> dict[str, Any]:
        return _profile_payload(name)

    @app.post("/personalities/save")
    async def _legacy_personalities_save(request: Request) -> Any:
        payload = await _save_payload_from_request(request)
        if isinstance(payload, JSONResponse):
            return payload
        return _save_profile(payload)

    @app.post("/personalities/apply")
    async def _legacy_personalities_apply(
        request: Request,
        payload: ApplyPayload | None = None,
        name: str | None = None,
        persist: Optional[bool] = None,
    ) -> Any:
        return await _apply_profile(request, payload, name, persist)

    @app.get("/voices")
    async def _voices() -> list[str]:
        loop = get_loop()
        if loop is None:
            return get_available_voices_for_backend()

        async def _get_v() -> list[str]:
            try:
                return await handler.get_available_voices()
            except Exception:
                return get_available_voices_for_backend()

        try:
            fut = asyncio.run_coroutine_threadsafe(_get_v(), loop)
            return fut.result(timeout=10)
        except Exception:
            return get_available_voices_for_backend()

    @app.get("/voices/current")
    async def _current_voice() -> dict[str, str]:
        fallback_voice = get_default_voice_for_backend()
        loop = get_loop()
        if loop is None:
            return {"voice": fallback_voice}

        def _get_current() -> str:
            try:
                return handler.get_current_voice()
            except Exception:
                return fallback_voice

        try:
            fut = asyncio.run_coroutine_threadsafe(asyncio.to_thread(_get_current), loop)
            return {"voice": fut.result(timeout=10)}
        except Exception:
            return {"voice": fallback_voice}

    @app.post("/voices/apply")
    async def _apply_voice(request: Request, voice: str | None = Query(None)) -> Any:
        selected_voice = str(voice or "")
        if not selected_voice:
            try:
                raw = await request.json()
            except Exception:
                raw = {}
            selected_voice = str(raw.get("voice", "") or "")
        if not selected_voice:
            return JSONResponse({"ok": False, "error": "missing_voice"}, status_code=400)
        loop = get_loop()
        if loop is None:
            return JSONResponse({"ok": False, "error": "loop_unavailable"}, status_code=503)

        async def _do() -> str:
            return await handler.change_voice(selected_voice)

        try:
            fut = asyncio.run_coroutine_threadsafe(_do(), loop)
            status = fut.result(timeout=10)
            current_profile = _current_choice()
            profile = store.load(current_profile)
            store.overwrite(
                current_profile,
                instructions=profile.instructions,
                tools_text=profile.tools_text,
                voice=selected_voice,
            )
            set_custom_profile(None if current_profile == DEFAULT_PROFILE_NAME else current_profile)
            return {"ok": True, "status": status}
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
