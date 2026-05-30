"""Lifecycle helpers for the local llama.cpp model servers."""

from __future__ import annotations
import os
import signal
import asyncio
import logging
import subprocess
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from reachy_mini_conversation_app.runtime.config import config


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalModelServerSpec:
    """One app-managed llama.cpp server."""

    label: str
    base_url: str
    hf_repo: str
    host: str
    port: int
    n_parallel: int
    context_size: int


class LocalModelServerManager:
    """Start missing local llama.cpp servers for the dashboard happy path."""

    def __init__(self, *, specs: list[LocalModelServerSpec] | None = None) -> None:
        """Initialize with optional explicit server specs for tests/custom launches."""
        self.specs = specs or _default_specs()
        self._processes: list[subprocess.Popen] = []

    async def start_up(self) -> None:
        """Start any configured localhost llama.cpp servers that are not already ready."""
        if not config.LOCAL_MODEL_SERVER_AUTOSTART:
            logger.info("Local model server autostart disabled.")
            return

        started_specs: list[LocalModelServerSpec] = []
        for spec in self.specs:
            if await _server_ready(spec.base_url):
                logger.info("%s server already ready at %s.", spec.label, spec.base_url)
                continue
            if not _is_local_bind_host(spec.host):
                logger.warning("%s server URL is not localhost; skipping autostart for %s.", spec.label, spec.base_url)
                continue
            self._start_server(spec)
            started_specs.append(spec)

        if started_specs:
            await asyncio.gather(*(self._wait_until_ready(spec) for spec in started_specs))

    async def shutdown(self) -> None:
        """Stop servers started by this manager, leaving externally owned servers alone."""
        for process in self._processes:
            if process.poll() is not None:
                continue
            try:
                process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                continue
        deadline = asyncio.get_running_loop().time() + 8.0
        for process in self._processes:
            while process.poll() is None and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.1)
            if process.poll() is None:
                process.terminate()
        self._processes.clear()

    def _start_server(self, spec: LocalModelServerSpec) -> None:
        command = [
            config.LOCAL_LLAMA_SERVER_BIN,
            "-hf",
            spec.hf_repo,
            "-np",
            str(spec.n_parallel),
            "-c",
            str(spec.context_size),
            "-fa",
            "on",
            "--host",
            spec.host,
            "--port",
            str(spec.port),
            "--no-webui",
        ]
        logger.info("Starting %s server: %s", spec.label, " ".join(command))
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=os.environ.copy())
        self._processes.append(process)

    async def _wait_until_ready(self, spec: LocalModelServerSpec) -> None:
        timeout_s = max(1.0, float(config.LOCAL_MODEL_SERVER_START_TIMEOUT_SECONDS))
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            if await _server_ready(spec.base_url):
                logger.info("%s server ready at %s.", spec.label, spec.base_url)
                return
            for process in self._processes:
                if process.poll() not in {None, 0}:
                    logger.warning("%s server process exited early with code %s.", spec.label, process.returncode)
                    return
            await asyncio.sleep(0.5)
        logger.warning("%s server was not ready after %.0fs at %s.", spec.label, timeout_s, spec.base_url)


def create_local_model_server_manager() -> LocalModelServerManager:
    """Create the default local llama.cpp server manager."""
    return LocalModelServerManager()


def _default_specs() -> list[LocalModelServerSpec]:
    return [
        _spec_from_base_url(
            label="Local chat model",
            base_url=config.LOCAL_CHAT_BASE_URL,
            hf_repo=config.LOCAL_CHAT_SERVER_HF,
            n_parallel=2,
            context_size=4096,
        ),
        _spec_from_base_url(
            label="Local router model",
            base_url=config.LOCAL_ROUTER_BASE_URL,
            hf_repo=config.LOCAL_ROUTER_SERVER_HF,
            n_parallel=1,
            context_size=1024,
        ),
        _spec_from_base_url(
            label="Local vision model",
            base_url=config.LOCAL_VISION_BASE_URL,
            hf_repo=config.LOCAL_VISION_SERVER_HF,
            n_parallel=1,
            context_size=2048,
        ),
    ]


def _spec_from_base_url(
    *,
    label: str,
    base_url: str,
    hf_repo: str,
    n_parallel: int,
    context_size: int,
) -> LocalModelServerSpec:
    parsed = urlsplit(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return LocalModelServerSpec(
        label=label,
        base_url=base_url.rstrip("/"),
        hf_repo=hf_repo,
        host="127.0.0.1" if host == "localhost" else host,
        port=port,
        n_parallel=n_parallel,
        context_size=context_size,
    )


def _is_local_bind_host(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost"}


async def _server_ready(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=0.6) as client:
            response = await client.get(f"{base_url.rstrip('/')}/models")
        return response.status_code == 200
    except Exception:
        return False
