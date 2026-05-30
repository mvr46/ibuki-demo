"""Tests for dashboard-friendly local llama.cpp server startup."""

from __future__ import annotations
import signal

import pytest

import reachy_mini_conversation_app.backends.local_model_servers as server_mod
from reachy_mini_conversation_app.backends.local_model_servers import LocalModelServerSpec, LocalModelServerManager


class _FakeProcess:
    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.returncode: int | None = None
        self.signals: list[int] = []

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        self.returncode = 0

    def terminate(self) -> None:
        self.returncode = -15


@pytest.mark.asyncio
async def test_local_model_server_manager_starts_missing_localhost_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing localhost model endpoint should be started with llama-server."""
    spec = LocalModelServerSpec(
        label="Local chat model",
        base_url="http://127.0.0.1:8080/v1",
        hf_repo="test/chat:Q4",
        host="127.0.0.1",
        port=8080,
        n_parallel=2,
        context_size=4096,
    )
    processes: list[_FakeProcess] = []
    ready = False

    async def fake_ready(_base_url: str) -> bool:
        return ready

    def fake_popen(command: list[str], **_kwargs: object) -> _FakeProcess:
        nonlocal ready
        process = _FakeProcess(command)
        processes.append(process)
        ready = True
        return process

    monkeypatch.setattr(server_mod, "_server_ready", fake_ready)
    monkeypatch.setattr(server_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(server_mod.config, "LOCAL_MODEL_SERVER_AUTOSTART", True)
    monkeypatch.setattr(server_mod.config, "LOCAL_LLAMA_SERVER_BIN", "llama-server")

    manager = LocalModelServerManager(specs=[spec])
    await manager.start_up()
    await manager.shutdown()

    assert len(processes) == 1
    assert processes[0].command == [
        "llama-server",
        "-hf",
        "test/chat:Q4",
        "-np",
        "2",
        "-c",
        "4096",
        "-fa",
        "on",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--no-webui",
    ]
    assert processes[0].signals == [signal.SIGINT]


@pytest.mark.asyncio
async def test_local_model_server_manager_skips_remote_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom non-local model endpoints should be probed but not process-managed."""
    spec = LocalModelServerSpec(
        label="Remote chat model",
        base_url="http://model.example.test/v1",
        hf_repo="test/chat:Q4",
        host="model.example.test",
        port=80,
        n_parallel=1,
        context_size=1024,
    )
    popen_calls: list[list[str]] = []

    async def fake_ready(_base_url: str) -> bool:
        return False

    def fake_popen(command: list[str], **_kwargs: object) -> _FakeProcess:
        popen_calls.append(command)
        return _FakeProcess(command)

    monkeypatch.setattr(server_mod, "_server_ready", fake_ready)
    monkeypatch.setattr(server_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(server_mod.config, "LOCAL_MODEL_SERVER_AUTOSTART", True)

    manager = LocalModelServerManager(specs=[spec])
    await manager.start_up()

    assert popen_calls == []
