"""Small audio streaming primitives used by the local robot loop."""

from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


class AdditionalOutputs:
    """Non-audio messages emitted alongside audio frames."""

    def __init__(self, *args: object) -> None:
        """Store one or more side-channel output payloads."""
        self.args = args


class AsyncStreamHandler(ABC):
    """Minimal async handler contract for robot audio streams."""

    def __init__(
        self,
        *,
        expected_layout: str,
        output_sample_rate: int,
        input_sample_rate: int,
    ) -> None:
        """Initialize stream metadata shared by all handlers."""
        self.expected_layout = expected_layout
        self.output_sample_rate = output_sample_rate
        self.input_sample_rate = input_sample_rate

    @abstractmethod
    async def receive(self, frame: tuple[int, NDArray[np.int16]]) -> None:
        """Receive an input audio frame."""
        ...

    @abstractmethod
    async def emit(self) -> Any:
        """Emit the next output item, if available."""
        ...

    @abstractmethod
    def copy(self) -> "AsyncStreamHandler":
        """Create a copy of this handler."""
        ...

    async def start_up(self) -> None:
        """Run optional async startup work."""


async def wait_for_item(queue: asyncio.Queue[Any], timeout: float = 0.1) -> Any | None:
    """Return the next queue item, or None when no item is ready."""
    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError):
        return None


def audio_to_int16(audio: NDArray[np.int16 | np.float32] | tuple[int, NDArray[np.int16 | np.float32]]) -> NDArray[np.int16]:
    """Convert PCM audio to int16."""
    if isinstance(audio, tuple):
        _, audio = audio
    if audio.dtype == np.int16:
        return cast(NDArray[np.int16], audio)
    if audio.dtype == np.float32:
        return (audio * 32767.0).astype(np.int16)
    raise TypeError(f"Unsupported audio data type: {audio.dtype}")


def audio_to_float32(
    audio: NDArray[np.int16 | np.float32] | tuple[int, NDArray[np.int16 | np.float32]],
) -> NDArray[np.float32]:
    """Convert PCM audio to float32 in the range [-1.0, 1.0)."""
    if isinstance(audio, tuple):
        _, audio = audio
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype == np.float32:
        return cast(NDArray[np.float32], audio)
    raise TypeError(f"Unsupported audio data type: {audio.dtype}")
