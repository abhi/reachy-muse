"""The three capabilities the companion serves, and stub engines for each.

C2 to C4 replace the stubs with real models. The server only sees these Protocols.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Protocol, Sequence, Tuple, Union

from musegadget.reachy_companion_protocol import MAX_LINE_BYTES, NarratedLine, VoiceRole


@dataclass(frozen=True)
class Partial:
    text: str


@dataclass(frozen=True)
class Endpoint:
    """The engine decided the turn is over before the robot closed the stream."""

    text: str


HearEvent = Union[Partial, Endpoint]


class HearingSession(Protocol):
    async def feed(self, pcm: bytes) -> Sequence[HearEvent]:
        """Take one 16 kHz s16le microphone frame. Return what it changed."""

    async def finish(self) -> str:
        """The robot ended the turn. Return the final transcript."""

    async def close(self) -> None:
        """The stream ended: endpoint, finish, abort, gap or disconnect. Release what the session holds."""


class Hearing(Protocol):
    name: str

    def open(self, pre_roll_frames: int) -> HearingSession: ...


class Voice(Protocol):
    name: str

    def speak(self, text: str, role: VoiceRole, out_rate: int) -> AsyncIterator[bytes]:
        """Yield s16le mono PCM at out_rate, in chunks of any size. The server frames them."""


class Narrator(Protocol):
    name: str

    async def acknowledge(self, request: str) -> Sequence[NarratedLine]: ...

    async def progress(self, request: str, status: str, already_said: Sequence[str]) -> Sequence[NarratedLine]: ...

    async def lines(self, request: str, reply: str) -> Sequence[NarratedLine]: ...


class ScriptedHearing:
    """Emits partials[i] after every frames_per_partial frames, and optionally ends the turn itself."""

    name = "scripted"

    def __init__(self, partials: Tuple[str, ...] = ("hello",), *, frames_per_partial: int = 5, endpoint_after: Optional[int] = None) -> None:
        self.partials = partials
        self.frames_per_partial = frames_per_partial
        self.endpoint_after = endpoint_after

    def open(self, pre_roll_frames: int) -> _ScriptedSession:
        return _ScriptedSession(self)


class _ScriptedSession:
    def __init__(self, script: ScriptedHearing) -> None:
        self.script = script
        self.frames = 0
        self.said = ""

    async def feed(self, pcm: bytes) -> Sequence[HearEvent]:
        self.frames += 1
        events = []
        index, due = divmod(self.frames, self.script.frames_per_partial)
        if due == 0 and index <= len(self.script.partials):
            self.said = self.script.partials[index - 1]
            events.append(Partial(self.said))
        if self.frames == self.script.endpoint_after:
            events.append(Endpoint(self.said))
        return events

    async def finish(self) -> str:
        return self.said

    async def close(self) -> None:
        pass


class ToneVoice:
    """A 440 Hz sine: `chunks` chunks of `chunk_ms` each, whatever the text."""

    name = "tone"

    def __init__(self, *, chunks: int = 3, chunk_ms: int = 100) -> None:
        self.chunks = chunks
        self.chunk_ms = chunk_ms

    async def speak(self, text: str, role: VoiceRole, out_rate: int) -> AsyncIterator[bytes]:
        count = out_rate * self.chunk_ms // 1000
        for index in range(self.chunks):
            start = index * count
            yield struct.pack(f"<{count}h", *(int(8000 * math.sin(2 * math.pi * 440 * (start + n) / out_rate)) for n in range(count)))


class EchoNarrator:
    """Speaks back the first three sentences of whatever it was given, with no expression."""

    name = "echo"

    async def acknowledge(self, request: str) -> Sequence[NarratedLine]:
        return _sentences(request)

    async def progress(self, request: str, status: str, already_said: Sequence[str]) -> Sequence[NarratedLine]:
        return _sentences(status)

    async def lines(self, request: str, reply: str) -> Sequence[NarratedLine]:
        return _sentences(reply)


def _sentences(text: str) -> Tuple[NarratedLine, ...]:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s][:3]
    return tuple(NarratedLine(s.encode("utf-8")[:MAX_LINE_BYTES].decode("utf-8", "ignore"), None) for s in sentences)
