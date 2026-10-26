from __future__ import annotations

import math
import re
import struct
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Optional, Protocol, Sequence, Tuple, Union

from reachy_muse.contract.expressions import Expression
from reachy_muse.contract.intent import Intent
from reachy_muse.device_command import DeviceCommand, parse_command
from reachy_muse.protocol import ANIMATION_CHANNELS, MAX_LINE_BYTES, Frames, NarratedLine, VoiceRole


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

    def speak(self, text: str, role: VoiceRole, out_rate: int, expression: Optional[Expression] = None) -> AsyncIterator[bytes]:
        """Yield s16le mono PCM at out_rate, in chunks of any size. The server frames them.

        `expression` is the one the narrator gave this line, if it narrated it; a voice may color the line with it.
        """


@dataclass(frozen=True)
class Narrated:
    intent: Optional[Intent]
    lines: Tuple[NarratedLine, ...]


class Narrator(Protocol):
    name: str

    async def acknowledge(self, request: str, *, previous: str = "", recently_said: Sequence[str] = (),
                          may_chat: bool = False) -> Narrated: ...

    async def progress(self, request: str, status: str, already_said: Sequence[str]) -> Sequence[NarratedLine]: ...

    async def lines(self, request: str, reply: str) -> Sequence[NarratedLine]: ...


class Classifier(Protocol):
    name: str

    async def classify(self, text: str, allowed: Sequence[DeviceCommand]) -> Optional[DeviceCommand]: ...


@dataclass(frozen=True)
class Animation:
    """Frames in ANIMATION_CHANNELS order and units; no frames means no motion."""

    fps: int
    frames: Frames


class Animator(Protocol):
    name: str

    async def animate(self, intent: str, duration_s: float) -> Animation: ...


class SpeechFirst:
    """Hearing, speech, narration and classify share the GPU with motion and go first.

    The server marks each of those ops busy while it runs; the motion planner waits for a quiet
    moment before each token, so motion only fills the gaps.
    """

    def __init__(self) -> None:
        self._busy = 0
        self._quiet = threading.Condition()

    @contextmanager
    def busy(self) -> Iterator[None]:
        with self._quiet:
            self._busy += 1
        try:
            yield
        finally:
            with self._quiet:
                self._busy -= 1
                self._quiet.notify_all()

    def wait_quiet(self, limit_s: float) -> None:
        """Block until no speech op runs, or ``limit_s`` passes, so a long one never starves motion for good."""
        with self._quiet:
            self._quiet.wait_for(lambda: self._busy == 0, timeout=limit_s)


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

    async def speak(self, text: str, role: VoiceRole, out_rate: int, expression: Optional[Expression] = None) -> AsyncIterator[bytes]:
        count = out_rate * self.chunk_ms // 1000
        for index in range(self.chunks):
            start = index * count
            yield struct.pack(f"<{count}h", *(int(8000 * math.sin(2 * math.pi * 440 * (start + n) / out_rate)) for n in range(count)))


class EchoNarrator:
    """Speaks back the first three sentences of whatever it was given, with no expression."""

    name = "echo"

    async def acknowledge(self, request: str, *, previous: str = "", recently_said: Sequence[str] = (),
                          may_chat: bool = False) -> Narrated:
        return Narrated(None, tuple(_sentences(request)))

    async def progress(self, request: str, status: str, already_said: Sequence[str]) -> Sequence[NarratedLine]:
        return _sentences(status)

    async def lines(self, request: str, reply: str) -> Sequence[NarratedLine]:
        return _sentences(reply)

    async def classify(self, text: str, allowed: Sequence[DeviceCommand]) -> Optional[DeviceCommand]:
        command = parse_command(text)
        return command if command in allowed else None


class NodAnimator:
    name = "nod"

    async def animate(self, intent: str, duration_s: float) -> Animation:
        fps = 25
        count = max(1, round(duration_s * fps))
        pitch = ANIMATION_CHANNELS.index("pitch")
        return Animation(fps, tuple(
            tuple(round(10 * math.sin(4 * math.pi * n / count), 1) if axis == pitch else 0.0
                  for axis in range(len(ANIMATION_CHANNELS)))
            for n in range(count)))


def _sentences(text: str) -> Tuple[NarratedLine, ...]:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s][:3]
    return tuple(NarratedLine(s.encode("utf-8")[:MAX_LINE_BYTES].decode("utf-8", "ignore"), None) for s in sentences)
