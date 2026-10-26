"""Kokoro standing in when Chatterbox fails to load, falls behind, or fails mid-line."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import aclosing
from typing import AsyncIterator, List, Optional, Protocol, Sequence

from reachy_muse.contract.expressions import Expression
from reachy_muse.protocol import VoiceRole

from reachy_muse.companion_server.voice_chatterbox import Chunk
from reachy_muse.companion_server.voice_kokoro import sentences
from reachy_muse.companion_server.voices import LoadableVoice

log = logging.getLogger("reachy_muse.companion_server")
_END = object()


class StreamingVoice(Protocol):
    name: str

    async def load(self) -> None: ...

    def stream(self, lines: Sequence[str], out_rate: int, expression: Optional[Expression] = None) -> AsyncIterator[Chunk]: ...


class FallbackVoice:
    """Speaks with the primary voice, and with the fallback whenever the primary cannot keep up.

    If the primary fails to load, the fallback speaks every line. If any audio chunk
    takes longer than `deadline_s`, or the primary fails mid-line, the fallback speaks the rest
    of that line. The fallback runs on its own thread, so a stuck primary does not hold it up.
    """

    deadline_s = 1.0
    prebuffer_s = 0.15

    def __init__(self, primary: StreamingVoice, fallback: LoadableVoice) -> None:
        self.primary: Optional[StreamingVoice] = primary
        self.fallback = fallback

    @property
    def name(self) -> str:
        return (self.primary or self.fallback).name

    async def load(self) -> None:
        await self.fallback.load()
        await self._load_primary()

    async def _load_primary(self) -> None:
        assert self.primary is not None
        try:
            await self.primary.load()
        except Exception as error:
            log.warning("voice %s failed to load (%s); %s speaks every line", self.primary.name, type(error).__name__,
                        self.fallback.name)
            self.primary = None

    async def speak(self, text: str, role: VoiceRole, out_rate: int, expression: Optional[Expression] = None) -> AsyncIterator[bytes]:
        lines = sentences(text)
        done = 0
        if self.primary is not None:
            loop = asyncio.get_running_loop()
            arrivals: asyncio.Queue = asyncio.Queue()
            pump = asyncio.create_task(_pump(self.primary.stream(lines, out_rate, expression), arrivals))
            playback = _UnderrunWatch(out_rate)
            held: List[Chunk] = []
            release_at: Optional[float] = None
            deadline = loop.time() + self.deadline_s
            try:
                while done < len(lines):
                    wake = min(deadline, release_at) if held and release_at is not None else deadline
                    try:
                        item = (arrivals.get_nowait() if not arrivals.empty() else
                                await asyncio.wait_for(arrivals.get(), max(0.0, wake - loop.time())))
                    except asyncio.TimeoutError:
                        if held and release_at is not None and loop.time() >= release_at:
                            for chunk in held:
                                playback.arrived(len(chunk.pcm), chunk.sentence + 1)
                                yield chunk.pcm
                            held = []
                            continue
                        log.warning("voice %s: sentence %d had no audio after %.1f s; %s speaks the rest", self.primary.name,
                                    done + 1, self.deadline_s, self.fallback.name)
                        break
                    if item is _END:
                        log.warning("voice %s ended in sentence %d; %s speaks the rest", self.primary.name,
                                    done + 1, self.fallback.name)
                        break
                    if isinstance(item, Exception):
                        log.warning("voice %s failed in sentence %d (%s); %s speaks the rest", self.primary.name, done + 1,
                                    type(item).__name__, self.fallback.name)
                        break
                    deadline = loop.time() + self.deadline_s
                    if item.pcm:
                        if release_at is None:
                            release_at = loop.time() + self.prebuffer_s
                        held.append(item)
                        if loop.time() >= release_at:
                            for chunk in held:
                                playback.arrived(len(chunk.pcm), chunk.sentence + 1)
                                yield chunk.pcm
                            held = []
                    if item.last:
                        done += 1
                for chunk in held:
                    playback.arrived(len(chunk.pcm), chunk.sentence + 1)
                    yield chunk.pcm
            finally:
                pump.cancel()
                await asyncio.gather(pump, return_exceptions=True)
        if done < len(lines):
            async for pcm in self.fallback.speak(" ".join(lines[done:]), role, out_rate, expression):
                yield pcm


async def _pump(chunks: AsyncIterator[Chunk], arrivals: asyncio.Queue) -> None:
    """Moves the primary's chunks to a queue, so waiting on the queue with a timeout never cancels the stream itself."""
    try:
        async with aclosing(chunks) as stream:
            async for chunk in stream:
                arrivals.put_nowait(chunk)
    except Exception as error:
        arrivals.put_nowait(error)
    finally:
        arrivals.put_nowait(_END)


class _UnderrunWatch:
    def __init__(self, rate: int) -> None:
        self.rate = rate
        self.started: Optional[float] = None
        self.sent_s = 0.0

    def arrived(self, pcm_bytes: int, sentence: int) -> None:
        now = time.monotonic()
        if self.started is None:
            self.started = now
        elif now > self.started + self.sent_s:
            gap = now - self.started - self.sent_s
            log.warning("voice underrun of %d ms in sentence %d", round(gap * 1000), sentence)
            self.started += gap
        self.sent_s += pcm_bytes / 2 / self.rate
