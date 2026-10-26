"""Generated motion for Reachy's longer moments: a greeting, thinking while Muse works, and a closing.

Generating a motion takes about a second on the companion, too long to wait for before a spoken
line, so each moment keeps one motion ready ahead of time and makes the next one after it plays.
A moment whose motion is not ready in time is skipped, and Reachy shows its fixed expressions
as it does with no companion. Intents come from the table below; no user words are sent.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, FrozenSet, Optional, Set, Tuple

from reachy_muse.capabilities import CapabilityUnavailable, Motion, MotionSource
from reachy_muse.contract.expressions import Expression
from reachy_muse.protocol import MAX_OUTSTANDING_ANIMATIONS

log = logging.getLogger(__name__)

# A closing waits at most this long after speech ends, then is skipped.
CLOSING_DEADLINE_S = 1.0
OUTPUT_WAIT_S = 30.0


class Moment(Enum):
    GREETING = "greeting"
    THINKING = "thinking"
    CELEBRATION = "celebration"
    APOLOGY = "apology"


@dataclass(frozen=True)
class MomentSpec:
    """``during`` are the states the motion may start in; ``start_within_s`` counts from its cue."""

    intents: Tuple[str, ...]
    duration_s: float
    during: FrozenSet[str]
    start_within_s: float
    loop: bool = False


AFTER_SPEECH = frozenset({"listening", "idle"})
MOMENTS: Dict[Moment, MomentSpec] = {
    Moment.THINKING: MomentSpec(("thinking hard", "pondering a question", "curious, working something out"),
                                4.0, frozenset({"thinking"}), 12.0, loop=True),
    Moment.GREETING: MomentSpec(("perking up to say hello", "a cheerful greeting", "waking up happy to see someone"),
                                2.5, frozenset({"listening"}), 3.0),
    Moment.CELEBRATION: MomentSpec(("celebrating good news", "happy and excited"), 2.5, AFTER_SPEECH, 1.5),
    Moment.APOLOGY: MomentSpec(("apologetic and sorry", "sad and sheepish"), 2.5, AFTER_SPEECH, 1.5),
}
# Thinking is the most frequent moment, then the greeting.
PREFETCH = (Moment.THINKING, Moment.GREETING, Moment.CELEBRATION, Moment.APOLOGY)
CLOSINGS: Dict[Expression, Moment] = {
    Expression.HAPPY: Moment.CELEBRATION,
    Expression.SURPRISED: Moment.CELEBRATION,
    Expression.SAD: Moment.APOLOGY,
    Expression.SHAKE: Moment.APOLOGY,
}


class MomentDirector:
    """Keeps a motion ready per moment and hands it to the hardware when its moment comes.

    With no source, a source whose companion lacks the animate op, or hardware that cannot play
    generated motion, every method returns at once and touches nothing. The hardware calls only
    queue or drop a motion for the motion thread, so they run on the event loop: a stop can never
    race a cue that is still handing its motion over.
    """

    def __init__(self, source: Optional[MotionSource], hardware) -> None:
        self._source = source
        self._hardware = hardware
        self._ready: Dict[Moment, Motion] = {}
        self._fetching: Dict[Moment, asyncio.Task] = {}
        self._slots = asyncio.Semaphore(MAX_OUTSTANDING_ANIMATIONS)
        self._intents = {moment: itertools.cycle(spec.intents) for moment, spec in MOMENTS.items()}
        self._refill_task: Optional[asyncio.Task] = None
        self._cue: Optional[asyncio.Task] = None
        self._closing: Optional[Tuple[Moment, asyncio.Task]] = None
        self._handed = False
        self._closed = False
        self._tasks: Set[asyncio.Task] = set()

    @property
    def active(self) -> bool:
        return (not self._closed and self._source is not None and self._source.available
                and callable(getattr(self._hardware, "play_motion", None)))

    async def keep_ready(self, every_s: float = 1.0) -> None:
        while not self._closed:
            self.refill()
            await asyncio.sleep(every_s)

    def refill(self) -> None:
        if self.active and (self._refill_task is None or self._refill_task.done()) and len(self._ready) < len(MOMENTS):
            self._refill_task = self._start(self._refill())

    def _start(self, work) -> asyncio.Task:
        task = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def close(self) -> None:
        self._closed = True
        try:
            self.stop()
        finally:
            tasks = tuple(self._tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._ready.clear()

    async def _refill(self) -> None:
        for moment in PREFETCH:
            if moment not in self._ready and not await self._fetch(moment):
                return

    def _fetch(self, moment: Moment) -> "asyncio.Future[bool]":
        """The one request in flight for ``moment``; it stores what it makes in ``_ready``."""
        task = self._fetching.get(moment)
        if task is None:
            task = self._fetching[moment] = self._start(self._generate(moment))
            task.add_done_callback(lambda done: self._fetching.pop(moment, None))
        return asyncio.shield(task)

    async def _generate(self, moment: Moment) -> bool:
        spec = MOMENTS[moment]
        async with self._slots:
            started = time.monotonic()
            try:
                motion = await self._source.generate(next(self._intents[moment]), spec.duration_s)
            except CapabilityUnavailable as exc:
                log.info("Reachy has no %s motion: %s", moment.value, exc)
                return False
        if motion is None:
            log.info("Reachy's companion made no %s motion", moment.value)
            return False
        self._ready[moment] = motion
        log.info("Reachy readied %.1f s of %s motion in %.2f s", len(motion.frames) / motion.fps, moment.value,
                 time.monotonic() - started)
        return True

    async def _take(self, moment: Moment) -> Optional[Motion]:
        if moment not in self._ready:
            await self._fetch(moment)
        motion = self._ready.pop(moment, None)
        self.refill()
        return motion

    def cue(self, moment: Moment) -> None:
        if self.active:
            self._replace_cue(self._play(moment, time.monotonic()))

    def prepare_closing(self, expression: Optional[Expression]) -> None:
        """Start readying the closing a reply's expression earns, if any, so it is ready when speech ends."""
        moment = CLOSINGS.get(expression)
        if moment is not None and self.active and self._closing is None:
            self._closing = (moment, self._start(self._take(moment)))

    def stop_cue(self) -> None:
        cue, self._cue = self._cue, None
        if cue is not None:
            cue.cancel()
        if self._handed:
            self._handed = False
            self._hardware.stop_motion()

    def stop(self) -> None:
        """Barge-in, a dropped request or a device command: fade out everything, closing included."""
        closing, self._closing = self._closing, None
        if closing is not None:
            closing[1].cancel()
        self.stop_cue()

    def finish_turn(self, output_idle: Callable[[], bool]) -> None:
        """The turn is done: fade out its thinking motion and play its closing once Reachy stops speaking."""
        closing, self._closing = self._closing, None
        self.stop_cue()
        if closing is not None:
            self._replace_cue(self._play_closing(*closing, output_idle))

    def _replace_cue(self, play) -> None:
        if self._cue is not None:
            self._cue.cancel()
        self._cue = self._start(play)

    async def _play_closing(self, moment: Moment, making: asyncio.Task, output_idle: Callable[[], bool]) -> None:
        try:
            deadline = time.monotonic() + OUTPUT_WAIT_S
            while not output_idle() and time.monotonic() < deadline:
                await asyncio.sleep(.05)
            spoken = time.monotonic()
            try:
                motion = await asyncio.wait_for(asyncio.shield(making), CLOSING_DEADLINE_S)
            except asyncio.TimeoutError:
                log.info("Reachy skipped its %s motion; it was not ready when speech ended", moment.value)
                return
            await self._play(moment, spoken, motion)
        finally:
            making.cancel()

    async def _play(self, moment: Moment, cued: float, motion: Optional[Motion] = None) -> None:
        spec = MOMENTS[moment]
        if motion is None:
            motion = await self._take(moment)
        remaining = cued + spec.start_within_s - time.monotonic()
        if motion is None or remaining <= 0:
            return
        self._handed = True
        self._hardware.play_motion(motion.frames, motion.fps, during=spec.during, loop=spec.loop,
                                   start_within_s=remaining)
        log.info("Reachy cued its %s motion %.2f s after the moment began", moment.value, time.monotonic() - cued)
