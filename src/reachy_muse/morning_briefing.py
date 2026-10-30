# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from datetime import time as wall_time
from pathlib import Path
from typing import Callable, Protocol, Tuple

from reachy_muse.arrival import ArrivalCamera, WatchPose
from reachy_muse.capabilities import SpokenLine
from reachy_muse.contract.expressions import Expression
from reachy_muse.contract.replies import spoken_reply
from reachy_muse.local_backends import split_sentences
from reachy_muse.protocol import MAX_LINE_BYTES

log = logging.getLogger(__name__)

DAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
IDLE_POLL_S = 5.0
HOLD_EVERY_S = 15.0
HOLD_LINES = ("Still gathering the market news.", "Checking the latest tech stories.", "Almost there.")
APOLOGY = "Sorry, I couldn't get your briefing this morning."


@dataclass(frozen=True)
class BriefingSchedule:
    start: wall_time
    end: wall_time
    days: frozenset[int]

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("morning briefing window must end after it starts on the same day")

    @classmethod
    def parse(cls, window: str = "04:50-06:00", days: str | None = None) -> BriefingSchedule:
        try:
            start_text, end_text = window.split("-")
            start, end = _clock_time(start_text), _clock_time(end_text)
        except ValueError:
            raise ValueError(f"morning briefing window must look like 04:50-06:00, not {window!r}") from None
        return cls(start, end, _weekdays(days or "daily"))

    def window_date(self, now_local: datetime) -> date | None:
        if now_local.weekday() in self.days and self.start <= now_local.time() < self.end:
            return now_local.date()
        return None

    def __str__(self) -> str:
        days = "daily" if len(self.days) == 7 else ",".join(DAY_NAMES[day] for day in sorted(self.days))
        return f"{self.start:%H:%M}-{self.end:%H:%M} {days}"


@dataclass(frozen=True)
class TestWindow:
    start: datetime
    end: datetime

    def window_date(self, now_local: datetime) -> date | None:
        return self.start.date() if self.start <= now_local < self.end else None

    def __str__(self) -> str:
        return f"test {self.start:%Y-%m-%d %H:%M:%S} to {self.end:%Y-%m-%d %H:%M:%S}"


def _clock_time(text: str) -> wall_time:
    hours, minutes = text.strip().split(":")
    if len(minutes) != 2:
        raise ValueError(text)
    return wall_time(int(hours), int(minutes))


def _weekdays(text: str) -> frozenset[int]:
    if text.strip().lower() == "daily":
        return frozenset(range(7))
    days = set()
    for item in text.lower().split(","):
        first, _, last = item.strip().partition("-")
        try:
            low = DAY_NAMES.index(first)
            high = DAY_NAMES.index(last) if last else low
        except ValueError:
            raise ValueError(f"morning briefing days must look like daily, mon-fri or mon,wed,sat, "
                             f"not {text!r}") from None
        if high < low:
            raise ValueError(f"morning briefing day range {item.strip()!r} runs backwards")
        days.update(range(low, high + 1))
    return frozenset(days)


class Clock(Protocol):
    def now_local(self) -> datetime: ...
    def monotonic(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now_local(self) -> datetime:
        return datetime.now().astimezone()

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class DayLedger:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._marked: date | None = None

    def done(self, day: date) -> bool:
        return self._marked == day or self._saved() == day.isoformat()

    def mark(self, day: date) -> None:
        self._marked = day
        if self.path is None:
            return
        partial = self.path.with_name(self.path.name + ".tmp")
        try:
            partial.write_text(json.dumps({"version": 1, "briefed": day.isoformat()}), encoding="utf-8")
            os.replace(partial, self.path)
        except OSError as exc:
            log.warning("Reachy morning briefing: could not record today's briefing: %s", type(exc).__name__)

    def _saved(self) -> str | None:
        if self.path is None:
            return None
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return saved.get("briefed") if isinstance(saved, dict) else None


def briefing_request(template: str, now_local: datetime) -> str:
    when = f"{now_local:%A, %B} {now_local.day}, {now_local:%H:%M} local time"
    return template.replace("{when}", when)


def briefing_lines(text: str) -> Tuple[SpokenLine, ...]:
    spoken, expression = spoken_reply(text)
    pieces = [piece for sentence in split_sentences(spoken) for piece in _fit(sentence)]
    return tuple(SpokenLine(piece, Expression.parse(expression) if index == len(pieces) - 1 else None, "answer")
                 for index, piece in enumerate(pieces))


def _fit(sentence: str) -> list[str]:
    if _bytes(sentence) <= MAX_LINE_BYTES:
        return [sentence]
    words = []
    for clause in re.split(r"(?<=[,;:])\s+", sentence):
        words.extend([clause] if _bytes(clause) <= MAX_LINE_BYTES else clause.split())
    lines = [words[0]]
    for word in words[1:]:
        if _bytes(lines[-1] + " " + word) <= MAX_LINE_BYTES:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def _bytes(text: str) -> int:
    return len(text.encode("utf-8"))


Lines = Tuple[SpokenLine, ...]


class MorningVoice(Protocol):
    async def greet(self, *, waiting: bool) -> None: ...

    async def fetch(self, request: str, *, attended: Callable[[], bool]) -> Lines | None: ...

    async def hold(self, line: str) -> None: ...

    async def brief(self, lines: Lines) -> float: ...


class MorningRoutine:
    def __init__(self, schedule: BriefingSchedule | TestWindow, request: str, camera: ArrivalCamera, ledger: DayLedger, *,
                 pose: WatchPose | None = None, clock: Clock = SystemClock(),
                 refresh_after_s: float = 2400.0, poll_s: float = 0.25) -> None:
        self.schedule = schedule
        self.request = request
        self.camera = camera
        self.ledger = ledger
        self.pose = pose
        self.clock = clock
        self.refresh_after_s = refresh_after_s
        self.poll_s = poll_s
        self._chat: tuple[str, str | None] | None = None
        self._arrived = False
        self._arrived_on: date | None = None
        self._window_day: date | None = None
        self._ready: Lines | None = None
        self._ready_at = 0.0
        self._refreshed = False
        self._camera_day: date | None = None

    async def watch_camera(self) -> None:
        diagnostics_at = self.clock.monotonic() + 60
        try:
            while True:
                day = self.schedule.window_date(self.clock.now_local())
                reason = "window closed"
                if day is not None and self.ledger.done(day):
                    day = None
                    reason = "briefing completed"
                if day != self._camera_day:
                    if self._camera_day is not None:
                        self.camera.stop_watching()
                        log.info("Arrival watcher stopped: %s", reason)
                    self._camera_day = day
                    if day is not None:
                        self.camera.start_watching(self.pose)
                        diagnostics_at = self.clock.monotonic() + 60
                        log.info("Arrival watcher started: window opened")
                if day is not None and self.clock.monotonic() >= diagnostics_at:
                    self.camera.log_arrival_diagnostics()
                    diagnostics_at = self.clock.monotonic() + 60
                await self.clock.sleep(self.poll_s if day is not None else IDLE_POLL_S)
        except Exception as exc:
            log.warning("Arrival watcher supervision failed: %s", type(exc).__name__)
            raise
        finally:
            if self._camera_day is not None:
                self.camera.stop_watching()
                self._camera_day = None
                log.info("Arrival watcher stopped: service shutdown")

    def select_chat(self, vm_id: str, session_id: str | None) -> None:
        chat = (vm_id, session_id)
        if self._chat != chat:
            self._chat = chat
            self._arrived = False
            self._arrived_on = None
            self._window_day = None
            self._ready = None
            self._ready_at = 0.0
            self._refreshed = False

    async def run(self, voice: MorningVoice) -> None:
        while True:
            day = self.schedule.window_date(self.clock.now_local())
            if day is None or self.ledger.done(day):
                await self.clock.sleep(IDLE_POLL_S)
                continue
            await self._watch(day, voice)

    async def _watch(self, day: date, voice: MorningVoice) -> None:
        first_watch = self._window_day != day
        if first_watch:
            self._window_day = day
            self._ready = None
            self._ready_at = 0.0
            self._refreshed = False
        if self._arrived_on == day:
            log.info("Reachy morning briefing: resuming the briefing after a reconnect")
            if (self._ready is not None and not self._refreshed
                    and self.clock.monotonic() - self._ready_at >= self.refresh_after_s):
                self._refreshed = True
                await self._brief(day, voice, None, None, fallback=self._ready)
            else:
                await self._brief(day, voice, self._ready, None)
            return
        self._arrived = False
        log.info("Reachy morning briefing: watching for arrival")
        pending = self._fetch(day, voice, attended=lambda: self._arrived) if first_watch else None
        fetch_started = self.clock.monotonic()
        try:
            while True:
                now = self.clock.monotonic()
                if pending is not None and pending.done():
                    lines, pending = pending.result(), None
                    if lines:
                        log.info("Reachy morning briefing: prefetched in %.1f s", now - fetch_started)
                if self.schedule.window_date(self.clock.now_local()) != day:
                    self._ready = None
                    log.info("Reachy morning briefing: window closed without an arrival")
                    return
                if self._camera_day == day and self.camera.arrival() is not None:
                    self._arrived = True
                    self._arrived_on = day
                    if self._ready is not None and pending is not None:
                        pending.cancel()
                        await asyncio.gather(pending, return_exceptions=True)
                        pending = None
                    log.info("Reachy morning briefing: arrival detected at %s", f"{self.clock.now_local():%H:%M}")
                    await self._brief(day, voice, self._ready, pending)
                    return
                if (self._ready is not None and not self._refreshed and pending is None
                        and now - self._ready_at >= self.refresh_after_s):
                    self._refreshed = True
                    pending = self._fetch(day, voice, attended=lambda: False)
                    fetch_started = now
                await self.clock.sleep(self.poll_s)
        finally:
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    async def _brief(self, day: date, voice: MorningVoice, ready: Lines | None,
                     pending: asyncio.Task | None, *, fallback: Lines | None = None) -> None:
        if ready is not None:
            await voice.greet(waiting=False)
            lines = ready
        else:
            await voice.greet(waiting=True)
            holds = itertools.cycle(HOLD_LINES)
            next_hold = self.clock.monotonic() + HOLD_EVERY_S

            async def holding(task: asyncio.Task) -> Lines | None:
                nonlocal next_hold
                while not task.done():
                    if self.clock.monotonic() >= next_hold:
                        await voice.hold(next(holds))
                        next_hold = self.clock.monotonic() + HOLD_EVERY_S
                    await self.clock.sleep(self.poll_s)
                return task.result()

            lines = await holding(pending) if pending is not None else None
            if lines is None:
                started = self.clock.monotonic()
                pending = self._fetch(day, voice, attended=lambda: True)
                try:
                    lines = await holding(pending)
                finally:
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                if lines:
                    log.info("Reachy morning briefing: fetched live in %.1f s", self.clock.monotonic() - started)
            if lines is None:
                lines = fallback
            if lines is None:
                await voice.hold(APOLOGY)
                self.ledger.mark(day)
                return
            if not lines:
                log.info("Reachy morning briefing: the user dropped it")
                self.ledger.mark(day)
                return
        spoken = await voice.brief(lines)
        self.ledger.mark(day)
        log.info("Reachy morning briefing: spoke briefing (%.0f s)", spoken)

    def _fetch(self, day: date, voice: MorningVoice, *, attended: Callable[[], bool]) -> asyncio.Task:
        request = briefing_request(self.request, self.clock.now_local())

        async def fetch() -> Lines | None:
            lines = await voice.fetch(request, attended=attended)
            if lines:
                self._ready, self._ready_at = lines, self.clock.monotonic()
            elif lines == () and attended():
                self._ready = None
                self.ledger.mark(day)
            return lines

        return asyncio.create_task(fetch())
