# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Short, speech-only direction history for wake attention. No audio is retained."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import math
from statistics import median
import time
from typing import Callable

from reachy_muse.contract.commands import POSE_LIMITS


@dataclass(frozen=True)
class DoAReading:
    """World bearing in radians, with left=0 and front=pi/2."""

    angle: float
    speech_detected: bool


@dataclass(frozen=True)
class SpeakerYaw:
    """Absolute SDK head and body yaw, in degrees. They are not additive."""

    head: float
    body: float


@dataclass(frozen=True)
class DirectionDecision:
    reason: str
    samples: int = 0
    span_ms: float = 0.0
    boundary_age_ms: float = math.nan
    last_age_ms: float = math.nan
    mad_deg: float = math.nan
    inlier: float = 0.0
    angle: float | None = None


@dataclass(frozen=True)
class DirectionSnapshot:
    samples: tuple[tuple[float, float], ...]
    captured_at: float
    empty_reason: str

    def decide(self, phrase_end: float | None) -> DirectionDecision:
        if phrase_end is None:
            return DirectionDecision("untimed")
        samples = [(at, angle) for at, angle in self.samples
                   if phrase_end - 1.5 <= at <= phrase_end]
        for index in range(len(samples) - 1, 0, -1):
            if samples[index][0] - samples[index - 1][0] > .35:
                samples = samples[index:]
                break
        age = self.captured_at - phrase_end
        span = samples[-1][0] - samples[0][0] if samples else 0.0
        last_age = phrase_end - samples[-1][0] if samples else math.nan
        direction = median(angle for _, angle in samples) if samples else None
        deviations = [abs(angle - direction) for _, angle in samples]
        mad = median(deviations) if deviations else math.nan
        inlier = sum(delta <= math.radians(20) for delta in deviations) / len(samples) if samples else 0.0
        if not samples:
            reason = self.empty_reason
        elif not 0 <= age <= 1.0 or last_age > .45:
            reason = "stale"
        elif len(samples) < 2 or span < .1:
            reason = "too_few"
        elif mad > math.radians(12) or median(deviations[-3:]) > math.radians(20) or inlier < .75:
            reason = "dispersed"
        else:
            reason = "ok"
        return DirectionDecision(reason, len(samples), span * 1000, age * 1000,
                                 last_age * 1000, math.degrees(mad), inlier,
                                 direction if reason == "ok" else None)


def speaker_yaw(angle: float) -> SpeakerYaw:
    """Map a world bearing into the app's safe absolute yaw range."""
    if (isinstance(angle, bool) or not isinstance(angle, (int, float))
            or not math.isfinite(angle) or not -math.pi / 2 <= angle <= 3 * math.pi / 2):
        raise ValueError("Direction must be finite radians between -pi/2 and 3pi/2")
    low, high = POSE_LIMITS["yaw"]
    head = max(low, min(high, math.degrees(math.pi / 2 - angle)))
    body = math.copysign(max(0.0, abs(head) - 20.0), head)
    low, high = POSE_LIMITS["body_yaw"]
    return SpeakerYaw(head, max(low, min(high, body)))


class WakeDirection:
    """One event-loop owner samples an off-thread, bounded-time DoA source.

    ``eligible`` must exclude robot output even when the microphone has AEC.
    Clearing the history also invalidates any read already in flight.
    """

    def __init__(self, read: Callable[[], DoAReading | None], *, clock: Callable = time.monotonic,
                 sleep: Callable = asyncio.sleep):
        self._read = read
        self._clock = clock
        self._sleep = sleep
        self._samples: deque[tuple[float, float]] = deque(maxlen=80)
        self._generation = 0
        self._empty_reason = "too_few"

    def clear(self, reason: str = "too_few") -> None:
        self._generation += 1
        self._samples.clear()
        self._empty_reason = reason

    def snapshot(self) -> DirectionSnapshot:
        return DirectionSnapshot(tuple(self._samples), self._clock(), self._empty_reason)

    def discard_through(self, at: float) -> None:
        while self._samples and self._samples[0][0] <= at:
            self._samples.popleft()

    def samples_between(self, start: float, end: float) -> tuple[tuple[float, float], ...]:
        return tuple((at, angle) for at, angle in self._samples if start <= at <= end)

    def estimate(self, phrase_end: float) -> float | None:
        """Median of the last speech segment within 1.5 seconds of the keyword end."""
        return self.snapshot().decide(phrase_end).angle

    async def sample_while(self, eligible: Callable[[], bool]) -> None:
        try:
            while True:
                if not eligible():
                    self.clear("ineligible")
                    await self._sleep(.05)
                    continue
                generation = self._generation
                started = self._clock()
                try:
                    reading = await asyncio.to_thread(self._read)
                except Exception:
                    reading = None
                delay = .2
                if generation != self._generation or not eligible():
                    self.clear("ineligible")
                elif self._clock() - started > .15:
                    self.clear("timeout")
                elif reading is None:
                    self.clear("unavailable")
                    delay = 1.0
                elif reading.speech_detected:
                    self._samples.append((started, reading.angle))
                    while self._samples and self._samples[0][0] < started - 3:
                        self._samples.popleft()
                await self._sleep(delay)
        finally:
            self.clear()
