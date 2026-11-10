# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Bounded streaming beat estimates and poses sampled by the existing motor owner."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading

import numpy as np

from reachy_muse.motion.layers import crossfade_weight


@dataclass(frozen=True)
class Beat:
    sample: int
    period: float
    index: int
    energy: float


class BeatTracker:
    """Ten millisecond onsets, with a median tempo over the last eight intervals.

    Samples are normalized mono PCM. Beat positions count source samples,
    independent of packet boundaries or decode speed. Four-beat groupings are
    inferred from the first onset, not a musical downbeat classifier.
    """

    def __init__(self, rate: int):
        self.rate = rate
        self.hop = max(1, round(rate * .01))
        self._pending = np.empty(0, dtype=np.float32)
        self._sample = 0
        self._previous = 0.0
        self._floor = 0.0
        self._last = -math.inf
        self._intervals = deque(maxlen=8)
        self.period = .5
        self.count = 0

    def feed(self, pcm) -> tuple[Beat, ...]:
        audio = np.concatenate((self._pending, pcm))
        count = len(audio) // self.hop
        self._pending = audio[count * self.hop:].copy()
        if not count:
            return ()
        frames = audio[:count * self.hop].reshape(count, self.hop)
        levels = np.sqrt(np.mean(frames * frames, axis=1))
        beats = []
        for level in levels:
            level = float(level)
            at = self._sample / self.rate
            if (level > max(.006, self._floor * 1.8, self._previous * 1.5)
                    and at - self._last >= .24):
                interval = at - self._last
                if .3 <= interval <= 1.0:
                    self._intervals.append(interval)
                    self.period = float(np.median(self._intervals))
                elif interval > 2:
                    self._intervals.clear()
                energy = min(1., max(0., (20 * math.log10(level + 1e-9) + 50) / 34))
                beats.append(Beat(self._sample, self.period, self.count, energy))
                self.count += 1
                self._last = at
            self._floor += .04 * (level - self._floor)
            self._previous = level
            self._sample += self.hop
        return tuple(beats)


def dance_pose(beats: float, energy: float) -> np.ndarray:
    """Small offsets in mm/degrees, with a smooth change of pattern every 16 beats."""
    def pattern(index):
        phase = 2 * math.pi * beats
        values = np.zeros(9)
        values[2] = -2.5 * math.cos(phase)
        values[4] = 5 * math.cos(phase)
        values[8] = 7 * math.sin(phase / 8)
        if index == 1:
            values[3] = 4 * math.sin(phase / 2)
            values[5] = 3 * math.sin(phase / 4)
        elif index == 2:
            values[0] = 2 * math.sin(phase / 2)
            values[2] *= .65
            values[4] *= .7
            values[8] *= .7
        distance = (beats + 2) % 4 - 2
        flick = math.cos(math.pi * distance / .7) ** 2 if abs(distance) < .35 else 0.
        values[6], values[7] = -16 * flick, 16 * flick
        return values

    bar_group = math.floor(beats / 16)
    blend = crossfade_weight(beats % 16, 2)
    return max(0., min(1., energy)) * (
        pattern((bar_group - 1) % 3) * (1 - blend) + pattern(bar_group % 3) * blend)


class DanceTimeline:
    def __init__(self):
        self._lock = threading.Lock()
        self._beats = deque(maxlen=256)
        self._active = False
        self._changed = 0.
        self._last = np.zeros(9)
        self._release = np.zeros(9)

    def offer(self, beats: tuple[Beat, ...], offset: int, start: float, rate: int) -> None:
        with self._lock:
            for beat in beats:
                self._beats.append((start + (beat.sample - offset) / rate, beat))

    def clear(self) -> None:
        with self._lock:
            self._beats.clear()

    def sample(self, now: float, *, eligible: bool) -> np.ndarray:
        with self._lock:
            while len(self._beats) > 1 and self._beats[1][0] <= now:
                self._beats.popleft()
            current = self._beats[0] if self._beats and self._beats[0][0] <= now else None
            active = eligible and current is not None and now - current[0] < 2
            if active != self._active:
                self._release = self._last.copy()
                self._changed = now
                self._active = active
            blend = crossfade_weight(now - self._changed)
            if active:
                at, beat = current
                phase = beat.index + (now - at) / beat.period
                decay = 1 - crossfade_weight(now - at - max(.6, beat.period * 1.25), .6)
                pose = dance_pose(phase, beat.energy * decay)
                self._last = self._release * (1 - blend) + pose * blend
            else:
                self._last = self._release * (1 - blend)
            return self._last.copy()
