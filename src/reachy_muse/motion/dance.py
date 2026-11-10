# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Bounded streaming beat estimates and poses sampled by the existing motor owner."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
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


DANCE_CAP = np.array([2., 2., 3., 5., 7., 4., 14., 14., 9.])
SMOOTH_S = .12
JERK_LIMIT = 8 * DANCE_CAP / SMOOTH_S ** 3


def feature_pose(frame) -> np.ndarray:
    phase = frame.phase * 2 * math.pi
    pose = np.zeros(9)
    if not frame.music or frame.loudness < .008:
        return pose
    pace = .7 if frame.character == "calm" else 1.
    pose[8] = 9 * frame.bass * math.sin(phase / 4)
    pose[2] = -3 * frame.snare
    pose[4] = 7 * frame.snare
    pose[6:8] = (-14 * frame.treble, 14 * frame.treble)
    pose[0] = 2 * frame.harmony * math.sin(phase / 16)
    pose[1] = 2 * frame.harmony * math.cos(phase / 16)
    pose[3] = 5 * frame.harmony * math.sin(phase / 8)
    pose[5] = 4 * frame.harmony * math.cos(phase / 8)
    if frame.regularity < .25 and frame.harmony > .1:
        pose += .08 * frame.harmony * dance_pose(frame.phase, frame.loudness)
    return np.clip(pose * frame.loudness * pace, -DANCE_CAP, DANCE_CAP)


class DanceTimeline:
    def __init__(self):
        self._lock = threading.Lock()
        self._beats = deque(maxlen=256)
        self._features = deque(maxlen=256)
        self._filter = np.zeros((3, 9))
        self._time = None
        self._target = np.zeros(9)
        self._current = None
        self._eligible = False

    def offer(self, beats: tuple[Beat, ...], offset: int, start: float, rate: int) -> None:
        with self._lock:
            for beat in beats:
                self._beats.append((start + (beat.sample - offset) / rate, beat))

    def offer_features(self, frames, offset: int, start: float, rate: int, plan=None, signatures=None) -> None:
        with self._lock:
            for frame in frames:
                seconds = frame.sample / rate
                section = plan.at(seconds) if plan is not None else None
                if section is not None:
                    frame = replace(frame, phase=(seconds - section.start) / frame.period)
                gain = plan.gain(seconds) if plan is not None else 1.
                pose = feature_pose(frame)
                if section is not None and signatures and section.family in signatures and frame.music:
                    pose += signatures[section.family].sample(section, seconds, frame)
                pose = np.clip(pose * gain, -DANCE_CAP, DANCE_CAP)
                self._features.append((start + (frame.sample - offset) / rate, pose))

    def clear(self) -> None:
        with self._lock:
            self._beats.clear()
            self._features.clear()
            self._current = None

    def _advance(self, now):
        step = min(max(0., now - self._time) / SMOOTH_S, 100.)
        a, b, c = self._filter - self._target
        self._filter = (np.array((a, b + a * step, c + b * step + a * step ** 2 / 2))
                        * math.exp(-step) + self._target)
        self._time = now

    def _advance_until(self, now):
        if self._current is not None:
            expiry = self._current[0] + .24
            if self._time < expiry < now:
                self._advance(expiry)
                self._target = np.zeros(9)
        self._advance(now)

    def sample(self, now: float, *, eligible: bool) -> np.ndarray:
        with self._lock:
            if self._time is None:
                self._time = now
                self._eligible = eligible
            now = max(now, self._time)
            while self._features and self._features[0][0] <= now:
                at, pose = self._features.popleft()
                self._advance_until(max(self._time, at))
                self._current = (at, pose)
                self._target = pose if self._eligible else np.zeros(9)
            self._advance_until(now)
            target = np.zeros(9)
            if eligible:
                if self._current is not None:
                    at, pose = self._current
                    if at <= now < at + .24:
                        target = pose
                elif not self._features:
                    while len(self._beats) > 1 and self._beats[1][0] <= now:
                        self._beats.popleft()
                    if self._beats:
                        at, beat = self._beats[0]
                        if at <= now < at + 1:
                            target = np.clip(dance_pose(beat.index + (now - at) / beat.period,
                                                       beat.energy), -DANCE_CAP, DANCE_CAP)
            self._target = target
            self._eligible = eligible
            return self._filter[2].copy()

    @property
    def jerk(self) -> np.ndarray:
        with self._lock:
            a, b, c = self._filter
            return (self._target - 3 * a + 3 * b - c) / SMOOTH_S ** 3
