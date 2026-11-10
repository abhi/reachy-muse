"""Compact chroma/timbre recurrence and energy structure in source seconds."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True, slots=True)
class Section:
    start: float
    end: float
    family: int
    occurrence: int
    chorus: bool
    energy: float
    build: float
    drop: bool
    period: float
    character: str


@dataclass(frozen=True)
class SongPlan:
    sections: tuple[Section, ...] = ()
    complete: bool = False

    def at(self, seconds: float) -> Section | None:
        # Plans are compact, but long files can still contain thousands of sections.
        low, high = 0, len(self.sections)
        while low < high:
            middle = (low + high) // 2
            if self.sections[middle].start <= seconds:
                low = middle + 1
            else:
                high = middle
        section = self.sections[low - 1] if low else None
        return section if section is not None and seconds < section.end else None

    def gain(self, seconds: float) -> float:
        section = self.at(seconds)
        gain = 1.
        if section is not None:
            gain += min(.36, .12 * section.occurrence) + .2 * section.build
            if section.drop and seconds < section.start + section.period * 4:
                gain *= 1.7
        if self.complete and self.sections:
            gain *= min(1., max(0., (self.sections[-1].end - seconds) / 4))
        return gain


def _unit(values):
    return values / max(float(np.linalg.norm(values)), 1e-9)


class StructureTracker:
    """One-second summaries, at most 64 motif families and 4096 sections."""

    def __init__(self, rate: int, *, rolling: bool = True):
        self.rate, self.rolling = rate, rolling
        self._frames = []
        self._second = 0
        self._sections = deque(maxlen=4096)
        self._families = []
        self._counts = []
        self._descriptor = None
        self._sum = None
        self._bins = 0
        self._energy = deque(maxlen=4)
        self._current = None
        self._latest = SongPlan()

    def feed(self, frames) -> None:
        for frame in frames:
            second = frame.sample // self.rate
            if second != self._second and self._frames:
                self._summarize((self._second + 1))
            self._second = second
            self._frames.append(frame)

    def _summarize(self, end):
        frames, self._frames = self._frames, []
        beginning = frames[0].sample // self.rate
        chroma = _unit(np.mean([f.chroma for f in frames], axis=0))
        timbre = _unit(np.mean([f.timbre for f in frames], axis=0))
        descriptor = _unit(np.concatenate((chroma * .9, timbre * .4)))
        energy = float(np.mean([f.loudness for f in frames]))
        period = float(np.median([f.period for f in frames]))
        character = frames[-1].character
        previous = self._energy[-1] if self._energy else energy
        drop = energy > max(.15, previous * 1.8) and energy - previous > .1
        self._energy.append(energy)
        rising = len(self._energy) == 4 and all(b > a * 1.03 for a, b in zip(self._energy, list(self._energy)[1:]))
        build = min(1., (energy - self._energy[0]) / .3) if rising else 0.
        novelty = self._descriptor is not None and float(descriptor @ self._descriptor) < .86
        boundary = self._current is None or (beginning - self._current.start >= 2 and (novelty or drop))
        if boundary:
            if self._current is not None:
                self._sections.append(self._current)
            scores = [float(descriptor @ prototype) for prototype in self._families]
            family = int(np.argmax(scores)) if scores and max(scores) >= .9 else len(self._families)
            if family == len(self._families):
                if family == 64:
                    family = int(np.argmax(scores))
                else:
                    self._families.append(descriptor)
                    self._counts.append(0)
            occurrence = self._counts[family]
            self._counts[family] += 1
            self._current = Section(beginning, end, family, occurrence, False, energy, build, drop, period, character)
            self._sum, self._bins = descriptor.copy(), 1
        else:
            self._bins += 1
            self._sum += descriptor
            current = self._current
            self._current = replace(current, end=end, energy=current.energy + (energy - current.energy) / self._bins,
                                    build=max(current.build, build))
        self._descriptor = _unit(self._sum)
        if self.rolling:
            while self._sections and self._sections[0].end < end - 120:
                self._sections.popleft()
        self._latest = self._plan(False)

    def _plan(self, complete):
        sections = (*self._sections, *((self._current,) if self._current is not None else ()))
        mean = float(np.mean([s.energy for s in sections])) if sections else 0.
        return SongPlan(tuple(replace(s, chorus=self._counts[s.family] > 1 and s.energy >= mean)
                              for s in sections), complete)

    def plan(self, *, complete=False) -> SongPlan:
        if complete:
            if self._frames:
                self._summarize((self._frames[-1].sample + 1) / self.rate)
            self._latest = self._plan(True)
        return self._latest
