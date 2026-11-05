# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Bounded sentence events and smooth, audio-timed punctuation accents."""

from __future__ import annotations

import math
import queue
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SentenceEvent:
    at: float
    epoch: int
    kind: Literal["start", "end"]
    value: tuple[str, int] | tuple[str, int, object]


class SentenceTimeline:
    def __init__(self) -> None:
        self.events: queue.Queue[SentenceEvent] = queue.Queue(maxsize=64)
        self.pending: list[SentenceEvent] = []
        self.epoch = 0
        self.accent = None
        self.previous = [0.0] * 9
        self.retargeted = 0.0
        self.last = [0.0] * 9
        self.release = None

    def offer(self, event: SentenceEvent) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full:
            pass

    def cancel(self, *, fade: bool = True) -> None:
        self.epoch += 1
        if not fade:
            self.accent = self.release = None
            self.last = [0.0] * 9

    def due(self, now: float) -> list[SentenceEvent]:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event.epoch == self.epoch:
                self.pending.append(event)
        self.pending = [event for event in self.pending if event.epoch == self.epoch]
        due = [event for event in self.pending if event.at <= now]
        self.pending = [event for event in self.pending if event.at > now][-64:]
        return due

    def set_accent(self, at: float, terminal: str, epoch: int, now: float) -> None:
        self.previous = self.offset(now)
        self.retargeted = now
        self.accent = (at, terminal, epoch)
        self.release = None

    def offset(self, now: float):
        """A small sinusoidal gesture, zero at both ends and bounded by construction."""
        result = [0.0] * 9
        if self.accent is not None and self.accent[2] != self.epoch:
            self.accent = None
            self.release = (now, self.last)
        if self.accent is None:
            if self.release is not None:
                started, source = self.release
                phase = min(1.0, max(0.0, (now - started) / 0.2))
                weight = 1 - phase * phase * (3 - 2 * phase)
                result = [value * weight for value in source]
                if phase == 1:
                    self.release = None
            self.last = result
            return result
        started, terminal, epoch = self.accent
        if now >= started + 0.5:
            self.accent = None
            self.last = result
            return result
        phase = max(0.0, (now - started) / 0.5)
        weight = math.sin(math.pi * phase) ** 2
        if terminal == "question":
            result[3] = 4 * weight
            result[6], result[7] = -8 * weight, 8 * weight
        else:
            result[4] = 4 * weight
            if terminal == "exclamation":
                result[6], result[7] = -10 * weight, 10 * weight
        blend = min(1.0, max(0.0, (now - self.retargeted) / 0.2))
        blend = blend * blend * (3 - 2 * blend)
        self.last = [old * (1 - blend) + new * blend for old, new in zip(self.previous, result)]
        return self.last
