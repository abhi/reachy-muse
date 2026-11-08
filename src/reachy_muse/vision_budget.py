# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Audio owns the CPU budget; local vision sheds work when capture falls behind."""

from __future__ import annotations

import logging
import math
import threading
import time

log = logging.getLogger(__name__)


class VisionBudget:
    def __init__(self, *, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._windows = {}
        self._healthy_since = None
        self._capture_at = None
        self._busy = False
        self.level = 0

    def audio_busy(self, busy: bool) -> None:
        with self._lock:
            self._busy = busy
            if busy:
                self._windows.pop("wake", None)

    def observe(self, source: str, audio_s: float, *, now=None) -> None:
        now = self._clock() if now is None else now
        with self._lock:
            if source == "capture":
                if audio_s:
                    self._capture_at = now
            started, total = self._windows.get(source, (now - audio_s, 0.0))
            total += audio_s
            self._windows[source] = (started, total)
            if now - started < 2:
                return
            coverage = total / (now - started)
            self._windows[source] = (now, 0.0)
            self._coverage(coverage, now, source)

    def _coverage(self, coverage, now, source):
        previous = self.level
        if coverage < .95:
            self.level = min(2, self.level + 1)
            self._healthy_since = None
        elif coverage >= .98:
            if self._healthy_since is None:
                self._healthy_since = now
            if self.level and now - self._healthy_since >= 20:
                self.level -= 1
                self._healthy_since = now
        else:
            self._healthy_since = None
        if self.level != previous:
            log.warning("Vision budget: %s; %s audio coverage %.3f",
                        ("normal", "throttled", "suspended")[self.level], source, coverage)

    def delay(self) -> float:
        with self._lock:
            if (self._busy or self.level == 2 or self._capture_at is not None
                    and self._clock() - self._capture_at > .5):
                return math.inf
            return 3.0 if self.level == 1 else 1.0
