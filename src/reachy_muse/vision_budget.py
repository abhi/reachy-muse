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
        self._suspended_at = None
        self._probing = False
        self._capture_at = None
        self._capturing = False
        self._busy = False
        self.level = 0

    def capture_active(self, active: bool) -> None:
        with self._lock:
            if self.level == 2:
                log.warning("Vision budget: resumed; capture %s", "restarted" if active else "stopped")
            self._capturing = active
            self._capture_at = self._clock() if active else None
            self._busy = False
            self._windows.clear()
            self._healthy_since = None
            self._suspended_at = None
            self._probing = False
            self.level = 0

    def audio_busy(self, busy: bool) -> None:
        with self._lock:
            self._busy = busy and self._capturing
            if busy:
                self._windows.pop("wake", None)

    def observe(self, source: str, audio_s: float, *, now=None) -> None:
        now = self._clock() if now is None else now
        with self._lock:
            if not self._capturing:
                return
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
        self._resume_if_due(now)
        previous = self.level
        if coverage < .95:
            self.level = min(1 if self._probing else 2, self.level + 1)
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
            self._suspended_at = now if self.level == 2 else None
            if self.level == 0:
                self._probing = False
            state = "resumed at reduced rate" if previous == 2 else ("normal", "throttled", "suspended")[self.level]
            log.warning("Vision budget: %s; %s audio coverage %.3f",
                        state, source, coverage)

    def _resume_if_due(self, now):
        if self._suspended_at is not None and now - self._suspended_at >= 60:
            self.level = 1
            self._probing = True
            self._suspended_at = None
            self._healthy_since = None
            log.warning("Vision budget: resumed at reduced rate; audio loss persists after 60s suspended")

    def delay(self) -> float:
        with self._lock:
            self._resume_if_due(self._clock())
            if (self._busy or self.level == 2 or self._capture_at is not None
                    and self._clock() - self._capture_at > .5):
                return math.inf
            return 3.0 if self.level == 1 else 1.0
