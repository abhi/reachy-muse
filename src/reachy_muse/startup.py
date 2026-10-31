# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Content-free timings for the path from process startup to microphone readiness."""

import logging
import time

log = logging.getLogger(__name__)


class StartupTimeline:
    def __init__(self):
        self._started = self._previous = time.monotonic()
        log.info("Reachy startup began")

    def mark(self, stage: str) -> None:
        now = time.monotonic()
        log.info("Reachy startup stage=%s elapsed=%.3fs total=%.3fs",
                 stage, now - self._previous, now - self._started)
        self._previous = now
