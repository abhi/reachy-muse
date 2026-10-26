# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""When each stage of one voice turn happened, logged as one line so every turn yields its latency breakdown.

``tools/turn_timeline.py`` turns these lines from the robot's journal into a table.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

# In the order a turn normally reaches them. Times are seconds after the turn's earliest stage.
STAGES = (
    "wake",            # the wake phrase opened the microphone; only on the first request after it
    "speech_end",      # the robot's voice detector last heard speech
    "endpoint",        # the turn ended: the companion's endpoint, or the robot's silence
    "transcript",      # the request's words were ready
    "dispatch",        # the request was handed to the Muse link
    "muse_ack",        # Muse accepted it
    "chat_audio",      # Reachy's own answer to small talk began to play; Muse never saw the turn
    "ack_audio",       # Reachy's acknowledgement began to play
    "first_status",    # Muse's first public status for it
    "first_text",      # the first words of Muse's answer
    "reply_done",      # Muse's first answer message was complete
    "narration_done",  # the answer's spoken lines were ready
    "first_audio",     # the answer began to play
)


class TurnTimeline:
    def __init__(self) -> None:
        self.marks: dict[str, float] = {}
        self.facts: dict[str, str] = {}
        self._reported = False

    def mark(self, stage: str, at: float | None = None) -> None:
        """Record the first time ``stage`` happened; later marks of the same stage are ignored."""
        if stage not in STAGES:
            raise ValueError(f"unknown turn stage {stage!r}")
        self.marks.setdefault(stage, time.monotonic() if at is None else at)

    def line(self) -> str:
        origin = min(self.marks.values(), default=0.0)
        stages = [f"{stage}={self.marks[stage] - origin:.3f}" for stage in STAGES if stage in self.marks]
        return " ".join(stages + [f"{key}={value}" for key, value in self.facts.items()])

    def report(self) -> None:
        """Log the timeline once, when the answer starts to play or the turn ends without one."""
        if not self._reported and self.marks:
            self._reported = True
            log.info("Reachy turn timeline %s", self.line())
