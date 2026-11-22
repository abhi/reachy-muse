# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Local gaze accompaniment, and what Muse may learn about the people Reachy hears and sees.

Never owns conversation input or reply audio.
"""

import asyncio
from dataclasses import dataclass
import logging
import math
import time
from typing import Callable

from reachy_muse.device_command import gaze_requested

log = logging.getLogger(__name__)
_SNAPSHOT_MAX_AGE_S = 2.0
_FRESH_PASS_S = .3
_REMEMBERED_S = 30.0
_SPEAKER_ARC = math.radians(20)
_VISIBLE_MAX_AGE_S = 10.0
# The robot's longest end-of-turn silence: an utterance becomes a request within it once its words end.
_HEARD_OUT_S = 3.0
_SPEAKERS = {"owner": "owner", "other": "someone_else"}


@dataclass(frozen=True)
class _Request:
    """What Muse's people commands may use of one queued request."""

    bearing: float | None          # the speaker's bearing, latched at the endpoint
    facts: dict[str, str]          # its timeline facts, where the companion's speaker label lands, maybe later


def people_result(speaker: str, snapshot) -> dict:
    """The reachy.people payload: ``speaker`` and the faces of a snapshot at most about 10 seconds old."""
    age = time.monotonic() - snapshot.captured_at if snapshot is not None else math.inf
    if not 0 <= age <= _VISIBLE_MAX_AGE_S:
        return {"speaker": speaker, "people_visible": None, "seen_s_ago": None}
    return {"speaker": speaker, "people_visible": len(snapshot.faces), "seen_s_ago": round(age)}


class PersonGaze:
    """Gaze tokens: one begins with each utterance, and admitting a newer one supersedes the rest.

    Muse's people commands answer about the latest request Reachy admitted. They wait a little for an
    admitted utterance still being heard, which may become that request; one that ends without a request,
    such as a cough, leaves the previous request standing. A release, such as when the listening window
    closes, drops every request.
    """

    def __init__(self, hardware, hearing: Callable[[], bool] = lambda: False):
        """``hearing`` tells whether an utterance is still being heard or turned into a request."""
        self.hardware = hardware
        self._hearing = hearing
        self._generation = 0
        self._admitted_generation = 0
        self._released_generation = 0
        self._holding = False
        self._facing = False           # Muse turned Reachy toward the speaker since the latest admission
        self._choosing = False         # a turn toward someone else waits for its camera pass
        self._admitted: set[int] = set()
        self._requests: dict[int, _Request] = {}

    @property
    def holding(self) -> bool:
        """A look this conversation chose still stands, so Reachy's own closing must not end it: a hold on
        someone else until a command or wake turn ends it, or the speaker Muse turned Reachy toward, until the
        next request begins."""
        return self._holding and self.hardware.person_gaze_held or self._facing

    def begin(self) -> int:
        self._generation += 1
        return self._generation

    def release(self) -> None:
        self._released_generation = self.begin()
        self.admit(self._released_generation)

    def queued(self, generation: int, bearing: float | None, facts: dict[str, str]) -> None:
        if generation > self._released_generation:
            self._requests[generation] = _Request(bearing, facts)
            self._forget_older()

    def admit(self, generation: int) -> bool:
        if generation <= self._admitted_generation:
            return generation == self._admitted_generation
        self._admitted_generation = generation
        self._admitted.add(generation)
        self._facing = False
        self._forget_older()
        if self._holding:
            self._holding = False
            self.hardware.release_person_gaze()
        return True

    async def people(self, deadline: float) -> dict:
        request = await self._heard_out(deadline)
        label = request.facts.get("speaker") if request is not None else None
        return people_result(_SPEAKERS.get(label, "unknown"), self.hardware.face_snapshot())

    async def look_at(self, target: str, deadline: float) -> str:
        """Face the latest admitted request's speaker, or someone else as a request to greet them does.

        As with Muse's other motion commands, a person turn that is still being chosen, queued or moving
        makes it busy.
        """
        request = await self._heard_out(deadline)
        if request is None or request.bearing is None:
            return "no_speaker_bearing"
        if self._choosing or self.hardware.person_turning:
            return "busy"
        generation = self._admitted_generation
        if target == "other":
            outcome, _, _ = await self._look_at_other(request.bearing, generation)
            return outcome
        if not self.hardware.face_speaker(request.bearing):
            return "busy"
        self._holding, self._facing = False, True
        return "turned"

    async def _heard_out(self, deadline: float) -> _Request | None:
        """The latest admitted request, once an admitted utterance still being heard has become one, or by
        ``deadline``; None if there is none, or the utterance is still heard by then."""
        until = min(time.monotonic() + _HEARD_OUT_S, deadline)
        while self._undecided() and time.monotonic() < until:
            await asyncio.sleep(.05)
        newest = self._admitted_generation
        if newest in self._requests:
            return self._requests[newest]
        if self._undecided() or newest == self._released_generation:
            return None
        latest = self._latest()
        return self._requests[latest] if latest is not None else None

    def _undecided(self) -> bool:
        newest = self._admitted_generation
        return newest not in self._requests and newest != self._released_generation and self._hearing()

    def _latest(self) -> int | None:
        admitted = self._admitted.intersection(self._requests)
        return max(admitted) if admitted else None

    def _forget_older(self) -> None:
        """Forget requests and tokens older than the last release or the latest admitted request; no answer
        reads them again."""
        oldest = max(self._released_generation, self._latest() or 0)
        self._requests = {key: request for key, request in self._requests.items() if key >= oldest}
        self._admitted = {key for key in self._admitted if key >= oldest}

    async def request(self, text: str | None, speaker_bearing: float | None, generation: int) -> None:
        """Face the person who is not speaking when the request asks Reachy to greet or look at them."""
        current = self.admit(generation)
        if not text or not text.strip():
            self._log("no_transcript")
            return
        if not gaze_requested(text):
            return
        if not current:
            self._log("busy")
            return
        self._log(*await self._look_at_other(speaker_bearing, generation))

    async def _look_at_other(self, speaker_bearing: float | None, generation: int) -> tuple[str, str, int]:
        """Face the person who is not speaking, preferring faces in view to faces seen recently.

        Returns the outcome, the source of the faces used and how many were in view. A remembered
        bearing that the current view covered is dropped: the camera just looked there.
        """
        if speaker_bearing is None:
            return "no_speaker_bearing", "none", 0
        cached = self.hardware.face_snapshot()
        view, source = cached, "snapshot"
        if not _recent(view):
            if self._choosing:
                return "busy", "none", 0
            self._choosing = True
            try:
                view, source = await self._fresh_pass(), "fresh"
            finally:
                self._choosing = False
        if generation != self._admitted_generation:
            return "busy", "none", 0
        if not _recent(view):
            view = None
        faces = view.faces if view is not None else ()
        speaker = math.pi / 2 - speaker_bearing
        others = _away_from(speaker, faces)
        if not others:
            remembered = self.hardware.recent_faces(time.monotonic() - _REMEMBERED_S)
            if view is not None:
                remembered = [face for face in remembered if not view.covers(face)]
            remembered = _away_from(speaker, remembered)
            if remembered:
                others, source = remembered, "memory"
        if not others:
            if view is not None:
                return "no_other" if faces else "no_faces", source, len(faces)
            return "stale" if cached is not None else "no_faces", "none", 0
        target = min(others, key=lambda face: abs(face.yaw - self.hardware.head_yaw))
        turned = self.hardware.look_at_person(target.yaw, target.pitch)
        if turned:
            self._holding = True
        return "turned" if turned else "busy", source, len(faces)

    async def _fresh_pass(self):
        try:
            return await asyncio.wait_for(asyncio.to_thread(self.hardware.refresh_faces, _FRESH_PASS_S),
                                          _FRESH_PASS_S)
        except asyncio.TimeoutError:
            return None

    @staticmethod
    def _log(outcome: str, source: str = "none", faces: int = 0) -> None:
        log.info("Reachy gaze request outcome=%s source=%s faces=%d", outcome, source, faces)


def _recent(snapshot) -> bool:
    return snapshot is not None and 0 <= time.monotonic() - snapshot.captured_at <= _SNAPSHOT_MAX_AGE_S


def _away_from(speaker_yaw: float, faces) -> list:
    return [face for face in faces if abs(face.yaw - speaker_yaw) > _SPEAKER_ARC]
