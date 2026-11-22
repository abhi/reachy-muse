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
_HEARD_KEPT = 16
_SPEAKERS = {"owner": "owner", "other": "someone_else"}


@dataclass(frozen=True)
class MuseActivity:
    """What Muse did in a realtime call, which tells which requests a command of its is about."""

    starts: tuple[float, ...] = ()      # when each of its responses began, oldest first
    open: bool = False                  # its latest response has not finished
    working_since: float | None = None  # when the work it is busy with began


@dataclass
class _Heard:
    """One utterance, as Muse's people commands see it."""

    bearing_so_far: Callable[[], float | None]
    admitted_at: float | None = None
    request: bool = False               # it ended as a request
    bearing: float | None = None        # its speaker's bearing, latched at that end
    facts: dict[str, str] | None = None  # its timeline facts once queued, where its speaker label lands

    @property
    def label(self) -> str | None:
        return self.facts.get("speaker") if self.facts is not None else None

    def speaker_bearing(self) -> float | None:
        return self.bearing if self.request else self.bearing_so_far()


def people_result(speaker: str, snapshot) -> dict:
    """The reachy.people payload: ``speaker`` and the faces of a snapshot at most about 10 seconds old."""
    age = time.monotonic() - snapshot.captured_at if snapshot is not None else math.inf
    if not 0 <= age <= _VISIBLE_MAX_AGE_S:
        return {"speaker": speaker, "people_visible": None, "seen_s_ago": None}
    return {"speaker": speaker, "people_visible": len(snapshot.faces), "seen_s_ago": round(age)}


class PersonGaze:
    """Gaze tokens: one begins with each utterance, and admitting a newer one supersedes the rest.

    Muse's people commands answer about the requests Muse is answering when it asks: in a realtime call, the
    ones its current response or work is about, and in chat, the request it was sent. An utterance still being
    heard may become such a request; one that ended without becoming a request, such as a cough, or that
    Reachy admitted after Muse asked, never counts. A release, such as when the listening window closes,
    forgets them all.
    """

    def __init__(self, hardware, hearing: Callable[[], int | None]):
        """``hearing`` gives the token of the utterance Reachy is hearing now, if any."""
        self.hardware = hardware
        self._hearing = hearing
        self._generation = 0
        self._admitted_generation = 0
        self._holding = False
        self._facing = False           # Muse turned Reachy toward the speaker since the latest admission
        self._choosing = False         # a turn toward someone else waits for its camera pass
        self._heard: dict[int, _Heard] = {}

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
        self._heard.clear()
        self.admit(self.begin())

    def opened(self, generation: int, bearing_so_far: Callable[[], float | None]) -> None:
        """An utterance began; ``bearing_so_far`` gives its speaker's bearing from the speech heard so far."""
        self._heard[generation] = _Heard(bearing_so_far)
        while len(self._heard) > _HEARD_KEPT:
            del self._heard[min(self._heard)]

    def requested(self, generation: int, bearing: float | None) -> None:
        heard = self._heard.get(generation)
        if heard is not None:
            heard.request, heard.bearing = True, bearing

    def queued(self, generation: int, facts: dict[str, str]) -> None:
        heard = self._heard.get(generation)
        if heard is not None:
            heard.facts = facts

    def admit(self, generation: int) -> bool:
        if generation <= self._admitted_generation:
            return generation == self._admitted_generation
        self._admitted_generation = generation
        heard = self._heard.get(generation)
        if heard is not None:
            heard.admitted_at = time.monotonic()
        self._facing = False
        if self._holding:
            self._holding = False
            self.hardware.release_person_gaze()
        return True

    async def people(self, deadline: float, activity: MuseActivity | None, chat: dict[str, str] | None) -> dict:
        """Who made the requests Muse is answering, once one still being heard has ended, and the faces Reachy
        saw last.

        Their known labels must agree; a request labeled unknown, or not labeled, counts for nothing.
        """
        asked_at = time.monotonic()
        until = min(asked_at + _HEARD_OUT_S, deadline)
        requests = self._answering(asked_at, activity, chat)
        while not all(heard.request for heard in requests) and time.monotonic() < until:
            await asyncio.sleep(.05)
            requests = self._answering(asked_at, activity, chat)
        labels = {heard.label for heard in requests}.intersection(_SPEAKERS)
        return people_result(_SPEAKERS[labels.pop()] if len(labels) == 1 else "unknown",
                             self.hardware.face_snapshot())

    async def look_at(self, target: str, deadline: float, activity: MuseActivity | None,
                      chat: dict[str, str] | None) -> str:
        """Face the speaker of the latest request Muse is answering, or someone else as a request to greet them
        does.

        As with Muse's other motion commands, a person turn that is still being chosen, queued or moving
        makes it busy.
        """
        requests = self._answering(time.monotonic(), activity, chat)
        bearing = requests[-1].speaker_bearing() if requests else None
        if bearing is None:
            return "no_speaker_bearing"
        if self._choosing or self.hardware.person_turning:
            return "busy"
        generation = self._admitted_generation
        if target == "other":
            outcome, _, _ = await self._look_at_other(bearing, generation)
            return outcome
        if not self.hardware.face_speaker(bearing):
            return "busy"
        self._holding, self._facing = False, True
        return "turned"

    def _answering(self, asked_at: float, activity: MuseActivity | None,
                   chat: dict[str, str] | None) -> list[_Heard]:
        """The requests Muse is answering, oldest first: the chat request whose timeline facts are ``chat``, or
        the call's requests by what Muse did in it, among those admitted by ``asked_at``."""
        if chat is not None:
            return [heard for heard in self._heard.values() if heard.facts is chat] or [
                _Heard(lambda: None, request=True, facts=chat)]
        if activity is None:
            return []
        hearing = self._hearing()
        return _answered([heard for generation, heard in sorted(self._heard.items())
                          if heard.admitted_at is not None and heard.admitted_at <= asked_at
                          and (heard.request or generation == hearing)], activity, asked_at)

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


def _answered(requests: list[_Heard], activity: MuseActivity, asked_at: float) -> list[_Heard]:
    """Which of ``requests``, oldest first, Muse answers when it asks at ``asked_at``.

    An open response made the command, so it answers the requests before it. Otherwise Muse answers the
    requests since its latest response, if any came, and else the ones that response answered. Work Muse began
    after its latest response answers the requests before the work and also those said during it, since
    Working cannot tell an aside from part of the request.
    """
    starts = [start for start in activity.starts if start <= asked_at]
    latest = starts[-1] if starts else -math.inf
    working_since = activity.working_since
    if working_since is not None and working_since <= latest:
        working_since = None
    since = [heard for heard in requests if heard.admitted_at > latest]
    if activity.open or not since and working_since is None:
        return _answered_by(requests, starts)
    if working_since is not None:
        before = [heard for heard in since if heard.admitted_at <= working_since]
        return (before or _answered_by(requests, starts)) + [heard for heard in since
                                                              if heard.admitted_at > working_since]
    return since


def _answered_by(requests: list[_Heard], starts: list[float]) -> list[_Heard]:
    """The requests the latest of ``starts`` answers: those since the response before it, or, when a response
    only went on with an earlier answer, since an earlier one."""
    if not starts:
        return []
    for since in [*reversed(starts[:-1]), -math.inf]:
        answered = [heard for heard in requests if since < heard.admitted_at <= starts[-1]]
        if answered:
            return answered
    return []


def _away_from(speaker_yaw: float, faces) -> list:
    return [face for face in faces if abs(face.yaw - speaker_yaw) > _SPEAKER_ARC]
