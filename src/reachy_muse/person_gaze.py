# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Local gaze accompaniment, and what Muse may learn about the people Reachy hears and sees.

Never owns conversation input or reply audio.
"""

import asyncio
import contextlib
from dataclasses import dataclass, replace
import logging
import math
import time
from typing import AsyncIterator, Callable

from reachy_muse.device_command import PhotoSubject, gaze_requested
from reachy_muse.local_face_tracking import ObservedFace
from reachy_muse.photo import PhotoAim
from reachy_muse.speaker_direction import DirectionDecision

log = logging.getLogger(__name__)
# Voiced speech Reachy hears of an utterance before it turns toward the speaker: about two DoA readings, so the
# bearing is steady, while the speaker is still talking.
_FOLLOW_AFTER_S = .5
_SNAPSHOT_MAX_AGE_S = 2.0
_FRESH_PASS_S = .3
_REMEMBERED_S = 30.0
_SPEAKER_ARC = math.radians(20)
_VISIBLE_MAX_AGE_S = 10.0
_LABEL_WAIT_S = 1.0
_HANDOFF_S = .1                       # left before Muse's deadline for an answer to reach the SDK
_HEARD_KEPT = 16
_SPEAKERS = {"owner": "owner", "other": "someone_else"}
# A photo's subject is centred within _AIM_TOLERANCE of the camera's axis, and found again within _AIM_FOUND of yaw.
_AIM_TOLERANCE = math.radians(6)
_AIM_FOUND = math.radians(15)
_AIM_WAIT_S = .75          # for a turn or move under way to end
_AIM_TURN_S = 1.15         # for the aim's turn, with a fade first
_AIM_CORRECT_S = .85       # left for one correction
_AIM_STILL_S = .2          # still after the turn before the confirming pass, for the camera's lag behind the pose
_AIM_RETRY_S = .05
_AIM_BUSY_MARGIN_S = 1.45  # left when Reachy stops retrying a turn other motion refuses
_GROUP_SHARE = .8          # of the camera's view that a group may span
# Within a degree of POSE_LIMITS' 35 degrees of yaw and 20 of pitch, the head turns no further.
_YAW_REACH = math.radians(34)
_PITCH_REACH = math.radians(19)
_KEEPS_VIEW = frozenset((PhotoSubject.SCENE, PhotoSubject.UNPARSED))
_LEVEL = frozenset((PhotoSubject.HELD, PhotoSubject.THIS))


@dataclass(frozen=True)
class MuseActivity:
    """What Muse did in a realtime call, which tells which requests a command of its is about."""

    starts: tuple[float, ...] = ()      # when each of its responses began, oldest first
    open: bool = False                  # its latest response has not finished
    silent: bool = False                # its latest response finished without audio, as one calling a tool does
    working_since: float | None = None  # when the work it is busy with began


@dataclass
class _Heard:
    """One utterance, as Muse's people commands see it."""

    speaker: object | None              # the companion's label slot for its voice, if the link labels voices
    bearing_so_far: Callable[[], float | None]
    admitted_at: float | None = None
    request: bool = False               # it ended as a request
    bearing: float | None = None        # its speaker's bearing, latched at that end
    facts: dict[str, str] | None = None  # its timeline facts once queued, where its speaker label lands too
    followed: bool = False              # Reachy decided whether to turn toward its speaker

    @property
    def label(self) -> str | None:
        if self.speaker is not None and self.speaker.label is not None:
            return self.speaker.label.value
        return self.facts.get("speaker") if self.facts is not None else None

    @property
    def awaiting_label(self) -> bool:
        return self.speaker is not None and self.speaker.label is None

    def speaker_bearing(self) -> float | None:
        return self.bearing if self.request else self.bearing_so_far()


@dataclass(frozen=True)
class _AimPoint:
    """Where a photo's subject is, in world radians, with yaw positive to the left and pitch positive below."""

    yaw: float
    pitch: float | None            # None frames it level, with the head's pitch held at 0
    source: str                    # fresh, memory or bearing
    members: tuple[tuple[float, float | None], ...] = ()  # a group's (yaw, pitch), with no pitch for the bearing


def people_result(speaker: str, snapshot) -> dict:
    """The reachy.people payload: ``speaker`` and the faces of a snapshot at most about 10 seconds old."""
    if not _within(snapshot, _VISIBLE_MAX_AGE_S):
        return {"speaker": speaker, "people_visible": None, "seen_s_ago": None}
    return {"speaker": speaker, "people_visible": len(snapshot.faces),
            "seen_s_ago": round(time.monotonic() - snapshot.captured_at)}


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
        self._aiming = False           # a photo's aim holds the camera, from its start until its frame is read
        self._aim_turned = False       # the aim made a person turn, which its end releases
        self._looking: asyncio.Future | None = None   # the camera pass in flight
        self._heard: dict[int, _Heard] = {}

    @property
    def holding(self) -> bool:
        """A look this conversation chose still stands, so Reachy's own closing must not end it: a photo's aim
        until its frame is read, a hold on someone else until a command or wake turn ends it, or the speaker Muse
        turned Reachy toward, until the next request begins."""
        return self._aiming or self._holding and self.hardware.person_gaze_held or self._facing

    def begin(self) -> int:
        self._generation += 1
        return self._generation

    def release(self) -> None:
        """Forget every utterance, and turn back from someone else Reachy was asked to face."""
        self._heard.clear()
        self._admit(self.begin(), turn_back=True)

    def opened(self, generation: int, speaker, bearing_so_far: Callable[[], float | None]) -> None:
        """An utterance began: ``speaker`` is the companion's label slot for its voice, if the link labels voices,
        and ``bearing_so_far`` gives its speaker's bearing from the speech heard so far."""
        self._heard[generation] = _Heard(speaker, bearing_so_far)
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
        """Take an utterance as the user's speech. A look on someone else ends where Reachy faces them, for the
        turn toward whoever speaks now to take over."""
        return self._admit(generation, turn_back=False)

    def _admit(self, generation: int, *, turn_back: bool) -> bool:
        if generation <= self._admitted_generation:
            return generation == self._admitted_generation
        self._admitted_generation = generation
        heard = self._heard.get(generation)
        if heard is not None:
            heard.admitted_at = time.monotonic()
        self._facing = False
        if self._holding:
            self._holding = False
            self.hardware.release_person_gaze(turn_back=turn_back)
        return True

    def follow(self, generation: int, voiced_s: float, direction: Callable[[], DirectionDecision]) -> None:
        """Turn toward whoever is speaking in the utterance ``generation`` once Reachy has admitted it, decided once
        per utterance: after ``_FOLLOW_AFTER_S`` of its voiced speech with a steady bearing, or at its end if it
        ends as a request first. ``direction`` decides that bearing from the speech heard so far.

        While the user has switched face tracking off, a person turn is chosen or holds Reachy, a photo's aim holds
        the camera, a turn Muse asked for since Reachy admitted the utterance stands, other motion owns the head, or
        Reachy turned toward a later speaker moments ago, it waits as long as the utterance goes on, and never turns
        for it later. Each decision logs one line.
        """
        heard = self._heard.get(generation)
        if (heard is None or heard.followed or generation != self._admitted_generation
                or voiced_s < _FOLLOW_AFTER_S and not heard.request):
            return
        angle = direction().angle
        if angle is None:
            outcome, delta = "unknown", None
        elif self._choosing or self._aiming:
            outcome, delta = "held", None
        elif self._facing:
            outcome, delta = "busy", None
        else:
            outcome, delta = self.hardware.follow_speaker(angle)
        if outcome not in ("turned", "same") and not heard.request:
            return
        heard.followed = True
        log.info("Reachy follows speaker outcome=%s delta_deg=%s", outcome, "none" if delta is None else round(delta))

    async def people(self, deadline: float, activity: MuseActivity | None, chat: dict[str, str] | None) -> dict:
        """Who made the requests Muse is answering, waiting up to a second for the companion's labels of them,
        and the faces of Reachy's latest camera look; when that is over about 10 seconds old, Reachy looks again
        meanwhile, within the same second.

        Their labels must agree. A request the companion labeled unknown, such as a short tail split off another,
        counts for nothing, but one with no label by then makes the speaker unknown.
        """
        asked_at = time.monotonic()
        until = min(asked_at + _LABEL_WAIT_S, deadline - _HANDOFF_S)
        snapshot = self.hardware.face_snapshot()
        looking = (None if _within(snapshot, _VISIBLE_MAX_AGE_S)
                   else asyncio.ensure_future(self._fresh_pass(min(_FRESH_PASS_S, until - asked_at))))
        try:
            requests = self._answering(asked_at, activity, chat)
            while any(heard.awaiting_label for heard in requests) and time.monotonic() < until:
                await asyncio.sleep(.05)
                requests = self._answering(asked_at, activity, chat)
            if looking is not None:
                snapshot = await looking or snapshot
        finally:
            if looking is not None:
                looking.cancel()
        labels = [heard.label for heard in requests]
        known = set(labels).intersection(_SPEAKERS)
        return people_result(_SPEAKERS[known.pop()] if len(known) == 1 and None not in labels else "unknown",
                             snapshot)

    async def look_at(self, target: str, deadline: float, activity: MuseActivity | None,
                      chat: dict[str, str] | None) -> str:
        """Face the speaker of the latest request Muse is answering, or someone else as a request to greet them
        does, with any camera pass done by ``deadline``.

        As with Muse's other motion commands, sleep, motion turned off, a photo's aim, or a person turn that is
        still being chosen, queued or moving makes it busy.
        """
        if self._aiming:
            return "busy"
        requests = self._answering(time.monotonic(), activity, chat)
        bearing = requests[-1].speaker_bearing() if requests else None
        if bearing is None:
            return "no_speaker_bearing"
        if self._choosing or self.hardware.person_turning:
            return "busy"
        generation = self._admitted_generation
        if target == "other":
            outcome, _, _ = await self._look_at_other(bearing, generation,
                                                      deadline - _HANDOFF_S - time.monotonic())
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
                _Heard(None, lambda: None, request=True, facts=chat)]
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

    async def _look_at_other(self, speaker_bearing: float | None, generation: int,
                             within_s: float = _FRESH_PASS_S) -> tuple[str, str, int]:
        """Face the person who is not speaking, preferring faces in view to faces seen recently; a camera pass
        takes at most ``within_s``.

        Returns the outcome, the source of the faces used and how many were in view. A remembered
        bearing that the current view covered is dropped: the camera just looked there. A photo's aim makes it busy.
        """
        if self._aiming:
            return "busy", "none", 0
        if speaker_bearing is None:
            return "no_speaker_bearing", "none", 0
        cached = self.hardware.face_snapshot()
        view, source = cached, "snapshot"
        if not _within(view, _SNAPSHOT_MAX_AGE_S):
            if self._choosing:
                return "busy", "none", 0
            self._choosing = True
            try:
                view, source = await self._fresh_pass(min(_FRESH_PASS_S, within_s)), "fresh"
            finally:
                self._choosing = False
        if generation != self._admitted_generation:
            return "busy", "none", 0
        if not _within(view, _SNAPSHOT_MAX_AGE_S):
            view = None
        faces = view.faces if view is not None else ()
        target, found = _choose_other(math.pi / 2 - speaker_bearing, faces, self._remembered(view),
                                      self.hardware.head_yaw)
        if target is None:
            if view is not None:
                return "no_other" if faces else "no_faces", source, len(faces)
            return "stale" if cached is not None else "no_faces", "none", 0
        if found == "memory":
            source = found
        turned = self.hardware.look_at_person(target.yaw, target.pitch)
        if turned:
            self._holding = True
        return "turned" if turned else "busy", source, len(faces)

    async def _fresh_pass(self, within_s: float):
        """A camera pass, waited for at most ``within_s``, or None. The face worker takes one pass at a time, so
        commands that need one meanwhile share the pass in flight."""
        if within_s <= 0:
            return None
        if self._looking is None or self._looking.done():
            self._looking = asyncio.ensure_future(asyncio.to_thread(self.hardware.refresh_faces, within_s))
        try:
            return await asyncio.wait_for(asyncio.shield(self._looking), within_s)
        except asyncio.TimeoutError:
            return None

    def _remembered(self, view) -> list:
        """Faces seen in the last 30 seconds, except those ``view`` saw and any where it looked, since the camera
        just looked there."""
        faces = self.hardware.recent_faces(time.monotonic() - _REMEMBERED_S)
        return [face for face in faces if view is None or face not in view.faces and not view.covers(face)]

    @contextlib.asynccontextmanager
    async def photo_aim(self, subject: PhotoSubject, speaker_bearing: float | None, deadline: float, *,
                        stop: Callable[[], bool] = lambda: False) -> AsyncIterator[PhotoAim]:
        """Point the camera at the photo's subject and keep it there until the block ends; on exit Reachy keeps
        facing it, with nothing turning back.

        Reachy finds the subject with its own camera passes, which the speaker's DoA ``speaker_bearing`` only
        guides, turns with the held person turn, and checks with a second pass, all by ``deadline``; ``stop``
        cancels the aim at its next step. Until the block ends nothing else turns Reachy.
        """
        started = time.monotonic()
        hold = getattr(self.hardware, "hold_view_for_photo", None)
        self._aiming = True
        if hold is not None:
            hold(True)
        try:
            speaker_yaw = None if speaker_bearing is None else math.pi / 2 - speaker_bearing
            aim = await self._aim(subject, speaker_yaw, deadline, stop)
            yield replace(aim, bearing=speaker_bearing is not None,
                          aim_ms=round((time.monotonic() - started) * 1000))
        finally:
            if self._aim_turned:
                self.hardware.release_person_gaze(turn_back=False)
            self._aiming = self._aim_turned = False
            if hold is not None:
                hold(False)

    async def _aim(self, subject: PhotoSubject, speaker_yaw: float | None, deadline: float,
                   stop: Callable[[], bool]) -> PhotoAim:
        """Wait out a view change, look for the subject, and turn toward it unless it is already centred."""
        hardware = self.hardware
        if subject in _KEEPS_VIEW or subject is PhotoSubject.THIS and speaker_yaw is None:
            return PhotoAim(subject)
        if not getattr(hardware, "motion_enabled", True) or not callable(getattr(hardware, "look_at_person", None)):
            return PhotoAim(subject, "off")
        await self._until(lambda: not hardware.view_changing, min(deadline, time.monotonic() + _AIM_WAIT_S), stop)
        if stop():
            return PhotoAim(subject, "cancelled")
        view = await self._fresh_pass(min(_FRESH_PASS_S, deadline - time.monotonic()))
        snapshot = view if view is not None else hardware.face_snapshot()
        point = _aim_point(subject, speaker_yaw, view, self._remembered(view), hardware.head_yaw,
                           snapshot.half_view if snapshot is not None else (0.0, 0.0))
        aim = PhotoAim(subject, faces=None if view is None else len(view.faces))
        if isinstance(point, str):
            return replace(aim, outcome=point)
        aim = replace(aim, source=point.source)
        if point.source == "fresh":
            centred = _centred(point, view)
            if centred is not None:
                return replace(aim, outcome=centred, off_deg=_off_deg(point, view))
        if stop():
            return replace(aim, outcome="cancelled")
        if not await self._turn_for_photo(point, deadline, stop):
            return replace(aim, outcome="cancelled" if stop() else "busy")
        return await self._confirm(replace(aim, turns=1), point, deadline, stop)

    async def _confirm(self, aim: PhotoAim, point: _AimPoint, deadline: float,
                       stop: Callable[[], bool]) -> PhotoAim:
        """Look for the subject again with the head still after its turn, and turn once more if it is off-centre."""
        if stop():
            return replace(aim, outcome="cancelled")
        settled = time.monotonic()
        if settled >= deadline:
            return replace(aim, outcome="unlocked")
        view = await self._fresh_pass(min(_FRESH_PASS_S, deadline - settled))
        if view is not None and view.captured_at < settled:
            view = await self._fresh_pass(min(_FRESH_PASS_S, deadline - time.monotonic()))
        if view is None or view.captured_at < settled:
            return replace(aim, outcome="bearing" if point.source == "bearing" else "unlocked")
        found = _refind(point, aim.subject, view)
        if found is None:
            if point.source == "bearing":
                return replace(aim, outcome="bearing", off_deg=_off_deg(point, view))
            return replace(aim, outcome="lost")
        aim = replace(aim, off_deg=_off_deg(found, view))
        centred = _centred(found, view)
        if centred is not None:
            return replace(aim, outcome=centred)
        if stop():
            return replace(aim, outcome="cancelled")
        hardware = self.hardware
        if deadline - time.monotonic() < _AIM_CORRECT_S or not hardware.look_at_person(
                found.yaw, 0.0 if found.pitch is None else found.pitch):
            return replace(aim, outcome="unlocked")
        await self._until(lambda: not hardware.person_turning, time.monotonic() + _AIM_WAIT_S, stop)
        return replace(aim, outcome="corrected", turns=2)

    async def _turn_for_photo(self, point: _AimPoint, deadline: float, stop: Callable[[], bool]) -> bool:
        """Turn toward ``point`` with the held person turn, retrying while other motion refuses it, then wait until
        the turn ends and the head has kept still; False when Reachy gave up first or ``stop`` was set."""
        hardware = self.hardware
        while not hardware.look_at_person(point.yaw, 0.0 if point.pitch is None else point.pitch):
            if stop() or time.monotonic() >= deadline - _AIM_BUSY_MARGIN_S:
                return False
            await asyncio.sleep(_AIM_RETRY_S)
        self._aim_turned = True
        await self._until(lambda: not hardware.person_turning, min(deadline, time.monotonic() + _AIM_TURN_S), stop)
        if stop():
            return False
        await asyncio.sleep(max(0.0, min(_AIM_STILL_S, deadline - time.monotonic())))
        return True

    @staticmethod
    async def _until(ready: Callable[[], bool], until: float, stop: Callable[[], bool]) -> None:
        """Wait until ``ready`` holds, checking every 50 ms, unless ``until`` passes or ``stop`` is set first."""
        while not ready() and not stop() and time.monotonic() < until:
            await asyncio.sleep(_AIM_RETRY_S)

    @staticmethod
    def _log(outcome: str, source: str = "none", faces: int = 0) -> None:
        log.info("Reachy gaze request outcome=%s source=%s faces=%d", outcome, source, faces)


def _within(snapshot, max_age_s: float) -> bool:
    return snapshot is not None and 0 <= time.monotonic() - snapshot.captured_at <= max_age_s


def _answered(requests: list[_Heard], activity: MuseActivity, asked_at: float) -> list[_Heard]:
    """Which of ``requests``, oldest first, Muse answers when it asks at ``asked_at``.

    Work Muse is busy with answers the requests since the response before the work began, or else that
    response's, and anything said during the work, which Working cannot tell from part of the request; the
    responses Muse began meanwhile, such as an acknowledgement, are part of the work. Otherwise a response that
    is still open, or that finished without audio as one calling a tool does, made the command: it answers the
    requests before it, back past the response before it, which may have been an acknowledgement. Otherwise
    Muse answers the requests since its latest response, if any came, and else the ones that response answered.
    """
    starts = [start for start in activity.starts if start <= asked_at]
    working_since = activity.working_since
    if working_since is not None:
        before = [start for start in starts if start < working_since]
        since = before[-1] if before else -math.inf
        asked = [heard for heard in requests if since < heard.admitted_at <= working_since]
        return (asked or _answered_by(requests, before, 1)) + [
            heard for heard in requests if heard.admitted_at > working_since]
    if activity.open or activity.silent:
        return _answered_by(requests, starts, 2)
    latest = starts[-1] if starts else -math.inf
    return [heard for heard in requests if heard.admitted_at > latest] or _answered_by(requests, starts, 1)


def _answered_by(requests: list[_Heard], starts: list[float], responses: int) -> list[_Heard]:
    """The requests that the latest ``responses`` of ``starts`` answer: those since the response before them,
    or, when those only went on with an earlier answer, since an earlier one."""
    if not starts:
        return []
    for since in [*reversed(starts[:-responses]), -math.inf]:
        answered = [heard for heard in requests if since < heard.admitted_at <= starts[-1]]
        if answered:
            return answered
    return []


def _away_from(speaker_yaw: float, faces) -> list:
    return [face for face in faces if abs(face.yaw - speaker_yaw) > _SPEAKER_ARC]


def _aim_point(subject: PhotoSubject, speaker_yaw: float | None, view, remembered, head_yaw: float,
               half_view: tuple[float, float]) -> _AimPoint | str:
    """Where to point the camera for ``subject``, from the first pass's ``view`` and the faces ``remembered``
    outside it, or why Reachy keeps its view: view, no_faces or no_other.

    "Me" frames the speaker's face and "held" the speaker level; "us" frames everyone the view fits, the speaker
    included; "other" frames the person who is not speaking or, without a bearing, everyone when the view fits them;
    "this" turns level only toward a speaker outside the view.
    """
    faces = view.faces if view is not None else ()
    if subject is PhotoSubject.THIS:
        if speaker_yaw is None or view is None or _in_view(speaker_yaw, view):
            return "view"
        return _AimPoint(speaker_yaw, None, "bearing")
    members = [((face.yaw, face.pitch), "fresh") for face in faces] + [
        ((face.yaw, face.pitch), "memory") for face in remembered]
    span = _GROUP_SHARE * 2 * half_view[0]
    if subject is PhotoSubject.OTHER:
        if not members:
            return "no_faces"
        if speaker_yaw is None:
            if len(members) == 1:
                return "no_other"
            if _span(members) <= span:
                return _group(members)
            speaker_yaw = _speaker_face(None, faces, remembered, head_yaw)[0].yaw
        face, source = _choose_other(speaker_yaw, faces, remembered, head_yaw)
        return "no_other" if face is None else _AimPoint(face.yaw, face.pitch, source)
    speaker, source = _speaker_face(speaker_yaw, faces, () if subject is PhotoSubject.HELD else remembered,
                                    head_yaw)
    if subject is PhotoSubject.US:
        anchor = (speaker_yaw, None) if speaker is None else (speaker.yaw, speaker.pitch)
        if speaker is None and speaker_yaw is not None:
            members.append((anchor, "bearing"))
        if len(members) > 1:
            return _group_point(members, span, anchor, head_yaw)
    if speaker is not None:
        return _AimPoint(speaker.yaw, None if subject is PhotoSubject.HELD else speaker.pitch, source)
    return "no_faces" if speaker_yaw is None else _AimPoint(speaker_yaw, None, "bearing")


def _speaker_face(speaker_yaw: float | None, faces, remembered,
                  head_yaw: float) -> tuple[ObservedFace | None, str]:
    """Whoever asked, in view before memory: the face nearest their bearing within 20 degrees or, without one, the
    largest face by box area, a tie going to the face nearest the head's yaw; and where it was found."""
    for found, source in ((faces, "fresh"), (remembered, "memory")):
        if speaker_yaw is not None:
            near = _nearest(found, speaker_yaw, _SPEAKER_ARC)
            if near is not None:
                return near, source
        elif found:
            return max(found, key=lambda face: (face.box[2] * face.box[3], -abs(face.yaw - head_yaw))), source
    return None, "none"


def _choose_other(speaker_yaw: float, faces, remembered, head_yaw: float) -> tuple[ObservedFace | None, str]:
    """Whoever is not speaking, in view before memory: of the faces more than 20 degrees from the speaker, the one
    needing the smallest head turn; and where it was found."""
    for found, source in ((faces, "fresh"), (remembered, "memory")):
        others = _away_from(speaker_yaw, found)
        if others:
            return min(others, key=lambda face: abs(face.yaw - head_yaw)), source
    return None, "none"


def _group_point(members, span: float, anchor: tuple[float, float | None], head_yaw: float) -> _AimPoint:
    """Frame the most ``members`` within ``span`` of yaw that include the speaker's ``anchor``, a tie going to the
    smaller turn. Each member is ((yaw, pitch), source)."""
    ordered = sorted(members, key=lambda member: member[0][0])
    runs = [ordered[start:end] for start in range(len(ordered)) for end in range(start + 1, len(ordered) + 1)
            if _span(ordered[start:end]) <= span and any(point == anchor for point, _ in ordered[start:end])]
    return _group(max(runs, key=lambda run: (len(run), -abs(_group(run).yaw - head_yaw))))


def _group(members) -> _AimPoint:
    """The middle of ``members``' yaws, at the mean pitch of those with a face, or level without one; it comes
    from the camera's pass if any member does. A single member is a point of its own."""
    points = tuple(point for point, _ in members)
    pitches = [pitch for _, pitch in points if pitch is not None]
    sources = {source for _, source in members}
    return _AimPoint((min(yaw for yaw, _ in points) + max(yaw for yaw, _ in points)) / 2,
                     sum(pitches) / len(pitches) if pitches else None,
                     next(source for source in ("fresh", "memory", "bearing") if source in sources),
                     points if len(points) > 1 else ())


def _span(members) -> float:
    yaws = [yaw for (yaw, _), _ in members]
    return max(yaws) - min(yaws)


def _refind(point: _AimPoint, subject: PhotoSubject, view) -> _AimPoint | None:
    """The subject in a later pass ``view``, or None when it is gone: a group's members each by the face within
    15 degrees of its yaw, keeping the speaker's bearing; the speaker by the face nearest their bearing within
    20 degrees, framed as ``subject`` asks; or else the face nearest the point within 15 degrees of yaw."""
    if point.members:
        found = []
        for yaw, pitch in point.members:
            face = None if pitch is None else _nearest(view.faces, yaw, _AIM_FOUND)
            if pitch is None or face is not None:
                found.append(((yaw, None) if face is None else (face.yaw, face.pitch), point.source))
        return _group(found) if any(pitch is not None for (_, pitch), _ in found) else None
    if point.source == "bearing":
        face = _nearest(view.faces, point.yaw, _SPEAKER_ARC)
        return None if face is None else _AimPoint(face.yaw, None if subject in _LEVEL else face.pitch, "bearing")
    face = _nearest(view.faces, point.yaw, _AIM_FOUND)
    return None if face is None else _AimPoint(face.yaw, None if point.pitch is None else face.pitch, point.source)


def _nearest(faces, yaw: float, within: float) -> ObservedFace | None:
    return min((face for face in faces if abs(face.yaw - yaw) <= within), key=lambda face: abs(face.yaw - yaw),
               default=None)


def _aim_error(point: _AimPoint, view) -> tuple[float, float | None]:
    """How far ``point`` lies from ``view``'s camera axis in yaw and, when it frames a face, in pitch."""
    return (math.remainder(point.yaw - view.heading, math.tau),
            None if point.pitch is None else point.pitch - view.elevation)


def _at_limit(point: _AimPoint, view) -> bool:
    """Every axis is within the tolerance, or the head is at its limit there with ``point`` beyond it."""
    return all(error is None or abs(error) <= _AIM_TOLERANCE or abs(axis) >= reach and error * axis > 0
               for error, axis, reach in zip(_aim_error(point, view), (view.heading, view.elevation),
                                             (_YAW_REACH, _PITCH_REACH)))


def _centred(point: _AimPoint, view) -> str | None:
    """locked when ``point`` is within the tolerance of ``view``'s axis, limit when it is as near as the head
    reaches, or None while a turn can bring it nearer."""
    if all(error is None or abs(error) <= _AIM_TOLERANCE for error in _aim_error(point, view)):
        return "locked"
    return "limit" if _at_limit(point, view) else None


def _off_deg(point: _AimPoint, view) -> int:
    return round(math.degrees(max(abs(error) for error in _aim_error(point, view) if error is not None)))


def _in_view(yaw: float, view) -> bool:
    return abs(math.remainder(yaw - view.heading, math.tau)) <= view.half_view[0]
