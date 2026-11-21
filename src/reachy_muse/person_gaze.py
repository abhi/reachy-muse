# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Local gaze accompaniment; never owns conversation input or reply audio."""

import asyncio
import logging
import math
import time

from reachy_muse.device_command import gaze_requested

log = logging.getLogger(__name__)
_SNAPSHOT_MAX_AGE_S = 2.0
_FRESH_PASS_S = .3
_REMEMBERED_S = 30.0
_SPEAKER_ARC = math.radians(20)


class PersonGaze:
    def __init__(self, hardware):
        self.hardware = hardware
        self._generation = 0
        self._admitted_generation = 0
        self._holding = False

    @property
    def holding(self) -> bool:
        """A gaze this conversation started is still shown; a command or wake turn can end it first."""
        return self._holding and self.hardware.person_gaze_held

    def begin(self) -> int:
        self._generation += 1
        return self._generation

    def release(self) -> None:
        self.admit(self.begin())

    def admit(self, generation: int) -> bool:
        if generation <= self._admitted_generation:
            return generation == self._admitted_generation
        self._admitted_generation = generation
        if self._holding:
            self._holding = False
            self.hardware.release_person_gaze()
        return True

    async def request(self, text: str | None, speaker_bearing: float | None, generation: int) -> None:
        """Face the person who is not speaking, preferring faces in view to faces seen recently.

        A remembered bearing that the current view covered is dropped: the camera just looked there.
        """
        current = self.admit(generation)
        if not text or not text.strip():
            self._log("no_transcript")
            return
        if not gaze_requested(text):
            return
        if not current:
            self._log("busy")
            return
        if speaker_bearing is None:
            self._log("no_speaker_bearing")
            return
        cached = self.hardware.face_snapshot()
        view, source = cached, "snapshot"
        if not _recent(view):
            view, source = await self._fresh_pass(), "fresh"
        if generation != self._admitted_generation:
            self._log("busy")
            return
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
                self._log("no_other" if faces else "no_faces", source, len(faces))
            else:
                self._log("stale" if cached is not None else "no_faces")
            return
        target = min(others, key=lambda face: abs(face.yaw - self.hardware.head_yaw))
        turned = self.hardware.look_at_person(target.yaw, target.pitch)
        if turned:
            self._holding = True
        self._log("turned" if turned else "busy", source, len(faces))

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
