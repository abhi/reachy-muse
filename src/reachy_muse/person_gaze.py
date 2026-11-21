# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Local gaze accompaniment; never owns conversation input or reply audio."""

import asyncio
import logging
import math
import re
import time

log = logging.getLogger(__name__)
_TARGET = r"(?:him|her|them|my friend|someone(?: else)?|somebody(?: else)?|the other person|my guest)"
_REQUEST = re.compile(
    rf"(?:please )?(?:(?:can|could|would) you )?(?:say (?:hi|hello) to {_TARGET}|greet {_TARGET}|look at {_TARGET})"
    r"(?: please)?[.!?]*", re.IGNORECASE)


def gaze_requested(text: str) -> bool:
    return _REQUEST.fullmatch(" ".join(text.split())) is not None


class PersonGaze:
    def __init__(self, hardware):
        self.hardware = hardware
        self._generation = 0
        self._admitted_generation = 0
        self._holding = False

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

    async def request(self, text: str | None, speaker_bearing: float | None,
                      generation: int | None = None) -> None:
        generation = self.begin() if generation is None else generation
        if not self.admit(generation):
            self._log("busy", 0)
            return
        if not text or not text.strip():
            self._log("no_transcript", 0)
            return
        if not gaze_requested(text):
            return
        if speaker_bearing is None:
            self._log("no_speaker_bearing", 0)
            return
        snapshot = self.hardware.face_snapshot()
        stale = snapshot is not None
        if snapshot is None or not 0 <= time.monotonic() - snapshot.captured_at <= 2.0:
            try:
                snapshot = await asyncio.wait_for(asyncio.to_thread(self.hardware.refresh_faces, .3), .3)
            except asyncio.TimeoutError:
                snapshot = None
        if generation != self._admitted_generation:
            self._log("busy", 0)
            return
        if snapshot is None or not 0 <= time.monotonic() - snapshot.captured_at <= 2.0:
            self._log("stale" if stale or snapshot is not None else "no_faces", 0)
            return
        faces = snapshot.faces
        if not faces:
            self._log("no_faces", 0)
            return
        speaker_yaw = math.pi / 2 - speaker_bearing
        others = [face for face in faces if abs(face.yaw - speaker_yaw) > math.radians(20)]
        if not others:
            self._log("no_other", len(faces))
            return
        target = min(others, key=lambda face: abs(face.yaw - self.hardware.head_yaw))
        turned = self.hardware.look_at_person(target.yaw, target.pitch)
        if turned:
            self._holding = True
        self._log("turned" if turned else "busy", len(faces))

    @staticmethod
    def _log(outcome: str, faces: int) -> None:
        log.info("Reachy gaze request outcome=%s faces=%d", outcome, faces)
