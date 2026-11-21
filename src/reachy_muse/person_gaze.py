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

    def release(self) -> None:
        self._generation += 1
        release = getattr(self.hardware, "release_person_gaze", None)
        if release is not None:
            release()

    async def request(self, text: str | None, speaker_bearing: float | None) -> None:
        self.release()
        if not text or not text.strip():
            self._log("no_transcript", 0)
            return
        if not gaze_requested(text):
            return
        if speaker_bearing is None:
            self._log("no_speaker_bearing", 0)
            return
        generation = self._generation
        read = getattr(self.hardware, "face_snapshot", None)
        snapshot = read() if read is not None else None
        stale = snapshot is not None
        if snapshot is None or not 0 <= time.monotonic() - snapshot.captured_at <= 2.0:
            refresh = getattr(self.hardware, "refresh_faces", None)
            try:
                snapshot = await asyncio.wait_for(asyncio.to_thread(refresh, .3), .3) if refresh is not None else None
            except asyncio.TimeoutError:
                snapshot = None
        if generation != self._generation:
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
        self._log("turned" if turned else "busy", len(faces))

    @staticmethod
    def _log(outcome: str, faces: int) -> None:
        log.info("Reachy gaze request outcome=%s faces=%d", outcome, faces)
