# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""One photo for Muse, taken only when the user asks Reachy for it.

A camera frame never leaves the capture: only a JPEG of at most ``PHOTO_MAX_BYTES`` does, in memory.
Nothing is written to disk.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from reachy_muse.turn_timeline import TurnTimeline

log = logging.getLogger(__name__)

PHOTO_REQUEST_TTL_S = 30.0
PHOTO_CAPTURE_S = 2.0
PHOTO_SEND_TIMEOUT_S = 20.0
PHOTO_MIME_TYPE = "image/jpeg"
PHOTO_FILENAME = "reachy-photo.jpg"
PHOTO_MAX_BYTES = 150 * 1024
# (widest size, quality), tried in order until a JPEG fits PHOTO_MAX_BYTES.
JPEG_LADDER = ((1280, 80), (1280, 70), (1280, 60), (960, 70), (960, 55), (640, 60))
JPEG_MAGIC = b"\xff\xd8\xff"

PHOTO_FAILED = "I couldn't take a photo."
PHOTO_NOT_SENT = "Muse didn't get the photo."
# What Reachy says when a photo request ends without Muse's answer, by outcome.
PHOTO_FAILURE_LINES = {
    "expired": "Please ask me again for the photo.",
    "camera_busy": PHOTO_FAILED,
    "no_frame": PHOTO_FAILED,
    "camera_failed": PHOTO_FAILED,
    "too_large": "That photo was too large to send.",
    "rejected": PHOTO_NOT_SENT,
    # The photo may have reached Muse with only its acknowledgement late.
    "send_timeout": "I'm not sure Muse got the photo.",
    "no_answer": "Muse didn't answer about the photo.",
}

Encoder = Callable[[object, int, int, int], bytes]   # (frame, width, height, quality) -> JPEG


@dataclass(frozen=True)
class PhotoRequest:
    """A one-use permission: Reachy's own recognizer heard an admitted turn ask for a photo.

    One capture attempt uses it up, whatever its outcome, and it lapses PHOTO_REQUEST_TTL_S after the turn ended.
    """

    words: str = field(repr=False)   # sent to Muse with the photo
    heard_at: float                  # the turn's endpoint, on the monotonic clock

    def expired(self, now: float) -> bool:
        return now - self.heard_at > PHOTO_REQUEST_TTL_S


def photo_allowed(timeline: TurnTimeline) -> bool:
    """Whether this turn's voice may ask for a photo: any voice Reachy admits, for now.

    Once the owner's voiceprint accuracy is measured, only the owner may ask, with
    ``timeline.facts.get("speaker") == "owner"``. A refused request goes on as an ordinary one.
    """
    return True


@dataclass(frozen=True)
class PhotoCapture:
    """One capture attempt: a JPEG and how it was made, or why there is none."""

    failure: str | None = None     # camera_busy, no_frame, camera_failed or too_large
    jpeg: bytes | None = field(default=None, repr=False)
    width: int | None = None
    height: int | None = None
    quality: int | None = None
    capture_ms: int | None = None  # from the request to a frame, including the wait for a still head
    encode_ms: int | None = None


@dataclass
class PhotoLog:
    """The one content-free line logged for each photo request."""

    path: str                      # chat, or realtime when the request was heard in a call
    outcome: str = "interrupted"
    width: int | None = None
    height: int | None = None
    quality: int | None = None
    size: int | None = None
    capture_ms: int | None = None
    encode_ms: int | None = None
    upload_ms: int | None = None

    def captured(self, capture: PhotoCapture) -> None:
        self.width, self.height, self.quality = capture.width, capture.height, capture.quality
        self.size = len(capture.jpeg) if capture.jpeg is not None else None
        self.capture_ms, self.encode_ms = capture.capture_ms, capture.encode_ms

    def write(self) -> None:
        log.info("Reachy photo outcome=%s path=%s width=%s height=%s quality=%s bytes=%s capture_ms=%s "
                 "encode_ms=%s upload_ms=%s", self.outcome, self.path,
                 *("none" if value is None else value for value in (
                     self.width, self.height, self.quality, self.size, self.capture_ms, self.encode_ms,
                     self.upload_ms)))


def fit_jpeg(frame, encode: Encoder, *, deadline: float) -> tuple[bytes, int, int, int] | None:
    """The first JPEG on ``JPEG_LADDER`` that fits, as (jpeg, width, height, quality); None if none does.

    Raises ``TimeoutError`` if ``deadline`` passes before a size and quality is tried.
    """
    height, width = frame.shape[:2]
    tried = set()
    for widest, quality in JPEG_LADDER:
        size = (min(width, widest), round(height * min(width, widest) / width))
        if (size, quality) in tried:
            continue
        tried.add((size, quality))
        if time.monotonic() >= deadline:
            raise TimeoutError("the photo was not encoded in time")
        jpeg = encode(frame, *size, quality)
        if not jpeg.startswith(JPEG_MAGIC):
            raise ValueError("the encoder did not make a JPEG")
        if len(jpeg) <= PHOTO_MAX_BYTES:
            return jpeg, *size, quality
    return None


def opencv_jpeg(frame, width: int, height: int, quality: int) -> bytes:
    """``frame`` as a JPEG of ``width`` by ``height``; OpenCV writes no metadata."""
    import cv2

    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not encoded:
        raise ValueError("OpenCV could not encode the photo")
    return jpeg.tobytes()


def shutter_cue(rate: int):
    """A camera shutter, two clicks over 120 ms, generated locally."""
    import numpy as np

    t = np.arange(round(rate * 0.12)) / rate
    clicks = np.exp(-t / .004) + .6 * np.exp(-np.maximum(t - .07, 0) / .006) * (t >= .07)
    return (0.12 * np.random.default_rng(1).uniform(-1, 1, len(t)) * clicks).astype(np.float32)
