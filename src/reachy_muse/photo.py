# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""One photo for Muse, taken only when the user asks Reachy for it.

A camera frame never leaves the capture: only a JPEG of at most ``PHOTO_MAX_BYTES`` does, in memory.
Nothing is written to disk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

PHOTO_CAPTURE_S = 2.0
PHOTO_MAX_BYTES = 150 * 1024
# (widest size, quality), tried in order until a JPEG fits PHOTO_MAX_BYTES.
JPEG_LADDER = ((1280, 80), (1280, 70), (1280, 60), (960, 70), (960, 55), (640, 60))
JPEG_MAGIC = b"\xff\xd8\xff"

Encoder = Callable[[object, int, int, int], bytes]   # (frame, width, height, quality) -> JPEG


@dataclass(frozen=True)
class PhotoCapture:
    """One capture attempt: a JPEG and how it was made, or why there is none."""

    failure: str | None = None     # camera_busy, no_frame, camera_failed or too_large
    jpeg: bytes | None = None
    width: int | None = None
    height: int | None = None
    quality: int | None = None
    capture_ms: int | None = None  # from the request to a frame, including the wait for a still head
    encode_ms: int | None = None


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
