# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Motion-loop stillness and local camera rotation compensation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class CameraGaze:
    generation: int = 0
    settled: bool = False
    observed_at: float = -math.inf
    rotation: np.ndarray | None = None


class GazeSettler:
    """Require both commanded and observed motion below 1 deg/s for 300 ms.

    Translation must also stay below 0.8 mm/s. A generation changes on every
    moving tick, including motion entirely between two camera observations.
    """

    def __init__(self):
        self.gaze = CameraGaze()
        self._previous = None
        self._since = None

    def invalidate(self):
        self.gaze = CameraGaze(self.gaze.generation + 1)
        self._previous = self._since = None

    def update(self, now, commanded, observed, rotation, *, eligible):
        import numpy as np

        previous = self._previous
        dt = now - self.gaze.observed_at
        quiet = eligible and previous is not None and 0 < dt <= .1
        if quiet:
            for current, old in zip((commanded, observed), previous):
                velocity = np.abs(np.asarray(current) - old) / dt
                quiet &= (np.isfinite(velocity).all() and np.linalg.norm(velocity[:3]) < .8
                          and np.linalg.norm(velocity[3:6]) < 1 and velocity[6] < 1)
        self._previous = (np.array(commanded, copy=True), np.array(observed, copy=True))
        if not quiet:
            self._since = None
        elif self._since is None:
            self._since = now
        self.gaze = CameraGaze(
            self.gaze.generation + int(not quiet),
            self._since is not None and now - self._since >= .3,
            now, rotation.copy())


class GazeProjection:
    """Project frames into one fixed head orientation for an evidence window.

    Inference sees the same coordinates as the motion gate and cached person
    boxes. Full frames exist only during a sample; only calibration is retained.
    Translation/parallax and unmodelled scene motion still face the torso and
    broad-motion rejection in WaveDetector.
    """

    def __init__(self, camera_matrix):
        import numpy as np

        matrix = np.array(camera_matrix, dtype=float, copy=True)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or matrix[0, 0] <= 0 or matrix[1, 1] <= 0):
            raise ValueError("Wave gaze requires camera calibration")
        self._matrix = matrix
        self._reference = None
        self._shape = None

    def reset(self):
        self._reference = self._shape = None

    def project(self, frame, rotation):
        import cv2
        import numpy as np
        from reachy_muse.arrival import _check_frame

        _check_frame(frame)
        rotation = np.asarray(rotation)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("Wave gaze requires a finite head rotation")
        shape = frame.shape[:2]
        if self._shape is not None and self._shape != shape:
            raise ValueError("Camera dimensions changed during wave evidence")
        if self._reference is None:
            self._reference, self._shape = rotation.copy(), shape
        # YOLOX already reduces to 416. Avoid warping a full 1080p frame on the CPU.
        height, width = shape
        scale = min(1.0, 416 / width)
        size = (round(width * scale), round(height * scale))
        scale_matrix = np.diag([size[0] / width, size[1] / height, 1.0])
        matrix = scale_matrix @ self._matrix
        # Camera right/down/forward -> robot forward/left/up.
        axes = np.array([[0., 0, 1], [-1, 0, 0], [0, -1, 0]])
        transform = matrix @ axes.T @ self._reference.T @ rotation @ axes @ np.linalg.inv(matrix)
        reduced = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR) if scale < 1 else frame
        projected = cv2.warpPerspective(reduced, transform, size, borderMode=cv2.BORDER_REPLICATE)
        # Map a cue's stabilized pixel center back to the original camera image.
        return projected, np.linalg.inv(scale_matrix) @ np.linalg.inv(transform)
