# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Local camera detection that supplies poses to the existing motion owner."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from pathlib import Path
import threading
import time
from typing import Callable, Sequence


YAW_LIMIT = math.radians(20)
PITCH_LIMIT = math.radians(12)
FACE_LOSS_SECONDS = 0.8
IDLE_SAMPLE_PERIOD = 0.5
SEEN_FACES = 256


def default_face_model() -> Path:
    return Path(__file__).parent / "data/face_tracking/face_detection_yunet_2023mar.onnx"


@dataclass(frozen=True)
class FaceTarget:
    """Absolute head angles in radians, with a monotonic observation time.

    ``paused_s`` is how long vision has been paused since then; that time never counts toward losing the face.
    """

    yaw: float
    pitch: float
    observed_at: float
    paused_s: float = 0.0

    def unseen_s(self, now: float) -> float:
        """Seconds by ``now`` that vision looked for the face without seeing it."""
        return now - self.observed_at - self.paused_s


def _resumed(target: FaceTarget | None, paused_at: float, now: float) -> FaceTarget | None:
    """``target`` once vision looks again at ``now``, after a pause that began at ``paused_at``."""
    if target is None:
        return None
    return replace(target, paused_s=target.paused_s + max(0.0, now - max(paused_at, target.observed_at)))


@dataclass(frozen=True)
class ObservedFace:
    yaw: float
    pitch: float
    box: tuple[float, float, float, float]
    score: float
    captured_at: float


@dataclass(frozen=True)
class FaceSnapshot:
    faces: tuple[ObservedFace, ...]
    captured_at: float
    heading: float                 # world yaw of the camera axis at capture
    elevation: float               # world pitch of the camera axis at capture
    half_view: tuple[float, float]  # half the camera's horizontal and vertical fields of view

    def covers(self, face: ObservedFace) -> bool:
        half_width, half_height = self.half_view
        return (abs(math.remainder(face.yaw - self.heading, math.tau)) <= half_width
                and abs(face.pitch - self.elevation) <= half_height)


@dataclass(frozen=True)
class FaceCapture:
    frame: object
    angles: tuple[float, float]
    rotation: object
    captured_at: float
    half_view: tuple[float, float]


@dataclass
class _FaceRequest:
    deadline: float
    done: threading.Event
    snapshot: FaceSnapshot | None = None


def _snapshot(faces, capture: FaceCapture) -> FaceSnapshot:
    result = []
    for face in faces:
        yaw, pitch = face.yaw_error, face.pitch_error
        ray = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch))
        x, y, z = [sum(a * b for a, b in zip(row, ray)) for row in capture.rotation]
        result.append(ObservedFace(math.atan2(y, x), math.atan2(-z, math.hypot(x, y)),
                                   face.box, face.score, capture.captured_at))
    forward_x, forward_y, forward_z = (row[0] for row in capture.rotation)
    return FaceSnapshot(tuple(result), capture.captured_at, math.atan2(forward_y, forward_x),
                        math.atan2(-forward_z, math.hypot(forward_x, forward_y)), capture.half_view)


@dataclass(frozen=True)
class _Face:
    """Detection geometry only; no image or identity is retained."""

    box: tuple[float, float, float, float]
    yaw_error: float
    pitch_error: float
    score: float

    @property
    def center(self) -> tuple[float, float]:
        x, y, width, height = self.box
        return x + width / 2, y + height / 2

    @property
    def area(self) -> float:
        return self.box[2] * self.box[3]


class _YuNetDetector:
    def __init__(self, model: Path, camera_matrix, distortion) -> None:
        import cv2
        import numpy as np

        if not model.is_file():
            raise FileNotFoundError("Local face detector model was not found")
        self._cv = cv2
        self._np = np
        self._matrix = np.array(camera_matrix, dtype=np.float64, copy=True)
        if (self._matrix.shape != (3, 3) or not np.isfinite(self._matrix).all()
                or self._matrix[0, 0] <= 0 or self._matrix[1, 1] <= 0):
            raise ValueError("Face tracking requires a calibrated camera matrix")
        self._distortion = np.array(distortion if distortion is not None else np.zeros(5),
                                    dtype=np.float64, copy=True)
        if self._distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(self._distortion).all():
            raise ValueError("Face tracking camera distortion is invalid")
        # Avoid occupying every CPU core while audio recognition is running.
        cv2.setNumThreads(1)
        self._detector = cv2.FaceDetectorYN.create(
            str(model), "", (320, 320), 0.7, 0.3, 100,
            cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
        )
        self._detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))

    def detect(self, frame) -> list[_Face]:
        np, cv = self._np, self._cv
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3
                or min(frame.shape[:2]) < 2):
            raise ValueError("Face tracking expects a BGR uint8 camera frame")
        height, width = frame.shape[:2]
        scale = min(320 / width, 320 / height)
        resized_width, resized_height = round(width * scale), round(height * scale)
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        left, top = (320 - resized_width) // 2, (320 - resized_height) // 2
        image[top:top + resized_height, left:left + resized_width] = cv.resize(
            frame, (resized_width, resized_height), interpolation=cv.INTER_AREA)
        _, rows = self._detector.detect(image)
        result = []
        if rows is None:
            return result
        for row in rows:
            if not np.isfinite(row).all() or float(row[-1]) < 0.7:
                continue
            x, y, box_width, box_height = [float(value) for value in row[:4]]
            u, v = (x + box_width / 2 - left) / scale, (y + box_height / 2 - top) / scale
            if not 0 <= u < width or not 0 <= v < height:
                continue
            x1, y1 = max(0.0, (x - left) / scale), max(0.0, (y - top) / scale)
            x2 = min(float(width), (x + box_width - left) / scale)
            y2 = min(float(height), (y + box_height - top) / scale)
            if x2 <= x1 or y2 <= y1:
                continue
            ray = cv.undistortPoints(np.array([[[u, v]]], dtype=np.float64),
                                     self._matrix, self._distortion)[0, 0]
            xn, yn = float(ray[0]), float(ray[1])
            if not math.isfinite(xn) or not math.isfinite(yn):
                continue
            result.append(_Face((x1 / width, y1 / height, (x2 - x1) / width,
                                 (y2 - y1) / height),
                                -math.atan2(xn, 1.0),
                                math.atan2(yn, math.sqrt(1.0 + xn * xn)), float(row[-1])))
        return result


class _FaceTracker:
    def __init__(self, sample_period: float, *, hold_center: float | None = None):
        self._period = sample_period
        self.loss_seconds = FACE_LOSS_SECONDS
        self._face: _Face | None = None
        self._target: FaceTarget | None = None
        self._hold_center = hold_center

    def update(self, faces: Sequence[_Face], angles: tuple[float, float],
               now: float) -> FaceTarget | None:
        faces = [face for face in faces if face.score >= 0.7 and face.area > 0
                 and all(math.isfinite(value) for value in (*face.box, face.yaw_error,
                                                            face.pitch_error, face.score))]
        if self._hold_center is not None:
            faces = [face for face in faces if abs(angles[0] + face.yaw_error - self._hold_center) <= YAW_LIMIT]
        if self._target is not None and self._target.unseen_s(now) >= self.loss_seconds:
            self._face = self._target = None
        chosen = None
        if self._face is not None:
            cx, cy = self._face.center
            nearby = [face for face in faces
                      if math.dist(face.center, (cx, cy)) <= max(0.12, math.sqrt(self._face.area) * 0.8)
                      and 0.25 <= face.area / self._face.area <= 4]
            if nearby:
                chosen = min(nearby, key=lambda face: math.dist(face.center, (cx, cy)))
        elif faces:
            chosen = (min(faces, key=lambda face: abs(angles[0] + face.yaw_error - self._hold_center))
                      if self._hold_center is not None
                      else max(faces, key=lambda face: face.area))
        if chosen is None:
            return self._target
        yaw, pitch = angles
        if not all(math.isfinite(value) for value in angles):
            return None
        yaw_error = chosen.yaw_error if abs(chosen.yaw_error) > math.radians(1) else 0.0
        pitch_error = chosen.pitch_error if abs(chosen.pitch_error) > math.radians(1) else 0.0
        desired_yaw = max(-YAW_LIMIT, min(YAW_LIMIT, yaw + yaw_error))
        desired_pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, pitch + pitch_error))
        previous = self._target or FaceTarget(yaw, pitch, now - self._period)
        elapsed = max(0.0, min(0.4, now - previous.observed_at))
        blend = 1 - math.exp(-elapsed / 0.3)

        def smooth(old, desired, rate, limit):
            delta = (desired - old) * blend
            delta = max(-rate * elapsed, min(rate * elapsed, delta))
            return max(-limit, min(limit, old + delta))

        self._target = FaceTarget(
            smooth(previous.yaw, desired_yaw, math.radians(35), YAW_LIMIT),
            smooth(previous.pitch, desired_pitch, math.radians(25), PITCH_LIMIT), now,
        )
        self._face = chosen
        return self._target

    def resume(self, paused_at: float, now: float) -> None:
        """Vision looks again at ``now``, after a pause that began at ``paused_at``."""
        self._target = _resumed(self._target, paused_at, now)


class FaceFollower:
    """A gated detector worker; the caller remains the sole motion writer.

    ``capture_frame(background=...)`` returns a ``FaceCapture`` with the head
    angles and rotation at capture, or None. ``camera_matrix`` and ``distortion``
    must describe its original frame. Frames and detector inputs exist only
    during one worker iteration.

    While the budget pauses detection, the follower keeps its target: Reachy
    goes on facing the person it followed. The pause never counts toward losing
    that face, so once vision looks again a face that has gone is dropped as
    any unseen face is. ``paused_s`` counts the seconds following waited out a
    pause, and ``held_s`` those of them it kept a target through.
    """

    def __init__(self, capture_frame: Callable[..., FaceCapture | None], model_path: Path | None = None, *,
                 camera_matrix, distortion=None, sample_hz: float = 5.0, clock: Callable = time.monotonic,
                 detector_factory: Callable | None = None, budget=None):
        if not math.isfinite(sample_hz) or not 1 <= sample_hz <= 15:
            raise ValueError("Face tracking sample rate must be between 1 and 15 Hz")
        self._budget = budget
        self._capture_frame = capture_frame
        self._model = Path(model_path) if model_path is not None else default_face_model()
        self._matrix = camera_matrix
        self._distortion = distortion
        self._detector_factory = detector_factory or _YuNetDetector
        self._period = 1.0 / sample_hz
        self._sample_period = self._period
        self._loss_seconds = FACE_LOSS_SECONDS
        self._clock = clock
        self._condition = threading.Condition()
        self._active = False
        self._closed = False
        self._generation = 0
        self._target: FaceTarget | None = None
        self._snapshot: FaceSnapshot | None = None
        self._seen: deque[ObservedFace] = deque(maxlen=SEEN_FACES)
        self._request: _FaceRequest | None = None
        self._hold_center: float | None = None
        self._paused_at: float | None = None
        self.paused_s = 0.0
        self.held_s = 0.0
        self.last_seen_at: float | None = None
        self._failure: Exception | None = None
        self._thread: threading.Thread | None = None

    @property
    def failure(self) -> Exception | None:
        with self._condition:
            return self._failure

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("Local face tracking is already closed")
            if self._thread is not None:
                return
            detector = self._detector_factory(self._model, self._matrix, self._distortion)
            self._thread = threading.Thread(target=self._run, args=(detector,),
                                            name="muse-face-detector", daemon=True)
            self._thread.start()

    def set_active(self, active: bool, *, idle: bool = False) -> None:
        with self._condition:
            period = max(self._period, IDLE_SAMPLE_PERIOD) if idle else self._period
            if self._closed or (active == self._active and period == self._sample_period):
                return
            if active != self._active:
                self._active = active
                self._generation += 1
                self._target = None
            self._sample_period = period
            self._condition.notify_all()

    def target(self) -> FaceTarget | None:
        with self._condition:
            return self._current()

    def _current(self) -> FaceTarget | None:
        target = self._target
        now = self._clock() if self._paused_at is None else self._paused_at
        if not self._active or target is None or target.unseen_s(now) >= self._loss_seconds:
            return None
        return target

    def snapshot(self) -> FaceSnapshot | None:
        with self._condition:
            return self._snapshot

    def seen_since(self, since: float) -> tuple[ObservedFace, ...]:
        """Faces from snapshots captured at or after ``since``, among the last ``SEEN_FACES`` seen."""
        with self._condition:
            return tuple(face for face in self._seen if face.captured_at >= since)

    def hold_center(self, yaw: float | None) -> None:
        with self._condition:
            self._hold_center = yaw
            self._generation += 1
            self._target = None
            self._condition.notify_all()

    def refresh(self, timeout: float = .3) -> FaceSnapshot | None:
        with self._condition:
            if self._closed or self._failure is not None or self._thread is None or self._request is not None:
                return None
            request = self._request = _FaceRequest(self._clock() + timeout, threading.Event())
            self._condition.notify_all()
        if request.done.wait(timeout):
            return request.snapshot
        return None

    def close(self, timeout: float = 2.0) -> None:
        with self._condition:
            self._closed = True
            self._active = False
            self._target = None
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise RuntimeError("Local face detector did not stop")

    def _run(self, detector) -> None:
        tracker = _FaceTracker(self._period)
        generation = -1
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._active or self._request is not None or self._closed)
                    if self._closed:
                        return
                    current_generation = self._generation
                    request = self._request
                    if request is not None and self._clock() >= request.deadline:
                        self._request = None
                        request.done.set()
                        continue
                if generation != current_generation:
                    tracker = _FaceTracker(self._period, hold_center=self._hold_center)
                    generation = current_generation
                started = self._clock()
                scale = self._budget.delay() if self._budget is not None else 1.0
                with self._condition:
                    if request is None and not math.isfinite(scale):
                        self._wait_out_pause(started)
                        continue
                    if self._paused_at is not None:
                        self._target = _resumed(self._target, self._paused_at, started)
                        tracker.resume(self._paused_at, started)
                        self._paused_at = None if math.isfinite(scale) else started
                try:
                    capture = self._capture_frame(background=request is None)
                    if request is not None and self._clock() >= request.deadline:
                        capture = None
                    if (request is None and self._budget is not None
                            and not math.isfinite(self._budget.delay())):
                        capture = None
                        with self._condition:
                            if self._paused_at is None:
                                self._paused_at = self._clock()
                    if capture is not None:
                        try:
                            faces = detector.detect(capture.frame)
                            snapshot = _snapshot(faces, capture)
                            target = tracker.update(faces, capture.angles, capture.captured_at)
                        finally:
                            del capture
                        with self._condition:
                            if not self._closed:
                                self._snapshot = snapshot
                                self._seen.extend(snapshot.faces)
                                if request is not None:
                                    request.snapshot = snapshot
                                if self._active and self._generation == generation:
                                    self._target = target
                                    if target is not None:
                                        self.last_seen_at = target.observed_at
                except Exception:
                    if request is None:
                        raise
                finally:
                    if request is not None:
                        with self._condition:
                            self._request = None
                            request.done.set()
                with self._condition:
                    if self._closed:
                        return
                    if self._active and self._generation == generation:
                        cost = self._clock() - started
                        period = max(self._sample_period * scale, cost * 5) if math.isfinite(scale) else .1
                        tracker.loss_seconds = self._loss_seconds = max(FACE_LOSS_SECONDS, period + cost + self._period)
                        self._condition.wait_for(
                            lambda: self._closed or self._request is not None or not self._active
                            or self._generation != generation,
                            timeout=max(.001, period - cost))
        except Exception as exc:
            # A traceback could otherwise retain its camera frame indefinitely.
            exc.__traceback__ = exc.__context__ = exc.__cause__ = None
            with self._condition:
                self._failure = exc
                self._active = False
                self._target = None

    def _wait_out_pause(self, now: float) -> None:
        """One wait while the budget pauses detection, with the condition held: the target stays, and the pause
        that began here, or earlier, stops the clock that loses its face."""
        if self._paused_at is None:
            self._paused_at = now
        holding = self._current() is not None
        self._condition.wait(.1)
        waited = self._clock() - now
        self.paused_s += waited
        if holding:
            self.held_s += waited
