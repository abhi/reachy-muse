# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Notice someone walking into the room with Reachy's camera; frames stay on the robot."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    import numpy as np


GATE_WIDTH = 160
PIXEL_DELTA = 18.0
BACKGROUND_RATE = 0.05
MOTION_MIN_FRACTION = 0.004
LIGHTING_CHANGE_FRACTION = 0.5
LIGHT_JUMP = 10.0
LIGHT_BURST_S = 8.0
LIGHT_DETECT_PERIOD_S = 0.25
SETTLE_S = 1.0
MOTION_MEMORY_S = 1.0
BURST_S = 3.0
DETECT_PERIOD_S = 0.5
PERSON_SCORE = 0.4
NMS_IOU = 0.6
MIN_BOX_HEIGHT = 0.08
MIN_BOX_WIDTH = 0.02
BOX_MOTION_FRACTION = 0.15
CONFIRM_DETECTIONS = 2
CONFIRM_WINDOW_S = 2.0
WATCH_HZ = 3.0
_YOLOX_SIZE = 416
log = logging.getLogger(__name__)


def default_person_model() -> Path:
    return Path(__file__).parent / "data/arrival/yolox_nano.onnx"


@dataclass(frozen=True)
class WatchPose:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0


class ArrivalCamera(Protocol):
    def start_watching(self, pose: WatchPose | None) -> None: ...

    def stop_watching(self) -> None: ...

    def arrival(self) -> float | None: ...

    def log_arrival_diagnostics(self) -> None: ...


@dataclass
class ArrivalStats:
    frames: int = 0
    frame_errors: int = 0
    brightness_total: float = 0.0
    motion_triggers: int = 0
    detector_runs: int = 0
    max_person_confidence: float = 0.0

    @property
    def brightness(self) -> float:
        return self.brightness_total / self.frames if self.frames else 0.0


@dataclass(frozen=True)
class PersonBox:
    x: float
    y: float
    width: float
    height: float
    score: float


class PersonDetector(Protocol):
    def detect(self, frame) -> list[PersonBox]: ...


def _check_frame(frame) -> None:
    import numpy as np

    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 2):
        raise ValueError("Arrival detection expects a BGR uint8 camera frame")


def _gate_image(frame):
    import numpy as np

    _check_frame(frame)
    height, width = frame.shape[:2]
    if width > GATE_WIDTH:
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (GATE_WIDTH, max(2, round(height * GATE_WIDTH / width))),
                          interpolation=cv2.INTER_AREA)
    else:
        gray = frame @ np.array([0.114, 0.587, 0.299])
    gray = gray.astype(np.float32)
    rows, cols = gray.shape
    padded = np.pad(gray, 1, mode="edge")
    return sum(padded[dy:dy + rows, dx:dx + cols] for dy in range(3) for dx in range(3)) / 9


@dataclass(frozen=True)
class GateReading:
    moving: bool
    lighting: bool
    moved_recently: np.ndarray
    light_on: bool = False


class MotionGate:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._background = None
        self._moved_at = None
        self._settle_until = -math.inf

    def update(self, frame, now: float) -> GateReading:
        import numpy as np

        gray = _gate_image(frame)
        if self._background is None or self._background.shape != gray.shape:
            self._background = gray
            self._moved_at = np.full(gray.shape, -np.inf)
            self._settle_until = now + SETTLE_S
            return GateReading(False, False, np.zeros(gray.shape, dtype=bool))
        difference = gray - self._background
        scene_brightening = float(np.median(difference))
        changed = np.abs(difference - scene_brightening) > PIXEL_DELTA
        fraction = float(changed.mean())
        lighting = abs(scene_brightening) > LIGHT_JUMP or fraction > LIGHTING_CHANGE_FRACTION
        if lighting:
            self._settle_until = now + SETTLE_S
        if now < self._settle_until:
            self._background = gray
            self._moved_at.fill(-np.inf)
            return GateReading(False, lighting, np.zeros(gray.shape, dtype=bool), scene_brightening > LIGHT_JUMP)
        self._background += BACKGROUND_RATE * (gray - self._background)
        moving = fraction >= MOTION_MIN_FRACTION
        if moving:
            self._moved_at[changed] = now
        return GateReading(moving, False, now - self._moved_at <= MOTION_MEMORY_S)


def _on_motion(box: PersonBox, moved_recently: np.ndarray) -> bool:
    if box.height < MIN_BOX_HEIGHT or box.width < MIN_BOX_WIDTH:
        return False
    rows, cols = moved_recently.shape
    top, left = int(box.y * rows), int(box.x * cols)
    bottom = max(top + 1, math.ceil((box.y + box.height) * rows))
    right = max(left + 1, math.ceil((box.x + box.width) * cols))
    return float(moved_recently[top:bottom, left:right].mean()) >= BOX_MOTION_FRACTION


class ArrivalDetector:
    def __init__(self, detector: PersonDetector) -> None:
        self._detector = detector
        self._gate = MotionGate()
        self.arrived_at: float | None = None
        self.motion_triggered = False
        self.detector_ran = False
        self.max_person_confidence = 0.0
        self.reachy_moved()

    def reachy_moved(self) -> None:
        self._gate.reset()
        self._burst_until = -math.inf
        self._light_until = -math.inf
        self._detected_at = -math.inf
        self._hits: list[float] = []

    def observe(self, frame, now: float) -> bool:
        self.motion_triggered = self.detector_ran = False
        self.max_person_confidence = 0.0
        if self.arrived_at is not None:
            return True
        reading = self._gate.update(frame, now)
        self.motion_triggered = reading.moving
        if reading.light_on:
            self._light_until = now + LIGHT_BURST_S
        if reading.moving or reading.lighting:
            self._burst_until = now + BURST_S
        light_burst = now < self._light_until
        period = LIGHT_DETECT_PERIOD_S if light_burst else DETECT_PERIOD_S
        if (not light_burst and now >= self._burst_until) or now - self._detected_at < period:
            return False
        self._detected_at = now
        if light_burst:
            frame = _brightened(frame)
        self.detector_ran = True
        boxes = self._detector.detect(frame)
        self.max_person_confidence = max(
            (box.score for box in boxes), default=getattr(self._detector, "max_person_confidence", 0.0))
        if any(box.score >= PERSON_SCORE and _on_motion(box, reading.moved_recently) for box in boxes):
            self._hits = [hit for hit in self._hits if now - hit <= CONFIRM_WINDOW_S] + [now]
            if len(self._hits) >= CONFIRM_DETECTIONS:
                self.arrived_at = now
        return self.arrived_at is not None


def _brightened(frame):
    import numpy as np

    brightness = float(frame.mean())
    if brightness >= 60:
        return frame
    gain = min(3.0, 80.0 / max(1.0, brightness))
    return np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)


class YoloxPersonDetector:
    def __init__(self, model: Path) -> None:
        if not model.is_file():
            raise FileNotFoundError("Local person detector model was not found")
        import cv2
        import numpy as np

        self._cv = cv2
        self._np = np
        cv2.setNumThreads(1)
        self._net = cv2.dnn.readNetFromONNX(str(model))
        grids, strides = [], []
        for stride in (8, 16, 32):
            side = _YOLOX_SIZE // stride
            xs, ys = np.meshgrid(np.arange(side), np.arange(side))
            grids.append(np.stack((xs, ys), 2).reshape(-1, 2))
            strides.append(np.full((side * side, 1), stride))
        self._grid = np.concatenate(grids).astype(np.float32)
        self._stride = np.concatenate(strides).astype(np.float32)
        self.max_person_confidence = 0.0

    def detect(self, frame) -> list[PersonBox]:
        np, cv = self._np, self._cv
        _check_frame(frame)
        height, width = frame.shape[:2]
        scale = min(_YOLOX_SIZE / width, _YOLOX_SIZE / height)
        resized = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = np.full((_YOLOX_SIZE, _YOLOX_SIZE, 3), 114, dtype=np.uint8)
        image[:resized[1], :resized[0]] = cv.resize(frame, resized, interpolation=cv.INTER_LINEAR)
        self._net.setInput(cv.dnn.blobFromImage(image))
        rows = self._net.forward().reshape(-1, 85)
        scores = rows[:, 4] * rows[:, 5]
        self.max_person_confidence = float(scores.max())
        keep = scores >= PERSON_SCORE
        rows, scores = rows[keep], scores[keep]
        stride = self._stride[keep]
        sizes = np.exp(rows[:, 2:4]) * stride
        corners = (rows[:, :2] + self._grid[keep]) * stride - sizes / 2
        boxes = np.concatenate([corners, sizes], axis=1) / scale
        chosen = cv.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), PERSON_SCORE, NMS_IOU)
        result = []
        for index in np.asarray(chosen, dtype=int).reshape(-1):
            x, y, box_width, box_height = (float(value) for value in boxes[index])
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width), x + box_width), min(float(height), y + box_height)
            if x2 > x1 and y2 > y1:
                result.append(PersonBox(x1 / width, y1 / height, (x2 - x1) / width,
                                        (y2 - y1) / height, float(scores[index])))
        return result


class ArrivalWatcher:
    def __init__(self, read_frame: Callable, is_still: Callable[[], bool],
                 model_path: Path | None = None, *, sample_hz: float = WATCH_HZ,
                 clock: Callable[[], float] = time.monotonic,
                 detector_factory: Callable | None = None):
        if not math.isfinite(sample_hz) or not 0.5 <= sample_hz <= 10:
            raise ValueError("Arrival watch rate must be between 0.5 and 10 Hz")
        self._read_frame = read_frame
        self._is_still = is_still
        self._model = Path(model_path) if model_path is not None else default_person_model()
        self._detector_factory = detector_factory or YoloxPersonDetector
        self._period = 1.0 / sample_hz
        self._clock = clock
        self._condition = threading.Condition()
        self._watching = False
        self._closed = False
        self._generation = 0
        self._arrival: float | None = None
        self._failure: Exception | None = None
        self._thread: threading.Thread | None = None
        self._stats = ArrivalStats()

    @property
    def failure(self) -> Exception | None:
        with self._condition:
            return self._failure

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("Arrival watching is already closed")
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="muse-arrival", daemon=True)
            self._thread.start()

    def set_watching(self, watching: bool) -> None:
        with self._condition:
            if self._closed or watching == self._watching:
                return
            if watching:
                self._arrival = None
                self._generation += 1
                self._stats = ArrivalStats()
            self._watching = watching
            self._condition.notify_all()

    def arrival(self) -> float | None:
        with self._condition:
            return self._arrival

    def take_stats(self) -> ArrivalStats:
        with self._condition:
            stats, self._stats = self._stats, ArrivalStats()
            return stats

    def close(self, timeout: float = 2.0) -> None:
        with self._condition:
            self._closed = True
            self._watching = False
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise RuntimeError("Arrival watcher did not stop")

    def _looking(self) -> bool:
        return self._watching and self._arrival is None

    def _run(self) -> None:
        detector = None
        decision = None
        generation = -1
        try:
            while True:
                with self._condition:
                    if not self._looking():
                        decision = None
                    self._condition.wait_for(lambda: self._looking() or self._closed)
                    if self._closed:
                        return
                    current_generation = self._generation
                if detector is None:
                    detector = self._detector_factory(self._model)
                if decision is None or generation != current_generation:
                    decision = ArrivalDetector(detector)
                    generation = current_generation
                started = self._clock()
                if not self._is_still():
                    decision.reachy_moved()
                else:
                    try:
                        frame = self._read_frame()
                        if frame is not None:
                            _check_frame(frame)
                    except Exception:
                        frame = None
                    if frame is not None:
                        try:
                            brightness = float(frame.mean())
                            with self._condition:
                                self._stats.frames += 1
                                self._stats.brightness_total += brightness
                            arrived = decision.observe(frame, self._clock())
                        finally:
                            del frame
                            with self._condition:
                                self._stats.motion_triggers += int(decision.motion_triggered)
                                self._stats.detector_runs += int(decision.detector_ran)
                                self._stats.max_person_confidence = max(
                                    self._stats.max_person_confidence, decision.max_person_confidence)
                        if arrived:
                            with self._condition:
                                if self._generation == generation and self._arrival is None:
                                    self._arrival = decision.arrived_at
                    else:
                        with self._condition:
                            self._stats.frame_errors += 1
                        decision.reachy_moved()
                with self._condition:
                    if self._closed:
                        return
                    if self._looking() and self._generation == generation:
                        self._condition.wait(max(0.0, self._period - (self._clock() - started)))
        except Exception as exc:
            log.warning("Arrival watcher failed: %s", type(exc).__name__)
            # A traceback could otherwise retain its camera frame indefinitely.
            exc.__traceback__ = exc.__context__ = exc.__cause__ = None
            with self._condition:
                self._failure = exc
                self._watching = False
