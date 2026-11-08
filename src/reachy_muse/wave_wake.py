# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Local, conservative raised-hand oscillation. No images leave this module."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import logging
import threading
import time

from reachy_muse.arrival import MotionGate, PersonBox, YoloxPersonDetector, _gate_image, default_person_model
from reachy_muse.camera_gaze import GazeProjection

log = logging.getLogger(__name__)


@dataclass
class WaveStats:
    frames: int = 0
    detector_runs: int = 0
    cues: int = 0
    max_score: float = 0.0
    frame_errors: int = 0


class WaveDetector:
    """A stable person, still torso and three raised lateral reversals in 1-3 s.

    Boxes use normalized image coordinates. The caller expires person detections
    and resets on robot motion. Only one reduced grayscale image is retained.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._previous = None
        self._at = -math.inf
        self._person = None
        self._paths = [deque(), deque()]
        self._completed_at = -math.inf
        self.score = 0.0

    def _position(self, previous, gray, changed, bounds):
        import numpy as np

        x1, y1, x2, y2 = bounds
        local = changed[y1:y2, x1:x2]
        delta = (gray - previous)[y1:y2, x1:x2]
        brighter = np.count_nonzero(local & (delta > 18))
        darker = np.count_nonzero(local & (delta < -18))
        # Translation uncovers background as well as covering it. Blinking does not.
        if min(brighter, darker) < 3 or min(brighter, darker) < .3 * max(brighter, darker):
            return None
        ys, xs = np.nonzero(local)
        if len(xs) < 6 or len(xs) > (x2 - x1) * (y2 - y1) * .45:
            return None
        if np.ptp(xs) > (x2 - x1) * .9 or np.ptp(ys) > (y2 - y1) * .65:
            return None
        return float(xs.mean() + x1), float(ys.mean() + y1)

    def observe(self, frame, boxes: list[PersonBox], now: float, *, confirmed: bool = True) -> bool:
        import numpy as np

        gray = _gate_image(frame)
        people = [b for b in boxes if all(math.isfinite(v) for v in
                  (b.x, b.y, b.width, b.height, b.score))
                  and b.score >= .6 and b.height >= .35 and b.width >= .15
                  and b.x >= 0 and b.y >= 0 and b.x + b.width <= 1
                  and b.y + b.height <= 1]
        if len(people) != 1 or not math.isfinite(now):
            self.reset()
            return False
        box = people[0]
        old = self._person
        if (old is None or abs(box.height / old.height - 1) > .2
                or math.dist((box.x + box.width / 2, box.y + box.height),
                             (old.x + old.width / 2, old.y + old.height)) > .12 * old.height
                or not 0 < now - self._at <= .6
                or self._previous is None or self._previous.shape != gray.shape):
            self.reset()
            self._previous, self._at, self._person = gray, now, box
            return False
        previous = self._previous
        self._previous, self._at = gray, now
        # Freeze the anchor during the gesture; detector box jitter is not hand motion.
        box = old
        rows, cols = gray.shape
        x, y, w, h = box.x * cols, box.y * rows, box.width * cols, box.height * rows
        delta = gray - previous
        light = float(np.median(delta))
        changed = np.abs(delta - light) > 18
        outside = changed.copy()
        outside[max(0, int(y - .15 * h)):min(rows, math.ceil(y + h)),
                max(0, int(x - .3 * w)):min(cols, math.ceil(x + 1.3 * w))] = False
        torso = changed[int(y + .38 * h):int(y + .85 * h),
                        int(x + .3 * w):int(x + .7 * w)]
        if abs(light) > 10 or outside.mean() > .08 or torso.size == 0 or torso.mean() > .08:
            self.reset()
            return False
        if confirmed and now - self._completed_at <= .75:
            self._completed_at = -math.inf
            return True
        self.score = 0.0
        for side, (left, right) in enumerate(((-.45, .32), (.68, 1.45))):
            bounds = (max(0, int(x + left * w)), max(0, int(y - .12 * h)),
                      min(cols, math.ceil(x + right * w)), min(rows, math.ceil(y + .32 * h)))
            position = self._position(previous, gray, changed, bounds)
            path = self._paths[side]
            if position is None:
                if path and now - path[-1][0] > .45:
                    path.clear()
                continue
            if path and now - path[-1][0] > .45:
                path.clear()
            path.append((now, *position))
            while path and now - path[0][0] > 3:
                path.popleft()
            if len(path) < 5 or max(p[2] for p in path) - min(p[2] for p in path) > .12 * h:
                continue
            direction, reversals, extreme = 0, 0, path[0][1]
            for _, px, _ in path:
                move = px - extreme
                if direction and move * direction >= 0:
                    extreme = px
                elif abs(move) >= .10 * w:
                    new_direction = 1 if move > 0 else -1
                    reversals += int(direction != 0 and new_direction != direction)
                    direction, extreme = new_direction, px
            self.score = max(self.score, min(1.0, reversals / 3))
            if reversals >= 3 and now - path[0][0] >= 1:
                self._paths = [deque(), deque()]
                self._completed_at = -math.inf if confirmed else now
                return confirmed
        return False


class WaveWatcher:
    """One bounded CPU worker. Disabling invalidates cues; draining destroys image history.

    Idle sampling is 2 Hz. Motion permits a 3 s burst at 8 Hz followed by a
    2 s rest, even under continuous motion. Person inference is at most 0.8 Hz
    in bursts. Throttling stretches the burst and sampling intervals by the same factor.
    Rearming requires 3 s without motion, including after pauses and reconnects.
    """

    def __init__(self, read_frame, is_still, model_path=None, *, clock=time.monotonic,
                 detector_factory=YoloxPersonDetector, read_gaze=None, camera_matrix=None, budget=None):
        self._budget = budget
        self._inference_after = -math.inf
        self._read_frame, self._is_still = read_frame, is_still
        self._read_gaze = read_gaze
        self._projection = GazeProjection(camera_matrix) if camera_matrix is not None else None
        self._model = model_path or default_person_model()
        self._factory, self._clock = detector_factory, clock
        self._condition = threading.Condition()
        self._work = threading.Lock()
        self._active = self._closed = False
        self._epoch = 0
        self._sample_epoch = None
        self._thread = None
        self._pending = None
        self.wave_position: tuple[float, float] | None = None
        self._cooldown_until = -math.inf
        self._rearming = False
        self._failed = False
        self._idle_tick = False
        self._generation = None
        self._reset()
        self._stats = WaveStats()

    def _reset(self):
        self._quiet_since = None
        self._gate = MotionGate()
        self._decision = WaveDetector()
        self._boxes = []
        self._burst_until = self._rest_until = self._detected_at = -math.inf
        self._confirmations = 0
        if self._projection is not None:
            self._projection.reset()

    def _gaze_valid(self, generation):
        gaze = self._read_gaze() if self._read_gaze is not None else None
        return (self._is_still() and (gaze is None or
                gaze.settled and gaze.generation == generation))

    def _discard_motion(self):
        self._reset()
        with self._condition:
            self._pending = None
            self.wave_position = None

    def start(self):
        with self._condition:
            if self._closed:
                raise RuntimeError("Wave watcher is closed")
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="muse-wave", daemon=True)
                self._thread.start()

    def set_active(self, active: bool):
        with self._condition:
            if active == self._active or self._closed or self._failed:
                return
            self._active = active
            self._epoch += 1
            self._pending = None
            self.wave_position = None
            self._condition.notify_all()

    def drain(self, timeout=2.0):
        if not self._work.acquire(timeout=timeout):
            return
        try:
            with self._condition:
                if self._active:
                    return
                self._reset()
                stats, self._stats = self._stats, WaveStats()
        finally:
            self._work.release()
        log.info("Wave stats frames=%d detector_runs=%d cues=%d max_score=%.3f frame_errors=%d",
                 stats.frames, stats.detector_runs, stats.cues, stats.max_score, stats.frame_errors)

    def _current(self, epoch):
        with self._condition:
            return self._active and not self._closed and self._epoch == epoch

    def take_wave(self) -> bool:
        with self._condition:
            pending, self._pending = self._pending, None
            self.wave_position = None
            if (self._active and not self._closed and pending is not None
                    and 0 <= self._clock() - pending[0] <= .75 and self._gaze_valid(pending[2])):
                self.wave_position = pending[1]
                return True
            return False

    def close(self, timeout=2.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            self._active = False
            self._closed = True
            self._pending = None
            self.wave_position = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(max(0.0, deadline - time.monotonic()))
            if self._thread.is_alive():
                raise RuntimeError("Wave watcher did not stop")
        else:
            self._reset()

    def _sample(self, detector):
        with self._condition:
            epoch = self._epoch
        if not self._current(epoch):
            self._discard_motion()
            return .5
        if epoch != self._sample_epoch:
            self._reset()
            self._sample_epoch = epoch
        now = self._clock()
        scale = self._budget.delay() if self._budget is not None else 1.0
        if not math.isfinite(scale):
            self._discard_motion()
            return .1
        gaze = self._read_gaze() if self._read_gaze is not None else None
        generation = gaze.generation if gaze is not None else None
        self._idle_tick = not self._idle_tick
        idle_period = .4 if self._idle_tick else .6
        if not self._gaze_valid(generation):
            self._discard_motion()
            return .5
        if generation != self._generation:
            self._discard_motion()
            self._generation = generation
        if now < self._rest_until:
            return .5
        if self._burst_until != -math.inf and now >= self._burst_until:
            self._reset()
            self._rest_until = now + 2
            return .5
        try:
            frame = self._read_frame()
            if frame is None:
                self._reset()
                self._stats.frame_errors += 1
                return .5
            if (not self._current(epoch) or not self._gaze_valid(generation) or self._budget is not None
                    and not math.isfinite(self._budget.delay())):
                self._discard_motion()
                return .5
            to_camera = None
            if self._projection is not None and gaze is not None:
                frame, to_camera = self._projection.project(frame, gaze.rotation)
            now = self._clock()
            reading = self._gate.update(frame, now)
            self._stats.frames += 1
            if reading.lighting:
                self._reset()
                return .5
            if self._rearming:
                if reading.moving:
                    self._quiet_since = None
                elif self._quiet_since is None:
                    self._quiet_since = now
                elif now - self._quiet_since >= 3 and now >= self._cooldown_until:
                    self._rearming = False
                    self._reset()
                return idle_period
            if self._burst_until == -math.inf:
                if not reading.moving:
                    # Avoid sampling a periodic wave at exactly the same phase.
                    return idle_period
                self._burst_until = now + 3 * scale
            if now >= self._inference_after and now - self._detected_at >= 1.25:
                started = self._clock()
                self._boxes = detector.detect(frame)
                cost = self._clock() - started
                self._inference_after = started + max(1.25 * scale, cost * 5)
                self._detected_at = now
                if len(self._boxes) == 1 and self._boxes[0].score >= .6:
                    self._confirmations += 1
                else:
                    self._confirmations = 0
                self._stats.detector_runs += 1
            if not self._current(epoch) or not self._gaze_valid(generation):
                self._discard_motion()
                return .5
            detected = self._decision.observe(frame, self._boxes, now, confirmed=self._confirmations >= 2)
            self._stats.max_score = max(self._stats.max_score, self._decision.score)
            if detected and self._confirmations >= 2:
                with self._condition:
                    if self._current(epoch) and self._gaze_valid(generation):
                        box = self._boxes[0]
                        height, width = frame.shape[:2]
                        position = ((box.x + box.width / 2) * width,
                                    (box.y + box.height / 2) * height)
                        if to_camera is not None:
                            pixel = to_camera @ [*position, 1.0]
                            position = (float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2]))
                        self._pending = (self._clock(), position, generation)
                        self._cooldown_until = self._clock() + 3
                        self._rearming = True
                        self._stats.cues += 1
                self._reset()
        except (ValueError, OSError):
            self._reset()
            self._stats.frame_errors += 1
            return .5
        finally:
            # Only the reduced previous image may survive an active iteration.
            frame = None
        return .125

    def _run(self):
        detector = None
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._active)
                    if self._closed:
                        return
                if self._budget is not None and not math.isfinite(self._budget.delay()):
                    with self._work:
                        self._discard_motion()
                    with self._condition:
                        self._condition.wait(.1)
                    continue
                started = self._clock()
                with self._work:
                    with self._condition:
                        if not self._active or self._closed:
                            continue
                        epoch = self._epoch
                    if detector is None:
                        detector = self._factory(self._model)
                    if not self._current(epoch):
                        self._discard_motion()
                        continue
                    period = self._sample(detector)
                    if not self._current(epoch):
                        self._discard_motion()
                    scale = self._budget.delay() if self._budget is not None else 1.0
                    if math.isfinite(scale):
                        period *= scale
                with self._condition:
                    if self._active and not self._closed:
                        self._condition.wait_for(lambda: self._closed or not self._active,
                                                 timeout=max(.001, period - (self._clock() - started)))
        except Exception:
            # Retain neither exception/traceback nor frames on failure.
            with self._condition:
                self._failed = True
                self._active = False
                self._pending = None
                self.wave_position = None
            log.warning("Wave stats failures=1")
        finally:
            with self._work:
                self._reset()
                stats, self._stats = self._stats, WaveStats()
            log.info("Wave stats frames=%d detector_runs=%d cues=%d max_score=%.3f frame_errors=%d",
                     stats.frames, stats.detector_runs, stats.cues, stats.max_score, stats.frame_errors)
