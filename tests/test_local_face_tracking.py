# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from dataclasses import replace
import math
import threading
import time

import pytest

from reachy_muse.local_face_tracking import (
    FACE_LOSS_SECONDS, PITCH_LIMIT, YAW_LIMIT, FaceCapture, FaceFollower, _Face, _FaceTracker,
    default_face_model,
)

LEVEL = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def face(x=0.4, y=0.3, width=0.2, height=0.3, yaw=0, pitch=0):
    return _Face((x, y, width, height), math.radians(yaw), math.radians(pitch), 0.95)


def wait_for(condition, timeout=1):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            raise AssertionError('Face detector did not reach the expected state')
        time.sleep(0.002)


def test_target_uses_current_head_pose_and_deadband():
    tracker = _FaceTracker(0.2)
    target = tracker.update([face(yaw=0.5, pitch=-0.5)],
                            (math.radians(8), math.radians(-4)), 1)
    assert target.yaw == pytest.approx(math.radians(8))
    assert target.pitch == pytest.approx(math.radians(-4))
    target = tracker.update([face(yaw=8, pitch=5)],
                            (math.radians(10), math.radians(-2)), 1.2)
    assert math.radians(8) < target.yaw < math.radians(18)
    assert math.radians(-4) < target.pitch < math.radians(3)
    assert target.observed_at == 1.2


def test_extreme_targets_stay_within_bounds_and_slew_rate():
    tracker = _FaceTracker(0.2)
    previous = (0.0, 0.0)
    for index in range(30):
        target = tracker.update([face(yaw=120, pitch=-90)], previous, 1 + index * 0.2)
        assert abs(target.yaw) <= YAW_LIMIT
        assert abs(target.pitch) <= PITCH_LIMIT
        assert abs(target.yaw - previous[0]) <= math.radians(7) + 1e-12
        assert abs(target.pitch - previous[1]) <= math.radians(5) + 1e-12
        previous = target.yaw, target.pitch
    assert target.yaw == pytest.approx(YAW_LIMIT, abs=1e-7)
    assert target.pitch == pytest.approx(-PITCH_LIMIT, abs=1e-7)


def test_hysteresis_keeps_existing_face_when_larger_face_arrives():
    tracker = _FaceTracker(0.2)
    first = face(x=0.1, width=0.15, yaw=8)
    other = face(x=0.65, width=0.3, yaw=-15)
    tracker.update([first], (0, 0), 1)
    target = tracker.update([other, face(x=0.11, width=0.15, yaw=8)], (0, 0), 1.2)
    assert target.yaw > 0


def test_largest_initial_face_is_selected_without_identity_recognition():
    target = _FaceTracker(0.2).update(
        [face(width=0.1, yaw=-8), face(x=0.65, width=0.3, yaw=8)], (0, 0), 1)
    assert target.yaw > 0


def test_unmatched_faces_do_not_extend_loss_timeout_and_reacquire_after_loss():
    tracker = _FaceTracker(0.2)
    original = tracker.update([face(x=0.1, width=0.1, yaw=8)], (0, 0), 1)
    different = face(x=0.75, width=0.2, yaw=-8)
    assert tracker.update([different], (0, 0), 1.2) == original
    assert tracker.update([], (0, 0), 1.6) == original
    assert tracker.update([], (0, 0), 1 + FACE_LOSS_SECONDS) is None
    assert tracker.update([different], (0, 0), 2).yaw < 0


@pytest.mark.parametrize('values', [(float('nan'), 0), (0, float('inf'))])
def test_invalid_head_snapshot_never_becomes_a_motion_target(values):
    assert _FaceTracker(0.2).update([face(yaw=8)], values, 1) is None


def test_default_packaged_model_is_small_and_hash_pinned():
    import hashlib
    import json

    model = default_face_model()
    manifest = json.loads((model.parent / 'SOURCE.json').read_text())
    assert model.stat().st_size == 232589
    assert hashlib.sha256(model.read_bytes()).hexdigest() == manifest['model_sha256']
    assert manifest['model_sha256'] == '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
    assert (model.parent / 'LICENSE').read_text().startswith('MIT License')


class Detector:
    def __init__(self, *args):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [face(yaw=10)]


def captures(read_frame, clock=time.monotonic):
    """A camera owner's captures of ``read_frame`` with the head level and facing forward."""
    def capture(*, background=True):
        frame = read_frame()
        return FaceCapture(frame, (0.0, 0.0), LEVEL, clock(), (0.0, 0.0)) if frame is not None else None
    return capture


def follower(read_frame, **kwargs):
    return FaceFollower(captures(read_frame, kwargs.get('clock', time.monotonic)),
                        camera_matrix=None, sample_hz=15, detector_factory=Detector, **kwargs)


def test_worker_is_inactive_until_enabled_and_clears_target_on_close():
    reads = []
    item = follower(lambda: reads.append('camera') or object())
    try:
        item.start()
        item.start()
        time.sleep(0.02)
        assert reads == []
        item.set_active(True)
        wait_for(lambda: item.target() is not None)
        assert item.target().yaw > 0
        item.set_active(False)
        assert item.target() is None
        count = len(reads)
        time.sleep(0.08)
        assert len(reads) == count
    finally:
        item.close()
    assert not item._thread.is_alive()
    assert item.target() is None


def test_deactivation_rejects_inflight_frame_result():
    entered, release = threading.Event(), threading.Event()

    def read():
        entered.set()
        assert release.wait(1)
        return object()

    item = follower(read)
    try:
        item.start()
        item.set_active(True)
        assert entered.wait(1)
        item.set_active(False)
        release.set()
        time.sleep(0.02)
        assert item.target() is None
    finally:
        release.set()
        item.close()


def test_old_session_target_is_not_reused_after_reactivation():
    reads = []
    item = follower(lambda: reads.append(1) or (object() if len(reads) == 1 else None))
    try:
        item.start()
        item.set_active(True)
        wait_for(lambda: item.target() is not None)
        item.set_active(False)
        item.set_active(True)
        wait_for(lambda: len(reads) >= 2)
        assert item.target() is None
    finally:
        item.close()


def test_no_new_camera_frame_expires_target_without_retaining_images():
    current = [1.0]
    reads = []
    item = follower(lambda: reads.append(1) or (object() if len(reads) == 1 else None),
                    clock=lambda: current[0])
    try:
        item.start()
        item.set_active(True)
        wait_for(lambda: item.target() is not None)
        current[0] += FACE_LOSS_SECONDS
        assert item.target() is None
        assert not any(name in vars(item) for name in ('frame', '_frame', 'image', '_image'))
    finally:
        item.close()


def test_camera_failure_is_reported_without_publishing_pose_or_retrying_forever():
    reads = []

    def failed():
        reads.append(1)
        raise RuntimeError('Camera unavailable')

    item = follower(failed)
    item.start()
    item.set_active(True)
    wait_for(lambda: item.failure is not None)
    assert isinstance(item.failure, RuntimeError)
    assert item.target() is None
    item.close()
    assert reads == [1]


def test_detector_failure_does_not_retain_camera_frame_in_traceback():
    import gc
    import weakref

    references = []

    class Frame:
        pass

    class FailedDetector:
        def __init__(self, *args):
            pass

        def detect(self, frame):
            raise RuntimeError('Detection unavailable')

    def read():
        frame = Frame()
        references.append(weakref.ref(frame))
        return frame

    item = FaceFollower(captures(read), camera_matrix=None, detector_factory=FailedDetector)
    item.start()
    item.set_active(True)
    wait_for(lambda: item.failure is not None)
    item.close()
    gc.collect()
    assert item.failure.__traceback__ is None
    assert item.failure.__context__ is None
    assert references[0]() is None


def test_missing_camera_calibration_fails_before_worker_starts(tmp_path):
    np = pytest.importorskip('numpy')
    pytest.importorskip('cv2')
    item = FaceFollower(captures(lambda: None), camera_matrix=None)
    with pytest.raises(ValueError, match='calibrated camera matrix'):
        item.start()
    assert item._thread is None


@pytest.mark.parametrize('rate', [0, 0.5, 16, float('inf'), float('nan')])
def test_unbounded_detector_rate_is_rejected(rate):
    with pytest.raises(ValueError, match='sample rate'):
        FaceFollower(captures(lambda: None), camera_matrix=None, sample_hz=rate)


def test_idle_sampling_uses_less_camera_work_and_wake_preserves_the_target():
    reads = []
    item = follower(lambda: reads.append(time.monotonic()) or object())
    try:
        item.start()
        item.set_active(True, idle=True)
        wait_for(lambda: item.target() is not None)
        time.sleep(.6)
        assert len(reads) == 2
        assert reads[1] - reads[0] >= .45
        target = item.target()
        assert target is not None
        item.set_active(True)
        assert item.target() is not None
        wait_for(lambda: len(reads) >= 5)
        assert reads[4] - reads[2] < .4
        item.set_active(False)
        assert item.target() is None
    finally:
        item.close()


def test_throttled_idle_gaze_keeps_the_same_face_between_samples():
    from types import SimpleNamespace

    now = [0.0]
    reads = []
    first = face(x=.1, width=.15, yaw=8)
    other = face(x=.65, width=.3, yaw=-15)

    class People:
        def __init__(self, *args):
            pass

        def detect(self, frame):
            return [first] if len(reads) == 1 else [other, first]

    item = FaceFollower(captures(lambda: reads.append(now[0]) or object(), lambda: now[0]),
                        camera_matrix=None, detector_factory=People, clock=lambda: now[0],
                        budget=SimpleNamespace(delay=lambda: 3))
    try:
        item.start()
        item.set_active(True, idle=True)
        wait_for(lambda: item.target() is not None)
        original = item.target()
        for at in (.8, 1.4):
            now[0] = at
            assert item.target() == original
        now[0] = 1.5
        wait_for(lambda: item.target() is not None and item.target().observed_at == 1.5, timeout=2)
        assert item.target().yaw > 0
        now[0] = 4
        assert item.target() is None
    finally:
        item.close()


def test_slow_detection_keeps_target_while_next_inference_runs():
    now = [0.0]
    entered, release = threading.Event(), threading.Event()

    class SlowDetector(Detector):
        def detect(self, frame):
            observed_at = now[0]
            if self.calls:
                entered.set()
                assert release.wait(2)
            now[0] = observed_at + .32
            return super().detect(frame)

    item = FaceFollower(captures(lambda: object(), lambda: now[0]),
                        camera_matrix=None, detector_factory=SlowDetector,
                        clock=lambda: now[0])
    try:
        item.start()
        item.set_active(True)
        wait_for(lambda: item.target() is not None)
        original = item.target()
        assert original.observed_at == 0
        now[0] = 1.6
        assert entered.wait(2)
        now[0] = 1.84
        assert item.target() == original
        release.set()
        wait_for(lambda: item.target() is not None and item.target().observed_at == 1.6)
        now[0] = 4
        assert item.target() is None
    finally:
        release.set()
        item.close()


def test_all_faces_keep_capture_pose_and_time_while_budget_pauses(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from reachy_muse.hardware import ReachyController
    from test_hardware import FakeRobot, pose_factory

    clock = [100.0]
    robot = FakeRobot()
    robot.pose = pose_factory(yaw=30, pitch=10, roll=20)
    robot.media.camera = SimpleNamespace(K=np.array([[900., 0, 640], [0, 900, 360], [0, 0, 1]]), D=np.zeros(5))
    paused = [False]
    budget = SimpleNamespace(delay=lambda: math.inf if paused[0] else 1.0)
    captures, calls = [], []

    def frame():
        assert controller._camera_lock.locked()
        captures.append(threading.get_ident())
        return np.zeros((10, 10, 3), dtype=np.uint8)

    robot.media.get_frame = frame

    class Detect:
        def detect(self, frame):
            calls.append(threading.get_ident())
            robot.pose = pose_factory(yaw=-30)
            paused[0] = True
            return [face(yaw=0, pitch=0), face(x=.7, width=.1, yaw=-30, pitch=5)]

    def factory(*args, **kwargs):
        return FaceFollower(*args, **kwargs, detector_factory=lambda *args: Detect(), clock=lambda: clock[0])

    controller = ReachyController(face_follow_model='fake', face_follower_factory=factory)
    controller._robot, controller._np, controller.vision_budget = robot, np, budget
    monkeypatch.setattr('reachy_muse.hardware.time', SimpleNamespace(monotonic=lambda: clock[0]))
    controller._start_face_follow()
    follower = controller._face_follower
    try:
        controller._update_face_tracking('listening')
        wait_for(lambda: controller.face_snapshot() is not None)
        snapshot = controller.face_snapshot()
        assert snapshot.captured_at == 100.0
        assert len(snapshot.faces) == 2
        first, second = snapshot.faces
        assert (math.degrees(first.yaw), math.degrees(first.pitch)) == pytest.approx((30, 10))
        assert (math.degrees(second.yaw), math.degrees(second.pitch)) == pytest.approx((1.460177, 23.468076), abs=.00001)
        assert (first.box, first.score, first.captured_at) == ((.4, .3, .2, .3), .95, 100.0)
        assert (math.degrees(snapshot.heading), math.degrees(snapshot.elevation)) == pytest.approx((30, 10))
        assert [math.degrees(angle) for angle in snapshot.half_view] == pytest.approx([35.417, 21.801], abs=.001)
        assert snapshot.covers(replace(first, yaw=math.radians(64), pitch=math.radians(-11)))
        assert not snapshot.covers(replace(first, yaw=math.radians(-6)))
        assert not snapshot.covers(replace(first, pitch=math.radians(-12)))
        clock[0] = 103.0
        wait_for(lambda: follower.target() is None)
        assert controller.face_snapshot() == snapshot
        refreshed = controller.refresh_faces()
        assert refreshed.captured_at == 103.0
        assert math.degrees(refreshed.faces[0].yaw) == pytest.approx(-30)
        assert len(calls) == len(captures) == 2
        assert set(calls) == set(captures)
        assert calls[0] != threading.get_ident()
    finally:
        follower.close()


def test_refresh_deadline_does_not_queue_more_inference_or_drop_old_snapshot():
    entered, release = threading.Event(), threading.Event()
    calls = []

    class Detect:
        def detect(self, frame):
            calls.append(frame)
            if len(calls) > 1:
                entered.set()
                assert release.wait(2)
            return [face()]

    item = FaceFollower(captures(lambda: object()), camera_matrix=None, detector_factory=lambda *args: Detect())
    item.start()
    try:
        first = item.refresh()
        assert len(first.faces) == 1
        started = time.monotonic()
        assert item.refresh(.03) is None
        assert time.monotonic() - started < .2
        assert entered.is_set()
        assert item.snapshot() == first
        assert item.refresh(.03) is None
        assert len(calls) == 2
    finally:
        release.set()
        item.close()


def test_seen_faces_keep_their_capture_times_within_a_bound(monkeypatch):
    monkeypatch.setattr('reachy_muse.local_face_tracking.SEEN_FACES', 3)
    now = [10.0]

    class Two:
        def detect(self, frame):
            return [face(yaw=-10), face(x=.7, yaw=20)]

    item = FaceFollower(captures(lambda: object(), lambda: now[0]), camera_matrix=None,
                        detector_factory=lambda *args: Two(), clock=lambda: now[0])
    item.start()
    try:
        assert item.refresh() is not None
        now[0] = 20.0
        assert item.refresh() is not None
        assert [(face.captured_at, round(math.degrees(face.yaw))) for face in item.seen_since(0)] == [
            (10.0, 20), (20.0, -10), (20.0, 20)]
        assert [face.captured_at for face in item.seen_since(15)] == [20.0, 20.0]
    finally:
        item.close()


def test_recent_views_keep_empty_passes_and_expire_within_the_same_bound(monkeypatch):
    monkeypatch.setattr('reachy_muse.local_face_tracking.SEEN_FACES', 3)
    now = [10.0]

    class Detector:
        def detect(self, frame):
            return [] if now[0] == 20.0 else [face(yaw=-10), face(x=.7, yaw=20)]

    item = FaceFollower(captures(lambda: object(), lambda: now[0]), camera_matrix=None,
                        detector_factory=lambda *args: Detector(), clock=lambda: now[0])
    item.start()
    try:
        for captured_at in (10.0, 20.0, 30.0, 40.0):
            now[0] = captured_at
            assert item.refresh() is not None
        assert [(view.captured_at, tuple(round(math.degrees(face.yaw)) for face in view.faces))
                for view in item.views_since(0)] == [(20.0, ()), (30.0, (-10, 20)), (40.0, (-10, 20))]
        assert [view.captured_at for view in item.views_since(35)] == [40.0]
    finally:
        item.close()
