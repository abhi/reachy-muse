# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

import time

import pytest

np = pytest.importorskip("numpy")

from reachy_muse.arrival import (
    ArrivalDetector, ArrivalWatcher, PersonBox, YoloxPersonDetector, default_person_model,
)

FPS = 3
HEIGHT, WIDTH = 120, 160
PERSON = 250


def room(brightness=1.0, shift=0):
    texture = np.random.default_rng(7).integers(20, 160, (HEIGHT, WIDTH + 40, 3))
    return (texture[:, 20 + shift:20 + shift + WIDTH] * brightness).astype(np.uint8)


LAMP_OFF = np.linspace(0.6, 0.1, WIDTH)[None, :, None]


def with_person(frame, left, top=40, width=20, height=60):
    frame = frame.copy()
    frame[max(0, top):top + height, max(0, left):max(0, left + width)] = PERSON
    return frame


class MarkedPeople:
    def __init__(self, *fixed: PersonBox):
        self.fixed = list(fixed)
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        rows, cols = np.nonzero((frame == PERSON).all(axis=2))
        boxes = list(self.fixed)
        if len(rows):
            boxes.append(PersonBox(cols.min() / WIDTH, rows.min() / HEIGHT,
                                   (cols.max() + 1 - cols.min()) / WIDTH,
                                   (rows.max() + 1 - rows.min()) / HEIGHT, 0.9))
        return boxes


def first_arrival(frames, detector, *, moving=()):
    decision = ArrivalDetector(detector)
    for index, frame in enumerate(frames):
        now = index / FPS
        if index in moving:
            decision.reachy_moved()
            continue
        if decision.observe(frame, now):
            return now
    return None


def empty_room(seconds, **options):
    return [room(**options)] * (seconds * FPS)


def test_a_person_walking_in_from_the_side_is_an_arrival_two_thirds_of_a_second_after_entering():
    walk = [with_person(room(), -12 + 8 * step) for step in range(20)]
    assert first_arrival(empty_room(2) + walk, MarkedPeople()) == pytest.approx(8 / 3)


def test_a_person_walking_away_with_their_back_to_reachy_is_an_arrival():
    walk = [with_person(room(), 70 - step // 2, top=30 + step, width=26 - step, height=70 - 2 * step)
            for step in range(12)]
    assert first_arrival(empty_room(2) + walk, MarkedPeople()) == pytest.approx(8 / 3)


def test_a_photo_of_a_person_never_counts_even_while_a_curtain_moves_elsewhere():
    rng = np.random.default_rng(3)
    poster = PersonBox(0.1, 0.2, 0.15, 0.5, 0.95)
    frames = []
    for _ in range(30 * FPS):
        frame = room()
        frame[10:70, 120:150] = rng.integers(0, 255, (60, 30, 3))
        frames.append(frame)
    detector = MarkedPeople(poster)
    assert first_arrival(frames, detector) is None
    assert detector.calls > 20


def test_a_light_switching_on_with_only_a_photo_of_a_person_in_the_room_is_not_an_arrival():
    detector = MarkedPeople(PersonBox(0.1, 0.2, 0.15, 0.5, 0.95))
    assert first_arrival(empty_room(2, brightness=LAMP_OFF) + empty_room(10), detector) is None
    assert detector.calls == 24


def test_a_light_switching_on_then_a_person_walking_in_is_an_arrival_once_exposure_settles():
    walk = [with_person(room(), -12 + 8 * step) for step in range(20)]
    assert first_arrival(empty_room(2, brightness=LAMP_OFF) + walk, MarkedPeople()) == pytest.approx(10 / 3)


def test_a_dim_light_on_burst_brightens_detection_but_still_requires_local_motion():
    class DimPeople:
        def __init__(self):
            self.brightness = []

        def detect(self, frame):
            self.brightness.append(round(float(frame.mean()), 1))
            return [PersonBox(0.25, 0.25, 0.125, 0.5, 0.8)] if frame.mean() >= 60 else []

    dark = np.full((HEIGHT, WIDTH, 3), 5, dtype=np.uint8)
    dim = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
    detector = DimPeople()
    frames = [dark] * 6 + [dim] * 6
    assert first_arrival(frames, detector) is None
    assert detector.brightness == [80.0] * 6
    walk = []
    for step in range(6):
        frame = dim.copy()
        frame[30:90, 40 + 4 * step:60 + 4 * step] = 90
        walk.append(frame)
    assert first_arrival(frames + walk, DimPeople()) == pytest.approx(13 / 3)


@pytest.mark.parametrize("brightness", [0.2, 0.5, 1.0])
def test_light_on_with_a_stationary_person_box_never_triggers_at_different_exposures(brightness):
    detector = MarkedPeople(PersonBox(0.1, 0.2, 0.15, 0.5, 0.95))
    frames = empty_room(2, brightness=0.01) + empty_room(12, brightness=brightness)
    assert first_arrival(frames, detector) is None
    assert detector.calls == 24


def test_reachy_turning_its_head_is_not_an_arrival_and_the_gate_starts_over():
    poster = PersonBox(0.1, 0.2, 0.15, 0.5, 0.95)
    turning = [room(shift=step) for step in range(9)]
    frames = empty_room(2) + turning + empty_room(4, shift=8)
    moving = range(2 * FPS, 2 * FPS + len(turning))
    assert first_arrival(frames, MarkedPeople(poster), moving=moving) is None
    walk = [with_person(room(shift=8), -12 + 8 * step) for step in range(20)]
    assert first_arrival(frames + walk, MarkedPeople(poster), moving=moving) == pytest.approx(9 + 2 / 3)


def test_the_same_head_turn_without_the_reset_would_have_counted_the_photo():
    poster = PersonBox(0.1, 0.2, 0.15, 0.5, 0.95)
    turning = [room(shift=step) for step in range(9)]
    assert first_arrival(empty_room(2) + turning + empty_room(4, shift=8), MarkedPeople(poster)) is not None


def test_a_single_detection_flicker_is_not_an_arrival():
    class Once(MarkedPeople):
        def detect(self, frame):
            boxes = super().detect(frame)
            return boxes if self.calls == 1 else []

    detector = Once()
    frames = empty_room(2) + [with_person(room(), -12 + 8 * step) for step in range(20)]
    assert first_arrival(frames, detector) is None
    assert detector.calls == 10


def camera_with_frame_clock(frames):
    state = {"index": -1, "reads": 0}

    def read():
        state["reads"] += 1
        state["index"] = min(state["index"] + 1, len(frames) - 1)
        return frames[state["index"]]

    return read, lambda: max(0, state["index"]) / FPS, state


def wait_for(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "the arrival watcher did not reach the expected state"
        time.sleep(0.002)


def test_watcher_latches_the_arrival_and_stops_reading_frames_until_watching_starts_again():
    frames = empty_room(2) + [with_person(room(), -12 + 8 * step) for step in range(20)] + empty_room(3)
    read, clock, camera = camera_with_frame_clock(frames)
    watcher = ArrivalWatcher(read, lambda: True, sample_hz=10, clock=clock,
                             detector_factory=lambda model: MarkedPeople())
    try:
        watcher.start()
        watcher.set_watching(True)
        wait_for(lambda: watcher.arrival() is not None)
        assert watcher.arrival() == pytest.approx(8 / 3)
        reads = camera["reads"]
        time.sleep(0.3)
        assert camera["reads"] == reads == 9
        watcher.set_watching(False)
        assert watcher.arrival() == pytest.approx(8 / 3)
        watcher.set_watching(True)
        assert watcher.arrival() is None
    finally:
        watcher.close()


def test_watcher_reads_no_frames_until_watching_nor_while_reachy_moves():
    reads = []
    still = [False]
    watcher = ArrivalWatcher(lambda: reads.append(1) or room(), lambda: still[0], sample_hz=10,
                             detector_factory=lambda model: MarkedPeople())
    try:
        watcher.start()
        time.sleep(0.15)
        assert reads == []
        watcher.set_watching(True)
        time.sleep(0.15)
        assert reads == []
        still[0] = True
        wait_for(lambda: len(reads) >= 2)
        assert watcher.arrival() is None and watcher.failure is None
    finally:
        watcher.close()


def test_watcher_failure_is_reported_without_retaining_the_frame(caplog):
    import gc
    import weakref

    references = []

    class Broken:
        def detect(self, frame):
            raise RuntimeError("Detection unavailable")

    frames = empty_room(2) + [with_person(room(), 10 * step) for step in range(10)]
    read_script, clock, _ = camera_with_frame_clock(frames)

    def read_tracked():
        frame = read_script().copy()
        references.append(weakref.ref(frame))
        return frame

    watcher = ArrivalWatcher(read_tracked, lambda: True, sample_hz=10, clock=clock,
                             detector_factory=lambda model: Broken())
    watcher.start()
    watcher.set_watching(True)
    wait_for(lambda: watcher.failure is not None)
    watcher.close()
    gc.collect()
    assert str(watcher.failure) == "Detection unavailable"
    assert watcher.failure.__traceback__ is None and watcher.failure.__context__ is None
    assert all(reference() is None for reference in references)
    assert [record.getMessage() for record in caplog.records] == ["Arrival watcher failed: RuntimeError"]


def test_frame_read_errors_recover_and_diagnostics_contain_only_interval_numbers(caplog):
    base = np.full((HEIGHT, WIDTH, 3), 20, dtype=np.uint8)
    frames = [base] * 6 + [with_person(base, -12 + 8 * step) for step in range(3)]
    read_script, clock, _ = camera_with_frame_clock(frames)
    reads = 0

    def read():
        nonlocal reads
        reads += 1
        if reads == 1:
            raise OSError("private camera details")
        return None if reads == 2 else read_script()

    watcher = ArrivalWatcher(read, lambda: True, sample_hz=10, clock=clock,
                             detector_factory=lambda model: MarkedPeople())
    try:
        watcher.start()
        watcher.set_watching(True)
        wait_for(lambda: watcher.arrival() is not None)
        watcher.set_watching(True)
        assert watcher.arrival() == pytest.approx(8 / 3)
        stats = watcher.take_stats()
        assert (stats.frames, stats.frame_errors, stats.motion_triggers, stats.detector_runs,
                stats.max_person_confidence) == (9, 2, 3, 2, 0.9)
        assert stats.brightness == pytest.approx(23.5138889)
        reset = watcher.take_stats()
        assert (reset.frames, reset.frame_errors, reset.brightness, reset.detector_runs,
                reset.motion_triggers, reset.max_person_confidence) == (0, 0, 0.0, 0, 0, 0.0)
        assert watcher.failure is None
        assert "private camera details" not in caplog.text
    finally:
        watcher.close()


def test_watcher_detects_a_walker_with_every_fourth_frame_missing():
    frames = empty_room(2) + [with_person(room(), -12 + 8 * step) for step in range(20)]
    frames = [None if index % 4 == 3 else frame for index, frame in enumerate(frames)]
    read, clock, camera = camera_with_frame_clock(frames)
    watcher = ArrivalWatcher(read, lambda: True, sample_hz=10, clock=clock,
                             detector_factory=lambda model: MarkedPeople())
    try:
        watcher.start()
        watcher.set_watching(True)
        wait_for(lambda: watcher.arrival() is not None or camera["index"] == len(frames) - 1)
    finally:
        watcher.close()
    assert watcher.arrival() == pytest.approx(8 / 3)


def test_watcher_settles_again_after_a_long_gap_in_camera_frames():
    frames = empty_room(2) + [with_person(room(), 10)] + [None] * 9
    frames += [with_person(room(), 70)] * (4 * FPS)
    read, clock, camera = camera_with_frame_clock(frames)
    watcher = ArrivalWatcher(read, lambda: True, sample_hz=10, clock=clock,
                             detector_factory=lambda model: MarkedPeople())
    try:
        watcher.start()
        watcher.set_watching(True)
        wait_for(lambda: watcher.arrival() is not None or camera["index"] == len(frames) - 1)
    finally:
        watcher.close()
    assert watcher.arrival() is None
    assert camera["index"] == len(frames) - 1
    assert watcher.failure is None


def test_a_missing_person_model_is_a_watcher_failure_not_a_crash(tmp_path):
    watcher = ArrivalWatcher(room, lambda: True, tmp_path / "absent.onnx", sample_hz=10,
                             detector_factory=lambda model: YoloxPersonDetector(model))
    watcher.start()
    watcher.set_watching(True)
    wait_for(lambda: watcher.failure is not None)
    watcher.close()
    assert isinstance(watcher.failure, (FileNotFoundError, ImportError))
    assert watcher.arrival() is None


def test_packaged_person_model_is_hash_pinned_and_apache_licensed():
    import hashlib
    import json

    model = default_person_model()
    manifest = json.loads((model.parent / "SOURCE.json").read_text())
    assert model.stat().st_size == 3659407
    assert hashlib.sha256(model.read_bytes()).hexdigest() == manifest["model_sha256"]
    assert manifest["model_sha256"] == "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"
    assert "Apache License" in (model.parent / "LICENSE").read_text()[:200]


def test_packaged_person_model_sees_nobody_in_a_blank_frame():
    pytest.importorskip("cv2")
    detector = YoloxPersonDetector(default_person_model())
    assert detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8)) == []


@pytest.mark.parametrize("rate", [0, 0.2, 11, float("inf"), float("nan")])
def test_unbounded_watch_rate_is_rejected(rate):
    with pytest.raises(ValueError, match="watch rate"):
        ArrivalWatcher(room, lambda: True, sample_hz=rate)
