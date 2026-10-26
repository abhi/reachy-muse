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
    assert detector.calls == 5


def test_a_light_switching_on_then_a_person_walking_in_is_an_arrival_once_exposure_settles():
    walk = [with_person(room(), -12 + 8 * step) for step in range(20)]
    assert first_arrival(empty_room(2, brightness=LAMP_OFF) + walk, MarkedPeople()) == pytest.approx(4)


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


def test_watcher_failure_is_reported_without_retaining_the_frame():
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
