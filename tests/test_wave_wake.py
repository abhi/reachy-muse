"""Literal gesture outcomes on pixels; no models, camera or network."""

import pytest

np = pytest.importorskip("numpy")

from bench.wave import BOX, NEGATIVES, clip
from reachy_muse.wave_wake import WaveDetector


def detections(kind, *, boxes=None, times=None):
    detector = WaveDetector()
    return [now for index, (now, frame, people) in enumerate(clip(kind))
            if detector.observe(frame, people if boxes is None else boxes,
                                now if times is None else times(index, now))]


def test_raised_hand_wakes_after_repeated_side_to_side_motion():
    assert detections("wave") == [2.125]


@pytest.mark.parametrize("kind", NEGATIVES)
def test_non_wave_motion_does_not_wake(kind):
    assert detections(kind) == []


def test_wave_requires_one_unambiguous_person():
    assert detections("wave", boxes=[]) == []
    assert detections("wave", boxes=[BOX, BOX]) == []


def test_sparse_or_out_of_order_frames_cannot_accumulate_a_wave():
    assert detections("wave", times=lambda index, now: index) == []
    assert detections("wave", times=lambda index, now: -now) == []


def test_robot_motion_reset_destroys_partial_wave():
    detector = WaveDetector()
    events = []
    for index, (now, frame, boxes) in enumerate(clip("wave")):
        if index % 6 == 0:
            detector.reset()
        if detector.observe(frame, boxes, now):
            events.append(now)
    assert events == []


def test_wave_replay_is_an_explicit_camera_only_limit():
    assert detections("replay") == [2.125]
