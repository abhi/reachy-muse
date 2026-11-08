"""Wave greetings use the existing motion writer and yield to speech."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_muse.hardware import ReachyController
from reachy_muse.motion import MotionStyle
from reachy_muse.motion.wave_back import wave_back_offset
from test_face_tracking_motion import drive


@pytest.mark.parametrize("state", ["idle", "listening", "sleeping"])
def test_wave_is_issued_once_and_ignored_inside_cooldown(monkeypatch, state):
    now = [10.0]
    monkeypatch.setattr("reachy_muse.hardware.time", SimpleNamespace(monotonic=lambda: now[0]))
    controller = ReachyController(wave_wake=True)
    controller.set_state(state)
    controller.gesture("wave")
    assert (controller._gesture_name, controller._gesture_started) == ("wave", 10.0)
    now[0] = 14.59
    controller.gesture("wave")
    assert controller._gesture_started == 10.0
    now[0] = 14.61
    controller.gesture("wave")
    assert controller._gesture_started == 14.61


def test_wave_during_speech_is_dropped_and_never_queued(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("reachy_muse.hardware.time", SimpleNamespace(monotonic=lambda: now[0]))
    controller = ReachyController(wave_wake=True)
    controller.set_state("speaking")
    controller.gesture("wave")
    assert controller._gesture_name is None
    controller.set_state("idle")
    assert controller._gesture_name is None
    controller.gesture("wave")
    assert controller._gesture_name == "wave"
    controller.set_state("speaking")
    assert controller._gesture_name is None


def test_procedural_wave_has_literal_rest_endpoints_and_both_antennas_move():
    assert wave_back_offset(0) == (0.0,) * 9
    assert wave_back_offset(1.6) == (0.0,) * 9
    assert wave_back_offset(2) == (0.0,) * 9
    middle = wave_back_offset(.8)
    assert middle[:6] == pytest.approx((0, 0, 3, 5, -3, 0))
    offsets = np.array([wave_back_offset(t) for t in np.arange(0, 1.601, .025)])
    assert np.ptp(offsets[:, 6]) > 40
    assert np.array_equal(offsets[:, 6], offsets[:, 7])
    assert offsets[:, 5].tolist() == [0.0] * 65


def test_still_mode_waves_at_the_current_gaze_and_returns_to_it(monkeypatch):
    antennas, before, still, after = [], [], [], []

    def action(controller, follower, tick):
        if tick == 70:
            before.append(controller._pose_parameters(controller._robot.pose))
            controller.gesture("wave")
        if 70 < tick < 134:
            antennas.append(np.degrees(controller._robot.antennas))
            still.append(controller._wave_still())
        if tick in (140, 150, 170):
            after.append(controller._wave_still())

    controller, _, samples = drive(monkeypatch, action, ticks=180, wave_wake=True,
                                    face_follow_idle=True, style=MotionStyle(breathing=False))
    assert set(still) == {False}
    assert after == [False, False, True]
    assert np.ptp(np.array(antennas), axis=0).tolist() == pytest.approx([44.8, 44.8], abs=1)
    assert samples[100]["yaw"] == pytest.approx(8, abs=.01)
    assert samples[100]["z"] > 2
    assert samples[180] == pytest.approx(before[0], abs=.03)
    assert math.degrees(controller._last_body_yaw) == 0
