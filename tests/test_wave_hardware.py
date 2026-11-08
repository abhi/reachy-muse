"""One reader and explicit priority for the morning camera watcher."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import math
import threading
import time
from types import SimpleNamespace

import pytest

from test_hardware import FakeRobot, np, pose_factory, start_controller
from reachy_muse.arrival import WatchPose
from reachy_muse.hardware import ReachyController
from reachy_muse import cli


def test_morning_watch_drains_wave_and_shares_the_camera_with_priority():
    events, readers = [], []

    class Watcher:
        failure = None

        def __init__(self, read, still, model, **kwargs):
            readers.append(read)
            self.name = "wave" if len(readers) == 1 else "arrival"

        def start(self):
            events.append((self.name, "start"))

        def set_active(self, active):
            events.append((self.name, active))

        set_watching = set_active

        def drain(self):
            events.append((self.name, "drain"))

        def take_wave(self):
            return True

        def close(self):
            events.append((self.name, "close"))

    robot = FakeRobot()
    robot.media.get_frame = lambda: "frame"
    controller = start_controller(robot, wave_wake=True, wave_watcher_factory=Watcher,
                                   arrival_model=Path("person.onnx"), arrival_watcher_factory=Watcher)
    try:
        controller.set_wave_watching(True)
        controller.start_watching(WatchPose())
        controller.set_wave_watching(True)
        assert controller.take_wave() is False
        assert [read() for read in readers] == ["frame", "frame"]
        controller.vision_budget.capture_active(True)
        controller.vision_budget.audio_busy(True)
        assert [read() for read in readers] == [None, "frame"]
        assert events == [("wave", "start"), ("wave", True), ("wave", False), ("wave", "drain"),
                          ("arrival", "start"), ("arrival", True), ("wave", False)]
        controller.stop_watching()
        controller.set_wave_watching(True)
        assert controller.take_wave() is True
    finally:
        controller.close()
    assert events[-2:] == [("wave", "close"), ("arrival", "close")]


def test_camera_reader_serializes_consumers_without_retaining_frames():
    robot = FakeRobot()
    inside = set()
    overlaps = []
    calls = []

    def read():
        thread = threading.get_ident()
        if inside:
            overlaps.append(True)
        inside.add(thread)
        time.sleep(.01)
        calls.append(1)
        inside.remove(thread)
        return len(calls)

    robot.media.get_frame = read
    controller = start_controller(robot)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: controller._read_camera_frame(), range(4)))
        assert sorted(results) == [1, 2, 3, 4]
        assert overlaps == []
    finally:
        controller.close()
    assert controller._read_camera_frame() is None


def test_wave_camera_is_opt_in_and_doctor_never_enables_it(monkeypatch):
    options = []
    monkeypatch.setattr("reachy_muse.hardware.ReachyController", lambda **kwargs: options.append(kwargs))
    for command in (["run"], ["run", "--wave-wake"], ["doctor"]):
        cli._hardware(cli.parser().parse_args(command))
    assert [option.get("wave_wake", False) for option in options] == [False, True, False]


def test_wave_can_authorize_realtime_without_an_acoustic_model(monkeypatch):
    monkeypatch.setattr("reachy_muse.realtime_call.realtime_supported", lambda: True)
    args = cli.parser().parse_args(["run", "--mode", "on-robot", "--muse-path", "realtime", "--wave-wake"])
    assert cli.realtime_path(args) == "realtime"


@pytest.mark.parametrize("x, yaw, expected", [(160, 0, 45), (480, 0, 135), (320, 20, 70)])
@pytest.mark.parametrize("delay, claimed", [(0, True), (.2, False)])
def test_wave_bearing_uses_current_head_pose_and_survives_camera_drain(monkeypatch, x, yaw, expected,
                                                                    delay, claimed):
    now = [1.0]
    monkeypatch.setattr("reachy_muse.hardware.time", SimpleNamespace(monotonic=lambda: now[0]))
    controller = ReachyController(wave_wake=True)
    controller._np, controller._robot = np, FakeRobot()
    controller._robot.pose = pose_factory(yaw=yaw)
    controller._robot.media.camera = SimpleNamespace(
        K=np.array([[160., 0, 320], [0, 160, 240], [0, 0, 1]]))

    class Watcher:
        wave_position = (x, 240)

        def take_wave(self):
            return True

        def set_active(self, active):
            self.wave_position = None

        def drain(self):
            pass

    controller._wave_watcher = Watcher()
    for tick in range(21):
        controller._update_wave_gaze(.5 + tick * .025, np.zeros(9), eligible=True)
    assert controller._wave_still() is True
    assert controller.take_wave() is True
    controller.set_wave_watching(False)
    now[0] += delay
    assert controller.claim_wave() is claimed
    assert controller.wave_direction == (pytest.approx(math.radians(expected)) if claimed else None)
    assert controller.claim_wave() is False
    assert controller.wave_direction is None


def test_head_motion_during_camera_drain_invalidates_the_claimed_wave():
    controller = ReachyController(wave_wake=True)
    controller._wave_watcher = SimpleNamespace(take_wave=lambda: True, wave_position=None)
    assert controller.take_wave() is True
    controller._wave_gaze.invalidate()
    assert controller.claim_wave() is False
    assert controller.wave_direction is None
