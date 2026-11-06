"""One reader and explicit priority for the morning camera watcher."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

from test_hardware import FakeRobot, start_controller
from reachy_muse.arrival import WatchPose
from reachy_muse import cli


def test_morning_watch_drains_wave_and_uses_the_same_camera_reader():
    events, readers = [], []

    class Watcher:
        failure = None

        def __init__(self, read, still, model):
            readers.append(read)
            self.name = "wave" if len(readers) == 1 else "arrival"

        def start(self):
            events.append((self.name, "start"))

        def set_active(self, active):
            events.append((self.name, active))

        set_watching = set_active

        def take_wave(self):
            return True

        def close(self):
            events.append((self.name, "close"))

    controller = start_controller(FakeRobot(), wave_wake=True, wave_watcher_factory=Watcher,
                                   arrival_model=Path("person.onnx"), arrival_watcher_factory=Watcher)
    try:
        controller.set_wave_watching(True)
        controller.start_watching(WatchPose())
        controller.set_wave_watching(True)
        assert controller.take_wave() is False
        assert readers[0] == readers[1]
        assert events == [("wave", "start"), ("wave", True), ("wave", False),
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
