"""Wave worker scheduling, suspension and stale-cue behavior."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

import reachy_muse.wave_wake as wave
from bench.wave import BOX


def until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(.002)


def test_periodic_wave_survives_idle_sampling_and_slow_person_inference():
    from bench.wave import scheduled
    assert scheduled("wave", .32) == {
        "reads": 17, "person_runs_at": [2.4, 3.72], "cues_at": [4.165]}
    assert scheduled("still", .32) == {"reads": 16, "person_runs_at": [], "cues_at": []}
    assert scheduled("curtain", .32) == {
        "reads": 26, "person_runs_at": [2.0, 3.32, 4.64], "cues_at": []}


def test_inactive_or_moving_robot_never_reads_camera():
    reads = []
    watcher = wave.WaveWatcher(lambda: reads.append(1), lambda: False,
                               detector_factory=lambda _: object())
    try:
        watcher.start()
        time.sleep(.03)
        assert reads == []
        watcher.set_active(True)
        time.sleep(.03)
        assert reads == []
        assert watcher.take_wave() is False
    finally:
        watcher.close()


def test_disabling_waits_for_in_flight_work_before_returning():
    entered, release = threading.Event(), threading.Event()

    def load(_):
        entered.set()
        assert release.wait(3)
        return object()

    watcher = wave.WaveWatcher(lambda: None, lambda: True, detector_factory=load)
    watcher.start()
    watcher.set_active(True)
    try:
        assert entered.wait(2)
        with ThreadPoolExecutor() as executor:
            disabled = executor.submit(watcher.set_active, False)
            time.sleep(.03)
            assert disabled.done() is False
            release.set()
            assert disabled.result(2) is None
        assert watcher.take_wave() is False
    finally:
        release.set()
        watcher.close()


@pytest.fixture
def immediate_motion(monkeypatch):
    class Gate:
        def update(self, frame, now):
            return SimpleNamespace(moving=True, lighting=False)

    class Cue:
        score = 1.0

        def observe(self, frame, boxes, now):
            return True

    monkeypatch.setattr(wave, "MotionGate", Gate)
    monkeypatch.setattr(wave, "WaveDetector", Cue)


def test_cooldown_survives_disable_and_enable_and_cue_is_consumed_once(immediate_motion):
    now = [0.0]
    reads, inferences = [], []

    def read():
        now[0] += .25
        reads.append(now[0])
        return np.zeros((120, 160, 3), np.uint8)

    class People:
        def detect(self, frame):
            inferences.append(now[0])
            return [BOX]

    watcher = wave.WaveWatcher(read, lambda: True, clock=lambda: now[0],
                               detector_factory=lambda _: People())
    try:
        watcher.start()
        watcher.set_active(True)
        until(watcher.take_wave)
        assert watcher.take_wave() is False
        assert inferences == [.25, 1.5]
        assert reads == [.25, .5, .75, 1.0, 1.25, 1.5]
        watcher.set_active(False)
        now[0] = 16.49
        watcher.set_active(True)
        time.sleep(.03)
        assert len(reads) == 6
        assert watcher.take_wave() is False
        watcher.set_active(False)
        now[0] = 16.5
        watcher.set_active(True)
        until(watcher.take_wave)
        assert inferences == [.25, 1.5, 16.75, 18.0]
    finally:
        watcher.close()


def test_old_cue_expires_instead_of_waking_a_later_conversation(immediate_motion):
    now = [0.0]
    calls = []

    def read():
        now[0] += .25
        return np.zeros((120, 160, 3), np.uint8)

    class People:
        def detect(self, frame):
            calls.append(now[0])
            return [BOX]

    watcher = wave.WaveWatcher(read, lambda: True, clock=lambda: now[0],
                               detector_factory=lambda _: People())
    try:
        watcher.start()
        watcher.set_active(True)
        until(lambda: len(calls) == 2)
        time.sleep(.02)
        now[0] += .76
        assert watcher.take_wave() is False
        assert watcher.take_wave() is False
    finally:
        watcher.close()


def test_failure_logs_only_counts_and_releases_camera_frame(immediate_motion, caplog):
    import gc
    import weakref
    references = []

    def read():
        frame = np.zeros((120, 160, 3), np.uint8)
        references.append(weakref.ref(frame))
        return frame

    class Broken:
        def detect(self, frame):
            raise RuntimeError("private payload that must not be logged")

    watcher = wave.WaveWatcher(read, lambda: True, detector_factory=lambda _: Broken())
    watcher.start()
    watcher.set_active(True)
    until(lambda: bool(caplog.records))
    watcher.close()
    gc.collect()
    assert [record.getMessage() for record in caplog.records] == ["Wave stats failures=1"]
    assert all(reference() is None for reference in references)
