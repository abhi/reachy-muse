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
        "reads": 26, "person_runs_at": [2.4, 4.095], "cues_at": [4.415]}
    assert scheduled("still", .32) == {"reads": 16, "person_runs_at": [], "cues_at": []}
    assert scheduled("curtain", .32) == {
        "reads": 27, "person_runs_at": [2.0, 3.695], "cues_at": []}


def test_short_wave_waits_for_the_second_person_observation():
    from bench.wave import scheduled
    assert scheduled("wave", 0, 1001, seconds=1.5)["cues_at"] == [3.25]


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

        def observe(self, frame, boxes, now, *, confirmed=True):
            return confirmed

    monkeypatch.setattr(wave, "MotionGate", Gate)
    monkeypatch.setattr(wave, "WaveDetector", Cue)


def test_cooldown_survives_pauses_and_long_waves_until_three_quiet_seconds(immediate_motion, monkeypatch):
    now, moving = [0.0], [True]
    inferences = []

    class Gate:
        def update(self, frame, now):
            return SimpleNamespace(moving=moving[0], lighting=False)

    class People:
        def detect(self, frame):
            inferences.append(now[0])
            return [BOX]

    monkeypatch.setattr(wave, "MotionGate", Gate)
    watcher = wave.WaveWatcher(lambda: np.zeros((120, 160, 3), np.uint8), lambda: True,
                               clock=lambda: now[0])
    watcher.set_active(True)
    cues = []
    for at in [0, 1.25, 2, 3, 4, 5, 6, 7, 8, 8.5, 9.75]:
        now[0] = at
        if at == 2:
            watcher.set_active(False)
            watcher.set_active(True)
        moving[0] = at < 5 or at > 8
        watcher._sample(People())
        if watcher.take_wave():
            cues.append(at)
            assert watcher.take_wave() is False
    assert cues == [1.25, 9.75]
    assert inferences == [0, 1.25, 8.5, 9.75]


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
        assert watcher.wave_position is None
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


def test_audio_priority_prevents_camera_and_model_work():
    from reachy_muse.vision_budget import VisionBudget
    budget = VisionBudget()
    budget.audio_busy(True)
    events = []
    watcher = wave.WaveWatcher(lambda: events.append("camera"), lambda: True, budget=budget,
                               detector_factory=lambda _: events.append("model"))
    watcher.start()
    watcher.set_active(True)
    try:
        time.sleep(.15)
        assert events == []
    finally:
        watcher.close()


def test_motion_resets_do_not_reset_person_compute_deadline(immediate_motion):
    now = [0.0]
    calls = []

    class Person:
        def detect(self, frame):
            calls.append(now[0])
            now[0] += .32
            return [BOX]

    watcher = wave.WaveWatcher(lambda: np.zeros((120, 160, 3), np.uint8), lambda: True,
                               clock=lambda: now[0])
    watcher.set_active(True)
    for at in [0, .5, 1, 1.5, 2]:
        now[0] = at
        watcher._discard_motion()
        watcher._sample(Person())
    assert calls == [0, 2]
