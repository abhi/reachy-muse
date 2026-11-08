# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest

import reachy_muse.arrival
import reachy_muse.hardware
from musegadget.identity import Identity
from reachy_muse.arrival import PersonBox
from reachy_muse.conversation import ReachyService
from reachy_muse.hardware import ReachyController
from reachy_muse.presence import Presence


def test_observations_expire_but_keep_rough_last_seen_age():
    state = Presence()
    assert state.result(50) == {"ok": True, "present": None, "last_seen_seconds": None}
    state = state.observe(True, 50)
    assert state.result(51) == {"ok": True, "present": True, "last_seen_seconds": 1}
    assert state.result(53.4) == {"ok": True, "present": None, "last_seen_seconds": 3}
    state = state.observe(False, 54)
    assert state.result(55) == {"ok": True, "present": False, "last_seen_seconds": 5}
    state = state.observe(None, 56)
    assert state.result(56) == {"ok": True, "present": None, "last_seen_seconds": 6}


def test_camera_check_reads_one_frame_and_retains_no_images(monkeypatch):
    import gc
    import weakref
    np = pytest.importorskip("numpy")
    reads, models, references = [], [], []
    now = [100.0]
    monkeypatch.setattr(reachy_muse.hardware.time, "monotonic", lambda: now[0])

    def frame():
        reads.append(True)
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        references.append(weakref.ref(image))
        return image

    class Detector:
        def __init__(self, model):
            models.append(model)

        def detect(self, image):
            assert image.shape == (24, 32, 3)
            return [PersonBox(0, 0, .2, .3, .9)] if now[0] == 100 else []

    monkeypatch.setattr(reachy_muse.arrival, "YoloxPersonDetector", Detector)
    controller = ReachyController(enable_motion=False)
    controller._robot = SimpleNamespace(media=SimpleNamespace(get_frame=frame))
    assert controller.presence() == {"ok": True, "present": True, "last_seen_seconds": 0}
    assert reads == [True]
    now[0] = 110.2
    assert controller.presence() == {"ok": True, "present": False, "last_seen_seconds": 10}
    assert reads == [True, True] and len(models) == 1
    gc.collect()
    assert all(reference() is None for reference in references)
    assert controller._state == "idle" and controller._commands.empty()


def test_presence_command_reads_camera_while_a_conversation_turn_is_running(monkeypatch):
    import asyncio
    import math
    from test_device_command_turns import rig_with, run, service
    from test_pins import Transcripts
    from test_wave_conversation import continuous_microphone

    rig = rig_with(monkeypatch)
    continuous_microphone(rig.hardware)
    reads, results = [], []
    controller = ReachyController(enable_motion=False)
    controller._robot = SimpleNamespace(media=SimpleNamespace(get_frame=lambda: reads.append(1) or object()))
    controller._presence_detector = SimpleNamespace(detect=lambda _: [PersonBox(0, 0, .2, .3, .9)])
    rig.hardware.vision_budget = controller.vision_budget
    rig.hardware.presence = controller.presence

    async def muse(session, request):
        assert conversation._voice[1]._turn_active
        await rig.wait_until_or_timeout(lambda: controller.vision_budget.delay() == math.inf)
        assert controller._read_camera_frame() is None
        results.append(await asyncio.to_thread(conversation.run_command, "reachy.presence", {}, 500))
        await session.answer(request, "Hello there.")

    conversation = service(rig, muse, Transcripts("Is anyone here?"))

    async def user():
        rig.say()

    run(rig, conversation, user)
    assert results == [{"ok": True, "payload": {"present": True, "last_seen_seconds": 0}}]
    assert reads == [1]


@pytest.mark.parametrize("found, last_seen_seconds", [(True, 4), (False, 14)])
def test_delayed_detection_expires_the_frame_and_keeps_its_last_seen_age(monkeypatch, found, last_seen_seconds):
    now = [100.0]
    monkeypatch.setattr(reachy_muse.hardware.time, "monotonic", lambda: now[0])

    class Detector:
        def detect(self, image):
            now[0] = 103.5
            return [PersonBox(0, 0, .2, .3, .9)] if found else []

    controller = ReachyController(enable_motion=False)
    controller._robot = SimpleNamespace(media=SimpleNamespace(get_frame=object))
    controller._presence_detector = Detector()
    controller._presence = Presence(True, 90, 90)
    assert controller.presence() == {"ok": True, "present": None, "last_seen_seconds": last_seen_seconds}


@pytest.mark.parametrize("failure", ["missing_camera", "no_frame", "read_error", "detector_error"])
def test_unavailable_camera_never_claims_absence(monkeypatch, failure, caplog):
    def frame():
        if failure == "read_error":
            raise RuntimeError("private camera information")
        return object() if failure == "detector_error" else None

    class BrokenDetector:
        def __init__(self, model):
            raise RuntimeError("private detector information")

    monkeypatch.setattr(reachy_muse.arrival, "YoloxPersonDetector", BrokenDetector)
    controller = ReachyController()
    if failure != "missing_camera":
        controller._robot = SimpleNamespace(media=SimpleNamespace(get_frame=frame))
    assert controller.presence() == {"ok": True, "present": None, "last_seen_seconds": None}
    assert "private" not in caplog.text


def test_watching_uses_detector_state_without_reading_a_frame_and_keeps_history(monkeypatch):
    now = [101.0]
    monkeypatch.setattr(reachy_muse.hardware.time, "monotonic", lambda: now[0])
    controller = ReachyController()
    controller._robot = SimpleNamespace(media=SimpleNamespace(
        get_frame=lambda: pytest.fail("a watching camera must not be read by a command")))
    controller._watching = True
    controller._arrival_watcher = SimpleNamespace(presence=lambda: Presence(True, 100, 100))
    assert controller.presence() == {"ok": True, "present": True, "last_seen_seconds": 1}
    now[0] = 120
    assert controller.presence() == {"ok": True, "present": None, "last_seen_seconds": 20}
    controller._arrival_watcher = None
    assert controller.presence() == {"ok": True, "present": None, "last_seen_seconds": 20}


def test_watcher_invalidates_presence_while_moving_or_failed():
    still = [True]
    watcher = reachy_muse.arrival.ArrivalWatcher(lambda: None, lambda: still[0], clock=lambda: 101)
    watcher.set_watching(True)
    watcher._presence = Presence(True, 100, 100)
    assert watcher.presence().result(101) == {"ok": True, "present": True, "last_seen_seconds": 1}
    still[0] = False
    assert watcher.presence().result(101) == {"ok": True, "present": None, "last_seen_seconds": 1}

    still[0] = True
    watcher._failure = RuntimeError("camera failed")
    assert watcher.presence().result(101) == {"ok": True, "present": None, "last_seen_seconds": 1}


def test_last_seen_survives_the_end_of_arrival_watching(monkeypatch):
    monkeypatch.setattr(reachy_muse.hardware.time, "monotonic", lambda: 120)
    controller = ReachyController()
    controller._arrival_watcher = SimpleNamespace(presence=lambda: Presence(None, 110, 100))
    assert controller.presence() == {"ok": True, "present": None, "last_seen_seconds": 20}


@pytest.mark.parametrize("blocked", ["camera", "detector", "lock"])
def test_presence_deadline_bounds_camera_detector_and_lock_waits(monkeypatch, blocked):
    stalled, resume = threading.Event(), threading.Event()
    reads = []
    monkeypatch.setattr(reachy_muse.hardware.time, "monotonic", lambda: 120)

    def block():
        stalled.set()
        resume.wait()

    def frame():
        reads.append(True)
        if blocked == "camera":
            block()
        return object()

    class Detector:
        def __init__(self, model):
            pass

        def detect(self, image):
            if blocked == "detector":
                block()
            return []

    monkeypatch.setattr(reachy_muse.arrival, "YoloxPersonDetector", Detector)
    controller = ReachyController(enable_motion=False)
    controller._robot = SimpleNamespace(media=SimpleNamespace(get_frame=frame))
    controller._presence = Presence(True, 100, 100)
    service = ReachyService(identity=Identity("02:00:00:ab:cd:ef"), executor=controller,
                           backends=lambda _: None)
    if blocked == "lock":
        controller._presence_lock.acquire()
        stalled.set()
    with ThreadPoolExecutor(max_workers=1) as commands:
        pending = commands.submit(service.run_command, "reachy.presence", {}, 10)
        try:
            assert stalled.wait(1)
            assert pending.result(timeout=.2) == {
                "ok": True, "payload": {"present": None, "last_seen_seconds": 20}}
            assert reads == ([] if blocked == "lock" else [True])
        finally:
            resume.set()
            if blocked == "lock":
                controller._presence_lock.release()
    assert controller._presence_lock.acquire(timeout=2)
    controller._presence_lock.release()
