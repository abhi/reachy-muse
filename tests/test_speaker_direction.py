"""Offline wake direction estimates, with a scripted source and clock."""

import asyncio
import math
import time
from types import SimpleNamespace

import pytest

from reachy_muse.hardware import ReachyController
from reachy_muse.speaker_direction import DoAReading, WakeDirection, speaker_yaw
from test_doa_http import doa_server
from test_hardware import FakeRobot, pose_factory


@pytest.mark.parametrize("angle, head, body", [
    (-math.pi / 6, 35, 15), (7 * math.pi / 6, -35, -15),
    (0, 35, 15), (math.pi / 6, 35, 15), (math.pi / 3, 30, 10),
    (math.radians(80), 10, 0), (math.pi / 2, 0, 0),
    (2 * math.pi / 3, -30, -10), (math.pi, -35, -15),
])
def test_world_direction_maps_to_absolute_sdk_yaw(angle, head, body):
    target = speaker_yaw(angle)
    assert (target.head, target.body) == pytest.approx((head, body))


@pytest.mark.parametrize("angle", [True, None, "left", -1.6, 4.8, math.inf, math.nan])
def test_invalid_directions_cannot_make_motor_targets(angle):
    with pytest.raises(ValueError, match="finite radians"):
        speaker_yaw(angle)


def test_daemon_read_uses_controller_address_and_returns_only_geometry(doa_server):
    controller, state = doa_server('{"angle": 0.5, "speech_detected": true, "unused": "discard"}')
    reading = controller.read_doa()
    assert (reading.angle, reading.speech_detected) == (0.5, True)
    assert state.paths == ["/api/state/doa"]


@pytest.mark.parametrize("payload", [
    'null', '[]', '{}', 'broken', '{"angle": true, "speech_detected": true}',
    '{"angle": NaN, "speech_detected": true}', '{"angle": 3.2, "speech_detected": true}',
    '{"angle": -0.1, "speech_detected": true}',
    '{"angle": 0, "speech_detected": 1}', '{"angle": "0", "speech_detected": true}',
])
def test_unsupported_daemon_data_falls_back_to_a_later_valid_read(doa_server, payload):
    controller, _ = doa_server(payload, '{"angle": 0, "speech_detected": false}')
    assert controller.read_doa() is None
    assert controller.read_doa() == DoAReading(0, False)


def test_unreachable_daemon_is_optional(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("socket.create_connection", unavailable)
    controller = ReachyController(host="daemon.invalid", port=8123)
    controller._robot = FakeRobot()
    assert controller.read_doa() is None
    assert speaker_yaw(math.pi / 2).head == 0


@pytest.mark.parametrize("yaw, roll, pitch, angle, head, body", [
    (30, 0, 0, 90, 30, 10), (-30, 0, 0, 90, -30, -10),
    (30, 0, 0, 60, 35, 15), (-30, 0, 0, 120, -35, -15),
    (35, 0, 0, 0, 35, 15), (-35, 0, 0, 180, -35, -15),
    (10, 60, 0, 60, 26.10211375, 6.10211375),
    (10, 0, 60, 60, 35, 15),
])
def test_reading_uses_the_current_head_rotation_before_yaw_limits(
        doa_server, yaw, roll, pitch, angle, head, body):
    controller, _ = doa_server(f'{{"angle": {math.radians(angle)}, "speech_detected": true}}')
    controller._robot.pose = pose_factory(yaw=yaw, roll=roll, pitch=pitch)
    reading = controller.read_doa()
    target = speaker_yaw(reading.angle)
    assert (target.head, target.body) == pytest.approx((head, body))


def test_direction_history_uses_each_readings_head_pose(doa_server):
    controller, _ = doa_server(*[f'{{"angle": {math.radians(angle)}, "speech_detected": true}}'
                                 for angle in (80, 50, 90, 60)])
    readings = []
    for yaw in (20, -10, 30, 0):
        controller._robot.pose = pose_factory(yaw=yaw)
        readings.append(controller.read_doa())

    def check(direction, clock, delays):
        assert direction.estimate(clock.now) == pytest.approx(math.pi / 3)
        target = speaker_yaw(direction.estimate(clock.now))
        assert (target.head, target.body) == pytest.approx((30, 10))

    sampled(readings, check)


async def sample_script(readings, check, *, eligible=None):
    """Run the actual sampler with fake time; park at the end for assertions."""
    clock = SimpleNamespace(now=100.0)
    remaining = list(readings)
    finished = asyncio.Event()
    delays = []

    def source():
        return remaining.pop(0)

    async def sleep(delay):
        delays.append(delay)
        clock.now += delay
        if not remaining:
            finished.set()
            await asyncio.Event().wait()
        await asyncio.sleep(0)

    direction = WakeDirection(source, clock=lambda: clock.now, sleep=sleep)
    task = asyncio.create_task(direction.sample_while(eligible or (lambda: True)))
    try:
        await asyncio.wait_for(finished.wait(), 2)
        check(direction, clock, delays)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def sampled(readings, check):
    asyncio.run(sample_script(readings, check))


def test_median_uses_speech_only_and_ignores_one_outlier():
    def check(direction, clock, delays):
        assert direction.estimate(clock.now) == pytest.approx(1.0384709049)
        assert delays == pytest.approx([.2, 1 / 15, 1 / 15, 1 / 15, 1 / 15])

    sampled([DoAReading(0, False), DoAReading(math.radians(59), True),
             DoAReading(math.radians(60), True), DoAReading(0, True),
             DoAReading(math.radians(61), True)], check)


@pytest.mark.parametrize("angles", [[60, 60], [0, 60, 120, 180], [0, 0, 180, 180]])
def test_sparse_or_disagreeing_samples_have_no_direction_then_good_speech_does(angles):
    def check(direction, clock, delays):
        assert direction.estimate(clock.now) is None

    sampled([DoAReading(math.radians(angle), True) for angle in angles], check)

    def consistent(direction, clock, delays):
        assert direction.estimate(clock.now) == pytest.approx(math.radians(80))

    sampled([DoAReading(math.radians(80), True)] * 4, consistent)


def test_phrase_boundary_excludes_later_speech_and_stale_or_cleared_history():
    def check(direction, clock, delays):
        # The keyword ended before a different speaker's later request audio.
        assert direction.estimate(100.2) == pytest.approx(math.radians(60))
        assert direction.estimate(clock.now) == pytest.approx(math.radians(120))
        clock.now += 1.1
        assert direction.estimate(100.2) is None
        direction.clear()
        assert direction.estimate(clock.now) is None

    sampled([DoAReading(math.radians(60), True)] * 4 + [DoAReading(0, False)] * 2
            + [DoAReading(math.radians(120), True)] * 4, check)


def test_a_short_pause_and_changed_speaker_cannot_select_the_earlier_speaker():
    def check(direction, clock, delays):
        assert direction.estimate(clock.now) is None

    sampled([DoAReading(math.radians(60), True)] * 18 + [DoAReading(0, False)]
            + [DoAReading(math.radians(120), True)] * 4, check)


def test_failed_source_clears_old_speech_but_new_speech_can_restore_attention():
    def check(direction, clock, delays):
        assert direction.estimate(clock.now) == pytest.approx(math.radians(120))
        assert delays[3] == 1.0

    sampled([DoAReading(math.radians(60), True)] * 3 + [None]
            + [DoAReading(math.radians(120), True)] * 3, check)


def test_inflight_read_is_rejected_across_a_listening_epoch():
    import threading

    async def run():
        gate = SimpleNamespace(eligible=True)
        entered, proceed = threading.Event(), threading.Event()
        count = 0

        def read():
            nonlocal count
            count += 1
            if count == 1:
                entered.set()
                assert proceed.wait(2)
                return DoAReading(0, True)
            return DoAReading(math.pi / 2, True)

        direction = WakeDirection(read)
        task = asyncio.create_task(direction.sample_while(lambda: gate.eligible))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            gate.eligible = False
            direction.clear()
            gate.eligible = True
            proceed.set()
            await asyncio.sleep(.45)
            assert direction.estimate(time.monotonic()) == pytest.approx(math.pi / 2)
        finally:
            proceed.set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
