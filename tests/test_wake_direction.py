"""Keyword admission freezes geometry before opening the microphone or greeting."""

import asyncio
import math
import time
from types import SimpleNamespace

import pytest

from reachy_muse.conversation import _CaptureBuffer
from reachy_muse.device_command import WakeWindow
from reachy_muse.speaker_direction import DoAReading, WakeDirection
from test_conversation import FakeHardware, bounded
from test_wake_handoff import handoff


class DirectionalHardware(FakeHardware):
    echo_cancelled_input = True
    motion_enabled = True

    def __init__(self):
        super().__init__()
        self.directions = []
        self.reads = 0
        self.speech = True

    def read_doa(self):
        self.reads += 1
        return DoAReading(math.pi / 3, self.speech)

    def look_at_speaker(self, angle):
        self.directions.append((angle, self.states[-1]))


async def wait_for_reads(hardware, count):
    async def wait():
        while hardware.reads < count:
            await asyncio.sleep(.005)

    await asyncio.wait_for(wait(), 2)
    await asyncio.sleep(.005)


def test_short_keyword_collects_direction_through_microphone_activity(handoff, monkeypatch):
    now = [100.0]
    sleeping, resume = asyncio.Queue(), asyncio.Queue()

    async def sleep(delay):
        await sleeping.put(now[0] + delay)
        await resume.get()

    monkeypatch.setattr("reachy_muse.conversation.time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr("reachy_muse.conversation.WakeDirection",
                        lambda read: WakeDirection(read, clock=lambda: now[0], sleep=sleep))
    hardware = DirectionalHardware()

    async def run(conversation, session, hw, wake, recognition, recordings, accepted, feed):
        next_poll = await sleeping.get()
        assert hw.reads == 0
        for frame in range(14):
            now[0] = 100 + (frame + 1) * .02
            await feed((.25, 1))
            if now[0] >= next_poll:
                await resume.put(None)
                next_poll = await sleeping.get()
        hw.speech = False
        now[0] = 100.30
        await feed((0, 1))
        assert hw.directions == [(pytest.approx(1.0471975512), "idle")]
        assert conversation._wake_direction.estimate(100.28) is None
        assert session.setup_messages == []

    asyncio.run(bounded(handoff(run, delayed=True, hardware=hardware)))


@pytest.mark.parametrize("asleep, trusted", [(False, True), (True, True), (True, False)])
def test_keyword_freezes_direction_before_wake_state_and_stops_sampling(handoff, asleep, trusted):
    hardware = DirectionalHardware()

    async def run(conversation, session, hw, wake, recognition, recordings, accepted, feed):
        np = pytest.importorskip("numpy")
        conversation.backends.voice.phrases["I'm awake."] = (np.full(80, .3, np.float32),)
        if asleep:
            conversation._wake_window = WakeWindow.ASLEEP
            hw.set_state("sleeping")
        conversation._direction_audio_until = math.inf
        await wait_for_reads(hw, 7)
        await feed((.25, 14), (0, 7))
        assert hw.directions == ([(pytest.approx(1.0471975512), "sleeping" if asleep else "idle")]
                                 if trusted else [])
        reads = hw.reads
        await asyncio.sleep(.1)
        assert hw.reads == reads
        assert conversation._wake_deadline > time.monotonic()
        assert session.setup_messages == []

    asyncio.run(bounded(handoff(run, trusted=trusted, recognized="Hey Muse.", hardware=hardware)))


@pytest.mark.parametrize("blocked", ["muted", "echo_tail", "queued_output", "realtime"])
def test_direction_gate_excludes_all_robot_audio_even_with_aec(handoff, blocked):
    hardware = DirectionalHardware()

    async def run(conversation, session, hw, wake, recognition, recordings, accepted, feed):
        conversation._direction_audio_until = math.inf
        await wait_for_reads(hw, 4)
        assert conversation._wake_direction.estimate(time.monotonic()) == pytest.approx(1.0471975512)
        if blocked == "muted":
            conversation._muted = True
        elif blocked == "echo_tail":
            conversation._echo_until = math.inf
        elif blocked == "queued_output":
            conversation._output_busy = True
        else:
            from types import SimpleNamespace
            conversation.call = SimpleNamespace(audible=True)
        assert conversation._input_blocked() is False
        conversation._attend_wake(time.monotonic())
        assert hw.directions == []
        # Release before the normal microphone worker inspects the fake call.
        conversation.call = None
        conversation._muted = False
        conversation._echo_until = 0
        conversation._output_busy = False
        conversation._direction_audio_until = math.inf
        await wait_for_reads(hw, hw.reads + 5)
        conversation._attend_wake(time.monotonic())
        assert hw.directions == [(pytest.approx(1.0471975512), "idle")]

    asyncio.run(bounded(handoff(run, hardware=hardware)))


def test_capture_gap_discards_geometry_before_a_new_wake(handoff):
    hardware = DirectionalHardware()

    async def run(conversation, session, hw, wake, recognition, recordings, accepted, feed):
        conversation._direction_audio_until = math.inf
        await wait_for_reads(hw, 5)
        await feed((.25, 14), (0, 7), gap=True)
        assert hw.directions == []
        assert conversation._wake_deadline > time.monotonic()
        assert hw.states[-1] == "listening"

    asyncio.run(bounded(handoff(run, hardware=hardware)))


@pytest.mark.parametrize("trusted, invalid, rate, expected", [
    (True, None, 16000, [pytest.approx(math.pi / 3)]),
    (False, None, 16000, []), (True, None, 48000, []),
    *[(True, invalid, 16000, []) for invalid in
      ("old_epoch", "old_feed", "boolean_tail", "negative_tail", "oversized_tail")],
])
def test_changing_speakers_require_a_verified_keyword_boundary(handoff, trusted, invalid, rate, expected):
    hardware = DirectionalHardware()
    hardware.sample_rate = rate

    async def run(conversation, session, hw, wake, recognition, recordings, accepted, feed):
        history = conversation._wake_direction = WakeDirection(hw.read_doa)
        now = time.monotonic()
        history._samples.extend((now + offset, math.radians(angle)) for offset, angle in [
            (-.82, 60), (-.75, 60), (-.68, 60), (-.2, 120), (-.13, 120), (-.06, 120),
        ])
        await feed((.25, 14), (.5, 20), (0, 7))
        assert [angle for angle, _ in hw.directions] == expected
        assert conversation._wake_deadline > time.monotonic()
        assert history.estimate(time.monotonic()) is None

    asyncio.run(bounded(handoff(run, trusted=trusted, invalid=invalid,
                               recognized="Hey Muse, find flights.", hardware=hardware)))


def test_silence_and_still_mode_preserve_the_current_wake_behavior(handoff):
    async def run(conversation, session, hw, wake, recognition, recordings, accepted, feed):
        await asyncio.sleep(.25)
        assert hw.reads == 0
        await feed((.25, 14), (0, 7))
        assert hw.directions == []
        assert conversation._wake_deadline > time.monotonic()
        assert hw.states[-1] == "listening"
        if not hw.motion_enabled:
            assert hw.reads == 0

    silent = DirectionalHardware()
    silent.speech = False
    asyncio.run(bounded(handoff(run, hardware=silent)))
    still = DirectionalHardware()
    still.motion_enabled = False
    asyncio.run(bounded(handoff(run, hardware=still)))


def test_capture_batches_keep_the_latest_capture_end_through_backlog():
    np = pytest.importorskip("numpy")

    async def run():
        buffer = _CaptureBuffer(16000)
        buffer.push(np.full(320, .1), captured_at=100.02)
        buffer.push(np.full(320, .2), captured_at=100.04)
        await asyncio.sleep(.01)
        batch = await buffer.get()
        assert batch.captured_at == 100.04
        assert batch.samples.tolist() == [.1] * 320 + [.2] * 320
        buffer.push(np.full(320, .3), captured_at=101.0, gap=True)
        batch = await buffer.get()
        assert (batch.captured_at, batch.gap) == (101.0, True)
        assert batch.samples.tolist() == [.3] * 320

    asyncio.run(run())


@pytest.mark.parametrize("delay", [.5, .9])
@pytest.mark.parametrize("dispersed", [False, True])
def test_candidate_geometry_survives_verification_replay_and_alignment(handoff, monkeypatch, caplog, delay, dispersed):
    from reachy_muse.wake_policy import Verification, WakePolicy

    now = [100.0]
    sleeping, resume = asyncio.Queue(), asyncio.Queue()

    async def sleep(seconds):
        await sleeping.put(now[0] + seconds)
        await resume.get()

    monkeypatch.setattr("reachy_muse.conversation.time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr("reachy_muse.conversation.WakeDirection",
                        lambda read: WakeDirection(read, clock=lambda: now[0], sleep=sleep))
    hardware = DirectionalHardware()
    read = hardware.read_doa

    def direction():
        reading = read()
        return DoAReading((0 if hardware.reads % 2 else math.pi) if dispersed else reading.angle,
                          hardware.speech)

    hardware.read_doa = direction
    caplog.set_level("INFO")

    async def run(conversation, session, hw, wake, recognition, recordings, accepted, feed):
        class Verify:
            async def verify(self, pcm, phrase):
                now[0] += delay - .2
                await resume.put(None)
                await sleeping.get()
                return Verification("timeout", "none")

        detect = wake.feed
        wake.candidate_replay = True
        wake.feed = lambda audio, align: detect(audio)
        stages = []

        def replay(threshold):
            stages.append("replay")
            now[0] += .1
            return True

        def align():
            stages.append("alignment")
            now[0] += .1

        wake.replay_candidate, wake.align_candidate = replay, align
        conversation._wake_policy = WakePolicy(Verify())
        next_poll = await sleeping.get()
        for frame in range(34):
            now[0] = 100 + (frame + 1) * .02
            await feed((.25, 1))
            if now[0] >= next_poll:
                await resume.put(None)
                next_poll = await sleeping.get()
        hw.speech = False
        now[0] = 101.0
        await feed((0, 16))
        assert stages == ["replay", "alignment"]
        assert hw.directions == ([] if dispersed else [(pytest.approx(math.pi / 3), "idle")])
        assert session.setup_messages == []
        lines = [record.message for record in caplog.records if record.message.startswith("Wake direction")]
        assert lines == [
            "Wake direction decision=dispersed samples=4 span_ms=620.0 boundary_age_ms=320.0 "
            "last_age_ms=0.0 mad_deg=90.0 inlier=0.00" if dispersed else
            "Wake direction decision=ok samples=4 span_ms=620.0 boundary_age_ms=320.0 "
            "last_age_ms=0.0 mad_deg=0.0 inlier=1.00 yaw=30.0"
        ]

    asyncio.run(bounded(handoff(run, delayed=True, hardware=hardware)))
