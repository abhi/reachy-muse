"""Keyword admission freezes geometry before opening the microphone or greeting."""

import asyncio
from dataclasses import replace
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
        assert conversation._wake_direction.snapshot().decide(100.28).angle is None
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
        phrase_end = time.monotonic()
        assert conversation._wake_direction.snapshot().decide(phrase_end).angle == pytest.approx(1.0471975512)
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
        assert history.samples_between(now - 1.5, now - .5) == ()
        assert [angle for _, angle in history.samples_between(now - .5, now)] == (
            [pytest.approx(2.0943951024)] * 3 if trusted and invalid is None and rate == 16000 else [])

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


class SteppedClock:
    """One virtual clock for the conversation and its DoA sampler. Each sampler sleep ends exactly on time, also
    while a worker thread holds the microphone loop, as keyword inference does on the robot."""

    def __init__(self, now=100.0):
        self.now, self.loop, self._next = now, None, None
        self._wakes, self._resume = asyncio.Queue(), asyncio.Queue()

    async def sleep(self, seconds):
        await self._wakes.put(self.now + seconds)
        await self._resume.get()

    async def advance(self, until):
        if self._next is None:
            self._next = await self._wakes.get()
        while self._next <= until:
            self.now = self._next
            await self._resume.put(None)
            self._next = await self._wakes.get()
        self.now = until

    def spend(self, seconds):
        """From a worker thread: ``seconds`` pass while the event loop, and the sampler on it, go on."""
        asyncio.run_coroutine_threadsafe(self.advance(self.now + seconds), self.loop).result(2)


class UnflaggingDaemon(DirectionalHardware):
    """The daemon never flags speech in its DoA readings."""

    def read_doa(self):
        return replace(super().read_doa(), speech_detected=False)


class LoadedDaemon(DirectionalHardware):
    """The daemon misses the 0.1 s deadline of the first DoA read, as on a loaded robot."""

    def read_doa(self):
        reading = super().read_doa()
        return reading if self.reads > 1 else None


@pytest.mark.parametrize("tail, inference, lag, daemon, echo, line", [
    # 17:12:47 on the robot: the candidate came 0.52 s after the keyword's last token, after 79 ms of inference.
    (.52, .0789, 0, DirectionalHardware, None,
     "Wake direction decision=ok samples=4 span_ms=600.0 boundary_age_ms=598.9 last_age_ms=30.0 mad_deg=0.0 "
     "inlier=1.00 yaw=30.0"),
    # 17:25:25: 0.32 s and 95 ms, with the streaming model placing the last token 0.2 s after the loud audio.
    (.32, .0954, .2, DirectionalHardware, None,
     "Wake direction decision=ok samples=4 span_ms=600.0 boundary_age_ms=415.4 last_age_ms=230.0 mad_deg=0.0 "
     "inlier=1.00 yaw=30.0"),
    # people1's timing: a 0.32 s tail and no inference, so the candidate beats the 0.5 s activity window.
    (.32, 0, 0, DirectionalHardware, None,
     "Wake direction decision=ok samples=4 span_ms=600.0 boundary_age_ms=320.0 last_age_ms=30.0 mad_deg=0.0 "
     "inlier=1.00 yaw=30.0"),
    # Reachy's own echo tail covers 100.80-101.00 s of the pause: the phrase's bearings go with it, and the line
    # says so.
    (.52, .0789, 0, DirectionalHardware, (40, 101.0),
     "Wake direction decision=ineligible samples=0 span_ms=0.0 boundary_age_ms=598.9 last_age_ms=nan mad_deg=nan "
     "inlier=0.00"),
    # The daemon never flags the phrase.
    (.52, .0789, 0, UnflaggingDaemon, None,
     "Wake direction decision=too_few samples=0 span_ms=0.0 boundary_age_ms=598.9 last_age_ms=nan mad_deg=nan "
     "inlier=0.00"),
    # The phrase is flagged, but its first DoA read fails: the sampler clears and waits a second before the next.
    (.52, .0789, 0, LoadedDaemon, None,
     "Wake direction decision=unavailable samples=0 span_ms=0.0 boundary_age_ms=598.9 last_age_ms=nan mad_deg=nan "
     "inlier=0.00"),
], ids=["robot-1712", "robot-1725", "people1", "robot-echo", "never-flagged", "failed-read"])
def test_a_pause_after_the_keyword_keeps_its_direction_until_the_robot_s_late_candidate(
        handoff, monkeypatch, caplog, tail, inference, lag, daemon, echo, line):
    """The user says the 0.68 s phrase of people1's test, then waits for the wake gesture, as the docs say to. The
    DoA flags speech only while they talk. The keyword is found on the robot's timing, with the robot's verifier.
    A wake that finds no direction says what emptied its window, so the robot's log can tell the causes apart."""
    from reachy_muse.wake_policy import Verification, WakePolicy
    from test_wake_handoff import Detection

    clock = SteppedClock()
    monkeypatch.setattr("reachy_muse.conversation.time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr("reachy_muse.conversation.WakeDirection",
                        lambda read: WakeDirection(read, clock=lambda: clock.now, sleep=clock.sleep))
    hardware = daemon()
    caplog.set_level("INFO")

    async def run(conversation, session, hw, wake, recognition, recordings, accepted, feed):
        np = pytest.importorskip("numpy")
        clock.loop = asyncio.get_running_loop()
        heard = SimpleNamespace(samples=0, boundary=None)

        def spot(audio, align=True):
            wake.feed_id += 1
            wake.last_detection = None
            loud = np.flatnonzero(audio == np.float32(.25))
            if len(loud):
                heard.boundary = heard.samples + int(loud[-1]) + 1 + round(lag * 16000)
            heard.samples += len(audio)
            if heard.boundary is None or heard.samples - heard.boundary < round(tail * 16000):
                return False
            clock.spend(inference)
            wake.last_detection = Detection(heard.samples - heard.boundary, wake.epoch, wake.feed_id)
            heard.boundary = None
            return True

        class Verify:
            async def verify(self, pcm, phrase):
                await clock.advance(clock.now + .079)
                return Verification("timeout", "none")

        wake.feed, wake.candidate_replay = spot, True
        wake.replay_candidate = lambda threshold: True
        wake.align_candidate = lambda: clock.spend(.328)
        conversation._wake_policy = WakePolicy(Verify())
        frame = 0
        for marker, frames in ((.25, 34), (0, 60)):
            hw.speech = bool(marker)
            for _ in range(frames):
                frame += 1
                await clock.advance(100 + frame * .02)
                if echo is not None and frame == echo[0]:
                    conversation._echo_until = echo[1]
                await feed((marker, 1))
        assert [record.message for record in caplog.records
                if record.message.startswith("Wake direction")] == [line]
        assert hw.directions == ([(pytest.approx(math.pi / 3), "idle")] if line.endswith("yaw=30.0") else [])
        assert conversation._wake_deadline is not None
        assert session.setup_messages == []

    asyncio.run(bounded(handoff(run, hardware=hardware)))
