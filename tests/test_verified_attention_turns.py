"""Verified invocation, speaker attention and request/reply ordering."""

import asyncio
import math
from types import SimpleNamespace

import pytest

from reachy_muse.speaker_direction import DoAReading
from reachy_muse.wake_policy import WakePolicy, WakeVerifier
from test_device_command_turns import played, rig_with, run, sent_to_muse, service
from test_pins import Transcripts, WAKE_SAMPLE, Wake, np


class TimedWake(Wake):
    def __init__(self):
        self.epoch = self.feed_id = 0
        self.last_detection = None

    def feed(self, samples):
        self.feed_id += 1
        self.last_detection = None
        positions = np.flatnonzero(samples == WAKE_SAMPLE)
        if not len(positions):
            return False
        self.last_detection = SimpleNamespace(
            epoch=self.epoch, feed_id=self.feed_id,
            post_wake_samples=len(samples) - int(positions[-1]) - 1)
        return True

    def reset(self):
        self.epoch += 1
        self.last_detection = None


@pytest.mark.parametrize("matches", [False, True])
def test_attention_waits_for_verified_wake_and_uses_candidate_direction(monkeypatch, matches):
    rig = rig_with(monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    reads, directions = [], []

    class Verifier:
        async def transcribe(self, wav):
            entered.set()
            await release.wait()
            return "hey muse" if matches else "hey music"

    def read_doa():
        angle = 2 * math.pi / 3 if entered.is_set() else math.pi / 3
        reads.append(angle)
        return DoAReading(angle, True)

    rig.hardware.read_doa = read_doa
    rig.hardware.look_at_speaker = lambda angle: directions.append((angle, rig.states()[-1]))
    policy = WakePolicy(WakeVerifier(local=Verifier(), budget_s=.3))
    transcripts = Transcripts("What is the weather tomorrow?")

    async def muse(session, request):
        assert directions == [(pytest.approx(math.pi / 3), "idle")]
        await session.answer(request, "Sunny tomorrow.")

    conversation = service(rig, muse, transcripts, wake_detector=TimedWake(),
                           wake_policy=policy, wake_timeout_s=30)

    async def user():
        await rig.wait_until_or_timeout(lambda: len(reads) >= 5)
        assert len(reads) >= 5
        rig.say(wake=True)
        await asyncio.wait_for(entered.wait(), 2)
        assert directions == []
        assert rig.states()[-1] == "idle"
        assert sent_to_muse(rig.recording.events) == []
        assert transcripts.texts == ["What is the weather tomorrow?"]
        await rig.wait_until_or_timeout(lambda: 2 * math.pi / 3 in reads, timeout=.15)
        assert 2 * math.pi / 3 in reads
        release.set()

    events = run(rig, conversation, user, rest=("idle", "listening"))
    if matches:
        assert directions == [(pytest.approx(math.pi / 3), "idle")]
        assert sent_to_muse(events) == ["What is the weather tomorrow?"]
        assert played(events) == ["tts:Sunny tomorrow."]
    else:
        assert directions == []
        assert sent_to_muse(events) == []
        assert played(events) == []
        assert transcripts.texts == ["What is the weather tomorrow?"]
