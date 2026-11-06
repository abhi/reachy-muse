"""Verified invocation, speaker attention and request/reply ordering."""

import asyncio
import math
import threading
from types import SimpleNamespace

import pytest

from reachy_muse.speaker_direction import DoAReading
from reachy_muse.wake_policy import WakePolicy, WakeVerifier
from reachy_muse.voice_verify import Decision
from test_device_command_turns import played, rig_with, run, sent_to_muse, service
from test_pins import Transcripts, Vad, WAKE_SAMPLE, Wake, np


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
@pytest.mark.parametrize("voice_matches", [None, False, True], ids=("voice-off", "voice-reject", "voice-match"))
def test_attention_waits_for_verified_wake_and_uses_candidate_direction(monkeypatch, matches, voice_matches):
    rig = rig_with(monkeypatch)
    monkeypatch.setattr("webrtcvad.Vad", lambda mode: Vad())
    entered, release = asyncio.Event(), asyncio.Event()
    reads, directions = [], []
    voice_entered, voice_release = threading.Event(), threading.Event()
    voice_checks = []

    class SpeakerVerifier:
        mode = "strict"
        enforcing = True

        def verify(self, samples, rate):
            voice_checks.append(samples.copy())
            if len(voice_checks) == 1:
                voice_entered.set()
                assert voice_release.wait(3)
            return Decision(1.0 if voice_matches else 0.0, voice_matches, voice_matches)

    rig.recording.label = lambda samples: rig.recording.labels.get(float(samples[0]), "voice-rejected")

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
                           wake_policy=policy, wake_timeout_s=30,
                           voice_verifier=SpeakerVerifier() if voice_matches is not None else None)

    async def user():
        await rig.wait_until_or_timeout(lambda: len(reads) >= 5)
        assert len(reads) >= 5
        rig.say(wake=True)
        await asyncio.wait_for(entered.wait(), 2)
        assert directions == []
        assert rig.states()[-1] == "idle"
        assert sent_to_muse(rig.recording.events) == []
        assert transcripts.texts == ["What is the weather tomorrow?"]
        assert voice_checks == []
        await rig.wait_until_or_timeout(lambda: 2 * math.pi / 3 in reads, timeout=.15)
        assert 2 * math.pi / 3 in reads
        release.set()
        if matches and voice_matches is not None:
            try:
                assert await asyncio.to_thread(voice_entered.wait, 2)
                assert directions == [(pytest.approx(math.pi / 3), "idle")]
                assert rig.states()[-1] == "idle"
                assert sent_to_muse(rig.recording.events) == []
                assert transcripts.texts == ["What is the weather tomorrow?"]
                assert set(voice_checks[0]) == {WAKE_SAMPLE}
            finally:
                voice_release.set()

    events = run(rig, conversation, user, rest=("idle", "listening"))
    assert directions == ([(pytest.approx(math.pi / 3), "idle")] if matches else [])
    if matches and voice_matches is not False:
        assert directions == [(pytest.approx(math.pi / 3), "idle")]
        assert sent_to_muse(events) == ["What is the weather tomorrow?"]
        assert played(events) == ["tts:Sunny tomorrow."]
        if voice_matches:
            assert len(voice_checks) == 2
            assert WAKE_SAMPLE not in voice_checks[1]
    else:
        assert sent_to_muse(events) == []
        assert played(events) == (["voice-rejected"] if matches else [])
        assert transcripts.texts == ["What is the weather tomorrow?"]
        assert len(voice_checks) == (1 if matches else 0)


def test_voice_rejection_chime_allows_the_next_verified_wake_and_reply(monkeypatch):
    rig = rig_with(monkeypatch)
    monkeypatch.setattr("webrtcvad.Vad", lambda mode: Vad())
    monkeypatch.setattr("reachy_muse.conversation.ECHO_TAIL_S", 0)
    rig.recording.label = lambda samples: rig.recording.labels.get(float(samples[0]), "voice-rejected")
    voice_checks = []

    class SpeakerVerifier:
        mode = "strict"
        enforcing = True

        def verify(self, samples, rate):
            voice_checks.append(samples.copy())
            allowed = len(voice_checks) > 1
            return Decision(float(allowed), allowed, allowed)

    class AcousticVerifier:
        async def transcribe(self, wav):
            return "hey muse"

    async def muse(session, request):
        await session.answer(request, "Sunny tomorrow.")

    transcripts = Transcripts("What is the weather tomorrow?")
    conversation = service(rig, muse, transcripts, wake_detector=TimedWake(),
                           wake_policy=WakePolicy(WakeVerifier(local=AcousticVerifier())),
                           voice_verifier=SpeakerVerifier(), wake_timeout_s=30)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "voice-rejected" in played(rig.recording.events))
        await rig.quiet("idle", "listening", span=.3)
        assert played(rig.recording.events) == ["voice-rejected"]
        assert sent_to_muse(rig.recording.events) == []
        assert transcripts.texts == ["What is the weather tomorrow?"]
        rig.say(wake=True)

    events = run(rig, conversation, user, rest=("idle", "listening"))
    assert sent_to_muse(events) == ["What is the weather tomorrow?"]
    assert played(events) == ["voice-rejected", "tts:Sunny tomorrow."]
    assert len(voice_checks) == 3
