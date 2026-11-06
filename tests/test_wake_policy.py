import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_muse.wake_policy import WakeContext, WakePolicy, WakeVerifier, contains_phrase


class Detector:
    phrase = "hey muse"
    threshold = .1
    boost = 1.0
    candidate_replay = True
    last_candidate = None

    def __init__(self, strong=True):
        self.strong = strong
        self.aligned = self.resets = 0
        self.floors = []

    def feed(self, audio, *, align):
        assert align is False
        return bool(np.any(audio))

    def replay_candidate(self, threshold):
        self.floors.append(threshold)
        return self.strong

    def align_candidate(self):
        self.aligned += 1

    def reset(self):
        self.resets += 1


class Local:
    def __init__(self, text="Hey, Muse!", *, hang=False, fail=False):
        self.text, self.hang, self.fail = text, hang, fail
        self.cancelled = False
        self.received = []

    async def transcribe(self, wav):
        self.received.append(wav)
        try:
            if self.hang:
                await asyncio.Event().wait()
            if self.fail:
                raise RuntimeError("secret transcript")
            return self.text
        finally:
            self.cancelled = True


class Companion(Local):
    async def transcribe_wake(self, pcm):
        return await self.transcribe(pcm)


@pytest.mark.parametrize("text, expected", [
    ("Hey, Muse!", True), ("well hey muse please listen", True),
    ("HAY MUSE", True), ("hay music", False), ("pay muse", False),
    ("hey music", False), ("hey museum", False), ("hi muse", False),
    ("hey my muse", False), ("muse", False), ("hey news", False),
])
def test_phrase_boundaries(text, expected):
    assert contains_phrase(text) is expected


@pytest.mark.parametrize("text, phrase, expected", [
    ("hey o'brien", "hey o'brien", True),
    ("Well, HEY O'BRIEN!", "hey o'brien", True),
    ("hey o'brien please listen", "hey o'brien", True),
    ("hey d'artagnan", "hey d'artagnan", True),
    ("hey o'brien", "hey o'brie", False),
    ("hey o'brien", "hey brian", False),
])
def test_custom_phrase_with_apostrophes(text, phrase, expected):
    assert contains_phrase(text, phrase) is expected


def test_companion_negative_is_final_even_when_local_matches():
    async def run():
        local, companion = Local(), Companion("hey music")
        result = await WakeVerifier(local=local, companion=lambda: companion).verify(b"\0\0" * 24000, "hey muse")
        assert (result.result, result.source) == ("no_match", "companion")
        assert local.cancelled is True
    asyncio.run(run())


def test_companion_timeout_uses_already_started_local_and_cancels_stream():
    async def run():
        local, companion = Local(), Companion(hang=True)
        result = await WakeVerifier(local=local, companion=lambda: companion,
                                    budget_s=.06, companion_s=.02).verify(b"\0\0" * 20, "hey muse")
        assert (result.result, result.source) == ("match", "local")
        assert companion.cancelled is True
        assert len(local.received) == 1
    asyncio.run(run())


def test_both_hung_verifiers_finish_within_one_budget_and_abort():
    async def run():
        local, companion = Local(hang=True), Companion(hang=True)
        verifier = WakeVerifier(local=local, companion=lambda: companion, budget_s=.03, companion_s=.01)
        result = await asyncio.wait_for(verifier.verify(b"\0\0" * 20, "hey muse"), .2)
        assert (result.result, result.source) == ("timeout", "none")
        assert (local.cancelled, companion.cancelled) == (True, True)
    asyncio.run(run())


@pytest.mark.parametrize("context, reason", [
    (WakeContext(playback=("You can say hey muse to wake me.",), speaking=True), "echo"),
    (WakeContext(speaking=True), "speaking"),
    (WakeContext(blocked=True), "speaking"),
    (WakeContext(armed=False), "cooldown"),
])
def test_rejected_candidates_are_content_free_and_skip_verifier(context, reason, caplog):
    async def run():
        local, detector = Local(), Detector()
        policy = WakePolicy(WakeVerifier(local=local))
        assert await policy.feed(detector, np.ones(1600, np.float32) / 10, context=context) is False
        assert (policy.last_decision.reason, policy.last_decision.score, detector.aligned) == (reason, None, 0)
        assert local.received == []
    caplog.set_level("INFO", logger="reachy_muse.wake_policy")
    asyncio.run(run())
    assert len(caplog.messages) == 1
    assert '"score": null' in caplog.messages[0]
    assert "muse" not in caplog.messages[0].lower()
    assert "secret" not in caplog.messages[0]


def test_duplex_wake_during_unrelated_known_output_requires_verification():
    async def run():
        detector = Detector(strong=False)
        policy = WakePolicy(WakeVerifier(local=Local()))
        context = WakeContext(speaking=True, playback=("The forecast is sunny.",))
        assert await policy.feed(detector, np.full(1600, .1), context=context) is True
        assert (policy.last_decision.reason, policy.last_decision.verifier, detector.aligned) == ("accepted", "local", 1)
        assert detector.floors == []
    asyncio.run(run())


def test_split_playback_echo_cannot_be_authorized_by_an_exact_verifier_match():
    async def run():
        local, detector = Local("Hey, Muse!"), Detector()
        policy = WakePolicy(WakeVerifier(local=local))
        context = WakeContext(speaking=True, playback=("Hey,", "Muse is here."))
        assert await policy.feed(detector, np.full(1600, .1), context=context) is False
        assert policy.last_decision.reason == "echo"
        assert local.received == []
        assert detector.aligned == 0
    asyncio.run(run())


@pytest.mark.parametrize("batched", [False, True])
def test_repeated_playback_keeps_its_order_when_guarding_a_split_wake(batched):
    from reachy_muse.conversation import VoiceConversation, _CaptureBuffer

    async def run():
        conversation = VoiceConversation(SimpleNamespace(), SimpleNamespace(echo_cancelled_input=True),
                                         backends=SimpleNamespace(motion=None))
        conversation._muse_audible("Muse is here.", 98)
        conversation._muse_audible("Hey,", 100)
        buffer = _CaptureBuffer(16000)
        if batched:
            buffer.push(np.full(800, .1), wake_context=conversation._wake_capture_context(100))
        conversation._muse_audible("Muse is here.", 100.1)
        buffer.push(np.full(800, .1), wake_context=conversation._wake_capture_context(100.1))
        captured = await buffer.get()
        local, detector = Local(), Detector()
        policy = WakePolicy(WakeVerifier(local=local))
        assert await policy.feed(detector, captured.samples, context=captured.wake_context) is False
        assert policy.last_decision.reason == "echo"
        assert captured.wake_context.playback == ("Muse is here.", "Hey,", "Muse is here.")
        assert local.received == []
        assert detector.aligned == 0

    asyncio.run(run())


def test_batching_repeated_playback_snapshots_does_not_invent_a_wake_phrase():
    from reachy_muse.conversation import VoiceConversation, _CaptureBuffer

    async def run():
        conversation = VoiceConversation(SimpleNamespace(), SimpleNamespace(echo_cancelled_input=True),
                                         backends=SimpleNamespace(motion=None))
        conversation._muse_audible("Muse is here.", 100)
        conversation._muse_audible("Hey,", 100.1)
        buffer = _CaptureBuffer(16000)
        for _ in range(2):
            buffer.push(np.full(800, .1), wake_context=conversation._wake_capture_context(100.1))
        captured = await buffer.get()
        policy = WakePolicy(WakeVerifier(local=Local()))
        assert await policy.feed(Detector(), captured.samples, context=captured.wake_context) is True
        assert captured.wake_context.playback == ("Muse is here.", "Hey,")

    asyncio.run(run())


@pytest.mark.parametrize("strong, expected, reason", [(True, True, "accepted"), (False, False, "below_threshold")])
def test_companion_down_requires_conservative_replay(strong, expected, reason):
    async def run():
        detector, policy = Detector(strong), WakePolicy()
        assert await policy.feed(detector, np.full(1600, .1)) is expected
        assert (policy.last_decision.reason, policy.last_decision.verifier) == (reason, "conservative_replay")
        assert detector.floors == [.25]
        assert policy.last_decision.replay_floor == (.25 if strong else None)
    asyncio.run(run())


def test_explicit_no_match_cannot_fall_back_to_acoustic_acceptance():
    async def run():
        detector = Detector(True)
        policy = WakePolicy(WakeVerifier(local=Local("hey news")))
        assert await policy.feed(detector, np.full(1600, .1)) is False
        assert policy.last_decision.reason == "verification_rejected"
        assert detector.floors == []
    asyncio.run(run())


def test_capture_gap_during_verification_cannot_authorize_handoff():
    async def run():
        policy = WakePolicy(WakeVerifier(local=Local()))
        assert await policy.feed(Detector(), np.full(1600, .1), valid=lambda: False) is False
        assert policy.last_decision.reason == "capture_gap"
    asyncio.run(run())


def test_cooldown_uses_audio_time_and_idle_does_not_erase_minute_levels(caplog):
    async def run():
        detector, policy = Detector(), WakePolicy()
        assert await policy.feed(detector, np.full(1600, .1)) is True
        policy.reset()
        assert await policy.feed(detector, np.full(1600, .1)) is False
        assert policy.last_decision.reason == "cooldown"
        for _ in range(60):
            assert await policy.feed(detector, np.zeros(16000, np.float32)) is False
        assert await policy.feed(detector, np.full(1600, .1)) is True
    caplog.set_level("INFO", logger="reachy_muse.wake_policy")
    asyncio.run(run())
    levels = [message for message in caplog.messages if message.startswith("Wake levels")]
    assert len(levels) == 1
    assert '"audio_s": 60.2' in levels[0]


def test_queued_and_batched_audio_keeps_capture_time_echo_evidence():
    from reachy_muse.conversation import _CaptureBuffer

    async def run():
        buffer = _CaptureBuffer(16000, capacity_s=3)
        buffer.push(np.full(800, .1), wake_context=WakeContext(speaking=True, playback=("Say hey muse.",)))
        buffer.push(np.full(800, .1), wake_context=WakeContext(playback=("The forecast is sunny.",)))
        captured = await buffer.get()
        assert captured.wake_context == WakeContext(speaking=True, playback=("Say hey muse.", "The forecast is sunny."))
        policy = WakePolicy(WakeVerifier(local=Local()))
        assert await policy.feed(Detector(), captured.samples, context=captured.wake_context) is False
        assert policy.last_decision.reason == "echo"
    asyncio.run(run())


def test_unknown_output_stays_blocked_when_batched_with_known_unrelated_text():
    from reachy_muse.conversation import _CaptureBuffer

    async def run():
        buffer = _CaptureBuffer(16000)
        buffer.push(np.full(800, .1), wake_context=WakeContext(blocked=True, speaking=True))
        buffer.push(np.full(800, .1), wake_context=WakeContext(playback=("A reminder.",)))
        captured = await buffer.get()
        policy = WakePolicy(WakeVerifier(local=Local()))
        assert await policy.feed(Detector(), captured.samples, context=captured.wake_context) is False
        assert policy.last_decision.reason == "speaking"
    asyncio.run(run())


def test_companion_wake_stream_always_aborts_and_forgets_audio():
    from reachy_muse.companion_client import CompanionLink
    from reachy_muse.protocol import CloseHow

    async def run():
        closed = []
        stream = SimpleNamespace(audio=bytearray(), settled=asyncio.Event(), failure=None, endpoint=None,
                                 finish=lambda: None, close=closed.append)
        stream.send = stream.audio.extend
        link = SimpleNamespace(open_hear=lambda: stream)
        task = asyncio.create_task(CompanionLink.transcribe_wake(link, b"\0\0" * 24000))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (closed, stream.audio) == ([CloseHow.ABORT], bytearray())
    asyncio.run(run())
