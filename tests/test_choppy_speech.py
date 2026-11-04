"""Playback keeps its samples through slow production and residual microphone echo."""

import asyncio
from dataclasses import replace
import threading
import time

import pytest

from reachy_muse.capabilities import Mode, Partial
from reachy_muse.conversation import VoiceConversation
from reachy_muse.local_backends import LocalHearing, backends_for
from test_conversation import FakeHardware, FakeSession
from test_duplex_voice import duplex
from test_streaming_input import StreamingRecognizer

np = pytest.importorskip("numpy")


@pytest.mark.parametrize("text, pauses", [
    (None, False),
    ("Good", False),
    ("Good question", False),
    ("Here is", False),
    ("What about Paris?", True),
    ("Stop.", True),
])
def test_residual_voice_activity_needs_distinct_words_to_pause(duplex, text, pauses):
    recognizer = StreamingRecognizer()

    class Speech:
        async def stream(self, text, rate):
            yield np.full(rate // 2, .25, np.float32)

    async def run(ctx):
        task = asyncio.create_task(ctx.conversation._speak(
            None, text="Good question, here is the answer.", stream_segment=True))
        try:
            await ctx.playback.get()
            await ctx.feed((.1, 3))
            stream, = recognizer.turns
            if text is not None:
                stream.partial = Partial(text, 1, 1)
            await ctx.feed((.1, 1))
            assert ctx.conversation._playback.user_speaking is pauses
            await ctx.feed((0, 8))
            await task
            assert np.concatenate(ctx.hardware.played).tolist() == [.25] * 4000
            assert ctx.hardware.cleared == 0
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(duplex(run, speech=Speech(), transcriber=recognizer, real_turn=True))


@pytest.mark.parametrize("hearing_mode", ["batch", "muse", "fallback"])
def test_hearing_without_live_partials_pauses_playback_on_voice_activity(duplex, hearing_mode):
    from reachy_muse.companion_client import CompanionHearing, CompanionLink, LinkState, _Connection
    from reachy_muse.failover import CompanionRoute, FailoverHearing
    from reachy_muse.protocol import Op

    class BatchRecognizer:
        async def transcribe(self, wav):
            pytest.fail("A short interruption should not reach batch recognition")

    class Speech:
        async def stream(self, text, rate):
            yield np.full(rate // 2, .25, np.float32)

    async def scenario():
        hearing = LocalHearing(None if hearing_mode == "muse" else BatchRecognizer())
        if hearing_mode == "fallback":
            link = CompanionLink("ws://localhost:9", token="test", out_rate=16000, insecure=True)
            link._connection = _Connection(None, (Op.HEAR,))
            link._set(LinkState.UP)
            hearing = FailoverHearing(CompanionHearing(link), hearing, CompanionRoute(link).hearing)

        async def run(ctx):
            task = asyncio.create_task(ctx.conversation._speak(
                None, text="An answer that must yield to the user.", stream_segment=True))
            try:
                await ctx.playback.get()
                await ctx.feed((.1, 3))
                if hearing_mode == "fallback":
                    assert not ctx.conversation._playback.user_speaking
                    link._connection.fail("disconnected")
                    link._set(LinkState.DOWN)
                    await ctx.feed((.1, 1))
                    await ctx.feed((.1, 1))
                assert ctx.conversation._playback.user_speaking
                played = len(ctx.hardware.played)
                await asyncio.sleep(.02)
                assert len(ctx.hardware.played) == played
                await ctx.feed((0, 8))
                await task
                assert np.concatenate(ctx.hardware.played).tolist() == [.25] * 4000
                assert ctx.session.setup_messages == []
                assert ctx.session.sent == []
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        await duplex(run, speech=Speech(), hearing=hearing, real_turn=True)

    asyncio.run(scenario())


def test_audio_progress_outlives_the_speech_idle_deadline(monkeypatch):
    import reachy_muse.conversation as conversation_module

    monkeypatch.setattr(conversation_module, "SPEECH_TIMEOUT_S", .3)

    class Speech:
        async def stream(self, text, rate):
            for _ in range(10):
                yield np.full(640, .125, np.float32)
                await asyncio.sleep(.015)

    async def run():
        session, hardware = FakeSession(), FakeHardware()
        conversation = VoiceConversation(session, hardware, backends=backends_for(
            Mode.ON_ROBOT, session, speech=Speech()))
        await conversation._speech_with_timeout(conversation._speak(
            None, text="A continuous answer.", stream_segment=True))
        assert np.concatenate(hardware.played).tolist() == [.125] * 6400
        assert hardware.cleared == 0

    asyncio.run(run())


def test_companion_pongs_cannot_hide_a_stalled_speaker_write(monkeypatch):
    import reachy_muse.conversation as conversation_module
    from reachy_muse.local_speech import PreparedSpeech

    monkeypatch.setattr(conversation_module, "SPEECH_TIMEOUT_S", .15)
    entered, release = threading.Event(), threading.Event()

    class Hardware(FakeHardware):
        def play_audio(self, samples):
            entered.set()
            release.wait()
            super().play_audio(samples)

    class Voice:
        speaks_text = True

        def prepare(self, line, rate):
            async def stream():
                yield np.full(800, .25, np.float32)
            return PreparedSpeech(stream(), rate, progress_at=time.monotonic)

    async def run():
        session, hardware = FakeSession(), Hardware()
        backends = replace(backends_for(Mode.ON_ROBOT, session, speech=object()), voice=Voice())
        conversation = VoiceConversation(session, hardware, backends=backends)
        task = asyncio.create_task(conversation._speech_with_timeout(conversation._speak(
            None, text="An answer.", stream_segment=True)))
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            await asyncio.sleep(.35)
            release.set()
            with pytest.raises(TimeoutError, match="stopped making progress"):
                await task
            assert np.concatenate(hardware.played).tolist() == [.25] * 320
            assert hardware.cleared == 1
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
