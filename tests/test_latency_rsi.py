"""Offline reproductions of measured startup waste, with no live transports."""

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("av")

from bench.latency_replay import OfflineSession, ReplaySpeech, SilentHardware, startup
from reachy_muse.capabilities import Backends, Mode, SpokenLine
from reachy_muse.contract.replies import ReplyStyle
from reachy_muse.conversation import VoiceConversation
from reachy_muse.local_backends import LocalVoice, backends_for


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "fake.onnx"
    path.write_bytes(b"never loaded")
    path.with_suffix(".onnx.json").write_text(json.dumps({"audio": {"sample_rate": 22050}}))
    return path


def test_reconnection_reaches_ready_with_identical_audio_and_no_new_synthesis(model):
    async def scenario():
        speech, progress = ReplaySpeech(model, 0), ReplaySpeech(model, 0)
        _, before = await startup(speech, progress)
        assert (len(speech.requests), len(progress.requests)) == (14, 35)
        _, after = await startup(speech, progress)
        assert after == before
        assert (len(speech.requests), len(progress.requests)) == (14, 35)
    asyncio.run(scenario())


def test_companion_fallback_reuses_recordings_but_not_route_or_narrator_state(model):
    async def scenario():
        speech = ReplaySpeech(model, 0)
        first = backends_for(Mode.COMPANION, object(), speech=speech, companion=object(), wake=True)
        await first.voice.warm(16000)
        second = backends_for(Mode.COMPANION, object(), speech=speech, companion=object(), wake=True)
        await second.voice.warm(16000)
        assert len(speech.requests) == 14
        assert first.companion is not second.companion
        assert first.narrator.local is not second.narrator.local
        assert second.voice.is_presynthesized("Yes?")
    asyncio.run(scenario())


def test_recordings_are_scoped_to_the_voice_and_output_rate(model):
    async def scenario():
        speech = ReplaySpeech(model, 0)
        for rate in (16000, 48000, 16000):
            voice = LocalVoice(speech, ("Yes?",))
            await voice.warm(rate)
            samples = [chunk async for chunk in voice.stream(SpokenLine("Yes?", None, "notice"), rate)]
            assert sum(len(chunk) for chunk in samples) == rate // 100
        assert speech.requests == [("Yes?", 16000), ("Yes?", 48000)]
        other = ReplaySpeech(model, 0)
        await LocalVoice(other, ("Yes?",)).warm(16000)
        assert other.requests == [("Yes?", 16000)]
        assert other.phrase_cache is not speech.phrase_cache
    asyncio.run(scenario())


def test_cancelled_warmup_keeps_completed_phrases_and_retries_incomplete_audio(model):
    class InterruptedSpeech(ReplaySpeech):
        def __init__(self):
            super().__init__(model, 0)
            self.block = True
            self.partial = asyncio.Event()

        async def stream(self, text, rate):
            async for chunk in super().stream(text, rate):
                yield chunk
            if text == "On it." and self.block:
                self.partial.set()
                await asyncio.Future()

    async def scenario():
        speech = InterruptedSpeech()
        warming = asyncio.create_task(LocalVoice(speech, ("Yes?", "On it.")).warm(16000))
        await asyncio.wait_for(speech.partial.wait(), 1)
        warming.cancel()
        with pytest.raises(asyncio.CancelledError):
            await warming
        speech.block = False
        voice = LocalVoice(speech, ("Yes?", "On it."))
        await voice.warm(16000)
        assert speech.requests == [("Yes?", 16000), ("On it.", 16000), ("On it.", 16000)]
        assert set(voice.phrases) == {"Yes?", "On it."}
    asyncio.run(scenario())


class BlockedVoice:
    def __init__(self):
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.closed = asyncio.Event()
        self.error = None

    async def warm(self, rate):
        self.started.set()
        try:
            await self.finish.wait()
            if self.error:
                raise self.error
        finally:
            self.closed.set()


def test_cold_start_warms_both_voices_together_and_waits_for_both():
    async def scenario():
        voice, progress = BlockedVoice(), BlockedVoice()
        backends = Backends(hearing=None, voice=voice, narrator=None, progress_voice=progress,
                            local_style=ReplyStyle.MARKER)
        conversation = VoiceConversation(None, SimpleNamespace(output_sample_rate=16000), backends=backends)
        warming = asyncio.create_task(conversation.warm_up())
        try:
            await asyncio.wait_for(asyncio.gather(voice.started.wait(), progress.started.wait()), 1)
            voice.finish.set()
            await voice.closed.wait()
            assert not warming.done()
            progress.finish.set()
            await warming
            assert voice.closed.is_set() and progress.closed.is_set()
        finally:
            warming.cancel()
            await asyncio.gather(warming, return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["main", "progress", "cancel"])
def test_failed_or_cancelled_startup_joins_every_warmup(failure):
    async def scenario():
        voice, progress = BlockedVoice(), BlockedVoice()
        backends = Backends(hearing=None, voice=voice, narrator=None, progress_voice=progress,
                            local_style=ReplyStyle.MARKER)
        conversation = VoiceConversation(None, SimpleNamespace(output_sample_rate=16000), backends=backends)
        warming = asyncio.create_task(conversation.warm_up())
        await asyncio.wait_for(asyncio.gather(voice.started.wait(), progress.started.wait()), 1)
        if failure == "cancel":
            warming.cancel()
        else:
            failing = voice if failure == "main" else progress
            failing.error = ValueError("synthesis failed")
            failing.finish.set()
        with pytest.raises(asyncio.CancelledError if failure == "cancel" else ValueError):
            await warming
        assert voice.closed.is_set() and progress.closed.is_set()
    asyncio.run(scenario())


@pytest.mark.parametrize("fail", [False, True])
def test_subscribe_overlaps_warmup_and_is_closed_if_startup_stops(fail):
    class NoCaptureHardware(SilentHardware):
        def read_audio(self):
            raise AssertionError("capture started before warm-up finished")

    async def scenario():
        session, voice = OfflineSession(), BlockedVoice()
        backends = Backends(hearing=None, voice=voice, narrator=None, local_style=ReplyStyle.MARKER)
        conversation = VoiceConversation(session, NoCaptureHardware(), backends=backends, owns_chat=True)
        running = asyncio.create_task(conversation.run())
        try:
            await asyncio.wait_for(session.subscription_started.wait(), 1)
            await asyncio.wait_for(voice.started.wait(), 1)
            assert not running.done()
            if fail:
                voice.error = ValueError("synthesis failed")
                voice.finish.set()
            else:
                running.cancel()
            with pytest.raises(ValueError if fail else asyncio.CancelledError):
                await running
            assert session.subscription_closed.is_set()
            assert voice.closed.is_set()
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
    asyncio.run(scenario())
