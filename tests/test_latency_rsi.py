"""Offline reproductions of measured startup waste, with no live transports."""

import asyncio
import json
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("av")

from bench.latency_replay import OfflineSession, ReplaySpeech, SilentHardware, startup
from musegadget.identity import Identity
from musegadget.link_client import Outcome
from reachy_muse.capabilities import Backends, Mode, SpokenLine
from reachy_muse.contract.replies import ReplyStyle
from reachy_muse.conversation import ReachyService, VoiceConversation
from reachy_muse.local_backends import LocalVoice, backends_for
from reachy_muse.local_speech import PiperSpeech
from test_local_speech import worker  # noqa: F401


# Playback digests from 06fd273, before fixed-phrase caching.
REPLAY_PCM_SHA256 = "604e5162d4d9936bd0d096bbf4c55348ad89947006ff966efd7578d2a298968f"
PIPER_PCM_SHA256 = "e6b58ee72dcfafbd99241948aed0cb6a2d8439bd2310e1140ef0a005aaba0f36"


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "fake.onnx"
    path.write_bytes(b"never loaded")
    path.with_suffix(".onnx.json").write_text(json.dumps({"audio": {"sample_rate": 22050}}))
    return path


def test_reconnection_reaches_ready_with_baseline_audio_and_no_new_synthesis(model):
    async def scenario():
        speech, progress = ReplaySpeech(model, 0), ReplaySpeech(model, 0)
        _, before = await startup(speech, progress)
        assert before == REPLAY_PCM_SHA256
        assert (len(speech.requests), len(progress.requests)) == (14, 35)
        _, after = await startup(speech, progress)
        assert after == REPLAY_PCM_SHA256
        assert (len(speech.requests), len(progress.requests)) == (14, 35)
    asyncio.run(scenario())


def test_cancelled_speech_restarts_worker_before_reconnect_is_ready(model, worker):
    async def scenario():
        speech, progress = PiperSpeech(model), PiperSpeech(model)
        stream = None
        try:
            _, before = await startup(speech, progress)
            assert before == PIPER_PCM_SHA256
            assert len(worker[1]) == 2
            process = speech._process
            stream = speech.stream("Cancelled reply.", 48000)
            await asyncio.wait_for(stream.__anext__(), 3)
            await stream.aclose()
            assert process.returncode == -9
            _, after = await startup(speech, progress)
            assert len(worker[1]) == 3
            assert worker[1][-1].returncode is None
            assert worker[2].read_text() == "Cancelled reply."
            assert after == PIPER_PCM_SHA256
            stream = speech.stream("An uncached answer.", 48000)
            samples = [chunk async for chunk in stream]
            assert sum(len(chunk) for chunk in samples) == 4800
            assert worker[2].read_text() == "An uncached answer."
            assert len(worker[1]) == 3
        finally:
            if stream is not None:
                await stream.aclose()
            await asyncio.gather(speech.close(), progress.close())
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
        for voice in (first.voice, second.voice):
            samples = [chunk async for chunk in voice.stream(SpokenLine("Yes?", None, "notice"), 16000)]
            np.testing.assert_array_equal(np.concatenate(samples), np.full(160, 0.71875, dtype=np.float32))
    asyncio.run(scenario())


def test_recordings_are_scoped_to_the_voice_and_output_rate(model):
    async def scenario():
        speech = ReplaySpeech(model, 0)
        for rate, count in ((16000, 160), (48000, 480), (16000, 160)):
            voice = LocalVoice(speech, ("Yes?",))
            await voice.warm(rate)
            samples = [chunk async for chunk in voice.stream(SpokenLine("Yes?", None, "notice"), rate)]
            np.testing.assert_array_equal(np.concatenate(samples), np.full(count, 0.71875, dtype=np.float32))
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
        for text, value in (("Yes?", 0.71875), ("On it.", 0.23828125)):
            samples = [chunk async for chunk in voice.stream(SpokenLine(text, None, "notice"), 16000)]
            np.testing.assert_array_equal(np.concatenate(samples), np.full(160, value, dtype=np.float32))
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


@pytest.mark.parametrize("piper_voice", ["main", "progress"])
@pytest.mark.parametrize("cancellations", [1, 3])
def test_cancelled_warmup_reaps_worker_spawned_during_sibling_cleanup(model, worker, monkeypatch,
                                                                      piper_voice, cancellations):
    async def scenario():
        spawn_started, start_spawn = asyncio.Event(), asyncio.Event()
        process_created, closing, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original = asyncio.create_subprocess_exec

        async def delayed(*args, **kwargs):
            spawn_started.set()
            await start_spawn.wait()
            process = await original(*args, **kwargs)
            process_created.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return process

        class ClosingVoice:
            async def warm(self, rate):
                try:
                    await asyncio.Future()
                finally:
                    closing.set()
                    await process_created.wait()
                    closed.set()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed)
        speech = PiperSpeech(model)
        voice, progress = LocalVoice(speech), ClosingVoice()
        if piper_voice == "progress":
            voice, progress = progress, voice
        backends = Backends(hearing=None, voice=voice, narrator=None, progress_voice=progress,
                            local_style=ReplyStyle.MARKER)
        conversation = VoiceConversation(None, SimpleNamespace(output_sample_rate=16000), backends=backends)
        warming = asyncio.create_task(conversation.warm_up())
        try:
            await asyncio.wait_for(spawn_started.wait(), 3)
            warming.cancel()
            await asyncio.wait_for(closing.wait(), 3)
            assert not warming.done()
            for _ in range(1, cancellations):
                warming.cancel()
                await asyncio.sleep(0)
            start_spawn.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(warming, 3)
            assert closed.is_set()
            assert len(worker[1]) == 1
            assert worker[1][0].returncode == -9
            assert speech._process is None
        finally:
            start_spawn.set()
            process_created.set()
            if not warming.done():
                warming.cancel()
            await asyncio.gather(warming, return_exceptions=True)
            await speech.close()
            for process in worker[1]:
                if process.returncode is None:
                    process.kill()
                await process.communicate()
    asyncio.run(scenario())


@pytest.mark.parametrize("piper_voice", ["main", "progress"])
@pytest.mark.parametrize("shutdown", ["stop", "link_loss"])
def test_service_shutdown_during_failed_warmup_reaps_spawning_worker(model, worker, monkeypatch,
                                                                     piper_voice, shutdown):
    async def scenario():
        process_created, release_process = asyncio.Event(), asyncio.Event()
        stop, disconnected = asyncio.Event(), asyncio.Event()
        original = asyncio.create_subprocess_exec

        async def delayed(*args, **kwargs):
            process = await original(*args, **kwargs)
            process_created.set()
            await release_process.wait()
            return process

        class SpawningVoice(LocalVoice):
            async def warm(self, rate):
                self.warming = asyncio.current_task()
                await super().warm(rate)

        class OfflineLink(OfflineSession):
            async def run(self, stop_link):
                self.stop_link = stop_link
                if shutdown == "link_loss":
                    await disconnected.wait()
                    return Outcome.CLOSED
                await stop_link.wait()
                return Outcome.STOPPED

        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed)
        speech = PiperSpeech(model)
        piper, failing = SpawningVoice(speech), BlockedVoice()
        failing.error = ValueError("synthesis failed")
        voice, progress = (piper, failing) if piper_voice == "main" else (failing, piper)
        backends = Backends(hearing=None, voice=voice, narrator=None, progress_voice=progress,
                            local_style=ReplyStyle.MARKER)
        session = OfflineLink()
        service = ReachyService(identity=Identity("02:00:00:ab:cd:ef"), executor=SilentHardware(),
                               backends=lambda session: backends)
        serving = asyncio.create_task(service.serve(session, {"vm_id": "vm-1"}, stop))

        async def wait_for_cleanup():
            while not piper.warming.cancelling():
                await asyncio.sleep(0)
            await asyncio.sleep(0)

        try:
            await asyncio.wait_for(process_created.wait(), 3)
            assert worker[1][0].returncode is None
            failing.finish.set()
            await asyncio.wait_for(wait_for_cleanup(), 3)
            (stop if shutdown == "stop" else disconnected).set()
            await asyncio.wait_for(session.stop_link.wait(), 3)
            release_process.set()
            outcome = await asyncio.wait_for(serving, 3)
            assert outcome is (Outcome.STOPPED if shutdown == "stop" else Outcome.CLOSED)
            assert worker[1][0].returncode == -9
            assert session.subscription_closed.is_set()
        finally:
            release_process.set()
            stop.set()
            disconnected.set()
            if not serving.done():
                serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)
            await speech.close()
            for process in worker[1]:
                if process.returncode is None:
                    process.kill()
                await process.communicate()
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
