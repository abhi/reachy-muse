"""CPU-only playback of prepared sentences and explicit voice restart boundaries."""

import asyncio
from dataclasses import replace

import numpy as np

from reachy_muse.capabilities import SpokenLine
from reachy_muse.local_speech import SpeechRestart
from test_expressive_playback import setup


def test_narrated_reply_prepares_the_following_sentence_while_current_audio_plays(monkeypatch):
    async def scenario():
        clock, hardware, conversation = setup(monkeypatch)
        events = []
        lock = asyncio.Lock()

        class Voice:
            speaks_text = True

            def prepare(self, line, rate):
                return None

            async def stream(self, line, rate):
                async with lock:
                    events.append(("synthesize", line.text, line.muse_message_id))
                    yield np.full(8, 1 if line.text == "First." else 2, dtype=np.float32)

        conversation.backends = replace(conversation.backends, voice=Voice())
        original = hardware.play_audio

        def push(samples, **kwargs):
            events.append(("audio", samples.tolist()))
            original(samples, **kwargs)

        hardware.play_audio = push
        await conversation._speak_lines("reply", [SpokenLine("First.", None, "answer"),
                                                  SpokenLine("Second.", None, "answer")])
        assert events[:2] == [("synthesize", "First.", "reply"),
                              ("synthesize", "Second.", "reply")]
        assert [item for item in events if item[0] == "audio"] == [
            ("audio", [1, 1, 1, 1]), ("audio", [1, 1, 1, 1]),
            ("audio", [2, 2, 2, 2]), ("audio", [2, 2, 2, 2])]
        assert [item[1] for item in hardware.schedule if item[0] == "audio"] == [0, 0, .04, .08]
        assert hardware.cleared == 0

    asyncio.run(scenario())


def test_dead_link_restart_drains_previous_voice_before_repeating_sentence(monkeypatch):
    async def scenario():
        clock, hardware, conversation = setup(monkeypatch)

        class Voice:
            speaks_text = True

            def prepare(self, line, rate):
                return None

            async def stream(self, line, rate):
                yield np.array([1, 2, 3, 4], dtype=np.float32)
                yield SpeechRestart()
                yield np.array([5, 6, 7, 8], dtype=np.float32)

        conversation.backends = replace(conversation.backends, voice=Voice())
        heard = []
        await conversation._speak("reply", text="Complete sentence.", stream_segment=True,
                                  on_start=lambda: heard.append("Complete sentence."))
        assert heard == ["Complete sentence."]
        assert [(item[1], item[2]) for item in hardware.schedule if item[0] == "audio"] == [
            (0, [1, 2, 3, 4]), (.09, [5, 6, 7, 8])]
        assert hardware.cleared == 0

    asyncio.run(scenario())


def test_slow_companion_source_keeps_waiting_while_its_link_makes_progress(monkeypatch):
    import time
    from reachy_muse import conversation as conversation_module
    from reachy_muse.capabilities import Mode
    from reachy_muse.conversation import VoiceConversation
    from reachy_muse.failover import CompanionRoute, FailoverVoice
    from reachy_muse.local_backends import backends_for
    from test_conversation import FakeHardware

    monkeypatch.setattr(conversation_module, "SPEECH_TIMEOUT_S", .08)

    class Link:
        up = True

        @property
        def last_progress_at(self):
            return time.monotonic()

    class Companion:
        available = True

        async def stream(self, line, rate):
            await asyncio.sleep(.18)
            yield np.full(80, .5, np.float32)

    class Local:
        async def stream(self, line, rate):
            yield np.full(80, -.5, np.float32)

    async def scenario():
        route = CompanionRoute(Link())
        route.begin_turn()
        voice = FailoverVoice(Companion(), Local(), route)
        hardware = FakeHardware()
        conversation = VoiceConversation(None, hardware, backends=replace(
            backends_for(Mode.ON_ROBOT, None, speech=object()), voice=voice, companion=route))
        await conversation._speech_with_timeout(conversation._speak(
            "reply", text="A slow, healthy reply.", stream_segment=True))
        assert np.concatenate(hardware.played).tolist() == [.5] * 80
        assert hardware.cleared == 0
        assert conversation._speech_source is None

    asyncio.run(scenario())


def test_next_replys_live_companion_cannot_hide_a_stalled_local_source(monkeypatch):
    import time
    import pytest
    from reachy_muse import conversation as conversation_module
    from reachy_muse.capabilities import Mode
    from reachy_muse.conversation import VoiceConversation
    from reachy_muse.failover import CompanionRoute, FailoverVoice
    from reachy_muse.local_backends import backends_for
    from test_conversation import FakeHardware

    monkeypatch.setattr(conversation_module, "SPEECH_TIMEOUT_S", .08)

    class Link:
        up = False

        @property
        def last_progress_at(self):
            return time.monotonic()

    async def scenario():
        local_started, local_closed, release_remote = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Local:
            async def stream(self, line, rate):
                local_started.set()
                try:
                    await asyncio.Event().wait()
                    yield np.full(80, -.5, np.float32)
                finally:
                    local_closed.set()

        class Companion:
            available = True

            async def stream(self, line, rate):
                await release_remote.wait()
                yield np.full(80, .75, np.float32)

        link = Link()
        route = CompanionRoute(link)
        route.begin_turn()
        voice = FailoverVoice(Companion(), Local(), route)
        current = voice.prepare(SpokenLine("Local reply.", None, "answer", "old"), 8000)
        await local_started.wait()
        link.up = True
        following = voice.prepare(SpokenLine("Recovered reply.", None, "answer", "new"), 8000)
        hardware = FakeHardware()
        conversation = VoiceConversation(None, hardware, backends=replace(
            backends_for(Mode.ON_ROBOT, None, speech=object()), voice=voice, companion=route))
        try:
            with pytest.raises(TimeoutError, match="stopped making progress"):
                await conversation._speech_with_timeout(conversation._speak(
                    "old", text="Local reply.", stream_segment=True, prepared_stream=current))
            assert local_closed.is_set()
            assert hardware.played == []
            release_remote.set()
            assert np.concatenate([chunk async for chunk in following]).tolist() == [.75] * 80
        finally:
            await current.aclose()
            await following.aclose()

    asyncio.run(scenario())


def test_reply_arriving_during_final_drain_does_not_announce_idle_between_sentences(monkeypatch):
    from reachy_muse.conversation import _SpeechJob
    from reachy_muse.expression_plan import OutputIdle

    async def scenario():
        _, hardware, conversation = setup(monkeypatch)
        idle = asyncio.Event()
        states = []
        drains = 0
        original_drain = conversation._drain_speech

        class Voice:
            speaks_text = True

            def prepare(self, line, rate):
                return None

            async def stream(self, line, rate):
                yield np.full(8, 1 if line.text == "First." else 2, np.float32)

        conversation.backends = replace(conversation.backends, voice=Voice())

        async def drain(cursor):
            nonlocal drains
            drains += 1
            if drains == 1:
                await conversation._output_queue.put(_SpeechJob("reply", "Second."))
                await asyncio.sleep(0)
            await original_drain(cursor)

        async def plan(event):
            if isinstance(event, OutputIdle):
                states.append(np.concatenate(hardware.played).tolist())
                idle.set()

        conversation._drain_speech = drain
        conversation._plan = plan
        await conversation._output_queue.put(_SpeechJob("reply", "First."))
        speaker = asyncio.create_task(conversation._play_output())
        try:
            await asyncio.wait_for(idle.wait(), 1)
            assert states == [[1] * 8 + [2] * 8]
            assert hardware.cleared == 0
        finally:
            speaker.cancel()
            await asyncio.gather(speaker, return_exceptions=True)

    asyncio.run(scenario())
