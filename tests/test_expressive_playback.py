import asyncio

import numpy as np
import pytest

import reachy_muse.conversation as conversation_module
from reachy_muse.capabilities import Mode
from reachy_muse.conversation import VoiceConversation, _SpeechSuperseded
from reachy_muse.local_backends import backends_for
from reachy_muse.speech_playback import PCMPlayer, SpeechPlayback
from test_conversation import FakeHardware
from test_speech_playback import Clock


class SentenceHardware(FakeHardware):
    output_sample_rate = 100

    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.schedule = []
        self.after_push = lambda: None

    def play_audio(self, samples, *, sentence=None):
        self.schedule.append(("audio", round(self.clock(), 3), samples.tolist(),
                              (sentence.expression.value, sentence.terminal) if sentence else None))
        super().play_audio(samples)
        self.after_push()

    def finish_sentence(self):
        self.schedule.append(("end", round(self.clock(), 3)))


class Speech:
    async def stream(self, text, rate):
        yield np.arange(8, dtype=np.float32) / 8


def setup(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(conversation_module, "PCMPlayer", lambda *args, **kwargs:
                        PCMPlayer(*args, **kwargs, clock=clock, sleep=clock.sleep))
    hardware = SentenceHardware(clock)
    conversation = VoiceConversation(None, hardware,
        backends=backends_for(Mode.ON_ROBOT, None, speech=Speech()))
    conversation._playback = SpeechPlayback(clock=clock)
    return clock, hardware, conversation


def test_sentence_expression_goes_with_first_pcm_and_end_follows_last_pcm(monkeypatch):
    async def scenario():
        clock, hardware, conversation = setup(monkeypatch)
        await conversation._speak("reply", text="Would you like more?", stream_segment=True,
            on_start=lambda: hardware.schedule.append(("heard", len(hardware.played))))
        assert hardware.schedule == [
            ("audio", 0, [0, .125, .25, .375], ("curious", "question")),
            ("heard", 1),
            ("audio", 0, [.5, .625, .75, .875], None),
            ("end", .04),
        ]
        assert clock() == pytest.approx(.13)
        assert hardware.states == []
    asyncio.run(scenario())


def test_pause_keeps_sentence_identity_and_does_not_repeat_its_onset(monkeypatch):
    async def scenario():
        clock, hardware, conversation = setup(monkeypatch)
        paused = asyncio.Event()

        def hold():
            if len(hardware.played) == 1:
                conversation._playback.set_user_speaking(True)

        hardware.after_push = hold
        original = hardware.set_state

        def state(value, **kwargs):
            original(value, **kwargs)
            if value == "listening":
                paused.set()

        hardware.set_state = state
        speaking = asyncio.create_task(conversation._speak("reply", text="Wonderful!", stream_segment=True))
        await paused.wait()
        clock.now = 1
        conversation._playback.set_user_speaking(False)
        await speaking
        assert hardware.schedule == [
            ("audio", 0, [0, .125, .25, .375], ("happy", "exclamation")),
            ("audio", 1, [.5, .625, .75, .875], None),
            ("end", 1),
        ]
        assert hardware.state_expressions == [("listening", None), ("speaking", None)]
    asyncio.run(scenario())


def test_revision_while_waiting_for_quiet_drops_old_sentence_and_its_motion(monkeypatch):
    async def scenario():
        _, hardware, conversation = setup(monkeypatch)
        current = [True]
        conversation._playback.set_user_speaking(True)
        speaking = asyncio.create_task(conversation._speak("reply", text="Wonderful!", stream_segment=True,
                                                          is_current=lambda: current[0]))
        await asyncio.sleep(0)
        current[0] = False
        conversation._playback.set_user_speaking(False)
        with pytest.raises(_SpeechSuperseded):
            await speaking
        await conversation._speak("replacement", text="I'm sorry.", stream_segment=True)
        assert hardware.schedule == [
            ("audio", 0, [0, .125, .25, .375], ("sad", "plain")),
            ("audio", 0, [.5, .625, .75, .875], None),
            ("end", .04),
        ]
        assert hardware.cleared == 1
    asyncio.run(scenario())
