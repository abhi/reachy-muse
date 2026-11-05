import asyncio
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_muse.capabilities import Mode, Motion
from reachy_muse.companion_client import CompanionMotion
from reachy_muse.conversation import VoiceConversation
from reachy_muse.local_backends import backends_for
from reachy_muse.protocol import Op
from test_conversation import FakeHardware, FakeSession


class Hardware(FakeHardware):
    library_expressions = True

    def __init__(self):
        super().__init__()
        self.sentences = []

    def play_audio(self, samples, *, sentence=None):
        self.sentences.append(sentence)
        self.played.append(samples.copy())

    def finish_sentence(self):
        pass


class Source:
    library_available = True
    available = True

    def __init__(self, delay):
        self.delay = delay
        self.calls = []
        self.cancelled = False

    async def generate(self, intent, duration):
        self.calls.append((intent, duration))
        try:
            await asyncio.sleep(self.delay)
            return Motion(25, ((1., 0., 0., 0., 0., 0., 0., 0., 0.),))
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.mark.parametrize('motion_delay,audio_delay,expected', [(.001, .02, True), (.2, .01, False), (.2, .08, False)])
def test_sentence_motion_never_delays_audio_and_uses_only_timely_result(motion_delay, audio_delay, expected):
    async def scenario():
        source = Source(motion_delay)
        hardware = Hardware()
        session = FakeSession()
        backend = replace(backends_for(Mode.MUSE_VOICE, session), motion=source)
        conversation = VoiceConversation(session, hardware, backends=backend)
        async def speech():
            await asyncio.sleep(audio_delay)
            yield np.ones(80, dtype=np.float32)
        started = asyncio.get_running_loop().time()
        await conversation._speak_owned('id', text='Wonderful!', expression='happy',
                                        prepared_stream=speech(), stream_segment=True, drain=False)
        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed < audio_delay + .08
        assert len(hardware.played) == 1
        assert (hardware.sentences[0].motion is not None) is expected
        assert source.calls == [('happy', 2.5)]
        if not expected:
            await asyncio.sleep(0)
            assert source.cancelled
    asyncio.run(scenario())


def test_library_name_and_live_capability_gate_expression_requests():
    link = SimpleNamespace(models=(("animate", "reachy-motion-library"),), can=lambda op: op is Op.ANIMATE)
    source = CompanionMotion(link)
    assert source.library_available
    link.models = (("animate", "reachy-motion"),)
    assert not source.library_available
    link.models = (("animate", "reachy-motion-library"),)
    link.can = lambda op: False
    assert not source.library_available
