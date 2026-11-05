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


@pytest.mark.timeout(1)
def test_first_audio_arrives_before_pending_motion_is_cancelled_and_late_result_is_ignored():
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()

        class LateSource:
            library_available = available = True
            cancelled = False

            async def generate(self, intent, duration):
                started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    await release.wait()
                finally:
                    finished.set()
                return Motion(25, ((1., 0., 0., 0., 0., 0., 0., 0., 0.),))

        source = LateSource()

        class ObservedHardware(Hardware):
            def play_audio(self, samples, *, sentence=None):
                super().play_audio(samples, sentence=sentence)

        hardware = ObservedHardware()
        session = FakeSession()
        conversation = VoiceConversation(session, hardware, backends=replace(
            backends_for(Mode.MUSE_VOICE, session), motion=source))

        async def speech():
            await started.wait()
            yield np.ones(80, dtype=np.float32)
            release.set()
            await finished.wait()
            yield np.ones(80, dtype=np.float32)

        await conversation._speak_owned('id', text='Hello!', prepared_stream=speech(),
                                        stream_segment=True, drain=False)
        assert source.cancelled
        assert hardware.sentences[0].motion is None
        assert hardware.sentences[1] is None
    asyncio.run(scenario())


def test_timeout_cancels_motion_before_slow_voice_and_uses_50ms(monkeypatch):
    async def scenario():
        timeouts = []
        cancelled = asyncio.Event()
        original = asyncio.wait_for

        async def observed_wait_for(future, timeout):
            timeouts.append(timeout)
            return await original(future, timeout)

        class SlowSource:
            library_available = available = True
            async def generate(self, intent, duration):
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        source = SlowSource()
        hardware = Hardware()
        session = FakeSession()
        conversation = VoiceConversation(session, hardware, backends=replace(
            backends_for(Mode.MUSE_VOICE, session), motion=source))

        async def speech():
            await cancelled.wait()
            await asyncio.sleep(0)
            yield np.ones(80, dtype=np.float32)

        monkeypatch.setattr(asyncio, 'wait_for', observed_wait_for)
        await conversation._speak_owned('id', text='Hello!', prepared_stream=speech(),
                                        stream_segment=True, drain=False)
        assert timeouts == [.05]
        assert hardware.sentences[0].motion is None
    asyncio.run(scenario())


@pytest.mark.parametrize('enabled,available', [(False, True), (True, False), (True, None)])
def test_disabled_or_non_library_source_does_not_request_expression(enabled, available):
    async def scenario():
        source = Source(.001)
        source.library_available = available
        hardware = Hardware()
        hardware.library_expressions = enabled
        session = FakeSession()
        conversation = VoiceConversation(session, hardware, backends=replace(
            backends_for(Mode.MUSE_VOICE, session), motion=source if available is not None else None))

        async def speech():
            await asyncio.sleep(.01)
            yield np.ones(80, dtype=np.float32)

        await conversation._speak_owned('id', text='Hello!', prepared_stream=speech(),
                                        stream_segment=True, drain=False)
        assert source.calls == []
        assert hardware.sentences[0].motion is None
    asyncio.run(scenario())


def test_cancelled_animation_keeps_its_slot_until_companion_answers():
    from reachy_muse.capabilities import CapabilityUnavailable
    from reachy_muse.companion_client import CompanionLink, LinkState, _Connection
    from reachy_muse.protocol import AnimateResult, encode

    async def scenario():
        link = CompanionLink('ws://localhost:9000', token='token', out_rate=16000, insecure=True)
        connection = _Connection(None, (Op.ANIMATE,))
        link._connection = connection
        link._set(LinkState.UP)
        first = asyncio.create_task(link.animate('happy', 2.5))
        second = asyncio.create_task(link.animate('thinking', 4.0))
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        with pytest.raises(CapabilityUnavailable, match='too many'):
            await link.animate('sad', 2.5)
        link._receive(connection, encode(AnimateResult(1, 25)))
        third = asyncio.create_task(link.animate('sad', 2.5))
        await asyncio.sleep(0)
        assert len(connection.animations) == 2
        second.cancel()
        third.cancel()
        await asyncio.gather(second, third, return_exceptions=True)
    asyncio.run(scenario())
