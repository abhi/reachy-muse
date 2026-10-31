import asyncio
import logging

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("av")

from bench.latency_replay import OfflineSession
from reachy_muse.capabilities import Backends, Mode
from reachy_muse.companion_client import CompanionLink
from reachy_muse.contract.replies import ReplyStyle
from reachy_muse.conversation import VoiceConversation
from reachy_muse.local_backends import LocalHearing, backends_for
from reachy_muse.startup import StartupTimeline
from test_conversation import FakeHardware
from test_latency_rsi import BlockedVoice


@pytest.mark.parametrize("fail", [False, True])
def test_ready_microphone_detects_wake_during_stalled_or_failed_cache_warming(caplog, fail):
    caplog.set_level(logging.INFO)

    class Wake:
        phrase = "hey muse"

        def __init__(self):
            self.heard = []

        def reset(self):
            pass

        def feed(self, samples):
            self.heard.append(float(samples[0]))
            return True

    async def scenario():
        session, hardware, wake = OfflineSession(), FakeHardware(), Wake()
        voice, progress = BlockedVoice(), BlockedVoice()
        backends = Backends(hearing=LocalHearing(), voice=voice, narrator=None,
                            progress_voice=progress, local_style=ReplyStyle.MARKER)
        conversation = VoiceConversation(session, hardware, backends=backends, owns_chat=True,
                                         wake_detector=wake)
        running = asyncio.create_task(conversation.run())
        try:
            await asyncio.wait_for(conversation._ready.wait(), 1)
            await asyncio.wait_for(asyncio.gather(voice.started.wait(), progress.started.wait()), 1)
            if fail:
                voice.error = ValueError("bad model")
                voice.finish.set()
                await asyncio.wait_for(progress.closed.wait(), 1)
            hardware.samples.put(np.full(320, .75, dtype=np.float32))

            async def heard_wake():
                while "Reachy heard its wake phrase; microphone open" not in caplog.text:
                    if running.done():
                        running.result()
                    await asyncio.sleep(.001)
            await asyncio.wait_for(heard_wake(), 1)
            assert wake.heard == [.75]
            assert conversation._wake_strip_required is True
            assert not running.done()
            assert "stage=chat_subscription" in caplog.text
            assert "stage=hearing" in caplog.text
            assert "stage=microphone_ready" in caplog.text
            if fail:
                assert "live synthesis remains available" in caplog.text
            else:
                assert not voice.closed.is_set() and not progress.closed.is_set()
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        assert voice.closed.is_set() and progress.closed.is_set()
        assert session.subscription_closed.is_set()
    asyncio.run(scenario())


def test_startup_timing_reports_stage_and_total_seconds(monkeypatch, caplog):
    ticks = iter([100.0, 101.25, 104.5])
    monkeypatch.setattr("reachy_muse.startup.time.monotonic", lambda: next(ticks))
    caplog.set_level(logging.INFO, logger="reachy_muse.startup")
    timeline = StartupTimeline()
    timeline.mark("hardware")
    timeline.mark("microphone_ready")
    assert caplog.messages == [
        "Reachy startup began",
        "Reachy startup stage=hardware elapsed=1.250s total=1.250s",
        "Reachy startup stage=microphone_ready elapsed=3.250s total=4.500s",
    ]


def test_connecting_companion_cannot_delay_ready_or_local_recording():
    async def scenario():
        # No transport is run: this companion stays CONNECTING for the whole test.
        link = CompanionLink("wss://localhost:9", token="test", out_rate=16000,
                             pin="sha256:" + "00" * 32)
        session, hardware = OfflineSession(), FakeHardware()
        backends = backends_for(Mode.COMPANION, session, companion=link)
        conversation = VoiceConversation(session, hardware, backends=backends, owns_chat=True)
        running = asyncio.create_task(conversation.run())
        try:
            await asyncio.wait_for(conversation._ready.wait(), 1)
            assert hardware.states[-1] == "idle"
            assert link.state.value == "connecting"
            assert backends.hearing.companion_ready is False
            # The same hearing instance already has a functioning local recorder.
            class Vad:
                def is_speech(self, pcm, rate):
                    return bool(np.any(np.frombuffer(pcm, dtype="<i2")))
            turn = backends.hearing.open_turn(16000, silence_s=.1, vad=Vad())
            turn.feed(np.full(8000, .3, dtype=np.float32))
            turn.feed(np.zeros(4000, dtype=np.float32))
            endpoint = await turn.take().ended.endpoint()
            assert (endpoint.text, endpoint.audio[:4]) == (None, b"RIFF")
            turn.abort()
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
    asyncio.run(scenario())
