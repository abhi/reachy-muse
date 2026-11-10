import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_muse import expression_plan as plan
from reachy_muse.conversation import VoiceConversation, _CaptureBuffer
from reachy_muse.music import MusicController
from reachy_muse.wake_policy import WakeContext, WakePolicy, WakeVerifier
from test_music import Sink, constant
from test_registered_commands import command_device
from test_wake_policy import Detector, Local


def test_wake_ducks_before_listening_holds_through_turn_and_hush_stops():
    async def run():
        service, voice, _ = command_device()
        sink = Sink()
        music = voice.music = MusicController(sink.output(), 16000, decoder=constant)
        try:
            music.play({"url": "https://private.invalid/audio"})
            assert not voice._has_output()
            assert not voice._input_blocked()
            assert voice._wake_capture_context(10).music is True
            assert voice._wake_capture_context(10).blocked is False
            original = voice.hardware.set_state
            def listening(state, **kwargs):
                assert music.dancing is False
                original(state, **kwargs)
            voice.hardware.set_state = listening
            await voice._admit_wake(10, strip_required=False)
            assert music.playing and not music.dancing
            voice._wake_deadline = None
            voice._wake_first_request_pending = False
            voice._turn_active = True
            voice._sync_music()
            assert not music.dancing
            voice.hardware.set_state = original
            await voice._apply_cue(plan.SetState(plan.State.IDLE))
            voice._turn_active = False
            voice._sync_music()
            assert music.dancing
            voice._hush()
            assert music.status() == {"playing": False, "elapsed": 0., "dancing": False}
            assert sink.clears == 1
        finally:
            music.close()
    asyncio.run(run())


def test_stronger_acoustic_threshold_applies_even_to_verified_wake():
    async def run():
        detector = Detector()
        policy = WakePolicy(WakeVerifier(local=Local()))
        assert await policy.feed(detector, np.ones(1600, np.float32) * .1, context=WakeContext(music=True))
        assert detector.floors == [.35]
        detector = Detector(strong=False)
        policy.reset(new_trial=True)
        assert not await policy.feed(detector, np.ones(1600, np.float32) * .1, context=WakeContext(music=True))
        assert policy.last_decision.reason == "music_threshold"
    asyncio.run(run())


def test_music_context_survives_capture_batching():
    async def run():
        capture = _CaptureBuffer(16000)
        capture.push(np.zeros(160, np.float32), wake_context=WakeContext(music=True))
        capture.push(np.zeros(160, np.float32), wake_context=WakeContext())
        assert (await capture.get()).wake_context.music is True
    asyncio.run(run())


def test_room_tap_receives_the_only_microphone_read_even_when_speech_is_blocked():
    async def run():
        reads, tapped = [], []
        def read():
            time.sleep(.005)
            reads.append(threading.current_thread().name)
            return np.full(160, .125, np.float32)
        def tap(pcm, at, *, gap):
            tapped.append((threading.current_thread().name, float(pcm[0]), len(pcm)))
        voice = SimpleNamespace(hardware=SimpleNamespace(read_audio=read, sample_rate=16000),
                                music=SimpleNamespace(tap=tap), _input_blocked=lambda: True, _wake_policy=None)
        capture = _CaptureBuffer(16000)
        task = asyncio.create_task(VoiceConversation._capture_microphone(voice, capture))
        try:
            chunk = await asyncio.wait_for(capture.get(), 1)
            assert chunk.discard is True
            assert set(reads) == {"reachy-microphone_0"}
            assert tapped and all(row == ("reachy-microphone_0", .125, 160) for row in tapped)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert len(tapped) == len(reads)
    asyncio.run(run())


def test_notification_holds_music_before_wave_and_releases_after_speech(monkeypatch, paired_inbox):
    from test_notifications import device, notify
    from test_inbox import notice
    monkeypatch.setattr("reachy_muse.notifications.ATTENTION_S", .01)
    async def run():
        service, voice, _ = device()
        sink = Sink()
        music = voice.music = MusicController(sink.output(), 16000, decoder=constant)
        try:
            music.play({"url": "https://private.invalid/audio"})
            seen = []
            voice.hardware.gesture = lambda name: seen.append((name, music.dancing))
            service._presence.mark("voice", time.monotonic())
            result = await notify(service, **{**notice(), "mode": "now"})
            assert result == {"ok": True, "payload": {"status": "delivered", "reason": "spoken"}}
            assert seen == [("wave", False)]
            assert music.playing
            voice._sync_music()
            assert music.dancing
        finally:
            music.close()
    asyncio.run(run())
