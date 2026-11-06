"""A visual invocation opens the existing conversation, without spoken keywords."""

import asyncio

import pytest

from test_device_command_turns import played, rig_with, run, sent_to_muse, service
from test_pins import Transcripts, Wake


class WaveCamera:
    def __init__(self, hardware):
        self.active = False
        self.pending = False
        self.transitions = []
        hardware.set_wave_watching = self.set_active
        hardware.take_wave = self.take

    def set_active(self, active):
        if active != self.active:
            self.transitions.append(active)
        self.active = active
        if not active:
            self.pending = False

    def take(self):
        result, self.pending = self.active and self.pending, False
        return result


@pytest.mark.parametrize("acoustic", [False, True])
def test_wave_greets_and_only_fresh_speech_reaches_muse(monkeypatch, acoustic):
    rig = rig_with(monkeypatch)
    camera = WaveCamera(rig.hardware)
    transcripts = Transcripts("What is the weather tomorrow?")

    async def muse(session, request):
        await session.answer(request, "Sunny tomorrow.")

    options = {"wake_detector": Wake()} if acoustic else {}
    conversation = service(rig, muse, transcripts, wave_wake=True, wake_timeout_s=30, **options)

    async def user():
        assert camera.active is True
        rig.say()  # Ambient speech before the gesture must never become a request.
        await rig.quiet()
        assert transcripts.texts == ["What is the weather tomorrow?"]
        camera.pending = True
        await rig.wait_until_or_timeout(lambda: "tts:Yes?" in played(rig.recording.events))
        await rig.quiet("listening", "idle")
        assert camera.active is False
        assert sent_to_muse(rig.recording.events) == []
        rig.say()

    events = run(rig, conversation, user, rest=("idle", "listening"))
    assert sent_to_muse(events) == ["What is the weather tomorrow?"]
    assert played(events) == ["tts:Yes?", "tts:Sunny tomorrow."]
    assert camera.transitions == [True, False]
    assert {"robot": "set_state", "state": "listening", "expression": "happy"} in events


def test_wave_wakes_sleep_once_without_a_speech_request(monkeypatch):
    rig = rig_with(monkeypatch)
    camera = WaveCamera(rig.hardware)
    transcripts = Transcripts("Hey Muse, go to sleep.")

    async def muse(session, request):
        raise AssertionError("A wave is not a request to Muse")

    conversation = service(rig, muse, transcripts, wave_wake=True, wake_detector=Wake())

    async def user():
        rig.say(wake=True)
        await rig.quiet("sleeping")
        assert camera.active is True
        camera.pending = True

    events = run(rig, conversation, user, rest=("listening", "idle"))
    assert played(events) == ["tts:Good night.", "tts:I'm awake."]
    assert sent_to_muse(events) == []
    assert camera.transitions == [True, False, True, False]


def test_wave_only_window_expires_and_requires_another_wave(monkeypatch):
    rig = rig_with(monkeypatch)
    camera = WaveCamera(rig.hardware)
    transcripts = Transcripts("Uninvited speech")

    async def muse(session, request):
        raise AssertionError("The expired visual window must stay closed")

    conversation = service(rig, muse, transcripts, wave_wake=True, wake_timeout_s=.3)

    async def user():
        camera.pending = True
        await rig.wait_until_or_timeout(lambda: "tts:Yes?" in played(rig.recording.events))
        await rig.quiet()
        assert camera.active is True
        rig.say()

    events = run(rig, conversation, user)
    assert played(events) == ["tts:Yes?"]
    assert sent_to_muse(events) == []
    assert transcripts.texts == ["Uninvited speech"]


def test_pre_wave_in_flight_microphone_read_is_discarded():
    import threading
    import numpy as np
    from reachy_muse.conversation import _CaptureBuffer, VoiceConversation
    from test_conversation import FakeHardware, FakeSession, cancel_task
    from reachy_muse.capabilities import Mode
    from reachy_muse.local_backends import backends_for

    async def scenario():
        hardware = FakeHardware()
        entered, release = threading.Event(), threading.Event()
        reads = []

        def read():
            reads.append(1)
            if len(reads) == 1:
                entered.set()
                release.wait(2)
                return np.full(320, .5, np.float32)
            return None

        hardware.read_audio = read
        conversation = VoiceConversation(FakeSession(), hardware,
                                         backends=backends_for(Mode.MUSE_VOICE, FakeSession()))
        capture = _CaptureBuffer(16000)
        task = asyncio.create_task(conversation._capture_microphone(capture))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            capture.discard_pending()
            release.set()
            while len(reads) < 2:
                await asyncio.sleep(.001)
            assert await capture.get() is None
            assert capture.samples == 0
        finally:
            release.set()
            await cancel_task(task)

    asyncio.run(scenario())


def test_cancelling_during_wave_activation_drains_and_disables_the_camera():
    import threading
    from reachy_muse.conversation import VoiceConversation
    from test_conversation import FakeHardware, FakeSession
    from reachy_muse.capabilities import Mode
    from reachy_muse.local_backends import backends_for

    async def scenario():
        hardware, session = FakeHardware(), FakeSession()
        camera = WaveCamera(hardware)
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        def activate(active):
            if active:
                entered.set()
                assert release.wait(3)
            camera.set_active(active)
            finished.set()

        hardware.set_wave_watching = activate
        session.chat_subscribed.set()
        conversation = VoiceConversation(session, hardware, wave_wake=True,
                                         backends=backends_for(Mode.MUSE_VOICE, session))
        task = asyncio.create_task(conversation._microphone())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await asyncio.sleep(.03)
            cleanup_finished_early = task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert await asyncio.to_thread(finished.wait, 2)
            assert camera.active is False
            assert camera.transitions == [True, False]
            assert cleanup_finished_early is False
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_wave_opens_realtime_after_camera_suspension_and_stays_suspended(monkeypatch):
    from reachy_muse.realtime_call import realtime_supported
    from test_realtime_turns import ScriptedCalls, StreamingTranscripts, rig_for, speak_live, room
    if not realtime_supported():
        pytest.skip("the Muse SDK has no voice call events")
    rig = rig_for(monkeypatch)
    camera = WaveCamera(rig.hardware)
    calls = ScriptedCalls(reply_s=.2)
    opened_while_camera_active = []

    def open_call(*args, **kwargs):
        opened_while_camera_active.append(camera.active)
        return calls(*args, **kwargs)

    async def muse(session, request):
        raise AssertionError("A realtime request must not also go to chat")

    conversation = service(rig, muse, StreamingTranscripts("What is the weather?"),
                           wave_wake=True, wake_timeout_s=30, muse_path="realtime",
                           voice_link=calls.link, voice_call=open_call)

    async def user():
        rig.say()
        await rig.quiet()
        assert calls.frames == []
        camera.pending = True
        await rig.wait_until_or_timeout(lambda: bool(calls.opened), 5)
        await rig.quiet("listening", "idle")
        assert calls.frames == []
        assert camera.active is False
        await speak_live(rig)
        await room(rig, .3)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        assert camera.active is False

    events = run(rig, conversation, user, rest=("idle", "listening"))
    assert opened_while_camera_active == [False]
    assert camera.transitions == [True, False]
    assert sent_to_muse(events) == []
    assert played(events) == ["tts:Yes?", "muse", "muse"]
