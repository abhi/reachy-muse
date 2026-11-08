"""A visual invocation opens the existing conversation, without spoken keywords."""

import asyncio

import pytest

from test_device_command_turns import played, rig_with, run, sent_to_muse, service
from test_pins import Transcripts, Wake


class WaveCamera:
    def __init__(self, hardware):
        self.active = False
        self.pending = False
        self.claimable = False
        self.transitions = []
        hardware.set_wave_watching = self.set_active
        hardware.take_wave = self.take
        hardware.claim_wave = self.claim

    def set_active(self, active):
        if active != self.active:
            self.transitions.append(active)
            self.pending = False
        self.active = active
        if not active:
            self.pending = False

    def take(self):
        result, self.pending = self.active and self.pending, False
        self.claimable = result
        return result

    def claim(self):
        result, self.claimable = self.claimable, False
        return result


def continuous_microphone(hardware):
    import queue
    import time
    from types import SimpleNamespace
    import numpy as np
    from reachy_muse.hardware import ReachyController

    class AppSink:
        def try_pull_sample(self, timeout):
            try:
                return hardware.samples.get_nowait()
            except queue.Empty:
                if timeout:
                    time.sleep(timeout / 1_000_000_000)
                    return np.zeros(320, dtype=np.float32)
                return None

    sink = AppSink()
    media = SimpleNamespace(audio=SimpleNamespace(_appsink_audio=sink),
                            get_audio_sample=lambda: sink.try_pull_sample(20_000_000))
    controller = ReachyController()
    controller._np, controller._robot = np, SimpleNamespace(media=media)
    hardware.read_audio = controller.read_audio
    hardware.discard_microphone = lambda: controller.discard_microphone()


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("acoustic", [False, True])
def test_wave_greets_and_only_fresh_speech_reaches_muse(monkeypatch, acoustic, continuous):
    rig = rig_with(monkeypatch)
    if continuous:
        continuous_microphone(rig.hardware)
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
        assert camera.active is True
        assert rig.hardware.gestures == ["wave"]
        assert sent_to_muse(rig.recording.events) == []
        rig.say()

    events = run(rig, conversation, user, rest=("idle", "listening"))
    assert sent_to_muse(events) == ["What is the weather tomorrow?"]
    assert played(events) == ["tts:Yes?", "tts:Sunny tomorrow."]
    assert camera.transitions == [True, False, True, False, True, False]
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
        await rig.wait_until_or_timeout(lambda: "tts:I'm awake." in played(rig.recording.events), 5)

    events = run(rig, conversation, user, rest=("listening", "idle"))
    assert played(events) == ["tts:Good night.", "tts:I'm awake."]
    assert sent_to_muse(events) == []
    assert rig.hardware.gestures == ["wave"]
    assert camera.transitions == [True, False, True, False, True, False]


def test_settled_pixel_wave_wakes_sleep_without_a_speech_request(monkeypatch):
    import time
    from bench.wave import clip
    from reachy_muse.wave_wake import WaveWatcher
    from test_face_tracking_motion import Follower
    from test_hardware import FakeRobot, start_controller

    rig = rig_with(monkeypatch)
    robot = FakeRobot()
    still = next(clip("still", 1001))
    gesture = list(clip("wave", 1001, seconds=3))
    started, seen = [], []

    def read():
        frame = still
        if started and time.monotonic() - started[0] >= 1.5:
            index = int((time.monotonic() - started[0] - 1.5) * 8)
            frame = gesture[min(index, len(gesture) - 1)]
        seen[:] = frame[2]
        return frame[1]

    class People:
        def detect(self, frame):
            return list(seen)

    robot.media.get_frame = read
    controller = start_controller(robot, wave_wake=True, face_follow_idle=True, sleep_on_close=False,
                                  wave_watcher_factory=lambda *args, **kwargs:
                                  WaveWatcher(*args, **kwargs, detector_factory=lambda _: People()))
    controller._face_follower = follower = Follower()
    follower.close = lambda: None
    original_state = rig.hardware.set_state

    def set_state(state, **kwargs):
        original_state(state, **kwargs)
        controller.set_state(state, **kwargs)

    rig.hardware.set_state = set_state
    rig.hardware.set_wave_watching = controller.set_wave_watching
    rig.hardware.take_wave = controller.take_wave
    rig.hardware.claim_wave = controller.claim_wave

    async def muse(session, request):
        raise AssertionError("A wave is not a request to Muse")

    conversation = service(rig, muse, Transcripts("Hey Muse, go to sleep."),
                           wave_wake=True, wake_detector=Wake())

    async def user():
        rig.say(wake=True)
        await rig.quiet("sleeping")
        await rig.wait_until_or_timeout(controller._wave_still, 6)
        assert follower.active is False
        started.append(time.monotonic())
        await rig.wait_until_or_timeout(lambda: "tts:I'm awake." in played(rig.recording.events), 10)

    try:
        events = run(rig, conversation, user, rest=("listening", "idle"))
        assert played(events) == ["tts:Good night.", "tts:I'm awake."]
        assert sent_to_muse(events) == []
        assert controller._failure is None
    finally:
        controller.close()


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
            discarding = asyncio.create_task(capture.discard_pending())
            await asyncio.sleep(0)
            release.set()
            await discarding
            while len(reads) < 2:
                await asyncio.sleep(.001)
            assert await capture.get() is None
            assert capture.samples == 0
        finally:
            release.set()
            await cancel_task(task)

    asyncio.run(scenario())


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("post_wave_speech", [False, True])
def test_wave_discards_sdk_backlog_before_recognition(monkeypatch, post_wave_speech, continuous):
    import io
    import threading
    import wave
    import numpy as np

    rig = rig_with(monkeypatch)
    if continuous:
        continuous_microphone(rig.hardware)
    camera = WaveCamera(rig.hardware)
    entered, release, drained = threading.Event(), threading.Event(), threading.Event()
    original_read = rig.hardware.read_audio
    original_discard = rig.hardware.discard_microphone
    heard = []

    def read():
        sample = original_read()
        if sample is not None and np.any(sample) and not entered.is_set():
            entered.set()
            assert release.wait(5)
        if sample is None and release.is_set():
            drained.set()
        return sample

    def discard():
        original_discard()
        drained.set()

    class Recognizer:
        async def transcribe(self, wav):
            with wave.open(io.BytesIO(wav), "rb") as audio:
                pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
            values = np.unique(pcm[pcm != 0]).tolist()
            heard.append(values)
            return "Post-wave speech." if values == [16384] else "Pre-wave speech."

    async def muse(session, request):
        await session.answer(request, "Received.")

    rig.hardware.read_audio = read
    rig.hardware.discard_microphone = discard
    conversation = service(rig, muse, Recognizer(), wave_wake=True, wake_timeout_s=30)

    async def user():
        rig.say()
        assert await asyncio.to_thread(entered.wait, 2)
        camera.pending = True
        await rig.wait_until_or_timeout(lambda: not camera.active)
        release.set()
        await rig.wait_until_or_timeout(lambda: "tts:Yes?" in played(rig.recording.events))
        await rig.quiet("listening", "idle")
        assert await asyncio.to_thread(drained.wait, 2)
        await rig.quiet("listening", "idle")
        if post_wave_speech:
            for value in [.5] * 25 + [0] * 25:
                rig.hardware.samples.put(np.full(320, value, dtype=np.float32))

    try:
        events = run(rig, conversation, user, rest=("listening", "idle"))
    finally:
        release.set()
    assert heard == ([[16384]] if post_wave_speech else [])
    assert sent_to_muse(events) == (["Post-wave speech."] if post_wave_speech else [])


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


@pytest.mark.parametrize("blocked", ["camera", "model"])
def test_service_cancellation_and_hardware_close_are_bounded_during_wave_work(monkeypatch, blocked):
    import threading
    import numpy as np
    from reachy_muse.hardware import ReachyController
    from reachy_muse.wave_wake import WaveWatcher
    from test_hardware import FakeRobot

    rig = rig_with(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def block():
        entered.set()
        assert release.wait(10)

    def read():
        if blocked == "camera":
            block()
        return np.zeros((120, 160, 3), np.uint8)

    def load(_):
        if blocked == "model":
            block()
        return object()

    controller = ReachyController(wave_wake=True, wave_watcher_factory=lambda *args, **kwargs:
                                 WaveWatcher(*args, **kwargs, detector_factory=load))
    controller._robot = FakeRobot()
    controller._robot.media.get_frame = read
    controller._wave_still = lambda: True
    controller._read_wave_gaze = lambda: None
    rig.hardware.set_wave_watching = controller.set_wave_watching
    rig.hardware.take_wave = controller.take_wave
    rig.hardware.claim_wave = controller.claim_wave

    async def muse(session, request):
        raise AssertionError("No speech was submitted")

    conversation = service(rig, muse, Transcripts(), wave_wake=True)

    async def scenario():
        task = asyncio.create_task(conversation.serve(rig.session, {"vm_id": "vm-1"}, asyncio.Event()))
        closing = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await rig.wait_until_or_timeout(lambda: not controller._wave_watcher._active)
            closing = asyncio.create_task(asyncio.to_thread(controller.close))
            await asyncio.wait({task, closing}, timeout=3)
            assert [task.done(), closing.done()] == [True, True]
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(RuntimeError, match="Wave watcher did not stop"):
                await closing
            assert not any(t.get_name() == "reachy-microphone-capture" for t in asyncio.all_tasks())
            assert controller.take_wave() is False
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, *([closing] if closing is not None else []), return_exceptions=True)

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        controller.close()


@pytest.mark.parametrize("muse_path", ["chat", "realtime"])
def test_arrival_watching_invalidates_a_wave_during_camera_drain(monkeypatch, muse_path):
    import threading
    from reachy_muse.arrival import WatchPose
    from reachy_muse.hardware import ReachyController
    from reachy_muse.realtime_call import realtime_supported
    from test_hardware import FakeRobot, FakeWatcher
    from test_realtime_turns import ScriptedCalls

    if muse_path == "realtime" and not realtime_supported():
        pytest.skip("the Muse SDK has no voice call events")
    rig = rig_with(monkeypatch)
    camera = WaveCamera(rig.hardware)
    consumed, draining, release = threading.Event(), threading.Event(), threading.Event()
    controller = ReachyController(wave_wake=True)
    controller._robot = FakeRobot()
    controller._arrival_watcher = arrival = FakeWatcher(None, None, None)

    class Watcher:
        def set_active(self, active):
            camera.set_active(active)

        def drain(self):
            if consumed.is_set() and not draining.is_set():
                draining.set()
                assert release.wait(5)

        def take_wave(self):
            wave = camera.take()
            if wave:
                consumed.set()
            return wave

        def close(self):
            pass

    controller._wave_watcher = Watcher()
    rig.hardware.set_wave_watching = controller.set_wave_watching
    rig.hardware.take_wave = controller.take_wave
    rig.hardware.claim_wave = lambda: controller.claim_wave()
    calls = ScriptedCalls()
    transcripts = Transcripts("Uninvited speech.")

    async def muse(session, request):
        await session.answer(request, "Received.")

    options = {"voice_link": calls.link, "voice_call": calls} if muse_path == "realtime" else {}
    conversation = service(rig, muse, transcripts, wave_wake=True, wake_timeout_s=30,
                           muse_path=muse_path, **options)

    async def user():
        camera.pending = True
        assert await asyncio.to_thread(draining.wait, 2)
        watching = asyncio.create_task(asyncio.to_thread(controller.start_watching, WatchPose()))
        try:
            await rig.wait_until_or_timeout(lambda: controller._watching)
            assert controller._watching is True
        finally:
            release.set()
            await watching
        await rig.quiet("idle", "listening")
        rig.say()

    try:
        events = run(rig, conversation, user, rest=("idle", "listening"))
        assert arrival.watching == [True]
    finally:
        release.set()
        controller.close()
    assert played(events) == []
    assert sent_to_muse(events) == []
    assert calls.opened == []
    assert transcripts.texts == ["Uninvited speech."]


@pytest.mark.parametrize("continuous", [False, True])
def test_wave_opens_realtime_after_suspension_and_watches_again_while_idle(monkeypatch, continuous):
    from reachy_muse.realtime_call import realtime_supported
    from test_realtime_turns import ScriptedCalls, StreamingTranscripts, rig_for, speak_live, room
    if not realtime_supported():
        pytest.skip("the Muse SDK has no voice call events")
    rig = rig_for(monkeypatch)
    if continuous:
        continuous_microphone(rig.hardware)
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
        await rig.wait_until_or_timeout(lambda: "tts:Yes?" in played(rig.recording.events), 5)
        await rig.quiet("listening", "idle")
        assert not any(any(frame) for frame in calls.frames)
        assert camera.active is True
        assert rig.hardware.gestures == ["wave"]
        await speak_live(rig)
        await room(rig, .3)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        assert camera.active is False

    events = run(rig, conversation, user, rest=("idle", "listening"))
    assert opened_while_camera_active == [False]
    assert camera.transitions[:4] == [True, False, True, False]
    assert camera.transitions[-2:] == [True, False]
    assert sent_to_muse(events) == []
    assert played(events) == ["tts:Yes?", "muse", "muse"]


def test_speech_during_wave_reaches_realtime_from_its_first_sample(monkeypatch):
    import threading
    import numpy as np
    from test_realtime_turns import ScriptedCalls, StreamingTranscripts, rig_for
    from reachy_muse.realtime_call import realtime_supported

    if not realtime_supported():
        pytest.skip("the Muse SDK has no voice call events")
    rig = rig_for(monkeypatch)
    rig.hardware.sample_rate = 24000
    camera = WaveCamera(rig.hardware)
    calls = ScriptedCalls(reply=None)
    flushing, release = threading.Event(), threading.Event()
    discard = rig.hardware.discard_microphone
    gesture = rig.hardware.gesture

    def discard_microphone():
        flushing.set()
        assert release.wait(3)
        discard()

    def speak_at_gesture(name):
        rig.hardware.samples.put(np.full(480, .75, np.float32))
        gesture(name)

    rig.hardware.discard_microphone = discard_microphone
    rig.hardware.gesture = speak_at_gesture

    async def muse(session, request):
        raise AssertionError("A realtime request must not also go to chat")

    conversation = service(rig, muse, StreamingTranscripts("What is the weather tomorrow?"),
                           wave_wake=True, wake_timeout_s=30, muse_path="realtime",
                           voice_link=calls.link, voice_call=calls)

    async def user():
        camera.pending = True
        assert await asyncio.to_thread(flushing.wait, 2)
        await asyncio.sleep(.1)
        release.set()
        await rig.wait_until_or_timeout(lambda: bool(rig.hardware.gestures))
        for value in [.25] * 50 + [.5] * 50 + [0] * 25:
            rig.hardware.samples.put(np.full(480, value, np.float32))
            await asyncio.sleep(.02)

    try:
        events = run(rig, conversation, user, rest=("idle", "listening"))
    finally:
        release.set()
    pcm = np.frombuffer(b"".join(calls.frames), "<i2")
    assert [int(np.count_nonzero(pcm == value)) for value in (24575, 8192, 16384)] == [480, 24000, 24000]
    np.testing.assert_array_equal(pcm[pcm != 0], [24575] * 480 + [8192] * 24000 + [16384] * 24000)
    assert sent_to_muse(events) == []
    assert played(events) == []


@pytest.mark.parametrize("muse_path, expected_requests, expected_playback", [
    ("chat", ["What is the weather tomorrow?"], ["tts:Sunny tomorrow."]),
    ("realtime", [], ["muse"]),
])
def test_short_request_during_wave_omits_the_delayed_greeting(
        monkeypatch, muse_path, expected_requests, expected_playback):
    import numpy as np
    from reachy_muse.realtime_call import realtime_supported
    from test_realtime_turns import ScriptedCalls, StreamingTranscripts, reply_events, rig_for, room, voiced

    if muse_path == "realtime" and not realtime_supported():
        pytest.skip("the Muse SDK has no voice call events")
    rig = rig_for(monkeypatch)
    camera = WaveCamera(rig.hardware)
    reply = asyncio.Event()

    async def answer(sdk, emit, script):
        emit(sdk.Working(True))
        await reply.wait()
        reply_events(sdk, emit, "r1", "Sunny tomorrow.", voiced(.2))
        emit(sdk.Working(False))

    calls = ScriptedCalls(answer=answer)

    async def muse(session, request):
        await session.answer(request, "Sunny tomorrow.")

    conversation = service(rig, muse, StreamingTranscripts("What is the weather tomorrow?"),
                           wave_wake=True, wake_timeout_s=30, muse_path=muse_path,
                           voice_link=calls.link, voice_call=calls)
    conversation.silence_s = .8

    async def user():
        camera.pending = True
        await rig.wait_until_or_timeout(lambda: bool(rig.hardware.gestures))
        for value in [.3] * 33 + [0] * 45:
            rig.hardware.samples.put(np.full(320, value, np.float32))
        try:
            await room(rig, 2)
        finally:
            reply.set()
        await rig.wait_until_or_timeout(lambda: expected_playback[-1] in played(rig.recording.events), 5)

    events = run(rig, conversation, user, rest=("idle", "listening"))
    assert sent_to_muse(events) == expected_requests
    assert played(events) == expected_playback


def test_another_wave_while_awake_gets_a_gesture_without_reopening_or_speaking(monkeypatch):
    rig = rig_with(monkeypatch)
    camera = WaveCamera(rig.hardware)

    async def muse(session, request):
        raise AssertionError("A wave is not a request to Muse")

    conversation = service(rig, muse, Transcripts(), wave_wake=True, wake_timeout_s=30)

    async def user():
        camera.pending = True
        await rig.wait_until_or_timeout(lambda: "tts:Yes?" in played(rig.recording.events), 5)
        await rig.quiet("listening", "idle")
        assert camera.active is True
        assert rig.hardware.gestures == ["wave"]
        camera.pending = True
        await rig.wait_until_or_timeout(lambda: len(rig.hardware.gestures) == 2)
        assert rig.hardware.gestures == ["wave", "wave"]

    events = run(rig, conversation, user, rest=("idle", "listening"))
    assert played(events) == ["tts:Yes?"]
    assert sent_to_muse(events) == []


def test_wave_during_reply_is_not_saved_for_after_speech(monkeypatch):
    from test_device_command_turns import LongVoice

    rig = rig_with(monkeypatch, chunk_s=.05)
    camera = WaveCamera(rig.hardware)

    async def muse(session, request):
        await session.answer(request, "Hello there.")

    conversation = service(rig, muse, Transcripts("Hey Muse, say hello."), wave_wake=True,
                           wake_detector=Wake(), wake_timeout_s=30, voice=LongVoice(rig.recording, 20))

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "tts:Hello there." in played(rig.recording.events), 5)
        assert camera.active is False
        camera.pending = True
        await rig.quiet("listening", "idle")
        assert camera.active is True
        assert rig.hardware.gestures == []

    events = run(rig, conversation, user, rest=("idle", "listening"))
    assert played(events) == ["tts:Hello there."]
    assert rig.hardware.gestures == []
