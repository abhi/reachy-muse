"""Gaze requests through microphone capture, scripted calls, camera and motor owners."""

import asyncio
import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_muse.hardware import ReachyController
from reachy_muse.local_face_tracking import FaceFollower, _Face
from reachy_muse.motion import MotionStyle
from reachy_muse.person_gaze import gaze_requested
from reachy_muse.speaker_direction import DoAReading
from test_device_command_turns import played, run, sent_to_muse
from test_hardware import FakeRobot, generated, pose_factory
from test_pins import CHUNK, VOICE_SAMPLE, WAKE_SAMPLE
from test_realtime_turns import (RATE, ScriptedCalls, StreamingTranscripts, realtime_service,
                                 reply_events, rig_for, voiced)


@pytest.mark.parametrize('text, expected', [
    ('Say hi to my friend.', True), ('Say hello to someone else.', True),
    ('Greet my guest.', True), ('Please look at her.', True), ('Could you look at them?', True),
    ('Look at me.', False), ('Look over here.', False), ('Watch me.', False),
    ("Don't say hi to him.", False), ('Do not greet her.', False),
    ('Never look at him.', False), ('Say "say hi to him".', False),
    ('"Say hi to him."', False), ("'Say hi to him.'", False),
    ('For example, say hi to him.', False), ('What does say hello to someone mean?', False),
])
def test_small_imperative_grammar(text, expected):
    assert gaze_requested(text) is expected


@pytest.fixture
def people(monkeypatch):
    rig = rig_for(monkeypatch)
    robot = FakeRobot()
    scene = SimpleNamespace(faces=[(30, .35), (-25, .12)], camera=True, busy=True, passes=0, doa=math.pi / 3)
    robot.media.camera = SimpleNamespace(K=np.eye(3), D=np.zeros(5))
    camera_threads = []

    def frame():
        assert controller._camera_lock.locked()
        camera_threads.append(threading.get_ident())
        return robot.pose.copy() if scene.camera else None

    robot.media.get_frame = frame

    class Detect:
        def detect(self, frame):
            scene.passes += 1
            faces = []
            for yaw, width in scene.faces:
                angle = math.radians(yaw)
                x, y, z = frame[:3, :3].T @ [math.cos(angle), math.sin(angle), 0]
                faces.append(_Face((.1 if yaw == 30 else .7, .3, width, .3), math.atan2(y, x),
                                   math.atan2(-z, math.hypot(x, y)), .95))
            return faces

    def factory(*args, **kwargs):
        return FaceFollower(*args, **kwargs, detector_factory=lambda *args: Detect())

    controller = ReachyController(robot_factory=lambda **kwargs: robot, pose_factory=pose_factory,
                                 pose_interpolator=lambda a, b, t: a * (1 - t) + b * t,
                                 face_follow_model='fake', face_follower_factory=factory,
                                 sleep_on_close=False, style=MotionStyle(breathing=False, expression_clips=False))
    controller.vision_budget = SimpleNamespace(delay=lambda: math.inf if scene.busy else 1.0)
    controller.start()
    original_state = rig.hardware.set_state

    def set_state(state, **kwargs):
        original_state(state, **kwargs)
        controller.set_state(state, **kwargs)

    rig.hardware.set_state = set_state
    rig.hardware.motion_enabled = True
    rig.hardware.face_tracking_enabled = True
    rig.hardware.head_yaw = 0.0
    rig.hardware.output_sample_rate = RATE
    rig.hardware.read_doa = lambda: DoAReading(scene.doa, True) if scene.doa is not None else None
    for name in ('face_snapshot', 'refresh_faces', 'look_at_speaker', 'look_at_person', 'release_person_gaze', 'allow_face_tracking'):
        setattr(rig.hardware, name, getattr(controller, name))
    try:
        yield rig, controller, robot, scene
        assert controller._failure is None
        assert len(robot.writers) == 1
        assert threading.get_ident() not in camera_threads
    finally:
        controller.close()


async def say_live(rig, *, wake=False):
    if wake:
        rig.hardware.samples.put(np.full(CHUNK, WAKE_SAMPLE, np.float32))
        await asyncio.sleep(.02)
    for _ in range(35):
        rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
        await asyncio.sleep(.02)
    rig.say(voiced_s=0)


@pytest.mark.parametrize('thinking', [False, True])
def test_other_face_turn_preempts_and_muse_reply_plays_in_full(people, caplog, thinking):
    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    reply = voiced(.6)
    reply_done = asyncio.Event()

    async def answer(sdk, emit, script):
        await rig.wait_until_or_timeout(lambda: controller._person_hold, 2)
        assert controller._person_hold
        scene.busy = False
        reply_events(sdk, emit, 'r1', 'Hello, nice to meet you!', reply)
        reply_done.set()

    calls = ScriptedCalls(answer=answer)
    service = realtime_service(rig, StreamingTranscripts('Say hi to my friend.', 'What time is it?'), calls)

    async def user():
        if thinking:
            controller.set_state('thinking')
            controller.play_motion(generated(8, yaw=10), 25, during={'thinking', 'listening'}, loop=True)
            await rig.wait_until_or_timeout(lambda: controller._generated is not None)
            assert controller._generated is not None
        await say_live(rig, wake=True)
        await asyncio.wait_for(reply_done.wait(), 3)
        await asyncio.sleep(.9)
        assert controller._person_hold
        assert controller._generated is None
        assert math.degrees(controller.head_yaw) == pytest.approx(-25, abs=1)
        assert math.degrees(robot.body_yaw) == pytest.approx(-5, abs=.2)
        assert len(np.concatenate(rig.hardware.played)) == 14400
        np.testing.assert_array_equal(np.concatenate(rig.hardware.played),
                                      np.frombuffer(reply, "<i2").astype(np.float32) / 32768)
        assert calls.barge_ins == 0
        assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
            'Reachy gaze request outcome=turned faces=2']
        rig.hardware.samples.put(np.full(CHUNK * 8, VOICE_SAMPLE, np.float32))
        await rig.wait_until_or_timeout(lambda: not controller._person_hold)
        assert not controller._person_hold
        rig.say(voiced_s=0)

    events = run(rig, service, user)
    assert sent_to_muse(events) == []
    assert set(played(events)) == {'muse'}
    assert 'Say hi' not in caplog.text and 'nice to meet' not in caplog.text


@pytest.mark.parametrize('case, text, expected', [
    ('single', 'Say hi to my friend.', 'no_other faces=1'),
    ('empty', 'Greet my guest.', 'no_faces faces=0'),
    ('stale', 'Look at the other person.', 'stale faces=0'),
    ('unknown', 'Say hello to someone else.', 'no_speaker_bearing faces=0'),
    ('missing', '', 'no_transcript faces=0'),
    ('busy', 'Greet my guest.', 'busy faces=2'),
    ('negative', "Don't say hi to him.", None),
    ('quoted', 'For example, "say hi to him".', None),
])
def test_request_negative_outcomes_keep_the_call_reply(people, caplog, case, text, expected):
    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    if case == 'single':
        scene.faces = [(30, .35)]
    if case == 'empty':
        scene.faces = []
    if case == 'stale':
        assert controller.refresh_faces() is not None
        time.sleep(2.1)
        scene.camera = False
    if case == 'unknown':
        scene.doa = None
    if case == 'busy':
        controller._command_active = True
    calls = ScriptedCalls(reply='Hello!', reply_s=.1)
    service = realtime_service(rig, StreamingTranscripts(text), calls)

    async def user():
        await say_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: 'muse' in played(rig.recording.events), 3)
        await asyncio.sleep(.4)
        assert controller._person_hold is False

    events = run(rig, service, user)
    assert set(played(events)) == {'muse'}
    assert sent_to_muse(events) == []
    assert calls.barge_ins == 0
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == (
        ['Reachy gaze request outcome=' + expected] if expected else [])


def test_look_at_me_remains_the_local_face_tracking_command(people, caplog):
    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    calls = ScriptedCalls(reply=None)
    service = realtime_service(rig, StreamingTranscripts('Look at me.'), calls)

    async def user():
        await say_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: calls.barge_ins == 1)
        assert controller._person_hold is False

    run(rig, service, user)
    assert calls.barge_ins == 1
    assert any('outcome=FACE_TRACKING_ON' in line for line in caplog.messages)
    assert not any(line.startswith('Reachy gaze request') for line in caplog.messages)


def test_cached_faces_choose_smallest_turn_and_release_when_window_closes(people, caplog):
    rig, controller, robot, scene = people
    scene.faces = [(30, .35), (-25, .12), (0, .1)]
    caplog.set_level('INFO')
    calls = ScriptedCalls(reply='Hello!', reply_s=.1)
    service = realtime_service(rig, StreamingTranscripts('Greet my guest.'), calls)

    async def user():
        assert await asyncio.to_thread(controller.refresh_faces) is not None
        await say_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: controller._person_hold)
        assert controller._person_hold
        assert controller._speaker_look.head == 0
        assert scene.passes == 1
        await rig.wait_until_or_timeout(lambda: service._voice[1]._wake_deadline is None, 4)
        await rig.wait_until_or_timeout(lambda: not controller._person_hold)
        assert not controller._person_hold

    run(rig, service, user)
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
        'Reachy gaze request outcome=turned faces=3']


@pytest.mark.parametrize("wake", [False, True])
def test_chat_keeps_endpoint_bearing_after_robot_audio_clears_the_ring(people, caplog, wake):
    from test_device_command_turns import service as chat_service
    from test_pins import Transcripts
    from test_realtime_turns import TimedWake

    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    entered, proceed = asyncio.Event(), asyncio.Event()

    class Transcribe(Transcripts):
        async def transcribe(self, wav):
            entered.set()
            await proceed.wait()
            return await super().transcribe(wav)

    async def muse(session, request):
        await session.answer(request, 'Hello there.')

    service = chat_service(rig, muse, Transcribe('Say hello to my guest.'),
                           wake_detector=TimedWake() if wake else None, wake_timeout_s=5)

    async def user():
        await say_live(rig, wake=wake)
        await asyncio.wait_for(entered.wait(), 2)
        voice = service._voice[1]
        await voice._announce('Just a moment.', started=None, defer_playback=False)
        await asyncio.sleep(.1)
        assert voice._wake_direction.estimate(time.monotonic()) is None
        proceed.set()
        await rig.wait_until_or_timeout(lambda: controller._person_hold)
        assert controller._person_hold
        assert controller._speaker_look.head == pytest.approx(-25)

    events = run(rig, service, user)
    assert sent_to_muse(events) == ['Say hello to my guest.']
    assert played(events) == ['tts:Just a moment.', 'tts:Hello there.']
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
        'Reachy gaze request outcome=turned faces=2']


def test_wake_gaze_is_fallback_only_within_its_window(people, caplog):
    from test_device_command_turns import service as make_service, never_asked
    from test_realtime_turns import TimedWake

    rig, controller, robot, scene = people
    caplog.set_level('INFO')

    class DelayedWake(TimedWake):
        samples = 0

        def feed(self, samples):
            detected = super().feed(samples)
            self.samples += len(samples)
            return detected and self.samples >= 9600

    calls = ScriptedCalls(reply='Hello!', reply_s=.1)
    service = make_service(rig, never_asked, StreamingTranscripts('Say hi to my friend.'),
                           wake_detector=DelayedWake(), wake_timeout_s=5, muse_path='realtime',
                           voice_link=calls.link, voice_call=calls)

    async def user():
        for _ in range(30):
            rig.hardware.samples.put(np.full(CHUNK, WAKE_SAMPLE, np.float32))
            await asyncio.sleep(.02)
        await rig.wait_until_or_timeout(lambda: controller.wake_gaze_bearing is not None)
        assert controller.wake_gaze_bearing == pytest.approx(math.pi / 3)
        rig.hardware.wake_gaze_bearing = controller.wake_gaze_bearing
        scene.doa = None
        scene.camera = True
        await say_live(rig)
        await rig.wait_until_or_timeout(lambda: controller._person_hold)
        assert controller._person_hold
        assert controller._speaker_look.head == pytest.approx(-25)

    run(rig, service, user)
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
        'Reachy gaze request outcome=turned faces=2']
