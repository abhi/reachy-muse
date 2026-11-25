"""Gaze requests through microphone capture, scripted calls, camera and motor owners."""

import asyncio
from dataclasses import replace
import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_muse.device_command import gaze_requested
from reachy_muse.hardware import ReachyController
from reachy_muse.local_face_tracking import FaceFollower, _Face
from reachy_muse.motion import MotionStyle
from reachy_muse.speaker_direction import DoAReading
from test_device_command_turns import played, run, sent_to_muse
from test_hardware import FakeRobot, generated, pose_factory
from test_pins import CHUNK, VOICE_SAMPLE, WAKE_SAMPLE, Rig
from test_realtime_turns import (RATE, CallHardware, ScriptedCalls, StreamingTranscripts, TimedWake,
                                 realtime_service, reply_events, room, voiced)


@pytest.mark.parametrize('text, expected', [
    ('Say hi to my friend.', True), ('Say hello to someone else.', True),
    ('Greet my guest.', True), ('Please look at her.', True), ('Could you look at them?', True),
    ('Okay, say hi to him.', True), ('Reachy, say hi to my friend.', True),
    ('Say hi to my friend, please.', True), ('Can you please say hi to my friend?', True),
    ('Hey Muse, would you greet the other person now?', True), ('Alright, look at her!', True),
    ('Look at me.', False), ('Look over here.', False), ('Watch me.', False),
    ("Don't say hi to him.", False), ('Don’t say hi to him.', False), ('Do not greet her.', False),
    ('Never look at him.', False), ('Please don’t look at them.', False), ('Say hi to him, not her.', False),
    ('Say "say hi to him".', False), ('"Say hi to him."', False), ("'Say hi to him.'", False),
    ('“Say hi to my friend.”', False), ('‘Greet my guest.’', False),
    ('For example, say hi to him.', False), ('What does say hello to someone mean?', False),
    ('Say hi to my friend Sarah.', True), ('Say hello to Sarah.', True), ('Say hello to my mom.', True),
    ('Greet our guests.', True), ('Can you say hi to everyone?', True),
    ("Say hi to my friend's dog.", True), ('Say hi to the kids.', True), ('Say hi to the kids, please.', True),
    ('Say hi to my little friend.', True), ('Say hi to Noah.', True),
    ('Say hi to the camera.', False), ('Say hi to Bob in the email.', False),
    ('Say hi to everyone on the team.', False), ('Say hello to the team in Slack.', False),
    ('Say hi to Mom on the phone.', False), ('Say hello to Alexa.', False), ('Say hi to the robot.', False),
    ('Say hi to the dog.', False), ('Greet the new year.', False),
    ('Say hi to me.', False), ('Say hello to yourself.', False), ('Say hi to Reachy.', False),
    ('Say hello to no one.', False), ('Say hi to my friend Sarah who sits by the window.', False),
    ('Look at this.', False), ('Look at that dog.', False), ('Look at Sarah.', False),
    ('Say hi to her for me.', False), ('Greet them for us, please.', False), ('Say hello to him from me.', False),
    ('Say hi to her when you call her.', False), ('Say hello to them next time.', False),
    ('Say hi to him later.', False), ('Greet my guest tomorrow.', False), ('Say hi to her if you see her.', False),
])
def test_small_imperative_grammar(text, expected):
    assert gaze_requested(text) is expected


class PeopleHardware(CallHardware):
    """The rig's microphone and speaker, with motion, gaze and faces from a real controller."""

    motion_enabled = True
    face_tracking_enabled = True
    output_sample_rate = RATE

    def __init__(self, recording, controller):
        super().__init__(recording)
        self.controller = controller
        for name in ('face_snapshot', 'refresh_faces', 'recent_faces', 'look_at_speaker', 'follow_speaker',
                     'look_at_person', 'face_speaker', 'release_person_gaze', 'allow_face_tracking'):
            setattr(self, name, getattr(controller, name))

    def set_state(self, state, **kwargs):
        super().set_state(state, **kwargs)
        self.controller.set_state(state, **kwargs)

    def gesture(self, name):
        super().gesture(name)
        self.controller.gesture(name)

    def set_call_open(self, call_open):
        self.controller.set_call_open(call_open)

    @property
    def head_yaw(self):
        return self.controller.head_yaw

    @property
    def wake_gaze_bearing(self):
        return self.controller.wake_gaze_bearing

    @property
    def person_gaze_held(self):
        return self.controller.person_gaze_held

    @property
    def person_turning(self):
        return self.controller.person_turning


class DelayedWake(TimedWake):
    """Fires after 0.6 s of wake audio, once DoA has had time to place the speaker."""

    samples = 0

    def feed(self, samples):
        detected = super().feed(samples)
        self.samples += len(samples)
        return detected and self.samples >= 9600


class PlacedWake(TimedWake):
    """Fires on the first wake audio after Reachy's DoA history can place the speaker.

    Under load a capture gap or a late DoA read clears that history, so 0.6 s of wake audio may not be enough.
    Reachy places the speaker as of the moment it captured the wake audio, which can precede the newest sample;
    with three samples the decision holds without it.
    """

    history = None
    heard = False
    examined = 0

    def feed(self, samples):
        phrase_end = time.monotonic()
        decision = self.history.snapshot().decide(phrase_end)
        self.heard = super().feed(samples) and decision.reason == 'ok' and decision.samples >= 3
        self.examined += 1
        return self.heard


@pytest.fixture
def people(monkeypatch):
    rig = Rig(monkeypatch)
    robot = FakeRobot()
    scene = SimpleNamespace(faces=[(30, .35), (-25, .12)], camera=True, busy=True, passes=0, doa=math.pi / 3,
                            age=0.0, before_capture=None, frame_reads=[])
    robot.media.camera = SimpleNamespace(K=np.array([[900., 0, 640], [0, 900, 360], [0, 0, 1]]), D=np.zeros(5))
    camera_threads = []

    def frame():
        assert controller._camera_lock.locked()
        camera_threads.append(threading.get_ident())
        scene.frame_reads.append(time.monotonic())
        return robot.pose.copy() if scene.camera else None

    robot.media.get_frame = frame

    class Detect:
        def detect(self, frame):
            scene.passes += 1
            faces = []
            for yaw, width, *above in scene.faces:
                angle, lift = math.radians(yaw), math.radians(above[0] if above else 0)
                x, y, z = frame[:3, :3].T @ [math.cos(lift) * math.cos(angle), math.cos(lift) * math.sin(angle),
                                             math.sin(lift)]
                faces.append(_Face((.1 if yaw == 30 else .7, .3, width, .3), math.atan2(y, x),
                                   math.atan2(-z, math.hypot(x, y)), .95))
            return faces

    def factory(capture_frame, *args, **kwargs):
        def capture(**options):
            if scene.before_capture is not None:
                scene.before_capture()
            captured = capture_frame(**options)
            return replace(captured, captured_at=captured.captured_at - scene.age) if captured is not None else None
        return FaceFollower(capture, *args, **kwargs, detector_factory=lambda *args: Detect())

    controller = ReachyController(robot_factory=lambda **kwargs: robot, pose_factory=pose_factory,
                                 pose_interpolator=lambda a, b, t: a * (1 - t) + b * t,
                                 face_follow_model='fake', face_follower_factory=factory,
                                 sleep_on_close=False, style=MotionStyle(breathing=False, expression_clips=False))
    controller.vision_budget = SimpleNamespace(delay=lambda: math.inf if scene.busy else 1.0)
    controller.start()
    rig.hardware = PeopleHardware(rig.recording, controller)
    rig.hardware.read_doa = lambda: DoAReading(scene.doa, True) if scene.doa is not None else None
    try:
        yield rig, controller, robot, scene
        assert controller._failure is None
        assert len(robot.writers) == 1
        assert threading.get_ident() not in camera_threads
    finally:
        controller.close()


async def say_live(rig, *, wake=False, voiced_s=.7, paced_silence_s=None):
    if wake:
        rig.hardware.samples.put(np.full(CHUNK, WAKE_SAMPLE, np.float32))
        await asyncio.sleep(.02)
    for _ in range(round(voiced_s * 16000 / CHUNK)):
        rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
        await asyncio.sleep(.02)
    if paced_silence_s is None:
        rig.say(voiced_s=0)
    else:
        await room(rig, paced_silence_s)


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
            'Reachy gaze request outcome=turned source=fresh faces=2']
        rig.hardware.samples.put(np.full(CHUNK * 8, VOICE_SAMPLE, np.float32))
        await rig.wait_until_or_timeout(lambda: not controller._person_hold)
        assert not controller._person_hold
        rig.say(voiced_s=0)

    events = run(rig, service, user)
    assert sent_to_muse(events) == []
    assert set(played(events)) == {'muse'}
    assert 'Say hi' not in caplog.text and 'nice to meet' not in caplog.text


def test_lost_transcript_does_not_carry_the_previous_speaker_into_the_next_request(people, caplog):
    from test_pins import FailingStreamingTranscriber

    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    reply = voiced(.2)

    class FailOnce(StreamingTranscripts):
        failed = False

        def open_turn(self):
            if not self.failed:
                self.failed = True
                return FailingStreamingTranscriber().open_turn()
            return super().open_turn()

    async def answer(sdk, emit, script):
        await rig.wait_until_or_timeout(lambda: controller._person_hold)
        assert controller._person_hold
        reply_events(sdk, emit, 'r1', 'Hello there!', reply)

    calls = ScriptedCalls(answer=answer)
    service = realtime_service(rig, FailOnce('Say hi to my friend.'), calls)

    async def user():
        await say_live(rig, wake=True, voiced_s=1.3)
        await rig.wait_until_or_timeout(lambda: not service._voice[1]._playback.user_speaking)
        assert not service._voice[1]._playback.user_speaking
        scene.doa = math.radians(115)
        await say_live(rig, voiced_s=.5)
        await rig.wait_until_or_timeout(lambda: controller._person_hold)
        assert controller._person_hold
        assert controller._speaker_look.head == pytest.approx(30)
        await rig.wait_until_or_timeout(lambda: abs(math.degrees(controller.head_yaw) - 30) < .2)
        assert math.degrees(controller.head_yaw) == pytest.approx(30, abs=.2)
        assert math.degrees(robot.body_yaw) == pytest.approx(10, abs=.2)
        await rig.wait_until_or_timeout(lambda: 'muse' in played(rig.recording.events))
        await rig.wait_until_or_timeout(lambda: not service._voice[1].call.audible)
        np.testing.assert_array_equal(np.concatenate(rig.hardware.played),
                                      np.frombuffer(reply, '<i2').astype(np.float32) / 32768)

    events = run(rig, service, user)
    assert calls.barge_ins == 0
    assert sent_to_muse(events) == []
    assert played(events) == ['muse']
    assert 'Reachy could not transcribe a turn Muse heard in its call; Muse answers it' in caplog.messages
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
        'Reachy gaze request outcome=turned source=fresh faces=2']


def test_buffered_greeting_excludes_its_speaker_after_wake_verification(people, caplog):
    from reachy_muse.wake_policy import Verification, WakePolicy
    from test_device_command_turns import service as make_service, never_asked

    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    verifying = asyncio.Event()
    reply = voiced(.2)
    scene.speech = True
    rig.hardware.read_doa = lambda: DoAReading(scene.doa, scene.speech)

    class Verify:
        async def verify(self, pcm, phrase):
            verifying.set()
            await asyncio.sleep(.9)
            return Verification('match', 'local')

    async def answer(sdk, emit, script):
        await rig.wait_until_or_timeout(lambda: any(line.startswith('Reachy gaze request') for line in caplog.messages))
        reply_events(sdk, emit, 'r1', 'Hello there!', reply)

    calls = ScriptedCalls(answer=answer)
    service = make_service(rig, never_asked, StreamingTranscripts('Say hi to my friend.'),
                           wake_detector=DelayedWake(), wake_policy=WakePolicy(Verify()),
                           wake_timeout_s=5, muse_path='realtime', voice_link=calls.link, voice_call=calls)
    service.silence_s = .8

    async def user():
        for _ in range(30):
            rig.hardware.samples.put(np.full(CHUNK, WAKE_SAMPLE, np.float32))
            await asyncio.sleep(.02)
        await asyncio.wait_for(verifying.wait(), 2)
        scene.doa = math.radians(115)
        for _ in range(35):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(.02)
        scene.speech = False
        await room(rig, .9)
        await rig.wait_until_or_timeout(lambda: controller._person_hold)
        assert [line for line in caplog.messages if line.startswith('Speaker gaze')][:1] == [
            'Speaker gaze yaw=30.0 body_yaw=10.0 decision=turn']
        assert controller._person_hold
        assert controller._speaker_look.head == pytest.approx(30)
        await rig.wait_until_or_timeout(lambda: abs(math.degrees(controller.head_yaw) - 30) < .2)
        assert math.degrees(controller.head_yaw) == pytest.approx(30, abs=.2)
        assert math.degrees(robot.body_yaw) == pytest.approx(10, abs=.2)
        await rig.wait_until_or_timeout(lambda: 'muse' in played(rig.recording.events))
        await rig.wait_until_or_timeout(lambda: not service._voice[1].call.audible)
        np.testing.assert_array_equal(np.concatenate(rig.hardware.played),
                                      np.frombuffer(reply, '<i2').astype(np.float32) / 32768)

    events = run(rig, service, user)
    assert calls.barge_ins == 0
    assert sent_to_muse(events) == []
    assert played(events) == ['muse']
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
        'Reachy gaze request outcome=turned source=fresh faces=2']
    assert 'Say hi' not in caplog.text and 'Hello there' not in caplog.text


def test_rejected_short_utterances_do_not_supply_the_next_speakers_bearing(people, caplog):
    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    reply = voiced(.2)

    async def answer(sdk, emit, script):
        await rig.wait_until_or_timeout(lambda: any(line.startswith('Reachy gaze request') for line in caplog.messages), 5)
        reply_events(sdk, emit, 'r1', 'Hello there!', reply)

    calls = ScriptedCalls(answer=answer)
    service = realtime_service(rig, StreamingTranscripts('', '', 'Say hi to my friend.'), calls)
    service.silence_s = .8
    service.wake_timeout_s = 5

    async def user():
        await say_live(rig, wake=True, voiced_s=.22, paced_silence_s=.9)
        await say_live(rig, voiced_s=.22, paced_silence_s=.9)
        assert controller.wake_gaze_bearing is None
        assert not service._voice[1]._playback.user_speaking
        assert not any(line.startswith('Reachy gaze request') for line in caplog.messages)
        scene.doa = None
        await say_live(rig, paced_silence_s=.9)
        await rig.wait_until_or_timeout(lambda: any(line.startswith('Reachy gaze request') for line in caplog.messages))
        assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
            'Reachy gaze request outcome=no_speaker_bearing source=none faces=0']
        assert controller._person_hold is None
        await rig.wait_until_or_timeout(lambda: 'muse' in played(rig.recording.events))
        await rig.wait_until_or_timeout(lambda: not service._voice[1].call.audible)
        assert math.degrees(controller.head_yaw) == pytest.approx(0, abs=.2)
        assert math.degrees(robot.body_yaw) == pytest.approx(0, abs=.2)
        np.testing.assert_array_equal(np.concatenate(rig.hardware.played),
                                      np.frombuffer(reply, '<i2').astype(np.float32) / 32768)

    events = run(rig, service, user)
    assert calls.barge_ins == 0
    assert sent_to_muse(events) == []
    assert played(events) == ['muse']
    assert 'Say hi' not in caplog.text and 'Hello there' not in caplog.text


@pytest.mark.parametrize('end', ['rejected_in_the_next_chunk', 'capture_gap'])
def test_an_ended_utterance_never_supplies_the_next_speakers_bearing(people, caplog, end):
    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    reply = voiced(.2)

    async def answer(sdk, emit, script):
        await rig.wait_until_or_timeout(lambda: any(line.startswith('Reachy gaze request') for line in caplog.messages), 5)
        reply_events(sdk, emit, 'r1', 'Hello there!', reply)

    calls = ScriptedCalls(answer=answer)
    service = realtime_service(rig, StreamingTranscripts('', 'Say hi to my friend.'), calls)
    service.silence_s = .8
    service.wake_timeout_s = 5

    async def user():
        if end == 'rejected_in_the_next_chunk':
            await say_live(rig, wake=True, voiced_s=.22, paced_silence_s=.02)
            scene.doa = None
            await room(rig, .58)
            rig.hardware.samples.put(np.concatenate([np.zeros(15 * CHUNK, np.float32),
                                                     np.full(10 * CHUNK, VOICE_SAMPLE, np.float32)]))
            await say_live(rig, voiced_s=.5, paced_silence_s=None)
        else:
            await say_live(rig, wake=True, voiced_s=.3, paced_silence_s=.1)
            await asyncio.sleep(.4)
            scene.doa = None
            await say_live(rig, voiced_s=.7, paced_silence_s=None)
        await room(rig, .9)
        await rig.wait_until_or_timeout(lambda: any(line.startswith('Reachy gaze request') for line in caplog.messages))
        assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
            'Reachy gaze request outcome=no_speaker_bearing source=none faces=0']
        await rig.wait_until_or_timeout(lambda: 'muse' in played(rig.recording.events))
        await rig.wait_until_or_timeout(lambda: not service._voice[1].call.audible)
        assert math.degrees(controller.head_yaw) == pytest.approx(0, abs=.2)
        assert math.degrees(robot.body_yaw) == pytest.approx(0, abs=.2)

    events = run(rig, service, user)
    assert calls.barge_ins == 0
    assert played(events) == ['muse']
    if end == 'capture_gap':
        assert any('capture lost continuity' in line for line in caplog.messages)


@pytest.mark.parametrize('companion_fails', [False, True])
def test_a_companion_failure_in_the_endpoint_silence_keeps_the_speakers_bearing(people, caplog, companion_fails):
    from musegadget.identity import Identity
    from reachy_muse.capabilities import Mode
    from reachy_muse.companion_client import CompanionHearing, CompanionLink, LinkState, _Connection
    from reachy_muse.conversation import ReachyService
    from reachy_muse.failover import CompanionRoute, FailoverHearing
    from reachy_muse.local_backends import backends_for
    from reachy_muse.protocol import Op
    from test_pins import PinSession, Transcripts, Vad

    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    link = CompanionLink('ws://localhost:9', token='test', out_rate=16000, insecure=True)
    link._connection = _Connection(None, (Op.HEAR,))
    link._set(LinkState.UP)

    async def muse(session, request):
        await session.answer(request, 'Hello there.')

    def backends(session):
        bundle = backends_for(Mode.ON_ROBOT, session, speech=rig.voice(), progress_speech=rig.voice('progress_tts'),
                              transcriber=Transcripts('Say hi to my friend.'))
        return replace(bundle, hearing=FailoverHearing(CompanionHearing(link), bundle.hearing,
                                                       CompanionRoute(link).hearing))

    async def prepare_chat(session, vm):
        return 'robot-chat'

    rig.session = PinSession(rig.recording, muse)
    service = ReachyService(identity=Identity('02:00:00:ab:cd:ef'), executor=rig.hardware, silence_s=.2,
                            speech_gate=Vad(), backends=backends, prepare_chat=prepare_chat, owns_chat=True)

    async def user():
        for _ in range(35):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(.02)
        await room(rig, .06)
        if companion_fails:
            link._connection.fail('disconnected')
            link._set(LinkState.DOWN)
        await room(rig, .6)
        await rig.wait_until_or_timeout(lambda: any(line.startswith('Reachy gaze request') for line in caplog.messages), 5)

    events = run(rig, service, user)
    assert sent_to_muse(events) == ['Say hi to my friend.']
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
        'Reachy gaze request outcome=turned source=fresh faces=2']


@pytest.mark.parametrize('case, text, expected', [
    ('single', 'Say hi to my friend.', 'no_other source=fresh faces=1'),
    ('empty', 'Greet my guest.', 'no_faces source=fresh faces=0'),
    ('stale', 'Look at the other person.', 'stale source=none faces=0'),
    ('unknown', 'Say hello to someone else.', 'no_speaker_bearing source=none faces=0'),
    ('missing', '', 'no_transcript source=none faces=0'),
    ('busy', 'Greet my guest.', 'busy source=fresh faces=2'),
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
        scene.faces = [(30, .35)]
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
        assert controller._person_hold is None

    events = run(rig, service, user)
    assert set(played(events)) == {'muse'}
    assert sent_to_muse(events) == []
    assert calls.barge_ins == 0
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == (
        ['Reachy gaze request outcome=' + expected] if expected else [])


def test_a_turn_ended_where_muse_s_reply_starts_keeps_its_speakers_bearing(people, caplog):
    """Reachy's own end of the turn would come 3 s after the user stops, but Muse's reply starts first and ends the
    turn there. The turn keeps the bearing taken while the user spoke, so Reachy turns to the friend at once."""
    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    rig.hardware.echo_cancelled_input = True
    calls = ScriptedCalls(reply='Hello!', reply_s=.1)
    service = realtime_service(rig, StreamingTranscripts('Say hi to my friend.'), calls)
    service.silence_s = 3

    async def user():
        await say_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: 'muse' in played(rig.recording.events), 3)
        answered = time.monotonic()
        await rig.wait_until_or_timeout(lambda: controller._person_hold is not None, 2)
        assert time.monotonic() - answered < 1
        await asyncio.sleep(.9)
        assert math.degrees(controller.head_yaw) == pytest.approx(-25, abs=1)
        assert math.degrees(robot.body_yaw) == pytest.approx(-5, abs=.2)

    events = run(rig, service, user)
    assert set(played(events)) == {'muse'}
    assert calls.barge_ins == 0
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
        'Reachy gaze request outcome=turned source=fresh faces=2']


@pytest.mark.parametrize('seen_s, remembered, current, expected, facing', [
    (10, [(30, .35), (-40, .12)], [(30, .35)], 'turned source=memory faces=1', (-35, -15)),
    (40, [(30, .35), (-40, .12)], [(30, .35)], 'no_other source=fresh faces=1', (30, 10)),
    (10, [(30, .35), (-5, .12)], [(30, .35), (-25, .12)], 'turned source=fresh faces=2', (-25, -5)),
])
def test_a_friend_seen_recently_but_out_of_view_is_remembered(people, caplog, seen_s, remembered, current,
                                                               expected, facing):
    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    scene.faces, scene.age = remembered, seen_s
    assert len(controller.refresh_faces().faces) == 2
    scene.faces, scene.age = current, 0.0
    calls = ScriptedCalls(reply='Hello!', reply_s=.1)
    service = realtime_service(rig, StreamingTranscripts('Say hi to my friend.'), calls)

    def turned():
        return round(math.degrees(controller.head_yaw)), round(math.degrees(robot.body_yaw))

    async def user():
        await say_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: any(line.startswith('Reachy gaze request') for line in caplog.messages))
        await rig.wait_until_or_timeout(lambda: turned() == facing)
        await asyncio.sleep(.3)
        assert turned() == facing

    events = run(rig, service, user)
    assert calls.barge_ins == 0
    assert set(played(events)) == {'muse'}
    assert scene.passes == 2
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
        'Reachy gaze request outcome=' + expected]


def test_memory_does_not_send_reachy_to_where_the_speaker_stood_earlier(people, caplog):
    """Ten seconds ago Reachy saw the user at +5 and the friend at -40. The user now speaks from +30."""
    from test_device_command_turns import service as make_service, never_asked

    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    scene.faces, scene.age = [(5, .35), (-40, .12)], 10
    assert len(controller.refresh_faces().faces) == 2
    scene.faces, scene.age = [(30, .35)], 0.0
    calls = ScriptedCalls(reply='Hello!', reply_s=.1)
    service = make_service(rig, never_asked, StreamingTranscripts('Say hi to my friend.'),
                           wake_detector=DelayedWake(), wake_timeout_s=5, muse_path='realtime',
                           voice_link=calls.link, voice_call=calls)
    seen = {}

    def facing():
        return round(math.degrees(controller.head_yaw)), round(math.degrees(robot.body_yaw))

    async def user():
        for _ in range(30):
            rig.hardware.samples.put(np.full(CHUNK, WAKE_SAMPLE, np.float32))
            await asyncio.sleep(.02)
        await rig.wait_until_or_timeout(lambda: facing() == (30, 10))
        seen['woken'] = facing()
        await say_live(rig)
        await rig.wait_until_or_timeout(lambda: any(line.startswith('Reachy gaze request') for line in caplog.messages))
        await asyncio.sleep(1.2)
        seen['greeting'] = facing()

    events = run(rig, service, user)
    assert set(played(events)) == {'muse'}
    assert seen['woken'] == (30, 10)
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
        'Reachy gaze request outcome=turned source=memory faces=1']
    assert seen['greeting'] == (-35, -15)


def test_no_face_frame_is_read_or_remembered_while_the_head_turns(people):
    from reachy_muse.hardware import SPEAKER_TURN_S

    rig, controller, robot, scene = people
    scene.faces, scene.busy = [], False
    turned = {}

    def turn_before_capturing():
        scene.before_capture = None
        scene.faces = [(-40, .12)]
        assert controller.look_at_speaker(math.pi / 3)
        deadline = time.monotonic() + 1
        while controller._speaker_turn is None and time.monotonic() < deadline:
            time.sleep(.002)
        turned['at'] = time.monotonic()

    controller.set_state('listening')
    deadline = time.monotonic() + 2
    while controller.face_snapshot() is None and time.monotonic() < deadline:
        time.sleep(.01)
    assert controller.face_snapshot() is not None
    scene.before_capture = turn_before_capturing
    time.sleep(SPEAKER_TURN_S + .5)
    assert 'at' in turned
    turning = turned['at'], turned['at'] + SPEAKER_TURN_S - .05
    assert [at for at in scene.frame_reads if turning[0] <= at <= turning[1]] == []
    assert [face.captured_at for face in controller.recent_faces(0) if turning[0] <= face.captured_at <= turning[1]] == []


def test_a_remembered_friend_standing_above_the_view_is_still_found(people, caplog):
    """Ten seconds ago Reachy saw the friend standing 35 degrees above level at -20; the camera's view now covers
    that yaw but not that height."""
    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    scene.faces, scene.age = [(30, .35), (-20, .12, 35)], 10
    assert len(controller.refresh_faces().faces) == 2
    scene.faces, scene.age = [(30, .35)], 0.0
    calls = ScriptedCalls(reply='Hello!', reply_s=.1)
    service = realtime_service(rig, StreamingTranscripts('Say hi to my friend.'), calls)

    def turned():
        return round(math.degrees(controller.head_yaw)), round(math.degrees(robot.body_yaw))

    async def user():
        await say_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: turned() == (-20, 0))
        assert turned() == (-20, 0)

    events = run(rig, service, user)
    assert set(played(events)) == {'muse'}
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
        'Reachy gaze request outcome=turned source=memory faces=1']


@pytest.mark.parametrize('text', ['Say hi to the camera.', 'Say hi to Bob in the email.',
                                  'Say hi to everyone on the team.', 'Say hello to Alexa.'])
def test_greetings_for_no_one_in_the_room_keep_reachy_on_the_speaker(people, caplog, text):
    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    calls = ScriptedCalls(reply='Sure.', reply_s=.1)
    service = realtime_service(rig, StreamingTranscripts(text), calls)
    seen = {}

    async def user():
        await say_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: 'muse' in played(rig.recording.events), 3)
        await asyncio.sleep(1.0)
        seen['facing'] = round(math.degrees(controller.head_yaw))

    events = run(rig, service, user)
    assert calls.barge_ins == 0
    assert set(played(events)) == {'muse'}
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == []
    assert seen['facing'] == 30
    assert controller._person_hold is None


@pytest.mark.parametrize('text', ['Okay, say hi to him.', 'Can you please say hi to my friend?',
                                  'Say hello to Sarah.'])
def test_natural_phrasings_turn_toward_the_friend(people, caplog, text):
    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    calls = ScriptedCalls(reply='Hello!', reply_s=.1)
    service = realtime_service(rig, StreamingTranscripts(text), calls)

    async def user():
        await say_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: round(math.degrees(controller.head_yaw)) == -25)
        assert round(math.degrees(controller.head_yaw)) == -25

    events = run(rig, service, user)
    assert calls.barge_ins == 0
    assert set(played(events)) == {'muse'}
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
        'Reachy gaze request outcome=turned source=fresh faces=2']


@pytest.mark.parametrize('text, gaze, barge_ins, command', [
    ('Look at him.', ['Reachy gaze request outcome=turned source=fresh faces=2'], 0, 'none'),
    ('Look at my friend.', ['Reachy gaze request outcome=turned source=fresh faces=2'], 0, 'none'),
    ('Could you look at the other person?', ['Reachy gaze request outcome=turned source=fresh faces=2'], 0, 'none'),
    ('Look at me.', [], 1, 'FACE_TRACKING_ON'),
])
def test_gaze_phrases_never_reach_the_command_classifier(people, caplog, text, gaze, barge_ins, command):
    from reachy_muse.device_command import DeviceCommand
    from test_device_command_turns import FakeModel, service as make_service, never_asked

    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    model = FakeModel(DeviceCommand.FACE_TRACKING_ON)
    calls = ScriptedCalls(reply='Hello!', reply_s=.1)
    service = make_service(rig, never_asked, StreamingTranscripts(text), model=model,
                           wake_detector=TimedWake(), wake_timeout_s=2, muse_path='realtime',
                           voice_link=calls.link, voice_call=calls)

    async def user():
        await say_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: controller._person_hold or calls.barge_ins or model.asked, 3)
        await asyncio.sleep(.3)

    run(rig, service, user)
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == gaze
    assert (calls.barge_ins, model.asked) == (barge_ins, [])
    assert [line.split(' outcome=')[1].split()[0] for line in caplog.messages
            if line.startswith('Realtime command transcript=arrived')] == [command]


def test_look_at_me_remains_the_local_face_tracking_command(people, caplog):
    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    calls = ScriptedCalls(reply=None)
    service = realtime_service(rig, StreamingTranscripts('Look at me.'), calls)

    async def user():
        await say_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: calls.barge_ins == 1)
        assert controller._person_hold is None

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
        'Reachy gaze request outcome=turned source=snapshot faces=3']


def test_a_turned_head_picks_the_other_face_needing_the_smallest_turn(people, caplog):
    from test_device_command_turns import service as make_service, never_asked

    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    scene.faces = [(30, .35), (-10, .12), (55, .12)]
    calls = ScriptedCalls(reply='Hello!', reply_s=.1)
    service = make_service(rig, never_asked, StreamingTranscripts('Say hi to my friend.'),
                           wake_detector=DelayedWake(), wake_timeout_s=5, muse_path='realtime',
                           voice_link=calls.link, voice_call=calls)

    async def user():
        for _ in range(30):
            rig.hardware.samples.put(np.full(CHUNK, WAKE_SAMPLE, np.float32))
            await asyncio.sleep(.02)
        await rig.wait_until_or_timeout(lambda: abs(math.degrees(controller.head_yaw) - 30) < .2)
        assert math.degrees(controller.head_yaw) == pytest.approx(30, abs=.2)
        await say_live(rig)
        await rig.wait_until_or_timeout(lambda: abs(math.degrees(controller.head_yaw) - 35) < .2)
        assert math.degrees(controller.head_yaw) == pytest.approx(35, abs=.2)
        assert math.degrees(robot.body_yaw) == pytest.approx(15, abs=.2)

    run(rig, service, user)
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
        'Reachy gaze request outcome=turned source=fresh faces=3']


@pytest.mark.parametrize('release', ['next_request', 'window_close'])
def test_releasing_the_greeting_turns_back_to_the_speaker(people, caplog, release):
    from test_device_command_turns import service as make_service, never_asked

    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    reply = voiced(.3)
    asked_again = asyncio.Event()

    async def answer(sdk, emit, script):
        await rig.wait_until_or_timeout(lambda: controller._person_hold, 3)
        reply_events(sdk, emit, 'r1', 'Hello there!', reply)
        await asked_again.wait()
        reply_events(sdk, emit, 'r2', 'It is noon.', reply)

    calls = ScriptedCalls(answer=answer)
    service = make_service(rig, never_asked, StreamingTranscripts('Say hi to my friend.', 'What time is it?'),
                           wake_detector=DelayedWake(), wake_timeout_s=5 if release == 'next_request' else 2,
                           muse_path='realtime', voice_link=calls.link, voice_call=calls)
    service.silence_s = .8

    def facing():
        return round(math.degrees(controller.head_yaw)), round(math.degrees(robot.body_yaw))

    async def user():
        for _ in range(30):
            rig.hardware.samples.put(np.full(CHUNK, WAKE_SAMPLE, np.float32))
            await asyncio.sleep(.02)
        await rig.wait_until_or_timeout(lambda: facing() == (30, 10))
        assert facing() == (30, 10)
        await say_live(rig, paced_silence_s=.9)
        await rig.wait_until_or_timeout(lambda: facing() == (-25, -5))
        assert facing() == (-25, -5)
        await rig.wait_until_or_timeout(lambda: 'muse' in played(rig.recording.events))
        await rig.wait_until_or_timeout(lambda: not service._voice[1].call.audible, 4)
        if release == 'next_request':
            await room(rig, .3)
            await say_live(rig, paced_silence_s=.9)
            asked_again.set()
        else:
            await rig.wait_until_or_timeout(lambda: service._voice[1]._wake_deadline is None, 4)
            assert service._voice[1]._wake_deadline is None
        await rig.wait_until_or_timeout(lambda: not controller._person_hold)
        await asyncio.sleep(1)
        assert facing() == (30, 10)
        assert (service._voice[1]._wake_deadline is not None) is (release == 'next_request')

    run(rig, service, user)
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
        'Reachy gaze request outcome=turned source=fresh faces=2']


def test_waiting_cues_leave_the_head_on_the_friend(people, caplog):
    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    held = []

    def waited():
        return any('cue=still_waiting' in line for line in caplog.messages)

    async def answer(sdk, emit, script):
        await rig.wait_until_or_timeout(waited, 8)
        await asyncio.sleep(1.8)
        reply_events(sdk, emit, 'r1', 'Hello there!', voiced(.2))

    calls = ScriptedCalls(answer=answer)
    service = realtime_service(rig, StreamingTranscripts('Say hi to my friend.'), calls)

    async def user():
        await say_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: round(math.degrees(controller.head_yaw)) == -25)
        await room(rig, .8)
        while 'muse' not in played(rig.recording.events) and len(held) < 500:
            held.append({**controller._pose_parameters(robot.pose), 'antenna': math.degrees(robot.antennas[0])})
            await room(rig, .02)

    events = run(rig, service, user)
    assert 'thinking' in rig.states()
    assert [line for line in caplog.messages if 'cue=still_waiting' in line] == [
        'Realtime turn ordinal=1 cue=still_waiting']
    assert {'robot': 'gesture', 'name': 'nod'} in events
    assert len(held) > 100
    assert {axis: (round(min(pose[axis] for pose in held), 1), round(max(pose[axis] for pose in held), 1))
            for axis in ('yaw', 'roll', 'z')} == {'yaw': (-25.0, -25.0), 'roll': (0.0, 0.0), 'z': (3.0, 3.0)}
    assert round(max(pose['pitch'] for pose in held) - min(pose['pitch'] for pose in held), 1) == 0.0
    assert max(pose['antenna'] for pose in held) - min(pose['antenna'] for pose in held) > 20


def test_muse_motion_commands_run_once_reachy_faces_the_friend(people, caplog):
    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    rig.hardware.run_command = controller.run_command
    reply = voiced(.3)

    async def answer(sdk, emit, script):
        await rig.wait_until_or_timeout(lambda: controller._person_hold, 3)
        reply_events(sdk, emit, 'r1', 'Hello, nice to meet you!', reply)

    calls = ScriptedCalls(answer=answer)
    service = realtime_service(rig, StreamingTranscripts('Say hi to my friend.'), calls)
    service.wake_timeout_s = 8
    results = {}

    async def user():
        await say_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: controller._person_hold)
        await rig.wait_until_or_timeout(lambda: 'muse' in played(rig.recording.events))
        await rig.wait_until_or_timeout(lambda: not service._voice[1].call.audible, 4)
        await room(rig, 1.5)
        results['held'] = round(math.degrees(controller.head_yaw)), controller._person_hold is not None
        results['expression'] = await asyncio.to_thread(
            service.run_command, 'reachy.expression', {'name': 'happy', 'duration': .5}, 5000)
        results['move'] = await asyncio.to_thread(
            service.run_command, 'reachy.move', {'pitch': -5, 'duration': .3, 'hold': 0}, 5000)
        results['released'] = controller._person_hold is None

    run(rig, service, user)
    assert results['held'] == (-25, True)
    assert (results['expression']['ok'], results['move']['ok']) == (True, True), results
    assert results['released'] is True


def test_a_reply_after_a_muse_command_ended_the_gaze_gets_its_closing_clip(people, caplog):
    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    commands = []

    def run_command(name, params, timeout_ms):
        commands.append(params.get('name', name))
        return controller.run_command(name, params, timeout_ms)

    rig.hardware.run_command = run_command
    moved = asyncio.Event()

    async def answer(sdk, emit, script):
        await rig.wait_until_or_timeout(lambda: controller._person_hold, 3)
        reply_events(sdk, emit, 'r1', 'Hello, nice to meet you!', voiced(.3))
        await moved.wait()
        reply_events(sdk, emit, 'r2', 'Hello, nice to meet you!', voiced(.3))

    calls = ScriptedCalls(answer=answer)
    service = realtime_service(rig, StreamingTranscripts('Say hi to my friend.'), calls)
    service.wake_timeout_s = 8

    async def user():
        await say_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: 'muse' in played(rig.recording.events))
        await rig.wait_until_or_timeout(lambda: not service._voice[1].call.audible, 4)
        await room(rig, 1)
        assert commands == []
        moving = await asyncio.to_thread(service.run_command, 'reachy.move', {'pitch': -5, 'duration': .3, 'hold': 0}, 5000)
        assert moving['ok'] is True
        assert controller._person_hold is None
        moved.set()
        await rig.wait_until_or_timeout(lambda: len(commands) > 1, 4)

    run(rig, service, user)
    assert commands[:2] == ['reachy.move', 'success1']


def test_a_lost_line_turns_back_from_the_friend_to_the_speaker(people, caplog, monkeypatch):
    from test_device_command_turns import service as make_service, never_asked

    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    monkeypatch.setattr('reachy_muse.conversation.CALL_PENDING_EXTENSION_S', 1)
    calls = ScriptedCalls(reply=None)
    service = make_service(rig, never_asked, StreamingTranscripts('Say hi to my friend.'),
                           wake_detector=DelayedWake(), wake_timeout_s=2, muse_path='realtime',
                           voice_link=calls.link, voice_call=calls)
    lost = 'tts:I lost my line to Muse. Please say that again.'
    seen = {}

    def facing():
        return round(math.degrees(controller.head_yaw)), round(math.degrees(robot.body_yaw))

    async def user():
        for _ in range(30):
            rig.hardware.samples.put(np.full(CHUNK, WAKE_SAMPLE, np.float32))
            await asyncio.sleep(.02)
        await rig.wait_until_or_timeout(lambda: facing() == (30, 10))
        await say_live(rig, paced_silence_s=.6)
        await rig.wait_until_or_timeout(lambda: facing() == (-25, -5))
        seen['greeting'] = facing()
        for _ in range(400):
            if lost in played(rig.recording.events):
                break
            await room(rig, .02)
        await room(rig, 1)
        seen['after_loss'] = facing()

    events = run(rig, service, user, rest=('listening', 'idle'))
    assert [line for line in caplog.messages if line.startswith('Realtime recovery ')] == [
        'Realtime recovery attempt=1 reason=window_expired']
    assert played(events) == [lost]
    assert len(calls.opened) == 2
    assert seen == {'greeting': (-25, -5), 'after_loss': (30, 10)}
    assert controller._person_hold is None


@pytest.mark.parametrize("wake", [False, True])
@pytest.mark.parametrize('silence_s', [.8, 1.2])
def test_chat_keeps_endpoint_bearing_after_robot_audio_clears_the_ring(people, caplog, wake, silence_s):
    from test_device_command_turns import service as chat_service
    from test_pins import Transcripts

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
    service.silence_s = silence_s

    async def user():
        await say_live(rig, wake=wake, paced_silence_s=silence_s + .1)
        await asyncio.wait_for(entered.wait(), 2)
        voice = service._voice[1]
        await voice._announce('Just a moment.', started=None, defer_playback=False)
        await asyncio.sleep(.1)
        assert voice._wake_direction.snapshot().samples == ()
        proceed.set()
        await rig.wait_until_or_timeout(lambda: controller._person_hold)
        assert controller._person_hold
        assert controller._speaker_look.head == pytest.approx(-25)

    events = run(rig, service, user)
    assert sent_to_muse(events) == ['Say hello to my guest.']
    assert played(events) == ['tts:Just a moment.', 'tts:Hello there.']
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
        'Reachy gaze request outcome=turned source=fresh faces=2']


def test_rejected_echo_keeps_person_gaze_through_the_reply(people, caplog):
    rig, controller, robot, scene = people
    rig.hardware.echo_cancelled_input = True
    caplog.set_level('INFO')
    reply = voiced(3)

    async def answer(sdk, emit, script):
        await rig.wait_until_or_timeout(lambda: controller._person_hold)
        assert controller._person_hold
        reply_events(sdk, emit, 'r1', 'Hello, nice to meet you!', reply)

    calls = ScriptedCalls(answer=answer)
    service = realtime_service(rig, StreamingTranscripts('Say hi to my friend.', 'Hello, nice to meet you!'), calls)

    async def user():
        await say_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: 'muse' in played(rig.recording.events))
        await say_live(rig)
        assert service._voice[1].call.audible
        assert controller._person_hold
        await rig.wait_until_or_timeout(lambda: any('drop=barge_in' in line for line in caplog.messages))
        assert controller._person_hold
        assert calls.barge_ins == 0
        assert calls.loud_after_reply == 0
        await rig.wait_until_or_timeout(lambda: not service._voice[1].call.audible, 5)
        assert controller._person_hold
        assert math.degrees(controller.head_yaw) == pytest.approx(-25, abs=.2)
        np.testing.assert_array_equal(np.concatenate(rig.hardware.played),
                                      np.frombuffer(reply, '<i2').astype(np.float32) / 32768)

    run(rig, service, user)
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
        'Reachy gaze request outcome=turned source=fresh faces=2']


@pytest.mark.parametrize('text, logged', [
    ('Say hello to my guest.', ['Reachy gaze request outcome=busy source=none faces=0']),
    ('Find me a recipe.', []),
])
@pytest.mark.parametrize('path', ['chat', 'realtime'])
def test_delayed_transcript_cannot_reacquire_gaze_during_the_next_utterance(people, caplog, monkeypatch, path,
                                                                            text, logged):
    from test_device_command_turns import service as chat_service
    from test_pins import Transcripts

    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    entered, proceed = asyncio.Event(), asyncio.Event()

    class Transcribe(Transcripts):
        async def transcribe(self, wav):
            entered.set()
            await proceed.wait()
            return await super().transcribe(wav)

    class Streaming(StreamingTranscripts):
        def open_turn(self):
            turn = super().open_turn()
            finish = turn.finish

            async def delayed():
                entered.set()
                await proceed.wait()
                return await finish()

            turn.finish = lambda: asyncio.create_task(delayed())
            return turn

    async def muse(session, request):
        await session.answer(request, 'Hello there.')

    finished = asyncio.Event()

    async def answer(sdk, emit, script):
        await finished.wait()
        reply_events(sdk, emit, 'r1', 'Here you go.', voiced(.2))

    service = chat_service(rig, muse, Transcribe(text, 'What time is it?'),
                           wake_detector=TimedWake(), wake_timeout_s=5)
    if path == 'realtime':
        monkeypatch.setattr('reachy_muse.conversation.CALL_COMMAND_DEADLINE_S', 2)
        service = realtime_service(rig, Streaming(text, 'What time is it?'), ScriptedCalls(answer=answer))

    async def user():
        await say_live(rig, wake=True)
        await asyncio.wait_for(entered.wait(), 2)
        for _ in range(15):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(.02)
        assert service._voice[1]._playback.user_speaking
        proceed.set()
        for _ in range(25):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(.02)
        assert service._voice[1]._playback.user_speaking
        assert controller._person_hold is None
        assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == logged
        rig.say(voiced_s=0)
        finished.set()

    run(rig, service, user)


@pytest.mark.parametrize('speaker_yaw', [30, 60])
def test_wake_gaze_is_fallback_only_within_its_window(people, caplog, speaker_yaw):
    from test_device_command_turns import service as make_service, never_asked

    rig, controller, robot, scene = people
    caplog.set_level('INFO')
    scene.doa = math.radians(90 - speaker_yaw)
    if speaker_yaw == 60:
        scene.faces = [(60, .35)]
    calls = ScriptedCalls(reply='Hello!', reply_s=.1)
    wake = PlacedWake()
    service = make_service(rig, never_asked, StreamingTranscripts('Say hi to my friend.'),
                           wake_detector=wake, wake_timeout_s=5, muse_path='realtime',
                           voice_link=calls.link, voice_call=calls)

    async def user():
        wake.history = service._voice[1]._wake_direction
        while not wake.heard:
            examined = wake.examined
            rig.hardware.samples.put(np.full(CHUNK, WAKE_SAMPLE, np.float32))
            await rig.wait_until_or_timeout(lambda: wake.examined > examined)
            await asyncio.sleep(.02)
        await rig.wait_until_or_timeout(lambda: controller.wake_gaze_bearing is not None)
        assert controller.wake_gaze_bearing == pytest.approx(math.radians(90 - speaker_yaw))
        scene.doa = None
        scene.camera = True
        # Queued at once: paced from the event loop, a stall of 0.27 s would look like a capture gap and split
        # the request in two.
        rig.say(voiced_s=.7)
        await rig.wait_until_or_timeout(lambda: any(line.startswith('Reachy gaze request') for line in caplog.messages))
        assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
            'Reachy gaze request outcome=' + ('turned source=fresh faces=2' if speaker_yaw == 30 else 'no_other source=fresh faces=1')]
        if speaker_yaw == 30:
            await rig.wait_until_or_timeout(lambda: controller._person_hold)
        assert (controller._person_hold is not None) is (speaker_yaw == 30)
        assert controller._speaker_look.head == pytest.approx(-25 if speaker_yaw == 30 else 35)
        assert controller.wake_gaze_bearing == pytest.approx(math.radians(90 - speaker_yaw))

    run(rig, service, user)
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
        'Reachy gaze request outcome=' + ('turned source=fresh faces=2' if speaker_yaw == 30 else 'no_other source=fresh faces=1')]


def test_limited_person_turn_does_not_reseed_on_the_speaker(people, caplog):
    rig, controller, robot, scene = people
    scene.faces = [(20, .35), (60, .12)]
    scene.doa = math.radians(70)
    caplog.set_level('INFO')
    calls = ScriptedCalls(reply='Hello!', reply_s=.1)
    service = realtime_service(rig, StreamingTranscripts('Say hi to my friend.'), calls)

    async def user():
        await say_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: controller._person_hold)
        assert controller._person_hold
        scene.busy = False
        await asyncio.sleep(1.5)
        assert controller._person_hold
        assert math.degrees(controller.head_yaw) == pytest.approx(35, abs=.2)
        assert math.degrees(robot.body_yaw) == pytest.approx(15, abs=.2)

    events = run(rig, service, user)
    assert set(played(events)) == {'muse'}
    assert calls.barge_ins == 0
    assert [line for line in caplog.messages if line.startswith('Reachy gaze request')] == [
        'Reachy gaze request outcome=turned source=fresh faces=2']
