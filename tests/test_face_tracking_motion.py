# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Exercise gaze at the motor output, including the clips that used to erase it."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_muse.arrival import WatchPose
from reachy_muse.contract.expressions import Expression
from reachy_muse.hardware import ReachyController, _Frames
from reachy_muse.local_face_tracking import FaceTarget
from reachy_muse.motion import MotionStyle
from reachy_muse.motion.layers import MotionLayers, Ramp
from reachy_muse.motion.sentence import SentenceMotion
from test_hardware import FakeRobot, FakeWatcher, generated, pose_factory


class Follower:
    failure = None
    active = False
    visible = True
    yaw = 8
    pitch = -4

    def __init__(self):
        self.activity = []
        self.rates = []

    def set_active(self, active, *, idle=False):
        self.active = active
        self.activity.append(active)
        self.rates.append(idle)

    def target(self):
        return (FaceTarget(math.radians(self.yaw), math.radians(self.pitch), 0)
                if self.active and self.visible else None)


def drive(monkeypatch, action, *, ticks=140, **options):
    robot = FakeRobot()
    controller = ReachyController(pose_factory=pose_factory,
                                 pose_interpolator=lambda a, b, t: a * (1 - t) + b * t,
                                 sleep_on_close=False, **options)
    controller._np, controller._robot = np, robot
    controller._output_sample_rate = 16000
    controller._start_music()
    controller._layers = MotionLayers(controller._style)
    controller._generated_gain = Ramp(1)
    controller._face_follower = follower = Follower()
    controller._require_running = lambda: None
    samples = {}

    class Clock:
        now, tick = 1000.0, 0

        def is_set(self):
            return self.tick >= ticks

        def wait(self, duration):
            self.now += duration
            if controller._started.is_set():
                self.tick += 1
                samples[self.tick] = controller._pose_parameters(robot.pose)
                action(controller, follower, self.tick)
            return self.is_set()

    clock = controller._stop = Clock()
    monkeypatch.setattr('reachy_muse.hardware.time', SimpleNamespace(monotonic=lambda: clock.now))
    controller._motion_loop()
    assert controller._failure is None
    return controller, follower, samples


@pytest.mark.parametrize('state', ['listening', 'thinking', 'speaking'])
def test_generated_gestures_follow_a_moving_face_throughout_conversation(monkeypatch, state):
    def action(controller, follower, tick):
        if tick == 1:
            controller.set_state(state)
            controller.play_motion(generated(8, yaw=3, pitch=2, roll=5, z=4), 25, during={state})
        if tick == 70:
            follower.yaw, follower.pitch = -8, 4

    _, follower, samples = drive(monkeypatch, action, style=MotionStyle(breathing=False))
    assert all(follower.activity[1:])
    assert samples[65] == pytest.approx(dict(x=0, y=0, z=4, roll=5, yaw=11, pitch=-2), abs=.1)
    assert samples[135] == pytest.approx(dict(x=0, y=0, z=4, roll=5, yaw=-5, pitch=6), abs=.1)


@pytest.mark.parametrize('source', ['library', 'sentence'])
def test_speech_clips_add_to_gaze_and_sentence_replacement_does_not_duplicate_it(monkeypatch, source):
    frames = generated(8, yaw=3, pitch=2, roll=5, z=4)
    move = _Frames(np.asarray(frames), 25, False, pose_factory)

    def action(controller, follower, tick):
        if tick == 1:
            controller._moves = SimpleNamespace(list_moves=lambda: ['success1'], get=lambda name: move)
        if tick in (1, 70):
            if source == 'library':
                controller.set_state('speaking', expression='happy')
            else:
                motion = SimpleNamespace(frames=frames, fps=25)
                controller.play_audio(np.ones(1600), sentence=SentenceMotion(Expression.HAPPY, motion=motion))
        if tick == 70:
            follower.yaw, follower.pitch = -8, 4

    _, follower, samples = drive(monkeypatch, action, style=MotionStyle(sway='sdk', breathing=False))
    assert all(follower.activity[1:])
    # Spoken clips use 0.6 intensity; gaze must retain its full strength.
    assert samples[65]['yaw'] == pytest.approx(9.8, abs=.1)
    assert samples[135]['yaw'] == pytest.approx(-6.2, abs=.1)
    assert samples[135]['pitch'] == pytest.approx(5.2, abs=.1)
    assert samples[135]['roll'] == pytest.approx(3.0, abs=.1)


def test_idle_follow_includes_still_mode_and_respects_sleep_and_user_switch():
    controller = ReachyController(face_follow_idle=True)
    controller._face_follower = follower = Follower()
    for state in ('idle', 'listening', 'thinking', 'speaking', 'sleeping', 'error'):
        controller._update_face_tracking(state)
    assert follower.activity == [True, True, True, True, False, False]
    assert follower.rates == [True, False, False, False, False, False]
    controller._style = MotionStyle(breathing=False)
    controller._update_face_tracking('idle')
    controller._update_face_tracking('listening')
    controller.allow_face_tracking(False)
    controller._update_face_tracking('speaking')
    assert follower.activity[-3:] == [True, True, False]
    assert follower.rates[-3:] == [True, False, False]


@pytest.mark.parametrize('idle, breathing, idle_message', [
    (False, True, 'Local face tracking off: idle'),
    (False, False, 'Local face tracking off: idle'),
    (True, True, 'Local face tracking on: idle follow'),
    (True, False, 'Local face tracking on: idle follow'),
])
def test_tracking_logs_only_transitions_with_content_free_reasons(caplog, idle, breathing, idle_message):
    caplog.set_level('INFO', logger='reachy_muse.hardware')
    controller = ReachyController(face_follow_idle=idle, style=MotionStyle(breathing=breathing))
    controller._face_follower = Follower()
    for state, commanded in [('idle', False), ('listening', False), ('thinking', False),
                             ('speaking', False), ('speaking', True), ('speaking', True),
                             ('speaking', False), ('sleeping', False)]:
        controller._update_face_tracking(state, commanded=commanded)
    assert [r.message for r in caplog.records] == [
        idle_message,
        'Local face tracking on: conversation',
        'Local face tracking off: command',
        'Local face tracking on: conversation',
        'Local face tracking off: sleeping',
    ]


def test_wave_watching_keeps_idle_face_following_active():
    controller = ReachyController(face_follow_idle=True, wave_wake=True)
    controller._face_follower = follower = Follower()
    controller._wave_watching = True
    controller._update_face_tracking('idle')
    assert follower.activity == [True]
    assert follower.rates == [True]


def test_full_strength_speech_clip_keeps_the_nod_on_top_of_gaze(monkeypatch):
    frames = generated(8, roll=5)

    def action(controller, follower, tick):
        if tick == 1:
            motion = SimpleNamespace(frames=frames, fps=25)
            controller.play_audio(np.ones(1600), sentence=SentenceMotion(Expression.HAPPY, motion=motion))
        if tick == 70:
            controller.gesture('nod')

    _, _, samples = drive(monkeypatch, action, style=MotionStyle(sway='sdk', breathing=False))
    pitches = [pose['pitch'] for tick, pose in samples.items() if 75 <= tick <= 130]
    assert max(pitches) > 1
    assert min(pitches) < -9
    assert samples[85]['yaw'] == pytest.approx(8, abs=.1)


@pytest.mark.parametrize('command', ['reachy.move', 'reachy.expression'])
def test_muse_commands_own_absolute_pose_then_gaze_returns(monkeypatch, command):
    from reachy_muse.hardware import _Command

    move = _Frames(np.asarray(generated(.8, yaw=30, pitch=18)), 25, False, pose_factory)
    params = {'yaw': 30, 'pitch': 18, 'duration': .2, 'hold': .2} if command == 'reachy.move' else {
        'name': 'test_move', 'intensity': 1,
    }
    request = _Command(command, params, 1020)
    commanded_poses = []

    def action(controller, follower, tick):
        if tick == 1:
            controller.set_state('listening')
            controller.play_motion(generated(8, roll=4), 25, during={'listening'})
            controller._moves = SimpleNamespace(list_moves=lambda: ['test_move'], get=lambda name: move)
        if tick == 70:
            controller._commands.put_nowait(request)
        if tick > 70 and not request.done.is_set() and not follower.active:
            commanded_poses.append(controller._pose_parameters(controller._robot.pose))

    _, follower, samples = drive(monkeypatch, action, ticks=240, style=MotionStyle(breathing=False))
    assert request.result['ok']
    assert any(abs(pose['yaw'] - 30) < .01 and abs(pose['pitch'] - 18) < .01 for pose in commanded_poses)
    assert follower.active
    assert samples[235]['yaw'] == pytest.approx(8, abs=.1)
    assert samples[235]['pitch'] == pytest.approx(-4, abs=.1)


@pytest.mark.parametrize('gesture', ['nod', 'shake', 'tilt'])
def test_recorded_command_keeps_its_pose_when_a_listening_gesture_arrives(monkeypatch, gesture):
    from reachy_muse.hardware import _Command

    move = _Frames(np.asarray(generated(5, pitch=10)), 25, False, pose_factory)
    request = _Command('reachy.expression', {'name': 'test_move', 'intensity': 1}, 1020)

    def action(controller, follower, tick):
        if tick == 1:
            controller.set_state('listening')
            controller._moves = SimpleNamespace(list_moves=lambda: ['test_move'], get=lambda name: move)
            controller._commands.put_nowait(request)
        if tick == 70:
            controller.gesture(gesture)

    _, _, samples = drive(monkeypatch, action, ticks=240, style=MotionStyle(breathing=False))
    assert request.result['ok']
    for tick in range(65, 140):
        assert samples[tick] == pytest.approx(dict(x=0, y=0, z=0, roll=0, pitch=10, yaw=0))


def test_gaze_plus_clip_is_bounded_and_lost_face_returns_to_normal_motion(monkeypatch):
    def action(controller, follower, tick):
        if tick == 1:
            follower.yaw, follower.pitch = 20, 12
            controller.set_state('listening')
            controller.play_motion(generated(8, yaw=15, pitch=10), 25, during={'listening'})
        if tick == 70:
            follower.target = lambda: None

    _, _, samples = drive(monkeypatch, action, style=MotionStyle(breathing=False))
    assert samples[65]['yaw'] == pytest.approx(35, abs=.1)
    assert samples[65]['pitch'] == pytest.approx(20)
    assert samples[135]['yaw'] == pytest.approx(15, abs=.1)
    assert samples[135]['pitch'] == pytest.approx(10, abs=.1)
    assert all(-35 <= pose['yaw'] <= 35 and -20 <= pose['pitch'] <= 20 for pose in samples.values())
    assert max(abs(samples[t]['yaw'] - samples[t - 1]['yaw']) for t in range(2, 140)) < 3


@pytest.mark.parametrize('idle, breathing, state, active, yaw', [
    (False, True, 'idle', False, 0),
    (False, False, 'idle', False, 0),
    (True, True, 'idle', True, 8),
    (True, False, 'idle', True, 8),
    (True, True, 'sleeping', False, 0),
    (True, False, 'sleeping', False, 0),
])
def test_idle_follow_reaches_motor_output_only_when_allowed(monkeypatch, idle, breathing, state,
                                                          active, yaw):
    def action(controller, follower, tick):
        if tick == 1:
            controller.set_state(state)

    _, follower, samples = drive(monkeypatch, action, face_follow_idle=idle,
                                 style=MotionStyle(breathing=breathing))
    assert follower.active is active
    assert samples[135]['yaw'] == pytest.approx(yaw, abs=.1)


def test_still_idle_follow_smoothly_tracks_a_moving_face_then_returns_to_rest(monkeypatch):
    def action(controller, follower, tick):
        if tick == 70:
            follower.yaw, follower.pitch = -8, 4
        if tick == 140:
            follower.visible = False

    controller, follower, samples = drive(monkeypatch, action, ticks=280, face_follow_idle=True,
                                          style=MotionStyle(breathing=False))
    assert follower.active is True
    assert samples[65] == pytest.approx(dict(x=0, y=0, z=0, roll=0, yaw=8, pitch=-4), abs=.1)
    assert samples[135] == pytest.approx(dict(x=0, y=0, z=0, roll=0, yaw=-8, pitch=4), abs=.1)
    for tick in range(230, 281):
        assert samples[tick] == pytest.approx(dict(x=0, y=0, z=0, roll=0, yaw=0, pitch=0), abs=.01)
    for axis in ('yaw', 'pitch'):
        assert max(abs(samples[tick][axis] - samples[tick - 1][axis])
                   for tick in range(2, 281)) < 1.5
    for _, _, antennas, body_yaw in controller._robot.frames:
        assert np.degrees(antennas) == pytest.approx([-10, 10], abs=.01)
        assert body_yaw == 0


@pytest.mark.parametrize('breathing', [True, False])
@pytest.mark.parametrize('resume', ['idle', 'listening', 'thinking', 'speaking'])
def test_arrival_watching_observes_faces_at_the_watch_pose_then_resumes_gaze(monkeypatch, resume, breathing):
    stillness, tracking = [], []

    def action(controller, follower, tick):
        if tick == 70:
            controller._arrival_watcher = FakeWatcher(None, controller._camera_still, None)
            controller.start_watching(WatchPose(-20, 5))
        if 300 <= tick < 350:
            follower.yaw = 8 if tick % 20 < 10 else -8
        if 330 <= tick < 350:
            stillness.append(controller._arrival_watcher.is_still())
            tracking.append(follower.active)
        if tick == 350:
            follower.yaw = 8
            if resume == 'idle':
                controller.stop_watching()
            else:
                controller.set_state(resume)

    _, follower, samples = drive(monkeypatch, action, ticks=520, face_follow_idle=True,
                                 style=MotionStyle(breathing=breathing))
    assert samples[65]['yaw'] == pytest.approx(8, abs=.1)
    assert samples[340]['yaw'] == pytest.approx(-20, abs=.1)
    assert samples[340]['pitch'] == pytest.approx(5, abs=.1)
    assert stillness == [True] * 20
    assert tracking == [True] * 20
    assert follower.active
    assert samples[515]['yaw'] == pytest.approx(8, abs=.1)
