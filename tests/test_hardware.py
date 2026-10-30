# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

import reachy_muse.hardware
from reachy_muse.arrival import ArrivalStats, WatchPose
from reachy_muse.hardware import EMOTIONS_DATASET, ReachyController, ReachyHardwareError
from reachy_muse.motion import MotionStyle

SDK_SWAY = MotionStyle(sway="sdk")


class OfflineLibrary:
    def __init__(self, dataset):
        raise OSError("huggingface.co is unreachable")


@pytest.fixture(autouse=True)
def offline_emotion_library(monkeypatch):
    """Every controller here starts on a robot whose emotion library cannot load."""
    monkeypatch.setitem(sys.modules, "reachy_mini.motion.recorded_move",
                        SimpleNamespace(RecordedMoves=OfflineLibrary))


def pose_factory(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, degrees=True, mm=True):
    roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    pose = np.eye(4)
    pose[:3, :3] = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    pose[:3, 3] = np.array([x, y, z]) / 1000
    return pose


class FakeMedia:
    def __init__(self):
        self.audio = self
        self.input = None
        self.output = []
        self.clear_count = 0
        self.recording = False
        self.playing = False

    def get_input_audio_samplerate(self):
        return 16000

    def get_output_audio_samplerate(self):
        return 16000

    def start_recording(self):
        self.recording = True

    def start_playing(self):
        self.playing = True

    def get_audio_sample(self):
        audio, self.input = self.input, None
        return audio

    def push_audio_sample(self, samples):
        self.output.append(samples.copy())

    def clear_player(self):
        self.clear_count += 1
        self.output.clear()


class FakeRobot:
    def __init__(self):
        self.media = FakeMedia()
        self.pose = np.eye(4)
        self.antennas = np.deg2rad([-10, 10])
        self.body_yaw = 0
        self.writers = set()
        self.frames = []
        self.closed = False
        self.wobbling = False
        self.wobble_events = []
        self.events = []
        self.fail_writes = False

    def _record_writer(self):
        self.writers.add(threading.get_ident())

    def enable_motors(self):
        self._record_writer()
        self.events.append("enable")

    def get_current_head_pose(self):
        return self.pose

    def get_current_joint_positions(self):
        return [self.body_yaw, 0, 0, 0, 0, 0, 0], self.antennas

    def enable_wobbling(self):
        self._record_writer()
        self.wobbling = True
        self.wobble_events.append(True)

    def disable_wobbling(self):
        self._record_writer()
        self.wobbling = False
        self.wobble_events.append(False)

    def set_target(self, head, antennas, body_yaw):
        self._record_writer()
        if self.fail_writes:
            raise RuntimeError("daemon disconnected")
        self.pose = head.copy()
        self.antennas = antennas.copy()
        self.body_yaw = body_yaw
        self.frames.append((time.monotonic(), head.copy(), antennas.copy(), body_yaw))

    def goto_sleep(self):
        self._record_writer()
        self.events.append("sleep")

    def goto_target(self, head, antennas, body_yaw, duration):
        self.events.append("park")
        self.set_target(head=head, antennas=antennas, body_yaw=body_yaw)

    def disable_motors(self):
        self._record_writer()
        self.events.append("disable")

    def __exit__(self, *args):
        self.media.playing = self.media.recording = False
        self.closed = True


def start_controller(robot, **options):
    controller = ReachyController(
        robot_factory=lambda **kwargs: robot,
        pose_factory=pose_factory,
        pose_interpolator=lambda first, last, t: first * (1 - t) + last * t,
        **options,
    )
    controller.start()
    return controller


@pytest.fixture
def hardware():
    robot = FakeRobot()
    controller = start_controller(robot)
    yield controller, robot
    controller.close()


@pytest.fixture
def sdk_sway_hardware():
    robot = FakeRobot()
    controller = start_controller(robot, style=SDK_SWAY)
    yield controller, robot
    controller.close()


def test_controller_installs_the_speaker_eq_profile_it_was_given(monkeypatch):
    import reachy_muse.speaker_eq
    installed = []
    monkeypatch.setattr(reachy_muse.speaker_eq, "install_speaker_eq",
                        lambda audio, *, gains: installed.append(gains))
    robot = FakeRobot()
    controller = ReachyController(robot_factory=lambda **kwargs: robot, pose_factory=pose_factory,
                                  pose_interpolator=lambda first, last, t: first,
                                  enable_motion=False, speaker_eq_gains=(1.0,) * 10)
    controller.start()
    controller.close()
    assert installed == [(1.0,) * 10]


def test_mic_becomes_mono_and_playback_flush_keeps_recording(hardware):
    controller, robot = hardware
    robot.media.input = np.array([[0.1, 0.3], [-0.1, -0.3]], dtype=np.float32)
    np.testing.assert_allclose(controller.read_audio(), [0.2, -0.2])
    assert controller.read_audio() is None
    assert controller.sample_rate == controller.output_sample_rate == 16000
    controller.play_audio(np.array([0.25, -0.5, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(robot.media.output[-1], [0.25, -0.5, 1.0])
    controller.clear_audio()
    assert robot.media.clear_count == 1
    assert robot.media.recording and robot.media.playing


def test_thinking_has_lively_ears_but_a_still_head_for_the_first_nine_seconds():
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    first, first_antennas = controller._state_pose("thinking", 0)
    later, later_antennas = controller._state_pose("thinking", 1.2)
    np.testing.assert_allclose(first, later, atol=1e-12)
    np.testing.assert_allclose(np.rad2deg(first_antennas), [-10, 10])
    assert not np.allclose(first_antennas, later_antennas)
    assert first[2, 3] == pytest.approx(0.004)
    times = np.arange(0, 9, 0.025)
    poses = [controller._state_pose("thinking", t) for t in times]
    antennas = np.array([np.rad2deg(ant) for _, ant in poses])
    assert np.ptp(antennas[:, 0]) > 50
    assert (np.abs(antennas) <= 40).all()
    for pose, _ in poses:
        np.testing.assert_allclose(pose, pose_factory(pitch=-3, z=4), atol=1e-12)
    assert np.count_nonzero(np.diff(np.sign(antennas[:, 0] + 10))) >= 12


def test_thinking_ear_bursts_have_a_quiet_pause_and_resume_without_a_jump():
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    for pause in (2.25, 2.7, 3.2, 3.59):
        _, antennas = controller._state_pose("thinking", pause)
        np.testing.assert_allclose(np.rad2deg(antennas), [-10, 10], atol=1e-12)
    poses = [controller._state_pose("thinking", t) for t in np.arange(0, 7.2, .025)]
    antennas = np.rad2deg([antenna for _, antenna in poses])
    assert max(abs(np.diff(antennas[:, 1]) / .025)) < 350
    resumed = antennas[(np.arange(0, 7.2, .025) >= 4.05) & (np.arange(0, 7.2, .025) < 5.25), 1]
    assert np.ptp(resumed) > 50


def test_thinking_head_glances_are_brief_alternate_and_return_to_center():
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    times = np.arange(0, 120, .025)
    coordinates = [controller._pose_parameters(controller._state_pose("thinking", t)[0]) for t in times]
    yaw = np.array([pose["yaw"] for pose in coordinates])
    moving = np.abs(yaw) > .01
    assert np.mean(moving) < .07
    assert np.count_nonzero(moving[1:] & ~moving[:-1]) == 4
    for midpoint, expected in ((9.9, 6), (39.9, -6), (69.9, 6), (99.9, -6)):
        pose = controller._pose_parameters(controller._state_pose("thinking", midpoint)[0])
        assert pose["yaw"] == pytest.approx(expected)
        assert pose["roll"] == pytest.approx(expected)
        assert pose["z"] == pytest.approx(6)
    for settled in (0, 8.9, 10.8, 20, 38.9, 40.8, 70.8, 100.8):
        pose, _ = controller._state_pose("thinking", settled)
        np.testing.assert_allclose(pose, pose_factory(pitch=-3, z=4), atol=1e-12)
    assert np.max(np.abs(yaw)) <= 6 + 1e-9
    assert max(abs(np.diff(yaw) / .025)) < 11
    for pose in coordinates:
        assert pose["x"] == 0 and pose["y"] == 0
        assert pose["pitch"] == pytest.approx(-3)


def test_progress_ear_accent_enters_and_leaves_the_thinking_beat_smoothly():
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    for age in (0, .001, 1.399, 1.4, 2):
        _, baseline = controller._state_pose("thinking", 2.6 + age)
        _, accented = controller._state_pose("thinking", 2.6 + age, progress_age=age)
        assert np.max(np.abs(np.rad2deg(accented - baseline))) < .001
    _, baseline = controller._state_pose("thinking", 3.3)
    _, accented = controller._state_pose("thinking", 3.3, progress_age=.7)
    assert np.max(np.abs(np.rad2deg(accented - baseline))) > 35


@pytest.mark.parametrize("thinking_age", [.7, 2.6, 9.2, 39.2])
def test_progress_ear_accent_is_pronounced_bounded_and_preserves_the_head_pose(thinking_age):
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    antennas = []
    for cue_age in np.arange(0, 1.4, .025):
        baseline, _ = controller._state_pose("thinking", thinking_age + cue_age)
        accented, ears = controller._state_pose("thinking", thinking_age + cue_age, progress_age=cue_age)
        np.testing.assert_allclose(accented, baseline, atol=1e-12)
        antennas.append(np.rad2deg(ears))
    antennas = np.array(antennas)
    assert np.ptp(antennas[:, 1]) > 70
    assert np.max(np.abs(antennas)) <= 60


def test_progress_cue_keeps_the_head_glance_clock_and_clears_when_thinking_ends(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(reachy_muse.hardware, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    robot = FakeRobot()
    controller._robot = robot
    controller.set_state("thinking")
    clock[0] = 109.9
    controller.cue_progress()
    elapsed = clock[0] - controller._state_started
    pose, _ = controller._state_pose("thinking", elapsed,
                                    progress_age=clock[0] - controller._progress_started)
    coordinates = controller._pose_parameters(pose)
    assert coordinates["yaw"] == pytest.approx(6)
    assert coordinates["roll"] == pytest.approx(6)
    assert robot.frames == [] and robot.writers == set()

    controller.set_state("speaking", expression="happy")
    controller.cue_progress()
    assert controller._progress_started is None
    clock[0] = 110.0
    controller.set_state("thinking")
    _, ears = controller._state_pose("thinking", 2.7, progress_age=controller._progress_started)
    np.testing.assert_allclose(np.rad2deg(ears), [-10, 10], atol=1e-12)


def test_reply_pose_matches_the_emotion():
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    happy, happy_antennas = controller._state_pose("speaking", .5, "happy")
    sad, sad_antennas = controller._state_pose("speaking", .5, "sad")
    assert happy[2, 3] == pytest.approx(0.007)
    assert sad[2, 3] == pytest.approx(-0.005)
    np.testing.assert_allclose(np.rad2deg(happy_antennas), [-35, 35])
    np.testing.assert_allclose(np.rad2deg(sad_antennas), [-55, 55])
    thoughtful, _ = controller._state_pose("speaking", .5, "thinking")
    assert controller._pose_parameters(thoughtful)["roll"] == pytest.approx(12)


@pytest.mark.parametrize("state, gesture, age, axis, expected", [
    ("listening", "nod", .45, "pitch", 5),
    ("thinking", "nod", .45, "pitch", 5),
    ("listening", "tilt", .8, "roll", 10),
])
def test_gestures_layer_on_the_resting_pose(state, gesture, age, axis, expected):
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    resting, _ = controller._state_pose(state, age)
    pose = controller._gesture_pose(resting, gesture, age)
    assert controller._pose_parameters(pose)[axis] == pytest.approx(expected)
    np.testing.assert_allclose(controller._gesture_pose(resting, gesture, 2), resting)


def test_gesture_survives_speech_onset_and_preserves_the_expression(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(reachy_muse.hardware, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    controller.set_state("thinking", expression="curious")
    controller.gesture("nod")
    assert (controller._state, controller._reply_expression) == ("thinking", "curious")
    clock[0] = 10.1
    controller.set_state("speaking", expression="happy")
    clock[0] = 11.05
    resting, _ = controller._state_pose("speaking", clock[0] - controller._state_started, "happy")
    pose = controller._gesture_pose(resting, controller._gesture_name, clock[0] - controller._gesture_started)
    coordinates = controller._pose_parameters(pose)
    assert coordinates["pitch"] == pytest.approx(-8)
    assert coordinates["roll"] < -5
    assert coordinates["z"] > 4
    assert controller._gesture_started == 10
    controller.gesture("nod")
    assert controller._gesture_started == 11.05


def test_thinking_variations_keep_the_glance_clock(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(reachy_muse.hardware, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    controller = ReachyController(pose_factory=pose_factory)
    controller.set_state("thinking")
    clock[0] = 15
    controller.set_state("thinking", expression="curious")
    assert controller._state_started == 10


def test_unmarked_speech_stays_neutral_instead_of_taking_the_thinking_pose():
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    for age in (0, .4, 1.2, 5):
        pose, antennas = controller._state_pose("speaking", age)
        np.testing.assert_allclose(pose, pose_factory(), atol=1e-12)
        np.testing.assert_allclose(np.rad2deg(antennas), [-10, 10])


def test_happy_speech_rises_holds_then_settles_while_face_gaze_remains_absolute():
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    controller._face_follower = SimpleNamespace(target=lambda: SimpleNamespace(
        yaw=math.radians(6), pitch=math.radians(-3)))
    poses, ears = [], []
    for age in (0, .4, .7, 1.8):
        pose, antennas = controller._state_pose("speaking", age, "happy")
        poses.append(controller._pose_parameters(controller._face_pose(pose)))
        ears.append(np.rad2deg(antennas))
    assert poses[0]["z"] == pytest.approx(0)
    assert poses[1]["z"] == pytest.approx(7)
    assert poses[2]["z"] == pytest.approx(7)
    assert 3 < poses[3]["z"] < 6
    assert poses[1]["roll"] == pytest.approx(-8)
    assert -8 < poses[3]["roll"] < -3
    assert all(pose["yaw"] == pytest.approx(6) and pose["pitch"] == pytest.approx(-3)
               for pose in poses)
    np.testing.assert_allclose(ears[0], [-10, 10])
    np.testing.assert_allclose(ears[1], [-35, 35])
    assert not np.allclose(ears[1], ears[3])


def test_thinking_antennas_have_distinct_timing_without_more_head_motion():
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    frames = [controller._state_pose("thinking", age) for age in np.arange(.45, 1.65, .025)]
    ears = np.rad2deg([antennas for _, antennas in frames])
    assert np.max(np.abs(ears.sum(axis=1))) > 10
    assert np.ptp(ears[:, 0]) > 50 and np.ptp(ears[:, 1]) > 50
    for pose, _ in frames:
        np.testing.assert_allclose(pose, pose_factory(pitch=-3, z=4), atol=1e-12)


def test_rare_thinking_glance_is_visible_without_changing_face_gaze():
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    controller._face_follower = SimpleNamespace(target=lambda: SimpleNamespace(yaw=0., pitch=0.))
    before, _ = controller._state_pose("thinking", 8.9)
    glance, _ = controller._state_pose("thinking", 9.9)
    after, _ = controller._state_pose("thinking", 10.8)
    coordinates = controller._pose_parameters(controller._face_pose(glance))
    assert coordinates == pytest.approx(dict(x=0, y=0, z=6, roll=6, yaw=0, pitch=0))
    np.testing.assert_allclose(controller._face_pose(before), controller._face_pose(after), atol=1e-12)


def test_consecutive_happy_sentences_each_start_a_motif_without_resetting_thinking(monkeypatch):
    clock = [100.]
    monkeypatch.setattr(reachy_muse.hardware, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    controller.set_state("speaking", expression="happy")
    clock[0] = 102.
    settled, _ = controller._state_pose("speaking", clock[0] - controller._state_started, "happy")
    controller.set_state("speaking", expression="happy")
    clock[0] += .4
    peak, _ = controller._state_pose("speaking", clock[0] - controller._state_started, "happy")
    assert controller._pose_parameters(peak)["z"] == pytest.approx(7)
    assert controller._pose_parameters(settled)["z"] < 6
    controller.set_state("thinking")
    thinking_started = controller._state_started
    clock[0] += 2
    controller.set_state("thinking")
    assert controller._state_started == thinking_started


def test_pose_smoothing_keeps_the_same_response_when_target_write_frequency_changes():
    from reachy_muse.hardware import _blend_factor
    fractions = []
    for interval in (.01, .025, .05):
        fraction = 0.
        for _ in range(round(.5 / interval)):
            fraction += (1 - fraction) * _blend_factor(.12, interval)
        fractions.append(fraction)
    assert fractions == pytest.approx([.922437, .922437, .922437], abs=1e-6)


@pytest.mark.parametrize("expression, axis, minimum_range", [("nod", "pitch", 4), ("shake", "yaw", 5)])
def test_gesture_keeps_face_tracking_audio_and_a_single_writer(
        expression, axis, minimum_range):
    robot = FakeRobot()
    robot.media.get_frame = lambda: None
    robot.media.camera = SimpleNamespace(K=np.eye(3), D=np.zeros(5))

    class Follower:
        active = False
        failure = None

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def set_active(self, active):
            self.active = active

        def target(self):
            return SimpleNamespace(yaw=math.radians(6), pitch=math.radians(-3)) if self.active else None

        def close(self):
            pass

    controller = ReachyController(robot_factory=lambda **kwargs: robot, pose_factory=pose_factory,
        pose_interpolator=lambda a, b, t: a * (1 - t) + b * t,
        face_follow_model="public-test-model", face_follower_factory=Follower, style=SDK_SWAY)
    try:
        controller.start()
        controller.set_state("listening")
        deadline = time.monotonic() + 2
        while controller._read_head_angles()[0] < math.radians(5.5):
            assert time.monotonic() < deadline
            time.sleep(.01)
        started = time.monotonic()
        controller.gesture(expression)
        controller.play_audio(np.ones(80, dtype=np.float32))
        assert robot.media.output and time.monotonic() - started < .1
        while time.monotonic() - started < .6:
            assert controller._face_follower.active
            time.sleep(.01)
        controller.set_state("speaking")
        while time.monotonic() - started < 1.8:
            assert controller._face_follower.active
            time.sleep(.01)
        frames = [frame for frame in robot.frames if started <= frame[0] <= started + 1.6]
        coordinates = [controller._pose_parameters(frame[1]) for frame in frames]
        assert np.ptp([pose[axis] for pose in coordinates]) > minimum_range
        assert all(abs(pose["yaw"]) <= 20 and abs(pose["pitch"]) <= 12 for pose in coordinates)
        assert controller._face_follower.active and robot.wobbling
        assert controller._commands.empty()
        assert len(robot.writers) == 1 and threading.get_ident() not in robot.writers
        controller.set_state("thinking")
    finally:
        controller.close()
    assert robot.closed and not robot.wobbling


@pytest.mark.parametrize("expression, axis, amplitude", [("nod", "pitch", 8), ("shake", "yaw", 10)])
def test_speech_gesture_holds_calibrated_peaks_and_returns_to_the_original_gaze(
        expression, axis, amplitude):
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    anchor = (2., -1.)
    anchor_angle = anchor[0 if axis == "yaw" else 1]
    for age, offset in ((0, 0), (.25, amplitude), (.35, amplitude), (.45, amplitude),
                        (.95, -amplitude), (1.05, -amplitude), (1.15, -amplitude),
                        (1.6, 0), (2, 0)):
        pose = controller._gesture_pose(pose_factory(yaw=anchor[0], pitch=anchor[1]), expression, age)
        coordinates = controller._pose_parameters(pose)
        assert coordinates[axis] == pytest.approx(anchor_angle + offset)
        assert coordinates["pitch" if axis == "yaw" else "yaw"] == pytest.approx(
            anchor[1 if axis == "yaw" else 0])


@pytest.mark.parametrize("expression", ["nod", "shake"])
@pytest.mark.parametrize("anchor", [(19., 11.), (-19., -11.)])
def test_speech_gesture_respects_absolute_bounds_when_face_gaze_is_near_the_limit(expression, anchor):
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    for age in np.arange(0, 1.7, .025):
        pose = controller._gesture_pose(pose_factory(yaw=anchor[0], pitch=anchor[1]), expression, age)
        coordinates = controller._pose_parameters(pose)
        assert abs(coordinates["yaw"]) <= 20 + 1e-9
        assert abs(coordinates["pitch"]) <= 12 + 1e-9
        assert coordinates["roll"] == pytest.approx(0)
        assert coordinates["z"] == pytest.approx(0)


def test_wake_listening_pose_has_a_happy_flourish_then_settles_to_attention():
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    listening, listening_antennas = controller._state_pose("listening", 0)
    assert controller._pose_parameters(listening)["pitch"] == pytest.approx(-3)
    assert listening[2, 3] == pytest.approx(.003)
    np.testing.assert_allclose(np.rad2deg(listening_antennas), [-12, 12])
    greeting, greeting_antennas = controller._state_pose("listening", .5, "happy")
    assert greeting[2, 3] == pytest.approx(.007)
    assert controller._pose_parameters(greeting)["roll"] == pytest.approx(-8)
    np.testing.assert_allclose(np.rad2deg(greeting_antennas), [-35, 35])
    frames = [controller._state_pose("listening", t, "happy") for t in np.arange(0, 1, .025)]
    antennas = np.rad2deg([antenna for _, antenna in frames])
    yaws = [controller._pose_parameters(pose)["yaw"] for pose, _ in frames]
    assert np.ptp(antennas[:, 0]) > 25 and np.ptp(antennas[:, 1]) > 25
    assert np.ptp(yaws) > 9
    assert np.max(np.abs(antennas)) <= 53
    for time_after_cue in (1.4, 2, 6):
        settled, settled_antennas = controller._state_pose("listening", time_after_cue, "happy")
        np.testing.assert_allclose(settled, listening, atol=1e-12)
        np.testing.assert_allclose(settled_antennas, listening_antennas, atol=1e-12)


def test_close_parks_before_releasing_torque_and_all_targets_share_a_writer(hardware):
    controller, robot = hardware
    assert controller.wobbling_available and not robot.wobbling
    controller.close()
    controller.close()
    assert robot.events == ["enable", "sleep", "disable"]
    assert robot.closed and not robot.wobbling
    assert len(robot.writers) == 1


def test_face_gaze_preserves_expression_translation_roll_and_bounded_head_angles():
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    target = SimpleNamespace(yaw=math.radians(20), pitch=math.radians(12))
    controller._face_follower = SimpleNamespace(target=lambda: target)
    base = pose_factory(x=2, z=7, roll=-8, yaw=25, pitch=15)
    result = controller._pose_parameters(controller._face_pose(base))
    assert result == pytest.approx(dict(x=2, y=0, z=7, roll=-8, yaw=20, pitch=12))
    controller._face_follower = SimpleNamespace(target=lambda: None)
    assert controller._face_pose(base) is base


def test_centered_face_target_is_absolute_and_does_not_double_expression_angles():
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    target = SimpleNamespace(yaw=math.radians(6), pitch=math.radians(-3))
    controller._face_follower = SimpleNamespace(target=lambda: target)
    base = pose_factory(yaw=6, pitch=-3, z=4, roll=3)
    result = controller._pose_parameters(controller._face_pose(base))
    assert result == pytest.approx(dict(x=0, y=0, z=4, roll=3, yaw=6, pitch=-3))


def test_camera_failure_disables_gaze_without_disabling_voice_or_motion(caplog):
    active = []
    controller = ReachyController()
    controller._face_follower = SimpleNamespace(failure=RuntimeError("camera unavailable"),
                                                set_active=active.append)
    controller._update_face_tracking("listening")
    controller._update_face_tracking("speaking")
    assert active == [False, False]
    assert not controller.face_tracking_enabled and controller.motion_enabled
    assert caplog.text.count("Local face tracking stopped") == 1


def test_face_follow_shares_motor_writer_and_yields_to_explicit_commands():
    robot = FakeRobot()
    robot.media.get_frame = lambda: None
    robot.media.camera = SimpleNamespace(K=np.eye(3), D=np.zeros(5))
    events = []

    class Follower:
        active = False

        def __init__(self, read_frame, model_path, **options):
            self.read_head_angles = options["read_head_angles"]
            assert options["camera_matrix"] is robot.media.camera.K

        def start(self):
            events.append("camera-start")

        def set_active(self, active):
            self.active = active

        def target(self):
            return SimpleNamespace(yaw=.2, pitch=.05) if self.active else None

        def close(self):
            assert not robot.closed
            assert controller._motion_thread is None
            events.append("camera-close")

    controller = ReachyController(
        robot_factory=lambda **kwargs: robot, pose_factory=pose_factory,
        pose_interpolator=lambda a, b, t: a * (1 - t) + b * t,
        face_follow_model="public-test-model", face_follower_factory=Follower,
    )
    try:
        controller.start()
        follower = controller._face_follower
        assert controller.face_tracking_enabled and not follower.active
        controller.set_state("listening")
        deadline = time.monotonic() + 2
        while not follower.active and time.monotonic() < deadline:
            time.sleep(.01)
        time.sleep(.2)
        assert follower.read_head_angles()[0] > .05
        original = controller._execute_motion

        def execute(request):
            assert not follower.active
            return original(request)

        controller._execute_motion = execute
        assert controller.run_command("reachy.move", {"yaw": -10, "duration": .2, "hold": 0})["ok"]
        controller.set_state("idle")
        deadline = time.monotonic() + 2
        while follower.active and time.monotonic() < deadline:
            time.sleep(.01)
        assert not follower.active
        assert len(robot.writers) == 1
    finally:
        controller.close()
    assert not controller.face_tracking_enabled and robot.closed
    assert events == ["camera-start", "camera-close"]


def test_audio_head_wobble_runs_only_during_answers_and_keeps_one_writer(sdk_sway_hardware):
    controller, robot = sdk_sway_hardware

    def wait_for_wobbling(expected):
        deadline = time.monotonic() + 1
        while robot.wobbling != expected:
            assert time.monotonic() < deadline
            time.sleep(.005)

    assert robot.wobble_events == [False]
    controller.set_state("thinking")
    time.sleep(.06)
    thinking_started = controller._state_started
    controller.set_state("thinking")
    assert controller._state_started == thinking_started
    assert robot.wobble_events == [False]
    controller.set_state("speaking", expression="happy")
    wait_for_wobbling(True)
    controller.set_state("speaking", expression="curious")
    time.sleep(.06)
    assert robot.wobble_events == [False, True]
    controller.set_state("thinking")
    wait_for_wobbling(False)
    controller.set_state("listening", expression="happy")
    time.sleep(.06)
    assert robot.wobble_events == [False, True, False]
    assert len(robot.writers) == 1


def test_unavailable_wobble_control_keeps_primary_motion_and_audio_working():
    class Robot(FakeRobot):
        def disable_wobbling(self):
            raise RuntimeError("speech motion unsupported")

    robot = Robot()
    controller = ReachyController(
        robot_factory=lambda **kwargs: robot, pose_factory=pose_factory,
        pose_interpolator=lambda first, last, t: first * (1 - t) + last * t, style=SDK_SWAY,
    )
    try:
        controller.start()
        assert not controller.wobbling_available
        controller.set_state("speaking", expression="happy")
        controller.play_audio(np.ones(80, dtype=np.float32))
        assert robot.media.output
        assert controller.run_command("reachy.expression", {"name": "nod", "duration": .2})["ok"]
    finally:
        controller.close()
    assert robot.closed and robot.events[-2:] == ["sleep", "disable"]


def test_partial_wobble_start_is_stopped_before_falling_back(sdk_sway_hardware, monkeypatch):
    controller, robot = sdk_sway_hardware
    enable = robot.enable_wobbling

    def interrupted_start():
        enable()
        raise RuntimeError("enable request failed")

    monkeypatch.setattr(robot, "enable_wobbling", interrupted_start)
    controller.set_state("speaking", expression="happy")
    deadline = time.monotonic() + 1
    while controller.wobbling_available:
        assert time.monotonic() < deadline
        time.sleep(.005)
    assert not robot.wobbling
    assert robot.wobble_events == [False, True, False]
    controller.play_audio(np.ones(80, dtype=np.float32))
    assert robot.media.output
    assert len(robot.writers) == 1


@pytest.mark.parametrize("stop_during_shutdown", [False, True])
def test_failed_wobble_stop_reports_failure_and_still_parks_and_closes_media(sdk_sway_hardware, monkeypatch,
                                                                          stop_during_shutdown):
    controller, robot = sdk_sway_hardware
    controller.set_state("speaking")
    deadline = time.monotonic() + 1
    while not robot.wobbling:
        assert time.monotonic() < deadline
        time.sleep(.005)

    def failed_stop():
        robot._record_writer()
        raise RuntimeError("stop request failed")

    monkeypatch.setattr(robot, "disable_wobbling", failed_stop)
    if not stop_during_shutdown:
        controller.set_state("thinking")
        deadline = time.monotonic() + 1
        while controller._failure is None:
            assert time.monotonic() < deadline
            time.sleep(.005)
        with pytest.raises(ReachyHardwareError, match="could not stop speech motion"):
            controller.read_audio()
    with pytest.raises(ReachyHardwareError, match="could not stop speech motion during shutdown"):
        controller.close()
    controller.close()
    assert robot.events[-2:] == ["sleep", "disable"]
    assert robot.closed and not robot.media.playing and not robot.media.recording
    assert len(robot.writers) == 1


def test_audio_only_start_does_not_enable_or_park_motors():
    robot = FakeRobot()
    controller = ReachyController(
        enable_motion=False, robot_factory=lambda **kwargs: robot,
        pose_factory=pose_factory,
        pose_interpolator=lambda first, last, t: first * (1 - t) + last * t,
    )
    controller.start()
    assert controller.run_command("reachy.expression", {"name": "nod"})["ok"] is False
    controller.close()
    assert robot.events == [] and robot.frames == []


def test_unknown_antenna_mode_is_rejected():
    with pytest.raises(ValueError, match="antenna_mode"):
        ReachyController(antenna_mode="automatic")


@pytest.mark.parametrize("mode,inactive,active", [
    ("left", (0,), 1), ("right", (1,), 0), ("none", (0, 1), None),
])
def test_partial_antenna_mode_preserves_measured_positions_through_motion_and_park(mode, inactive, active):
    robot = FakeRobot()
    measured = np.deg2rad([-43, 57])
    robot.antennas = measured.copy()
    controller = ReachyController(
        antenna_mode=mode, robot_factory=lambda **kwargs: robot, pose_factory=pose_factory,
        pose_interpolator=lambda first, last, t: first * (1 - t) + last * t,
    )
    controller.start()
    try:
        for index in inactive:
            key = ("right_antenna", "left_antenna")[index]
            result = controller.run_command("reachy.move", {key: 0, "x": 4})
            assert result == {"ok": False, "error": f"{key} is inactive for antenna_mode={mode}"}
        controller.set_state("thinking")
        result = controller.run_command("reachy.expression", {"name": "happy", "duration": 0.2})
        assert result["ok"]
        params = {"x": 4, "duration": 0.2, "hold": 0}
        if active is not None:
            params[("right_antenna", "left_antenna")[active]] = -30 if active == 0 else 30
        result = controller.run_command("reachy.move", params)
        assert result["ok"]
        completed = [frame for frame in robot.frames if np.isclose(frame[1][0, 3], 0.004)]
        assert completed
        if active is not None:
            assert math.degrees(completed[-1][2][active]) == pytest.approx(-30 if active == 0 else 30)
    finally:
        controller.close()
    assert robot.events == ["enable", "park", "disable"]
    assert robot.closed and len(robot.writers) == 1
    for frame in robot.frames:
        for index in inactive:
            assert frame[2][index] == measured[index]
    assert robot.frames[-1][1][2, 3] == pytest.approx(-0.01)
    assert controller._pose_parameters(robot.frames[-1][1])["pitch"] == pytest.approx(10)
    if active is not None:
        assert np.ptp([frame[2][active] for frame in robot.frames]) > math.radians(20)
        assert math.degrees(robot.frames[-1][2][active]) == pytest.approx(-10 if active == 0 else 10)


def test_progress_cue_animates_the_left_ear_without_writing_from_the_caller_thread():
    robot = FakeRobot()
    measured_right = math.radians(-43)
    robot.antennas[0] = measured_right
    controller = ReachyController(
        antenna_mode="left", robot_factory=lambda **kwargs: robot, pose_factory=pose_factory,
        pose_interpolator=lambda first, last, t: first * (1 - t) + last * t,
    )
    controller.start()
    try:
        controller.set_state("thinking")
        cue_start = time.monotonic()
        controller.cue_progress()
        deadline = cue_start + 2
        while not robot.frames or robot.frames[-1][0] < cue_start + 1.4:
            assert time.monotonic() < deadline
            time.sleep(.005)
        cue_frames = [frame for frame in robot.frames if cue_start <= frame[0] <= cue_start + 1.4]
        assert len(cue_frames) >= 30
        assert np.ptp([math.degrees(frame[2][1]) for frame in cue_frames]) > 65
        assert max(abs(math.degrees(frame[2][1])) for frame in cue_frames) <= 60
        assert not robot.wobbling
    finally:
        controller.close()
    assert all(frame[2][0] == measured_right for frame in robot.frames)
    assert len(robot.writers) == 1 and threading.get_ident() not in robot.writers


@pytest.mark.parametrize("params", [
    {"roll": 21}, {"pitch": -21}, {"x": 16}, {"left_antenna": 91},
    {"body_yaw": float("nan")}, {"yaw": True}, {"yaw": 0, "duration": 0},
    {"yaw": 0, "hold": 6}, {"yaw": 0, "unexpected": "value"}, {},
])
def test_invalid_command_cannot_reach_actuators(hardware, params):
    controller, robot = hardware
    result = controller.run_command("reachy.move", params)
    assert result["ok"] is False
    assert all(abs(frame[1][0, 3]) < 0.000001 for frame in robot.frames)


def test_move_reaches_literal_pose_and_state_updates_wait_until_it_finishes(hardware):
    controller, robot = hardware
    outcomes = []
    worker = threading.Thread(target=lambda: outcomes.append(controller.run_command(
        "reachy.move", {"x": 6, "roll": 12, "right_antenna": -45,
                       "body_yaw": 20, "duration": 0.2, "hold": 0.2})))
    worker.start()
    time.sleep(0.12)
    controller.set_state("error")
    worker.join(2)
    assert outcomes and outcomes[0]["ok"] is True
    completed = [frame for frame in robot.frames if np.isclose(frame[1][0, 3], 0.006)]
    assert completed
    _, head, antennas, body = completed[-1]
    np.testing.assert_allclose(head, pose_factory(x=6, roll=12), atol=0.0002)
    assert math.isclose(antennas[0], -math.pi / 4)
    assert math.isclose(body, math.radians(20))
    assert len(robot.writers) == 1


def test_timed_out_command_stops_before_target(hardware):
    controller, robot = hardware
    result = controller.run_command("reachy.move", {"x": 15, "duration": 1}, timeout_ms=50)
    assert result == {"ok": False, "error": "Reachy Mini command timed out"}
    time.sleep(0.1)
    assert max(frame[1][0, 3] for frame in robot.frames) < 0.001


def test_builtin_expression_works_without_the_emotion_library(hardware):
    controller, robot = hardware
    result = controller.run_command("reachy.expression", {"name": "happy", "duration": 0.2})
    assert result["ok"] is True
    assert result["payload"]["source"] == "builtin"
    assert max(frame[1][2, 3] for frame in robot.frames) >= 0.004


def test_exact_recorded_expression_loads_official_library_and_bounds_the_motion():
    robot = FakeRobot()
    datasets = []

    class Move:
        duration = 0.15

        def evaluate(self, elapsed):
            return pose_factory(z=100, roll=50), np.array([-4.0, 4.0]), 4

    class Moves:
        def __init__(self, dataset):
            datasets.append(dataset)

        def list_moves(self):
            return ["happy1"]

        def get(self, name):
            assert name == "happy1"
            return Move()

    controller = start_controller(robot, moves_factory=Moves)
    try:
        result = controller.run_command("reachy.expression", {"name": "happy1", "intensity": 1})
    finally:
        controller.close()
    assert result["ok"] is True
    assert result["payload"]["source"] == "recorded"
    assert datasets == [EMOTIONS_DATASET]
    assert any(np.allclose(frame[1], pose_factory(z=20, roll=20), atol=0.001) for frame in robot.frames)
    assert max(abs(value) for frame in robot.frames for value in frame[2]) <= math.pi / 2
    assert max(abs(frame[3]) for frame in robot.frames) <= math.radians(35)


def test_motion_failure_is_reported_by_audio_calls(hardware):
    controller, robot = hardware
    robot.fail_writes = True
    deadline = time.monotonic() + 1
    while controller._failure is None and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(ReachyHardwareError, match="daemon disconnected"):
        controller.read_audio()


def test_stopped_controller_reports_a_hardware_error():
    controller = ReachyController()
    with pytest.raises(ReachyHardwareError, match="not running"):
        controller.read_audio()


def test_invalid_microphone_audio_reports_a_hardware_error(hardware):
    controller, robot = hardware
    robot.media.input = np.array([np.nan], dtype=np.float32)
    with pytest.raises(ReachyHardwareError, match="invalid microphone audio"):
        controller.read_audio()


def test_speaker_backend_fault_reports_a_hardware_error(hardware, monkeypatch):
    controller, robot = hardware

    def disconnected(samples):
        raise ConnectionError("speaker pipeline closed")

    monkeypatch.setattr(robot.media, "push_audio_sample", disconnected)
    with pytest.raises(ReachyHardwareError, match="speaker pipeline closed"):
        controller.play_audio(np.zeros(160, dtype=np.float32))
    with pytest.raises(ValueError, match="finite mono array"):
        controller.play_audio(np.array([np.nan], dtype=np.float32))


def test_unknown_initial_motor_mode_fails_before_enabling_or_parking():
    robot = FakeRobot()
    robot.client = SimpleNamespace(get_status=lambda: SimpleNamespace(backend_status=None))
    controller = ReachyController(
        robot_factory=lambda **kwargs: robot, pose_factory=pose_factory,
        pose_interpolator=lambda first, last, t: first * (1 - t) + last * t,
    )
    with pytest.raises(ReachyHardwareError, match="initial motor control mode"):
        controller.start()
    assert robot.events == []
    assert robot.closed


def test_close_keeps_motors_that_were_already_enabled():
    robot = FakeRobot()
    robot.client = SimpleNamespace(get_status=lambda: SimpleNamespace(
        backend_status=SimpleNamespace(motor_control_mode="enabled")))
    controller = ReachyController(
        robot_factory=lambda **kwargs: robot, pose_factory=pose_factory,
        pose_interpolator=lambda first, last, t: first * (1 - t) + last * t,
    )
    controller.start()
    controller.close()
    assert robot.events == ["enable"]
    assert robot.closed


def test_slow_expression_download_can_time_out_and_does_not_block_parking():
    robot = FakeRobot()
    release = threading.Event()

    def slow_library(dataset):
        release.wait(3)
        return SimpleNamespace(list_moves=lambda: ["happy1"])

    controller = start_controller(robot, moves_factory=slow_library)
    try:
        result = controller.run_command("reachy.expression", {"name": "happy1"}, timeout_ms=50)
        assert result == {"ok": False, "error": "Reachy Mini command timed out"}
        started = time.monotonic()
        controller.close()
        assert time.monotonic() - started < 1
        assert robot.events[-2:] == ["sleep", "disable"]
        frame_count = len(robot.frames)
        release.set()
        time.sleep(0.05)
        assert len(robot.frames) == frame_count
    finally:
        release.set()


def loud_speech(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * 16000)) / 16000
    return (0.3 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)


def frames_between(robot, start: float, end: float) -> list:
    return [frame for frame in robot.frames if start <= frame[0] <= end]


def head_yaws(controller, frames) -> list[float]:
    return [controller._pose_parameters(frame[1])["yaw"] for frame in frames]


def test_speech_sway_follows_the_played_audio_and_settles_after_a_flush(hardware):
    controller, robot = hardware
    controller.set_state("speaking")
    time.sleep(.5)
    silent = time.monotonic()
    time.sleep(.3)
    controller.set_state("thinking")
    time.sleep(.5)
    thinking = time.monotonic()
    controller.play_audio(loud_speech(.6))
    time.sleep(.8)
    controller.set_state("speaking")
    time.sleep(.5)
    played = time.monotonic()
    for chunk in np.array_split(loud_speech(3.0), 30):
        controller.play_audio(chunk)
    time.sleep(1.0)
    flushed = time.monotonic()
    controller.clear_audio()
    time.sleep(1.8)
    assert np.ptp(head_yaws(controller, frames_between(robot, silent, silent + .3))) < .01
    assert np.ptp(head_yaws(controller, frames_between(robot, thinking, thinking + .8))) < .01
    assert np.ptp(head_yaws(controller, frames_between(robot, played + .2, flushed))) > 5
    assert np.ptp(head_yaws(controller, frames_between(robot, flushed + 1.5, flushed + 1.8))) < .05
    assert robot.wobble_events == [False]
    assert len(robot.writers) == 1


def test_layered_motion_is_clamped_to_the_pose_limits_before_it_reaches_the_motors(hardware):
    controller, robot = hardware

    class Overshoot:
        def breathing_offset(self, now, active):
            return np.zeros(9)

        def sway_offset(self, now, active):
            return np.array([40, 40, 40, 40, 40, 60, 120, 120, 60], dtype=float)

    controller._layers = Overshoot()
    time.sleep(.2)
    swapped = time.monotonic() + .05
    time.sleep(.2)
    frames = frames_between(robot, swapped, swapped + .15)
    assert len(frames) >= 4
    for _, head, antennas, body_yaw in frames:
        assert controller._pose_parameters(head) == pytest.approx(
            {"x": 15, "y": 15, "z": 20, "roll": 20, "pitch": 20, "yaw": 35})
        np.testing.assert_allclose(np.degrees(antennas), [90, 90])
        assert math.degrees(body_yaw) == pytest.approx(35)


def antenna_spread(frames) -> float:
    return float(np.ptp([math.degrees(frame[2][0]) for frame in frames]))


@pytest.mark.parametrize("breathing, idle_spread", [(True, (25, 31)), (False, (0, .01))])
def test_idle_breathing_sways_the_antennas_and_stops_while_speaking(breathing, idle_spread):
    robot = FakeRobot()
    controller = start_controller(robot, style=MotionStyle(breathing=breathing))
    try:
        time.sleep(.5)
        idle = time.monotonic()
        time.sleep(2.0)
        controller.set_state("speaking")
        time.sleep(1.4)
        speaking = time.monotonic()
        time.sleep(.6)
    finally:
        controller.close()
    assert idle_spread[0] <= antenna_spread(frames_between(robot, idle, idle + 2.0)) <= idle_spread[1]
    assert antenna_spread(frames_between(robot, speaking, speaking + .6)) < .1


class TiltedMove:
    """A recorded move that holds a 16-degree roll for one second."""

    duration = 1.0

    def evaluate(self, elapsed):
        assert 0 <= elapsed < 1.0
        return pose_factory(roll=16), np.deg2rad([-10, 10]), 0.0


def move_library(requested: list | None = None):
    """An emotion library factory whose moves are all TiltedMove; it records the names asked for."""
    requested = [] if requested is None else requested

    class Library:
        def __init__(self, dataset):
            assert dataset == EMOTIONS_DATASET

        def list_moves(self):
            return ["tilted1", "success1", "surprised1", "inquiring2"]

        def get(self, name):
            requested.append(name)
            return TiltedMove()

    return Library


def roll_at(controller, robot, moment: float) -> float:
    frame = min(robot.frames, key=lambda frame: abs(frame[0] - moment))
    return controller._pose_parameters(frame[1])["roll"]


def test_recorded_expression_crossfades_in_and_out_of_the_resting_pose():
    robot = FakeRobot()
    controller = start_controller(robot, moves_factory=move_library())
    try:
        controller.set_state("listening")
        time.sleep(.5)
        outcomes = []
        called = time.monotonic()
        worker = threading.Thread(target=lambda: outcomes.append(controller.run_command(
            "reachy.expression", {"name": "tilted1", "intensity": 1})))
        worker.start()
        while controller._clip is None:
            assert time.monotonic() < called + 1
            time.sleep(.002)
        clip_start = controller._clip.timeline.started
        worker.join(3)
        answered = time.monotonic()
    finally:
        controller.close()
    assert outcomes[0]["ok"] is True and outcomes[0]["payload"]["duration"] == 1.0
    assert answered - clip_start == pytest.approx(1.4, abs=.1)
    rolls = [roll_at(controller, robot, clip_start + t) for t in (0.2, 0.4, 0.9, 1.2, 1.45)]
    assert rolls == pytest.approx([8, 16, 16, 8, 0], abs=1.5)

def rolls_between(controller, robot, start: float, end: float) -> list[float]:
    return [controller._pose_parameters(frame[1])["roll"] for frame in frames_between(robot, start, end)]


@pytest.mark.parametrize("expression, clip", [
    ("happy", "success1"), ("surprised", "surprised1"), ("curious", "inquiring2"), ("sad", None),
])
def test_a_spoken_expression_plays_its_recorded_clip(expression, clip):
    robot = FakeRobot()
    requested = []
    controller = start_controller(robot, moves_factory=move_library(requested))
    try:
        time.sleep(.3)
        controller.set_state("speaking", expression=expression)
        time.sleep(.3)
    finally:
        controller.close()
    assert requested == ([clip] if clip else [])


def test_a_spoken_clip_crossfades_over_the_static_pose_and_leaves_it_behind():
    robot = FakeRobot()
    controller = start_controller(robot, moves_factory=move_library())
    try:
        time.sleep(.3)
        spoken = time.monotonic()
        controller.set_state("speaking", expression="happy")
        time.sleep(1.2)
        controller.set_state("speaking", expression="happy")
        time.sleep(1.2)
        after = time.monotonic()
        time.sleep(.3)
    finally:
        controller.close()
    # TiltedMove rolls +16 degrees at intensity 0.6; the static happy pose rolls the other way.
    assert max(rolls_between(controller, robot, spoken, spoken + 1.2)) == pytest.approx(9.6, abs=.2)
    assert max(rolls_between(controller, robot, after, after + .3)) < -4


@pytest.mark.parametrize("options", [{}, {"style": MotionStyle(expression_clips=False)}],
                         ids=["offline library", "clips turned off"])
def test_a_spoken_expression_falls_back_to_its_static_pose(options):
    robot = FakeRobot()
    if options:
        options["moves_factory"] = move_library()
    controller = start_controller(robot, **options)
    try:
        time.sleep(.3)
        spoken = time.monotonic()
        controller.set_state("speaking", expression="happy")
        time.sleep(1.2)
    finally:
        controller.close()
    rolls = rolls_between(controller, robot, spoken, spoken + 1.2)
    assert max(rolls) < .5 and min(rolls) < -4


def test_a_command_during_a_spoken_clip_waits_for_its_fade_out_without_a_jump():
    robot = FakeRobot()
    controller = start_controller(robot, moves_factory=move_library())
    try:
        time.sleep(.3)
        spoken = time.monotonic()
        controller.set_state("speaking", expression="happy")
        time.sleep(.5)
        called = time.monotonic()
        result = controller.run_command("reachy.move", {"yaw": 10, "duration": .2, "hold": 0})
        answered = time.monotonic()
    finally:
        controller.close()
    assert result["ok"] is True
    assert answered - called == pytest.approx(.6, abs=.1)
    rolls = rolls_between(controller, robot, spoken, answered)
    assert max(rolls) > 9
    # The 17-degree swing from the clip back to the static pose spreads over the 0.4 s fade.
    assert max(abs(later - earlier) for earlier, later in zip(rolls, rolls[1:])) < 3


def test_sleeping_bows_the_head_and_folds_the_antennas_without_the_idle_bob():
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    for now in (0, 2.1, 7.4):
        pose, antennas = controller._state_pose("sleeping", now)
        np.testing.assert_allclose(pose, pose_factory(pitch=15, z=-10), atol=1e-12)
        np.testing.assert_allclose(np.rad2deg(antennas), [-80, 80])


def test_face_tracking_turned_off_keeps_the_follower_idle_until_turned_back_on():
    active = []
    controller = ReachyController()
    controller._face_follower = SimpleNamespace(set_active=active.append)
    controller.allow_face_tracking(False)
    controller._update_face_tracking("listening")
    controller.allow_face_tracking(True)
    controller._update_face_tracking("listening")
    controller._update_face_tracking("sleeping")
    assert active == [False, True, False]


@pytest.fixture
def volume_daemon():
    import http.server
    import json

    daemon = SimpleNamespace(volume=95, sets=[], refuse=False)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, body):
            data = json.dumps(body).encode()
            self.send_response(500 if daemon.refuse else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            assert self.path == "/api/volume/current"
            self.reply({"volume": daemon.volume, "platform": "linux", "device": "speaker"})

        def do_POST(self):
            assert self.path == "/api/volume/set"
            daemon.volume = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["volume"]
            daemon.sets.append(daemon.volume)
            self.reply({"volume": daemon.volume, "platform": "linux", "device": "speaker"})

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    daemon.port = server.server_address[1]
    yield daemon
    server.shutdown()
    server.server_close()


def test_volume_steps_through_the_daemon_and_stays_within_range(volume_daemon):
    controller = ReachyController(host="127.0.0.1", port=volume_daemon.port)
    assert controller.change_volume(10) == 100
    assert controller.change_volume(-30) == 70
    assert volume_daemon.sets == [100, 70]


def test_a_volume_the_daemon_refuses_or_cannot_reach_is_none(volume_daemon, caplog):
    volume_daemon.refuse = True
    assert ReachyController(host="127.0.0.1", port=volume_daemon.port).change_volume(10) is None
    with __import__("socket").socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    assert ReachyController(host="127.0.0.1", port=closed_port).change_volume(10) is None
    assert volume_daemon.sets == []
    assert caplog.text.count("Reachy could not change its speaker volume") == 2


def test_going_to_sleep_glides_into_the_bowed_pose_and_stops_breathing_and_sway():
    robot = FakeRobot()
    controller = start_controller(robot)
    try:
        controller.set_state("speaking")
        controller.play_audio(loud_speech(.6))
        time.sleep(.3)
        asleep = time.monotonic()
        controller.set_state("sleeping")
        time.sleep(4.0)
        settled = time.monotonic()
        time.sleep(1.0)
    finally:
        controller.close()
    gliding = frames_between(robot, asleep, settled)
    steps = [np.abs(np.degrees(later[2] - earlier[2])).max() for earlier, later in zip(gliding, gliding[1:])]
    # The 70-degree antenna fold eases in over many ticks instead of jumping.
    assert max(steps) < 10 and len([step for step in steps if step > 1]) > 10
    resting = frames_between(robot, settled, settled + 1.0)
    assert antenna_spread(resting) < .1
    np.testing.assert_allclose(np.degrees(resting[-1][2]), [-80, 80], atol=1)
    assert controller._pose_parameters(resting[-1][1]) == pytest.approx(
        {"x": 0, "y": 0, "z": -10, "roll": 0, "pitch": 15, "yaw": 0}, abs=.5)
    assert np.ptp(head_yaws(controller, resting)) < .01


def test_waking_lifts_the_head_and_unfolds_the_antennas_over_a_second_then_breathes_again():
    robot = FakeRobot()
    controller = start_controller(robot)
    try:
        controller.set_state("sleeping")
        time.sleep(4.0)
        woke = time.monotonic()
        controller.set_state("listening", expression="happy")
        time.sleep(3.5)
    finally:
        controller.close()

    def antennas_at(moment):
        frame = min(robot.frames, key=lambda frame: abs(frame[0] - moment))
        return np.degrees(frame[2])

    def pitch_at(moment):
        frame = min(robot.frames, key=lambda frame: abs(frame[0] - moment))
        return controller._pose_parameters(frame[1])["pitch"]

    assert antennas_at(woke - .05) == pytest.approx([-80, 80], abs=1)
    assert pitch_at(woke - .05) == pytest.approx(15, abs=.5)
    # Eased: barely moving after 0.1 s, about halfway at 0.5 s, unfolded by 1 s.
    assert antennas_at(woke + .1)[1] > 77
    assert 35 < antennas_at(woke + .5)[1] < 55
    assert antennas_at(woke + 1.0)[1] == pytest.approx(10, abs=3)
    assert pitch_at(woke + 1.0) < 2
    lifting = frames_between(robot, woke, woke + 1.4)
    steps = [np.abs(np.degrees(later[2] - earlier[2])).max() for earlier, later in zip(lifting, lifting[1:])]
    assert max(steps) < 6
    assert antenna_spread(frames_between(robot, woke + 1.5, woke + 3.5)) > 10


def generated(seconds, **values):
    pose = {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0, "right_antenna": -10, "left_antenna": 10,
            "body_yaw": 0, **values}
    return np.tile([pose[axis] for axis in ("x", "y", "z", "roll", "pitch", "yaw", "right_antenna", "left_antenna",
                                             "body_yaw")], (round(seconds * 25), 1)).astype(float)


def pose_at(controller, robot, moment):
    frame = min(robot.frames, key=lambda frame: abs(frame[0] - moment))
    return {**controller._pose_parameters(frame[1]), "antennas": np.degrees(frame[2]).tolist(),
            "body_yaw": math.degrees(frame[3])}


def wait_for_generated(controller, timeout=2.0):
    deadline = time.monotonic() + timeout
    while controller._generated is None:
        assert time.monotonic() < deadline, "the generated motion never started"
        time.sleep(.002)
    return controller._generated.timeline.started


def test_generated_motion_is_clamped_then_crossfades_out_when_the_answer_starts():
    robot = FakeRobot()
    controller = start_controller(robot)
    try:
        controller.set_state("thinking")
        time.sleep(.3)
        controller.play_motion(generated(1.0, roll=50, x=40, body_yaw=80, right_antenna=-150, left_antenna=120),
                               25, during={"thinking"}, loop=True, start_within_s=1.0)
        started = wait_for_generated(controller)
        time.sleep(1.6)
        answered = time.monotonic()
        controller.set_state("speaking")
        time.sleep(.6)
    finally:
        controller.close()
    held = pose_at(controller, robot, started + 1.2)
    assert (held["roll"], held["x"], held["body_yaw"]) == pytest.approx((20, 15, 35), abs=.01)
    assert held["antennas"] == pytest.approx([-90, 90], abs=.01)
    assert pose_at(controller, robot, answered - .05)["roll"] == pytest.approx(20, abs=.01)
    assert pose_at(controller, robot, answered + .5)["roll"] == pytest.approx(0, abs=.5)
    rolls = rolls_between(controller, robot, started, answered + .5)
    assert max(abs(later - earlier) for earlier, later in zip(rolls, rolls[1:])) < 3


def test_generated_motion_waits_for_its_state_and_is_dropped_once_its_window_passes():
    robot = FakeRobot()
    controller = start_controller(robot)
    try:
        time.sleep(.2)
        controller.play_motion(generated(1.0, roll=15), 25, during={"listening"}, start_within_s=.3)
        time.sleep(.5)
        listening = time.monotonic()
        controller.set_state("listening")
        time.sleep(.8)
    finally:
        controller.close()
    assert controller._generated is None
    assert max(abs(roll) for roll in rolls_between(controller, robot, listening, listening + .8)) < .5


def test_a_greeting_after_waking_plays_once_the_wake_lift_has_finished():
    robot = FakeRobot()
    controller = start_controller(robot)
    try:
        controller.set_state("sleeping")
        time.sleep(3.0)
        woke = time.monotonic()
        controller.set_state("listening", expression="happy")
        controller.play_motion(generated(1.5, roll=15), 25, during={"listening"}, start_within_s=3.0)
        started = wait_for_generated(controller, 3.0)
        time.sleep(.6)
    finally:
        controller.close()
    assert started - woke == pytest.approx(1.4, abs=.1)
    assert pose_at(controller, robot, woke + .9)["antennas"][1] == pytest.approx(10, abs=4)
    assert pose_at(controller, robot, started + .5)["roll"] == pytest.approx(15, abs=.01)


def test_stopping_generated_motion_fades_it_out_within_the_crossfade():
    robot = FakeRobot()
    controller = start_controller(robot)
    try:
        controller.set_state("listening")
        controller.play_motion(generated(3.0, pitch=-15), 25, during={"listening"})
        started = wait_for_generated(controller)
        time.sleep(.6)
        stopped = time.monotonic()
        controller.stop_motion()
        time.sleep(.6)
    finally:
        controller.close()
    assert pose_at(controller, robot, started + .5)["pitch"] == pytest.approx(-15, abs=.01)
    assert abs(pose_at(controller, robot, stopped + .45)["pitch"] + 3) < 1.1
    assert controller._generated is None


def test_generated_motion_keeps_an_inactive_antenna_where_it_was():
    robot = FakeRobot()
    robot.antennas = np.deg2rad([-30, 30])
    controller = start_controller(robot, antenna_mode="left")
    try:
        controller.set_state("listening")
        controller.play_motion(generated(1.0, right_antenna=80, left_antenna=-60), 25, during={"listening"})
        started = wait_for_generated(controller)
        time.sleep(.6)
    finally:
        controller.close()
    assert pose_at(controller, robot, started + .5)["antennas"] == pytest.approx([-30, -60], abs=.01)


def test_with_breathing_off_reachy_is_still_when_idle_and_after_generated_motion_ends():
    robot = FakeRobot()
    controller = start_controller(robot, style=MotionStyle(breathing=False))
    try:
        time.sleep(1.5)
        idle = time.monotonic()
        time.sleep(1.0)
        controller.play_motion(generated(.8, yaw=12, z=8), 25, during={"idle"})
        started = wait_for_generated(controller)
        time.sleep(1.4)
        ended = time.monotonic()
        time.sleep(1.5)
    finally:
        controller.close()
    assert pose_at(controller, robot, started + .6)["yaw"] == pytest.approx(12, abs=.01)
    for since in (idle, ended):
        frames = frames_between(robot, since, since + 1.0)
        assert len(frames) > 20
        assert np.ptp([controller._pose_parameters(frame[1])["z"] for frame in frames]) < .01
        assert np.ptp(head_yaws(controller, frames)) < .01
        assert antenna_spread(frames) < .01
    assert pose_at(controller, robot, ended + .5)["z"] == pytest.approx(0, abs=.01)


def test_generated_motion_pauses_face_tracking_and_hands_it_back():
    robot = FakeRobot()
    robot.media.get_frame = lambda: None
    robot.media.camera = SimpleNamespace(K=np.eye(3), D=np.zeros(5))
    activity = []

    class Follower:
        failure = None

        def __init__(self, read_frame, model_path, **options):
            pass

        def start(self):
            pass

        def set_active(self, active):
            if activity[-1:] != [active]:
                activity.append(active)

        def target(self):
            return None

        def close(self):
            pass

    controller = start_controller(robot, face_follow_model="public-test-model", face_follower_factory=Follower)
    try:
        controller.set_state("listening")
        time.sleep(.2)
        controller.play_motion(generated(.4), 25, during={"listening"})
        wait_for_generated(controller)
        time.sleep(1.2)
    finally:
        controller.close()
    assert activity == [True, False, True, False]


def test_generated_motion_refuses_frames_that_are_not_poses(hardware):
    controller, _ = hardware
    for frames, fps in ((np.zeros((4, 8)), 25), (np.full((4, 9), np.nan), 25), (np.zeros((0, 9)), 25), (np.zeros((4, 9)), 0)):
        with pytest.raises(ValueError, match="nine values"):
            controller.play_motion(frames, fps, during={"idle"})


def test_a_nod_during_generated_motion_shows_and_the_motion_comes_back_after_it():
    robot = FakeRobot()
    controller = start_controller(robot)
    try:
        controller.set_state("thinking")
        controller.play_motion(generated(1.0, roll=15), 25, during={"thinking"}, loop=True)
        started = wait_for_generated(controller)
        time.sleep(.6)
        nodded = time.monotonic()
        controller.gesture("nod")
        time.sleep(2.4)
    finally:
        controller.close()
    assert pose_at(controller, robot, nodded - .05)["roll"] == pytest.approx(15, abs=.01)
    during = pose_at(controller, robot, nodded + 1.05)
    assert during["roll"] == pytest.approx(0, abs=.5) and during["pitch"] < -9
    assert pose_at(controller, robot, nodded + 2.2)["roll"] == pytest.approx(15, abs=.01)
    rolls = rolls_between(controller, robot, started, nodded + 2.4)
    assert max(abs(later - earlier) for earlier, later in zip(rolls, rolls[1:])) < 3


class FakeWatcher:
    def __init__(self, read_frame, is_still, model_path):
        self.read_frame, self.is_still, self.model_path = read_frame, is_still, model_path
        self.watching = []
        self.seen_at = None
        self.failure = None
        self.closed = False

    def start(self):
        pass

    def set_watching(self, watching):
        self.watching.append(watching)

    def arrival(self):
        return self.seen_at

    def take_stats(self):
        return ArrivalStats(12, 2, 360.0, 3, 4, 0.375)

    def close(self):
        self.closed = True


def watched_controller(robot, **options):
    robot.media.get_frame = lambda: None
    watchers = []

    def factory(*args):
        watchers.append(FakeWatcher(*args))
        return watchers[-1]

    controller = start_controller(robot, arrival_model=Path("public-test-model"),
                                  arrival_watcher_factory=factory, **options)
    return controller, watchers


def test_watching_turns_the_resting_head_toward_the_door_gently_without_breathing_then_returns(caplog):
    caplog.set_level("INFO", logger="reachy_muse.hardware")
    robot = FakeRobot()
    controller, watchers = watched_controller(robot)
    try:
        time.sleep(1.0)
        turned = time.monotonic()
        controller.start_watching(WatchPose(30, 5))
        controller.stop_watching()
        controller.start_watching(WatchPose(40, 5))
        time.sleep(5.0)
        settled = time.monotonic()
        time.sleep(1.0)
        controller.stop_watching()
        left = time.monotonic()
        time.sleep(5.0)
    finally:
        controller.close()
    assert pose_at(controller, robot, settled)["yaw"] == pytest.approx(35, abs=.3)
    assert pose_at(controller, robot, settled)["pitch"] == pytest.approx(5, abs=.3)
    turning = frames_between(robot, turned, settled)
    yaws = head_yaws(controller, turning)
    speeds = [abs(later - earlier) / (after[0] - before[0])
              for earlier, later, before, after in zip(yaws, yaws[1:], turning, turning[1:])]
    assert max(speeds) < 25
    watching = frames_between(robot, settled, settled + 1.0)
    assert np.ptp([controller._pose_parameters(frame[1])["z"] for frame in watching]) < .01
    assert antenna_spread(watching) < .01
    assert pose_at(controller, robot, left + 4.9)["yaw"] == pytest.approx(0, abs=.3)
    assert pose_at(controller, robot, left + 4.9)["pitch"] == pytest.approx(0, abs=1)
    assert watchers[0].model_path == Path("public-test-model") and watchers[0].watching == [True, False, True, False]
    assert watchers[0].closed
    assert caplog.text.count("Watch pose limited to yaw 35, pitch 5 degrees") == 1


def test_sleeping_while_watching_keeps_the_bow_height_and_folded_antennas_but_faces_the_door(monkeypatch):
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    clock = [100.0]
    monkeypatch.setattr(reachy_muse.hardware.time, "monotonic", lambda: clock[0])
    controller.start_watching(WatchPose(-8, 4))
    np.testing.assert_allclose(controller._state_pose("sleeping", 3.0)[0], pose_factory(pitch=15, z=-10),
                               atol=1e-12)
    clock[0] += 2.0
    np.testing.assert_allclose(controller._state_pose("sleeping", 3.0)[0], pose_factory(yaw=-4, pitch=9.5, z=-10),
                               atol=1e-12)
    clock[0] += 2.0
    pose, antennas = controller._state_pose("sleeping", 3.0)
    np.testing.assert_allclose(pose, pose_factory(yaw=-8, pitch=4, z=-10), atol=1e-12)
    np.testing.assert_allclose(np.rad2deg(antennas), [-80, 80])
    listening, _ = controller._state_pose("listening", 3.0)
    np.testing.assert_allclose(listening, pose_factory(pitch=-3, z=3), atol=1e-12)
    controller.stop_watching()
    clock[0] += 4.0
    np.testing.assert_allclose(controller._state_pose("sleeping", 3.0)[0], pose_factory(pitch=15, z=-10),
                               atol=1e-12)


def test_sleeping_without_a_configured_watch_pose_levels_the_head_then_restores_sleep(monkeypatch):
    controller = ReachyController(pose_factory=pose_factory)
    controller._np = np
    controller.set_state("sleeping")
    clock = [100.0]
    monkeypatch.setattr(reachy_muse.hardware.time, "monotonic", lambda: clock[0])
    controller.start_watching(None)
    clock[0] += 4.0
    pose, antennas = controller._state_pose("sleeping", 10.0)
    np.testing.assert_allclose(pose, pose_factory(pitch=0, yaw=0, z=-10), atol=1e-12)
    np.testing.assert_allclose(np.rad2deg(antennas), [-80, 80])
    assert controller._state == "sleeping"
    controller.stop_watching()
    clock[0] += 4.0
    np.testing.assert_allclose(controller._state_pose("sleeping", 14.0)[0],
                               pose_factory(pitch=15, z=-10), atol=1e-12)


def wait_until(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "the camera never reached the expected stillness"
        time.sleep(.005)


def test_camera_is_still_only_while_resting_without_a_gesture_or_speech():
    robot = FakeRobot()
    controller, watchers = watched_controller(robot)
    try:
        controller.start_watching(None)
        still = watchers[0].is_still
        wait_until(still)
        controller.set_state("error")
        time.sleep(.1)
        assert not still()
        wait_until(still, timeout=6.0)
        controller.gesture("nod")
        time.sleep(.1)
        assert not still()
        wait_until(still)
        controller.set_state("speaking")
        time.sleep(.1)
        assert not still()
        controller.set_state("sleeping")
        time.sleep(.3)
        assert not still()
        wait_until(still, timeout=6.0)
    finally:
        controller.close()
    assert not still()


def test_disconnect_during_a_request_restores_camera_stillness(monkeypatch):
    import asyncio

    from musegadget.link_client import Outcome
    from test_device_command_turns import SpeakerHardware, rig_with, sent_to_muse, service
    from test_pins import Transcripts

    controller, watchers = watched_controller(FakeRobot())

    class Hardware(SpeakerHardware):
        def set_state(self, state, *, expression=None):
            super().set_state(state, expression=expression)
            controller.set_state(state, expression=expression)

    rig = rig_with(monkeypatch)
    rig.hardware = Hardware(rig.recording)

    async def check():
        disconnected = asyncio.Event()
        stop = asyncio.Event()

        async def pending(session, request):
            await disconnected.wait()

        async def link(stop_link):
            await disconnected.wait()
            return Outcome.CLOSED

        conversation = service(rig, pending, Transcripts("What's the weather tomorrow?"))
        rig.session.run = link
        serving = asyncio.create_task(conversation.serve(rig.session, {"vm_id": "vm-1"}, stop))
        try:
            await rig.quiet()
            await rig.wait_until_or_timeout(watchers[0].is_still, 5)
            assert watchers[0].is_still()
            rig.say()
            await rig.wait_until_or_timeout(lambda: sent_to_muse(rig.recording.events), 5)
            assert sent_to_muse(rig.recording.events) == ["What's the weather tomorrow?"]
            assert controller._state == "thinking"
            await rig.wait_until_or_timeout(lambda: not watchers[0].is_still())
            assert not watchers[0].is_still()
            disconnected.set()
            assert await asyncio.wait_for(serving, 5) is Outcome.CLOSED
            await rig.wait_until_or_timeout(watchers[0].is_still, 6)
            assert watchers[0].is_still() is True
            assert watchers[0].watching == [True]
        finally:
            disconnected.set()
            await asyncio.wait_for(serving, 5)
            await asyncio.gather(*rig.session.scripts)

    try:
        controller.start_watching(None)
        asyncio.run(asyncio.wait_for(check(), 20))
    finally:
        controller.close()


def test_arrival_is_none_with_one_warning_when_there_is_no_person_model(caplog):
    robot = FakeRobot()
    controller = start_controller(robot)
    try:
        controller.start_watching(WatchPose(10, 0))
        controller.start_watching(WatchPose(10, 0))
        assert controller.arrival() is None
    finally:
        controller.close()
    assert caplog.text.count("Arrival detection is unavailable") == 1


def test_arrival_reports_the_watcher_time_and_a_failed_camera_is_none_with_one_warning(caplog):
    robot = FakeRobot()
    controller, watchers = watched_controller(robot)
    try:
        controller.start_watching(None)
        watchers[0].seen_at = 1234.5
        assert controller.arrival() == 1234.5
        watchers[0].failure = RuntimeError("camera unavailable")
        assert controller.arrival() is None
        assert controller.arrival() is None
    finally:
        controller.close()
    assert caplog.text.count("Arrival detection is unavailable: RuntimeError") == 1


def test_a_failed_camera_watcher_is_replaced_the_next_time_reachy_watches(caplog):
    robot = FakeRobot()
    controller, watchers = watched_controller(robot)
    try:
        controller.start_watching(None)
        watchers[0].failure = RuntimeError("camera unavailable")
        assert controller.arrival() is None
        controller.stop_watching()
        controller.start_watching(None)
        watchers[1].seen_at = 99.0
        arrived = controller.arrival()
        watchers[1].failure = OSError("camera gone")
        assert controller.arrival() is None
    finally:
        controller.close()
    assert (len(watchers), watchers[0].closed, watchers[1].watching, arrived) == (2, True, [True], 99.0)
    assert caplog.text.count("Arrival detection is unavailable: RuntimeError") == 1
    assert caplog.text.count("Arrival detection is unavailable: OSError") == 1


def test_minute_diagnostics_are_numeric_and_restart_a_failed_watcher(caplog):
    caplog.set_level("INFO", logger="reachy_muse.hardware")
    robot = FakeRobot()
    controller, watchers = watched_controller(robot)
    try:
        controller.start_watching(None)
        controller.set_state("sleeping")
        controller._read_head_angles = lambda: (math.radians(-12), math.radians(3))
        controller._camera_still = lambda: True
        watchers[0].failure = RuntimeError("private camera details")
        controller.log_arrival_diagnostics()
        watchers[1].seen_at = 99.0
        assert controller.arrival() == 99.0
        assert watchers[0].closed is True
    finally:
        controller.close()
    lines = [record.getMessage() for record in caplog.records if record.getMessage().startswith("Arrival stats")]
    assert lines == ["Arrival stats frames=12 frame_errors=2 brightness=30.0 motion_triggers=3 "
                     "detector_runs=4 max_person_confidence=0.375 pitch=3.0 yaw=-12.0 asleep=1 still=1"]
    assert "private camera details" not in caplog.text
