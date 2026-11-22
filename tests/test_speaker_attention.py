"""Run the real motion owner against fake motors and a deterministic clock."""

import math
import threading
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

import reachy_muse.hardware as hardware
from reachy_muse.hardware import ReachyController, _Command
from reachy_muse.motion import MotionStyle
from reachy_muse.motion.layers import MotionLayers, Ramp
from test_doa_http import doa_server
from test_hardware import FakeRobot, generated, pose_factory


def simulate(monkeypatch, events, *, ticks=120, follower=None):
    class Robot(FakeRobot):
        def set_target(self, **values):
            super().set_target(**values)
            _, head, antennas, body = self.frames[-1]
            self.frames[-1] = (clock.now, head, antennas, body)

    robot = Robot()
    controller = ReachyController(pose_factory=pose_factory,
                                  pose_interpolator=lambda a, b, t: a * (1 - t) + b * t,
                                  sleep_on_close=False, style=MotionStyle(breathing=False, expression_clips=False))
    controller._np, controller._robot = np, robot
    controller._output_sample_rate = 16000
    controller._layers, controller._generated_gain = MotionLayers(controller._style), Ramp(1)
    controller._face_follower = follower

    class Clock:
        now, tick = 1000.0, 0
        times = {}

        def is_set(self):
            return self.tick >= ticks

        def wait(self, duration):
            self.now += duration
            if controller._started.is_set():
                self.tick += 1
                self.times[self.tick] = self.now
                if self.tick in events:
                    events[self.tick](controller)
            return self.is_set()

    clock = controller._stop = Clock()
    monkeypatch.setattr(hardware, "time", SimpleNamespace(monotonic=lambda: clock.now))
    controller._motion_loop()
    assert controller._failure is None
    return controller, robot, clock


def pose_at(controller, robot, at):
    _, head, antennas, body = min(robot.frames, key=lambda frame: abs(frame[0] - at))
    return {**controller._pose_parameters(head), "body_yaw": math.degrees(body),
            "right_antenna": math.degrees(antennas[0]), "left_antenna": math.degrees(antennas[1])}


@pytest.mark.parametrize("angle, head, body", [(math.radians(80), 10, 0), (math.pi / 3, 30, 10), (math.pi, -35, -15)])
def test_wake_yaw_eases_over_650ms_and_holds_without_a_face(monkeypatch, angle, head, body):
    def wake(controller):
        controller.look_at_speaker(angle)
        controller.set_state("listening", expression="happy")

    controller, robot, clock = simulate(monkeypatch, {5: wake, 50: lambda c: c.set_state("thinking")})
    start = clock.times[5]
    half = pose_at(controller, robot, start + .325)
    final = pose_at(controller, robot, start + .65)
    assert (half["yaw"], half["body_yaw"]) == pytest.approx((head / 2, body / 2), abs=.01)
    assert (final["yaw"], final["body_yaw"]) == pytest.approx((head, body), abs=.01)
    held = pose_at(controller, robot, clock.now)
    assert (held["yaw"], held["body_yaw"]) == pytest.approx((head, body), abs=.01)
    early = pose_at(controller, robot, start + .1)
    assert abs(early["yaw"]) < 1.1
    yaws = [controller._pose_parameters(frame[1])["yaw"] for frame in robot.frames]
    assert max(abs(b - a) for a, b in zip(yaws, yaws[1:])) < 2.6
    assert len(robot.writers) == 1


def test_repeated_wake_from_the_same_speaker_keeps_the_turned_gaze(monkeypatch, doa_server):
    source, _ = doa_server(*[f'{{"angle": {angle}, "speech_detected": true}}'
                             for angle in (math.pi / 3, math.pi / 2)])

    def wake(controller):
        controller._connect_options = source._connect_options
        reading = controller.read_doa()
        controller.look_at_speaker(reading.angle)
        controller.set_state("listening")

    controller, robot, clock = simulate(monkeypatch, {5: wake, 60: wake}, ticks=120)
    first = pose_at(controller, robot, clock.times[55])
    second = pose_at(controller, robot, clock.now)
    assert (first["yaw"], first["body_yaw"]) == pytest.approx((30, 10), abs=.01)
    assert (second["yaw"], second["body_yaw"]) == pytest.approx((30, 10), abs=.01)


@pytest.mark.parametrize("watch_tick", [10, 60])
def test_arrival_watch_replaces_wake_direction_at_the_motors(monkeypatch, watch_tick):
    from reachy_muse.arrival import WatchPose
    from test_hardware import FakeWatcher

    def watch(controller):
        controller._arrival_watcher = FakeWatcher(None, None, None)
        controller.set_state("idle")
        controller.start_watching(WatchPose(-20, 5))

    controller, robot, clock = simulate(monkeypatch, {
        5: lambda c: c.look_at_speaker(math.pi / 3),
        watch_tick: watch,
        320: lambda c: c.stop_watching(),
    }, ticks=520)
    watching = pose_at(controller, robot, clock.times[300])
    assert (watching["yaw"], watching["pitch"], watching["body_yaw"]) == pytest.approx((-20, 5, 0), abs=.1)
    rested = pose_at(controller, robot, clock.now)
    assert (rested["yaw"], rested["pitch"], rested["body_yaw"]) == pytest.approx((0, 0, 0), abs=.1)


def test_direction_turn_and_sleep_lift_run_together_and_sleep_releases_the_hold(monkeypatch):
    def wake(controller):
        controller.look_at_speaker(math.pi / 3)
        controller.set_state("listening", expression="happy")

    controller, robot, clock = simulate(monkeypatch, {
        2: lambda c: c.set_state("sleeping"), 180: wake, 300: lambda c: c.set_state("sleeping"),
    }, ticks=470)
    start = clock.times[180]
    halfway = pose_at(controller, robot, start + .325)
    assert (halfway["yaw"], halfway["body_yaw"]) == pytest.approx((15, 5), abs=.01)
    assert halfway["left_antenna"] > 60
    final_yaw = pose_at(controller, robot, start + .65)
    assert (final_yaw["yaw"], final_yaw["body_yaw"]) == pytest.approx((30, 10), abs=.01)
    lifted = pose_at(controller, robot, start + 1)
    assert (lifted["pitch"], lifted["left_antenna"]) == pytest.approx((-3, 12), abs=.5)
    asleep = pose_at(controller, robot, clock.now)
    assert (asleep["yaw"], asleep["body_yaw"], asleep["pitch"], asleep["z"], asleep["left_antenna"]) == pytest.approx(
        (0, 0, 15, -10, 80), abs=.1)


def test_face_tracking_starts_after_turn_and_holds_the_last_face_when_lost(monkeypatch):
    class Follower:
        failure = None
        active = None
        observation = None

        def __init__(self):
            self.activity = []

        def set_active(self, active):
            if self.active != active:
                self.activity.append(active)
                self.active = active
                self.observation = None

        def target(self):
            return self.observation if self.active else None

    follower = Follower()

    def wake(controller):
        controller.look_at_speaker(math.pi / 3)

    def face(controller):
        follower.observation = SimpleNamespace(yaw=math.radians(-5), pitch=math.radians(-3))

    def lost(controller):
        follower.observation = None

    controller, robot, clock = simulate(monkeypatch, {
        2: lambda c: c.set_state("listening"), 5: wake, 36: face, 80: lost,
    }, follower=follower)
    after_turn = pose_at(controller, robot, clock.times[5] + .65)
    assert (after_turn["yaw"], after_turn["body_yaw"]) == pytest.approx((30, 10), abs=.01)
    held = pose_at(controller, robot, clock.now)
    assert (held["yaw"], held["body_yaw"]) == pytest.approx((25, 10), abs=.01)
    assert follower.activity == [False, True, False, True]
    assert np.degrees(controller._read_head_angles()) == pytest.approx((-5, -3), abs=.01)


def test_idle_face_gaze_yields_to_wake_then_conversation_tracking(monkeypatch):
    from test_face_tracking_motion import Follower

    follower = Follower()
    follower.yaw = -8
    during_turn = []

    def wake(controller):
        controller.look_at_speaker(math.pi / 3)
        controller.set_state("listening")

    controller, robot, clock = simulate(monkeypatch, {
        1: lambda c: setattr(c, "_face_follow_idle", True),
        80: wake,
        90: lambda c: during_turn.append(follower.active),
        110: lambda c: setattr(follower, "yaw", -5),
    }, ticks=180, follower=follower)
    before = pose_at(controller, robot, clock.times[79])
    halfway = pose_at(controller, robot, clock.times[80] + .325)
    turned = pose_at(controller, robot, clock.times[80] + .65)
    handed_over = pose_at(controller, robot, clock.now)
    assert before["yaw"] == pytest.approx(-8, abs=.02)
    assert (halfway["yaw"], halfway["body_yaw"]) == pytest.approx((11, 5), abs=.02)
    assert (turned["yaw"], turned["body_yaw"]) == pytest.approx((30, 10), abs=.02)
    assert (handed_over["yaw"], handed_over["body_yaw"]) == pytest.approx((25, 10), abs=.02)
    assert during_turn == [False]
    assert follower.active is True


@pytest.mark.parametrize("clip", ["generated", "recorded"])
def test_existing_clip_wins_and_another_wake_can_turn_afterwards(monkeypatch, clip, caplog):
    caplog.set_level("INFO", logger="reachy_muse.hardware")
    request = _Command("reachy.expression", {"name": "offline", "intensity": 1.0}, 1010)

    def play(controller):
        controller.set_state("listening")
        if clip == "generated":
            controller.play_motion(generated(1, yaw=-12), 25, during={"listening"})
        else:
            move = SimpleNamespace(duration=1.0, evaluate=lambda elapsed: (pose_factory(yaw=-12), np.zeros(2), 0))
            controller._moves = SimpleNamespace(list_moves=lambda: ["offline"], get=lambda _: move)
            controller._commands.put_nowait(request)

    controller, robot, clock = simulate(monkeypatch, {
        2: play, 25: lambda c: c.look_at_speaker(math.pi / 3),
        150: lambda c: c.look_at_speaker(math.pi / 3),
    }, ticks=190)
    showing = pose_at(controller, robot, clock.times[25])
    assert showing["yaw"] == pytest.approx(-12, abs=.01)
    rested = pose_at(controller, robot, clock.times[145])
    assert (rested["yaw"], rested["body_yaw"]) == pytest.approx((0, 0), abs=.1)
    turned = pose_at(controller, robot, clock.now)
    assert (turned["yaw"], turned["body_yaw"]) == pytest.approx((30, 10), abs=.01)
    assert [record.message for record in caplog.records if record.message.startswith("Speaker gaze")] == [
        "Speaker gaze yaw=30.0 body_yaw=10.0 decision=busy",
        "Speaker gaze yaw=30.0 body_yaw=10.0 decision=turn",
    ]


def test_an_explicit_command_cancels_the_turn_and_held_direction(monkeypatch):
    request = _Command("reachy.move", {"yaw": -10, "body_yaw": -5, "duration": .2, "hold": .1}, 1010)
    controller, robot, clock = simulate(monkeypatch, {
        2: lambda c: c.look_at_speaker(math.pi / 3), 15: lambda c: c._commands.put_nowait(request),
    })
    assert request.result["payload"]["coordinates"] == pytest.approx(
        {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": -10})
    rested = pose_at(controller, robot, clock.now)
    assert (rested["yaw"], rested["body_yaw"]) == pytest.approx((0, 0), abs=.1)
    half_turn = pose_at(controller, robot, clock.times[15])
    assert half_turn["yaw"] == pytest.approx(12.85, abs=.1)


def test_a_wake_during_blocking_motion_does_not_turn_after_the_command(monkeypatch):
    request = _Command("reachy.move", {"yaw": -12, "duration": .8, "hold": .1}, 1010)
    controller, robot, clock = simulate(monkeypatch, {
        2: lambda c: c._commands.put_nowait(request), 10: lambda c: c.look_at_speaker(math.pi / 3),
        150: lambda c: c.look_at_speaker(math.pi / 3),
    }, ticks=190)
    assert request.result["payload"]["coordinates"]["yaw"] == -12
    rested = pose_at(controller, robot, clock.times[145])
    assert (rested["yaw"], rested["body_yaw"]) == pytest.approx((0, 0), abs=.1)
    turned = pose_at(controller, robot, clock.now)
    assert (turned["yaw"], turned["body_yaw"]) == pytest.approx((30, 10), abs=.01)


def test_command_claims_attention_before_disabling_face_tracking(monkeypatch):
    request = _Command("reachy.move", {"yaw": -12, "duration": .2, "hold": .1}, 1010)

    class Follower:
        failure = None
        controller = None
        attempted = False

        def set_active(self, active):
            if active or self.controller is None or self.attempted:
                return
            self.attempted = True
            thread = threading.Thread(target=self.controller.look_at_speaker, args=(math.pi / 3,))
            thread.start()
            thread.join(2)
            assert not thread.is_alive()

        def target(self):
            return None

    follower = Follower()

    def command(controller):
        follower.controller = controller
        controller._commands.put_nowait(request)

    controller, robot, clock = simulate(monkeypatch, {2: command}, ticks=100, follower=follower)
    assert follower.attempted
    assert request.result["ok"]
    rested = pose_at(controller, robot, clock.now)
    assert (rested["yaw"], rested["body_yaw"]) == pytest.approx((0, 0), abs=.1)


@pytest.mark.parametrize("gesture", ["nod", "tilt"])
@pytest.mark.parametrize("angle, yaw", [(0, 35), (math.pi, -35)])
def test_gestures_keep_a_side_directed_wake_gaze(monkeypatch, gesture, angle, yaw):
    controller, robot, clock = simulate(monkeypatch, {
        5: lambda c: c.look_at_speaker(angle), 50: lambda c: c.gesture(gesture),
    }, ticks=130)
    before = pose_at(controller, robot, clock.times[49])
    assert before["yaw"] == pytest.approx(yaw, abs=.01)
    during = [controller._pose_parameters(frame[1])["yaw"] for frame in robot.frames
              if clock.times[50] <= frame[0] <= clock.times[50] + 1.6]
    assert (min(during), max(during)) == pytest.approx((yaw, yaw), abs=.01)


@pytest.mark.parametrize("queued_together", [False, True])
def test_a_later_greeting_clip_waits_for_the_direction_turn(monkeypatch, queued_together):
    def greet(controller):
        controller.play_motion(generated(1, yaw=-12), 25, during={"listening"})

    def wake(controller):
        controller.set_state("listening", expression="happy")
        controller.look_at_speaker(math.pi / 3)
        if queued_together:
            greet(controller)

    controller, robot, clock = simulate(monkeypatch, {
        2: wake, 5: lambda c: None if queued_together else greet(c),
    }, ticks=150)
    halfway = pose_at(controller, robot, clock.times[2] + .325)
    assert (halfway["yaw"], halfway["body_yaw"]) == pytest.approx((15, 5), abs=.01)
    showing = pose_at(controller, robot, clock.times[2] + 1.2)
    # Expressions remain offsets from the held gaze after the wake turn.
    assert showing["yaw"] == pytest.approx(18, abs=.01)
    held = pose_at(controller, robot, clock.now)
    assert (held["yaw"], held["body_yaw"]) == pytest.approx((30, 10), abs=.01)


def test_a_direction_request_does_not_wake_a_sleeping_robot_by_itself(monkeypatch, caplog):
    caplog.set_level("INFO", logger="reachy_muse.hardware")
    controller, robot, clock = simulate(monkeypatch, {
        2: lambda c: c.set_state("sleeping"), 180: lambda c: c.look_at_speaker(math.pi / 3),
        240: lambda c: c.set_state("listening"),
    }, ticks=340)
    sleeping = pose_at(controller, robot, clock.times[235])
    assert (sleeping["yaw"], sleeping["body_yaw"], sleeping["pitch"]) == pytest.approx((0, 0, 15), abs=.1)
    awake = pose_at(controller, robot, clock.now)
    assert (awake["yaw"], awake["body_yaw"], awake["pitch"]) == pytest.approx((0, 0, -3), abs=.1)
    assert [record.message for record in caplog.records if record.message.startswith("Speaker gaze")] == [
        "Speaker gaze yaw=30.0 body_yaw=10.0 decision=stale",
    ]


def test_still_mode_keeps_wake_state_and_refuses_direction_motion(caplog):
    caplog.set_level("INFO", logger="reachy_muse.hardware")
    controller = ReachyController(enable_motion=False)
    controller.set_state("sleeping")
    controller.look_at_speaker(math.pi / 3)
    controller.set_state("listening", expression="happy")
    assert (controller.motion_enabled, controller._state, controller._speaker_pending) == (False, "listening", None)
    assert [record.message for record in caplog.records] == ["Speaker gaze angle=1.047 decision=still"]


def play_thinking(source):
    def play(controller):
        controller.set_state('thinking')
        if source == 'generated':
            controller.play_motion(generated(8, yaw=12), 25, during={'thinking'}, loop=True)
        else:
            from reachy_muse.hardware import _Clip, _Frames
            from reachy_muse.motion.layers import ClipTimeline
            controller._clip = _Clip(_Frames(np.asarray(generated(8, yaw=12)), 25, False, pose_factory),
                                     ClipTimeline(1001.5, 8), 1)
    return play


def look_at_friend(controller):
    assert controller.look_at_person(math.radians(-30), math.radians(6)) is True


@pytest.mark.parametrize('source', ['generated', 'clip'])
def test_person_turn_fades_motion_holds_and_releases(monkeypatch, source):
    controller, robot, clock = simulate(monkeypatch, {
        2: play_thinking(source), 25: look_at_friend, 130: lambda c: c.release_person_gaze(),
    }, ticks=230)
    turned = pose_at(controller, robot, clock.times[100])
    assert (turned['yaw'], turned['body_yaw'], turned['pitch']) == pytest.approx((-30, -10, 6), abs=.2)
    released = pose_at(controller, robot, clock.now)
    assert (released['yaw'], released['body_yaw']) == pytest.approx((0, 0), abs=.1)
    assert controller._person_hold is None
    assert controller._generated is controller._clip is None
    yaws = [controller._pose_parameters(frame[1])['yaw'] for frame in robot.frames]
    assert max(abs(b - a) for a, b in zip(yaws, yaws[1:])) < 4
    assert len(robot.writers) == 1


def listen_to_speaker_at_30(controller):
    controller.look_at_speaker(math.pi / 3)
    controller.set_state('listening')


@pytest.mark.parametrize('replacement', ['same_tick', 'after_release'])
def test_a_second_person_gaze_still_turns_back_to_the_first_speaker(monkeypatch, replacement):
    def replace_now(controller):
        controller.release_person_gaze()
        assert controller.look_at_person(math.radians(-10), 0)

    events = {5: listen_to_speaker_at_30, 60: lambda c: c.look_at_person(math.radians(-30), 0),
              180: lambda c: c.release_person_gaze()}
    if replacement == 'same_tick':
        events[120] = replace_now
    else:
        events[120] = lambda c: c.release_person_gaze()
        events[125] = lambda c: c.look_at_person(math.radians(-10), 0)
    controller, robot, clock = simulate(monkeypatch, events, ticks=260)
    assert round(pose_at(controller, robot, clock.times[175])['yaw']) == -10
    final = pose_at(controller, robot, clock.now)
    assert (round(final['yaw']), round(final['body_yaw'])) == (30, 10)


def test_a_wake_turn_during_a_person_gaze_ends_it(monkeypatch):
    controller, robot, clock = simulate(monkeypatch, {
        5: listen_to_speaker_at_30, 60: lambda c: c.look_at_person(math.radians(-30), 0),
        120: lambda c: c.look_at_speaker(math.radians(110)), 180: lambda c: c.release_person_gaze(),
    }, ticks=260)
    assert round(pose_at(controller, robot, clock.times[175])['yaw']) == -20
    assert controller._person_hold is None
    assert round(pose_at(controller, robot, clock.now)['yaw']) == -20


@pytest.mark.parametrize('source', ['generated', 'clip'])
def test_commands_wait_only_while_reachy_turns_toward_the_person(monkeypatch, source):
    turning = _Command('reachy.move', {'yaw': 0, 'duration': .2, 'hold': 0}, 1010)
    facing = _Command('reachy.move', {'yaw': 10, 'duration': .2, 'hold': .5}, 1010)
    controller, robot, clock = simulate(monkeypatch, {
        2: play_thinking(source), 25: look_at_friend, 30: lambda c: c._commands.put_nowait(turning),
        100: lambda c: c._commands.put_nowait(facing),
    }, ticks=200)
    assert turning.result == {'ok': False, 'error': 'Reachy is still turning toward a person'}
    turned = pose_at(controller, robot, clock.times[99])
    assert (turned['yaw'], turned['body_yaw'], turned['pitch']) == pytest.approx((-30, -10, 6), abs=.2)
    assert facing.result['ok'] is True
    assert pose_at(controller, robot, clock.times[100] + .4)['yaw'] == pytest.approx(10, abs=.2)
    rested = pose_at(controller, robot, clock.now)
    assert (rested['yaw'], rested['body_yaw']) == pytest.approx((0, 0), abs=.2)
    assert controller._person_hold is None
    assert len(robot.writers) == 1


def face_the_speaker_at_30(controller):
    assert controller.face_speaker(math.pi / 3) is True


@pytest.mark.parametrize('source', ['generated', 'clip'])
def test_a_turn_toward_the_speaker_refuses_commands_only_while_it_moves_and_holds_nothing(monkeypatch, source):
    turning = _Command('reachy.move', {'yaw': 0, 'duration': .2, 'hold': 0}, 1010)
    seen = []

    def command_mid_turn(controller):
        seen.append(('moving', controller._person_pending is None and controller._speaker_turn is not None))
        controller._commands.put_nowait(turning)

    controller, robot, clock = simulate(monkeypatch, {
        2: play_thinking(source), 25: face_the_speaker_at_30, 55: command_mid_turn,
        110: lambda c: seen.append(('turning', c.person_turning)), 130: lambda c: c.release_person_gaze(),
    }, ticks=230)
    assert turning.result == {'ok': False, 'error': 'Reachy is still turning toward a person'}
    assert seen == [('moving', True), ('turning', False)]
    assert controller._person_hold is None
    for at in (clock.times[100], clock.now):
        faced = pose_at(controller, robot, at)
        assert (faced['yaw'], faced['body_yaw']) == pytest.approx((30, 10), abs=.2)


def test_sleep_releases_person_gaze(monkeypatch):
    controller, robot, clock = simulate(monkeypatch, {
        5: lambda c: c.look_at_person(math.radians(80), math.radians(-80)),
        90: lambda c: c.set_state('sleeping'),
    }, ticks=200)
    turned = pose_at(controller, robot, clock.times[70])
    assert (turned['yaw'], turned['body_yaw'], turned['pitch']) == pytest.approx((35, 15, -20), abs=.1)
    assert not controller._person_hold
    assert pose_at(controller, robot, clock.now)['yaw'] == pytest.approx(0, abs=.1)


class CenteringFollower:
    """A face follower that sees no face, and records where Reachy centers it."""

    failure = None
    active = False

    def __init__(self):
        self.centers = []

    def set_active(self, active, *, idle=False):
        self.active = active

    def target(self):
        return None

    def hold_center(self, yaw):
        self.centers.append(yaw)


def follow_from(yaw, results):
    """A later speaker at head yaw ``yaw`` in degrees; the outcome of Reachy's turn toward them lands in ``results``."""
    return lambda controller: results.append(controller.follow_speaker(math.radians(90 - yaw)))


@pytest.mark.parametrize("speaker, delta, faced, center", [(-25, -55, (-25, -5), 0), (-60, -65, (-35, -15), -25)])
def test_a_later_speaker_gets_a_wake_turn_and_the_follower_starts_on_them(monkeypatch, caplog, speaker, delta, faced,
                                                                         center):
    caplog.set_level("INFO", logger="reachy_muse.hardware")
    follower, results = CenteringFollower(), []
    controller, robot, clock = simulate(monkeypatch, {5: listen_to_speaker_at_30, 60: follow_from(speaker, results)},
                                        follower=follower)
    assert results == [("turned", pytest.approx(delta))]
    halfway = pose_at(controller, robot, clock.times[60] + .325)
    assert (halfway["yaw"], halfway["body_yaw"]) == pytest.approx(((30 + faced[0]) / 2, (10 + faced[1]) / 2), abs=.01)
    turned = pose_at(controller, robot, clock.now)
    assert (turned["yaw"], turned["body_yaw"]) == pytest.approx(faced, abs=.01)
    assert follower.centers == [pytest.approx(math.radians(center))]
    assert controller.wake_gaze_bearing == pytest.approx(math.pi / 3)
    assert [record.message for record in caplog.records if record.message.startswith("Speaker gaze")] == [
        "Speaker gaze yaw=30.0 body_yaw=10.0 decision=turn",
        f"Speaker gaze yaw={faced[0]:.1f} body_yaw={faced[1]:.1f} decision=turn"]


def test_a_speaker_the_head_faces_or_one_right_after_a_turn_gets_no_turn(monkeypatch):
    results = []
    controller, robot, clock = simulate(monkeypatch, {
        5: listen_to_speaker_at_30, 60: follow_from(15, results), 61: follow_from(-25, results),
        100: follow_from(30, results), 150: follow_from(30, results),
    }, ticks=200)
    assert results == [("same", pytest.approx(-15)), ("turned", pytest.approx(-55)), ("too_soon", pytest.approx(55)),
                       ("turned", pytest.approx(55))]
    waited = pose_at(controller, robot, clock.times[149])
    assert (waited["yaw"], waited["body_yaw"]) == pytest.approx((-25, -5), abs=.01)
    turned = pose_at(controller, robot, clock.now)
    assert (turned["yaw"], turned["body_yaw"]) == pytest.approx((30, 10), abs=.01)


def greet_friend_at_minus_30(controller):
    assert controller.look_at_person(math.radians(-30), 0) is True


@pytest.mark.parametrize("owner, outcome, faced", [
    ("tracking_off", "off", (0, 0)),
    ("greeting", "held", (-30, -10)),
    ("greeting_pending", "held", (-30, -10)),
    ("muse_turn", "busy", (-30, -10)),
    ("command", "busy", (0, 0)),
    ("thinking_motion", "busy", (12, 0)),
    ("sleep", "busy", (0, 0)),
])
def test_a_later_speaker_waits_while_a_person_turn_or_other_motion_owns_the_head(monkeypatch, owner, outcome, faced):
    """A later speaker stands at +60 degrees, where the head would turn to 35."""
    results = []
    follow = follow_from(60, results)
    move = _Command("reachy.move", {"yaw": -30, "duration": .3, "hold": .4}, 1010)
    controller, robot, clock = simulate(monkeypatch, {
        "tracking_off": {2: lambda c: c.allow_face_tracking(False), 30: follow},
        "greeting": {10: greet_friend_at_minus_30, 40: follow},
        "greeting_pending": {40: lambda c: (greet_friend_at_minus_30(c), follow(c))},
        "muse_turn": {10: lambda c: c.face_speaker(math.radians(120)), 15: follow},
        "command": {10: lambda c: c._commands.put_nowait(move), 20: follow},
        "thinking_motion": {2: play_thinking("generated"), 30: follow},
        "sleep": {2: lambda c: c.set_state("sleeping"), 30: follow},
    }[owner], ticks=200)
    assert results == [(outcome, None)]
    final = pose_at(controller, robot, clock.now)
    assert (round(final["yaw"]), round(final["body_yaw"])) == faced


def test_a_later_speaker_gets_a_turn_through_a_nod(monkeypatch):
    results = []
    controller, robot, clock = simulate(monkeypatch, {
        5: listen_to_speaker_at_30, 50: lambda c: c.gesture("nod"), 55: follow_from(-25, results),
    }, ticks=160)
    assert results == [("turned", pytest.approx(-55))]
    turned = pose_at(controller, robot, clock.now)
    assert (turned["yaw"], turned["body_yaw"]) == pytest.approx((-25, -5), abs=.01)

