"""Visual admission, wake attention and a reply through the real motion owner."""

import math
import time
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from reachy_muse.motion import MotionStyle
from reachy_muse.speaker_direction import DoAReading
from reachy_muse.wake_policy import Verification, WakePolicy
from reachy_muse.voice_verify import Decision
from test_device_command_turns import played, rig_with, run, sent_to_muse, service
from test_face_tracking_motion import Follower
from test_hardware import FakeRobot, generated, np, start_controller
from test_pins import Transcripts, Vad, Wake


class Tick(NamedTuple):
    """One motion tick, recorded on the motion thread as it wrote the robot's targets."""

    began_after: float  # when the previous tick wrote; this tick read its clock later
    written: float
    turning: bool
    following: bool
    generated: object
    yaw: float
    body_yaw: float


@pytest.mark.parametrize("visual", [False, True])
@pytest.mark.parametrize("acoustic", [False, True])
@pytest.mark.parametrize("verify_voice", [False, True])
def test_wave_turn_hands_over_to_face_tracking_through_a_whole_reply(monkeypatch, visual, acoustic, verify_voice):
    rig = rig_with(monkeypatch)
    monkeypatch.setattr("webrtcvad.Vad", lambda mode: Vad())
    robot = FakeRobot()
    robot.media.camera = SimpleNamespace(
        K=np.array([[320., 0, 320], [0, 320, 240], [0, 0, 1]]), D=np.zeros(5))
    controller = start_controller(robot, wave_wake=True, face_follow_idle=True, sleep_on_close=False,
                                  style=MotionStyle(breathing=False, expression_clips=False, sway="sdk"))
    follower = Follower()
    follower.visible = False
    follower.close = lambda: None
    controller._face_follower = follower

    class Camera:
        active = False
        pending = False
        wave_position = None

        def set_active(self, active):
            self.active = active
            if not active:
                self.pending = False
                self.wave_position = None

        def take_wave(self):
            detected, self.pending = self.active and self.pending, False
            self.wave_position = (320 - 320 / math.sqrt(3), 240) if detected and visual else None
            return detected

        def close(self):
            self.set_active(False)

        def drain(self):
            pass

    camera = controller._wave_watcher = Camera()
    directions, reads, tracking = [], [], []
    original_state = rig.hardware.set_state

    def set_state(state, **kwargs):
        original_state(state, **kwargs)
        controller.set_state(state, **kwargs)

    def read_doa():
        reads.append(1)
        return DoAReading(2 * math.pi / 3, True)

    def claim_wave():
        claimed = controller.claim_wave()
        rig.hardware.wave_direction = controller.wave_direction
        return claimed

    def look_at_speaker(angle):
        assert not camera.active
        directions.append(angle)
        controller.look_at_speaker(angle)

    rig.hardware.set_state = set_state
    rig.hardware.set_wave_watching = controller.set_wave_watching
    rig.hardware.take_wave = controller.take_wave
    rig.hardware.claim_wave = claim_wave
    rig.hardware.read_doa = read_doa
    rig.hardware.look_at_speaker = look_at_speaker
    rig.hardware.motion_enabled = True
    rig.hardware.play_motion = controller.play_motion
    rig.hardware.stop_motion = controller.stop_motion
    expected_yaw = 30 if visual else -30
    expected_body = 10 if visual else -10
    follower.yaw, follower.pitch = -5, -3
    ticks = []
    write = robot.set_target

    def set_target(head, antennas, body_yaw):
        began_after = ticks[-1].written if ticks else -math.inf
        write(head=head, antennas=antennas, body_yaw=body_yaw)
        ticks.append(Tick(began_after, time.monotonic(), controller._speaker_turn is not None, follower.active,
                          controller._generated, controller._pose_parameters(head)["yaw"],
                          math.degrees(body_yaw)))

    robot.set_target = set_target

    async def first_tick(predicate, since=0):
        """Wait for the motion thread to record a tick from ``since`` on that matches; return its index.

        Checks read these records rather than the live controller, which may be part way through a tick.
        """
        def match():
            return next((index for index in range(since, len(ticks)) if predicate(ticks[index])), None)

        await rig.wait_until_or_timeout(lambda: match() is not None)
        index = match()
        assert index is not None, "no motion tick matched"
        return index

    async def muse(session, request):
        since = len(ticks)
        controller.play_motion(generated(3, yaw=3, roll=5), 25, during={"thinking", "speaking"})
        timeline = ticks[await first_tick(lambda tick: tick.generated is not None, since)].generated.timeline
        faded_in = ticks[await first_tick(
            lambda tick: tick.began_after >= timeline.started + timeline.fade_in_s, since)]
        tracking.append((faded_in.following, faded_in.yaw))
        await session.answer(request, "Tomorrow will be sunny.")

    class RejectAcousticWake:
        calls = 0

        async def verify(self, pcm, phrase):
            self.calls += 1
            return Verification("no_match", "local")

    verifier = RejectAcousticWake()
    voice_checks = []

    class SpeakerVerifier:
        mode = "strict"
        enforcing = True

        def verify(self, samples, rate):
            voice_checks.append(samples.copy())
            return Decision(1.0, True, True)

    options = {"wake_detector": Wake(), "wake_policy": WakePolicy(verifier)} if acoustic else {}
    conversation = service(rig, muse, Transcripts("What is the weather tomorrow?"),
                           wave_wake=True, wake_timeout_s=30,
                           voice_verifier=SpeakerVerifier() if verify_voice else None, **options)

    async def user():
        await rig.wait_until_or_timeout(lambda: conversation._voice is not None)
        conversation._voice[1]._direction_audio_until = math.inf
        await rig.wait_until_or_timeout(lambda: len(reads) >= 5)
        camera.pending = True
        turn = await first_tick(lambda tick: tick.turning)
        assert not ticks[turn].following
        assert sent_to_muse(rig.recording.events) == []
        handover = ticks[await first_tick(lambda tick: tick.following and not tick.turning, turn)]
        assert handover.yaw == pytest.approx(expected_yaw, abs=.2)
        assert handover.body_yaw == pytest.approx(expected_body, abs=.2)
        assert directions == [pytest.approx(math.pi / 3 if visual else 2 * math.pi / 3, abs=.01)]
        await rig.quiet("listening", "idle")
        follower.visible = True
        await rig.wait_until_or_timeout(
            lambda: abs(controller._pose_parameters(robot.pose)["yaw"] - (expected_yaw - 5)) < .2)
        assert voice_checks == []
        rig.say()
        # run() ends the conversation once listening goes 0.6 s without an event, which a slow microphone can outlast.
        await rig.wait_until_or_timeout(lambda: sent_to_muse(rig.recording.events))

    try:
        events = run(rig, conversation, user, rest=("listening", "idle"))
        assert controller._failure is None
        assert sent_to_muse(events) == ["What is the weather tomorrow?"]
        assert played(events) == ["tts:Yes?", "tts:Tomorrow will be sunny."]
        assert tracking == [(True, pytest.approx(expected_yaw - 2, abs=.2))]
        assert verifier.calls == 0
        assert len(voice_checks) == (1 if verify_voice else 0)
    finally:
        controller.close()
