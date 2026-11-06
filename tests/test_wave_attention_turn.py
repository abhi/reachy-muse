"""Visual admission, wake attention and a reply through the real motion owner."""

import asyncio
import math
from types import SimpleNamespace

import pytest

from reachy_muse.motion import MotionStyle
from reachy_muse.speaker_direction import DoAReading
from reachy_muse.wake_policy import Verification, WakePolicy
from reachy_muse.voice_verify import Decision
from test_device_command_turns import played, rig_with, run, sent_to_muse, service
from test_face_tracking_motion import Follower
from test_hardware import FakeRobot, generated, np, start_controller
from test_pins import Transcripts, Vad, Wake


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

    async def muse(session, request):
        controller.play_motion(generated(3, yaw=3, roll=5), 25, during={"thinking", "speaking"})
        await asyncio.sleep(.5)
        tracking.append((follower.active, controller._pose_parameters(robot.pose)["yaw"]))
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
        await rig.wait_until_or_timeout(lambda: len(reads) >= 5)
        camera.pending = True
        await rig.wait_until_or_timeout(lambda: controller._speaker_turn is not None)
        assert not follower.active
        assert sent_to_muse(rig.recording.events) == []
        await rig.wait_until_or_timeout(lambda: controller._speaker_turn is None and follower.active)
        assert controller._pose_parameters(robot.pose)["yaw"] == pytest.approx(expected_yaw, abs=.2)
        assert math.degrees(robot.body_yaw) == pytest.approx(expected_body, abs=.2)
        assert directions == [pytest.approx(math.pi / 3 if visual else 2 * math.pi / 3, abs=.01)]
        await rig.quiet("listening", "idle")
        follower.visible = True
        await rig.wait_until_or_timeout(
            lambda: abs(controller._pose_parameters(robot.pose)["yaw"] - (expected_yaw - 5)) < .2)
        assert voice_checks == []
        rig.say()

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
