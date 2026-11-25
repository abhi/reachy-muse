"""A spoken photo request points Reachy's camera at what its words ask it to frame before the frame is read.

People speak into Reachy's real microphone loop, in a realtime call or in chat, from fake DoA bearings, while the
real motion owner and face follower turn Reachy toward fake faces that the head pose places, as in people1's tests:
the owner at +30 degrees with the larger face and a friend at -25. Each frame the photo thread reads records where
the head pointed then.
"""

import asyncio
import math
import re
import threading
import time

import numpy as np
import pytest

from reachy_muse.device_command import PhotoSubject
from reachy_muse.local_face_tracking import FaceTarget
from reachy_muse.motion import MotionStyle
from reachy_muse.person_gaze import PersonGaze
from reachy_muse.photo import shutter_cue
from reachy_muse.speaker_direction import DoAReading
from test_device_command_turns import played, run, service as make_service
from test_hardware import FakeRobot, generated, start_controller
from test_people_commands import InvokedSession
from test_person_gaze import people, say_live  # noqa: F401  # people is the fixture shared with people1's tests
from test_photo_capture import Encoder, until as wait_until
from test_photo_turns import PHOTO_MESSAGE, fields, photo_lines, sent
from test_pins import CHUNK, VOICE_SAMPLE, WAKE_SAMPLE, Transcripts, Wake
from test_realtime_turns import ScriptedCalls, StreamingTranscripts, realtime_service, reply_events, room, voiced

ANSWER = "You look great."
OPENING = {"realtime": "Hey Muse, ", "chat": "Reachy, "}
# What the robot's microphones report while their speech flag never fires.
ROBOT_DOA = DoAReading(1.5533430576324463, False)


def bearing(yaw):
    """The DoA bearing of someone standing at head yaw ``yaw``, in degrees."""
    return math.radians(90 - yaw)


def doa_as_on_the_robot(rig, scene):
    """DoA that places speech only while ``scene.doa`` is set, and otherwise reports as the robot does."""
    rig.hardware.read_doa = lambda: ROBOT_DOA if scene.doa is None else DoAReading(scene.doa, True)


class PoseCamera:
    """Records, for each frame the photo thread reads, the head's (yaw, pitch, body yaw) in whole degrees, and
    labels the shutter sound when it plays. With ``late_s`` the photo thread starts that late, as under load."""

    def __init__(self, rig, controller, robot, *, late_s=0.0):
        self.shots = []
        self.read_at = []
        self.encoder = controller._photo_encoder = Encoder()
        self._capturing = None
        read, capture, play = robot.media.get_frame, controller.capture_photo, rig.hardware.play_audio
        cue = shutter_cue(rig.hardware.output_sample_rate)

        def capture_photo(timeout_s):
            time.sleep(late_s)
            self._capturing = threading.get_ident()
            try:
                return capture(timeout_s)
            finally:
                self._capturing = None

        def get_frame():
            if threading.get_ident() == self._capturing:
                pose = controller._pose_parameters(robot.pose)
                self.shots.append((round(pose["yaw"]), round(pose["pitch"]), round(math.degrees(robot.body_yaw))))
                self.read_at.append(time.monotonic())
            return read()

        def play_audio(samples, **motion):
            if len(samples) and any(np.array_equal(cue[start:start + len(samples)], samples)
                                    for start in np.flatnonzero(cue == samples[0])):
                event = {"robot": "play", "audio": "shutter"}
                if rig.recording.events[-1:] != [event]:
                    rig.recording.add(event)
                return
            play(samples, **motion)

        robot.media.get_frame = get_frame
        rig.hardware.capture_photo = capture_photo
        rig.hardware.play_audio = play_audio


async def muse_answers(session, request):
    await session.answer(request, ANSWER)


def photo_service(rig, path, *said, calls=None, muse=muse_answers):
    """Reachy hearing ``said`` in turn, in a realtime call that the wake phrase opens or in chat, with photo
    requests on; Muse answers each chat message with ANSWER. Returns the service and the scripted calls."""
    if path == "realtime":
        calls = calls or ScriptedCalls(reply=None)
        return realtime_service(rig, StreamingTranscripts(*said), calls, muse=muse, photo_requests=True,
                                wake_timeout_s=5), calls
    return make_service(rig, muse, Transcripts(*said), photo_requests=True), None


async def until(rig, predicate, timeout=5):
    await rig.wait_until_or_timeout(predicate, timeout)
    assert predicate()


def answered(rig):
    return f"tts:{ANSWER}" in played(rig.recording.events)


def aimed(caplog):
    """The photo line's aim fields."""
    [line] = photo_lines(caplog)
    return {key: value for key, value in fields(line).items()
            if key in ("subject", "aim", "bearing", "source", "faces", "turns", "off_deg")}


def aim_fields(text):
    return dict(field.split("=") for field in text.split())


def facing(controller, robot):
    return round(math.degrees(controller.head_yaw)), round(math.degrees(robot.body_yaw))


def gaze_lines(caplog):
    return [line for line in caplog.messages if line.startswith("Reachy gaze request")]


def follow_lines(caplog):
    return [line for line in caplog.messages if line.startswith("Reachy follows speaker")]


def recording_holds(rig, controller):
    """Each hold_view_for_photo call, as whether it held the view and whether a turn or move was under way then."""
    holds = []

    def hold_view_for_photo(active):
        holds.append((active, controller.view_changing))
        controller.hold_view_for_photo(active)

    rig.hardware.hold_view_for_photo = hold_view_for_photo
    return holds


def ask(people, caplog, path, words, *, voiced_s=None, late_s=0.0):
    """The user asks for a photo in ``words``, after the wake phrase in a call or naming Reachy in chat, and Muse's
    answer plays. Returns the camera and the events.

    In a call the request lasts 1.5 s by default, so the wake greeting's head wiggle has settled when the camera
    reads; a photo that keeps the view would show it otherwise.
    """
    voiced_s = voiced_s or (1.5 if path == "realtime" else .7)
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    camera = PoseCamera(rig, controller, robot, late_s=late_s)
    service, calls = photo_service(rig, path, OPENING[path] + words)
    still_open = []

    async def user():
        await say_live(rig, wake=path == "realtime", voiced_s=voiced_s)
        await until(rig, lambda: answered(rig), 8)
        if path == "realtime":
            call = service._voice[1].call
            still_open.append(call is not None and not call.done)

    events = run(rig, service, user, rest=("listening", "idle"))
    assert [(event["muse"], event["text"]) for event in sent(events)] == [
        ("send_image", PHOTO_MESSAGE.format(words if path == "realtime" else OPENING[path] + words))]
    assert answered(rig)
    if path == "realtime":
        assert (calls.barge_ins, still_open) == (1, [True])
    return camera, events


LOCKED = "aim=locked bearing=no source=fresh faces=2"
KEPT = "aim=view bearing=no source=none faces=none turns=0 off_deg=none"
# The listening pose looks 3 degrees up; a turn toward someone levels the head.
STRAIGHT_AHEAD = [(0, -3, 0)]
FRAMED = [
    pytest.param("take a photo of me", None, [(30, 0, 10)], f"subject=me {LOCKED} turns=1 off_deg=0", 2, id="me"),
    pytest.param("can you see me?", None, [(30, 0, 10)], f"subject=me {LOCKED} turns=1 off_deg=0", 2,
                 id="see_me"),
    # The words an owner says next after a photo that missed them, as on 11-24.
    pytest.param("take a photo of me again", None, [(30, 0, 10)], f"subject=me {LOCKED} turns=1 off_deg=0", 2,
                 id="me_again"),
    pytest.param("take a photo of me sitting down", None, [(30, 0, 10)], f"subject=me {LOCKED} turns=1 off_deg=0", 2,
                 id="me_sitting_down"),
    pytest.param("take a photo and tell me what I look like", None, [(30, 0, 10)],
                 f"subject=me {LOCKED} turns=1 off_deg=0", 2, id="what_i_look_like"),
    pytest.param("take a photo of me", [(30, .35, 10), (-25, .12)], [(30, -10, 10)],
                 f"subject=me {LOCKED} turns=1 off_deg=0", 2, id="me_standing"),
    pytest.param("what am I holding?", [(30, .35, 10), (-25, .12)], [(30, 0, 10)],
                 f"subject=held {LOCKED} turns=1 off_deg=0", 2, id="held_standing"),
    # The middle of the two, +5, is already within 6 degrees of the camera's axis.
    pytest.param("take a photo of my friend", [(30, .35), (-20, .12)], STRAIGHT_AHEAD,
                 f"subject=other {LOCKED} turns=0 off_deg=5", 1, id="friend_centred"),
    pytest.param("take a photo of my friend", [(30, .35), (-10, .12)], [(10, 0, 0)],
                 f"subject=other {LOCKED} turns=1 off_deg=0", 2, id="friend"),
    pytest.param("take a photo of us", [(30, .35), (10, .12)], [(20, 0, 0)], f"subject=us {LOCKED} turns=1 off_deg=0",
                 2, id="us"),
    pytest.param("what is this?", None, STRAIGHT_AHEAD, f"subject=this {KEPT}", 0, id="this"),
    pytest.param("what do you see?", None, STRAIGHT_AHEAD, f"subject=scene {KEPT}", 0, id="scene"),
    pytest.param("what am I pointing at?", None, STRAIGHT_AHEAD, f"subject=scene {KEPT}", 0, id="pointing"),
    pytest.param("take a photo of my coffee", None, STRAIGHT_AHEAD, f"subject=unparsed {KEPT}", 0, id="unparsed"),
    pytest.param("take a photo of my friend's face", None, [(0, -3, 0)],
                 "subject=unparsed aim=view bearing=no source=none faces=none turns=0 off_deg=none", 0,
                 id="friends_face"),
    pytest.param("take a photo of my daughter's dress", None, [(0, -3, 0)],
                 "subject=unparsed aim=view bearing=no source=none faces=none turns=0 off_deg=none", 0,
                 id="daughters_dress"),
]


@pytest.mark.parametrize("words, faces, shots, aim, passes", FRAMED)
@pytest.mark.parametrize("path", ["realtime", "chat"])
def test_each_subject_frames_what_its_words_ask_for_without_doa(people, caplog, path, words, faces, shots, aim,
                                                                 passes):
    rig, controller, robot, scene = people
    scene.doa = None
    if faces is not None:
        scene.faces = faces
    camera, _ = ask(people, caplog, path, words)
    assert camera.shots == shots
    assert aimed(caplog) == aim_fields(aim)
    assert scene.passes == passes


ABOUT_A_THING = [
    "look at this, does this look okay?",
    "take a photo, does this look ripe?",
    "take a photo, does this look like poison ivy?",
    "take a photo, how does this look?",
    "take a photo, do these look the same?",
    "take a photo, how do these look?",
    "take a photo and tell me if these look ripe",
    "take a photo and tell me whether these look the same",
    "take a photo, how should I look after this plant?",
]


@pytest.mark.parametrize("words", ABOUT_A_THING)
@pytest.mark.parametrize("path", ["realtime", "chat"])
def test_a_question_about_how_a_thing_looks_keeps_the_view(people, caplog, path, words):
    rig, controller, robot, scene = people
    scene.doa = None
    camera, _ = ask(people, caplog, path, words)
    assert (aimed(caplog), camera.shots, scene.passes) == (aim_fields(f"subject=this {KEPT}"), STRAIGHT_AHEAD, 0)


@pytest.mark.parametrize("question", ["does this look good on me", "do these look okay on me"])
@pytest.mark.parametrize("suffix", ["", ", please", ", Reachy"])
@pytest.mark.parametrize("path", ["realtime", "chat"])
def test_a_look_question_ending_on_me_frames_the_speaker(people, caplog, path, question, suffix):
    rig, controller, robot, scene = people
    scene.doa = None
    camera, _ = ask(people, caplog, path, f"take a photo, {question}{suffix}?")
    assert camera.shots == [(30, 0, 10)]
    assert aimed(caplog) == aim_fields(
        "subject=me aim=locked bearing=no source=fresh faces=2 turns=1 off_deg=0")
    assert scene.passes == 2


def test_with_doa_the_speaker_s_own_face_is_framed_not_the_largest(people, caplog):
    """The friend, whose face is the smaller, asks; people2 has already turned Reachy toward them."""
    rig, controller, robot, scene = people
    scene.doa = bearing(-25)
    camera, _ = ask(people, caplog, "realtime", "take a photo of me")
    assert camera.shots == [(-25, -3, -5)]
    assert aimed(caplog) == aim_fields("subject=me aim=locked bearing=yes source=fresh faces=2 turns=0 off_deg=3")
    assert scene.passes == 1


def test_with_doa_a_photo_of_my_friend_frames_the_one_not_speaking(people, caplog):
    rig, controller, robot, scene = people
    scene.doa = bearing(30)
    camera, _ = ask(people, caplog, "realtime", "take a photo of my friend")
    assert camera.shots == [(-25, 0, -5)]
    assert aimed(caplog) == aim_fields("subject=other aim=locked bearing=yes source=fresh faces=2 turns=1 off_deg=0")
    assert scene.passes == 2


def test_a_follow_turn_still_moving_at_the_grant_is_waited_out(people, caplog, monkeypatch):
    """The request is shorter than people2 waits for, so it turns Reachy toward the owner at the request's end."""
    rig, controller, robot, scene = people
    monkeypatch.setattr("reachy_muse.person_gaze._FOLLOW_AFTER_S", 1.0)
    scene.doa = bearing(30)
    holds = recording_holds(rig, controller)
    camera, _ = ask(people, caplog, "realtime", "take a photo of me", voiced_s=.7)
    assert holds == [(True, True), (False, False)]
    assert follow_lines(caplog) == ["Reachy follows speaker outcome=turned delta_deg=30"]
    assert camera.shots == [(30, -3, 10)]
    assert aimed(caplog) == aim_fields("subject=me aim=locked bearing=yes source=fresh faces=2 turns=0 off_deg=3")
    assert scene.passes == 1


def test_a_bearing_outside_the_view_is_turned_to_then_centred_on_the_face(people, caplog):
    """The camera sees 15 degrees to each side, and the owner at +30 speaks from a bearing heard at +18, close
    enough that people2 does not turn."""
    rig, controller, robot, scene = people
    scene.view_deg, scene.faces, scene.doa = 15, [(30, .35)], bearing(18)
    camera, _ = ask(people, caplog, "realtime", "take a photo of me")
    assert follow_lines(caplog) == ["Reachy follows speaker outcome=same delta_deg=18"]
    assert camera.shots == [(30, 0, 10)]
    assert aimed(caplog) == aim_fields(
        "subject=me aim=corrected bearing=yes source=bearing faces=0 turns=2 off_deg=12")
    assert scene.passes == 2


@pytest.mark.parametrize("speaker, shots, aim, passes", [
    (18, [(18, 0, 0)], "subject=this aim=locked bearing=yes source=bearing faces=0 turns=1 off_deg=0", 2),
    (10, STRAIGHT_AHEAD, "subject=this aim=view bearing=yes source=none faces=1 turns=0 off_deg=none", 1),
])
def test_what_is_this_turns_level_only_to_a_speaker_outside_the_view(people, caplog, speaker, shots, aim, passes):
    rig, controller, robot, scene = people
    scene.view_deg, scene.faces, scene.doa = 15, [(speaker, .35, 10)], bearing(speaker)
    camera, _ = ask(people, caplog, "realtime", "what is this?")
    assert camera.shots == shots
    assert aimed(caplog) == aim_fields(aim)
    assert scene.passes == passes


def test_a_face_beyond_the_head_s_reach_is_framed_at_the_limit(people, caplog):
    rig, controller, robot, scene = people
    scene.faces, scene.doa = [(50, .35)], None
    camera, _ = ask(people, caplog, "chat", "take a photo of me")
    assert camera.shots == [(35, 0, 15)]
    assert aimed(caplog) == aim_fields("subject=me aim=limit bearing=no source=fresh faces=1 turns=1 off_deg=15")


def moves_after_the_first_pass(scene, faces):
    def move():
        if scene.passes == 1:
            scene.faces = faces
    scene.before_capture = move


def test_a_subject_who_moves_during_the_aim_gets_one_correction(people, caplog):
    rig, controller, robot, scene = people
    scene.doa = None
    moves_after_the_first_pass(scene, [(18, .35), (-25, .12)])
    camera, _ = ask(people, caplog, "chat", "take a photo of me")
    assert camera.shots == [(18, 0, 0)]
    assert aimed(caplog) == aim_fields("subject=me aim=corrected bearing=no source=fresh faces=2 turns=2 off_deg=12")
    assert scene.passes == 2


def test_never_mind_during_the_correction_cancels_the_aim(people):
    """The owner steps 12 degrees after the first pass, and "never mind" lands as Reachy turns to correct for it."""
    rig, controller, robot, scene = people
    moves_after_the_first_pass(scene, [(18, .35), (-25, .12)])
    turns = []
    look_at_person = rig.hardware.look_at_person

    def turn(yaw, pitch):
        turns.append(round(math.degrees(yaw)))
        return look_at_person(yaw, pitch)

    rig.hardware.look_at_person = turn
    gaze = PersonGaze(rig.hardware, hearing=lambda: None)

    async def photo_of_me():
        async with gaze.photo_aim(PhotoSubject.ME, None, time.monotonic() + 2.5, stop=lambda: len(turns) == 2) as aim:
            return aim

    aim = asyncio.run(asyncio.wait_for(photo_of_me(), 10))
    assert (aim.outcome, aim.turns, turns) == ("cancelled", 2, [30, 18])


def test_a_group_member_the_second_pass_misses_stays_in_the_frame(people, caplog):
    """The friend at +10 is missed by the confirming pass, as small faces often are, so Reachy keeps framing them
    with the owner at +30 rather than turning to the owner alone."""
    rig, controller, robot, scene = people
    scene.faces, scene.doa = [(30, .35), (10, .12)], None
    moves_after_the_first_pass(scene, [(30, .35)])
    camera, _ = ask(people, caplog, "chat", "take a photo of us")
    assert camera.shots == [(20, 0, 0)]
    assert aimed(caplog) == aim_fields("subject=us aim=locked bearing=no source=fresh faces=2 turns=1 off_deg=0")


def test_a_camera_pass_never_steers_the_head_while_the_view_is_held(people, caplog):
    """The owner steps 4 degrees further after the first pass, and the photo thread starts late. The second pass
    finds them within the tolerance, and the face follower would have eased the head toward them meanwhile."""
    rig, controller, robot, scene = people
    scene.doa = None
    moves_after_the_first_pass(scene, [(34, .35), (-25, .12)])
    camera, _ = ask(people, caplog, "chat", "take a photo of me", late_s=.3)
    assert camera.shots == [(30, 0, 10)]
    assert aimed(caplog) == aim_fields("subject=me aim=locked bearing=no source=fresh faces=2 turns=1 off_deg=4")


@pytest.mark.parametrize("words, subject", [("take a photo of me", "me"), ("what do you see?", "scene")])
@pytest.mark.parametrize("pass_s, late_s", [(.12, 0.0), (0.0, .3)], ids=["robot_pass_time", "late_photo_thread"])
def test_the_face_reachy_follows_when_asked_stays_framed(people, caplog, words, subject, pass_s, late_s):
    """In a call the wake phrase opened without a bearing, as on 11-24, Reachy faces the owner at +15 by face
    following alone, within a degree, with no turn to give it a gaze. Their photo needs no turn, and a scene
    question keeps the view, so the frame shows what Reachy faced when they asked. A camera pass takes ``pass_s``,
    as on the robot, or the photo thread starts ``late_s`` late."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    scene.faces, scene.doa, scene.busy = [(15, .35)], None, False
    scene.before_capture = lambda: time.sleep(pass_s)
    camera = PoseCamera(rig, controller, robot, late_s=late_s)
    held_at = []
    hold = controller.hold_view_for_photo
    rig.hardware.hold_view_for_photo = lambda active: (held_at.append(time.monotonic()) if active else None,
                                                       hold(active))
    service = realtime_service(rig, StreamingTranscripts("What time is it?", f"Reachy, {words}"), ScriptedCalls(),
                               muse=muse_answers, photo_requests=True, wake_timeout_s=20)
    seen = {}

    async def user():
        await say_live(rig, wake=True)
        voice = service._voice[1]
        await until(rig, lambda: "muse" in played(rig.recording.events) and not voice.call.audible, 8)
        await until(rig, lambda: abs(math.degrees(controller.head_yaw) - 15) < 1.5)
        await room(rig, .5)
        seen["before"] = round(math.degrees(controller.head_yaw)), controller._speaker_look
        await say_live(rig, voiced_s=1.5)
        await until(rig, lambda: answered(rig), 8)

    run(rig, service, user, rest=("listening", "idle"))
    before, look = seen["before"]
    assert (before in (14, 15), look) == (True, None)
    aiming = {round(controller._pose_parameters(pose)["yaw"]) for at, pose, _, _ in list(robot.frames)
              if held_at[0] <= at <= camera.read_at[0]}
    assert (aiming <= {before}, [(yaw, body) for yaw, _, body in camera.shots]) == (True, [(before, 0)])
    assert {key: aimed(caplog)[key] for key in ("subject", "turns")} == {"subject": subject, "turns": "0"}


def test_a_wake_turn_that_starts_as_the_view_is_held_keeps_facing_the_speaker():
    robot = FakeRobot()
    controller = start_controller(robot, photo_encoder=Encoder(),
                                  style=MotionStyle(breathing=False, expression_clips=False))
    held = threading.Event()

    class Follower:
        failure = None
        active = False

        def set_active(self, active, *, idle=False):
            if self.active and not active and controller._speaker_turn is not None:
                held.wait(1)
            self.active = active

        def hold_center(self, yaw):
            pass

        def target(self):
            return FaceTarget(math.radians(-15), 0.0, time.monotonic()) if self.active else None

        def close(self):
            pass

    controller._face_follower = Follower()
    try:
        controller.set_state("listening")
        assert wait_until(lambda: round(math.degrees(controller.head_yaw)) == -15)
        assert controller.look_at_speaker(math.radians(60))
        assert wait_until(lambda: controller._speaker_turn is not None)
        controller.hold_view_for_photo(True)
        held.set()
        assert wait_until(lambda: controller._speaker_turn is None)
        time.sleep(.5)
        facing = round(math.degrees(controller.head_yaw))
        controller.hold_view_for_photo(False)
    finally:
        controller.close()
    assert facing == 30


def test_disabling_face_tracking_recentres_the_head_while_the_photo_view_is_held(people):
    rig, controller, robot, scene = people
    scene.faces, scene.doa, scene.busy = [(15, .35)], None, False
    controller.set_state("listening")

    async def stop_looking():
        await until(rig, lambda: abs(controller._pose_parameters(robot.pose)["yaw"] - 15) < 1)
        controller.hold_view_for_photo(True)
        try:
            controller.allow_face_tracking(False)
            await rig.wait_until_or_timeout(lambda: abs(controller._pose_parameters(robot.pose)["yaw"]) < 1, 5)
            assert controller._photo_view.is_set()
            assert controller._pose_parameters(robot.pose)["yaw"] == pytest.approx(0, abs=1)
        finally:
            controller.hold_view_for_photo(False)

    asyncio.run(stop_looking())


@pytest.mark.parametrize("words, shots, aim", [
    ("take a photo of me", [(30, 0, 10)], "subject=me aim=locked bearing=no source=memory faces=0 turns=1 off_deg=0"),
    ("what am I holding?", STRAIGHT_AHEAD,
     "subject=held aim=no_faces bearing=no source=none faces=0 turns=0 off_deg=none"),
])
def test_a_remembered_face_is_turned_to_for_me_but_not_for_what_am_i_holding(people, caplog, words, shots, aim):
    """Ten seconds ago Reachy saw the owner at +30; now the camera sees 15 degrees to each side, and no bearing."""
    rig, controller, robot, scene = people
    scene.faces, scene.age, scene.doa = [(30, .35)], 10, None
    assert len(controller.refresh_faces().faces) == 1
    scene.age, scene.view_deg = 0.0, 15
    camera, _ = ask(people, caplog, "chat", words)
    assert camera.shots == shots
    assert aimed(caplog) == aim_fields(aim)


def test_one_person_remembered_twice_is_nobody_else_for_a_photo_of_my_friend(people, caplog):
    """Ten seconds ago two camera passes saw only the owner, at +30; now the camera sees 15 degrees to each side
    and nobody, and Reachy hears no bearing. Two sightings of one person are one face, so Reachy keeps its view
    rather than turning to photograph whoever most likely asks."""
    rig, controller, robot, scene = people
    scene.faces, scene.age, scene.doa = [(30, .35)], 10, None
    for _ in range(2):
        assert len(controller.refresh_faces().faces) == 1
    scene.age, scene.view_deg = 0.0, 15
    camera, _ = ask(people, caplog, "chat", "take a photo of my friend")
    assert camera.shots == STRAIGHT_AHEAD
    assert aimed(caplog) == aim_fields(
        "subject=other aim=no_other bearing=no source=none faces=0 turns=0 off_deg=none")


def seen_while_facing(controller, scene, head, sightings):
    assert controller.look_at_person(math.radians(head), 0.0)
    assert wait_until(lambda: not controller.person_turning and round(math.degrees(controller.head_yaw)) == head)
    for faces, age in sightings:
        scene.faces, scene.age = faces, age
        assert len(controller.refresh_faces().faces) == len(faces)
    controller.release_person_gaze(turn_back=True)
    assert wait_until(lambda: not controller.view_changing and abs(math.degrees(controller.head_yaw)) < 1)


@pytest.mark.parametrize("sightings", [
    [(50, 20), (33, 10)],
    [(50, 25), (43, 20), (36, 15), (33, 10)],
], ids=["jump", "steps"])
def test_an_owner_who_moved_is_nobody_else_for_a_photo_of_my_friend(people, caplog, sightings):
    rig, controller, robot, scene = people
    scene.doa, scene.view_deg = None, 35
    seen_while_facing(controller, scene, 20, [([(yaw, .35)], age) for yaw, age in sightings])
    scene.faces, scene.age = [(33, .35)], 0.0
    camera, _ = ask(people, caplog, "chat", "take a photo of my friend")
    assert camera.shots == [(0, -3, 0)]
    assert aimed(caplog) == aim_fields(
        "subject=other aim=no_other bearing=no source=none faces=1 turns=0 off_deg=none")


@pytest.mark.parametrize("together", [True, False], ids=["one_pass", "separate_passes"])
def test_two_people_seen_before_a_photo_stay_two(people, caplog, together):
    rig, controller, robot, scene = people
    scene.doa, scene.view_deg = None, 35
    if together:
        seen_while_facing(controller, scene, 20, [([(50, .12), (33, .35)], 10)])
    else:
        seen_while_facing(controller, scene, 20, [([(50, .12)], 20)])
        scene.faces, scene.age = [(33, .35)], 10
        assert len(controller.refresh_faces().faces) == 1
    scene.faces, scene.age = [(33, .35)], 0.0
    camera, _ = ask(people, caplog, "chat", "take a photo of my friend")
    assert (camera.shots, aimed(caplog)["aim"]) == ([(35, 0, 15)], "limit")


def test_without_faces_or_doa_the_photo_shows_what_reachy_faces(people, caplog):
    rig, controller, robot, scene = people
    scene.faces, scene.doa = [], None
    camera, _ = ask(people, caplog, "chat", "take a photo of me")
    assert camera.shots == STRAIGHT_AHEAD
    assert aimed(caplog) == aim_fields("subject=me aim=no_faces bearing=no source=none faces=0 turns=0 off_deg=none")


def test_a_subject_whose_face_is_gone_at_the_check_is_lost(people, caplog):
    """The owner steps out of view after the first pass; Reachy still faces where they were."""
    rig, controller, robot, scene = people
    scene.doa = None
    moves_after_the_first_pass(scene, [(-25, .12)])
    camera, _ = ask(people, caplog, "chat", "take a photo of me")
    assert camera.shots == [(30, 0, 10)]
    assert aimed(caplog) == aim_fields("subject=me aim=lost bearing=no source=fresh faces=2 turns=1 off_deg=none")


def test_without_time_for_a_correction_the_photo_is_unlocked(people, caplog, monkeypatch):
    """The owner steps 12 degrees after the first pass, and the aim's deadline leaves no time to turn again."""
    rig, controller, robot, scene = people
    monkeypatch.setattr("reachy_muse.conversation.PHOTO_AIM_S", 1.3)
    scene.doa = None
    moves_after_the_first_pass(scene, [(18, .35), (-25, .12)])
    camera, _ = ask(people, caplog, "chat", "take a photo of me")
    assert camera.shots == [(30, 0, 10)]
    assert aimed(caplog) == aim_fields("subject=me aim=unlocked bearing=no source=fresh faces=2 turns=1 off_deg=12")


def test_a_move_that_keeps_the_head_past_the_aim_s_patience_leaves_the_photo_where_it_points(people, caplog):
    """Muse's move holds the head from just before the grant for longer than the aim retries its turn, 1.05 s, and
    than the photo then waits for a still head, a second."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    scene.doa = None
    rig.hardware.run_command = controller.run_command
    camera = PoseCamera(rig, controller, robot)
    service, _ = photo_service(rig, "chat", "Reachy, take a photo of me.")
    rig.session = InvokedSession(rig.recording, muse_answers)
    moved = {}

    async def user():
        for _ in range(35):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
        moving = asyncio.create_task(rig.session.invoke("reachy.move", {"yaw": -20, "duration": .3, "hold": 2.5}))
        await until(rig, lambda: controller.view_changing)
        rig.say(voiced_s=0)
        moved["result"] = await moving
        await until(rig, lambda: answered(rig), 8)

    run(rig, service, user, rest=("listening", "idle"))
    assert moved["result"]["ok"] is True
    assert camera.shots == [(-20, -3, 0)]
    assert aimed(caplog) == aim_fields("subject=me aim=busy bearing=no source=fresh faces=2 turns=0 off_deg=none")


def test_a_thinking_motion_at_the_grant_fades_before_the_aim_s_turn(people, caplog):
    """A thinking loop starts while Reachy transcribes the request, and still sways the head at the grant."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    scene.doa = None
    camera = PoseCamera(rig, controller, robot)
    heard, proceed = asyncio.Event(), asyncio.Event()

    class Transcribing(Transcripts):
        async def transcribe(self, wav):
            heard.set()
            await proceed.wait()
            return await super().transcribe(wav)

    service = make_service(rig, muse_answers, Transcribing("Reachy, take a photo of me."), photo_requests=True)
    swaying = {}

    async def user():
        await say_live(rig)
        await asyncio.wait_for(heard.wait(), 3)
        controller.play_motion(generated(8, yaw=10), 25, during={"thinking"}, loop=True)
        await until(rig, lambda: controller._generated is not None)
        swaying["at_the_grant"] = True
        proceed.set()
        await until(rig, lambda: answered(rig), 8)
        swaying["after"] = controller._generated is not None

    run(rig, service, user, rest=("listening", "idle"))
    assert swaying == {"at_the_grant": True, "after": False}
    assert camera.shots == [(30, 0, 10)]
    assert aimed(caplog) == aim_fields("subject=me aim=locked bearing=no source=fresh faces=2 turns=1 off_deg=0")


def test_after_say_hi_to_my_friend_a_photo_of_me_frames_the_owner(people, caplog):
    """The owner greets the friend with a bearing, then asks for a photo of themselves without one."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    camera = PoseCamera(rig, controller, robot)

    async def greets(sdk, emit, calls):
        await rig.wait_until_or_timeout(lambda: controller._person_hold, 3)
        reply_events(sdk, emit, "r1", "Hello there!", voiced(.3))

    service, calls = photo_service(rig, "realtime", "Say hi to my friend.", "Reachy, take a photo of me.",
                                   calls=ScriptedCalls(answer=greets))
    seen = {}

    async def user():
        scene.doa = bearing(30)
        await say_live(rig, wake=True)
        await until(rig, lambda: facing(controller, robot) == (-25, -5))
        await until(rig, lambda: "muse" in played(rig.recording.events) and not service._voice[1].call.audible)
        seen["greeted"] = facing(controller, robot)
        scene.doa = None
        await room(rig, .3)
        await say_live(rig, voiced_s=1.5)
        await until(rig, lambda: answered(rig), 8)

    run(rig, service, user, rest=("listening", "idle"))
    assert seen["greeted"] == (-25, -5)
    assert gaze_lines(caplog) == ["Reachy gaze request outcome=turned source=fresh faces=2"]
    assert camera.shots == [(30, 0, 10)]
    assert aimed(caplog) == aim_fields("subject=me aim=locked bearing=no source=fresh faces=2 turns=1 off_deg=0")


def test_muse_s_move_waits_until_the_photo_s_frame_is_read(people, caplog):
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    scene.doa = None
    rig.hardware.run_command = controller.run_command
    holds = recording_holds(rig, controller)
    camera = PoseCamera(rig, controller, robot)
    service, _ = photo_service(rig, "chat", "Reachy, take a photo of me.")
    rig.session = InvokedSession(rig.recording, muse_answers)
    moved = {}

    async def user():
        await say_live(rig)
        await until(rig, lambda: holds[:1] == [(True, False)])
        moved["result"] = await rig.session.invoke("reachy.move", {"yaw": -20, "duration": .5, "hold": 0})
        moved["at"] = time.monotonic()
        await until(rig, lambda: answered(rig), 8)

    run(rig, service, user, rest=("listening", "idle"))
    assert moved["result"]["ok"] is True
    assert camera.shots == [(30, 0, 10)]
    assert camera.read_at[0] < moved["at"]
    assert round(min(controller._pose_parameters(pose)["yaw"] for at, pose, _, _ in robot.frames
                     if at > camera.read_at[0])) == -20


def test_a_move_running_at_the_grant_finishes_before_the_aim_turns(people, caplog):
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    scene.doa = None
    rig.hardware.run_command = controller.run_command
    holds = recording_holds(rig, controller)
    turns = []

    def look_at_person(yaw, pitch):
        turns.append((time.monotonic(), controller.look_at_person(yaw, pitch)))
        return turns[-1][1]

    rig.hardware.look_at_person = look_at_person
    camera = PoseCamera(rig, controller, robot)
    service, _ = photo_service(rig, "chat", "Reachy, take a photo of me.")
    rig.session = InvokedSession(rig.recording, muse_answers)
    moved = {}

    async def user():
        for _ in range(35):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
        moving = asyncio.create_task(rig.session.invoke("reachy.move", {"yaw": -20, "duration": .3, "hold": .6}))
        await until(rig, lambda: controller.view_changing)
        rig.say(voiced_s=0)
        moved["result"] = await moving
        await until(rig, lambda: answered(rig), 8)

    run(rig, service, user, rest=("listening", "idle"))
    assert moved["result"]["ok"] is True
    assert holds[0] == (True, True)
    assert turns[-1][1] is True
    assert camera.shots == [(30, 0, 10)]
    assert aimed(caplog) == aim_fields("subject=me aim=locked bearing=no source=fresh faces=2 turns=1 off_deg=0")


def test_nothing_else_turns_reachy_during_the_aim(people, caplog, monkeypatch):
    """While the second camera pass is held up, the friend asks Reachy to greet them, from a bearing of their own,
    and Muse asks to turn Reachy and who is there."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    monkeypatch.setattr("reachy_muse.person_gaze._FRESH_PASS_S", 5.0)
    monkeypatch.setattr("reachy_muse.conversation.PHOTO_AIM_S", 8.0)
    scene.doa = None
    doa_as_on_the_robot(rig, scene)
    confirming, checked = threading.Event(), threading.Event()

    def hold_up_the_second_pass():
        if scene.passes == 1:
            scene.before_capture = None
            confirming.set()
            checked.wait(5)

    scene.before_capture = hold_up_the_second_pass
    camera = PoseCamera(rig, controller, robot)
    service, _ = photo_service(rig, "chat", "Reachy, take a photo of me.", "Reachy, say hi to my friend.")
    rig.session = InvokedSession(rig.recording, muse_answers)
    asked = {}

    async def user():
        await say_live(rig)
        await until(rig, confirming.is_set)
        scene.doa = bearing(-25)
        await say_live(rig)
        await until(rig, lambda: gaze_lines(caplog) and len(follow_lines(caplog)) == 2)
        asked["look_at"] = (await rig.session.invoke("reachy.look_at", {"target": "speaker"}))["payload"]
        people_result = (await rig.session.invoke("reachy.people"))["payload"]
        asked["people"] = {key: people_result[key] for key in ("speaker", "people_visible")}
        asked["facing"] = facing(controller, robot)
        checked.set()
        await until(rig, lambda: played(rig.recording.events).count(f"tts:{ANSWER}") == 2, 8)

    try:
        run(rig, service, user, rest=("listening", "idle"))
    finally:
        checked.set()
    assert follow_lines(caplog) == ["Reachy follows speaker outcome=unknown delta_deg=none",
                                    "Reachy follows speaker outcome=held delta_deg=none"]
    assert gaze_lines(caplog) == ["Reachy gaze request outcome=busy source=none faces=0"]
    assert asked == {"look_at": {"outcome": "busy"}, "people": {"speaker": "unknown", "people_visible": 2},
                     "facing": (30, 10)}
    assert camera.shots == [(30, 0, 10)]
    assert aimed(caplog) == aim_fields("subject=me aim=locked bearing=no source=fresh faces=2 turns=1 off_deg=0")


def test_a_greeting_whose_camera_pass_began_before_the_aim_leaves_the_aim_alone(people, caplog):
    """The owner at +30 asks Reachy to greet their friend, and its camera pass is still under way when a photo of
    the owner begins its aim, which shares that pass. The greeting is then busy, so only the aim turns Reachy."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    looking, released = threading.Event(), threading.Event()

    def hold_up_the_pass():
        looking.set()
        released.wait(5)

    scene.before_capture = hold_up_the_pass
    holds = recording_holds(rig, controller)
    turns = []
    look_at_person = rig.hardware.look_at_person

    def turn(yaw, pitch):
        turns.append(round(math.degrees(yaw)))
        return look_at_person(yaw, pitch)

    rig.hardware.look_at_person = turn
    gaze = PersonGaze(rig.hardware, hearing=lambda: None)

    async def photo_of_me():
        async with gaze.photo_aim(PhotoSubject.ME, None, time.monotonic() + 2.5) as aim:
            return aim

    async def race():
        greeting = asyncio.create_task(gaze.request("Say hi to my friend.", bearing(30), gaze.begin()))
        await until(rig, looking.is_set)
        aiming = asyncio.create_task(photo_of_me())
        await until(rig, lambda: holds)
        scene.before_capture = None
        released.set()
        await greeting
        return await aiming

    try:
        aim = asyncio.run(asyncio.wait_for(race(), 10))
    finally:
        released.set()
    assert gaze_lines(caplog) == ["Reachy gaze request outcome=busy source=none faces=0"]
    assert (aim.outcome, aim.turns, turns, facing(controller, robot)) == ("locked", 1, [30], (30, 10))


def test_never_mind_during_the_aim_reads_no_photo_frame(people, caplog):
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    scene.doa = None
    looking = threading.Event()

    def slow_pass():
        looking.set()
        time.sleep(.25)

    scene.before_capture = slow_pass
    camera = PoseCamera(rig, controller, robot)
    service, _ = photo_service(rig, "chat", "Reachy, take a photo of me.", "Never mind.")

    async def user():
        await say_live(rig)
        await until(rig, looking.is_set)
        await say_live(rig, voiced_s=.3)
        await until(rig, lambda: photo_lines(caplog), 5)

    events = run(rig, service, user, rest=("listening", "idle"))
    assert camera.shots == []
    assert {key: fields(photo_lines(caplog)[0])[key] for key in ("outcome", "aim", "turns")} == {
        "outcome": "cancelled", "aim": "cancelled", "turns": "1"}
    assert sent(events) == []
    assert played(events) == ["tts:Okay, I'll drop that."]


def test_muse_s_move_waits_out_the_turn_of_an_aim_never_mind_ends(people, caplog):
    """Muse asks for a move once the aim holds the view, and "Never mind" lands while the aim's turn toward the
    owner still moves. The move runs once that turn ends, rather than being refused as a turn toward a person."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    scene.doa = None
    looking = threading.Event()

    def slow_pass():
        looking.set()
        time.sleep(.25)

    scene.before_capture = slow_pass
    holds = recording_holds(rig, controller)
    rig.hardware.run_command = controller.run_command
    service, _ = photo_service(rig, "chat", "Reachy, take a photo of me.", "Never mind.")
    rig.session = InvokedSession(rig.recording, muse_answers)
    moved = {}

    async def move():
        await until(rig, lambda: holds[:1] == [(True, False)])
        moved["result"] = await rig.session.invoke("reachy.move", {"yaw": -20, "duration": .3, "hold": 0})

    async def user():
        moving = asyncio.create_task(move())
        await say_live(rig)
        await until(rig, looking.is_set)
        await say_live(rig, voiced_s=.3)
        await until(rig, lambda: photo_lines(caplog), 5)
        await moving

    run(rig, service, user, rest=("listening", "idle"))
    assert holds == [(True, False), (False, True)]
    assert moved["result"]["ok"] is True, moved["result"]
    assert {key: fields(photo_lines(caplog)[0])[key] for key in ("aim", "turns")} == {"aim": "cancelled", "turns": "1"}
    yaws = [controller._pose_parameters(pose)["yaw"] for _, pose, _, _ in list(robot.frames)]
    faced = next(index for index, yaw in enumerate(yaws) if round(yaw) == 30)
    assert (round(max(yaws[:faced + 1])), round(min(yaws[faced:]))) == (30, -20)
    assert not controller._photo_view.is_set()


def test_a_photo_of_me_said_over_muse_s_reply_is_aimed_after_the_barge_in(people, caplog):
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    rig.hardware.echo_cancelled_input = True
    scene.doa = None
    camera = PoseCamera(rig, controller, robot)

    async def answer(sdk, emit, calls):
        ordinal = len(calls.answering)
        if ordinal == 1:
            reply_events(sdk, emit, "r1", "Tomorrow will be sunny and warm all day long.", voiced(3))
        elif ordinal == 3:
            reply_events(sdk, emit, "r3", "Anytime.", voiced(.3))

    calls = ScriptedCalls(answer=answer, every_turn=True)
    service, _ = photo_service(rig, "realtime", "What's the weather tomorrow?", "Reachy, take a photo of me.",
                               "Thanks!", calls=calls)
    seen = {}

    async def user():
        await say_live(rig, wake=True)
        voice = service._voice[1]
        await until(rig, lambda: voice.call.audible)
        await room(rig, .3)
        await say_live(rig, voiced_s=1.5)
        await until(rig, lambda: answered(rig), 8)
        seen["call"] = voice.call
        await until(rig, lambda: not voice._has_output())
        await room(rig, .5)
        await say_live(rig)
        await until(rig, lambda: played(rig.recording.events)[-1:] == ["muse"], 8)
        seen["still"] = voice.call is seen["call"] and not seen["call"].done

    events = run(rig, service, user, rest=("listening", "idle"))
    assert calls.barge_ins == 2
    assert seen["still"] is True
    assert camera.shots == [(30, 0, 10)]
    assert [(event["muse"], event["text"]) for event in sent(events)] == [
        ("send_image", PHOTO_MESSAGE.format("Reachy, take a photo of me."))]
    assert aimed(caplog) == aim_fields("subject=me aim=locked bearing=no source=fresh faces=2 turns=1 off_deg=0")


def test_a_camera_pass_that_never_finishes_bounds_the_aim(people, caplog):
    rig, controller, robot, scene = people
    scene.doa = None
    released = threading.Event()
    scene.before_capture = lambda: released.wait(5)
    try:
        camera, _ = ask(people, caplog, "chat", "take a photo of me")
    finally:
        released.set()
    assert camera.shots == STRAIGHT_AHEAD
    assert aimed(caplog) == aim_fields("subject=me aim=no_faces bearing=no source=none faces=none turns=0 off_deg=none")
    assert int(fields(photo_lines(caplog)[0])["aim_ms"]) < 2600


def test_one_photo_frame_per_grant(people, caplog):
    """Even with a correction, the aim's most turns and passes."""
    rig, controller, robot, scene = people
    scene.doa = None
    moves_after_the_first_pass(scene, [(18, .35), (-25, .12)])
    camera, events = ask(people, caplog, "chat", "take a photo of me")
    assert (len(camera.shots), len(camera.encoder.calls), len(sent(events))) == (1, 1, 1)
    assert scene.passes <= 3
    assert fields(photo_lines(caplog)[0])["turns"] == "2"


def test_the_call_s_closing_motion_waits_for_the_photo(people, caplog):
    """Muse's reply, which earns a closing, has played out, and Muse works on until the aim has begun, so Reachy
    weighs the reply's closing during the aim. A reply cut off by the photo request earns no closing at all."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    scene.doa = None
    commands = []

    def run_command(name, params, timeout_ms):
        commands.append(params.get("name", name))
        return controller.run_command(name, params, timeout_ms)

    rig.hardware.run_command = run_command
    holds = recording_holds(rig, controller)
    camera = PoseCamera(rig, controller, robot)

    async def answer(sdk, emit, calls):
        emit(sdk.Working(True))
        reply_events(sdk, emit, "r1", "Hello Sarah, nice to see you!", voiced(.6))
        await rig.wait_until_or_timeout(lambda: holds, 10)
        emit(sdk.Working(False))

    service, _ = photo_service(rig, "realtime", "Who is here?", "Reachy, take a photo of me.",
                               calls=ScriptedCalls(answer=answer))

    async def user():
        await say_live(rig, wake=True)
        voice = service._voice[1]
        await until(rig, lambda: "muse" in played(rig.recording.events) and not voice.call.audible)
        await room(rig, .3)
        await say_live(rig, voiced_s=1.5)
        await until(rig, lambda: answered(rig), 8)
        await until(rig, lambda: not voice._call_tasks)

    run(rig, service, user, rest=("listening", "idle"))
    assert camera.shots == [(30, 0, 10)]
    assert commands == []
    assert [line for line in caplog.messages if line.startswith("Reachy closed Muse's realtime reply")] == []


@pytest.mark.parametrize("people", [{"faces": False}], indirect=True)
@pytest.mark.parametrize("speaker, shots, aim", [
    (30, [(30, 0, 10)], "subject=me aim=bearing bearing=yes source=bearing faces=none turns=1 off_deg=none"),
    (None, STRAIGHT_AHEAD, "subject=me aim=no_faces bearing=no source=none faces=none turns=0 off_deg=none"),
])
def test_without_a_face_worker_a_bearing_turns_reachy_level_to_the_speaker(people, caplog, speaker, shots, aim):
    rig, controller, robot, scene = people
    scene.doa = None if speaker is None else bearing(speaker)
    camera, _ = ask(people, caplog, "chat", "take a photo of me")
    assert camera.shots == shots
    assert aimed(caplog) == aim_fields(aim)


def test_the_photo_line_has_only_content_free_aim_fields(people, caplog):
    rig, controller, robot, scene = people
    scene.doa = bearing(30)
    ask(people, caplog, "realtime", "take a photo of my friend Sarah")
    [line] = photo_lines(caplog)
    assert set(fields(line)) == {"outcome", "path", "subject", "aim", "bearing", "source", "faces", "turns", "off_deg",
                                 "aim_ms", "width", "height", "quality", "bytes", "capture_ms", "encode_ms", "upload_ms"}
    assert all(re.fullmatch(r"[a-z_]+|\d+|none", value) for value in fields(line).values())
    assert aimed(caplog) == aim_fields("subject=other aim=locked bearing=yes source=fresh faces=2 turns=1 off_deg=0")
    logged = "\n".join(record.getMessage() for record in caplog.records)
    for word in ("Sarah", "friend"):
        assert word not in logged


def unaddressed_photo_phrases_in_an_open_window(people, caplog):
    """Later in the window "Hey Muse" opened, photo phrases nobody addressed to Reachy, such as a television's."""
    rig = people[0]
    said = ("What do you see?", "Take a picture of us.", "What am I holding?")
    service = make_service(rig, muse_answers, Transcripts("Hey Muse, what's the weather tomorrow?", *said),
                           wake_detector=Wake(), wake_timeout_s=30, photo_requests=True)

    async def user():
        rig.say(wake=True)
        for count in range(1, len(said) + 1):
            await until(rig, lambda: played(rig.recording.events).count(f"tts:{ANSWER}") == count, 8)
            await rig.quiet("listening")
            rig.say()
        await until(rig, lambda: played(rig.recording.events).count(f"tts:{ANSWER}") == len(said) + 1, 8)

    events = run(rig, service, user, rest=("listening",))
    assert [(event["muse"], event["text"]) for event in sent(events)] == [
        ("send_chat", "what's the weather tomorrow?"), *(("send_chat", text) for text in said)]


def a_photo_request_while_muse_works_on_another(people, caplog):
    rig = people[0]

    async def muse(session, request):
        await rig.wait_until_or_timeout(lambda: "tts:One at a time, please." in played(rig.recording.events), 10)
        await session.answer(request, ANSWER)

    service = make_service(rig, muse, Transcripts("What's the weather tomorrow?", "Reachy, take a photo of me."),
                           photo_requests=True)

    async def user():
        rig.say()
        await rig.quiet("thinking")
        rig.say()
        await until(rig, lambda: answered(rig), 8)

    events = run(rig, service, user, rest=("listening", "idle"))
    assert [event["muse"] for event in sent(events)] == ["send_chat"]
    assert "tts:One at a time, please." in played(events)


def a_call_s_lagging_partial(people, caplog):
    """The companion's partial at Reachy's endpoint lacks the request's end, which its final words bring."""
    from dataclasses import replace
    from reachy_muse.companion_client import CompanionHearing, CompanionLink, LinkState, _Connection
    from reachy_muse.protocol import HearEndpoint, HearPartial, Op, encode

    rig = people[0]
    link = CompanionLink("ws://localhost:9", token="test", out_rate=16000, insecure=True)
    link._connection = connection = _Connection(None, (Op.HEAR,))
    link._set(LinkState.UP)
    opened = link.open_hear

    def open_hear(outcomes=None):
        stream = opened(outcomes)
        link._receive(connection, encode(HearPartial(stream.id, 0, "take a photo of me")))
        finish = stream.finish

        def finish_with_the_final():
            finish()
            asyncio.get_running_loop().call_later(.3, lambda: link._receive(
                connection, encode(HearEndpoint(stream.id, "take a photo of me in five minutes", True))))

        stream.finish = finish_with_the_final
        return stream

    link.open_hear = open_hear
    service = realtime_service(rig, None, ScriptedCalls(reply=None), muse=muse_answers, photo_requests=True,
                               wake_timeout_s=5)
    backends = service.backends
    service.backends = lambda session: replace(backends(session), hearing=CompanionHearing(link))

    async def user():
        rig.say(wake=True)
        await until(rig, lambda: any(line.startswith("Realtime command ") for line in caplog.messages), 5)
        await asyncio.sleep(1)

    events = run(rig, service, user, rest=("thinking", "listening", "idle"))
    assert sent(events) == []


def muse_s_commands_and_a_briefing(people, caplog):
    from types import SimpleNamespace
    from reachy_muse.arrival import PersonBox
    from test_morning_turns import Camera, briefing, morning

    rig, controller = people[0], people[1]
    controller._presence_detector = SimpleNamespace(detect=lambda image: [PersonBox(0, 0, .2, .3, .9)])
    for name in ("presence", "presence_observations", "run_command"):
        setattr(rig.hardware, name, getattr(controller, name))
    watched = Camera()
    service = make_service(rig, briefing, Transcripts(), morning=morning(watched), photo_requests=True)
    results = {}

    async def user():
        await until(rig, lambda: sent(rig.recording.events), 5)
        watched.arrive()
        await until(rig, lambda: watched.calls == ["start", "stop"], 10)
        await rig.quiet()
        for name, params in (("reachy.photo", {"text": "What do you see?"}), ("reachy.presence", {}),
                             ("reachy.notify", {"id": "n1", "text": "Your package arrived.", "mode": "now",
                                                "kind": "notification"})):
            results[name] = await asyncio.to_thread(service.run_command, name, params, 5000)
        await rig.quiet()

    events = run(rig, service, user)
    assert {name: result.get("error") or result["payload"].get("status") or result["payload"]["present"]
            for name, result in results.items()} == {
        "reachy.photo": "unsupported command: reachy.photo", "reachy.presence": True, "reachy.notify": "delivered"}
    assert played(events)[-1] == "tts:Your package arrived."


@pytest.mark.usefixtures("paired_inbox")
@pytest.mark.parametrize("asked", [unaddressed_photo_phrases_in_an_open_window,
                                   a_photo_request_while_muse_works_on_another, a_call_s_lagging_partial,
                                   muse_s_commands_and_a_briefing], ids=lambda asked: asked.__name__)
def test_requests_that_take_no_photo_never_aim(people, caplog, asked):
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    holds = recording_holds(rig, controller)
    camera = PoseCamera(rig, controller, robot)
    asked(people, caplog)
    assert (holds, camera.shots, scene.passes) == ([], [], 0)
    assert [fields(line)["outcome"] for line in photo_lines(caplog)] == (
        ["busy"] if asked is a_photo_request_while_muse_works_on_another else [])
