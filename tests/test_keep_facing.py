"""Reachy keeps facing the person it follows while audio pauses its camera, and looks thoughtful only while Muse
really works.

Conversations run through ReachyService.serve with the real microphone loop and TimedWake, the real motion owner, a
real FaceFollower and the controller's own VisionBudget. Fakes stand in only at the edges: the microphone's audio,
DoA, camera frames and Muse's link. The camera sees a face only within its field of view of the head.
"""

import asyncio
import math
from pathlib import Path
import queue
import re
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_muse.hardware import ReachyController
from reachy_muse.local_face_tracking import FACE_LOSS_SECONDS, FaceCapture, FaceFollower, _Face
from reachy_muse.motion import MotionStyle
from reachy_muse.speaker_direction import DoAReading
from test_device_command_turns import LongVoice, never_asked, played, run, service as make_service
from test_hardware import FakeRobot, pose_factory
from test_person_gaze import PeopleHardware, PlacedWake
from test_photo_capture import Encoder
from test_pins import CHUNK, VOICE_SAMPLE, WAKE_SAMPLE, Rig, Transcripts
from test_realtime_turns import (RATE, ScriptedCalls, StreamingTranscripts, TimedWake, reply_events, voiced,
                                 voiced_ms)

OWNER, FRIEND = 30, -25
HALF_VIEW = math.atan2(640, 900)
REPLY = "It will be sunny all day."
REPLY_S = 3.0


def bearing(yaw):
    """The DoA bearing of someone standing at head yaw ``yaw``, in degrees."""
    return math.radians(90 - yaw)


def facing(controller, robot):
    return round(math.degrees(controller.head_yaw)), round(math.degrees(robot.body_yaw))


def near(controller, yaw, within=1.5):
    return abs(math.degrees(controller.head_yaw) - yaw) < within


def logged(caplog, prefix):
    return [line for line in caplog.messages if line.startswith(prefix)]


def followed(caplog):
    return [line.removeprefix("Reachy follows speaker ") for line in logged(caplog, "Reachy follows speaker ")]


async def until(rig, predicate, timeout=5):
    await rig.wait_until_or_timeout(predicate, timeout)
    assert predicate()


class Faces:
    """The camera's view of ``scene.faces``, each a (yaw in degrees, box left edge, box width): the faces within the
    camera's field of view of the head, where they appear in it."""

    def __init__(self, scene):
        self.scene = scene

    def detect(self, frame):
        seen = []
        for yaw, left, width in self.scene.faces:
            angle = math.radians(yaw)
            x, y, z = frame[:3, :3].T @ [math.cos(angle), math.sin(angle), 0.0]
            if abs(math.atan2(y, x)) <= HALF_VIEW:
                seen.append(_Face((left, .3, width, .3), math.atan2(y, x), math.atan2(-z, math.hypot(x, y)), .95))
        return seen


def controller_for(robot, scene, **options):
    robot.media.camera = SimpleNamespace(K=np.array([[900., 0, 640], [0, 900, 360], [0, 0, 1]]), D=np.zeros(5))
    robot.media.get_frame = lambda: robot.pose.copy()

    def follower(*args, **kwargs):
        return FaceFollower(*args, **kwargs, detector_factory=lambda *unused: Faces(scene))

    return ReachyController(robot_factory=lambda **kwargs: robot, pose_factory=pose_factory,
                            pose_interpolator=lambda a, b, t: a * (1 - t) + b * t, face_follow_model=Path("fake"),
                            face_follower_factory=follower, sleep_on_close=False,
                            style=MotionStyle(breathing=False, expression_clips=False), **options)


class RoomHardware(PeopleHardware):
    """The people rig's hardware with the controller's own vision budget, which the microphone loop marks busy
    while audio needs the CPU. Its microphone, as Reachy's does, delivers 20 ms of audio every 20 ms: the sounds the
    test queued, in order, and the room's silence while there are none."""

    def __init__(self, recording, controller):
        super().__init__(recording, controller)
        self.vision_budget = controller.vision_budget
        self._next = time.monotonic()

    @property
    def face_follow_seconds(self):
        return self.controller.face_follow_seconds

    def read_audio(self):
        wait = self._next - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._next = max(self._next + CHUNK / self.sample_rate, time.monotonic() - .5)
        try:
            return self.samples.get_nowait()
        except queue.Empty:
            return np.zeros(CHUNK, np.float32)


@pytest.fixture
def camera_room(monkeypatch):
    """Builds Reachy in a room: ``scene.faces`` are where people's faces are, and ``scene.doa`` the bearing of whoever
    speaks, or None when Reachy cannot place them."""
    built = []

    def build(**options):
        rig = Rig(monkeypatch)
        robot = FakeRobot()
        scene = SimpleNamespace(faces=[], doa=None)
        controller = controller_for(robot, scene, **options)
        built.append((controller, robot))
        controller.start()
        rig.hardware = RoomHardware(rig.recording, controller)
        rig.hardware.read_doa = lambda: DoAReading(scene.doa, True) if scene.doa is not None else None
        return rig, controller, robot, scene

    yield build
    for controller, robot in built:
        try:
            assert controller._failure is None
            assert len(robot.writers) == 1
        finally:
            controller.close()


def muse_replies(rig, path, *texts, wake_detector=None):
    """Reachy hearing ``texts`` in turn, in a realtime call or in chat, and Muse answering each after half a second's
    work with a reply ``REPLY_S`` long. Returns the service, and ``answered(count)``, which waits until Muse's
    ``count``-th reply has played out."""
    wake_detector = wake_detector or TimedWake()
    if path == "realtime":
        async def answer(sdk, emit, calls):
            ordinal = len(calls.answering)
            emit(sdk.Working(True))
            await asyncio.sleep(.5)
            reply_events(sdk, emit, f"r{ordinal}", REPLY, voiced(REPLY_S))
            emit(sdk.Working(False))

        calls = ScriptedCalls(answer=answer, every_turn=True)
        service = make_service(rig, never_asked, StreamingTranscripts(*texts), wake_detector=wake_detector,
                               wake_timeout_s=5, muse_path="realtime", voice_link=calls.link, voice_call=calls)
        samples = round(REPLY_S * RATE)
    else:
        async def muse(session, request):
            await asyncio.sleep(.5)
            await session.answer(request, REPLY)

        chunks = round(REPLY_S * rig.hardware.output_sample_rate / 80)
        service = make_service(rig, muse, StreamingTranscripts(*texts), wake_detector=wake_detector, wake_timeout_s=5,
                               voice=LongVoice(rig.recording, chunks))
        samples = chunks * 80

    async def answered(count):
        voice = service._voice[1]
        await until(rig, lambda: sum(map(len, rig.hardware.played)) >= count * samples and not voice._has_output(), 15)

    return service, answered


class Watch:
    """The head's yaw in degrees, and whether the camera is paused, every 50 ms until stopped."""

    def __init__(self, controller):
        self.samples = []

        async def sample():
            while True:
                self.samples.append((math.degrees(controller.head_yaw),
                                     not math.isfinite(controller.vision_budget.delay())))
                await asyncio.sleep(.05)

        self._task = asyncio.create_task(sample())

    async def stop(self):
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        return self.samples


async def say_live(rig, *, wake=False, voiced_s=.7, quiet_s=.5, watch=None):
    """The user's voice for ``voiced_s``, then the room's quiet for ``quiet_s``, as the microphone plays them. Returns
    ``watch()`` at each 20 ms of the voice."""
    voice = [np.full(CHUNK, WAKE_SAMPLE, np.float32)] * wake + [
        np.full(CHUNK, VOICE_SAMPLE, np.float32)] * round(voiced_s * 16000 / CHUNK)
    quiet = [np.zeros(CHUNK, np.float32)] * round(quiet_s * 16000 / CHUNK)
    for chunk in voice + quiet:
        rig.hardware.samples.put(chunk)
    seen = []
    while rig.hardware.samples.qsize() > len(quiet):
        if watch is not None:
            seen.append(watch())
        await asyncio.sleep(CHUNK / 16000)
    while not rig.hardware.samples.empty():
        await asyncio.sleep(.01)
    return seen


async def speak_from(rig, scene, yaw, *, voiced_s=.7):
    """Someone at head yaw ``yaw`` speaks, then the room is quiet for half a second. Returns the head's yaw, in whole
    degrees, while they spoke."""
    scene.doa = bearing(yaw)
    return await say_live(rig, voiced_s=voiced_s, watch=lambda: round(math.degrees(rig.hardware.controller.head_yaw)))


class Placed(PlacedWake):
    """PlacedWake, deaf until the test hands it Reachy's DoA history, since this microphone hears the room from the
    start."""

    def feed(self, samples):
        if self.history is None:
            TimedWake.feed(self, samples)
            return False
        return super().feed(samples)


async def wake_from(rig, service, scene, wake, yaw):
    """Someone at head yaw ``yaw`` says the wake phrase until Reachy has placed them."""
    scene.doa = bearing(yaw)
    wake.history = service._voice[1]._wake_direction
    while not wake.heard:
        examined = wake.examined
        rig.hardware.samples.put(np.full(CHUNK, WAKE_SAMPLE, np.float32))
        await rig.wait_until_or_timeout(lambda: wake.examined > examined)


def summary(caplog):
    """The working, paused and held seconds of the call's summary line."""
    line, = logged(caplog, "Realtime call_summary ")
    return tuple(float(value) for value in re.search(
        r" working_s=(\d+\.\d) face_paused_s=(\d+\.\d) face_held_s=(\d+\.\d)$", line).groups())


@pytest.mark.parametrize("path", ["realtime", "chat"])
def test_reachy_keeps_facing_the_owner_while_audio_pauses_its_camera_and_follows_once_it_looks_again(
        camera_room, caplog, path):
    """The owner stands at +15 degrees, near enough to straight ahead that Reachy needs no turn toward them, and
    Reachy cannot place their voice, as on 11-24. Their request and Muse's reply keep the camera paused; Reachy keeps
    facing them all the while, where it used to look straight ahead. Once the conversation is quiet the camera looks
    again, and Reachy follows the owner as they step to +5 degrees."""
    rig, controller, robot, scene = camera_room(face_follow_idle=True)
    caplog.set_level("INFO")
    scene.faces = [(15, .4, .3)]
    service, answered = muse_replies(rig, path, "What's the weather like today?")
    seen = {}

    async def user():
        await until(rig, lambda: near(controller, 15))
        watching = Watch(controller)
        await say_live(rig, wake=True)
        await answered(1)
        seen["conversation"] = await watching.stop()
        seen["body"] = facing(controller, robot)[1]
        await until(rig, lambda: math.isfinite(controller.vision_budget.delay()))
        scene.faces = [(5, .4, .3)]
        await until(rig, lambda: near(controller, 5))
        seen["moved"] = facing(controller, robot)
        if path == "realtime":
            await until(rig, lambda: logged(caplog, "Realtime call_summary "), 10)

    run(rig, service, user)
    yaws = [yaw for yaw, _ in seen["conversation"]]
    assert (min(yaws), max(yaws)) == (pytest.approx(15, abs=1.5), pytest.approx(15, abs=1.5))
    assert sum(paused for _, paused in seen["conversation"]) >= .8 * len(seen["conversation"])
    assert seen["body"] == 0
    assert seen["moved"] in ((4, 0), (5, 0), (6, 0))
    assert logged(caplog, "Speaker gaze") == []
    if path == "realtime":
        working, paused, held = summary(caplog)
        assert working == pytest.approx(.5, abs=.2)
        assert paused >= REPLY_S
        assert held == pytest.approx(paused, abs=.3)


@pytest.mark.parametrize("path", ["realtime", "chat"])
def test_a_face_that_left_while_the_camera_paused_is_dropped_once_it_looks_again(camera_room, caplog, path):
    """The owner walks away while Muse replies: the paused camera cannot see it, so Reachy keeps facing where they
    were. Once it looks again, it drops the face it no longer sees within the usual loss time, counted without the
    pause, and settles back to straight ahead."""
    rig, controller, robot, scene = camera_room(face_follow_idle=True)
    caplog.set_level("INFO")
    scene.faces = [(15, .4, .3)]
    service, answered = muse_replies(rig, path, "What's the weather like today?")
    seen = {}

    async def user():
        await until(rig, lambda: near(controller, 15))
        await say_live(rig, wake=True)
        await until(rig, lambda: rig.hardware.played, 10)
        scene.faces = []
        watching = Watch(controller)
        await answered(1)
        seen["reply"] = await watching.stop()
        await until(rig, lambda: math.isfinite(controller.vision_budget.delay()))
        looked = time.monotonic()
        await until(rig, lambda: controller._face_follower.target() is None)
        seen["dropped_s"] = time.monotonic() - looked
        await until(rig, lambda: near(controller, 0, .5))

    run(rig, service, user)
    yaws = [yaw for yaw, _ in seen["reply"]]
    assert (min(yaws), max(yaws)) == (pytest.approx(15, abs=1.5), pytest.approx(15, abs=1.5))
    assert seen["dropped_s"] < FACE_LOSS_SECONDS + .5
    assert facing(controller, robot) == (0, 0)


@pytest.mark.parametrize("path", ["realtime", "chat"])
def test_a_turn_toward_the_friend_while_the_camera_pauses_is_not_pulled_back_to_the_owner(camera_room, caplog, path):
    """The owner at +30 wakes Reachy, which turns to them and, between turns, follows their face. The friend at -25
    then talks: Reachy turns toward them while their speech and Muse's reply keep the camera paused, and the owner's
    face it was following never pulls it back."""
    rig, controller, robot, scene = camera_room()
    caplog.set_level("INFO")
    scene.faces = [(OWNER, .1, .35), (FRIEND, .7, .3)]
    wake = Placed()
    service, answered = muse_replies(rig, path, "What's the weather?", "And tomorrow?", wake_detector=wake)
    seen = {}

    async def user():
        await wake_from(rig, service, scene, wake, OWNER)
        await until(rig, lambda: facing(controller, robot) == (30, 10))
        await speak_from(rig, scene, OWNER)
        await answered(1)
        await until(rig, lambda: controller._face_follower.target() is not None)
        seen["owner"] = facing(controller, robot)
        seen["friend"] = await speak_from(rig, scene, FRIEND, voiced_s=2.0)
        watching = Watch(controller)
        await answered(2)
        seen["reply"] = await watching.stop()
        await until(rig, lambda: controller._face_follower.target() is not None)
        seen["after"] = facing(controller, robot)

    run(rig, service, user)
    assert seen["owner"] == (30, 10)
    assert seen["friend"][0] == 30 and seen["friend"][-1] == -25
    yaws = [yaw for yaw, _ in seen["reply"]]
    assert (min(yaws), max(yaws)) == (pytest.approx(-25, abs=.5), pytest.approx(-25, abs=.5))
    assert seen["after"] == (-25, -5)
    assert followed(caplog) == ["outcome=same delta_deg=0", "outcome=turned delta_deg=-55"]
    assert logged(caplog, "Speaker gaze") == ["Speaker gaze yaw=30.0 body_yaw=10.0 decision=turn",
                                              "Speaker gaze yaw=-25.0 body_yaw=-5.0 decision=turn"]


# ---- every turn replaces the target the follower held, on the real motion owner


class CaptureClock:
    """Reachy's microphone capture as the vision budget sees it: continuous, with every second covered."""

    def __init__(self, budget):
        self._budget = budget
        self._stop = threading.Event()
        budget.capture_active(True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        last = time.monotonic()
        while not self._stop.wait(.02):
            now = time.monotonic()
            self._budget.observe("capture", now - last, now=now)
            last = now

    def close(self):
        self._stop.set()
        self._thread.join(1)


@pytest.fixture
def owner_followed():
    """Reachy listening, following the owner's face at +15 degrees, with the friend at -25 out of view; then audio
    pauses the camera."""
    robot = FakeRobot()
    scene = SimpleNamespace(faces=[(15, .1, .35), (FRIEND, .7, .3)])
    controller = controller_for(robot, scene, photo_encoder=Encoder())
    controller.start()
    clock = CaptureClock(controller.vision_budget)
    try:
        controller.set_state("listening")
        deadline = time.monotonic() + 5
        while not near(controller, 15) and time.monotonic() < deadline:
            time.sleep(.02)
        assert near(controller, 15)
        controller.vision_budget.audio_busy(True)
        time.sleep(1.5)
        assert controller._face_follower.target() is not None
        assert near(controller, 15)
        yield controller, robot, scene
        assert controller._failure is None
        assert len(robot.writers) == 1
    finally:
        clock.close()
        controller.close()


def heads_for(controller, seconds):
    samples = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        samples.append(math.degrees(controller.head_yaw))
        time.sleep(.025)
    return samples


@pytest.mark.parametrize("turn", ["wake", "follow", "look_at_speaker", "look_at_other"])
def test_a_turn_while_the_camera_pauses_replaces_the_face_it_held(owner_followed, turn):
    """A wake turn, a turn toward a later speaker, and Muse's look_at, toward the speaker or someone else, each turn
    Reachy to the friend while the camera stays paused. The owner's face the follower held never pulls Reachy back,
    and once the camera looks again the follower starts on the friend, where Reachy now looks."""
    controller, robot, scene = owner_followed
    started = {
        "wake": lambda: controller.look_at_speaker(bearing(FRIEND)),
        "follow": lambda: controller.follow_speaker(bearing(FRIEND))[0] == "turned",
        "look_at_speaker": lambda: controller.face_speaker(bearing(FRIEND)),
        "look_at_other": lambda: controller.look_at_person(math.radians(FRIEND), 0),
    }[turn]()
    assert started is True
    time.sleep(1.0)
    assert facing(controller, robot) == (-25, -5)
    paused = heads_for(controller, 1.5)
    assert (min(paused), max(paused)) == (pytest.approx(-25, abs=.1), pytest.approx(-25, abs=.1))
    controller.vision_budget.audio_busy(False)
    deadline = time.monotonic() + 3
    while controller._face_follower.target() is None and time.monotonic() < deadline:
        time.sleep(.02)
    target = controller._face_follower.target()
    assert math.degrees(target.yaw) == pytest.approx(0, abs=1)
    looking = heads_for(controller, 1.0)
    assert (min(looking), max(looking)) == (pytest.approx(-25, abs=1), pytest.approx(-25, abs=1))
    assert facing(controller, robot) == (-25, -5)


@pytest.mark.parametrize("ending", ["sleep", "idle", "stop_looking"])
def test_sleep_idle_and_stop_looking_still_end_the_face_held_through_a_pause(owner_followed, ending):
    """Going to sleep, going idle without idle follow, and the user asking Reachy to stop looking at them each stop
    face following, held face or not: Reachy settles straight ahead though the camera stays paused."""
    controller, robot, scene = owner_followed
    {"sleep": lambda: controller.set_state("sleeping"), "idle": lambda: controller.set_state("idle"),
     "stop_looking": lambda: controller.allow_face_tracking(False)}[ending]()
    time.sleep(2.0)
    assert controller._face_follower.target() is None
    assert facing(controller, robot) == (0, 0)


def test_a_photo_while_the_camera_pauses_is_taken_facing_the_person_followed(owner_followed):
    """A photo neither turns Reachy nor moves the follower's centre: it frames the owner Reachy kept facing, and the
    head stays on them afterwards."""
    controller, robot, scene = owner_followed
    held = math.degrees(controller._face_follower.target().yaw)
    frames = []
    robot.media.get_frame = lambda: frames.append(math.degrees(controller.head_yaw)) or robot.pose.copy()
    photo = controller.capture_photo()
    assert photo.failure is None
    assert frames == [pytest.approx(held, abs=.1)]
    after = heads_for(controller, 1.0)
    assert (min(after), max(after)) == (pytest.approx(held, abs=.1), pytest.approx(held, abs=.1))


def test_a_photo_the_owner_asks_for_is_framed_on_the_face_reachy_keeps_facing(camera_room, monkeypatch):
    """The owner at +15, whom Reachy faces by their face alone, asks in chat for a photo of themselves. Audio pauses
    the camera from their words to the shutter, and a detection takes 0.1 s, about what YuNet takes on the robot.
    Whatever runs before the shutter, Reachy faces the face it holds when the frame is read."""
    detect = Faces.detect

    def slow(self, frame):
        time.sleep(.1)
        return detect(self, frame)

    monkeypatch.setattr(Faces, "detect", slow)
    rig, controller, robot, scene = camera_room(face_follow_idle=True, photo_encoder=Encoder())
    scene.faces = [(15, .4, .3)]
    read, framed = robot.media.get_frame, []

    def frame():
        if controller._photo_hold.is_set():
            framed.append((math.degrees(controller.head_yaw), math.degrees(controller._face_follower.target().yaw)))
        return read()

    robot.media.get_frame = frame
    rig.hardware.capture_photo = controller.capture_photo

    async def muse(session, request):
        await session.answer(request, REPLY)

    service = make_service(rig, muse, Transcripts("Reachy, take a photo of me."), photo_requests=True)

    async def user():
        await until(rig, lambda: near(controller, 15))
        await say_live(rig)
        await until(rig, lambda: f"tts:{REPLY}" in played(rig.recording.events), 10)

    run(rig, service, user, rest=("listening", "idle"))
    [(head, held)] = framed
    assert held == pytest.approx(15, abs=1.5)
    assert head == pytest.approx(held, abs=.5)


def test_a_gaze_cleared_while_the_camera_pauses_drops_the_face_seen_from_it():
    """Reachy turned toward the owner's voice at +15 and follows their face at +25. Arrival watching then clears that
    gaze while audio pauses the camera. The face's target is ten degrees from the cleared gaze, so it is dropped
    rather than held: Reachy settles straight ahead, not ten degrees off it."""
    robot = FakeRobot()
    scene = SimpleNamespace(faces=[(25, .4, .3)])
    controller = controller_for(robot, scene, face_follow_idle=True)
    controller.start()
    clock = CaptureClock(controller.vision_budget)
    try:
        controller.look_at_speaker(bearing(15))
        controller.set_state("listening")
        deadline = time.monotonic() + 5
        while not near(controller, 25) and time.monotonic() < deadline:
            time.sleep(.02)
        assert near(controller, 25)
        controller.vision_budget.audio_busy(True)
        time.sleep(.3)
        controller.set_state("idle")
        controller.start_watching(None)
        time.sleep(.3)
        assert controller._face_follower.target() is None
        controller.stop_watching()
        time.sleep(2.0)
        assert facing(controller, robot) == (0, 0)
    finally:
        clock.close()
        controller.close()
    assert controller._failure is None


# ---- the follower's own clock


class Paused:
    """A vision budget the test pauses; ``asked`` counts the worker's checks while paused."""

    def __init__(self):
        self.paused = False
        self.asked = 0

    def delay(self):
        if self.paused:
            self.asked += 1
            return math.inf
        return 1.0


def follower_with(now, budget, visible, reads):
    class Detect:
        def __init__(self, *args):
            pass

        def detect(self, frame):
            return [_Face((.4, .3, .2, .3), math.radians(10), 0.0, .95)] if visible[0] else []

    def capture(*, background=True):
        reads.append(now[0])
        return FaceCapture(object(), (0.0, 0.0), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), now[0], (0.0, 0.0))

    return FaceFollower(capture, camera_matrix=None, sample_hz=15, detector_factory=Detect, clock=lambda: now[0],
                        budget=budget)


def wait_for(condition, timeout=1):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "the face follower did not reach the expected state"
        time.sleep(.002)


def test_a_paused_camera_keeps_the_face_and_loses_it_only_after_it_looks_without_seeing_it():
    now, budget, visible, reads = [10.0], Paused(), [True], []
    item = follower_with(now, budget, visible, reads)
    item.start()
    try:
        item.set_active(True)
        wait_for(lambda: item.target() is not None)
        followed = item.target()
        budget.paused = True
        wait_for(lambda: budget.asked >= 2)
        now[0] = 70.0
        assert item.target() == followed
        visible[0] = False
        count = len(reads)
        budget.paused = False
        wait_for(lambda: len(reads) > count + 1)
        now[0] = 70.0 + FACE_LOSS_SECONDS - .01
        assert (item.target().yaw, item.target().observed_at) == (followed.yaw, 10.0)
        assert item.last_seen_at == 10.0
        now[0] = 70.0 + FACE_LOSS_SECONDS + .01
        assert item.target() is None
    finally:
        item.close()


def test_a_face_lost_before_the_camera_paused_does_not_come_back():
    now, budget, visible, reads = [10.0], Paused(), [True], []
    item = follower_with(now, budget, visible, reads)
    item.start()
    try:
        item.set_active(True)
        wait_for(lambda: item.target() is not None)
        visible[0] = False
        now[0] = 10.0 + FACE_LOSS_SECONDS
        count = len(reads)
        wait_for(lambda: len(reads) > count)
        assert item.target() is None
        budget.paused = True
        wait_for(lambda: budget.asked >= 2)
        assert item.target() is None
    finally:
        item.close()


def test_a_look_requested_while_the_camera_pauses_keeps_the_face_it_follows():
    """Muse's reachy.people asks for a fresh look while audio pauses the camera: the face it follows is still the
    one seen there, however long the pause."""
    now, budget, visible, reads = [10.0], Paused(), [True], []
    item = follower_with(now, budget, visible, reads)
    item.start()
    try:
        item.set_active(True)
        wait_for(lambda: item.target() is not None)
        budget.paused = True
        wait_for(lambda: budget.asked >= 2)
        now[0] = 40.0
        assert item.refresh() is not None
        assert item.target().observed_at == 40.0
        now[0] = 100.0
        assert item.target().observed_at == 40.0
    finally:
        item.close()


# ---- thinking only while Muse really works, and listening when the user speaks


def realtime(rig, answer, *texts, **fields):
    calls = ScriptedCalls(answer=answer, every_turn=True)
    service = make_service(rig, never_asked, StreamingTranscripts(*texts), wake_detector=TimedWake(),
                           muse_path="realtime", voice_link=calls.link, voice_call=calls,
                           wake_timeout_s=fields.pop("wake_timeout_s", 10), **fields)
    return service, calls


def states_since(rig, start):
    return [event["state"] for event in rig.recording.events[start:] if event.get("robot") == "set_state"]


async def states_while_talking(rig, voiced_s=1.0):
    """The user's voice, then half a second of quiet; returns Reachy's state at each 20 ms of the voice."""
    return await say_live(rig, voiced_s=voiced_s, watch=lambda: rig.states()[-1])


def test_working_left_on_after_muse_s_answers_never_shows_the_thinking_pose_over_the_owner(camera_room, caplog):
    """The robot's SDK leaves Working on after Muse's first answer, as on 11-24. Muse's later events all belong to its
    spoken replies, so nothing shows it at work: Reachy listens after each reply and while the owner talks, and
    thinks only while a request waits for Muse."""
    rig, controller, robot, scene = camera_room()
    caplog.set_level("INFO")

    async def answer(sdk, emit, calls):
        ordinal = len(calls.answering)
        if ordinal == 1:
            emit(sdk.Working(True))
        await asyncio.sleep(.5)
        reply_events(sdk, emit, f"r{ordinal}", REPLY, voiced(1.0))

    service, calls = realtime(rig, answer, "What's the weather?", "And tomorrow?")
    seen = {}

    async def reply_then_rest(count):
        voice = service._voice[1]
        await until(rig, lambda: sum(map(len, rig.hardware.played)) > (count - 1) * RATE, 10)
        replying = len(rig.recording.events)
        await until(rig, lambda: sum(map(len, rig.hardware.played)) >= count * RATE and not voice.call.audible, 10)
        await asyncio.sleep(1.0)
        return states_since(rig, replying)

    async def user():
        await say_live(rig, wake=True)
        seen["after_reply"] = await reply_then_rest(1)
        seen["working"] = service._voice[1].call.working
        seen["while_talking"] = await states_while_talking(rig)
        seen["after_second_reply"] = await reply_then_rest(2)

    run(rig, service, user, rest=("listening", "idle"))
    assert seen["working"] is True
    assert "thinking" not in seen["after_reply"] and seen["after_reply"][-1:] == ["listening"]
    assert set(seen["while_talking"]) == {"listening"}
    assert "thinking" not in seen["after_second_reply"] and seen["after_second_reply"][-1:] == ["listening"]
    assert set(played(rig.recording.events)) == {"muse"}


def test_the_owner_speaking_while_reachy_looks_thoughtful_gets_it_listening(camera_room, caplog):
    """Muse is slow to answer, so the request waits and Reachy looks thoughtful. The owner adds something meanwhile,
    and Reachy listens to them; their admitted speech, not Muse's work, decides the pose."""
    rig, controller, robot, scene = camera_room()
    caplog.set_level("INFO")
    released = asyncio.Event()

    async def answer(sdk, emit, calls):
        ordinal = len(calls.answering)
        if ordinal == 1:
            await released.wait()
            reply_events(sdk, emit, "r1", REPLY, voiced(.5))

    service, calls = realtime(rig, answer, "Book me a table for two.", "For seven o'clock.")
    seen = {}

    async def user():
        await say_live(rig, wake=True)
        voice = service._voice[1]
        await until(rig, lambda: voice.call.pending and rig.states()[-1:] == ["thinking"], 10)
        seen["while_talking"] = await states_while_talking(rig)
        released.set()
        await until(rig, lambda: rig.hardware.played and not voice.call.audible, 10)

    run(rig, service, user, rest=("listening", "idle"))
    assert seen["while_talking"].index("listening") <= 10
    assert seen["while_talking"][-1] == "listening"


def test_speech_the_microphone_does_not_admit_leaves_reachy_thoughtful(camera_room, caplog):
    """The request waits for Muse past the follow-up window, so Reachy needs its wake phrase again. Speech without it
    never reaches the call, so Reachy stays thoughtful through it."""
    rig, controller, robot, scene = camera_room()
    caplog.set_level("INFO")

    async def answer(sdk, emit, calls):
        await asyncio.Event().wait()

    service, calls = realtime(rig, answer, "Book me a table for two.", wake_timeout_s=2)
    seen = {}

    async def user():
        await say_live(rig, wake=True)
        voice = service._voice[1]
        await until(rig, lambda: voice.call.pending and rig.states()[-1:] == ["thinking"], 10)
        await until(rig, lambda: time.monotonic() > voice._wake_deadline + .2)
        sent = len(calls.frames)
        seen["while_talking"] = await states_while_talking(rig)
        seen["reached_muse_ms"] = voiced_ms(calls.frames[sent:])

    run(rig, service, user, rest=("thinking", "listening", "idle"))
    assert set(seen["while_talking"]) == {"thinking"}
    assert seen["reached_muse_ms"] == 0


# ---- the closed window's pose while only Working holds the call open


def yaw_span(samples):
    """The lowest and highest head yaw among ``Watch`` samples, in whole degrees."""
    yaws = [round(yaw) for yaw, _ in samples]
    return min(yaws), max(yaws)


@pytest.mark.parametrize("face_follow_idle", [False, True], ids=["default", "idle follow"])
def test_while_only_working_holds_the_call_reachy_keeps_facing_the_owner_through_muse_s_late_answer(
        camera_room, caplog, face_follow_idle):
    """The owner at +15 degrees, whom Reachy faces by face following alone. Muse answers, the robot's SDK leaves
    Working on, and the owner stays quiet past the follow-up window: Reachy shows the closed window's pose, but the
    call stays open and Muse may still answer in it. With or without idle follow, Reachy keeps following the owner,
    and Muse's late answer, which pauses the camera, plays facing them, as any reply in a call does."""
    rig, controller, robot, scene = camera_room(face_follow_idle=face_follow_idle)
    caplog.set_level("INFO")
    scene.faces = [(15, .4, .3)]
    released = asyncio.Event()

    async def answer(sdk, emit, calls):
        emit(sdk.Working(True))
        reply_events(sdk, emit, "r1", REPLY, voiced(1.0))
        await released.wait()
        reply_events(sdk, emit, "r2", "And warm all week.", voiced(2.0))

    service, calls = realtime(rig, answer, "What's the weather?", wake_timeout_s=3)
    seen = {}

    async def user():
        await say_live(rig, wake=True)
        voice = service._voice[1]
        await until(rig, lambda: rig.hardware.played and not voice.call.audible, 10)
        await until(rig, lambda: near(controller, 15), 3)
        await until(rig, lambda: rig.states()[-1:] == ["idle"], 8)
        watching = Watch(controller)
        await asyncio.sleep(2.0)
        seen["closed"] = (rig.states()[-1], voice.call is not None and not voice.call.done,
                          yaw_span(await watching.stop()))
        released.set()
        await until(rig, lambda: voice.call.audible, 5)
        watching = Watch(controller)
        await until(rig, lambda: sum(map(len, rig.hardware.played)) >= 3 * RATE and not voice.call.audible, 10)
        seen["late_answer"] = yaw_span(await watching.stop())

    run(rig, service, user, rest=("listening", "idle", "thinking"), timeout=40)
    assert seen["closed"][:2] == ("idle", True)
    assert seen["closed"][2] in ((14, 14), (14, 15), (15, 15))
    assert seen["late_answer"] in ((14, 14), (14, 15), (15, 15))


def test_once_the_window_closes_and_hangs_up_the_call_reachy_stops_following_without_idle_follow(camera_room, caplog):
    """Muse answers and its work settles, so the follow-up window closes and hangs up the call. Without idle follow,
    Reachy stops following the owner and settles straight ahead, as when any window closes."""
    rig, controller, robot, scene = camera_room()
    caplog.set_level("INFO")
    scene.faces = [(15, .4, .3)]

    async def answer(sdk, emit, calls):
        emit(sdk.Working(True))
        reply_events(sdk, emit, "r1", REPLY, voiced(1.0))
        emit(sdk.Working(False))

    service, calls = realtime(rig, answer, "What's the weather?", wake_timeout_s=2)
    seen = {}

    async def user():
        await say_live(rig, wake=True)
        voice = service._voice[1]
        await until(rig, lambda: rig.hardware.played and not voice.call.audible, 10)
        await until(rig, lambda: near(controller, 15), 3)
        await until(rig, lambda: voice.call is None, 8)
        await rig.wait_until_or_timeout(lambda: near(controller, 0, .5), 6)
        seen["closed"] = (rig.states()[-1], facing(controller, robot))

    run(rig, service, user, rest=("idle",), timeout=30)
    assert seen["closed"] == ("idle", (0, 0))
