import asyncio
import dataclasses
import math
import time

import pytest

import reachy_muse.conversation
from musegadget.identity import Identity
from reachy_muse.capabilities import Mode, Motion
from reachy_muse.contract.prompts import STOP_REQUEST
from reachy_muse.local_backends import backends_for
from reachy_muse.motion import MotionStyle
from reachy_muse.motion.moments import MOMENTS
from test_device_command_turns import SpeakerHardware, played
from test_pins import PinSession, Rig, Transcripts, Vad, Wake

np = pytest.importorskip("numpy")

TABLE_INTENTS = {intent for spec in MOMENTS.values() for intent in spec.intents}


class FakeMotion:
    def __init__(self, *, delay_s=0.0, slow=None, available=True):
        self.delay_s = delay_s
        self.slow = slow or {}
        self.available = available
        self.asked = []

    async def generate(self, intent, duration_s):
        self.asked.append(intent)
        await asyncio.sleep(self.slow.get(intent, self.delay_s))
        return Motion(25, ((0.0,) * 9,) * round(duration_s * 25))


class MotionHardware(SpeakerHardware):
    def play_motion(self, frames, fps, *, during, loop=False, start_within_s=1.0):
        self.recording.add({"robot": "motion", "during": sorted(during), "loop": loop, "frames": len(frames)})

    def stop_motion(self):
        self.recording.add({"robot": "stop_motion"})


def service(rig, muse, transcripts, *, motion, **fields):
    rig.session = PinSession(rig.recording, muse)

    def backends(session):
        bundle = backends_for(Mode.ON_ROBOT, session, speech=rig.voice(), progress_speech=rig.voice("progress_tts"),
                              transcriber=transcripts, wake="wake_detector" in fields)
        return dataclasses.replace(bundle, motion=motion)

    async def prepare_chat(session, vm):
        return "robot-chat"
    return reachy_muse.conversation.ReachyService(
        identity=Identity("02:00:00:ab:cd:ef"), executor=rig.hardware, silence_s=.2, speech_gate=Vad(),
        backends=backends, prepare_chat=prepare_chat, owns_chat=True, **fields)


def run(rig, service, user, *, rest=("idle",), settle=0.0):
    async def converse():
        stop = asyncio.Event()
        session = asyncio.create_task(service.serve(rig.session, {"vm_id": "vm-1"}, stop))
        try:
            await rig.quiet()
            await user()
            await rig.quiet(*rest)
            await asyncio.sleep(settle)
        finally:
            stop.set()
            await asyncio.wait_for(session, 5)
            await asyncio.gather(*rig.session.scripts)
    asyncio.run(asyncio.wait_for(converse(), 30))
    return rig.recording.events


def rig_with(monkeypatch, hardware=MotionHardware):
    rig = Rig(monkeypatch)
    rig.hardware = hardware(rig.recording)
    return rig


def story(events):
    told = []
    for event in events:
        if event.get("muse") == "send_chat":
            told.append(f"muse: {event['text']}")
        elif event.get("robot") == "motion":
            told.append(f"motion during {'/'.join(event['during'])}{' looping' if event['loop'] else ''}, "
                        f"{event['frames']} frames")
        elif event.get("robot") == "stop_motion":
            told.append("stop motion")
        elif event.get("robot") == "play":
            told.append(f"play {event['audio']}")
        elif event.get("robot") == "set_state" and event["state"] in ("speaking", "sleeping"):
            told.append(f"{event['state']}{' ' + event['expression'] if event['expression'] else ''}")
    return [line for index, line in enumerate(told) if index == 0 or line != told[index - 1]]


def answer_after(delay_s, reply):
    async def muse(session, request):
        await asyncio.sleep(delay_s)
        await session.answer(request, reply)
    return muse


def test_thinking_motion_starts_at_dispatch_fades_when_the_answer_starts_and_good_news_closes_with_a_celebration(
        monkeypatch):
    rig = rig_with(monkeypatch)
    motion = FakeMotion(delay_s=.05)
    conversation = service(rig, answer_after(1.2, "Good news, it will be sunny. [reachy:happy]"),
                           Transcripts("What's the weather tomorrow?"), motion=motion)

    async def user():
        rig.say()

    events = run(rig, conversation, user, settle=.5)
    assert story(events) == [
        "muse: What's the weather tomorrow?",
        "motion during thinking looping, 100 frames",
        "play tts:Hmm, good question.",
        "stop motion",
        "speaking happy",
        "play tts:Good news, it will be sunny.",
        "motion during idle/listening, 62 frames",
        "stop motion",
    ]
    assert set(motion.asked) <= TABLE_INTENTS
    assert motion.asked[:4] == ["thinking hard", "perking up to say hello", "celebrating good news",
                                "apologetic and sorry"]


def test_thinking_motion_not_ready_at_dispatch_is_asked_for_then_and_still_plays_while_muse_works(monkeypatch):
    rig = rig_with(monkeypatch)
    motion = FakeMotion(available=False)
    conversation = service(rig, answer_after(1.5, "It will rain. [reachy:sad]"), Transcripts("Will it rain?"),
                           motion=motion)

    async def user():
        motion.available = True
        rig.say()

    events = run(rig, conversation, user, settle=.5)
    assert motion.asked[0] == "thinking hard"
    assert story(events) == [
        "muse: Will it rain?",
        "motion during thinking looping, 100 frames",
        "play tts:Hmm, good question.",
        "stop motion",
        "speaking sad",
        "play tts:It will rain.",
        "motion during idle/listening, 62 frames",
        "stop motion",
    ]


def test_a_closing_not_ready_when_speech_ends_is_skipped(monkeypatch):
    rig = rig_with(monkeypatch)
    motion = FakeMotion(slow={"celebrating good news": 5.0})
    conversation = service(rig, answer_after(.5, "You won the raffle! [reachy:happy]"),
                           Transcripts("Did I win the raffle?"), motion=motion)

    async def user():
        await asyncio.sleep(1.0)
        rig.say()

    events = run(rig, conversation, user, settle=1.5)
    assert story(events)[-2:] == ["speaking happy", "play tts:You won the raffle!"]


def test_a_reply_with_no_closing_expression_ends_without_motion(monkeypatch):
    rig = rig_with(monkeypatch)
    conversation = service(rig, answer_after(1.0, "It is three o'clock. [reachy:nod]"), Transcripts("What time is it?"),
                           motion=FakeMotion())

    async def user():
        rig.say()

    events = run(rig, conversation, user, settle=.5)
    assert story(events)[-3:] == ["stop motion", "speaking nod", "play tts:It is three o'clock."]


def test_never_mind_while_muse_works_cancels_the_thinking_motion(monkeypatch):
    rig = rig_with(monkeypatch)
    transcripts = Transcripts("Search for the latest robot news.", "Never mind.")

    async def muse(session, request):
        if request == 1:
            await rig.wait_until_or_timeout(lambda: session.requests == 2, 10)
            await session.answer(request, "Robots learned to fold laundry. [reachy:happy]")
        else:
            await session.answer(request, "Okay. [reachy:nod]")

    conversation = service(rig, muse, transcripts, motion=FakeMotion())

    async def user():
        rig.say()
        await rig.quiet("thinking")
        rig.say()

    events = run(rig, conversation, user, settle=.5)
    assert story(events) == [
        "muse: Search for the latest robot news.",
        "motion during thinking looping, 100 frames",
        "play tts:Hmm, good question.",
        "stop motion",
        f"muse: {STOP_REQUEST}",
        "play tts:Okay, I'll drop that.",
    ]
    assert "tts:Robots learned to fold laundry." not in played(events)


def test_go_to_sleep_while_muse_works_fades_the_thinking_motion_before_the_sleep_pose(monkeypatch):
    rig = rig_with(monkeypatch)
    transcripts = Transcripts("Hey Muse, search for the latest robot news.", "Go to sleep.")

    async def muse(session, request):
        if request == 1:
            await rig.wait_until_or_timeout(lambda: session.requests == 2, 10)
            await session.answer(request, "Robots learned to fold laundry. [reachy:happy]")
        else:
            await session.answer(request, "Okay, I stopped. [reachy:nod]")

    conversation = service(rig, muse, transcripts, motion=FakeMotion(), wake_detector=Wake(), wake_timeout_s=30)

    async def user():
        rig.say(wake=True)
        await rig.quiet("thinking")
        rig.say()

    events = run(rig, conversation, user, rest=("sleeping",))
    assert story(events) == [
        "muse: search for the latest robot news.",
        "motion during thinking looping, 100 frames",
        "play tts:Hmm, good question.",
        "stop motion",
        f"muse: {STOP_REQUEST}",
        "play tts:Good night.",
        "sleeping",
    ]


def test_hey_muse_alone_gets_a_greeting_motion_with_its_yes(monkeypatch):
    rig = rig_with(monkeypatch)
    conversation = service(rig, answer_after(0, "unused"), Transcripts("Hey Muse."), motion=FakeMotion(),
                           wake_detector=Wake(), wake_timeout_s=1)

    async def user():
        rig.say(wake=True)

    events = run(rig, conversation, user, settle=.3)
    assert story(events) == ["motion during listening, 62 frames", "play tts:Yes?", "stop motion"]


def test_waking_from_sleep_greets_with_motion_after_good_morning(monkeypatch):
    rig = rig_with(monkeypatch)
    conversation = service(rig, answer_after(0, "unused"), Transcripts("Hey Muse, go to sleep.", "Wake up."),
                           motion=FakeMotion(), wake_detector=Wake(), wake_timeout_s=30)

    async def user():
        rig.say(wake=True)
        await rig.quiet("sleeping")
        rig.say(wake=True)
        await rig.quiet("listening")

    events = run(rig, conversation, user, rest=("listening",), settle=.3)
    assert story(events) == ["play tts:Good night.", "sleeping", "motion during listening, 62 frames",
                             "play tts:I'm awake.", "stop motion"]


@pytest.mark.parametrize("motion", [None, FakeMotion(available=False)], ids=["no source", "no animate op"])
def test_without_generated_motion_a_turn_matches_today_exactly(monkeypatch, motion):
    def one_turn(source):
        rig = rig_with(monkeypatch)
        conversation = service(rig, answer_after(1.2, "Good news, it will be sunny. [reachy:happy]"),
                               Transcripts("What's the weather tomorrow?"), motion=source)

        async def user():
            rig.say()
        return run(rig, conversation, user)

    with_director = one_turn(motion)
    assert not any(event.get("robot") in ("motion", "stop_motion") for event in with_director)
    assert [event for event in with_director if event.get("robot") != "play"] == \
        [event for event in one_turn(None) if event.get("robot") != "play"]


def test_with_breathing_off_reachy_is_still_after_the_closing_motion_ends(monkeypatch):
    from test_hardware import FakeRobot, pose_factory
    from reachy_muse.hardware import ReachyController

    robot = FakeRobot()
    controller = ReachyController(robot_factory=lambda **kwargs: robot, pose_factory=pose_factory,
                                  pose_interpolator=lambda first, last, t: first * (1 - t) + last * t,
                                  style=MotionStyle(breathing=False))

    class RobotHardware(MotionHardware):
        def set_state(self, state, *, expression=None):
            super().set_state(state, expression=expression)
            controller.set_state(state, expression=expression)

        def gesture(self, name):
            super().gesture(name)
            controller.gesture(name)

        def play_motion(self, frames, fps, **options):
            super().play_motion(frames, fps, **options)
            controller.play_motion([[0, 0, 0, 0, 0, 20, -10, 10, 0]] * len(frames), fps, **options)

        def stop_motion(self):
            super().stop_motion()
            controller.stop_motion()

    rig = rig_with(monkeypatch, RobotHardware)
    conversation = service(rig, answer_after(1.0, "Good news, it will be sunny. [reachy:happy]"),
                           Transcripts("What's the weather tomorrow?"), motion=FakeMotion())
    controller.start()
    try:
        async def user():
            rig.say()

        events = run(rig, conversation, user, settle=5.0)
        ended = time.monotonic()
    finally:
        controller.close()

    assert story(events)[-2:] == ["motion during idle/listening, 62 frames", "stop motion"]
    yaws = [controller._pose_parameters(frame[1])["yaw"] for frame in robot.frames]
    assert max(yaws) == pytest.approx(20, abs=.01)
    resting = [frame for frame in robot.frames if ended - 1.0 <= frame[0] <= ended]
    assert len(resting) > 20
    for column in range(3):
        assert np.ptp([frame[1][column, 3] for frame in resting]) < 1e-6
    assert np.ptp([math.degrees(frame[2][0]) for frame in resting]) < .01
    assert np.ptp([controller._pose_parameters(frame[1])["yaw"] for frame in resting]) < .01
