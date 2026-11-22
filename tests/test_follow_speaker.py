"""Reachy faces whoever is talking in an open listening window.

People speak into the real microphone loop at real-time pace, from fake DoA bearings, while the real motion owner
turns Reachy: the owner stands at +30 degrees and a friend at -25, where people1's fake faces are.
"""

import asyncio
import itertools
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_muse.speaker_direction import DoAReading
from test_device_command_turns import never_asked, played, run, service as make_service
from test_person_gaze import DelayedWake, people  # noqa: F401  # people is the fixture shared with people1's tests
from test_pins import CHUNK, VOICE_SAMPLE, WAKE_SAMPLE, Transcripts
from test_realtime_turns import (ScriptedCalls, StreamingTranscripts, TimedWake, realtime_service, reply_events, room,
                                 voiced)

OWNER, FRIEND = 30, -25
REPLY = voiced(.2)


def bearing(yaw):
    """The DoA bearing of someone standing at head yaw ``yaw``, in degrees."""
    return math.radians(90 - yaw)


def facing(controller, robot):
    return round(math.degrees(controller.head_yaw)), round(math.degrees(robot.body_yaw))


def followed(caplog):
    prefix = "Reachy follows speaker "
    return [line.removeprefix(prefix) for line in caplog.messages if line.startswith(prefix)]


async def until(rig, predicate, timeout=5):
    await rig.wait_until_or_timeout(predicate, timeout)
    assert predicate()


async def speak_from(rig, scene, yaw, *, voiced_s=.7, quiet_s=.5, wake=False):
    """Someone at head yaw ``yaw``, or nowhere Reachy can place with None, speaks for ``voiced_s`` at real-time pace,
    then the room is quiet for ``quiet_s``. Returns the head's yaw, in whole degrees, while they spoke."""
    scene.doa = None if yaw is None else bearing(yaw)
    if wake:
        rig.hardware.samples.put(np.full(CHUNK, WAKE_SAMPLE, np.float32))
        await asyncio.sleep(CHUNK / 16000)
    heads = []
    for _ in range(round(voiced_s * 16000 / CHUNK)):
        rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
        await asyncio.sleep(CHUNK / 16000)
        heads.append(round(math.degrees(rig.hardware.controller.head_yaw)))
    await room(rig, quiet_s)
    return heads


def answering():
    """Muse in a realtime call, answering each request with a short reply."""
    async def answer(sdk, emit, calls):
        reply_events(sdk, emit, f"r{len(calls.answering)}", "Sure.", REPLY)
    return ScriptedCalls(answer=answer, every_turn=True)


def answering_after(done):
    """Muse in a realtime call, answering each request with a short reply once ``done`` is set."""
    async def answer(sdk, emit, calls):
        ordinal = len(calls.answering)
        await done.wait()
        reply_events(sdk, emit, f"r{ordinal}", "Sure.", REPLY)
    return ScriptedCalls(answer=answer, every_turn=True)


async def replied(rig, service, count):
    """Until Muse's ``count``-th realtime reply has played out."""
    voice = service._voice[1]
    await until(rig, lambda: sum(map(len, rig.hardware.played)) >= count * len(REPLY) // 2
                and not voice.call.audible)


def muse_answers(rig, path, texts, *, wake_detector=None):
    """Reachy hearing ``texts`` in turn, in a realtime call or in chat, and Muse answering each with a short reply.

    Returns the service, the scripted calls, and ``answered(count)``, which waits until Muse's ``count``-th answer
    has played out."""
    calls = answering()
    wake_detector = wake_detector or TimedWake()
    if path == "realtime":
        service = make_service(rig, never_asked, StreamingTranscripts(*texts), wake_detector=wake_detector,
                               wake_timeout_s=5, muse_path="realtime", voice_link=calls.link, voice_call=calls)
    else:
        async def muse(session, request):
            await session.answer(request, "Sure.")

        service = make_service(rig, muse, Transcripts(*texts), wake_detector=wake_detector, wake_timeout_s=5)

    async def answered(count):
        if path == "realtime":
            await replied(rig, service, count)
        else:
            voice = service._voice[1]
            await until(rig, lambda: played(rig.recording.events).count("tts:Sure.") >= count
                        and not voice._has_output())

    return service, calls, answered


@pytest.mark.parametrize("path", ["realtime", "chat"])
def test_reachy_faces_each_of_two_people_as_they_take_turns(people, caplog, path):
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    service, _, answered = muse_answers(rig, path, ("What's the weather?", "And tomorrow?", "Thanks!"))
    seen = []

    async def user():
        for count, (yaw, wake) in enumerate([(OWNER, True), (FRIEND, False), (OWNER, False)], 1):
            await speak_from(rig, scene, yaw, wake=wake)
            await answered(count)
            seen.append(facing(controller, robot))
            await room(rig, 1.5)

    run(rig, service, user)
    assert seen == [(30, 10), (-25, -5), (30, 10)]
    assert followed(caplog) == ["outcome=turned delta_deg=30", "outcome=turned delta_deg=-55",
                                "outcome=turned delta_deg=55"]


async def sounds(rig, scene, yaw, chunks=(3, 2, 2, 2), apart=9):
    """Short sounds from someone at head yaw ``yaw``, of ``chunks`` voiced 20 ms chunks each with ``apart`` quiet chunks
    between, at real-time pace."""
    scene.doa = bearing(yaw)
    for index, voiced_chunks in enumerate(chunks):
        if index:
            await room(rig, apart * CHUNK / 16000)
        for _ in range(voiced_chunks):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)


@pytest.mark.parametrize("path", ["realtime", "chat"])
def test_a_sound_too_short_to_be_a_request_never_turns_reachy(people, caplog, path):
    """Four short sounds from the friend's side, such as a laugh: 180 ms of voice in all, too little for a request,
    over 0.68 s."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    service, calls, answered = muse_answers(rig, path, ("What's the weather?", "", "And tomorrow?"))
    seen = {}

    async def user():
        await speak_from(rig, scene, OWNER, wake=True)
        await answered(1)
        await room(rig, 1.5)
        await sounds(rig, scene, FRIEND)
        await room(rig, 1.0)
        seen["after"] = facing(controller, robot)

    run(rig, service, user)
    assert seen["after"] == (30, 10)
    if path == "realtime":
        assert len(calls.answering) == 1
    else:
        assert [line for line in caplog.messages if line.startswith("Reachy queued a voice turn")] == [
            "Reachy queued a voice turn; 1 pending"]
    assert followed(caplog) == ["outcome=turned delta_deg=30"]


class LateWake(TimedWake):
    """Fires 0.3 s into the request that follows 0.6 s of wake audio, as verification and alignment delay a real
    detection, and hands that 0.3 s of the request to the recorder as audio after the keyword."""

    def __init__(self):
        self.wake_samples = self.after = 0

    def feed(self, samples):
        self.feed_id += 1
        if np.any(samples == WAKE_SAMPLE):
            self.wake_samples += len(samples)
            return False
        if self.wake_samples < 9600:
            return False
        self.after += len(samples)
        if self.after < 4800:
            return False
        self.last_detection = SimpleNamespace(epoch=self.epoch, feed_id=self.feed_id, post_wake_samples=self.after)
        self.wake_samples = self.after = 0
        return True


def test_the_waker_s_request_heard_during_the_wake_turn_gets_no_second_turn(people, caplog):
    """The owner says "Hey Muse" and the request in one breath, and Reachy weighs the request partway through its
    wake turn toward them."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    calls = answering()
    service = make_service(rig, never_asked, StreamingTranscripts("What's the weather?"), wake_detector=LateWake(),
                           wake_timeout_s=5, muse_path="realtime", voice_link=calls.link, voice_call=calls)

    async def user():
        scene.doa = bearing(OWNER)
        for _ in range(30):
            rig.hardware.samples.put(np.full(CHUNK, WAKE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
        await speak_from(rig, scene, OWNER, voiced_s=1.2)
        await replied(rig, service, 1)

    run(rig, service, user)
    yaws = [(at, controller._pose_parameters(head)["yaw"]) for at, head, _, _ in robot.frames]
    crossing = {mark: next(at for at, yaw in yaws if yaw >= mark) for mark in (3, 27)}
    assert [line for line in caplog.messages if line.startswith("Wake direction decision=ok")] != []
    # The eased 0.65 s wake turn passes 10% (3 degrees) and 90% (27 degrees) of its way 0.3 s apart, once.
    assert round(crossing[27] - crossing[3], 1) == .3
    assert [line for line in caplog.messages if line.startswith("Speaker gaze")] == [
        "Speaker gaze yaw=30.0 body_yaw=10.0 decision=turn"]
    assert followed(caplog) == ["outcome=same delta_deg=0"]


def test_the_friend_answering_a_greeting_keeps_reachy_on_them_until_the_owner_speaks(people, caplog):
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    calls = answering()
    service = make_service(rig, never_asked, StreamingTranscripts("Say hi to my friend.", "Hi Reachy!",
                                                                  "What time is it?"),
                           wake_detector=DelayedWake(), wake_timeout_s=5, muse_path="realtime",
                           voice_link=calls.link, voice_call=calls)
    seen = {}

    async def user():
        scene.doa = bearing(OWNER)
        for _ in range(30):
            rig.hardware.samples.put(np.full(CHUNK, WAKE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
        await until(rig, lambda: facing(controller, robot) == (30, 10))
        await speak_from(rig, scene, OWNER)
        await until(rig, lambda: facing(controller, robot) == (-25, -5))
        await replied(rig, service, 1)
        seen["friend"] = await speak_from(rig, scene, FRIEND)
        await replied(rig, service, 2)
        seen["answered"] = facing(controller, robot), controller.person_gaze_held
        seen["owner"] = await speak_from(rig, scene, OWNER)
        await until(rig, lambda: facing(controller, robot) == (30, 10))

    run(rig, service, user)
    assert set(seen["friend"]) == {-25}
    assert seen["answered"] == ((-25, -5), False)
    assert seen["owner"][0] == -25
    assert [line for line in caplog.messages if line.startswith("Reachy gaze request")] == [
        "Reachy gaze request outcome=turned source=fresh faces=2"]
    assert followed(caplog) == ["outcome=same delta_deg=0", "outcome=same delta_deg=0", "outcome=turned delta_deg=55"]


@pytest.mark.parametrize("heard", ["Sure thing, it will be sunny all day.", "Okay."])
def test_echo_and_unconfirmed_sound_over_muse_s_reply_turn_reachy_toward_no_one(people, caplog, heard):
    """What the microphone hears over Muse's reply runs on past it, from the friend's side, but never confirms:
    Muse's own words heard back, or a single word."""
    rig, controller, robot, scene = people
    rig.hardware.echo_cancelled_input = True
    caplog.set_level("INFO")

    async def answer(sdk, emit, calls):
        reply_events(sdk, emit, "r1", "Sure thing, it will be sunny all day.", voiced(1.5))

    calls = ScriptedCalls(answer=answer)
    service = realtime_service(rig, StreamingTranscripts("What's the weather?", heard), calls, wake_timeout_s=5)
    seen = {}

    async def user():
        await speak_from(rig, scene, OWNER, wake=True)
        await until(rig, lambda: service._voice[1].call.audible)
        await room(rig, .3)
        seen["heads"] = await speak_from(rig, scene, FRIEND, voiced_s=2.5)
        seen["after"] = facing(controller, robot)

    run(rig, service, user)
    assert set(seen["heads"]) == {30}
    assert seen["after"] == (30, 10)
    assert calls.barge_ins == 0
    assert [line for line in caplog.messages if "drop=barge_in" in line] != []
    assert followed(caplog) == ["outcome=turned delta_deg=30"]


@pytest.mark.parametrize("where", ["unknown", "dispersed"])
def test_a_later_speaker_reachy_cannot_place_gets_no_turn(people, caplog, where):
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    scene.dispersed = False
    spread = itertools.cycle([bearing(FRIEND), bearing(60)])
    rig.hardware.read_doa = lambda: (DoAReading(next(spread), True) if scene.dispersed
                                     else None if scene.doa is None else DoAReading(scene.doa, True))
    service = realtime_service(rig, StreamingTranscripts("What's the weather?", "And tomorrow?"), answering(),
                               wake_timeout_s=5)
    seen = {}

    async def user():
        await speak_from(rig, scene, OWNER, wake=True)
        await replied(rig, service, 1)
        await room(rig, 1.5)
        scene.dispersed = where == "dispersed"
        seen["heads"] = await speak_from(rig, scene, None if where == "unknown" else FRIEND, voiced_s=1.5)
        await replied(rig, service, 2)
        seen["after"] = facing(controller, robot)

    run(rig, service, user)
    assert set(seen["heads"]) == {30}
    assert seen["after"] == (30, 10)
    assert followed(caplog) == ["outcome=turned delta_deg=30", "outcome=unknown delta_deg=none"]


def test_after_stop_looking_at_me_reachy_turns_toward_no_one_who_speaks(people, caplog):
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    service = realtime_service(rig, StreamingTranscripts("Stop looking at me.", "Where did you go?"), answering(),
                               wake_timeout_s=5)
    seen = {}

    async def user():
        await speak_from(rig, scene, OWNER, wake=True)
        await until(rig, lambda: "tts:Okay, I'll stop following you." in played(rig.recording.events)
                    and not service._voice[1]._has_output())
        await room(rig, 1.5)
        seen["heads"] = await speak_from(rig, scene, FRIEND, voiced_s=1.5)

    run(rig, service, user)
    assert set(seen["heads"]) == {30}
    assert followed(caplog) == ["outcome=turned delta_deg=30", "outcome=off delta_deg=none"]


def test_quick_remarks_from_two_people_do_not_swing_reachy_back_and_forth(people, caplog, monkeypatch):
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    monkeypatch.setattr("reachy_muse.hardware.FOLLOW_DWELL_S", 4.0)
    done = asyncio.Event()
    service = realtime_service(rig, StreamingTranscripts("Is it raining?", "No.", "Are you sure?"),
                               answering_after(done), wake_timeout_s=5)
    heads = []

    async def user():
        heads.extend(await speak_from(rig, scene, OWNER, wake=True, quiet_s=.3))
        heads.extend(await speak_from(rig, scene, FRIEND, quiet_s=.3))
        heads.extend(await speak_from(rig, scene, OWNER, quiet_s=.3))
        done.set()

    run(rig, service, user)
    assert heads[0] == 0 and set(heads[heads.index(30):]) == {30}
    assert followed(caplog) == ["outcome=turned delta_deg=30", "outcome=too_soon delta_deg=-55",
                                "outcome=same delta_deg=0"]


def test_a_later_speaker_who_talks_on_past_the_dwell_gets_the_turn_once(people, caplog, monkeypatch):
    """The friend starts right after Reachy turned to the owner and talks on; then the owner talks on past the
    dwell too, in the same utterance, while Muse waits to answer."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    monkeypatch.setattr("reachy_muse.realtime_call.WAITING_CUE_S", 30.0)
    done = asyncio.Event()
    service = realtime_service(rig, StreamingTranscripts("Is it raining?", "No, it stopped an hour ago."),
                               answering_after(done), wake_timeout_s=5)
    seen = {}

    async def user():
        await speak_from(rig, scene, OWNER, wake=True, quiet_s=.3)
        seen["friend"] = await speak_from(rig, scene, FRIEND, voiced_s=3.0, quiet_s=0)
        seen["owner"] = await speak_from(rig, scene, OWNER, voiced_s=2.5, quiet_s=.3)
        done.set()

    run(rig, service, user)
    assert set(seen["friend"][25:70]) == {30}
    assert seen["friend"][-1] == -25
    assert set(seen["owner"]) == {-25}
    assert followed(caplog) == ["outcome=turned delta_deg=30", "outcome=turned delta_deg=-55"]


@pytest.mark.parametrize("choosing", [False, True])
def test_reachy_stays_on_someone_muse_asked_it_to_face_while_a_later_speaker_talks(people, caplog, monkeypatch,
                                                                                   choosing):
    """While the friend talks, Muse asks Reachy to face someone other than them, the owner, and Reachy decides
    about the friend while it faces the owner or, with a slow camera pass, while it still chooses whom to face."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    monkeypatch.setattr("reachy_muse.person_gaze._FOLLOW_AFTER_S", 1.0)
    monkeypatch.setattr("reachy_muse.person_gaze._FRESH_PASS_S", 2.0)
    looked = threading.Event()
    if choosing:
        def slow_pass():
            scene.before_capture = None
            looked.wait(2)

        scene.before_capture = slow_pass
    service = realtime_service(rig, StreamingTranscripts("What's the weather?", "Is she there?"), answering(),
                               wake_timeout_s=5)
    seen = {}

    async def user():
        await speak_from(rig, scene, OWNER, wake=True)
        await replied(rig, service, 1)
        await room(rig, 1.5)
        talking = asyncio.create_task(speak_from(rig, scene, FRIEND, voiced_s=2.0))
        await asyncio.sleep(.5)
        looking = asyncio.create_task(asyncio.to_thread(service.run_command, "reachy.look_at", {"target": "other"},
                                                        5000))
        await asyncio.sleep(.9)
        looked.set()
        seen["look_at"] = await looking
        seen["heads"] = await talking
        seen["held"] = controller.person_gaze_held

    run(rig, service, user)
    assert seen["look_at"] == {"ok": True, "payload": {"outcome": "turned"}}
    assert set(seen["heads"]) == {30}
    assert seen["held"] is True
    assert followed(caplog) == ["outcome=turned delta_deg=30", "outcome=held delta_deg=none"]


def test_a_turn_muse_asked_for_stands_while_a_later_speaker_talks(people, caplog, monkeypatch):
    """Muse calls a tool for the owner's request; while the friend talks, it turns Reachy toward the speaker of the
    request it is answering, the owner."""
    from reachy_muse.hardware import SPEAKER_TURN_S

    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    monkeypatch.setattr("reachy_muse.person_gaze._FOLLOW_AFTER_S", 1.0)

    async def answer(sdk, emit, calls):
        ordinal = len(calls.answering)
        if ordinal == 1:
            emit(sdk.ResponseStarted("tool"))
            emit(sdk.ResponseDone("tool", "completed"))
        else:
            reply_events(sdk, emit, f"r{ordinal}", "Sure.", REPLY)

    service = realtime_service(rig, StreamingTranscripts("What's the weather?", "Who are you?"),
                               ScriptedCalls(answer=answer, every_turn=True), wake_timeout_s=5)
    seen = {}

    async def user():
        await speak_from(rig, scene, OWNER, wake=True, quiet_s=1.5)
        talking = asyncio.create_task(speak_from(rig, scene, FRIEND, voiced_s=1.0 + SPEAKER_TURN_S + .5))
        await asyncio.sleep(.5)
        seen["look_at"] = await asyncio.to_thread(service.run_command, "reachy.look_at", {"target": "speaker"}, 5000)
        seen["heads"] = await talking

    run(rig, service, user)
    assert seen["look_at"] == {"ok": True, "payload": {"outcome": "turned"}}
    assert set(seen["heads"]) == {30}
    assert followed(caplog) == ["outcome=turned delta_deg=30", "outcome=busy delta_deg=none"]


def test_a_later_speaker_gets_no_turn_while_muse_moves_reachy_nor_after(people, caplog):
    rig, controller, robot, scene = people
    caplog.set_level("INFO")

    def run_command(name, params, timeout_ms):
        if name == "reachy.move":
            return controller.run_command(name, params, timeout_ms)
        return {"ok": True}

    rig.hardware.run_command = run_command
    service = realtime_service(rig, StreamingTranscripts("What's the weather?", "And tomorrow?"), answering(),
                               wake_timeout_s=5)
    seen = {}

    async def user():
        await speak_from(rig, scene, OWNER, wake=True)
        await replied(rig, service, 1)
        await room(rig, 1.5)
        talking = asyncio.create_task(speak_from(rig, scene, FRIEND, voiced_s=1.2))
        await asyncio.sleep(.2)
        moved = await asyncio.to_thread(service.run_command, "reachy.move",
                                        {"yaw": 0, "duration": .3, "hold": 1.5}, 5000)
        seen["moved"] = moved["ok"]
        seen["heads"] = await talking
        await asyncio.sleep(1.0)
        seen["after"] = facing(controller, robot)

    run(rig, service, user)
    assert seen["moved"] is True
    assert min(seen["heads"]) == 0
    assert seen["after"] == (0, 0)
    assert followed(caplog) == ["outcome=turned delta_deg=30", "outcome=busy delta_deg=none"]
