"""Muse's people commands through ReachyService.serve.

Each request reaches Reachy through its microphone loop, with fake DoA and camera faces, while the fake
companion hears it and labels the voice. Muse calls the commands with link.invoke, through the SDK's own
handler, and learns only what the owner allowed.
"""

import asyncio
from dataclasses import replace
import itertools
import math
import re
import threading
import time

import numpy as np
import pytest
from musegadget.link_client import MAX_CONCURRENT_INVOKES, LinkSession

from fake_companion import END_OF_SPEECH_FRAMES, TOKEN, FakeCompanion
from reachy_muse import companion_client
from reachy_muse.companion_client import CompanionHearing, CompanionLink
from reachy_muse.failover import CompanionRoute, FailoverHearing
from reachy_muse.hardware import SPEAKER_TURN_S
from reachy_muse.protocol import HearClose, HearOpen, Op, Sender, SpeakerLabel, decode
from test_device_command_turns import never_asked, played, service as chat_service
from test_person_gaze import people, say_live  # noqa: F401  # people is the fixture shared with people1's tests
from test_pins import CHUNK, VOICE_SAMPLE, PinSession
from test_realtime_turns import (ScriptedCalls, StreamingTranscripts, TimedWake, realtime_service, reply_events, room,
                                 voiced)

REQUEST = "Who is here with me, Sarah?"
REPLY = "Hello Sarah, nice to see you!"
REPLY_AUDIO = voiced(.2)
OUTCOMES = ("turned", "no_other", "no_faces", "no_speaker_bearing", "stale", "busy")
ALLOWED_LOG_LINES = (
    r"Reachy people speaker=(owner|someone_else|unknown) visible=(none|\d+)",
    r"Reachy people outcome=(invalid_parameters|timeout)",
    rf"Reachy look_at target=(speaker|other) outcome=({'|'.join(OUTCOMES)}|timeout)",
    r"Reachy look_at outcome=invalid_parameters",
    r"invoke reachy\.(people|look_at)",
    r"reachy\.(people|look_at) (ok|failed) in \d+ ms",
)


class InvokedSession(PinSession):
    """Muse's end of the registered link: ``invoke`` hands a link.invoke to the SDK's own handler."""

    def __init__(self, recording, muse=never_asked):
        super().__init__(recording, muse)
        self._register_id = "register"
        self._tasks = set()
        self._invokes = asyncio.Semaphore(MAX_CONCURRENT_INVOKES)
        self._waiting = {}
        self._ids = itertools.count(1)
        self.results = []

    async def send(self, message):
        self._waiting.pop(message["id"]).set_result(message)

    async def invoke(self, command, params=None):
        id = f"invoke-{next(self._ids)}"
        result = self._waiting[id] = asyncio.get_running_loop().create_future()
        LinkSession._handle(self, {"method": "link.invoke", "id": id, "command": command,
                                   "params": {} if params is None else params})
        self.results.append(await asyncio.wait_for(result, 8))
        return self.results[-1]


class Muse:
    """Muse's side of one realtime call: before replying to its ``n``-th request it runs ``then(n)``,
    such as its tool calls, and it answers every request."""

    def __init__(self, then):
        async def answer(sdk, emit, calls):
            ordinal = len(calls.answering)
            await then(ordinal)
            reply_events(sdk, emit, f"r{ordinal}", REPLY, REPLY_AUDIO)

        self.calls = ScriptedCalls(answer=answer, every_turn=True)


def companion(speaker=SpeakerLabel.OWNER, request=REQUEST, *, ops=(Op.HEAR, Op.SPEAK, Op.NARRATE, Op.SPEAKER),
              **knobs):
    return FakeCompanion(ops=ops, partials=(request,), speaker=speaker, **knobs)


def converse(rig, calls, fake, user, *, silence_s=None, wake_timeout_s=2, muse=never_asked):
    """Reachy in a realtime call with ``calls``, or in chat with ``muse`` when ``calls`` is None, that hears
    through ``fake``, from rest until ``user(voice)`` is done and it rests again."""
    if calls is None:
        service = chat_service(rig, muse, StreamingTranscripts(), wake_detector=TimedWake(),
                               wake_timeout_s=wake_timeout_s)
    else:
        service = realtime_service(rig, StreamingTranscripts(), calls, wake_timeout_s=wake_timeout_s)
    if silence_s is not None:
        service.silence_s = silence_s
    rig.session = InvokedSession(rig.recording, muse)

    async def scenario():
        async with fake:
            link = CompanionLink(fake.url.removesuffix("/v1"), token=TOKEN, out_rate=rig.hardware.output_sample_rate,
                                 insecure=True, backoff_s=(5, 5))
            backends = service.backends

            def hearing_on_the_companion(session):
                bundle = backends(session)
                return replace(bundle, hearing=FailoverHearing(CompanionHearing(link), bundle.hearing,
                                                               CompanionRoute(link).hearing))

            service.backends = hearing_on_the_companion
            linking = asyncio.create_task(link.run())
            stop = asyncio.Event()
            serving = asyncio.create_task(service.serve(rig.session, {"vm_id": "vm-1"}, stop))
            try:
                await asyncio.wait_for(link.settled(), 2)
                assert link.can(Op.SPEAKER) == (Op.SPEAKER in fake.ops)
                await rig.quiet()
                await user(service._voice[1])
                await rig.quiet()
            finally:
                stop.set()
                await asyncio.wait_for(serving, 5)
                linking.cancel()
                await asyncio.gather(linking, return_exceptions=True)

    asyncio.run(asyncio.wait_for(scenario(), 30))


def logged(caplog, prefix):
    return [line for line in caplog.messages if line.startswith(prefix)]


async def until(rig, predicate, timeout=5):
    await rig.wait_until_or_timeout(predicate, timeout)
    assert predicate()


async def heard(rig, caplog, count=1):
    """Until Reachy has taken its ``count``-th request in the call."""
    await until(rig, lambda: len(logged(caplog, "Realtime command ")) >= count)


async def labeled(rig, caplog, label, count=1):
    """Until the companion's ``count``-th label is on its request."""
    await until(rig, lambda: len(logged(caplog, "Reachy speaker=")) >= count)
    assert logged(caplog, "Reachy speaker=")[count - 1] == f"Reachy speaker={label.value}"


async def replied(rig, voice, count=1):
    """Until Muse's ``count``-th reply has played out."""
    await until(rig, lambda: sum(map(len, rig.hardware.played)) == count * len(REPLY_AUDIO) // 2
                and (voice.call is None or not voice.call.audible))


def label_arrivals(monkeypatch):
    """When each speaker label reaches Reachy."""
    arrived = []
    original = companion_client._SpeakerSlot.arrived

    def arrived_at(slot, label):
        arrived.append(time.monotonic())
        original(slot, label)

    monkeypatch.setattr(companion_client._SpeakerSlot, "arrived", arrived_at)
    return arrived


def closed_by_reachy(fake):
    """Of the hear streams Reachy opened, in order, whether Reachy closed each itself."""
    messages = [decode(frame, sender=Sender.ROBOT) for frame in fake.received if isinstance(frame, str)]
    closed = {message.stream for message in messages if isinstance(message, HearClose)}
    return [message.stream in closed for message in messages if isinstance(message, HearOpen)]


def rest_of_reply(sdk, emit, response_id, text, pcm):
    """A reply whose ResponseStarted Muse already sent."""
    for word in text.split(" "):
        emit(sdk.TranscriptDelta(response_id, word + " "))
    for offset in range(0, len(pcm), 9600):
        emit(sdk.AudioChunk(response_id, offset // 2, pcm[offset:offset + 9600]))
    emit(sdk.TranscriptDone(response_id, text))
    emit(sdk.ResponseDone(response_id, "completed"))


def facing(controller, robot):
    return round(math.degrees(controller.head_yaw)), round(math.degrees(robot.body_yaw))


def assert_allowed(results, caplog, *, words=(REQUEST, REPLY, "Sarah")):
    """Every result carries only the allowed fields, and every line about the commands only fixed words."""
    for result in results:
        assert set(result) <= {"method", "id", "ok", "payload", "error"}
        payload = result.get("payload")
        if payload is None:
            assert result["ok"] is False and result["error"] in ("invalid_parameters", "timeout")
        elif set(payload) == {"outcome"}:
            assert payload["outcome"] in OUTCOMES
        else:
            assert set(payload) == {"speaker", "people_visible", "seen_s_ago"}
            assert payload["speaker"] in ("owner", "someone_else", "unknown")
            for count in (payload["people_visible"], payload["seen_s_ago"]):
                assert count is None or type(count) is int and count >= 0
            assert (payload["people_visible"] is None) == (payload["seen_s_ago"] is None)
    for line in caplog.messages:
        if re.search(r"reachy\.(people|look_at)|Reachy (people|look_at)", line):
            assert any(re.fullmatch(pattern, line) for pattern in ALLOWED_LOG_LINES), line
    for text in words:
        assert text not in caplog.text


# ---- reachy.people


@pytest.mark.parametrize("label, speaker", [
    (SpeakerLabel.OWNER, "owner"),
    (SpeakerLabel.OTHER, "someone_else"),
    (SpeakerLabel.UNKNOWN, "unknown"),
    (None, "unknown"),
])
def test_muse_learns_whether_its_owner_asked_and_how_many_people_reachy_sees(people, caplog, label, speaker):
    """Without a label, Reachy waits its full second for one, so the snapshot taken just before is a second old."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")

    async def ask(ordinal):
        await heard(rig, caplog)
        if label is not None:
            await labeled(rig, caplog, label)
        assert len((await asyncio.to_thread(controller.refresh_faces)).faces) == 2
        await rig.session.invoke("reachy.people")

    muse = Muse(ask)

    async def user(voice):
        await say_live(rig, wake=True)
        await replied(rig, voice)

    converse(rig, muse.calls, companion(label), user)
    assert rig.session.results == [{"method": "link.result", "id": "invoke-1", "ok": True, "payload": {
        "speaker": speaker, "people_visible": 2, "seen_s_ago": 1 if label is None else 0}}]
    assert logged(caplog, "Reachy people") == [f"Reachy people speaker={speaker} visible=2"]
    assert logged(caplog, "Reachy speaker=") == ([f"Reachy speaker={label.value}"] if label is not None else [])
    assert set(played(rig.recording.events)) == {"muse"}
    assert muse.calls.barge_ins == 0
    assert_allowed(rig.session.results, caplog)


@pytest.mark.parametrize("faces, age, visible, seen", [
    ([(30, .35), (-25, .12)], 0, 2, 0),
    ([(30, .35)], 0, 1, 0),
    ([], 0, 0, 0),
    ([(30, .35), (-25, .12)], 9, 2, 9),
    ([(30, .35), (-25, .12)], 11, None, None),
    (None, None, None, None),
])
def test_visible_people_come_from_a_face_snapshot_at_most_about_ten_seconds_old(people, caplog, faces, age,
                                                                                visible, seen):
    rig, controller, robot, scene = people
    caplog.set_level("INFO")

    async def ask(ordinal):
        await labeled(rig, caplog, SpeakerLabel.OWNER)
        if faces is not None:
            scene.faces, scene.age = faces, age
            assert len((await asyncio.to_thread(controller.refresh_faces)).faces) == len(faces)
        await rig.session.invoke("reachy.people")

    muse = Muse(ask)

    async def user(voice):
        await say_live(rig, wake=True)
        await replied(rig, voice)

    converse(rig, muse.calls, companion(), user)
    assert [result["payload"] for result in rig.session.results] == [
        {"speaker": "owner", "people_visible": visible, "seen_s_ago": seen}]
    assert logged(caplog, "Reachy people") == [
        f"Reachy people speaker=owner visible={'none' if visible is None else visible}"]
    assert_allowed(rig.session.results, caplog)


def test_a_label_from_a_closed_listening_window_is_unknown(people, caplog):
    rig, controller, robot, scene = people
    caplog.set_level("INFO")

    async def ask(ordinal):
        await labeled(rig, caplog, SpeakerLabel.OWNER)
        await rig.session.invoke("reachy.people")

    muse = Muse(ask)

    async def user(voice):
        await say_live(rig, wake=True)
        await replied(rig, voice)
        await until(rig, lambda: "Reachy's wake window closed after inactivity" in caplog.messages, 8)
        await rig.session.invoke("reachy.people")

    converse(rig, muse.calls, companion(), user)
    assert [result["payload"]["speaker"] for result in rig.session.results] == ["owner", "unknown"]
    assert logged(caplog, "Reachy people") == ["Reachy people speaker=owner visible=none",
                                               "Reachy people speaker=unknown visible=none"]
    assert_allowed(rig.session.results, caplog)


@pytest.mark.parametrize("asked, speaker", [
    ("after_an_unlabeled_request", "unknown"),
    ("while_the_next_request_is_spoken", "someone_else"),
    ("while_a_long_request_is_spoken", "unknown"),
])
def test_the_owners_label_never_stands_for_a_later_request(people, caplog, asked, speaker):
    """The owner asks first, then someone else does. Asked while that request is still spoken, Reachy waits
    up to a second for its label; a request without a label, or still unfinished by then, is unknown."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    fake = companion()
    waited = []

    async def ask(ordinal):
        if ordinal == 1:
            await labeled(rig, caplog, SpeakerLabel.OWNER)
            await rig.session.invoke("reachy.people")
        elif asked == "after_an_unlabeled_request":
            await heard(rig, caplog, 2)
            await rig.session.invoke("reachy.people")

    muse = Muse(ask)

    async def user(voice):
        await say_live(rig, wake=True)
        await replied(rig, voice)
        await room(rig, .5)
        fake.speaker = None if asked == "after_an_unlabeled_request" else SpeakerLabel.OTHER
        if asked == "after_an_unlabeled_request":
            await say_live(rig)
        else:
            speaking = asyncio.create_task(say_live(rig, voiced_s=.5 if asked == "while_the_next_request_is_spoken"
                                                    else 4.5))
            await until(rig, lambda: voice._playback.user_speaking)
            started = asyncio.get_running_loop().time()
            await rig.session.invoke("reachy.people")
            waited.append((asyncio.get_running_loop().time() - started, voice._playback.user_speaking))
            await speaking
        await replied(rig, voice, 2)

    converse(rig, muse.calls, fake, user, wake_timeout_s=10)
    assert [result["payload"]["speaker"] for result in rig.session.results] == ["owner", speaker]
    assert len(logged(caplog, "Realtime command ")) == 2
    if asked == "while_a_long_request_is_spoken":
        assert len(waited) == 1 and .95 <= waited[0][0] < 1.5 and waited[0][1] is True
    assert_allowed(rig.session.results, caplog)


def test_a_sound_that_never_becomes_a_request_leaves_the_owners_request_standing(people, caplog):
    """A cough after the owner's request is too short to be a request, so Muse still hears about the owner's."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")

    async def ask(ordinal):
        await labeled(rig, caplog, SpeakerLabel.OWNER)
        for _ in range(5):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
        await room(rig, .6)
        assert len(logged(caplog, "Realtime command ")) == 1
        await rig.session.invoke("reachy.people")
        await rig.session.invoke("reachy.look_at", {"target": "speaker"})

    muse = Muse(ask)

    async def user(voice):
        await say_live(rig, wake=True)
        await replied(rig, voice)
        await until(rig, lambda: facing(controller, robot) == (30, 10))

    converse(rig, muse.calls, companion(), user)
    assert [result["payload"] for result in rig.session.results] == [
        {"speaker": "owner", "people_visible": None, "seen_s_ago": None}, {"outcome": "turned"}]
    assert_allowed(rig.session.results, caplog)


def test_muse_asking_before_reachys_own_end_of_the_request_hears_about_it(people, caplog):
    """On the robot Reachy ends a request after more than a second of silence, and Muse often acts first."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    asked = []

    async def ask(ordinal):
        asked.append(len(logged(caplog, "Realtime command ")))
        await rig.session.invoke("reachy.people")
        await rig.session.invoke("reachy.look_at", {"target": "speaker"})

    muse = Muse(ask)

    async def user(voice):
        await say_live(rig, wake=True, paced_silence_s=2)
        await replied(rig, voice)
        await until(rig, lambda: facing(controller, robot) == (30, 10))

    converse(rig, muse.calls, companion(), user, silence_s=1.2)
    assert asked == [0]
    assert [result["payload"] for result in rig.session.results] == [
        {"speaker": "owner", "people_visible": None, "seen_s_ago": None}, {"outcome": "turned"}]
    assert_allowed(rig.session.results, caplog)


def test_muses_own_voice_heard_back_never_stands_for_the_user(people, caplog):
    """While Muse's reply plays, its echo reaches the microphone and the companion labels it someone else.
    Reachy never admits that echo as a request, so the owner's request is still the one Muse hears about."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    rig.hardware.echo_cancelled_input = True
    fake = companion()

    async def answer(sdk, emit, calls):
        await labeled(rig, caplog, SpeakerLabel.OWNER)
        reply_events(sdk, emit, "r1", REPLY, voiced(2.5))

    async def user(voice):
        await say_live(rig, wake=True)
        await until(rig, lambda: voice.call is not None and voice.call.audible)
        fake.partials, fake.speaker = (REPLY,), SpeakerLabel.OTHER
        await say_live(rig, voiced_s=.6)
        await labeled(rig, caplog, SpeakerLabel.OTHER, 2)
        await heard(rig, caplog, 2)
        await rig.session.invoke("reachy.people")

    calls = ScriptedCalls(answer=answer)
    converse(rig, calls, fake, user)
    assert [result["payload"]["speaker"] for result in rig.session.results] == ["owner"]
    assert logged(caplog, "Realtime command ")[1].endswith(" drop=barge_in")
    assert calls.barge_ins == 0
    assert_allowed(rig.session.results, caplog)


@pytest.mark.parametrize("ended_by", ["reachy", "companion_before_the_label", "companion_after_the_label"])
def test_reachy_people_answers_as_soon_as_the_label_lands(people, caplog, monkeypatch, ended_by):
    """Reachy ends a request after 2 s of silence, as on the robot. The companion labels it at its own end of
    speech plus its speaker check, 0.65 s after the voice. Most requests end at the companion's endpoint, about
    0.94 s after the voice; an endpoint can also come before the label, or a slow final pass leaves the end to
    Reachy. Muse asks at its own end of speech and hears as soon as the label lands, whatever ends the request;
    it replies once the request has ended."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    arrived = label_arrivals(monkeypatch)
    answered = []
    endpoint_quiet_frames = {"reachy": None, "companion_before_the_label": END_OF_SPEECH_FRAMES + 10,
                             "companion_after_the_label": 47}[ended_by]
    fake = companion(endpoint_quiet_frames=endpoint_quiet_frames, speaker_check_s=.55)

    async def ask(ordinal):
        await rig.session.invoke("reachy.people")
        answered.append(time.monotonic())
        await heard(rig, caplog)

    muse = Muse(ask)

    async def user(voice):
        await say_live(rig, wake=True, paced_silence_s=2.6)
        await replied(rig, voice)

    converse(rig, muse.calls, fake, user, silence_s=2.0)
    assert [result["payload"]["speaker"] for result in rig.session.results] == ["owner"]
    assert closed_by_reachy(fake) == [ended_by == "reachy"]
    assert len(arrived) == 1 and 0 <= answered[0] - arrived[0] < .3
    assert_allowed(rig.session.results, caplog)


def test_without_voice_labels_reachy_people_answers_at_once(people, caplog):
    """A companion that labels no voices, as in production today: Muse, asking before Reachy's own end of the
    request, hears unknown at once."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    waited = []

    async def ask(ordinal):
        started = time.monotonic()
        await rig.session.invoke("reachy.people")
        waited.append(time.monotonic() - started)

    muse = Muse(ask)

    async def user(voice):
        await say_live(rig, wake=True, paced_silence_s=2.6)
        await replied(rig, voice)

    converse(rig, muse.calls, companion(ops=(Op.HEAR, Op.SPEAK, Op.NARRATE)), user, silence_s=2.0)
    assert [result["payload"]["speaker"] for result in rig.session.results] == ["unknown"]
    assert len(waited) == 1 and waited[0] < .3
    assert_allowed(rig.session.results, caplog)


def test_an_aside_while_muse_prepares_its_answer_never_stands_for_the_guest(people, caplog):
    """A guest asks; while Muse prepares its answer, the owner says something aside, and Muse asks who made the
    request it is answering. Muse heard both before answering, and their labels disagree."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    fake = companion(SpeakerLabel.OTHER)
    aside = asyncio.Event()
    voices = []

    async def answer(sdk, emit, calls):
        await labeled(rig, caplog, SpeakerLabel.OTHER)
        aside.set()
        await until(rig, lambda: voices[0]._playback.user_speaking)
        await rig.session.invoke("reachy.people")
        reply_events(sdk, emit, "r1", REPLY, REPLY_AUDIO)

    async def user(voice):
        voices.append(voice)
        await say_live(rig, wake=True)
        await asyncio.wait_for(aside.wait(), 5)
        fake.speaker = SpeakerLabel.OWNER
        await say_live(rig, voiced_s=.5)
        await until(rig, lambda: len(rig.session.results) == 1, 6)
        await replied(rig, voice)

    converse(rig, ScriptedCalls(answer=answer), fake, user, wake_timeout_s=10)
    assert logged(caplog, "Reachy speaker=") == ["Reachy speaker=other", "Reachy speaker=owner"]
    assert [result["payload"]["speaker"] for result in rig.session.results] == ["unknown"]
    assert_allowed(rig.session.results, caplog)


def test_a_response_that_asks_is_about_the_request_before_it_not_an_aside_during_it(people, caplog):
    """A guest at -30 degrees asks, and Muse's response starts; before it says a word, the owner at +30 adds
    something. That response asks who made the request and turns toward them: the guest, not the owner."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    fake = companion(SpeakerLabel.OTHER)
    scene.doa = 2 * math.pi / 3
    started, aside_heard = asyncio.Event(), asyncio.Event()

    async def answer(sdk, emit, calls):
        emit(sdk.ResponseStarted("r1"))
        started.set()
        await asyncio.wait_for(aside_heard.wait(), 5)
        await rig.session.invoke("reachy.people")
        await rig.session.invoke("reachy.look_at", {"target": "speaker"})
        rest_of_reply(sdk, emit, "r1", REPLY, REPLY_AUDIO)

    async def user(voice):
        await say_live(rig, wake=True)
        await asyncio.wait_for(started.wait(), 5)
        fake.speaker, scene.doa = SpeakerLabel.OWNER, math.pi / 3
        await say_live(rig, voiced_s=.5)
        await heard(rig, caplog, 2)
        await labeled(rig, caplog, SpeakerLabel.OWNER, 2)
        aside_heard.set()
        await replied(rig, voice)
        await until(rig, lambda: facing(controller, robot) == (-30, -10))

    converse(rig, ScriptedCalls(answer=answer), fake, user, wake_timeout_s=10)
    assert [result["payload"].get("speaker", result["payload"].get("outcome"))
            for result in rig.session.results] == ["someone_else", "turned"]
    assert facing(controller, robot) == (-30, -10)
    assert_allowed(rig.session.results, caplog)


@pytest.mark.parametrize("aside, speaker", [(False, "someone_else"), (True, "unknown")])
def test_while_muse_works_on_a_request_it_hears_about_whoever_made_it(people, caplog, aside, speaker):
    """The owner asks and Muse answers. Then a guest asks, and Muse works on it before answering; meanwhile the
    owner may remark aside, which Muse's work could take as part of the guest's request."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    fake = companion()
    remarked = asyncio.Event()

    async def answer(sdk, emit, calls):
        ordinal = len(calls.answering)
        if ordinal == 2:
            emit(sdk.Working(True))
            if aside:
                await asyncio.wait_for(remarked.wait(), 5)
            await rig.session.invoke("reachy.people")
            emit(sdk.Working(False))
        if ordinal <= 2:
            reply_events(sdk, emit, f"r{ordinal}", REPLY, REPLY_AUDIO)

    async def user(voice):
        await say_live(rig, wake=True)
        await replied(rig, voice)
        await room(rig, .5)
        fake.speaker = SpeakerLabel.OTHER
        await say_live(rig)
        if aside:
            await labeled(rig, caplog, SpeakerLabel.OTHER, 2)
            fake.speaker = SpeakerLabel.OWNER
            await say_live(rig, voiced_s=.5)
            await labeled(rig, caplog, SpeakerLabel.OWNER, 3)
            remarked.set()
        await until(rig, lambda: len(rig.session.results) == 1, 6)
        await replied(rig, voice, 2)

    converse(rig, ScriptedCalls(answer=answer, every_turn=True), fake, user, wake_timeout_s=10)
    assert [result["payload"]["speaker"] for result in rig.session.results] == [speaker]
    assert_allowed(rig.session.results, caplog)


def test_in_chat_muse_hears_about_the_request_it_was_sent_not_one_said_while_it_works(people, caplog):
    """In chat, the owner asks and Muse works on it; a guest asks something else meanwhile, which waits its
    turn. Muse's answer to the owner hears about the owner, and its answer to the guest about someone else."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    fake = companion()

    async def muse(session, request):
        if request == 1:
            await until(rig, lambda: "Reachy heard a new follow-up while Muse worked" in caplog.messages, 8)
            await labeled(rig, caplog, SpeakerLabel.OTHER, 2)
        await rig.session.invoke("reachy.people")
        await session.answer(request, REPLY)

    async def user(voice):
        await say_live(rig, wake=True)
        await until(rig, lambda: rig.session.requests == 1)
        fake.speaker = SpeakerLabel.OTHER
        await room(rig, .3)
        await say_live(rig)
        await until(rig, lambda: len(rig.session.results) == 2, 10)

    converse(rig, None, fake, user, muse=muse, wake_timeout_s=10)
    assert [result["payload"]["speaker"] for result in rig.session.results] == ["owner", "someone_else"]
    assert_allowed(rig.session.results, caplog)


# ---- reachy.look_at


@pytest.mark.parametrize("target, turned, held", [("speaker", (30, 10), False), ("other", (-25, -5), True)])
def test_muse_turns_reachy_to_the_speaker_or_holds_it_on_someone_else(people, caplog, target, turned, held):
    """The speaker stands at +30 degrees and a friend at -25. Only a look at the friend is held, and the next
    request ends it, as a spoken greeting's does."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    seen = {}

    async def ask(ordinal):
        if ordinal == 1:
            await heard(rig, caplog)
            await rig.session.invoke("reachy.look_at", {"target": target})

    muse = Muse(ask)

    async def user(voice):
        await say_live(rig, wake=True)
        await replied(rig, voice)
        await until(rig, lambda: facing(controller, robot) == turned)
        seen["after_the_reply"] = facing(controller, robot), controller.person_gaze_held
        await room(rig, .5)
        await say_live(rig)
        await replied(rig, voice, 2)
        await asyncio.sleep(SPEAKER_TURN_S + .3)
        seen["after_the_next_request"] = facing(controller, robot)

    converse(rig, muse.calls, companion(), user)
    assert rig.session.results == [{"method": "link.result", "id": "invoke-1", "ok": True,
                                    "payload": {"outcome": "turned"}}]
    assert logged(caplog, "Reachy look_at") == [f"Reachy look_at target={target} outcome=turned"]
    assert logged(caplog, "Reachy gaze request") == []
    assert seen == {"after_the_reply": (turned, held), "after_the_next_request": (0, 0) if held else turned}
    assert muse.calls.barge_ins == 0
    assert_allowed(rig.session.results, caplog)


@pytest.mark.parametrize("looked", [True, False])
def test_reachys_own_closing_keeps_the_turn_toward_the_speaker_muse_asked_for(people, caplog, looked):
    """Muse turns Reachy toward the speaker in a response of its own, then replies in another. Without companion
    motion, Reachy closes a reply with a recorded clip, a motion command that would end that turn; it closes
    with one only when Muse did not turn it."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    rig.hardware.run_command = controller.run_command

    async def answer(sdk, emit, calls):
        await heard(rig, caplog)
        if looked:
            emit(sdk.ResponseStarted("tool"))
            await rig.session.invoke("reachy.look_at", {"target": "speaker"})
            emit(sdk.ResponseDone("tool", "completed"))
        reply_events(sdk, emit, "r1", REPLY, voiced(.6))

    async def user(voice):
        await say_live(rig, wake=True)
        await until(rig, lambda: "muse" in played(rig.recording.events) and not voice.call.audible)
        await until(rig, lambda: not voice._call_tasks)

    converse(rig, ScriptedCalls(answer=answer), companion(), user)
    assert [result["payload"] for result in rig.session.results] == ([{"outcome": "turned"}] if looked else [])
    assert [line for line in caplog.messages if line.startswith("Reachy closed Muse's realtime reply")] == (
        [] if looked else ["Reachy closed Muse's realtime reply with happy"])
    assert facing(controller, robot) == ((30, 10) if looked else (0, 0))


@pytest.mark.parametrize("target, case, outcome", [
    ("other", "only_the_speaker", "no_other"),
    ("other", "no_faces", "no_faces"),
    ("other", "stale", "stale"),
    ("other", "no_direction", "no_speaker_bearing"),
    ("speaker", "no_direction", "no_speaker_bearing"),
])
def test_look_at_says_why_reachy_stays_put(people, caplog, target, case, outcome):
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    if case in ("only_the_speaker", "stale"):
        scene.faces = [(30, .35)]
    if case == "no_faces":
        scene.faces = []
    if case == "stale":
        scene.age = 3
        assert len(controller.refresh_faces().faces) == 1
        scene.camera = False
    if case == "no_direction":
        scene.doa = None

    async def ask(ordinal):
        await heard(rig, caplog)
        await rig.session.invoke("reachy.look_at", {"target": target})

    muse = Muse(ask)

    async def user(voice):
        await say_live(rig, wake=True)
        await replied(rig, voice)
        await asyncio.sleep(SPEAKER_TURN_S)

    converse(rig, muse.calls, companion(), user)
    assert [result["payload"] for result in rig.session.results] == [{"outcome": outcome}]
    assert logged(caplog, "Reachy look_at") == [f"Reachy look_at target={target} outcome={outcome}"]
    assert facing(controller, robot) == (0, 0)
    assert not controller.person_gaze_held
    assert_allowed(rig.session.results, caplog)


@pytest.mark.parametrize("first, target, turned", [
    ("greeting", "speaker", (30, 10)),
    ("greeting", "other", (-25, -5)),
    ("speaker", "other", (-25, -5)),
])
def test_look_at_is_busy_only_while_reachy_still_turns_toward_a_person(people, caplog, monkeypatch, first, target,
                                                                       turned):
    """A person turn is under way: Reachy's own for a spoken greeting, or the one Muse asked for toward the
    speaker, asked again past the thinking motion's fade, while it moves. Muse's look_at is busy while Reachy
    turns, and turns it once that turn has settled."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    monkeypatch.setattr("reachy_muse.hardware.SPEAKER_TURN_S", 2.0)
    outcomes = []

    async def ask(ordinal):
        if first == "greeting":
            await until(rig, lambda: logged(caplog, "Reachy gaze request") == [
                "Reachy gaze request outcome=turned source=fresh faces=2"])
        else:
            await heard(rig, caplog)
            await rig.session.invoke("reachy.look_at", {"target": "speaker"})
            await asyncio.sleep(.8)
        outcomes.append((controller.person_turning, (await rig.session.invoke(
            "reachy.look_at", {"target": target}))["payload"]["outcome"]))
        await until(rig, lambda: not controller.person_turning, 4)
        outcomes.append((controller.person_turning, (await rig.session.invoke(
            "reachy.look_at", {"target": target}))["payload"]["outcome"]))

    muse = Muse(ask)

    async def user(voice):
        await say_live(rig, wake=True)
        await replied(rig, voice)
        await until(rig, lambda: facing(controller, robot) == turned, 4)

    converse(rig, muse.calls, companion(request="Say hi to my friend." if first == "greeting" else REQUEST), user)
    assert outcomes == [(True, "busy"), (False, "turned")]
    assert logged(caplog, "Reachy look_at") == [
        *(["Reachy look_at target=speaker outcome=turned"] if first == "speaker" else []),
        f"Reachy look_at target={target} outcome=busy", f"Reachy look_at target={target} outcome=turned"]
    assert_allowed(rig.session.results, caplog, words=(REPLY, "Say hi", "Sarah"))


def test_look_at_is_busy_while_a_greeting_still_chooses_whom_to_face(people, caplog, monkeypatch):
    """The user asks Reachy itself to greet their friend, and Muse asks to look while Reachy's camera pass for
    that greeting is still running: the greeting's turn is already pending."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    monkeypatch.setattr("reachy_muse.person_gaze._FRESH_PASS_S", 2.0)
    held, release = threading.Event(), threading.Event()

    def hold_the_pass():
        scene.before_capture = None
        held.set()
        release.wait(2)

    scene.before_capture = hold_the_pass

    async def ask(ordinal):
        await until(rig, held.is_set)
        await rig.session.invoke("reachy.look_at", {"target": "other"})
        release.set()
        await until(rig, lambda: bool(logged(caplog, "Reachy gaze request")))

    muse = Muse(ask)

    async def user(voice):
        await say_live(rig, wake=True)
        await replied(rig, voice)
        await until(rig, lambda: facing(controller, robot) == (-25, -5))

    converse(rig, muse.calls, companion(request="Say hi to my friend."), user)
    assert [result["payload"] for result in rig.session.results] == [{"outcome": "busy"}]
    assert logged(caplog, "Reachy look_at") + logged(caplog, "Reachy gaze request") == [
        "Reachy look_at target=other outcome=busy", "Reachy gaze request outcome=turned source=fresh faces=2"]
    assert scene.passes == 1
    assert_allowed(rig.session.results, caplog, words=(REPLY, "Say hi"))


def test_nothing_beyond_the_allowed_fields_reaches_muse_or_the_log(people, caplog, monkeypatch):
    """A guest names someone in the request, Muse names them in its reply and in invalid parameters; Muse
    still gets only the allowed fields, and the log only fixed words."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    monkeypatch.setattr("reachy_muse.hardware.SPEAKER_TURN_S", 2.0)

    async def ask(ordinal):
        await labeled(rig, caplog, SpeakerLabel.OTHER)
        assert len((await asyncio.to_thread(controller.refresh_faces)).faces) == 2
        for command, params in (("reachy.people", {"name": "Sarah"}), ("reachy.look_at", {"target": "Sarah"}),
                                ("reachy.look_at", {}), ("reachy.people", {}),
                                ("reachy.look_at", {"target": "other"}), ("reachy.look_at", {"target": "other"})):
            await rig.session.invoke(command, params)

    muse = Muse(ask)

    async def user(voice):
        await say_live(rig, wake=True)
        await replied(rig, voice)

    converse(rig, muse.calls, companion(SpeakerLabel.OTHER), user)
    assert [{key: value for key, value in result.items() if key != "id"} for result in rig.session.results] == [
        {"method": "link.result", "ok": False, "error": "invalid_parameters"},
        {"method": "link.result", "ok": False, "error": "invalid_parameters"},
        {"method": "link.result", "ok": False, "error": "invalid_parameters"},
        {"method": "link.result", "ok": True,
         "payload": {"speaker": "someone_else", "people_visible": 2, "seen_s_ago": 0}},
        {"method": "link.result", "ok": True, "payload": {"outcome": "turned"}},
        {"method": "link.result", "ok": True, "payload": {"outcome": "busy"}},
    ]
    assert logged(caplog, "Reachy people") + logged(caplog, "Reachy look_at") == [
        "Reachy people outcome=invalid_parameters",
        "Reachy people speaker=someone_else visible=2",
        "Reachy look_at outcome=invalid_parameters",
        "Reachy look_at outcome=invalid_parameters",
        "Reachy look_at target=other outcome=turned",
        "Reachy look_at target=other outcome=busy",
    ]
    assert_allowed(rig.session.results, caplog)


def test_a_request_muse_answers_only_by_turning_reachy_is_answered(people, caplog):
    """Turning Reachy is Muse acting on the request, as a nod is, so a response without words answers it."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")

    async def answer(sdk, emit, calls):
        await heard(rig, caplog)
        await rig.session.invoke("reachy.look_at", {"target": "other"})
        emit(sdk.ResponseStarted("r1"))
        emit(sdk.ResponseDone("r1", "completed"))

    async def user(voice):
        await say_live(rig, wake=True)
        await until(rig, lambda: bool(logged(caplog, "Realtime turn ordinal=1 outcome=")))

    converse(rig, ScriptedCalls(answer=answer), companion(), user)
    assert [line.split()[3] for line in logged(caplog, "Realtime turn ordinal=1 ")] == ["outcome=silent"]
    assert [result["payload"] for result in rig.session.results] == [{"outcome": "turned"}]
    assert played(rig.recording.events) == []
