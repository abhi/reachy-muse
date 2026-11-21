"""Muse's people commands through ReachyService.serve.

Each request reaches Reachy through its microphone loop, with fake DoA and camera faces, while the fake
companion hears it and labels the voice. Muse calls the commands with link.invoke, through the SDK's own
handler, and learns only what the owner allowed.
"""

import asyncio
from dataclasses import replace
import itertools
import re

import pytest
from musegadget.link_client import MAX_CONCURRENT_INVOKES, LinkSession

from fake_companion import TOKEN, FakeCompanion
from reachy_muse.companion_client import CompanionHearing, CompanionLink
from reachy_muse.failover import CompanionRoute, FailoverHearing
from reachy_muse.protocol import Op, SpeakerLabel
from test_device_command_turns import never_asked, played
from test_person_gaze import people, say_live  # noqa: F401  # people is the fixture shared with people1's tests
from test_pins import PinSession
from test_realtime_turns import ScriptedCalls, StreamingTranscripts, realtime_service, reply_events, room, voiced

REQUEST = "Who is here with me, Sarah?"
REPLY = "Hello Sarah, nice to see you!"
REPLY_AUDIO = voiced(.2)
ALLOWED_LOG_LINES = (
    r"Reachy people speaker=(owner|someone_else|unknown) visible=(none|\d+)",
    r"Reachy people outcome=(invalid_parameters|timeout)",
    r"invoke reachy\.people",
    r"reachy\.people (ok|failed) in \d+ ms",
)


class InvokedSession(PinSession):
    """Muse's end of the registered link: ``invoke`` hands a link.invoke to the SDK's own handler."""

    def __init__(self, recording):
        super().__init__(recording, never_asked)
        self._register_id = "register"
        self._tasks = set()
        self._invokes = asyncio.Semaphore(MAX_CONCURRENT_INVOKES)
        self._sent = asyncio.Queue()
        self._ids = itertools.count(1)
        self.results = []

    async def send(self, message):
        await self._sent.put(message)

    async def invoke(self, command, params=None):
        message = {"method": "link.invoke", "id": f"invoke-{next(self._ids)}", "command": command,
                   "params": {} if params is None else params}
        LinkSession._handle(self, message)
        result = await asyncio.wait_for(self._sent.get(), 3)
        assert result["id"] == message["id"]
        self.results.append(result)
        return result


class Muse:
    """Muse's side of one realtime call: before replying to its ``n``-th request it runs ``then(n)``,
    such as its tool calls, and it answers every request."""

    def __init__(self, then):
        async def answer(sdk, emit, calls):
            ordinal = len(calls.answering)
            await then(ordinal)
            reply_events(sdk, emit, f"r{ordinal}", REPLY, REPLY_AUDIO)

        self.calls = ScriptedCalls(answer=answer, every_turn=True)


def companion(speaker=SpeakerLabel.OWNER):
    return FakeCompanion(ops=(Op.HEAR, Op.SPEAK, Op.NARRATE, Op.SPEAKER), partials=(REQUEST,), speaker=speaker)


def converse(rig, muse, fake, user):
    """Reachy in a realtime call that hears through ``fake``, from rest until ``user(voice)`` is done and it
    rests again."""
    service = realtime_service(rig, StreamingTranscripts(), muse.calls)
    rig.session = InvokedSession(rig.recording)

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
                assert link.can(Op.SPEAKER)
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


def assert_allowed(results, caplog):
    """Every result carries only the allowed fields, and every line about the commands only fixed words."""
    for result in results:
        assert set(result) <= {"method", "id", "ok", "payload", "error"}
        if "payload" in result:
            payload = result["payload"]
            assert set(payload) == {"speaker", "people_visible", "seen_s_ago"}
            assert payload["speaker"] in ("owner", "someone_else", "unknown")
            for count in (payload["people_visible"], payload["seen_s_ago"]):
                assert count is None or type(count) is int and count >= 0
            assert (payload["people_visible"] is None) == (payload["seen_s_ago"] is None)
        else:
            assert result["ok"] is False and result["error"] in ("invalid_parameters", "timeout")
    for line in caplog.messages:
        if "people" in line:
            assert any(re.fullmatch(pattern, line) for pattern in ALLOWED_LOG_LINES), line
    for words in ("Sarah", REQUEST, REPLY):
        assert words not in caplog.text


@pytest.mark.parametrize("label, speaker", [
    (SpeakerLabel.OWNER, "owner"),
    (SpeakerLabel.OTHER, "someone_else"),
    (SpeakerLabel.UNKNOWN, "unknown"),
    (None, "unknown"),
])
def test_muse_learns_whether_its_owner_asked_and_how_many_people_reachy_sees(people, caplog, label, speaker):
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

    converse(rig, muse, companion(label), user)
    assert rig.session.results == [{"method": "link.result", "id": "invoke-1", "ok": True,
                                    "payload": {"speaker": speaker, "people_visible": 2, "seen_s_ago": 0}}]
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

    converse(rig, muse, companion(), user)
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

    converse(rig, muse, companion(), user)
    assert [result["payload"]["speaker"] for result in rig.session.results] == ["owner", "unknown"]
    assert logged(caplog, "Reachy people") == ["Reachy people speaker=owner visible=none",
                                               "Reachy people speaker=unknown visible=none"]
    assert_allowed(rig.session.results, caplog)


@pytest.mark.parametrize("asked", ["after_an_unlabeled_request", "while_the_next_request_is_spoken"])
def test_the_owners_label_never_stands_for_a_later_request(people, caplog, asked):
    """The owner asks first. Then another request comes without a label, or Muse asks before Reachy's end of it."""
    rig, controller, robot, scene = people
    caplog.set_level("INFO")
    fake = companion()

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
        if asked == "after_an_unlabeled_request":
            fake.speaker = None
            await say_live(rig)
        else:
            speaking = asyncio.create_task(say_live(rig, voiced_s=1.2))
            await until(rig, lambda: voice._playback.user_speaking)
            await rig.session.invoke("reachy.people")
            assert len(logged(caplog, "Realtime command ")) == 1
            await speaking
        await replied(rig, voice, 2)

    converse(rig, muse, fake, user)
    assert [result["payload"]["speaker"] for result in rig.session.results] == ["owner", "unknown"]
    assert len(logged(caplog, "Realtime command ")) == 2
    assert_allowed(rig.session.results, caplog)
