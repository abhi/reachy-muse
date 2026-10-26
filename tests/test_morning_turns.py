import asyncio
import time
from datetime import datetime, timedelta

import pytest

from reachy_muse.contract.prompts import STOP_REQUEST
from reachy_muse.morning_briefing import BriefingSchedule, DayLedger, MorningRoutine
from test_device_command_turns import LongVoice, played, rig_with, run, sent_to_muse, service
from test_pins import Transcripts, Wake

np = pytest.importorskip("numpy")

SENTENCES = ["Good morning.", "Pick one task and finish it first.", "Stocks rose yesterday.",
             "The Nasdaq gained one percent.", "A new phone chip shipped.", "That's your briefing."]
BRIEFING = " ".join(SENTENCES) + " [reachy:happy]"
SPOKEN = ["tts:Good morning!", *(f"tts:{sentence}" for sentence in SENTENCES)]


class WallClock:
    def __init__(self):
        self.opened = time.monotonic()

    def now_local(self):
        return datetime(2026, 10, 10, 4, 50) + timedelta(seconds=time.monotonic() - self.opened)

    def monotonic(self):
        return time.monotonic()

    async def sleep(self, seconds):
        await asyncio.sleep(seconds)


class Camera:
    def __init__(self):
        self.arrived = None
        self.calls = []

    def start_watching(self, pose):
        self.calls.append("start")

    def stop_watching(self):
        self.calls.append("stop")

    def arrival(self):
        return self.arrived

    def arrive(self):
        self.arrived = time.monotonic()


def morning(camera, **options):
    return MorningRoutine(BriefingSchedule.parse("04:50-06:00"), "Brief me.", camera, DayLedger(None),
                          clock=WallClock(), poll_s=.02, **options)


def lines_played_without_repeats(events):
    audio = played(events)
    return [line for index, line in enumerate(audio) if index == 0 or audio[index - 1] != line]


def states(events):
    return [event["state"] for event in events if event.get("robot") == "set_state"]


async def briefing(session, request):
    await session.answer(request, BRIEFING)


def test_a_held_briefing_is_spoken_whole_after_good_morning_and_never_before_arrival(monkeypatch):
    rig = rig_with(monkeypatch)
    camera = Camera()
    before_arrival = []

    async def user():
        await rig.wait_until_or_timeout(lambda: sent_to_muse(rig.recording.events), 5)
        await rig.quiet()
        before_arrival.extend(rig.recording.events)
        camera.arrive()
        await rig.wait_until_or_timeout(lambda: camera.calls == ["start", "stop"], 10)

    events = run(rig, service(rig, briefing, Transcripts(), morning=morning(camera)), user)
    assert sent_to_muse(before_arrival) == ["Brief me."]
    assert played(before_arrival) == []
    assert "thinking" not in states(before_arrival)
    assert played(events) == SPOKEN
    assert {"robot": "set_state", "state": "listening", "expression": "happy"} in events[len(before_arrival):]


def test_stop_talking_during_the_briefing_cuts_the_rest(monkeypatch):
    rig = rig_with(monkeypatch, chunk_s=.1)
    rig.hardware.echo_cancelled_input = True
    camera = Camera()

    async def user():
        await rig.wait_until_or_timeout(lambda: sent_to_muse(rig.recording.events), 5)
        await rig.quiet()
        camera.arrive()
        await rig.wait_until_or_timeout(lambda: {"robot": "play", "audio": SPOKEN[1]} in rig.recording.events, 5)
        rig.say()

    conversation = service(rig, briefing, Transcripts("Stop talking."), morning=morning(camera),
                           voice=LongVoice(rig.recording, chunks=20))
    events = run(rig, conversation, user)
    assert sent_to_muse(events) == ["Brief me."]
    assert played(events) == SPOKEN[:2]


def test_stop_talking_hushes_the_briefing_while_a_separate_request_is_pending(monkeypatch):
    rig = rig_with(monkeypatch, chunk_s=.03)
    rig.hardware.echo_cancelled_input = True
    camera = Camera()

    async def muse(session, request):
        if request == 1:
            await session.answer(request, BRIEFING)
        elif request == 2:
            await rig.wait_until_or_timeout(lambda: session.requests == 3, 6)
            await session.answer(request, "Sunny tomorrow. [reachy:happy]")
        else:
            await session.answer(request, "Okay, I stopped. [reachy:nod]")

    async def user():
        await rig.wait_until_or_timeout(lambda: sent_to_muse(rig.recording.events), 5)
        await rig.quiet()
        rig.say()
        await rig.wait_until_or_timeout(lambda: rig.session.requests == 2, 5)
        camera.arrive()
        await rig.wait_until_or_timeout(lambda: {"robot": "play", "audio": SPOKEN[1]} in rig.recording.events, 5)
        rig.say()

    conversation = service(rig, muse, Transcripts("What's the weather tomorrow?", "Stop talking."),
                           morning=morning(camera), voice=LongVoice(rig.recording, chunks=20))
    events = run(rig, conversation, user)
    assert sent_to_muse(events) == ["Brief me.", "What's the weather tomorrow?", STOP_REQUEST]
    assert [line for line in lines_played_without_repeats(events) if line in SPOKEN] == SPOKEN[:2]
    assert "tts:Okay, I'll drop that." not in played(events)


def test_after_the_briefing_a_question_needs_no_wake_phrase(monkeypatch):
    rig = rig_with(monkeypatch)
    camera = Camera()

    async def muse(session, request):
        await session.answer(request, BRIEFING if request == 1 else "Nvidia's chips led the gains. [reachy:nod]")

    async def user():
        await rig.wait_until_or_timeout(lambda: sent_to_muse(rig.recording.events), 5)
        await rig.quiet()
        camera.arrive()
        await rig.wait_until_or_timeout(lambda: camera.calls == ["start", "stop"], 10)
        await rig.quiet("listening")
        rig.say()

    conversation = service(rig, muse, Transcripts("Tell me more about Nvidia."), morning=morning(camera),
                           wake_detector=Wake(), wake_timeout_s=30)
    events = run(rig, conversation, user, rest=("listening", "idle"))
    assert sent_to_muse(events) == ["Brief me.", "Tell me more about Nvidia."]
    assert played(events)[:7] == SPOKEN
    assert played(events)[-1] == "tts:Nvidia's chips led the gains."


def test_an_arrival_wakes_a_sleeping_reachy_before_it_greets(monkeypatch):
    rig = rig_with(monkeypatch)
    camera = Camera()

    async def user():
        await rig.wait_until_or_timeout(lambda: sent_to_muse(rig.recording.events), 5)
        rig.say(wake=True)
        await rig.quiet("sleeping")
        camera.arrive()
        await rig.wait_until_or_timeout(lambda: camera.calls == ["start", "stop"], 10)

    conversation = service(rig, briefing, Transcripts("Hey Muse, go to sleep."), morning=morning(camera),
                           wake_detector=Wake(), wake_timeout_s=30)
    events = run(rig, conversation, user, rest=("listening", "idle"))
    assert played(events) == ["tts:Good night.", *SPOKEN]
    after_sleep = states(events)[states(events).index("sleeping"):]
    awake = after_sleep[after_sleep.index("listening"):]
    assert awake[:2] == ["listening", "listening"]
    assert "sleeping" not in awake


def test_a_question_asked_during_the_silent_prefetch_is_answered_after_it(monkeypatch):
    rig = rig_with(monkeypatch)
    camera = Camera()

    async def muse(session, request):
        if request == 1:
            await rig.wait_until_or_timeout(lambda: {"robot": "gesture", "name": "nod"} in rig.recording.events, 10)
            await session.answer(request, BRIEFING)
        else:
            await session.answer(request, "Sunny and mild. [reachy:happy]")

    async def user():
        await rig.wait_until_or_timeout(lambda: sent_to_muse(rig.recording.events), 5)
        rig.say()

    conversation = service(rig, muse, Transcripts("What's the weather today?"), morning=morning(camera))
    events = run(rig, conversation, user)
    assert sent_to_muse(events) == ["Brief me.", "What's the weather today?"]
    assert played(events)[-1] == "tts:Sunny and mild."
    assert "tts:Good morning!" not in played(events)


def test_stop_talking_works_while_a_refresh_was_still_running_at_arrival(monkeypatch):
    rig = rig_with(monkeypatch, chunk_s=.1)
    rig.hardware.echo_cancelled_input = True
    camera = Camera()
    arrived_at = []

    async def muse(session, request):
        if request == 2:
            await rig.wait_until_or_timeout(lambda: session.requests == 3, 6)
            await session.answer(request, "Markets fell overnight. [reachy:sad]")
        else:
            await session.answer(request, BRIEFING if request == 1 else "Okay, I stopped. [reachy:nod]")

    async def user():
        await rig.wait_until_or_timeout(lambda: len(sent_to_muse(rig.recording.events)) == 2, 5)
        arrived_at.append(len(rig.recording.events))
        camera.arrive()
        await rig.wait_until_or_timeout(lambda: {"robot": "play", "audio": SPOKEN[1]} in rig.recording.events, 5)
        rig.say()

    conversation = service(rig, muse, Transcripts("Stop talking."), morning=morning(camera, refresh_after_s=.3),
                           voice=LongVoice(rig.recording, chunks=20))
    events = run(rig, conversation, user)
    assert sent_to_muse(events) == ["Brief me.", "Brief me.", STOP_REQUEST]
    assert lines_played_without_repeats(events) == SPOKEN[:2]
    assert "thinking" not in states(events[arrived_at[0]:])


def test_stop_talking_stops_a_briefing_longer_than_the_speaker_queue(monkeypatch):
    rig = rig_with(monkeypatch, chunk_s=.1)
    rig.hardware.echo_cancelled_input = True
    camera = Camera()
    sentences = [f"Story number {index} is short." for index in range(1, 41)]

    async def muse(session, request):
        await session.answer(request, " ".join(sentences))

    async def user():
        await rig.wait_until_or_timeout(lambda: sent_to_muse(rig.recording.events), 5)
        await rig.quiet()
        camera.arrive()
        await rig.wait_until_or_timeout(
            lambda: {"robot": "play", "audio": "tts:Story number 1 is short."} in rig.recording.events, 5)
        rig.say()

    conversation = service(rig, muse, Transcripts("Stop talking."), morning=morning(camera),
                           voice=LongVoice(rig.recording, chunks=20))
    events = run(rig, conversation, user)
    assert played(events) == ["tts:Good morning!", "tts:Story number 1 is short."]


@pytest.mark.parametrize("vm_id, session_id, arrived", [
    ("vm-1", "robot-chat", False),
    ("vm-2", "robot-chat", False),
    ("vm-1", "other-chat", False),
    ("vm-2", "robot-chat", True),
    ("vm-1", "other-chat", True),
])
def test_a_link_reconnect_reuses_briefing_state_only_in_the_same_vm_and_chat(monkeypatch, vm_id, session_id, arrived):
    from musegadget.link_client import Outcome
    from reachy_muse.conversation import VoiceConversation
    from test_pins import PinSession

    rig = rig_with(monkeypatch)
    camera = Camera()
    conversation = service(rig, briefing, Transcripts(), morning=morning(camera))

    async def prepare_chat(session, vm):
        return session.chat

    async def next_briefing(session, request):
        await session.answer(request, "Markets fell overnight. [reachy:sad]")

    conversation.prepare_chat = prepare_chat

    async def check():
        disconnected = asyncio.Event()
        stop = asyncio.Event()
        greeting = asyncio.Event()
        first_session = rig.session
        greet = VoiceConversation.greet

        async def interrupted_greeting(voice, *, waiting):
            if voice.session is first_session and arrived:
                greeting.set()
                await asyncio.Event().wait()
            await greet(voice, waiting=waiting)

        monkeypatch.setattr(VoiceConversation, "greet", interrupted_greeting)

        async def link(stop_link):
            await disconnected.wait()
            return Outcome.CLOSED

        rig.session.run = link
        first = asyncio.create_task(conversation.serve(rig.session, {"vm_id": "vm-1"}, stop))
        try:
            await rig.wait_until_or_timeout(lambda: sent_to_muse(rig.recording.events), 5)
            await rig.quiet()
            if arrived:
                camera.arrive()
                await asyncio.wait_for(greeting.wait(), 5)
        finally:
            disconnected.set()
            assert await asyncio.wait_for(first, 5) is Outcome.CLOSED
            await asyncio.gather(*rig.session.scripts)
        camera.arrived = None
        rig.session = PinSession(rig.recording, next_briefing)
        rig.session.chat = session_id
        second = asyncio.create_task(conversation.serve(rig.session, {"vm_id": vm_id}, stop))
        try:
            await rig.wait_until_or_timeout(lambda: sum(event.get("muse") == "subscribe_chat"
                                                       for event in rig.recording.events) == 2, 5)
            await rig.quiet()
            assert played(rig.recording.events) == []
            assert camera.calls == ["start", "stop", "start"]
            camera.arrive()
            await rig.wait_until_or_timeout(lambda: camera.calls == ["start", "stop"] * 2, 10)
            await rig.quiet()
        finally:
            stop.set()
            await asyncio.wait_for(second, 5)
            await asyncio.gather(*rig.session.scripts)

    asyncio.run(asyncio.wait_for(check(), 20))
    same_chat = (vm_id, session_id) == ("vm-1", "robot-chat")
    assert played(rig.recording.events) == (SPOKEN if same_chat else [
        "tts:Good morning!", "tts:Markets fell overnight."])
    assert sent_to_muse(rig.recording.events) == ["Brief me."] * (1 if same_chat else 2)
