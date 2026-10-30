"""Whole turns that wake Reachy from the sleep that "go to sleep" puts it in."""

import asyncio

import pytest

from musegadget.link_client import Outcome
from test_device_command_turns import never_asked, played, rig_with, run, sent_to_muse, service
from test_morning_turns import Camera, SPOKEN, briefing, morning
from test_pins import PinSession, Transcripts, Wake

np = pytest.importorskip("numpy")

GO_TO_SLEEP = "Hey Muse, go to sleep."


def after_sleep(states):
    return states[states.index("sleeping"):]


def awake_after_sleep(states):
    """The states after Reachy first left its sleep pose."""
    asleep = after_sleep(states)
    return [state for index, state in enumerate(asleep) if "sleeping" not in asleep[index:]]


def asleep_then(monkeypatch, second, muse=never_asked, *, wake=True, rest=("listening", "idle")):
    rig = rig_with(monkeypatch)
    transcripts = Transcripts(GO_TO_SLEEP, second)
    conversation = service(rig, muse, transcripts, wake_detector=Wake(), wake_timeout_s=30)

    async def user():
        rig.say(wake=True)
        await rig.quiet("sleeping")
        rig.say(wake=wake)

    events = run(rig, conversation, user, rest=rest)
    return events, rig, transcripts


def test_the_wake_phrase_alone_wakes_reachy_and_greets_without_muse(monkeypatch):
    events, rig, _ = asleep_then(monkeypatch, "Hey Muse.")
    assert sent_to_muse(events) == []
    assert played(events) == ["tts:Good night.", "tts:I'm awake."]
    assert awake_after_sleep(rig.states())[0] == "listening"


def test_a_question_in_the_same_breath_wakes_reachy_and_goes_to_muse_word_for_word(monkeypatch):
    async def muse(session, request):
        await session.answer(request, "Sunny and warm all day. [reachy:happy]")

    events, rig, _ = asleep_then(monkeypatch, "Hey Muse, what's the weather tomorrow?", muse)
    assert sent_to_muse(events) == ["what's the weather tomorrow?"]
    assert played(events) == ["tts:Good night.", "tts:Sunny and warm all day."]
    states = awake_after_sleep(rig.states())
    assert states[0] == "listening" and "speaking" in states


def test_speech_without_the_wake_phrase_leaves_reachy_asleep(monkeypatch):
    events, rig, transcripts = asleep_then(monkeypatch, "What's the weather tomorrow?", wake=False,
                                           rest=("sleeping",))
    assert sent_to_muse(events) == []
    assert played(events) == ["tts:Good night."]
    assert transcripts.texts == ["What's the weather tomorrow?"]
    assert set(after_sleep(rig.states())) == {"sleeping"}


def test_hey_muse_wake_up_wakes_reachy_once_with_one_greeting(monkeypatch):
    events, rig, _ = asleep_then(monkeypatch, "Hey Muse, wake up.")
    assert sent_to_muse(events) == []
    assert played(events) == ["tts:Good night.", "tts:I'm awake."]
    assert awake_after_sleep(rig.states())[0] == "listening"


@pytest.mark.parametrize("wake", ["Hey Muse.", "Hey Muse, wake up.", "arrival"])
def test_reconnecting_preserves_sleep_until_a_wake_or_morning_arrival(monkeypatch, wake):
    rig = rig_with(monkeypatch)
    camera = Camera()
    transcripts = Transcripts(GO_TO_SLEEP, *([] if wake == "arrival" else [wake]))
    conversation = service(rig, never_asked, transcripts, wake_detector=Wake(), wake_timeout_s=30)

    async def check():
        stop = asyncio.Event()
        disconnected = asyncio.Event()

        async def link(stop_link):
            await disconnected.wait()
            return Outcome.CLOSED

        rig.session.run = link
        first = asyncio.create_task(conversation.serve(rig.session, {"vm_id": "vm-1"}, stop))
        try:
            await rig.quiet()
            rig.say(wake=True)
            await rig.quiet("sleeping")
        finally:
            disconnected.set()
            assert await asyncio.wait_for(first, 5) is Outcome.CLOSED
            await asyncio.gather(*rig.session.scripts)

        assert rig.states()[-1] == "sleeping"
        camera_task = None
        if wake == "arrival":
            conversation.morning = morning(camera)
            camera_task = asyncio.create_task(conversation.morning.watch_camera())
        rig.session = PinSession(rig.recording, briefing if wake == "arrival" else never_asked)
        states_before = len(rig.states())
        second = asyncio.create_task(conversation.serve(rig.session, {"vm_id": "vm-1"}, stop))
        try:
            await rig.wait_until_or_timeout(lambda: len(rig.states()) > states_before, 5)
            await rig.quiet("sleeping", "idle", "error")
            assert set(rig.states()[states_before:]) == {"sleeping"}
            rig.say()
            await rig.quiet("sleeping")
            assert set(rig.states()[states_before:]) == {"sleeping"}
            assert played(rig.recording.events) == ["tts:Good night."]
            assert transcripts.texts == ([] if wake == "arrival" else [wake])
            if wake == "arrival":
                camera.arrive()
                await rig.wait_until_or_timeout(lambda: camera.calls == ["start", "stop"], 10)
            else:
                rig.say(wake=True)
            await rig.quiet("listening", "idle")
            assert played(rig.recording.events) == [
                "tts:Good night.", *(SPOKEN if wake == "arrival" else ["tts:I'm awake."])]
            assert awake_after_sleep(rig.states())[0] == "listening"
            assert sent_to_muse(rig.recording.events) == (["Brief me."] if wake == "arrival" else [])
            assert not second.done()
        finally:
            stop.set()
            await asyncio.wait_for(second, 5)
            await asyncio.gather(*rig.session.scripts)
            if camera_task is not None:
                camera_task.cancel()
                await asyncio.gather(camera_task, return_exceptions=True)

        stop.clear()
        rig.session = PinSession(rig.recording, never_asked)
        states_before = len(rig.states())
        third = asyncio.create_task(conversation.serve(rig.session, {"vm_id": "vm-1"}, stop))
        try:
            await rig.wait_until_or_timeout(lambda: len(rig.states()) > states_before, 5)
            await rig.quiet()
            assert set(rig.states()[states_before:]) == {"idle"}
        finally:
            stop.set()
            await asyncio.wait_for(third, 5)
            await asyncio.gather(*rig.session.scripts)

    asyncio.run(asyncio.wait_for(check(), 30))
