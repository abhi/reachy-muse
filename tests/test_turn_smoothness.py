"""Whole turns against a companion that is alive but busy, as on November 5 at 06:11.

Its pongs keep arriving, so the link stays up, but its hearing never ends the turn and its
narration answers late. The robot must still end the turn at its own silence endpoint and
let the user hear that it is working.
"""

import asyncio

import pytest

from fake_companion import FakeCompanion
from reachy_muse.companion_client import LinkState
from reachy_muse.protocol import NarrateOp
from test_companion import PARTIALS, QUESTION, Conversation, answer, hear_streams_opened, scenario, spoken_by_companion
from test_pins import ScriptedStreamingTranscriber, Transcripts

pytest.importorskip("numpy")
pytest.importorskip("av")


def test_a_companion_that_never_endpoints_cannot_hold_the_turn_open(monkeypatch):
    async def run():
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=12, no_endpoint=True) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3], answer("Castles kept dragons out."),
                                 transcript="what the robot heard", ping_interval_s=.05, ping_timeout_s=.2)
            waited, transitions = [], []

            async def user():
                for turn in (1, 2):
                    await robot.speak(.8)
                    stopped = asyncio.get_running_loop().time()
                    await robot.until(lambda: len(robot.requests()) == turn, 10)
                    waited.append(asyncio.get_running_loop().time() - stopped)
                    await robot.quiet()
                transitions.extend(robot.link.transitions)
            await robot.run(user)
            # The robot's .2 s silence endpoint, the 1.5 s endpoint grace, and scheduling.
            assert all(1.5 < wait < 2.5 for wait in waited), waited
            assert robot.requests() == ["what the robot heard"] * 2
            assert hear_streams_opened(companion) == [1, 3]
            assert transitions == [LinkState.CONNECTING, LinkState.UP]
            assert robot.spoken() == ["companion", "companion"]
    scenario(run())



def test_a_late_companion_hands_its_audio_to_muse_when_the_robot_only_streams(monkeypatch):
    async def run():
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=12, no_endpoint=True) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3], answer("Castles kept dragons out."),
                                 transcriber=ScriptedStreamingTranscriber(), ping_interval_s=.05, ping_timeout_s=.2)

            async def user():
                await robot.speak(.8)
                await robot.until(lambda: robot.session.scripts, 5)
            await robot.run(user)
            assert [event["muse"] for event in robot.recording.events if "muse" in event][-1:] == ["send_voice"]
            assert robot.spoken() == ["companion"]
    scenario(run())


class StreamingRecognizer(ScriptedStreamingTranscriber, Transcripts):
    """Streams live turns and transcribes a finished recording, as Sherpa does."""

    def __init__(self, *texts):
        ScriptedStreamingTranscriber.__init__(self)
        Transcripts.__init__(self, *texts)


def test_a_cancel_heard_past_a_late_companion_still_stops_muse(monkeypatch):
    async def muse(session, request):
        if request == 1:
            while session.requests < 2:
                await asyncio.sleep(.01)
            await session.answer(request, "Robots learned to fold laundry.")
        else:
            await session.answer(request, "Okay, I stopped.")

    async def run():
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=12, no_endpoint=True) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3], muse,
                                 transcriber=StreamingRecognizer("Can you search for the latest robot news?",
                                                                 "Never mind."),
                                 ping_interval_s=.05, ping_timeout_s=.2)

            async def user():
                await robot.speak(.8)
                await robot.until(lambda: robot.requests(), 5)
                await robot.speak(.8)
                await robot.until(lambda: len(robot.requests()) == 2, 5)
            await robot.run(user)
            assert robot.requests() == ["Can you search for the latest robot news?",
                                        "Please stop working on my previous request."]
            assert [event["muse"] for event in robot.recording.events
                    if event.get("muse") in ("send_chat", "send_voice")] == ["send_chat", "send_chat"]
    scenario(run())


def test_late_acknowledgement_narration_still_acknowledges_the_request(monkeypatch):
    async def run():
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=12,
                                 narrate_delay_s={NarrateOp.ACKNOWLEDGE: 3.0}) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3], answer("Castles kept dragons out.", 2.5))
            await robot.run(lambda: robot.turns(1))
            assert robot.requests() == [QUESTION]
            assert robot.spoken() == ["tts:Hmm, good question.", "companion"]
            assert spoken_by_companion(companion) == ["Castles kept dragons out."]
    scenario(run())


def test_reachy_nods_when_the_user_pauses_before_any_words_arrive(monkeypatch):
    async def run():
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=10_000) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3], answer("Castles kept dragons out."), silence_s=1.0)
            moved = []

            def motion():
                return [event.get("name") or event["state"] for event in robot.recording.events
                        if event.get("robot") in ("gesture", "set_state")]

            async def user():
                await robot.speak(.8)
                stopped = asyncio.get_running_loop().time()
                await robot.until(lambda: "nod" in motion(), 2)
                moved.append(asyncio.get_running_loop().time() - stopped)
                await robot.until(lambda: robot.requests(), 3)
            await robot.run(user)
            assert .5 < moved[0] < 1.0, moved
            assert motion()[:4] == ["idle", "listening", "nod", "thinking"]
    scenario(run())


def test_a_neutral_narration_still_lets_each_sentence_show_its_feeling(monkeypatch):
    from reachy_muse.contract.expressions import Expression

    reply = "Great news, the castle is open. Sadly the moat is closed. Would you like directions?"

    async def run():
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=12, expression=Expression.NEUTRAL) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3], answer(reply))
            await robot.run(lambda: robot.turns(1))
            assert spoken_by_companion(companion) == [
                "Great news, the castle is open.", "Sadly the moat is closed.", "Would you like directions?"]
            assert [event["expression"] for event in robot.recording.events
                    if event.get("state") == "speaking"] == ["happy", "sad", "curious"]
    scenario(run())


def test_a_companion_voice_that_never_starts_cannot_hold_the_answer(monkeypatch):
    import reachy_muse.companion_client

    monkeypatch.setattr(reachy_muse.companion_client, "SPEECH_STALL_S", .5)

    async def run():
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=12, mute=True) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3], answer("Castles kept dragons out."),
                                 ping_interval_s=.05, ping_timeout_s=.2)
            transitions = []

            async def user():
                await robot.speak(.8)
                await robot.until(lambda: robot.spoken(), 3)
                transitions.extend(robot.link.transitions)
            await robot.run(user)
            assert spoken_by_companion(companion) == ["Castles kept dragons out."]
            assert robot.spoken() == ["tts:Castles kept dragons out."]
            assert transitions == [LinkState.CONNECTING, LinkState.UP]
    scenario(run())
