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
from test_pins import ScriptedStreamingTranscriber

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


def test_late_acknowledgement_narration_still_acknowledges_the_request(monkeypatch):
    async def run():
        async with FakeCompanion(partials=PARTIALS, frames_per_partial=12,
                                 narrate_delay_s={NarrateOp.ACKNOWLEDGE: 3.0}) as companion:
            robot = Conversation(monkeypatch, companion.url[:-3], answer("Castles kept dragons out.", 2.5))
            await robot.run(lambda: robot.turns(1))
            assert robot.requests() == [QUESTION]
            assert spoken_by_companion(companion) == ["Hmm, good question.", "Castles kept dragons out."]
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
            assert motion()[:3] == ["idle", "listening", "nod"]
    scenario(run())
