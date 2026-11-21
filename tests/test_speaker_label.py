"""The robot records the companion's speaker label on the turn's timeline and logs it in fixed words."""

import asyncio
import contextlib
import logging

import pytest
from websockets.asyncio.server import serve

from fake_companion import TOKEN, FakeCompanion
from reachy_muse.companion_client import CompanionHearing, CompanionLink
from reachy_muse.companion_server.engines import EchoNarrator, ScriptedHearing, ToneVoice
from reachy_muse.companion_server.owner_voice import PRINT_FILE, OwnerVoice, VoicePrint
from reachy_muse.companion_server.server import CompanionServer, Engines
from reachy_muse.companion_server.watchdog import EngineExecutor
from reachy_muse.protocol import (
    VERSION, NarrateOp, NarrateResult, Op, Sender, SpeakerLabel, SpeakerObservation, Welcome, decode, encode,
)
from reachy_muse.turn_timeline import TurnTimeline
from test_pins import Vad

np = pytest.importorskip("numpy")
pytest.importorskip("av")

MODEL = "a" * 64
OWNER_LEVEL, OTHER_LEVEL = .3, .7


class Encoder:
    """A quiet voice embeds like the owner's print, a loud one far from it."""

    model_id = MODEL
    dimension = 2

    def embed(self, samples, rate):
        return np.array([1.0, 0.0] if np.mean(np.abs(samples)) < .5 else [0.0, 1.0])


def enrolled(state):
    VoicePrint(MODEL, (1.0, 0.0)).save(state / PRINT_FILE)


@contextlib.asynccontextmanager
async def companion_engines(state, **engines):
    with EngineExecutor("speaker") as thread:
        speaker = OwnerVoice(Encoder(), state / PRINT_FILE, .74, name="fake-speaker", vad=Vad, thread=thread)
        async with CompanionServer(Engines(speaker=speaker, narrator=EchoNarrator(), **engines), TOKEN) as server:
            yield server.url.removesuffix("/v1")


@contextlib.asynccontextmanager
async def linked(url):
    link = CompanionLink(url, token=TOKEN, out_rate=16000, insecure=True, backoff_s=(5, 5))
    running = asyncio.create_task(link.run())
    try:
        await asyncio.wait_for(link.settled(), 2)
        yield link
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


async def spoken_turn(link, level):
    """Two seconds of one voice, then the silence that ends the turn on the robot."""
    turn = CompanionHearing(link).open_turn(16000, silence_s=.3, vad=Vad())
    turn.feed(np.full(32000, level, dtype=np.float32))
    turn.feed(np.zeros(8000, dtype=np.float32))
    ended = turn.take().ended
    assert ended is not None
    return ended


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(.01)


def speaker_logs(caplog):
    return [message for message in caplog.messages if message.startswith("Reachy speaker")]


def test_the_label_reaches_the_turns_timeline_facts_whichever_comes_first(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="reachy_muse.companion_client")
    enrolled(tmp_path)

    async def run():
        async with companion_engines(tmp_path, hearing=ScriptedHearing(("hello",))) as url, linked(url) as link:
            assert link.can(Op.SPEAKER)
            queued_first = await spoken_turn(link, OWNER_LEVEL)
            early = TurnTimeline()
            queued_first.note_speaker(early.facts)
            assert (await queued_first.endpoint()).text == "hello"
            await until(lambda: "speaker" in early.facts)
            labeled_first = await spoken_turn(link, OTHER_LEVEL)
            await labeled_first.endpoint()
            await until(lambda: "Reachy speaker=other" in caplog.messages)
            late = TurnTimeline()
            labeled_first.note_speaker(late.facts)
            return early.facts, late.facts

    assert asyncio.run(asyncio.wait_for(run(), 10)) == ({"speaker": "owner"}, {"speaker": "other"})
    assert speaker_logs(caplog) == ["Reachy speaker=owner", "Reachy speaker=other"]


@pytest.mark.parametrize("older", ["without a speaker model", "fake companion"])
def test_a_new_robot_hears_as_before_with_a_companion_that_never_grants_speaker(tmp_path, caplog, older):
    caplog.set_level(logging.INFO, logger="reachy_muse.companion_client")

    async def run():
        async with contextlib.AsyncExitStack() as stack:
            if older == "fake companion":
                url = (await stack.enter_async_context(FakeCompanion())).url.removesuffix("/v1")
            else:
                server = await stack.enter_async_context(CompanionServer(
                    Engines(hearing=ScriptedHearing(("hello",)), narrator=EchoNarrator()), TOKEN))
                url = server.url.removesuffix("/v1")
            link = await stack.enter_async_context(linked(url))
            assert (link.can(Op.HEAR), link.can(Op.SPEAKER)) == (True, False)
            ended = await spoken_turn(link, OWNER_LEVEL)
            timeline = TurnTimeline()
            ended.note_speaker(timeline.facts)
            heard = (await ended.endpoint()).text
            assert (await link.narrate(NarrateOp.ACKNOWLEDGE, "Barrier.")).lines[0].text == "Barrier."
            return heard, timeline.facts, link.up

    assert asyncio.run(asyncio.wait_for(run(), 10)) == ("hello", {}, True)
    assert speaker_logs(caplog) == []


def test_a_label_for_a_stream_the_robot_is_not_waiting_on_is_ignored(caplog):
    caplog.set_level(logging.INFO, logger="reachy_muse.companion_client")

    async def companion(ws):
        await ws.recv()
        await ws.send(encode(Welcome(VERSION, (Op.HEAR, Op.NARRATE, Op.SPEAKER), ())))
        await ws.send(encode(SpeakerObservation(99, 0, 16000, SpeakerLabel.OWNER)))
        async for frame in ws:
            request = decode(frame, sender=Sender.ROBOT)
            await ws.send(encode(NarrateResult(request.id, ())))

    async def run():
        async with serve(companion, "127.0.0.1", 0) as server, \
                linked(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}") as link:
            await link.narrate(NarrateOp.ACKNOWLEDGE, "Barrier.")
            return link.up

    assert asyncio.run(asyncio.wait_for(run(), 10)) is True
    assert speaker_logs(caplog) == []


def test_a_whole_turn_logs_its_speaker_in_the_turn_timeline(monkeypatch, tmp_path, caplog):
    from test_companion import Conversation, answer, scenario

    caplog.set_level(logging.INFO)
    enrolled(tmp_path)

    async def run():
        hearing = ScriptedHearing(("tell me about castles",), frames_per_partial=12)
        async with companion_engines(tmp_path, hearing=hearing, voice=ToneVoice()) as url:
            robot = Conversation(monkeypatch, url, answer("Castles had tall walls."))

            async def user():
                await robot.speak(1.5)
                await robot.until(lambda: robot.session.scripts and robot.session.scripts[-1].done(), 15)

            await robot.run(user)
            assert robot.requests() == ["tell me about castles"]

    scenario(run())
    timelines = [message for message in caplog.messages if message.startswith("Reachy turn timeline ")]
    assert len(timelines) == 1
    facts = dict(field.split("=") for field in timelines[0].split()[3:])
    assert (facts["forced"], facts["speaker"]) == ("yes", "owner")
    assert speaker_logs(caplog) == ["Reachy speaker=owner"]
