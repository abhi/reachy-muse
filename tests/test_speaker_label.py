"""The robot records the companion's speaker label on the turn's timeline and logs it in fixed words."""

import asyncio
import contextlib
import logging
import sys
import threading
from types import SimpleNamespace

import pytest
from websockets.asyncio.server import serve

from fake_companion import TOKEN, FakeCompanion
from reachy_muse.companion_client import CompanionHearing, CompanionLink
from reachy_muse.companion_server import hearing_voxtral
from reachy_muse.companion_server.engines import EchoNarrator, ScriptedHearing, ToneVoice
from reachy_muse.companion_server.owner_voice import PRINT_FILE, OwnerVoice, VoicePrint, _speaker_thread
from reachy_muse.companion_server.server import CompanionServer, Engines
from reachy_muse.protocol import (
    VERSION, Error, Narrate, NarrateOp, NarrateResult, Op, Sender, SpeakerLabel, SpeakerObservation, Welcome, decode,
    encode,
)
from reachy_muse.turn_timeline import TurnTimeline
from test_pins import Vad

np = pytest.importorskip("numpy")
pytest.importorskip("av")

MODEL = "a" * 64
OWNER_LEVEL, OTHER_LEVEL = .3, .7


class Encoder:
    """A quiet voice embeds like the owner's print, a loud one far from it. ``gate`` holds each check."""

    model_id = MODEL
    dimension = 2

    def __init__(self, gate=None):
        self.gate = gate
        self.entered, self.done = threading.Event(), threading.Event()

    def embed(self, samples, rate):
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(5)
        self.done.set()
        return np.array([1.0, 0.0] if np.mean(np.abs(samples)) < .5 else [0.0, 1.0])


class RecordingLink(CompanionLink):
    """The robot's real link, keeping every message the companion sent, in order."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received = []

    def _receive(self, connection, frame):
        if isinstance(frame, str):
            self.received.append(decode(frame, sender=Sender.COMPANION))
        super()._receive(connection, frame)


def enrolled(state):
    VoicePrint(MODEL, (1.0, 0.0)).save(state / PRINT_FILE)


@contextlib.asynccontextmanager
async def companion_engines(state, encoder=None, **engines):
    with _speaker_thread() as thread:
        speaker = OwnerVoice(encoder or Encoder(), state / PRINT_FILE, .74, name="fake-speaker", vad=Vad, thread=thread)
        async with CompanionServer(Engines(speaker=speaker, narrator=EchoNarrator(), **engines), TOKEN) as server:
            yield server.url.removesuffix("/v1")


@contextlib.asynccontextmanager
async def linked(url, link_type=CompanionLink):
    link = link_type(url, token=TOKEN, out_rate=16000, insecure=True, backoff_s=(5, 5))
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


def test_a_turn_queued_before_its_label_gets_it_when_it_comes(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="reachy_muse.companion_client")
    enrolled(tmp_path)

    async def run():
        facts = []
        async with companion_engines(tmp_path, hearing=ScriptedHearing(("hello",))) as url, linked(url) as link:
            assert link.can(Op.SPEAKER)
            for level in (OWNER_LEVEL, OTHER_LEVEL):
                ended = await spoken_turn(link, level)
                timeline = TurnTimeline()
                ended.note_speaker(timeline.facts)
                assert (await ended.endpoint()).text == "hello"
                await until(lambda: "speaker" in timeline.facts)
                facts.append(timeline.facts)
        return facts

    assert asyncio.run(asyncio.wait_for(run(), 10)) == [{"speaker": "owner"}, {"speaker": "other"}]
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


def test_a_label_still_in_flight_when_the_robot_aborts_reaches_the_turn(caplog):
    """A companion sends nothing once it has handled an abort, so a label arriving later left before it."""
    from reachy_muse.protocol import CloseHow, HearClose, HearEndpoint

    caplog.set_level(logging.INFO, logger="reachy_muse.companion_client")

    async def companion(ws):
        await ws.recv()
        await ws.send(encode(Welcome(VERSION, (Op.HEAR, Op.NARRATE, Op.SPEAKER), ())))
        async for frame in ws:
            if isinstance(frame, bytes):
                continue
            request = decode(frame, sender=Sender.ROBOT)
            if isinstance(request, HearClose) and request.how is CloseHow.FINISH:
                await ws.send(encode(HearEndpoint(request.stream, "hello", True)))
            elif isinstance(request, HearClose):
                await ws.send(encode(SpeakerObservation(request.stream, 0, 32000, SpeakerLabel.OWNER)))
            elif isinstance(request, Narrate):
                await ws.send(encode(NarrateResult(request.id, ())))

    async def run():
        async with serve(companion, "127.0.0.1", 0) as server, \
                linked(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}") as link:
            ended = await spoken_turn(link, OWNER_LEVEL)
            timeline = TurnTimeline()
            ended.note_speaker(timeline.facts)
            await ended.endpoint()
            ended.cancel()
            await link.narrate(NarrateOp.ACKNOWLEDGE, "Barrier.")
            return timeline.facts

    assert asyncio.run(asyncio.wait_for(run(), 10)) == {"speaker": "owner"}
    assert speaker_logs(caplog) == ["Reachy speaker=owner"]


def test_a_call_command_cut_off_where_muse_s_reply_began_keeps_its_label_and_final_words(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="reachy_muse.companion_client")
    enrolled(tmp_path)

    async def run():
        async with companion_engines(tmp_path, hearing=ScriptedHearing(("stop listening",))) as url, \
                linked(url) as link:
            turn = CompanionHearing(link).open_turn(16000, silence_s=.3, vad=Vad(), commands_only=True)
            turn.feed(np.full(32000, OWNER_LEVEL, dtype=np.float32))
            assert turn.take().ended is None
            turn.feed(np.full(320, OWNER_LEVEL, dtype=np.float32), turn_ends_at=0)
            ended = turn.take().ended
            timeline = TurnTimeline()
            ended.note_speaker(timeline.facts)
            endpoint = await ended.endpoint()
            await until(lambda: "speaker" in timeline.facts)
            turn.abort()
            return endpoint.text, endpoint.forced, timeline.facts

    assert asyncio.run(asyncio.wait_for(run(), 10)) == ("stop listening", True, {"speaker": "owner"})
    assert speaker_logs(caplog) == ["Reachy speaker=owner"]


# ---- realtime calls: the robot aborts each command stream at its own end of speech


@pytest.fixture
def final_pass(monkeypatch):
    """Parakeet stand-in for the real Voxtral session, held until the test sets it."""
    held = threading.Event()
    monkeypatch.setitem(sys.modules, "mlx.core", SimpleNamespace(array=lambda value: value))
    monkeypatch.setitem(sys.modules, "parakeet_mlx.audio", SimpleNamespace(get_logmel=lambda audio, config: audio))
    try:
        yield held
    finally:
        held.set()


def voxtral(final_pass):
    """The real hearing session and its endpointing, on fake models: speech is any nonzero sample, and Smart
    Turn calls a turn complete at its 1.2 s check, so the companion ends speech after 1.2 s of silence."""

    class Stream:
        spoken = False

        def feed(self, samples):
            pass

        def step(self, *, max_decode_tokens):
            spoken, self.spoken = self.spoken, True
            return [] if spoken else ["stop listening"]

    def generate(mel):
        assert final_pass.wait(5)
        return [SimpleNamespace(text="stop listening")]

    def speech(_, inputs):
        return np.array([[float(np.any(inputs["input"][0, hearing_voxtral.VAD_CONTEXT:]))]]), inputs["state"]

    hearing = hearing_voxtral.VoxtralHearing(hearing_voxtral.HearingConfig(turn_check_after_s=1.2))
    hearing._models = SimpleNamespace(
        voxtral=SimpleNamespace(create_streaming_session=lambda **kwargs: Stream()),
        parakeet=SimpleNamespace(generate=generate, preprocessor_config=None),
        vad=SimpleNamespace(run=speech),
        turn=SimpleNamespace(run=lambda *args: (np.array([[1.0]]),)),
        features=lambda *args, **kwargs: SimpleNamespace(input_features=np.zeros((1, 80, 800))))
    return hearing


def seconds(value, level=0.0):
    return np.full(round(value * 16000), level, dtype=np.float32)


def under(link, stream=1):
    return [type(message).__name__ for message in link.received if getattr(message, "stream", None) == stream]


@pytest.mark.parametrize("commands_only", [True, False], ids=["call", "chat"])
def test_the_label_reaches_the_turn_at_the_companions_end_of_speech_before_the_robots(tmp_path, caplog, final_pass,
                                                                                    commands_only):
    """The companion ends speech after 1.2 s of silence and the robot after 2.0 s. In a call the robot then aborts;
    in chat it finishes. Either way the label is on the turn first, while Parakeet still holds the endpoint."""
    caplog.set_level(logging.INFO, logger="reachy_muse.companion_client")
    enrolled(tmp_path)
    hearing = voxtral(final_pass)

    async def run():
        async with companion_engines(tmp_path, hearing=hearing) as url, linked(url, RecordingLink) as link:
            turn = CompanionHearing(link).open_turn(16000, silence_s=2.0, vad=Vad(), commands_only=commands_only)
            for audio in (seconds(2, OWNER_LEVEL), seconds(1.3)):
                turn.feed(audio)
                assert turn.take().ended is None
            await until(lambda: "SpeakerObservation" in under(link))
            turn.feed(seconds(.8))
            ended = turn.take().ended
            timeline = TurnTimeline()
            ended.note_speaker(timeline.facts)
            at_the_robots_end = dict(timeline.facts)
            await link.narrate(NarrateOp.ACKNOWLEDGE, "Barrier.")
            final_pass.set()
            words = (await ended.endpoint()).text
            await link.narrate(NarrateOp.ACKNOWLEDGE, "Barrier.")
            return at_the_robots_end, words, under(link)

    try:
        assert asyncio.run(asyncio.wait_for(run(), 10)) == (
            {"speaker": "owner"}, "stop listening",
            ["HearPartial", "SpeakerObservation"] + ([] if commands_only else ["HearEndpoint"]))
    finally:
        final_pass.set()
        hearing._thread.shutdown()
    assert speaker_logs(caplog) == ["Reachy speaker=owner"]


def test_a_robot_abort_before_the_companions_end_of_speech_gets_no_label_and_no_error(tmp_path, caplog, final_pass):
    caplog.set_level(logging.INFO, logger="reachy_muse.companion_client")
    enrolled(tmp_path)
    hearing = voxtral(final_pass)

    async def run():
        async with companion_engines(tmp_path, hearing=hearing) as url, linked(url, RecordingLink) as link:
            turn = CompanionHearing(link).open_turn(16000, silence_s=.5, vad=Vad(), commands_only=True)
            turn.feed(seconds(2, OWNER_LEVEL))
            assert turn.take().ended is None
            await until(lambda: "HearPartial" in under(link))
            turn.feed(seconds(.6))
            ended = turn.take().ended
            timeline = TurnTimeline()
            ended.note_speaker(timeline.facts)
            await link.narrate(NarrateOp.ACKNOWLEDGE, "Barrier.")
            final_pass.set()
            await asyncio.sleep(.2)
            await link.narrate(NarrateOp.ACKNOWLEDGE, "Barrier.")
            errors = [message for message in link.received if isinstance(message, Error)]
            return timeline.facts, under(link), errors, link.up

    try:
        assert asyncio.run(asyncio.wait_for(run(), 10)) == ({}, ["HearPartial"], [], True)
    finally:
        final_pass.set()
        hearing._thread.shutdown()
    assert speaker_logs(caplog) == []


def test_an_abort_while_the_speaker_check_runs_sends_nothing_after_it(tmp_path, caplog, final_pass):
    caplog.set_level(logging.INFO, logger="reachy_muse.companion_client")
    enrolled(tmp_path)
    check = threading.Event()
    encoder = Encoder(check)
    hearing = voxtral(final_pass)

    async def run():
        async with companion_engines(tmp_path, encoder, hearing=hearing) as url, linked(url, RecordingLink) as link:
            turn = CompanionHearing(link).open_turn(16000, silence_s=2.0, vad=Vad(), commands_only=True)
            for audio in (seconds(2, OWNER_LEVEL), seconds(1.3)):
                turn.feed(audio)
                assert turn.take().ended is None
            assert await asyncio.to_thread(encoder.entered.wait, 2)
            turn.feed(seconds(.8))
            ended = turn.take().ended
            timeline = TurnTimeline()
            ended.note_speaker(timeline.facts)
            await link.narrate(NarrateOp.ACKNOWLEDGE, "Barrier.")
            check.set()
            final_pass.set()
            assert await asyncio.to_thread(encoder.done.wait, 2)
            await asyncio.sleep(.2)
            await link.narrate(NarrateOp.ACKNOWLEDGE, "Barrier.")
            return timeline.facts, under(link)

    try:
        assert asyncio.run(asyncio.wait_for(run(), 10)) == ({}, ["HearPartial"])
    finally:
        check.set()
        final_pass.set()
        hearing._thread.shutdown()
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
