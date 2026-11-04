"""The failover wrappers against scripted companion and on-robot parts."""

import asyncio
import io
import wave

import pytest

import reachy_muse.failover
from reachy_muse.contract.intent import Intent
from reachy_muse.companion_client import CompanionOpening
from reachy_muse.capabilities import Heard, Hold, CapabilityUnavailable, HeardAudio, SpokenLine
from reachy_muse.companion_client import (
    CompanionHearing, CompanionHearingTurn, CompanionLink, LinkState, _Connection, _HearStream,
)
from reachy_muse.protocol import HearEndpoint, Op, encode
from reachy_muse.failover import CompanionRoute, FailoverHearing, FailoverNarrator
from reachy_muse.local_backends import LocalHearing
from reachy_muse.voice_audio import SpeechAudio, SpeechEnd

np = pytest.importorskip("numpy")


class EndedLate:
    truncated = False

    def __init__(self, end_sample):
        self.end_sample = end_sample


class ScriptedTurn:
    """A hearing turn whose takes are scripted: a HeardAudio to return or an error to raise."""

    active = True
    speech_active = True
    partial = None

    def __init__(self, takes):
        self.fed = []
        self._takes = list(takes)

    def feed(self, samples):
        self.fed.append(samples.copy())

    def take(self):
        outcome = self._takes.pop(0) if self._takes else HeardAudio()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def abort(self):
        pass


class ScriptedHearing:
    def __init__(self, *turns, available=True):
        self.available = available
        self.turns = list(turns)
        self.opened = []

    def open_turn(self, sample_rate, **options):
        self.opened.append(self.turns.pop(0))
        return self.opened[-1]


def test_a_late_endpoint_keeps_the_next_utterances_opening_for_a_replay():
    # The first utterance ends at sample 500; its endpoint arrives after 1000 samples were fed.
    companion = ScriptedHearing(ScriptedTurn([HeardAudio(EndedLate(500)), CapabilityUnavailable("gone")]))
    local = ScriptedHearing(ScriptedTurn([]))
    turn = FailoverHearing(companion, local, CompanionRoute(Link()).hearing).open_turn(16000)
    for chunk in range(10):
        turn.feed(np.full(100, chunk, dtype=np.float32))
    assert turn.take().ended is not None
    assert turn.take().ended is None
    turn.feed(np.full(100, 10, dtype=np.float32))
    replayed, = local.opened[0].fed
    assert replayed.tolist() == [float(n // 100) for n in range(500, 1100)]


@pytest.mark.parametrize("opening_samples", [160, 320, 640])
@pytest.mark.parametrize("silence_samples", [0, 11200])
def test_a_connecting_companion_preserves_the_opening_before_vad_activation(opening_samples, silence_samples):
    pytest.importorskip("av")

    class Vad:
        def is_speech(self, pcm, rate):
            return bool(np.any(np.frombuffer(pcm, dtype="<i2")))

    link = CompanionLink("ws://localhost:9", token="test", out_rate=16000, insecure=True)
    hearing = FailoverHearing(CompanionHearing(link), LocalHearing(), CompanionRoute(link).hearing)
    turn = hearing.open_turn(16000, silence_s=.1, vad=Vad())
    turn.feed(np.zeros(silence_samples, dtype=np.float32))
    turn.feed(np.full(opening_samples, .25, dtype=np.float32))
    assert turn.take().ended is None
    assert link.state is LinkState.CONNECTING
    link._connection = _Connection(None, (Op.HEAR,))
    link._set(LinkState.UP)
    assert turn.take().ended is None
    turn.feed(np.concatenate((np.full(640 - opening_samples, .25, dtype=np.float32),
                              np.full(7360, .5, dtype=np.float32))))
    assert turn.take().ended is None
    turn.feed(np.zeros(1600, dtype=np.float32))
    ended = turn.take().ended
    if ended is None:
        stream, = link._connection.hearing
        link._receive(link._connection, encode(HearEndpoint(stream, "first", True)))
        ended = turn.take().ended
    endpoint = asyncio.run(ended.endpoint())
    with wave.open(io.BytesIO(endpoint.audio), "rb") as wav:
        recorded = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    pre_roll = [0] * 2240 if silence_samples else []
    np.testing.assert_array_equal(recorded, pre_roll + [8192] * 640 + [16384] * 7360 + [0] * 1600)

    assert turn.take().ended is None
    turn.feed(np.full(8000, .75, dtype=np.float32))
    assert turn.take().ended is None
    stream, = link._connection.hearing.values()
    assert bytes(stream.audio) == b"\x00\x60" * 8000
    turn.abort()


@pytest.mark.parametrize("ended_first_turn", [False, True])
@pytest.mark.parametrize("ended_local_turn", [False, True])
def test_a_disconnect_after_handoff_replays_all_unfinished_audio(ended_first_turn, ended_local_turn):
    pytest.importorskip("av")

    class Vad:
        def is_speech(self, pcm, rate):
            return bool(np.any(np.frombuffer(pcm, dtype="<i2")))

    link = CompanionLink("ws://localhost:9", token="test", out_rate=16000, insecure=True)
    hearing = FailoverHearing(CompanionHearing(link), LocalHearing(), CompanionRoute(link).hearing)
    turn = hearing.open_turn(16000, silence_s=.1, vad=Vad())
    if ended_local_turn:
        turn.feed(np.full(8000, .75, dtype=np.float32))
        assert turn.take().ended is None
        turn.feed(np.zeros(1600, dtype=np.float32))
        assert turn.take().ended is not None
    turn.feed(np.zeros(11200, dtype=np.float32))
    turn.feed(np.full(160, .25, dtype=np.float32))
    assert turn.take().ended is None
    assert link.state is LinkState.CONNECTING
    link._connection = _Connection(None, (Op.HEAR,))
    link._set(LinkState.UP)
    assert turn.take().ended is None
    turn.feed(np.concatenate((np.full(480, .25, dtype=np.float32),
                              np.full(7360, .5, dtype=np.float32))))
    assert turn.take().ended is None
    stream, = link._connection.hearing.values()
    expected = [0] * 2240 + [8192] * 640 + [16384] * 7360
    np.testing.assert_array_equal(np.frombuffer(stream.audio, dtype="<i2"), expected)
    if ended_first_turn:
        turn.feed(np.zeros(1600, dtype=np.float32))
        assert turn.take().ended is None
        turn.feed(np.full(8000, .75, dtype=np.float32))
        assert turn.take().ended is None
        link._receive(link._connection, encode(HearEndpoint(stream.id, "first", True)))
        assert turn.take().ended is not None
        expected = [24576] * 8000
    link._connection.fail("disconnected")
    link._set(LinkState.DOWN)
    assert turn.take().ended is None
    turn.feed(np.zeros(1920, dtype=np.float32))
    ended = turn.take().ended
    assert ended is not None
    endpoint = asyncio.run(ended.endpoint())
    with wave.open(io.BytesIO(endpoint.audio), "rb") as wav:
        recorded = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert np.count_nonzero(recorded == 8192) == (0 if ended_first_turn else 640)
    np.testing.assert_array_equal(recorded, expected + [0] * 1600)
    turn.abort()


class Connection:
    closed = False
    probe = None

    def __init__(self):
        self.hearing = {}

    def send(self, frame):
        pass


class HearingLink:
    def __init__(self):
        self.connection = Connection()
        self._next = -1

    def open_hear(self):
        self._next += 2
        stream = self.connection.hearing[self._next] = _HearStream(self.connection, self._next)
        return stream


class Recorder:
    active = False
    speech_active = False
    quiet_s = 0.0
    total_quiet_s = 0.0
    last_truncated = False

    def __init__(self, *events):
        self._events = list(events)

    def take_audio_events(self):
        return self._events.pop(0) if self._events else []

    def feed(self, samples):
        return None

    def reset(self):
        pass


def test_an_ended_turn_waiting_behind_another_is_returned_before_the_companion_fails():
    voice = SpeechAudio(b"\0\0" * 160)
    link = HearingLink()
    turn = CompanionHearingTurn(link, Recorder([voice, SpeechEnd(True), voice, SpeechEnd(True)], [voice]))
    assert turn.take().ended is None
    link.connection.hearing[1].inbox.append(HearEndpoint(1, "first", True))
    link.connection.hearing[3].inbox.append(HearEndpoint(3, "second", True))
    first = turn.take().ended
    link.connection.hearing[5].failure = CapabilityUnavailable("gone", link_down=False)
    second = turn.take().ended

    async def texts():
        return [(await ended.endpoint()).text for ended in (first, second)]
    assert asyncio.run(texts()) == ["first", "second"]
    with pytest.raises(CapabilityUnavailable, match="gone"):
        turn.take()


class Link:
    up = True


class InventingNarrator:
    available = True

    def __init__(self):
        self.asked = 0

    async def acknowledge(self, request):
        self.asked += 1
        return CompanionOpening(Intent.QUESTION, (SpokenLine("It took 17 minutes.", None, "ack"),))


class RobotNarrator:
    def hold(self, intent, heard):
        return Hold(intent, SpokenLine("Okay.", None, "ack"))


def test_a_rested_narrator_is_asked_again_after_the_cooldown(monkeypatch):
    monkeypatch.setattr(reachy_muse.failover, "COOLDOWN_S", .2)
    companion = InventingNarrator()
    route = CompanionRoute(Link())
    route.begin_turn()
    narrator = FailoverNarrator(companion, RobotNarrator(), route)

    async def run():
        said = [(await narrator.acknowledge(Heard("castles"))).line.text for _ in range(4)]
        await asyncio.sleep(.3)
        said.append((await narrator.acknowledge(Heard("castles"))).line.text)
        return said
    assert asyncio.run(run()) == ["Okay."] * 5
    assert companion.asked == 4


def test_rested_hearing_stays_on_the_robot_and_is_tried_again_after_the_cooldown(monkeypatch):
    monkeypatch.setattr(reachy_muse.failover, "COOLDOWN_S", .2)
    gone = CapabilityUnavailable("the companion did not end the turn", link_down=False)
    lost = CapabilityUnavailable("the link dropped")
    companion = ScriptedHearing(*(ScriptedTurn([outcome]) for outcome in (gone, lost, gone, gone)), ScriptedTurn([]))
    local = ScriptedHearing(*(ScriptedTurn([]) for _ in range(6)))
    hearing = FailoverHearing(companion, local, CompanionRoute(Link()).hearing)

    def window():
        hearing.open_turn(16000).take()
        return (len(companion.opened), len(local.opened))

    async def run():
        heard = [window() for _ in range(5)]
        await asyncio.sleep(.3)
        return heard + [window()]
    # A lost link is no strike, so the third failure that rests hearing is the fourth window's. The
    # fifth window is heard on the robot alone, and after the cooldown the companion hears again.
    assert asyncio.run(run()) == [(1, 1), (2, 2), (3, 3), (4, 4), (4, 5), (5, 5)]


def test_a_dead_link_restarts_the_sentence_and_latches_that_reply_only():
    from reachy_muse.failover import FailoverVoice
    from reachy_muse.local_speech import SpeechRestart

    class Companion:
        available = True

        async def stream(self, line, output_rate):
            yield np.array([1], dtype=np.float32)
            if line.text == "First sentence.":
                raise CapabilityUnavailable("connection lost")
            yield np.array([2], dtype=np.float32)

    class Local:
        async def stream(self, line, output_rate):
            yield np.array({"First sentence.": [10, 11], "Second sentence.": [20, 21]}[line.text],
                           dtype=np.float32)

    async def run():
        route = CompanionRoute(Link())
        route.begin_turn()
        voice = FailoverVoice(Companion(), Local(), route)
        first = voice.stream(SpokenLine("First sentence.", None, "answer", "reply-a"), 16000)
        second = voice.stream(SpokenLine("Second sentence.", None, "answer", "reply-a"), 16000)
        route.begin_turn()
        chunks = []
        for stream in (first, second):
            chunks.append(["restart" if isinstance(chunk, SpeechRestart) else chunk.tolist()
                           async for chunk in stream])
        next_reply = voice.stream(SpokenLine("Next reply.", None, "answer", "reply-b"), 16000)
        chunks.append([chunk.tolist() async for chunk in next_reply])
        return chunks

    assert asyncio.run(run()) == [[[1.0], "restart", [10.0, 11.0]], [[20.0, 21.0]], [[1.0], [2.0]]]


def test_a_slow_companion_chunk_is_never_replaced_with_local_audio():
    from reachy_muse.failover import FailoverVoice

    class Companion:
        available = True

        async def stream(self, line, output_rate):
            yield np.array([1], dtype=np.float32)
            await asyncio.sleep(2.1)
            yield np.array([2], dtype=np.float32)

    class Local:
        async def stream(self, line, output_rate):
            yield np.array([9], dtype=np.float32)

    async def run():
        route = CompanionRoute(Link())
        route.begin_turn()
        voice = FailoverVoice(Companion(), Local(), route)
        return [chunk.tolist() async for chunk in voice.stream(SpokenLine("Slow sentence.", None, "answer"), 16000)]

    assert asyncio.run(run()) == [[1.0], [2.0]]


def test_a_live_companion_error_does_not_silently_change_voice():
    from reachy_muse.failover import FailoverVoice

    class Companion:
        available = True

        async def stream(self, line, output_rate):
            if line.text == "Rejected.":
                raise CapabilityUnavailable("overloaded", link_down=False)
            yield np.array([1, 2], dtype=np.float32)

    class Local:
        async def stream(self, line, output_rate):
            yield np.array([9], dtype=np.float32)

    async def run():
        route = CompanionRoute(Link())
        route.begin_turn()
        voice = FailoverVoice(Companion(), Local(), route)
        with pytest.raises(CapabilityUnavailable, match="overloaded"):
            _ = [chunk async for chunk in voice.stream(SpokenLine("Rejected.", None, "answer"), 16000)]
        route.begin_turn()
        return [chunk.tolist() async for chunk in voice.stream(SpokenLine("Accepted.", None, "answer"), 16000)]

    assert asyncio.run(run()) == [[1.0, 2.0]]
