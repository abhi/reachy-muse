"""Muse realtime voice calls driven through Reachy's whole conversation, with a scripted call."""

import asyncio
import time

import pytest

from reachy_muse.realtime_call import VoiceLink, realtime_supported
from test_device_command_turns import SpeakerHardware, never_asked, played, run, sent_to_muse, service
from test_pins import CHUNK, VOICE_SAMPLE, WAKE_SAMPLE, FailingStreamingTranscriber, Rig, Transcripts, Wake

np = pytest.importorskip("numpy")
pytest.importorskip("av")
pytestmark = pytest.mark.skipif(not realtime_supported(), reason="the Muse SDK has no voice call events")

RATE = 24000
REPLY = "Great news, it will be sunny tomorrow! Want me to set a reminder?"


def voiced(seconds, amp=4000):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * 150 * t)).astype("<i2").tobytes()


class CallHardware(SpeakerHardware):
    """Muse's realtime voice has no text label; it is recorded as ``muse``, with its sentence motion."""

    def play_audio(self, samples, *, sentence=None, accents=()):
        if sentence is not None:
            self.recording.add({"robot": "sentence", "expression": sentence.expression.value,
                                "terminal": sentence.terminal})
        if float(samples[0]) in self.recording.labels:
            return super().play_audio(samples)
        event = {"robot": "play", "audio": "muse"}
        if self.recording.events[-1:] != [event]:
            self.recording.add(event)
        self.played.append(samples.copy())

    def finish_sentence(self):
        pass


class ScriptedCalls:
    """Stands in for the SDK ``VoiceCall``: once the microphone frames it receives show the user
    spoke and then paused, it plays ``answer`` (by default ``reply``'s transcript, audio and
    lifecycle events). ``hang_up`` ends the call with an error from inside an answer."""

    def __init__(self, *, reply=REPLY, reply_s=1.0, fail=None, fail_after_frames=0, answer_early=False,
                 answer=None):
        self.reply = reply
        self.reply_s = reply_s
        self.answer_early = answer_early
        self.answer = answer or (answers(reply, reply_s) if reply is not None else None)
        self.loud_after_reply = 0
        self.fail = fail
        self.fail_after_frames = fail_after_frames
        self.opened = []
        self.frames = []
        self.frame_times = []
        self.barge_ins = 0
        self.links = 0
        self.hung_up = None
        self.answering = []

    def hang_up(self, error):
        self.hung_up.set_exception(error)

    async def link(self, vm):
        self.links += 1

        async def close():
            pass
        return VoiceLink(object(), lambda: True, close)

    def __call__(self, session, **options):
        self.opened.append({key: value for key, value in options.items() if key != "on_event"})
        return _Call(self, options["on_event"])


def reply_events(sdk, emit, response_id, text, pcm):
    emit(sdk.ResponseStarted(response_id))
    for word in text.split(" "):
        emit(sdk.TranscriptDelta(response_id, word + " "))
    for offset in range(0, len(pcm), 9600):
        emit(sdk.AudioChunk(response_id, offset // 2, pcm[offset:offset + 9600]))
    emit(sdk.TranscriptDone(response_id, text))
    emit(sdk.ResponseDone(response_id, "completed"))


def answers(reply, reply_s):
    async def answer(sdk, emit, script):
        emit(sdk.Working(True))
        reply_events(sdk, emit, "r1", reply, voiced(1.2) + b"\0\0" * int(.3 * RATE) + voiced(reply_s))
        emit(sdk.Working(False))
    return answer


class _Call:
    def __init__(self, script, on_event):
        self.script = script
        self.on_event = on_event
        self.ready = asyncio.Event()

    async def run(self, source, output, *, realtime_input=False):
        self.ready.set()
        if self.script.fail is not None and self.script.fail_after_frames == 0:
            raise self.script.fail
        self.script.hung_up = asyncio.get_running_loop().create_future()
        listening = asyncio.create_task(self._listen(source))
        try:
            await asyncio.wait({listening, self.script.hung_up}, return_when=asyncio.FIRST_COMPLETED)
            for done in (listening, self.script.hung_up):
                if done.done():
                    done.result()
        finally:
            listening.cancel()
            await asyncio.gather(listening, return_exceptions=True)

    async def _listen(self, source):
        from musegadget import voice_call as sdk

        loud = paused = 0
        answered = False
        async for frame in source:
            self.script.frames.append(frame)
            self.script.frame_times.append(time.monotonic())
            if np.sqrt(np.mean(np.frombuffer(frame, "<i2").astype(float) ** 2)) >= 300:
                loud += 1
                self.script.loud_after_reply += answered
            elif loud >= 10:
                paused += 1
            if self.script.fail is not None and len(self.script.frames) >= self.script.fail_after_frames:
                raise self.script.fail
            due = paused == (2 if self.script.answer_early else 5)
            if due and not answered and self.script.answer is not None:
                answered = True
                self.script.answering.append(
                    asyncio.create_task(self.script.answer(sdk, self.on_event, self.script)))
        await asyncio.Event().wait()

    async def barge_in(self):
        self.script.barge_ins += 1


def realtime_service(rig, transcripts, calls, muse=never_asked, **fields):
    return service(rig, muse, transcripts, wake_detector=Wake(), wake_timeout_s=2, muse_path="realtime",
                   voice_link=calls.link, voice_call=calls, **fields)


async def speak_live(rig, *, wake=False):
    """The user's voice at real-time pace, then half a second of quiet."""
    chunks = [np.full(CHUNK, WAKE_SAMPLE, np.float32)] if wake else []
    chunks += [np.full(CHUNK, VOICE_SAMPLE, np.float32)] * 25 + [np.zeros(CHUNK, np.float32)] * 25
    for chunk in chunks:
        rig.hardware.samples.put(chunk)
        await asyncio.sleep(CHUNK / 16000)


def rig_for(monkeypatch):
    rig = Rig(monkeypatch)
    rig.hardware = CallHardware(rig.recording)
    return rig


def test_after_the_wake_word_muse_hears_the_microphone_and_answers_with_its_own_voice(monkeypatch):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls()
    transcripts = Transcripts("Hey Muse, what's the weather tomorrow?")
    conversation = realtime_service(rig, transcripts, calls)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)

    events = run(rig, conversation, user)
    assert sent_to_muse(events) == []
    assert calls.opened == [{"device_id": "homelink-abcdef", "thread_id": "robot-chat", "half_duplex": True,
                             "ready_timeout": 8.0, "idle_timeout": 900.0, "overall_timeout": 900.0}]
    assert all(len(frame) == 1920 for frame in calls.frames) and len(calls.frames) >= 10
    assert set(played(events)) == {"muse"}
    assert transcripts.texts == []
    assert "speaking" in rig.states()
    assert [(event["expression"], event["terminal"]) for event in events if event.get("robot") == "sentence"] == [
        ("happy", "exclamation"), ("curious", "question")]
    assert {"robot": "command", "name": "reachy.expression",
            "params": {"name": "success1", "intensity": 0.6}} in events


def test_the_user_talking_over_muse_stops_its_reply(monkeypatch):
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    rig.hardware.chunk_s = .05
    calls = ScriptedCalls(reply_s=6.0)
    transcripts = Transcripts("Hey Muse, tell me a long story.", "Actually, never mind.")
    conversation = realtime_service(rig, transcripts, calls)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        await asyncio.sleep(.3)
        rig.say()
        await rig.wait_until_or_timeout(lambda: calls.barge_ins > 0, 5)

    events = run(rig, conversation, user)
    assert calls.opened[0]["half_duplex"] is False
    assert calls.barge_ins == 1
    assert {"robot": "clear_audio"} in events
    assert sent_to_muse(events) == []
    assert len(rig.hardware.played) < 6.0 * 8000 / 400


def test_a_spoken_device_command_runs_on_the_robot_and_muse_is_cut_off(monkeypatch):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(reply=None)
    conversation = realtime_service(rig, Transcripts("Hey Muse, go to sleep."), calls)

    async def user():
        rig.say(wake=True)
        await rig.quiet("sleeping")

    events = run(rig, conversation, user, rest=("sleeping",))
    assert sent_to_muse(events) == []
    assert played(events) == ["tts:Good night."]
    assert calls.barge_ins == 1
    assert rig.states()[-1] == "sleeping"


def test_a_call_that_fails_before_any_audio_reaches_muse_sends_the_request_by_chat(monkeypatch):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(fail=ConnectionError("no media"))

    async def muse(session, request):
        await session.answer(request, "Sunny tomorrow. [reachy:happy]")

    conversation = realtime_service(rig, Transcripts("Hey Muse, what's the weather tomorrow?"), calls, muse,
                                    voice=rig.voice())

    async def user():
        rig.say(wake=True)

    events = run(rig, conversation, user)
    assert sent_to_muse(events) == ["what's the weather tomorrow?"]
    assert played(events)[-1] == "tts:Sunny tomorrow."


def test_a_call_lost_after_muse_heard_the_request_asks_again_and_the_next_request_goes_by_chat(monkeypatch):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(fail=ConnectionError("media reset"), fail_after_frames=5)

    async def muse(session, request):
        await session.answer(request, "Sunny tomorrow. [reachy:happy]")

    transcripts = Transcripts("Hey Muse, what's the weather tomorrow?", "Hey Muse, what's the weather tomorrow?")
    conversation = realtime_service(rig, transcripts, calls, muse, voice=rig.voice())

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(
            lambda: "tts:I lost my line to Muse. Please say that again." in played(rig.recording.events), 5)
        await rig.quiet()
        rig.say(wake=True)

    events = run(rig, conversation, user)
    assert len(calls.opened) == 1
    assert sent_to_muse(events) == ["what's the weather tomorrow?"]
    assert played(events)[-2:] == ["tts:I lost my line to Muse. Please say that again.", "tts:Sunny tomorrow."]


def test_muse_answering_before_reachy_s_endpoint_keeps_the_call_listening_for_the_follow_up(monkeypatch):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(answer_early=True)
    transcripts = Transcripts("And tomorrow?")
    conversation = realtime_service(rig, transcripts, calls)

    async def user():
        await speak_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        voice = conversation._voice[1]
        await rig.wait_until_or_timeout(lambda: not voice._has_output() and not voice._input_blocked(), 5)
        await speak_live(rig)
        await rig.wait_until_or_timeout(lambda: transcripts.texts == [], 5)

    events = run(rig, conversation, user)
    assert calls.loud_after_reply >= 10
    assert sent_to_muse(events) == []
    assert "tts:Yes?" not in played(events)
    assert len(calls.opened) == 1


async def keeps_working(sdk, emit, script):
    emit(sdk.Working(True))


async def room(rig, seconds):
    """A quiet room at real-time pace."""
    for _ in range(round(seconds * 16000 / CHUNK)):
        rig.hardware.samples.put(np.zeros(CHUNK, np.float32))
        await asyncio.sleep(CHUNK / 16000)


def test_while_muse_works_past_the_follow_up_window_the_room_stops_reaching_muse_but_the_call_stays(monkeypatch):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(answer=keeps_working)
    conversation = realtime_service(rig, Transcripts("Book me a table for two."), calls)
    heard = {}

    async def user():
        await speak_live(rig, wake=True)
        await room(rig, 6.0)
        voice = conversation._voice[1]
        heard.update(until=time.monotonic(), call_open=voice.call is not None and not voice.call.done)

    run(rig, conversation, user, rest=("thinking",))
    assert [at for at in calls.frame_times if at > heard["until"] - 2.0] == []
    assert heard["call_open"] is True


async def answer_then_hang_up(sdk, emit, script):
    await answers(REPLY, 1.0)(sdk, emit, script)
    await asyncio.sleep(3.5)
    script.hang_up(ConnectionError("media reset"))


def test_a_call_lost_after_muse_answered_does_not_ask_the_user_to_repeat(monkeypatch):
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    calls = ScriptedCalls(answer_early=True, answer=answer_then_hang_up)
    conversation = realtime_service(rig, Transcripts("Hey Muse, what's the weather tomorrow?"), calls)

    async def user():
        await speak_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: calls.hung_up.done(), 8)
        await room(rig, 1.0)

    events = run(rig, conversation, user)
    assert set(played(events)) == {"muse"}


@pytest.mark.parametrize("transcriber", [Transcripts("And tomorrow?"), FailingStreamingTranscriber()],
                         ids=["no wake boundary", "transcription failed"])
def test_reachy_s_own_recognition_of_a_request_muse_heard_never_speaks_over_muse(monkeypatch, transcriber):
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    calls = ScriptedCalls(answer_early=True)
    conversation = realtime_service(rig, transcriber, calls)

    async def user():
        await speak_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        await room(rig, 3.0)

    events = run(rig, conversation, user)
    assert set(played(events)) == {"muse"}


def test_a_short_sound_over_muse_s_reply_does_not_stop_it(monkeypatch):
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    rig.hardware.chunk_s = .05
    calls = ScriptedCalls(reply_s=3.0)
    transcripts = Transcripts("Hey Muse, tell me a story.", "Mm-hm.")
    conversation = realtime_service(rig, transcripts, calls)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        await asyncio.sleep(.3)
        rig.say(voiced_s=.1)
        await rig.wait_until_or_timeout(lambda: calls.barge_ins > 0, 2)

    events = run(rig, conversation, user)
    assert calls.barge_ins == 0
    assert events.count({"robot": "clear_audio"}) == 1
