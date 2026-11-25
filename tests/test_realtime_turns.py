"""Muse realtime voice calls driven through Reachy's whole conversation, with a scripted call."""

import asyncio
import base64
import json
import re
import time
from types import SimpleNamespace

import pytest

from reachy_muse.realtime_call import VoiceLink, realtime_supported
from test_device_command_turns import FakeModel, SpeakerHardware, never_asked, played, run, sent_to_muse, service
from test_pins import CHUNK, VOICE_SAMPLE, WAKE_SAMPLE, FailingStreamingTranscriber, Rig, Transcripts, Wake
from reachy_muse.capabilities import Partial

np = pytest.importorskip("numpy")
pytest.importorskip("av")
pytestmark = pytest.mark.skipif(not realtime_supported(), reason="the Muse SDK has no voice call events")

RATE = 24000
REPLY = "Great news, it will be sunny tomorrow! Want me to set a reminder?"


class StreamingTranscripts(Transcripts):
    """Scripted streaming words; batch recognition is counted separately."""

    def __init__(self, *texts):
        super().__init__(*texts)
        self.batch_calls = 0

    async def transcribe(self, wav):
        self.batch_calls += 1
        return await super().transcribe(wav)

    def open_turn(self):
        text = self.texts.pop(0) if self.texts else ""

        class Turn:
            partial = Partial(text, 1, id(self))

            def feed(self, pcm):
                pass

            def finish(self):
                result = asyncio.get_running_loop().create_future()
                result.set_result(text)
                return result

            def abort(self):
                pass
        return Turn()


def test_realtime_never_runs_batch_recognition(monkeypatch, caplog):
    caplog.set_level("INFO", logger="reachy_muse.conversation")
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls()

    class BatchOnly(Transcripts):
        batch_calls = 0

        async def transcribe(self, wav):
            self.batch_calls += 1
            return "Hey Muse, tell me a story."

    recognizer = BatchOnly()
    conversation = realtime_service(rig, recognizer, calls)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)

    events = run(rig, conversation, user)
    assert recognizer.batch_calls == 0
    assert sent_to_muse(events) == []
    assert set(played(events)) == {"muse"}
    assert [line for line in caplog.messages if line.startswith("Realtime command ")] == [
        "Realtime command transcript=missing words=0 wake_prefix=none outcome=none drop=none"]


def test_missing_companion_command_transcript_is_logged_after_timed_wake(monkeypatch, caplog):
    from dataclasses import replace
    from reachy_muse.companion_client import CompanionHearing, CompanionLink, LinkState, _Connection
    from reachy_muse.device_command import WakeWindow
    from reachy_muse.protocol import CloseHow, HearClose, Op, Sender, decode

    caplog.set_level("INFO", logger="reachy_muse.conversation")
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls()
    link = CompanionLink("ws://localhost:9", token="test", out_rate=16000, insecure=True)
    link._connection = connection = _Connection(None, (Op.HEAR,))
    link._set(LinkState.UP)
    conversation = realtime_service(rig, None, calls)
    backends = conversation.backends
    conversation.backends = lambda session: replace(backends(session), hearing=CompanionHearing(link))

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: any(line.startswith("Realtime command ") for line in caplog.messages), 3)
        voice = conversation._voice[1]
        assert voice.call is not None
        assert voice._wake_window is WakeWindow.AWAKE

    events = run(rig, conversation, user)
    sent = [decode(frame, sender=Sender.ROBOT) for frame in
            (connection.outbox.get_nowait() for _ in range(connection.outbox.qsize())) if isinstance(frame, str)]
    assert [message.how for message in sent if isinstance(message, HearClose)] == [CloseHow.ABORT]
    assert sent_to_muse(events) == []
    assert set(played(events)) == {"muse"}
    assert [line for line in caplog.messages if line.startswith("Realtime command ")] == [
        "Realtime command transcript=missing words=0 wake_prefix=none outcome=none drop=none"]


@pytest.mark.parametrize("partial, reply, diagnostic", [
    ("", REPLY, "Realtime command transcript=arrived words=0 wake_prefix=none outcome=none drop=barge_in"),
    ("Great news", REPLY, "Realtime command transcript=arrived words=2 wake_prefix=none outcome=none drop=barge_in"),
    ("sunny", REPLY, "Realtime command transcript=arrived words=1 wake_prefix=none outcome=none drop=barge_in"),
    ("Want me to set a reminder?", REPLY, "Realtime command transcript=arrived words=6 wake_prefix=none outcome=none drop=barge_in"),
    ("stop talking", "You can tell me to stop talking.",
     "Realtime command transcript=arrived words=2 wake_prefix=none outcome=none drop=barge_in"),
])
def test_echo_or_unrecognized_noise_during_playback_neither_interrupts_nor_reaches_muse(
        monkeypatch, caplog, partial, reply, diagnostic):
    caplog.set_level("INFO", logger="reachy_muse.conversation")
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    calls = ScriptedCalls(reply_s=4, reply=reply)
    transcripts = StreamingTranscripts("Hey Muse, tell me a story.", partial)
    conversation = realtime_service(rig, transcripts, calls)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        await asyncio.sleep(.3)
        await rig.wait_until_or_timeout(lambda: any(line.startswith("Realtime command ") for line in caplog.messages), 3)
        caplog.clear()
        await speak_live(rig)
        await room(rig, .4)

    events = run(rig, conversation, user)
    assert calls.barge_ins == 0
    assert calls.loud_after_reply == 0
    assert transcripts.batch_calls == 0
    assert sent_to_muse(events) == []
    assert set(played(events)) == {"muse"}
    assert [line for line in caplog.messages if line.startswith("Realtime command ")] == [diagnostic]


def test_failed_recognition_after_an_interruption_keeps_the_next_reply_s_echo_out(monkeypatch):
    from reachy_muse.streaming_transcription import StreamingWorkerError

    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    emit_reply = []

    async def answer(sdk, emit, script):
        emit_reply.append(lambda: reply_events(sdk, emit, "r2", REPLY, voiced(3)))
        reply_events(sdk, emit, "r1", REPLY, voiced(3))

    class FailedInterruption(StreamingTranscripts):
        def open_turn(self):
            turn = super().open_turn()
            if turn.partial.text == "Actually, tell me about trains.":
                def feed(pcm):
                    if not np.any(np.frombuffer(pcm, "<i2")[-320:]):
                        raise StreamingWorkerError("command worker stopped")
                turn.feed = feed
            return turn

    calls = ScriptedCalls(answer=answer)
    transcripts = FailedInterruption("Tell me a story.", "Actually, tell me about trains.", REPLY)
    conversation = realtime_service(rig, transcripts, calls)
    echo_frames = []

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        await speak_live(rig)
        assert calls.barge_ins == 1
        assert calls.loud_after_reply >= 10
        before = calls.loud_after_reply
        emit_reply[0]()
        await room(rig, .3)
        await speak_live(rig)
        echo_frames.append(calls.loud_after_reply - before)

    events = run(rig, conversation, user)
    assert echo_frames == [0]
    assert calls.barge_ins == 1
    assert sent_to_muse(events) == []


@pytest.mark.parametrize("ending", ["finished", "flushed", "unplayed"])
def test_a_command_repeating_an_expired_reply_still_interrupts_the_current_reply(monkeypatch, ending):
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    emit_reply = []

    async def answer(sdk, emit, script):
        emit_reply.append(lambda: reply_events(sdk, emit, "r2", REPLY, voiced(4)))
        if ending == "unplayed":
            emit(sdk.TranscriptDone("r1", "You can tell me to stop talking."))
        else:
            reply_events(sdk, emit, "r1", "You can tell me to stop talking.",
                         voiced(4 if ending == "flushed" else .5))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, StreamingTranscripts("Tell me a story.", "stop talking"), calls)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: bool(emit_reply), 5)
        voice = conversation._voice[1]
        if ending == "flushed":
            await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
            await voice.call.barge_in()
        await rig.wait_until_or_timeout(lambda: not voice.call.audible and time.monotonic() > voice._echo_until, 3)
        emit_reply[0]()
        await room(rig, .3)
        await speak_live(rig)

    events = run(rig, conversation, user)
    assert calls.barge_ins == (2 if ending == "flushed" else 1)
    assert calls.loud_after_reply >= 10
    assert sent_to_muse(events) == []


@pytest.mark.parametrize("text, volume, face, spoken", [
    ("Hey Muse, volume down.", [-10], [], []),
    ("Hey Muse, stop following me.", [], [False], ["tts:Okay, I'll stop following you."]),
    ("Hey Muse, follow my face.", [], [True], ["tts:Okay, I'll look at you."]),
])
@pytest.mark.parametrize("deadline", [0, .25])
def test_streamed_device_commands_during_a_call_still_run(monkeypatch, text, volume, face, spoken, deadline):
    monkeypatch.setattr("reachy_muse.conversation.CALL_COMMAND_DEADLINE_S", deadline)
    rig = rig_for(monkeypatch)
    faces = []
    rig.hardware.face_tracking_enabled = True
    rig.hardware.allow_face_tracking = faces.append
    calls = ScriptedCalls(reply=None)
    recognizer = StreamingTranscripts(text)
    conversation = realtime_service(rig, recognizer, calls)

    async def user():
        rig.say(wake=True)

    events = run(rig, conversation, user)
    assert rig.hardware.volume_steps == volume
    assert faces == face
    assert played(events) == spoken
    assert calls.barge_ins == 1
    assert sent_to_muse(events) == []
    assert recognizer.batch_calls == 0


@pytest.mark.parametrize("during_reply, diagnostic", [
    (False, "Realtime command transcript=deadline words=0 wake_prefix=none outcome=none drop=deadline"),
    (True, "Realtime command transcript=deadline words=0 wake_prefix=none outcome=none drop=barge_in"),
])
def test_stalled_streaming_final_does_not_hold_up_the_call_or_produce_a_retry_cue(
        monkeypatch, caplog, during_reply, diagnostic):
    caplog.set_level("INFO", logger="reachy_muse.conversation")
    from test_streaming_input import StreamingRecognizer

    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    recognizer = StreamingRecognizer()
    calls = ScriptedCalls(reply_s=8 if during_reply else 1)
    conversation = realtime_service(rig, recognizer, calls)
    settled = []

    async def user():
        await speak_live(rig, wake=True)
        await room(rig, .5)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        if during_reply:
            assert conversation._voice[1].call.audible
            caplog.clear()
            await speak_live(rig)
            await room(rig, .5)
        settled.append(recognizer.turns[-1].result.cancelled())

    events = run(rig, conversation, user)
    assert settled == [True]
    assert [line for line in caplog.messages if line.startswith("Realtime command ")] == [diagnostic]
    assert set(played(events)) == {"muse"}
    assert sent_to_muse(events) == []


def voiced(seconds, amp=4000):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * 150 * t)).astype("<i2").tobytes()


def _voice(frames):
    return np.abs(np.frombuffer(b"".join(frames), "<i2").astype(int)) >= VOICE_SAMPLE * 32767 / 2


def voiced_ms(frames):
    """Milliseconds of the user's voice, at ``VOICE_SAMPLE``, in the microphone frames a call received."""
    return round(np.count_nonzero(_voice(frames)) * 1000 / RATE)


def echo_frames(frames):
    """How many call frames carry Muse's own voice, which swings below zero; the user's test voice never does."""
    return sum(int(np.frombuffer(frame, "<i2").min()) < -2000 for frame in frames)


def gaps_ms(frames):
    """Each stretch, in milliseconds, missing from the user's voice between its first and last sample in call frames."""
    voice = _voice(frames)
    if not voice.any():
        return []
    inside = ~voice[np.argmax(voice):len(voice) - np.argmax(voice[::-1])]
    changes = np.flatnonzero(np.diff(np.concatenate(([0], inside.astype(int), [0]))))
    return [round((end - start) * 1000 / RATE) for start, end in zip(changes[::2], changes[1::2])]


def companion_hearing(conversation, *, forced_endpoints=False):
    """Route the conversation's hearing through a companion link whose replies the test sends; returns the link.

    With ``forced_endpoints``, the link also answers each stream the robot finishes as the companion does: with a
    forced endpoint holding the stream's latest partial."""
    from dataclasses import replace
    from reachy_muse.companion_client import CompanionHearing, CompanionLink, LinkState, _Connection
    from reachy_muse.protocol import CloseHow, HearClose, HearEndpoint, HearPartial, Op, Sender, decode, encode

    link = CompanionLink("ws://localhost:9", token="test", out_rate=16000, insecure=True)
    link._connection = connection = _Connection(None, (Op.HEAR,))
    link._set(LinkState.UP)
    if forced_endpoints:
        partials, receive, send = {}, link._receive, connection.send

        def received(on, frame):
            message = decode(frame, sender=Sender.COMPANION)
            if isinstance(message, HearPartial):
                partials[message.stream] = message.text
            receive(on, frame)

        def sent(frame):
            send(frame)
            if isinstance(frame, HearClose) and frame.how is CloseHow.FINISH:
                endpoint = HearEndpoint(frame.stream, partials.get(frame.stream, ""), True)
                asyncio.get_running_loop().call_soon(received, connection, encode(endpoint))

        link._receive, connection.send = received, sent
    backends = conversation.backends
    conversation.backends = lambda session: replace(backends(session), hearing=CompanionHearing(link))
    return link


class CallHardware(SpeakerHardware):
    """Muse's realtime voice has no text label; it is recorded as ``muse``, with its sentence motion.
    ``muse_played`` holds when each chunk of it played, and its samples."""

    def __init__(self, recording, chunk_s=0.0):
        super().__init__(recording, chunk_s)
        self.muse_played = []

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
        self.muse_played.append((time.monotonic(), len(samples)))

    def finish_sentence(self):
        pass


def muse_played_after(rig, mark):
    """Seconds of Muse's voice Reachy played after the time ``mark``."""
    samples = sum(count for at, count in rig.hardware.muse_played if at > mark)
    return round(samples / rig.hardware.output_sample_rate, 2)


class ScriptedCalls:
    """Stands in for the SDK ``VoiceCall``: once the microphone frames it receives show the user
    spoke and then paused, it plays ``answer`` (by default ``reply``'s transcript, audio and
    lifecycle events), for the first request only unless ``every_turn``. ``paced`` takes the
    frames as the SDK sends them to Muse, no faster than real time. ``hang_up`` ends the call
    with an error from inside an answer."""

    def __init__(self, *, reply=REPLY, reply_s=1.0, fail=None, fail_after_frames=0, answer_early=False,
                 answer=None, every_turn=False, paced=False):
        self.reply = reply
        self.reply_s = reply_s
        self.answer_early = answer_early
        self.every_turn = every_turn
        self.paced = paced
        self.answer = answer or (answers(reply, reply_s) if reply is not None else None)
        self.loud_after_reply = 0
        self.fail = fail
        self.fail_after_frames = fail_after_frames
        self.opened = []
        self.frames = []
        self.frame_times = []
        self.barge_ins = 0
        self.links = 0
        self.closed_links = 0
        self.closed_calls = 0
        self.call_frames = []
        self.hung_up = None
        self.answering = []

    def hang_up(self, error):
        self.hung_up.set_exception(error)

    async def link(self, vm):
        self.links += 1

        async def close():
            self.closed_links += 1
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
        self.held = False
        self.frames = []
        script.call_frames.append(self.frames)

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
            self.script.closed_calls += 1

    async def _listen(self, source):
        from musegadget import voice_call as sdk

        loud = paused = 0
        answered = False
        async for frame in (self._paced(source) if self.script.paced else source):
            self.frames.append(frame)
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
                self.script.answering.append(
                    asyncio.create_task(self.script.answer(sdk, self.on_event, self.script)))
                if self.script.every_turn:
                    loud = paused = 0
                else:
                    answered = True
        await asyncio.Event().wait()

    @staticmethod
    async def _paced(source):
        next_send = time.monotonic()
        async for frame in source:
            await asyncio.sleep(max(0, next_send - time.monotonic()))
            yield frame
            next_send = time.monotonic() + len(frame) / (2 * RATE)

    async def barge_in(self):
        self.script.barge_ins += 1


def realtime_service(rig, transcripts, calls, muse=never_asked, **fields):
    return service(rig, muse, transcripts, wake_detector=TimedWake(),
                   wake_timeout_s=fields.pop("wake_timeout_s", 2), muse_path="realtime",
                   voice_link=calls.link, voice_call=calls, **fields)


class TimedWake(Wake):
    epoch = 0
    feed_id = 0

    def feed(self, samples):
        self.feed_id += 1
        detected = super().feed(samples)
        if detected:
            self.last_detection = SimpleNamespace(epoch=self.epoch, feed_id=self.feed_id, post_wake_samples=0)
        return detected

    def reset(self):
        self.epoch += 1


@pytest.mark.parametrize("sample_rate", [16000, 48000])
def test_a_wake_without_a_trustworthy_audio_boundary_preserves_the_request_by_chat(monkeypatch, sample_rate):
    rig = rig_for(monkeypatch)
    rig.hardware.sample_rate = sample_rate
    calls = ScriptedCalls()

    class DelayedWake(TimedWake):
        def feed(self, samples):
            super().feed(samples)
            self.last_detection = SimpleNamespace(epoch=self.epoch, feed_id=self.feed_id, post_wake_samples=0)
            return self.feed_id == 26

    wake = Wake() if sample_rate == 16000 else DelayedWake()

    async def muse(session, request):
        await session.answer(request, "Sunny tomorrow.")

    conversation = service(rig, muse, StreamingTranscripts("Hey Muse, what's the weather tomorrow?"),
                           wake_detector=wake, wake_timeout_s=2, muse_path="realtime",
                           voice_link=calls.link, voice_call=calls)

    async def user():
        for index in range(51):
            value = VOICE_SAMPLE if index < 25 else WAKE_SAMPLE if index == 25 else 0
            rig.hardware.samples.put(np.full(sample_rate // 50, value, np.float32))
            await asyncio.sleep(.02)

    events = run(rig, conversation, user)
    assert sent_to_muse(events) == ["what's the weather tomorrow?"]
    assert played(events) == ["tts:Sunny tomorrow."]
    assert calls.opened == []


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
    calls = ScriptedCalls(reply="Great news! Want me to set a reminder?")
    transcripts = StreamingTranscripts("Hey Muse, what's the weather tomorrow?")
    conversation = realtime_service(rig, transcripts, calls)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)

    events = run(rig, conversation, user)
    assert sent_to_muse(events) == []
    assert calls.opened == [{"device_id": "homelink-abcdef", "thread_id": "robot-chat", "half_duplex": True,
                             "ready_timeout": 8.0, "idle_timeout": 900.0, "overall_timeout": 900.0}]
    assert all(len(frame) == 1920 for frame in calls.frames) and len(calls.frames) >= 10
    # The .75 wake marker must stay local; allow resampler overshoot on .3 user audio.
    assert max(int(np.max(np.abs(np.frombuffer(frame, "<i2")))) for frame in calls.frames) < 16000
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
    transcripts = StreamingTranscripts("Hey Muse, tell me a long story.", "Actually, never mind.")
    conversation = realtime_service(rig, transcripts, calls)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        await asyncio.sleep(.3)
        rig.say()
        await rig.wait_until_or_timeout(lambda: calls.barge_ins > 0, 5)

    events = run(rig, conversation, user, rest=("thinking",))
    assert calls.opened[0]["half_duplex"] is False
    assert calls.barge_ins == 1
    assert {"robot": "clear_audio"} in events
    assert sent_to_muse(events) == []
    assert len(rig.hardware.played) < 6.0 * 8000 / 400


@pytest.mark.parametrize("recognition, heard", [("partial", True), ("final", False)])
def test_words_recognized_after_speech_still_interrupt_playback(monkeypatch, recognition, heard):
    """The user says "stop talking" over Muse's reply, and its words are recognized only after it is said. Muse hears
    it when a partial confirms it in the recognition window; when only its final words do, the command is Reachy's
    alone, and Muse never hears it."""
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    calls = ScriptedCalls(reply_s=6)

    class DelayedTranscripts(StreamingTranscripts):
        def open_turn(self):
            turn = super().open_turn()
            partial = turn.partial
            if partial.text == "stop talking":
                turn.partial = Partial("", 0, partial.utterance_id)

                def feed(pcm):
                    if recognition == "partial" and not np.any(np.frombuffer(pcm, "<i2")[-320:]):
                        turn.partial = partial

                def finish():
                    async def recognize():
                        await asyncio.sleep(.05)
                        return partial.text
                    return asyncio.create_task(recognize())

                turn.feed = feed
                turn.finish = finish
            return turn

    transcripts = DelayedTranscripts("Hey Muse, tell me a story.", "stop talking")
    conversation = realtime_service(rig, transcripts, calls)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        await asyncio.sleep(.3)
        await speak_live(rig)
        await room(rig, .5)

    events = run(rig, conversation, user)
    assert calls.barge_ins == 1
    if heard:
        assert calls.loud_after_reply >= 10
    else:
        assert calls.loud_after_reply == 0
    assert sent_to_muse(events) == []


def test_a_request_the_call_fails_under_before_its_words_confirm_it_asks_the_user_to_say_it_again(monkeypatch):
    """The user asks something over Muse's reply, and its words are recognized only at its endpoint, by which time the
    call has failed. Muse never heard the request and the call can take it no more, so Reachy asks for it again."""
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    calls = ScriptedCalls(reply_s=6)

    class LateWords(StreamingTranscripts):
        def open_turn(self):
            turn = super().open_turn()
            text = turn.partial.text
            if text == "What about Friday then?":
                turn.partial = Partial("", 0, turn.partial.utterance_id)

                def finish():
                    async def recognize():
                        calls.hang_up(ConnectionError("media reset"))
                        await asyncio.sleep(.1)
                        return text
                    return asyncio.create_task(recognize())

                turn.finish = finish
            return turn

    conversation = realtime_service(rig, LateWords("Hey Muse, tell me a story.", "What about Friday then?"), calls)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        await asyncio.sleep(.3)
        await speak_live(rig)
        await rig.wait_until_or_timeout(
            lambda: "tts:I missed part of that. Please say it again." in played(rig.recording.events), 3)

    events = run(rig, conversation, user)
    assert played(events)[-1] == "tts:I missed part of that. Please say it again."
    assert sent_to_muse(events) == []


def test_a_spoken_device_command_runs_on_the_robot_and_muse_is_cut_off(monkeypatch):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(reply=None)
    conversation = realtime_service(rig, StreamingTranscripts("Hey Muse, go to sleep."), calls)

    async def user():
        rig.say(wake=True)
        await rig.quiet("sleeping")

    events = run(rig, conversation, user, rest=("sleeping",))
    assert sent_to_muse(events) == []
    assert played(events) == ["tts:Good night."]
    assert calls.barge_ins == 0
    assert rig.states()[-1] == "sleeping"


@pytest.mark.parametrize("text, diagnostic", [
    ("stop listening.", "Realtime command transcript=arrived words=2 wake_prefix=none outcome=STOP_LISTENING drop=none"),
    ("Hey Mew stop listening.", "Realtime command transcript=arrived words=4 wake_prefix=tolerant outcome=STOP_LISTENING drop=none"),
    ("Hey mu se stop listening.", "Realtime command transcript=arrived words=5 wake_prefix=tolerant outcome=STOP_LISTENING drop=none"),
    ("Hey, Muse. Stop listening.", "Realtime command transcript=arrived words=4 wake_prefix=exact outcome=STOP_LISTENING drop=none"),
])
def test_timed_wake_command_closes_call_and_rejects_late_audio(monkeypatch, caplog, text, diagnostic):
    caplog.set_level("INFO", logger="reachy_muse.conversation")
    from musegadget import voice_call as sdk
    from reachy_muse.device_command import WakeWindow

    rig = rig_for(monkeypatch)

    class LateCalls(ScriptedCalls):
        def __call__(self, session, **options):
            self.emit = options["on_event"]
            return super().__call__(session, **options)

    calls = LateCalls(reply=None)
    conversation = realtime_service(rig, StreamingTranscripts(text), calls)
    observed = []

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: bool(calls.opened), 3)
        voice = conversation._voice[1]
        await rig.wait_until_or_timeout(lambda: "tts:Okay, I'll stop listening." in played(rig.recording.events), 3)
        observed.append((voice.call is None, voice._wake_window, voice._wake_deadline))
        reply_events(sdk, calls.emit, "late", REPLY, voiced(.2))
        await asyncio.sleep(.1)

    events = run(rig, conversation, user)
    assert observed == [(True, WakeWindow.DISMISSED, None)]
    assert played(events) == ["tts:Okay, I'll stop listening."]
    assert sent_to_muse(events) == []
    assert [line for line in caplog.messages if line.startswith("Realtime command ")] == [diagnostic]


@pytest.mark.parametrize("path", ["chat", "realtime"])
@pytest.mark.parametrize("speaking", [False, True], ids=["silent", "speaking"])
@pytest.mark.parametrize("text", ["Hey Mew stop listening.", "Hey news, that's all."])
def test_open_window_reinvocation_dismisses_and_stops_reply(monkeypatch, caplog, path, speaking, text):
    caplog.set_level("INFO", logger="reachy_muse.conversation")
    from musegadget import voice_call as sdk
    from reachy_muse.device_command import WakeWindow
    from test_device_command_turns import LongVoice

    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    calls = ScriptedCalls(reply_s=8 if speaking else .1)

    async def muse(session, request):
        await session.answer(request, "Once upon a time.")

    transcripts = StreamingTranscripts("Tell me a story.", text)
    conversation = service(rig, muse, transcripts, wake_detector=TimedWake(), wake_timeout_s=30,
                           muse_path=path, voice_link=calls.link, voice_call=calls,
                           voice=LongVoice(rig.recording, 200) if speaking and path == "chat" else rig.voice())
    observed = []
    followup_at = []

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: bool(played(rig.recording.events)), 3)
        voice = conversation._voice[1]
        call = voice.call
        if speaking:
            assert voice._has_output()
        else:
            await rig.quiet("listening")
        if path == "realtime":
            await rig.wait_until_or_timeout(lambda: any(line.startswith("Realtime command ") for line in caplog.messages), 3)
        followup_at.append(len(rig.recording.events))
        caplog.clear()
        await speak_live(rig)
        await rig.wait_until_or_timeout(lambda: "tts:Okay, I'll stop listening." in played(rig.recording.events), 3)
        observed.append((voice.call is None, voice._wake_window, voice._wake_deadline))
        if call is not None:
            reply_events(sdk, call.on_event, "late", REPLY, voiced(.2))
        await asyncio.sleep(.1)

    events = run(rig, conversation, user)
    assert observed == [(True, WakeWindow.DISMISSED, None)]
    assert played(events)[0] == ("muse" if path == "realtime" else "tts:Once upon a time.")
    if speaking:
        cleared = next(index for index, event in enumerate(events)
                       if event == {"robot": "clear_audio"} and any(e.get("robot") == "play" for e in events[:index]))
        assert played(events[cleared:]) == ["tts:Okay, I'll stop listening."]
    else:
        assert played(events[followup_at[0]:]) == ["tts:Okay, I'll stop listening."]
    assert sent_to_muse(events) == (["Tell me a story."] if path == "chat" else [])
    assert [line for line in caplog.messages if line.startswith("Realtime command ")] == ([
        "Realtime command transcript=arrived words=4 wake_prefix=tolerant outcome=STOP_LISTENING drop=none"
    ] if path == "realtime" else [])


@pytest.mark.parametrize("path", ["chat", "realtime"])
@pytest.mark.parametrize("text, diagnostic", [
    ("They must go to sleep.", "Realtime command transcript=arrived words=5 wake_prefix=none outcome=none drop=none"),
    ("Hey, I must go to sleep.", "Realtime command transcript=arrived words=6 wake_prefix=none outcome=none drop=none"),
    ("Hey, my users go to sleep.", "Realtime command transcript=arrived words=6 wake_prefix=none outcome=none drop=none"),
])
def test_first_request_after_a_timed_wake_keeps_the_wake_window_and_call_open(monkeypatch, caplog, path, text, diagnostic):
    from reachy_muse.device_command import WakeWindow

    caplog.set_level("INFO", logger="reachy_muse.conversation")
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(reply_s=.1)

    async def muse(session, request):
        await session.answer(request, "They're tired.")

    conversation = service(rig, muse, StreamingTranscripts(text),
                           wake_detector=TimedWake(), wake_timeout_s=30, muse_path=path,
                           voice_link=calls.link, voice_call=calls)
    observed = []

    async def user():
        rig.say(wake=True)
        await rig.quiet("listening", "sleeping")
        voice = conversation._voice[1]
        observed.append((voice._wake_window, voice._wake_deadline is not None,
                         voice.call is not None and not voice.call.done))

    events = run(rig, conversation, user, rest=("listening", "sleeping"))
    assert observed == [(WakeWindow.AWAKE, True, path == "realtime")]
    assert set(played(events)) == ({"tts:They're tired."} if path == "chat" else {"muse"})
    assert sent_to_muse(events) == ([text] if path == "chat" else [])
    assert calls.barge_ins == 0
    assert not any("handled a device command" in message for message in caplog.messages)
    assert [line for line in caplog.messages if line.startswith("Realtime command ")] == ([diagnostic] if path == "realtime" else [])


@pytest.mark.parametrize("path", ["chat", "realtime"])
@pytest.mark.parametrize("text, diagnostic", [
    ("They must go to sleep.", "Realtime command transcript=arrived words=5 wake_prefix=none outcome=none drop=none"),
    ("Hey, I must go to sleep.", "Realtime command transcript=arrived words=6 wake_prefix=none outcome=none drop=none"),
    ("Hey, my users go to sleep.", "Realtime command transcript=arrived words=6 wake_prefix=none outcome=none drop=none"),
    ("Hey Mew, stop listening to the radio", "Realtime command transcript=arrived words=7 wake_prefix=none outcome=none drop=none"),
    ("Hey Mew what's the weather", "Realtime command transcript=arrived words=5 wake_prefix=none outcome=none drop=none"),
    ("The news stopped listening to him", "Realtime command transcript=arrived words=6 wake_prefix=none outcome=none drop=none"),
])
def test_ordinary_followup_keeps_the_wake_window_and_call_open(monkeypatch, caplog, path, text, diagnostic):
    from reachy_muse.device_command import WakeWindow

    caplog.set_level("INFO", logger="reachy_muse.conversation")
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(reply_s=.1)
    transcripts = StreamingTranscripts("Tell me a story.", text)

    async def muse(session, request):
        await session.answer(request, "Once upon a time.")

    conversation = service(rig, muse, transcripts, wake_detector=TimedWake(), wake_timeout_s=30,
                           muse_path=path, voice_link=calls.link, voice_call=calls)
    observed = []

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: bool(played(rig.recording.events)), 3)
        await rig.quiet("listening")
        voice = conversation._voice[1]
        call = voice.call
        caplog.clear()
        rig.say()
        await rig.quiet("thinking" if path == "realtime" else "listening")
        observed.append((voice._wake_window, voice._wake_deadline is not None,
                         voice.call is call, call is None or not call.done))

    events = run(rig, conversation, user, rest=("thinking" if path == "realtime" else "listening",))
    assert observed == [(WakeWindow.AWAKE, True, True, True)]
    assert "tts:Good night." not in played(events)
    assert sent_to_muse(events) == (["Tell me a story.", text] if path == "chat" else [])
    assert not any("handled a device command" in message for message in caplog.messages)
    assert [line for line in caplog.messages if line.startswith("Realtime command ")] == ([diagnostic] if path == "realtime" else [])


@pytest.mark.parametrize("path", ["chat", "realtime"])
@pytest.mark.parametrize("text, diagnostic", [
    ("Hey Mew stop talking", "Realtime command transcript=arrived words=4 wake_prefix=tolerant outcome=unavailable drop=none"),
    ("Hey Mew volume up", "Realtime command transcript=arrived words=4 wake_prefix=tolerant outcome=unavailable drop=none"),
    ("Hey Mew look at me", "Realtime command transcript=arrived words=5 wake_prefix=tolerant outcome=unavailable drop=none"),
])
def test_unavailable_followup_recovery_preserves_the_request(monkeypatch, caplog, path, text, diagnostic):
    from reachy_muse.device_command import WakeWindow

    caplog.set_level("INFO", logger="reachy_muse.conversation")
    rig = rig_for(monkeypatch)
    rig.hardware.change_volume = None
    rig.hardware.face_tracking_enabled = False
    calls = ScriptedCalls(reply_s=.1)

    async def muse(session, request):
        await session.answer(request, "Once upon a time.")

    conversation = service(rig, muse, StreamingTranscripts("Tell me a story.", text),
                           wake_detector=TimedWake(), wake_timeout_s=30, muse_path=path,
                           voice_link=calls.link, voice_call=calls)
    observed = []

    async def user():
        rig.say(wake=True)
        await rig.quiet("listening")
        caplog.clear()
        rig.say()
        await rig.quiet("thinking" if path == "realtime" else "listening")
        voice = conversation._voice[1]
        observed.append((voice._wake_window, voice.call is not None))

    events = run(rig, conversation, user, rest=("thinking" if path == "realtime" else "listening",))
    assert observed == [(WakeWindow.AWAKE, path == "realtime")]
    assert sent_to_muse(events) == (["Tell me a story.", text] if path == "chat" else [])
    assert not any("handled a device command" in line for line in caplog.messages)
    assert [line for line in caplog.messages if line.startswith("Realtime command ")] == ([diagnostic] if path == "realtime" else [])


def test_tolerant_stop_talking_cuts_an_early_reply(monkeypatch):
    from musegadget import voice_call as sdk

    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(reply=None)

    class AfterPlayback(StreamingTranscripts):
        def open_turn(self):
            turn = super().open_turn()

            def finish():
                async def recognize():
                    call = conversation._voice[1].call
                    reply_events(sdk, call.on_event, "early", REPLY, voiced(5))
                    await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 2)
                    assert "muse" in played(rig.recording.events)
                    return "Hey mu be quiet"
                return asyncio.create_task(recognize())

            turn.finish = finish
            return turn

    conversation = realtime_service(rig, AfterPlayback(""), calls)

    async def user():
        rig.say(wake=True)

    events = run(rig, conversation, user)
    assert played(events)[0] == "muse"
    cleared = events.index({"robot": "clear_audio"})
    assert played(events[cleared:]) == []
    assert calls.barge_ins == 1
    assert sent_to_muse(events) == []


@pytest.mark.parametrize("command, acknowledgement, rest", [
    ("stop listening", "Okay, I'll stop listening.", "idle"),
    ("go to sleep", "Good night.", "sleeping"),
])
def test_dismissal_closes_the_call_before_acknowledging_and_uploads_no_more_audio(
        monkeypatch, command, acknowledgement, rest):
    from test_device_command_turns import LongVoice

    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    calls = ScriptedCalls(reply=None)
    conversation = realtime_service(rig, StreamingTranscripts(f"Hey Muse, {command}.", "Book a table."), calls,
                                    voice=LongVoice(rig.recording, 200))
    observed = []
    resumed = []

    async def user():
        rig.say(wake=True)
        label = f"tts:{acknowledgement}"
        await rig.wait_until_or_timeout(lambda: label in played(rig.recording.events), 5)
        assert label in played(rig.recording.events)
        call = conversation._voice[1].call
        before = len(calls.frames)
        await speak_live(rig)
        observed.append((call is None, len(calls.frames) - before))
        await rig.quiet(rest)
        before = len(calls.frames)
        await speak_live(rig, wake=True)
        resumed.append(len(calls.frames) - before)

    events = run(rig, conversation, user, rest=("thinking",))
    assert observed == [(True, 0)]
    assert resumed[0] >= 10
    assert len(calls.opened) == 2
    assert sent_to_muse(events) == []


@pytest.mark.parametrize("call_text, acknowledgement", [
    ("book a table for two", None),
    ("stop listening", "tts:Okay, I'll stop listening."),
])
def test_a_call_opened_during_a_pending_chat_request_never_replays_its_turn_by_chat(
        monkeypatch, call_text, acknowledgement):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(reply=None)
    release = asyncio.Event()

    async def muse(session, request):
        await release.wait()
        await session.answer(request, "Sunny tomorrow.")

    class FirstWakeUntimed(TimedWake):
        def feed(self, samples):
            detected = super().feed(samples)
            if detected and not calls.opened and rig.session.requests == 0:
                self.last_detection = None
            return detected

    conversation = service(rig, muse, StreamingTranscripts(
        "Hey Muse, what's the weather tomorrow?", f"Hey Muse, {call_text}."),
        wake_detector=FirstWakeUntimed(), wake_timeout_s=.3, muse_path="realtime",
        voice_link=calls.link, voice_call=calls)

    async def user():
        try:
            rig.say(wake=True)
            await rig.quiet("thinking")
            await speak_live(rig, wake=True)
            await room(rig, .3)
            if acknowledgement is not None:
                assert acknowledgement in played(rig.recording.events)
                assert conversation._voice[1].call is None
                before = len(calls.frames)
                await speak_live(rig)
                assert len(calls.frames) == before
        finally:
            release.set()

    events = run(rig, conversation, user)
    assert len(calls.frames) >= 10
    expected = ["what's the weather tomorrow?"]
    if acknowledgement is not None:
        expected.append("Please stop working on my previous request.")
    assert sent_to_muse(events) == expected
    if acknowledgement is not None:
        assert acknowledgement in played(events)


@pytest.mark.parametrize("failure_at", ["opening", "recognition"])
def test_a_call_that_fails_before_any_audio_reaches_muse_sends_the_request_by_chat(monkeypatch, failure_at):
    rig = rig_for(monkeypatch)
    finalizing = asyncio.Event()
    release = asyncio.Event()

    class WaitingCalls(ScriptedCalls):
        def __call__(self, session, **options):
            call = super().__call__(session, **options)

            async def run(source, output, *, realtime_input=False):
                self.hung_up = asyncio.get_running_loop().create_future()
                await self.hung_up

            call.run = run
            return call

    class DelayedTranscripts(StreamingTranscripts):
        def open_turn(self):
            turn = super().open_turn()

            def finish():
                async def recognize():
                    finalizing.set()
                    await release.wait()
                    return turn.partial.text
                return asyncio.create_task(recognize())

            turn.finish = finish
            return turn

    calls = WaitingCalls() if failure_at == "recognition" else ScriptedCalls(fail=ConnectionError("no media"))
    recognizer = DelayedTranscripts if failure_at == "recognition" else StreamingTranscripts

    async def muse(session, request):
        await session.answer(request, "Sunny tomorrow. [reachy:happy]")

    conversation = realtime_service(rig, recognizer("Hey Muse, what's the weather tomorrow?"), calls, muse,
                                    voice=rig.voice())

    async def user():
        rig.say(wake=True)
        if failure_at == "recognition":
            try:
                await asyncio.wait_for(finalizing.wait(), 3)
                await room(rig, .1)
                calls.hang_up(ConnectionError("no media"))
                await rig.wait_until_or_timeout(lambda: conversation._voice[1].call is None, 3)
            finally:
                release.set()

    events = run(rig, conversation, user)
    assert sent_to_muse(events) == ["what's the weather tomorrow?"]
    assert played(events)[-1] == "tts:Sunny tomorrow."
    assert calls.frames == []


def test_missing_companion_partial_after_a_failed_call_transcribes_once_and_sends_one_chat(monkeypatch):
    from dataclasses import replace
    from reachy_muse.companion_client import CompanionHearing, CompanionLink, LinkState, _Connection
    from reachy_muse.failover import CompanionRoute, FailoverHearing
    from reachy_muse.protocol import CloseHow, HearClose, Op, Sender, decode

    rig = rig_for(monkeypatch)
    link = CompanionLink("ws://localhost:9", token="test", out_rate=16000, insecure=True)
    link._connection = connection = _Connection(None, (Op.HEAR,))
    link._set(LinkState.UP)

    class WaitingCalls(ScriptedCalls):
        def __call__(self, session, **options):
            call = super().__call__(session, **options)

            async def run(source, output, *, realtime_input=False):
                self.hung_up = asyncio.get_running_loop().create_future()
                await self.hung_up

            call.run = run
            return call

    calls = WaitingCalls()
    transcripts = StreamingTranscripts("Tell me a story.")

    async def muse(session, request):
        await session.answer(request, "Once upon a time.")

    conversation = realtime_service(rig, transcripts, calls, muse)
    backends = conversation.backends

    def with_companion(session):
        bundle = backends(session)
        return replace(bundle, hearing=FailoverHearing(
            CompanionHearing(link), bundle.hearing, CompanionRoute(link).hearing))

    conversation.backends = with_companion

    async def user():
        speaking = asyncio.create_task(speak_live(rig, wake=True))
        try:
            await rig.wait_until_or_timeout(lambda: bool(connection.hearing) and calls.hung_up is not None, 3)
            assert len(connection.hearing) == 1
            assert conversation._voice[1].call.audio_sent is False
            calls.hang_up(ConnectionError("no media"))
        finally:
            await speaking

    events = run(rig, conversation, user)
    sent = [decode(frame, sender=Sender.ROBOT) for frame in
            (connection.outbox.get_nowait() for _ in range(connection.outbox.qsize())) if isinstance(frame, str)]
    assert [message.how for message in sent if isinstance(message, HearClose)] == [CloseHow.ABORT]
    assert transcripts.batch_calls == 1
    assert sent_to_muse(events) == ["Tell me a story."]
    assert [event for event in events if event.get("muse") == "send_voice"] == []
    assert rig.session.requests == 1
    assert calls.frames == []
    assert played(events)[-1] == "tts:Once upon a time."


def test_a_call_lost_after_muse_heard_the_request_asks_again_and_the_next_request_goes_by_chat(monkeypatch):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(fail=ConnectionError("media reset"), fail_after_frames=5)

    async def muse(session, request):
        await session.answer(request, "Sunny tomorrow. [reachy:happy]")

    transcripts = StreamingTranscripts("Hey Muse, what's the weather tomorrow?", "Hey Muse, what's the weather tomorrow?")
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


@pytest.mark.parametrize("reason, spoken", [
    ("remote_end", ["tts:I lost my line to Muse. Please say that again."]),
    ("remote_stop", []), ("idle_timeout", []), ("overall_timeout", []),
])
def test_an_ended_call_never_resends_uploaded_speech_by_chat(monkeypatch, reason, spoken):
    rig = rig_for(monkeypatch)

    class EOFCalls(ScriptedCalls):
        def __call__(self, session, **options):
            frames = self.frames

            class Call:
                async def run(self, source, output, *, realtime_input=False):
                    async for frame in source:
                        frames.append(frame)
                        if len(frames) == 5:
                            return reason

            return Call()

    calls = EOFCalls()

    async def muse(session, request):
        await session.answer(request, "The reservation is booked.")

    conversation = realtime_service(rig, StreamingTranscripts("Hey Muse, book a table for two."), calls, muse)

    async def user():
        rig.say(wake=True)

    events = run(rig, conversation, user)
    assert sent_to_muse(events) == []
    assert played(events) == spoken
    assert len(calls.frames) == 5


def test_muse_answering_before_reachy_s_endpoint_keeps_the_call_listening_for_the_follow_up(monkeypatch):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(answer_early=True)
    transcripts = StreamingTranscripts("And tomorrow?")
    conversation = realtime_service(rig, transcripts, calls)

    async def user():
        await speak_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        voice = conversation._voice[1]
        await rig.wait_until_or_timeout(lambda: not voice._has_output() and not voice._input_blocked(), 5)
        await speak_live(rig)
        await rig.wait_until_or_timeout(lambda: transcripts.texts == [], 5)

    events = run(rig, conversation, user, rest=("thinking",))
    assert calls.loud_after_reply >= 10
    assert sent_to_muse(events) == []
    assert "tts:Yes?" not in played(events)
    assert len(calls.opened) == 1


async def keeps_working(sdk, emit, script):
    emit(sdk.Working(True))


def test_reachy_say_is_busy_while_a_realtime_reply_is_audible(monkeypatch):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(reply_s=2)
    conversation = realtime_service(rig, StreamingTranscripts("Tell me a story."), calls)
    results = []

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        results.append(await asyncio.to_thread(conversation.run_command, "reachy.say",
                                               {"text": "A notification."}, 2000))

    events = run(rig, conversation, user)
    assert results == [{"ok": True, "payload": {"spoken": False, "reason": "busy"}}]
    assert "tts:A notification." not in played(events)


async def room(rig, seconds):
    """A quiet room at real-time pace."""
    for _ in range(round(seconds * 16000 / CHUNK)):
        rig.hardware.samples.put(np.zeros(CHUNK, np.float32))
        await asyncio.sleep(CHUNK / 16000)


def test_unconfirmed_sound_crossing_the_deadline_does_not_hang_up_a_reply(monkeypatch, caplog):
    caplog.set_level("INFO", logger="reachy_muse.conversation")
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True

    async def answer(sdk, emit, script):
        reply_events(sdk, emit, "r1", REPLY, voiced(7.5))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, StreamingTranscripts("Tell me a story.", "Okay."), calls)

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        await room(rig, voice._wake_deadline - time.monotonic() - .3)
        for _ in range(15):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
        await room(rig, .5)
        assert voice.call is call
        assert call.audible
        assert not call.done
        await rig.wait_until_or_timeout(lambda: not call.audible, 8)
        assert voice.call is call
        assert not call.done
        assert sum(len(chunk) for chunk in rig.hardware.played) / rig.hardware.output_sample_rate == pytest.approx(7.5, abs=.003)

    events = run(rig, conversation, user)
    assert calls.barge_ins == 0
    assert len(calls.opened) == 1
    assert set(played(events)) == {"muse"}
    assert sent_to_muse(events) == []
    assert "Realtime command transcript=arrived words=1 wake_prefix=none outcome=none drop=barge_in" in caplog.messages
    assert "Reachy's wake window closed after empty recognition and inactivity" not in caplog.messages


def muse_heard(seconds):
    """Muse's test voice, as ``voiced`` makes it, the way Reachy's microphone hears it from the speaker."""
    t = np.arange(round(seconds * 16000)) / 16000
    return (4000 / 32768 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)


def echo_of(rig, seconds):
    """The last ``seconds`` that Reachy's speaker played, as its microphone hears them."""
    played = np.concatenate(rig.hardware.played)[-round(seconds * rig.hardware.output_sample_rate):]
    return np.repeat(played, rig.hardware.sample_rate // rig.hardware.output_sample_rate)


@pytest.mark.parametrize("sound, seconds, words, reached, diagnostics", [
    ("voice", .5, "Actually, tell me about trains.", True,
     ["Realtime command transcript=arrived words=5 wake_prefix=none outcome=none drop=none"]),
    ("voice", .3, "Okay.", False,
     ["Realtime command transcript=arrived words=1 wake_prefix=none outcome=none drop=barge_in"]),
    ("voice", .5, "No-no-no!", True,
     ["Realtime command transcript=arrived words=1 wake_prefix=none outcome=none drop=none"]),
    ("voice", .5, "Wait-wait!", True,
     ["Realtime command transcript=arrived words=1 wake_prefix=none outcome=none drop=none"]),
    ("voice", .5, "Oh, wait!", True,
     ["Realtime command transcript=arrived words=2 wake_prefix=none outcome=none drop=none"]),
    ("voice", .5, "Uh, no.", True,
     ["Realtime command transcript=arrived words=2 wake_prefix=none outcome=none drop=none"]),
    ("cough", .16, "", False, []),
    ("echo", .5, "it will be sunny tomorrow", False,
     ["Realtime command transcript=arrived words=5 wake_prefix=none outcome=none drop=barge_in"]),
    ("uncancelled echo", .5, "it will be sunny tomorrow", False, []),
])
def test_while_muse_talks_past_the_wake_deadline_only_confirmed_speech_reaches_it(
        monkeypatch, caplog, sound, seconds, words, reached, diagnostics):
    caplog.set_level("INFO", logger="reachy_muse.conversation")
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = sound != "uncancelled echo"

    async def answer(sdk, emit, script):
        reply_events(sdk, emit, "r1", REPLY, voiced(7.5))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, StreamingTranscripts("Tell me a story.", words), calls)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        await room(rig, voice._wake_deadline - time.monotonic() + .5)
        assert call.audible
        caplog.clear()
        before = len(calls.frames)
        audio = (echo_of(rig, seconds) if sound.endswith("echo")
                 else np.full(round(seconds * 16000), VOICE_SAMPLE, np.float32))
        for start in range(0, len(audio), CHUNK):
            rig.hardware.samples.put(audio[start:start + CHUNK])
            await asyncio.sleep(CHUNK / 16000)
        await room(rig, .5)
        heard.append((voiced_ms(calls.frames[before:]), echo_frames(calls.frames[before:]), calls.barge_ins,
                      voice.call is call and not call.done))
        await rig.wait_until_or_timeout(lambda: not call.audible, 8)
        heard.append(sum(len(chunk) for chunk in rig.hardware.played) / rig.hardware.output_sample_rate)

    events = run(rig, conversation, user, rest=("listening", "idle", "thinking"))
    assert heard[0] == (500 if reached else 0, 0, int(reached), True)
    assert (heard[1] < 7) if reached else (heard[1] == pytest.approx(7.5, abs=.003))
    assert [line for line in caplog.messages
            if line.startswith("Realtime command ") or "could not transcribe" in line] == diagnostics
    assert set(played(events)) == {"muse"}
    assert sent_to_muse(events) == []


BACK_CHANNELS = ("Mm-hmm.", "Mm hmm.", "Mmhmm.", "Mhm.", "M-hm.", "M hm.", "Uh-huh.", "Uh huh.", "Uhhuh.",
                 "Uh-uh.", "Uh uh.", "Uhuh.", "Uh-oh.", "Uh oh.", "Uhoh.", "Hmm.", "Mm.", "Uh.", "Um.", "Ah.", "Oh.")


def test_back_channel_sounds_over_muse_s_reply_never_cut_it_off(monkeypatch, caplog):
    """Each back-channel sound, written with a hyphen, a space or neither, as half a second of voice over Muse."""
    caplog.set_level("INFO", logger="reachy_muse.conversation")
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True

    async def answer(sdk, emit, script):
        reply_events(sdk, emit, "r1", REPLY, voiced(22))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, StreamingTranscripts("Tell me a long story.", *BACK_CHANNELS), calls,
                                    wake_timeout_s=30)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        call = conversation._voice[1].call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        caplog.clear()
        before = len(calls.frames)
        for _ in BACK_CHANNELS:
            for _ in range(25):
                rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
                await asyncio.sleep(CHUNK / 16000)
            await room(rig, .3)
        heard.append((voiced_ms(calls.frames[before:]), calls.barge_ins, call.audible))
        await rig.wait_until_or_timeout(lambda: not call.audible, 8)
        heard.append(sum(len(chunk) for chunk in rig.hardware.played) / rig.hardware.output_sample_rate)

    run(rig, conversation, user, rest=("listening", "idle", "thinking"), timeout=45)
    assert heard[0] == (0, 0, True)
    assert heard[1] == pytest.approx(22, abs=.003)
    assert [line for line in caplog.messages if line.startswith("Realtime command ")] == [
        f"Realtime command transcript=arrived words={len(sound.split())} wake_prefix=none outcome=none drop=barge_in"
        for sound in BACK_CHANNELS]


def test_a_remark_that_never_interrupted_muse_is_dropped_when_muse_s_next_reply_ends_it(monkeypatch, caplog):
    """The user says "mm-hmm" over the end of Muse's reply, which it never interrupts, and Muse's next reply follows
    long before Reachy's own end of the turn: that reply ends the remark's turn, which is dropped as on main and is
    not counted as a turn the reply answers."""
    caplog.set_level("INFO", logger="reachy_muse.conversation")
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    next_reply = []

    async def answer(sdk, emit, script):
        next_reply.append(lambda: reply_events(sdk, emit, "r2", REPLY, voiced(1)))
        reply_events(sdk, emit, "r1", REPLY, voiced(1))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, StreamingTranscripts("Tell me a story.", "Mm-hmm."), calls,
                                    wake_timeout_s=30)
    conversation.silence_s = 1.2
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        call = conversation._voice[1].call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        caplog.clear()
        for _ in range(25):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
        next_reply[0]()
        await room(rig, .8)
        heard.append((calls.barge_ins, call.turns,
                      [line for line in caplog.messages if line.startswith("Realtime command ")]))

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking"), timeout=30)
    assert heard == [(0, 1, ["Realtime command transcript=arrived words=1 wake_prefix=none outcome=none drop=barge_in"])]


def test_words_muse_s_next_reply_ends_before_they_confirm_are_dropped_though_their_final_words_would_confirm(
        monkeypatch, caplog):
    """The user adds "and the one after that" in short words over the end of Muse's acknowledgement, and Muse's own
    next reply, its answer, starts 0.5 s after the last word and ends the turn. No partial confirmed the words, so
    Muse never heard them, and that reply cannot answer them: the turn is dropped, as on main, though its final words
    would confirm it, and Muse's answer plays on whole instead of being cut for words it never heard."""
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    answered = asyncio.Event()

    async def answer(sdk, emit, script):
        if answered.is_set():
            return
        answered.set()
        reply_events(sdk, emit, "r1", "Let me check the calendar for you.", voiced(3))
        await asyncio.sleep(3.8)
        reply_events(sdk, emit, "r2", "You have a dentist appointment at nine tomorrow.", voiced(4))

    calls = ScriptedCalls(answer=answer, every_turn=True, paced=True)
    conversation = realtime_service(rig, None, calls, wake_timeout_s=10)
    conversation.silence_s = 2.0
    link = companion_hearing(conversation, forced_endpoints=True)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        call = conversation._voice[1].call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        await room(rig, 2)
        caplog.clear()
        before = len(calls.frames)
        await say_over_reply(rig, link, "and the one after that",
                             ((200, 120), (160, 60), (200, 100), (160, 40), (240, 0)))
        await room(rig, 6)
        heard.append((voiced_ms(calls.frames[before:]), calls.barge_ins,
                      [line.split(" wake_prefix=")[1] for line in caplog.messages
                       if line.startswith("Realtime command ")],
                      [line for line in caplog.messages if line.startswith("Realtime turn ordinal=2 ")]))
        heard.append(round(sum(count for _, count in rig.hardware.muse_played) / rig.hardware.output_sample_rate, 1))

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking"), timeout=40)
    assert heard == [(0, 0, ["none outcome=none drop=barge_in"], []), 7.0]


def test_words_muse_s_next_reply_begins_after_while_reachy_weighs_them_are_dropped(monkeypatch, caplog):
    """The user adds "and the night after that" in short words over the end of Muse's acknowledgement, and only the
    turn's final words confirm it. Reachy's command model weighs the words for a second, and Muse's next reply, its
    answer, begins meanwhile: that reply ended the turn before Muse heard any of it. The turn is dropped, as when the
    reply begins before the final words, and Muse's answer plays on whole."""
    from reachy_muse import device_command

    caplog.set_level("INFO")
    monkeypatch.setattr(device_command, "CLASSIFY_DEADLINE_S", 2.0)
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    next_reply = []

    async def answer(sdk, emit, script):
        if next_reply:
            return
        next_reply.append(lambda: reply_events(sdk, emit, "r2", "You have a dentist appointment at nine tomorrow.",
                                               voiced(4)))
        reply_events(sdk, emit, "r1", "Let me check the calendar for you.", voiced(3))

    class Weighing(FakeModel):
        async def classify(self, text, allowed):
            next_reply[0]()
            return await super().classify(text, allowed)

    calls = ScriptedCalls(answer=answer, every_turn=True, paced=True)
    conversation = realtime_service(rig, None, calls, wake_timeout_s=10, model=Weighing(None, delay_s=1))
    conversation.silence_s = 2.0
    link = companion_hearing(conversation, forced_endpoints=True)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        call = conversation._voice[1].call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        await room(rig, 1)
        caplog.clear()
        before = len(calls.frames)
        await say_over_reply(rig, link, "and the night after that",
                             ((200, 120), (160, 60), (200, 100), (160, 40), (240, 0)))
        await room(rig, 7)
        heard.append((voiced_ms(calls.frames[before:]), calls.barge_ins,
                      [line.split(" wake_prefix=")[1] for line in caplog.messages
                       if line.startswith("Realtime command ")],
                      [line for line in caplog.messages if line.startswith("Realtime turn ordinal=2 ")]))
        heard.append(round(sum(count for _, count in rig.hardware.muse_played) / rig.hardware.output_sample_rate, 1))

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking"), timeout=40)
    assert heard == [(0, 0, ["none outcome=none drop=barge_in"], []), 7.0]


# A request as Reachy's voice detector hears it over Muse's reply: each word's voiced ms, and the quiet ms after it.
STRETCHES = {
    "each word its own stretch": ((260, 120), (160, 40), (140, 60), (240, 40), (80, 40), (280, 80), (100, 40),
                                  (160, 100), (280, 0)),
    "one long stretch before the partial has two words": ((380, 0), (180, 0), (140, 60), (240, 40), (80, 40),
                                                          (280, 80), (100, 40), (160, 100), (280, 0)),
    "fluent, ending on a long stretch": ((380, 100), (180, 0), (140, 60), (240, 0), (80, 0), (280, 80), (100, 0),
                                         (160, 0), (280, 0)),
}
# The companion's streaming recognizer runs this far behind the voice.
COMPANION_LAG_S = .48
LONG_REPLY = ("I can't see you from here, but I can still help. Tell me what you're wearing and I'll suggest "
              "something for tonight, or ask me about the weather first.")


async def say_over_reply(rig, link, words, stretches, *, final_after_s=None, partial_words=None, sound=None):
    """Say ``words`` at real-time pace in ``stretches``, in the user's test voice or ``sound``. The companion's
    partial holds each word COMPANION_LAG_S after its voice ends, up to ``partial_words`` words; with
    ``final_after_s``, the companion's own endpoint brings all the words that long after the last one."""
    from reachy_muse.protocol import HearEndpoint, HearPartial, encode

    said, voice, ends = words.split(), [], []
    for voiced_ms_, quiet_ms in stretches:
        voice += [True] * (voiced_ms_ // 20) + [False] * (quiet_ms // 20)
        ends.append((len(voice) - quiet_ms // 20) * .02)
    known, stream, shown = set(link._connection.hearing), None, 0
    for frame in range(len(voice) + round(((final_after_s or 0) + COMPANION_LAG_S + .1) / .02)):
        voiced = frame < len(voice) and voice[frame]
        chunk = np.full(CHUNK, VOICE_SAMPLE if voiced else 0, np.float32)
        if voiced and sound is not None:
            chunk = sound[frame * CHUNK % len(sound):][:CHUNK].copy()
        rig.hardware.samples.put(chunk)
        await asyncio.sleep(CHUNK / 16000)
        if stream is None:
            stream = next((opened for opened in link._connection.hearing if opened not in known), None)
        if stream is None or stream not in link._connection.hearing:
            continue
        heard = min(sum(end + COMPANION_LAG_S <= (frame + 1) * .02 for end in ends), partial_words or len(said))
        if heard != shown:
            shown = heard
            link._receive(link._connection, encode(HearPartial(stream, heard, " ".join(said[:heard]))))
        if final_after_s is not None and frame + 1 == len(voice) + round(final_after_s / .02):
            link._receive(link._connection, encode(HearEndpoint(stream, words, False)))


@pytest.mark.parametrize("words, stretches", [
    ("Reachy, what should I wear to the dinner tonight?", "each word its own stretch"),
    ("Reachy, what should I wear to the dinner tonight?", "one long stretch before the partial has two words"),
    ("What should I wear to the dinner tonight, Reachy?", "each word its own stretch"),
    ("Can you take a photo of me now, Reachy?", "each word its own stretch")],
    ids=["each word its own stretch", "one long stretch before the partial has two words",
         "opening with a word of Muse's reply", "opening with another word of Muse's reply"])
def test_a_request_over_a_long_reply_that_no_partial_confirms_reaches_muse_at_reachy_s_endpoint(
        monkeypatch, caplog, words, stretches):
    """The user asks something over Muse's reply as the robot hears it: gaps between the words, the companion's
    partials behind the voice, and the robot's 2 s of silence. No partial confirms the request while it is said,
    but its words at Reachy's endpoint do: Muse is cut off, hears every word of it, and answers it. That holds for a
    request opening, as most do, with a word Muse's reply also holds, which alone could be Muse's echo."""
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    replies = []

    async def answer(sdk, emit, script):
        replies.append(len(replies) + 1)
        if replies == [1]:
            reply_events(sdk, emit, "r1", LONG_REPLY, voiced(12.8))
        else:
            reply_events(sdk, emit, "r2", "A navy shirt would work well.", voiced(1))

    calls = ScriptedCalls(answer=answer, every_turn=True)
    conversation = realtime_service(rig, None, calls, wake_timeout_s=10)
    conversation.silence_s = 2.0
    link = companion_hearing(conversation, forced_endpoints=True)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        call = conversation._voice[1].call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        await room(rig, 3)
        caplog.clear()
        before = len(calls.frames)
        await say_over_reply(rig, link, words, STRETCHES[stretches])
        await room(rig, 1.5)
        await rig.wait_until_or_timeout(lambda: replies == [1, 2] and not call.audible, 5)
        heard.append((voiced_ms(calls.frames[before:]), calls.barge_ins, call.pending, [
            line for line in caplog.messages if line.startswith(("Realtime command ", "Realtime turn ordinal=2 "))
            or "could not transcribe" in line]))
        heard.append(round(sum(len(chunk) for chunk in rig.hardware.played) / rig.hardware.output_sample_rate, 1))

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking"), timeout=40)
    voice_ms = sum(voiced_ms_ for voiced_ms_, _ in STRETCHES[stretches])
    assert heard[0][:3] == (voice_ms, 1, [])
    assert [line.split(" voice_to_reply_ms=")[0] for line in heard[0][3]] == [
        "Realtime command transcript=arrived words=9 wake_prefix=none outcome=none drop=none",
        "Realtime turn ordinal=2 outcome=answered words=9"]
    assert heard[1] < 12.8


def gapped(words):
    """Each of ``words`` its own stretch, as in STRETCHES["each word its own stretch"], with no quiet after the last."""
    pattern = STRETCHES["each word its own stretch"]
    stretches = [pattern[index % len(pattern)] for index in range(len(words.split()))]
    return (*stretches[:-1], (stretches[-1][0], 0))


def said_over_reply(monkeypatch, caplog, words, stretches, *, reply=LONG_REPLY, after_s=3, sound=None):
    """Say ``words`` in ``stretches``, ``after_s`` into Muse's 12.8 s ``reply``, in the user's test voice or ``sound``.
    Returns the barge-ins, the user's voiced ms and the frames of Muse's own voice that reached Muse, and Reachy's
    command lines, once Reachy's endpoint has passed; then how many seconds of the reply played."""
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True

    async def answer(sdk, emit, script):
        reply_events(sdk, emit, "r1", reply, voiced(12.8))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, None, calls, wake_timeout_s=10)
    conversation.silence_s = 2.0
    link = companion_hearing(conversation, forced_endpoints=True)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        call = conversation._voice[1].call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        await room(rig, after_s)
        caplog.clear()
        before = len(calls.frames)
        await say_over_reply(rig, link, words, stretches, sound=sound)
        await room(rig, 1.6)
        heard.append((calls.barge_ins, voiced_ms(calls.frames[before:]), echo_frames(calls.frames[before:]),
                      [line for line in caplog.messages
                       if line.startswith("Realtime command ") or "could not transcribe" in line]))
        await rig.wait_until_or_timeout(lambda: not call.audible, 12)
        heard.append(round(sum(len(chunk) for chunk in rig.hardware.played) / rig.hardware.output_sample_rate, 2))

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking"), timeout=40)
    return heard


def said_at(reply, words):
    """About when Muse's test voice says ``words``, in seconds into ``reply``: that voice has no pauses to place the
    words by, so Reachy places them at 15 characters a second, with a pause at each comma and sentence break."""
    before = reply[:reply.index(words)]
    return len(before) / 15 + .15 * before.count(",") + .35 * len(re.findall(r"[.!?] ", before))


MUSEUM_REPLY = ("The museum is open until six tonight, and the cafe is on the second floor, next to the gift shop. "
                "The new robot exhibit is on the ground floor, and it is very popular, so go early.")


@pytest.mark.parametrize("reply, words", [
    (LONG_REPLY, "What time is it?"), (LONG_REPLY, "Can you repeat that?"), (LONG_REPLY, "What about Friday then?"),
    (MUSEUM_REPLY, "Is it free on Sundays?")],
    ids=["What time is it?", "Can you repeat that?", "What about Friday then?", "Is it free on Sundays?"])
def test_a_short_request_opening_with_a_word_of_muse_s_reply_reaches_muse_at_reachy_s_endpoint(
        monkeypatch, caplog, reply, words):
    """A short request said in short words over Muse's reply opens with a word the reply also holds. The companion's
    partial is that one word until most of the request is said, so no partial confirms it. The voice under that word
    still counts toward the turn's final words: Muse is cut off and hears every word."""
    heard = said_over_reply(monkeypatch, caplog, words, gapped(words), reply=reply)
    assert heard[0] == (1, sum(voiced_ms_ for voiced_ms_, _ in gapped(words)), 0, [
        f"Realtime command transcript=arrived words={len(words.split())} wake_prefix=none outcome=none drop=none"])


OFFER_REPLY = ("I can read you the news, check the weather, or set a reminder for you. Just tell me which one you "
               "want, and I can also play some music while you get ready.")
FORECAST_REPLY = ("Tomorrow will be sunny with a high of 72. Would you like the forecast for next week too? I can also "
                  "tell you about the pollen count, or what to wear.")


@pytest.mark.parametrize("reply, then, words, stretches", [
    (FORECAST_REPLY, "I can also", "Yes, the forecast for next week please.",
     gapped("Yes, the forecast for next week please.")),
    (OFFER_REPLY, "Just tell me", "Okay, set a reminder for me.", gapped("Okay, set a reminder for me.")),
    (FORECAST_REPLY, "I can also", "Yes, the forecast for next week please.",
     STRETCHES["fluent, ending on a long stretch"][:7])],
    ids=["the forecast Muse offered", "the reminder Muse offered", "control: said fluently"])
def test_a_request_taking_up_muse_s_offer_in_its_words_reaches_muse_at_reachy_s_endpoint(
        monkeypatch, caplog, reply, then, words, stretches):
    """Muse makes an offer and talks on, and the user takes the offer up over the reply in short words, in Muse's own
    words, as Muse's echo of the offer would hold them. But echo holds only words Muse played while the voice was
    heard, and Muse made the offer before then: Muse is cut off and hears every word. Said fluently, the same words
    confirm by a partial."""
    heard = said_over_reply(monkeypatch, caplog, words, stretches, reply=reply, after_s=said_at(reply, then))
    assert heard[0] == (1, sum(voiced_ms_ for voiced_ms_, _ in stretches), 0, [
        f"Realtime command transcript=arrived words={len(words.split())} wake_prefix=none outcome=none drop=none"])


@pytest.mark.parametrize("words", ["Got it.", "Oh, nice.", "Mm-hmm, okay."])
def test_a_short_remark_only_its_endpoint_confirms_never_cuts_muse_off(monkeypatch, caplog, words):
    """A remark of two words or fewer besides back-channel sounds, said in short words over Muse's reply, is confirmed
    only at Reachy's endpoint, two seconds after it. It asks Muse nothing, so it is dropped as unconfirmed: none of it
    reaches Muse, and Muse's reply plays on whole."""
    assert said_over_reply(monkeypatch, caplog, words, gapped(words)) == [(0, 0, 0, [
        f"Realtime command transcript=arrived words={len(words.split())} wake_prefix=none outcome=none drop=barge_in"]),
        12.8]


@pytest.mark.parametrize("words, outcome", [("Never mind.", "STOP_TALKING"), ("Stop talking.", "STOP_TALKING")])
def test_a_cancel_or_silence_phrase_only_its_endpoint_confirms_still_cuts_muse_off(monkeypatch, caplog, words, outcome):
    """As the remark above, but the words ask Muse to stop: Reachy stops Muse's reply at its endpoint."""
    heard = said_over_reply(monkeypatch, caplog, words, gapped(words))
    assert heard[0][0] == 1
    assert heard[0][3] == [
        f"Realtime command transcript=arrived words=2 wake_prefix=none outcome={outcome} drop=none"]
    assert heard[1] < 12.8


@pytest.mark.parametrize("words, stretches, outcome, heard", [
    ("Reachy, turn the volume down please.", STRETCHES["fluent, ending on a long stretch"][3:], "VOLUME_DOWN", True),
    ("Reachy, turn the volume down please.", STRETCHES["each word its own stretch"][3:], "VOLUME_DOWN", False),
    ("Stop talking.", ((200, 120), (240, 0)), "STOP_TALKING", False)],
    ids=["volume down confirmed while said", "volume down", "stop talking"])
def test_a_device_command_over_a_long_reply_is_obeyed_and_muse_never_answers_it(
        monkeypatch, caplog, words, stretches, outcome, heard):
    """As the request above, with a device command. Said in stretches each under 300 ms, it is confirmed only at
    Reachy's endpoint: Reachy obeys it and cuts Muse off, and Muse never hears it. Muse answers every turn it hears,
    and the call takes the microphone at the SDK's pace, so a command handed to Muse at its endpoint would reach Muse
    after Reachy's cut and be answered. Said fluently, a partial confirms it as it is said, so Muse hears it, and
    Reachy's cut drops Muse's answer. Either way, none of Muse's voice plays once Reachy has obeyed."""
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    replies = []

    async def answer(sdk, emit, script):
        replies.append(len(replies) + 1)
        if replies == [1]:
            reply_events(sdk, emit, "r1", LONG_REPLY, voiced(12.8))
        else:
            reply_events(sdk, emit, f"r{len(replies)}", "Okay, done.", voiced(2))

    calls = ScriptedCalls(answer=answer, every_turn=True, paced=True)
    conversation = realtime_service(rig, None, calls, wake_timeout_s=10)
    conversation.silence_s = 2.0
    link = companion_hearing(conversation, forced_endpoints=True)
    obeyed, voice_to_muse = [], []

    async def user():
        await speak_live(rig, wake=True)
        call = conversation._voice[1].call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        await room(rig, 3)
        caplog.clear()
        before = len(calls.frames)
        await say_over_reply(rig, link, words, stretches)
        await room(rig, 1.6)
        await rig.wait_until_or_timeout(
            lambda: any(line.startswith("Realtime command ") for line in caplog.messages), .2)
        obeyed.append(time.monotonic())
        await room(rig, 5)
        voice_to_muse.append(voiced_ms(calls.frames[before:]))

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking"), timeout=40)
    assert [line for line in caplog.messages if line.startswith("Realtime command ")] == [
        f"Realtime command transcript=arrived words={len(words.split())} wake_prefix=none outcome={outcome} drop=none"]
    assert rig.hardware.volume_steps == ([-10] if outcome == "VOLUME_DOWN" else [])
    assert (voice_to_muse[0] > 0, muse_played_after(rig, obeyed[0])) == (heard, 0.0)


def test_a_command_its_final_words_confirmed_never_reaches_muse_through_the_turn_said_after_it(monkeypatch, caplog):
    """The user says a command in short words over Muse's reply and talks straight on past the companion's end of it,
    so the words after it share its gate. Those words are a turn of their own, decided while Reachy's command model
    still weighs the command. Only the command's turn, whose final words confirmed the gate, decides where the held
    audio goes, so the later turn never sends Muse the command. The words said on go where the command goes."""
    from reachy_muse import device_command
    from reachy_muse.device_command import DeviceCommand
    from reachy_muse.protocol import HearEndpoint, encode
    from test_device_command_turns import QUIET_DOWN

    caplog.set_level("INFO")
    monkeypatch.setattr(device_command, "CLASSIFY_DEADLINE_S", 2.0)
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True

    async def answer(sdk, emit, script):
        reply_events(sdk, emit, "r1", LONG_REPLY, voiced(12.8))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, None, calls, wake_timeout_s=10,
                                    model=FakeModel(DeviceCommand.VOLUME_DOWN, delay_s=1))
    conversation.silence_s = 2.0
    link = companion_hearing(conversation)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        call = conversation._voice[1].call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        await room(rig, 3)
        caplog.clear()
        before, known, streams, voice = len(calls.frames), set(link._connection.hearing), [], []
        for voiced_ms_, quiet_ms in STRETCHES["each word its own stretch"][:7]:
            voice += [True] * (voiced_ms_ // 20) + [False] * (quiet_ms // 20)
        said_on = len(voice)
        voice += [True] * 10 + [False] * 10
        for frame, voiced in enumerate(voice):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE if voiced else 0, np.float32))
            await asyncio.sleep(CHUNK / 16000)
            streams += [stream for stream in link._connection.hearing if stream not in known and stream not in streams]
            if frame == said_on:
                link._receive(link._connection, encode(HearEndpoint(streams[0], QUIET_DOWN, False)))
            if frame == said_on + 10:
                link._receive(link._connection, encode(HearEndpoint(streams[1], "and thanks", False)))
        await room(rig, 1.5)
        heard.append((voiced_ms(calls.frames[before:]), rig.hardware.volume_steps,
                      [line.split(" wake_prefix=")[1] for line in caplog.messages
                       if line.startswith("Realtime command ")]))

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking"), timeout=40)
    assert heard == [(0, [-10], ["none outcome=none drop=none", "none outcome=VOLUME_DOWN drop=none"])]


@pytest.mark.parametrize("last_word_ms", [280, 520], ids=["its endpoint confirms it", "its last word confirms it"])
def test_reachy_never_nods_that_muse_is_slow_while_a_request_held_over_its_reply_still_reaches_muse(
        monkeypatch, caplog, last_word_ms):
    """A long request said over Muse's reply in short words, with a pause in it, is confirmed late: at Reachy's
    endpoint, or by a partial once its last word is a stretch long enough. Either way it reaches Muse at the SDK's
    pace for longer than Reachy waits before nodding that Muse is slow. Muse answers once the request has gone 0.6 s
    without voice, as its own voice detector does. Reachy's wait starts once Muse has the whole request, so it never
    nods."""
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    replies = []

    async def answer(sdk, emit, script):
        replies.append(len(replies) + 1)
        if replies == [1]:
            reply_events(sdk, emit, "r1", LONG_REPLY, voiced(12.8))
        elif replies == [1, 2]:
            while True:
                loud = [at for frame, at in zip(script.frames, script.frame_times)
                        if np.frombuffer(frame, "<i2").max() >= 5000]
                if time.monotonic() - loud[-1] >= .6:
                    break
                await asyncio.sleep(.02)
            reply_events(sdk, emit, "r2", "A navy shirt would work well.", voiced(2))

    calls = ScriptedCalls(answer=answer, every_turn=True, paced=True)
    conversation = realtime_service(rig, None, calls, wake_timeout_s=10)
    conversation.silence_s = 2.0
    link = companion_hearing(conversation, forced_endpoints=True)
    pattern = STRETCHES["each word its own stretch"]

    async def user():
        await speak_live(rig, wake=True)
        call = conversation._voice[1].call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        await room(rig, 2.5)
        caplog.clear()
        await say_over_reply(rig, link, "Reachy, I'm going to a dinner tonight at a fancy place downtown, what "
                             "should I wear to look good?", pattern[:8] + ((280, 60),) + pattern[:2] + ((280, 1000),)
                             + pattern[:6] + ((last_word_ms, 0),))
        await room(rig, 12)

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking"), timeout=60)
    assert [line.split(" voice_to_reply_ms=")[0] for line in caplog.messages
            if line.startswith("Realtime turn ordinal=2 ")] == ["Realtime turn ordinal=2 outcome=answered words=19"]


NUMBER_REPLY = ("Right now it's 72 degrees and sunny, and by eight tonight it will drop to 64, so bring a jacket "
                "if you go out later.")
WEATHER_REPLY = ("Tomorrow will be sunny with a high of 72, and the weekend looks rainy, so you might want an "
                 "umbrella on Saturday. Would you like the forecast for next week too?")


@pytest.mark.parametrize("reply, said, words, stretches", [
    *((LONG_REPLY, "Tell me", "Tell me what you're wearing and I'll suggest something", STRETCHES[name])
      for name in STRETCHES),
    (LONG_REPLY, "Tell me", "Tell me what you are wearing and I'll suggest",
     gapped("Tell me what you are wearing and I'll suggest")),
    (NUMBER_REPLY, "it's 72", "it's seventy two degrees and sunny and by eight",
     gapped("it's seventy two degrees and sunny and by eight")),
    (LONG_REPLY, "Tell me", "Tell me what your wearing and I'll suggest something",
     gapped("Tell me what your wearing and I'll suggest something")),
    (LONG_REPLY, "wearing", "wearing suggest something tonight ask about the weather first",
     ((240, 360),) * 8 + ((240, 0),)),
    (LONG_REPLY, "Tell me", "tell me what your wear in and I just", gapped("tell me what your wear in and I just")),
    (WEATHER_REPLY, "and the weekend", "the week and looks rain he so you",
     gapped("the week and looks rain he so you")),
    (LONG_REPLY, "something for", "some four tonight", gapped("some four tonight"))],
    ids=[*STRETCHES, "a contraction written out", "a number written out", "a substitution", "fragments",
         "five in nine words wrong", "four in eight words wrong", "two in three words wrong"])
def test_muse_s_echo_heard_in_gapped_stretches_over_its_reply_never_reaches_it_at_reachy_s_endpoint(
        monkeypatch, caplog, reply, said, words, stretches):
    """As the request above, but the sound is Muse's own reply coming back through the speaker from when Muse says
    ``said``, and the companion hears its words: with the robot's 2 s of silence, the turn's words at Reachy's
    endpoint confirm nothing. So too when the companion writes some words differently from Muse's transcript, gets
    two in five of them wrong, or hears only some of them."""
    assert said_over_reply(monkeypatch, caplog, words, stretches, reply=reply, after_s=said_at(reply, said),
                           sound=muse_heard(1)) == [(0, 0, 0, [
        f"Realtime command transcript=arrived words={len(words.split())} wake_prefix=none outcome=none drop=barge_in"]),
        12.8]


def test_echo_the_companion_hears_no_words_in_never_reaches_muse_ahead_of_the_request_after_it(monkeypatch):
    """Muse's voice comes back through the speaker in three short bursts the companion finds no words in, and a second
    later the user asks something in the same breath, fluently enough that a partial confirms it. Muse gets the
    request alone, as on main, and not its own voice in front of it."""
    from reachy_muse.protocol import HearPartial, encode

    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True

    async def answer(sdk, emit, script):
        reply_events(sdk, emit, "r1", LONG_REPLY, voiced(12.8))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, None, calls, wake_timeout_s=10)
    conversation.silence_s = 2.0
    link = companion_hearing(conversation, forced_endpoints=True)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        call = conversation._voice[1].call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        await room(rig, 3)
        before, known, echo = len(calls.frames), set(link._connection.hearing), muse_heard(1)
        sounds = (["echo"] * 10 + [None] * 5 + ["echo"] * 9 + [None] * 5 + ["echo"] * 10 + [None] * 50
                  + ["user"] * 25 + [None] * 5 + ["user"] * 25 + [None] * 40)
        # Each partial shows COMPANION_LAG_S after the user's stretch of voice that ends it.
        partials = {113 + 24: "What about", 143 + 24: "What about Friday then"}
        for frame, sound in enumerate(sounds):
            chunk = (echo[frame * CHUNK % len(echo):][:CHUNK].copy() if sound == "echo"
                     else np.full(CHUNK, VOICE_SAMPLE if sound == "user" else 0, np.float32))
            rig.hardware.samples.put(chunk)
            await asyncio.sleep(CHUNK / 16000)
            stream = next((opened for opened in link._connection.hearing if opened not in known), None)
            if stream is not None and frame in partials:
                link._receive(link._connection, encode(HearPartial(stream, frame, partials[frame])))
        heard.append((voiced_ms(calls.frames[before:]), echo_frames(calls.frames[before:]), calls.barge_ins))

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking"), timeout=40)
    assert heard == [(1000, 0, 1)]


@pytest.mark.parametrize("muse, barge_ins, pending", [("works", 0, [1, 2, 3]), ("talks", 1, [2, 3]),
                                                     ("has finished", 0, [2, 3])])
def test_speech_that_runs_on_past_a_companion_endpoint_reaches_muse_without_a_gap(monkeypatch, muse, barge_ins, pending):
    """The companion ends the user's turn while the user is still talking, after the wake deadline.

    A command model that takes its time, as on the robot, holds up each turn's acceptance."""
    from reachy_muse.protocol import HearEndpoint, HearPartial, encode

    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    words = "Actually, I love your voice, tell me about trains."

    async def answer(sdk, emit, script):
        if muse == "works":
            emit(sdk.Working(True))
        else:
            reply_events(sdk, emit, "r1", REPLY, voiced(7.5 if muse == "talks" else .5))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, None, calls, model=FakeModel(None, delay_s=.25))
    link = companion_hearing(conversation)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        if muse == "has finished":
            while not ("muse" in played(rig.recording.events) and not call.audible):
                await room(rig, .04)
        deadline = voice._wake_deadline
        await room(rig, deadline - time.monotonic() + (-.3 if muse == "has finished" else .3))
        due = {"works": lambda: chunks >= 25, "talks": lambda: calls.barge_ins > 0,
               "has finished": lambda: time.monotonic() > deadline}[muse]
        before = len(calls.frames)
        streams, ended = [], False
        for chunks in range(1, 51):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
            for stream in [stream for stream in link._connection.hearing if stream not in streams]:
                streams.append(stream)
                link._receive(link._connection, encode(HearPartial(stream, 1, words)))
            if streams and not ended and due():
                link._receive(link._connection, encode(HearEndpoint(streams[0], words, False)))
                ended = True
        for _ in range(100):
            if call.turns == 3:
                break
            await room(rig, .04)
        heard.append((gaps_ms(calls.frames[before:]), voiced_ms(calls.frames[before:]), calls.barge_ins,
                      [turn.ordinal for turn in call.pending]))

    run(rig, conversation, user, rest=("thinking", "listening"))
    assert heard == [([], 1000, barge_ins, pending)]


@pytest.mark.parametrize("turn_ends, pause_s", [("mid-speech", 0.0), ("mid-speech", .3), ("after speech", 0.0)])
def test_after_a_barge_in_muse_s_next_reply_never_streams_back_into_muse(monkeypatch, turn_ends, pause_s):
    """The user talks over Muse until Reachy cuts the reply off, says a little more and stops, and Muse answers at
    once. The microphone hears that answer, and the companion transcribes it into whatever turn is open.
    Mid-speech, the companion ends the user's first turn while the user is still talking."""
    from reachy_muse.protocol import HearEndpoint, HearPartial, encode

    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    said = {}
    next_reply = []

    def hear(stream, text):
        said[stream] = text
        link._receive(link._connection, encode(HearPartial(stream, len(said), text)))

    async def answer(sdk, emit, script):
        next_reply.append(lambda: reply_events(sdk, emit, "r2", REPLY, voiced(3)))
        reply_events(sdk, emit, "r1", REPLY, voiced(7.5))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, None, calls, wake_timeout_s=30)
    link = companion_hearing(conversation)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        await room(rig, .5)
        ended = False
        for chunks in range(1, 41):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
            for stream in [stream for stream in link._connection.hearing if stream not in said]:
                hear(stream, "No, wait, I mean boats." if said else "Actually, I love your voice, tell me about trains.")
            if turn_ends == "mid-speech" and not ended and calls.barge_ins and chunks >= 30:
                link._receive(link._connection, encode(HearEndpoint(next(iter(said)), "Actually, I love your voice", False)))
                ended = True
        await room(rig, pause_s)
        next_reply[0]()
        await rig.wait_until_or_timeout(lambda: call.audible, 1)
        open_before = [stream for stream in link._connection.hearing if stream in said]
        before = len(calls.frames)
        echo = muse_heard(1.0)
        for start in range(0, len(echo), CHUNK):
            rig.hardware.samples.put(echo[start:start + CHUNK])
            await asyncio.sleep(CHUNK / 16000)
            for stream in [stream for stream in link._connection.hearing if stream not in said]:
                hear(stream, "Great news, it will be sunny tomorrow!")
            if start == 10 * CHUNK:
                for stream in open_before:
                    hear(stream, said[stream] + " Great news, it will be sunny tomorrow!")
        await room(rig, .5)
        heard.append((calls.barge_ins, ended, echo_frames(calls.frames[before:]), len(calls.frames) - before >= 25,
                      voice.call is call))

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking"), timeout=30)
    assert heard == [(1, turn_ends == "mid-speech", 0, True, True)]


@pytest.mark.parametrize("partial_at_reply", [
    "Actually, I love your voice, tell me about tra", "Actually, I love your voice"], ids=["mid-word", "a word behind"])
def test_a_transcript_still_catching_up_when_muse_answers_gets_the_whole_turn_and_no_echo(
        monkeypatch, caplog, partial_at_reply):
    """The user talks over Muse until Reachy cuts the reply off, stops, and Muse answers at once. The companion's
    transcript of the user's turn has not caught up with the voice when the answer starts: its forced endpoint, a
    moment later, holds the whole turn. The echo of the answer is transcribed on a stream opened after it starts."""
    from reachy_muse.protocol import HearEndpoint, HearPartial, encode

    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    words = "Actually, I love your voice, tell me about trains."
    next_reply = []

    async def answer(sdk, emit, script):
        next_reply.append(lambda: reply_events(sdk, emit, "r2", REPLY, voiced(3)))
        reply_events(sdk, emit, "r1", REPLY, voiced(7.5))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, None, calls, wake_timeout_s=30)
    link = companion_hearing(conversation)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        await room(rig, .5)
        streams = []
        for _ in range(40):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
            for stream in [stream for stream in link._connection.hearing if stream not in streams]:
                streams.append(stream)
                link._receive(link._connection, encode(HearPartial(stream, 1, partial_at_reply)))
        next_reply[0]()
        await rig.wait_until_or_timeout(lambda: call.audible, 1)
        caplog.clear()
        before = len(calls.frames)
        echo = muse_heard(1.2)
        for frame, start in enumerate(range(0, len(echo), CHUNK)):
            rig.hardware.samples.put(echo[start:start + CHUNK])
            await asyncio.sleep(CHUNK / 16000)
            if frame == 3:
                link._receive(link._connection, encode(HearPartial(streams[0], 2, words)))
                link._receive(link._connection, encode(HearEndpoint(streams[0], words, True)))
            for stream in [stream for stream in link._connection.hearing if stream not in streams]:
                streams.append(stream)
                link._receive(link._connection, encode(HearPartial(
                    stream, 1, "Great news, it will be sunny tomorrow!")))
        await room(rig, .5)
        heard.append((calls.barge_ins, echo_frames(calls.frames[before:]), voice.call is call))

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking"), timeout=30)
    assert heard == [(1, 0, True)]
    assert [line for line in caplog.messages if line.startswith("Realtime command ")][0] == (
        "Realtime command transcript=arrived words=9 wake_prefix=none outcome=none drop=none")


def test_saying_it_again_over_muse_s_next_reply_interrupts_that_reply_too(monkeypatch):
    """The user talks over Muse, Muse starts its next reply while the user is still talking, and the user says it
    again. The companion ends the first turn mid-speech, so the repeat is a turn of its own with the same words."""
    from reachy_muse.protocol import HearEndpoint, HearPartial, encode

    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    words = "No, wait, I mean boats."
    next_reply = []

    async def answer(sdk, emit, script):
        next_reply.append(lambda: reply_events(sdk, emit, "r2", REPLY, voiced(6)))
        reply_events(sdk, emit, "r1", REPLY, voiced(7.5))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, None, calls, wake_timeout_s=30)
    link = companion_hearing(conversation)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        await room(rig, .5)
        streams, repeat = [], None
        for chunks in range(1, 91):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
            for stream in [stream for stream in link._connection.hearing if stream not in streams]:
                streams.append(stream)
                link._receive(link._connection, encode(HearPartial(stream, 1, words if len(streams) == 1
                                                                   else "No, wait, I mean boats!")))
            if chunks == 40:
                heard.append(calls.barge_ins)
                next_reply[0]()
                await rig.wait_until_or_timeout(lambda: call.audible, 1)
            if chunks == 45:
                link._receive(link._connection, encode(HearEndpoint(streams[0], words, False)))
            if chunks == 60:
                repeat = len(calls.frames)
        await room(rig, .5)
        heard.append((calls.barge_ins, voiced_ms(calls.frames[repeat:]) >= 500, len(streams), call.audible))

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking"), timeout=30)
    assert heard == [1, (2, True, 2, False)]


@pytest.mark.parametrize("command, replies, done", [
    ("Stop listening.", 1, "dismissed"), ("Volume down.", 1, [-10]), ("Stop listening.", 2, "dismissed")])
def test_a_command_ended_by_muse_s_reply_is_obeyed_on_its_last_partial_if_the_companion_never_ends_it(
        monkeypatch, command, replies, done):
    """The user says a device command over Muse's reply, which cuts the reply off, and Muse starts ``replies``
    more before Reachy's endpoint for the turn: Muse's own endpoint usually comes first, and the first of them ends
    the turn. The companion never answers the turn's finish, so the command's latest partial stands in."""
    from reachy_muse.companion_client import ENDPOINT_WAIT_S
    from reachy_muse.device_command import WakeWindow
    from reachy_muse.protocol import HearPartial, encode

    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    next_reply = []

    async def answer(sdk, emit, script):
        def answer_again():
            reply_events(sdk, emit, "r2", "Okay.", voiced(2 if replies == 1 else .2))
            if replies == 2:
                reply_events(sdk, emit, "r3", "Done.", voiced(1))

        next_reply.append(answer_again)
        reply_events(sdk, emit, "r1", REPLY, voiced(7.5))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, None, calls, wake_timeout_s=30)
    link = companion_hearing(conversation)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        await room(rig, .5)
        streams = []
        for _ in range(30):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
            for stream in [stream for stream in link._connection.hearing if stream not in streams]:
                streams.append(stream)
                link._receive(link._connection, encode(HearPartial(stream, 1, command)))
        heard.append(calls.barge_ins)

        def outcome():
            return "dismissed" if voice._wake_window is WakeWindow.DISMISSED else rig.hardware.volume_steps

        replied = time.monotonic()
        next_reply[0]()
        while outcome() != done and time.monotonic() < replied + ENDPOINT_WAIT_S + 2:
            await room(rig, .02)
        heard.append((outcome(), ENDPOINT_WAIT_S <= time.monotonic() - replied < ENDPOINT_WAIT_S + .5))

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking", "sleeping"), timeout=30)
    assert heard == [1, (done, True)]


@pytest.mark.parametrize("command, done", [("Stop listening.", "dismissed"), ("Volume down.", [-10])])
def test_a_command_ended_by_muse_s_reply_is_obeyed_on_its_last_partial_if_reachy_s_recognizer_never_ends_it(
        monkeypatch, command, done):
    """As with the companion, on Reachy's own streaming recognition: the user says a device command over Muse's
    reply, which cuts the reply off, and Muse starts its next reply before Reachy's endpoint for the turn. The
    recognizer never brings the turn's final words, so the command's latest partial stands in."""
    from reachy_muse.companion_client import ENDPOINT_WAIT_S
    from reachy_muse.device_command import WakeWindow

    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    next_reply = []

    async def answer(sdk, emit, script):
        next_reply.append(lambda: reply_events(sdk, emit, "r2", "Okay.", voiced(2)))
        reply_events(sdk, emit, "r1", REPLY, voiced(7.5))

    class NoFinal(StreamingTranscripts):
        def open_turn(self):
            turn = super().open_turn()
            if turn.partial.text == command:
                turn.finish = lambda: asyncio.get_running_loop().create_future()
            return turn

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, NoFinal("Hey Muse, tell me about trains.", command), calls,
                                    wake_timeout_s=30)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        await room(rig, .5)
        for _ in range(30):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
        heard.append(calls.barge_ins)

        def outcome():
            return "dismissed" if voice._wake_window is WakeWindow.DISMISSED else rig.hardware.volume_steps

        replied = time.monotonic()
        next_reply[0]()
        while outcome() != done and time.monotonic() < replied + ENDPOINT_WAIT_S + 2:
            await room(rig, .02)
        heard.append((outcome(), ENDPOINT_WAIT_S <= time.monotonic() - replied < ENDPOINT_WAIT_S + .5))

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking", "sleeping"), timeout=30)
    assert heard == [1, (done, True)]


# The reviewer's repros of the open turn ending where Muse's next reply starts, with the companion sending words
# said after that start on a stream opened after it, and answering a finish with a forced endpoint.

ECHO = muse_heard(1.0)


@pytest.mark.parametrize("command, outcome", [("Stop listening.", "dismissed"), ("Volume down.", [-10])])
def test_a_command_said_over_muse_is_obeyed_even_if_muse_answers_before_the_turn_ends(monkeypatch, command, outcome):
    """The user says a device command over Muse's reply, which cuts the reply; Muse starts its next reply before
    Reachy's endpoint for that turn. Reachy must still obey the command, as on main."""
    from reachy_muse.device_command import WakeWindow
    from reachy_muse.protocol import HearPartial, encode

    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    next_reply = []

    async def answer(sdk, emit, script):
        next_reply.append(lambda: reply_events(sdk, emit, "r2", "Okay.", voiced(2)))
        reply_events(sdk, emit, "r1", REPLY, voiced(7.5))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, None, calls, wake_timeout_s=30)
    link = companion_hearing(conversation, forced_endpoints=True)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        await room(rig, .5)
        streams = []
        for _ in range(30):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
            for stream in [stream for stream in link._connection.hearing if stream not in streams]:
                streams.append(stream)
                link._receive(link._connection, encode(HearPartial(stream, 1, command)))
        assert calls.barge_ins == 1
        next_reply[0]()
        await room(rig, 1.0)
        heard.append("dismissed" if voice._wake_window is WakeWindow.DISMISSED else rig.hardware.volume_steps)

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking", "sleeping"), timeout=30)
    assert heard == [outcome]


def test_a_short_afterthought_said_as_muse_starts_answering_reaches_muse(monkeypatch):
    """The user talks over Muse until Reachy cuts the reply, keeps talking as Muse starts its next reply, and adds
    "and boats too" in that reply's first 0.3 s. Those words come as partials of a stream opened after it starts."""
    from reachy_muse.protocol import HearPartial, encode

    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    next_reply = []

    async def answer(sdk, emit, script):
        next_reply.append(lambda: reply_events(sdk, emit, "r2", REPLY, voiced(4)))
        reply_events(sdk, emit, "r1", REPLY, voiced(7.5))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, None, calls, wake_timeout_s=30)
    conversation.silence_s = 1.2               # Reachy's end of turn with a companion, later than its words
    link = companion_hearing(conversation)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        await room(rig, .5)
        streams = []
        for _ in range(40):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
            for stream in [stream for stream in link._connection.hearing if stream not in streams]:
                streams.append(stream)
                link._receive(link._connection, encode(HearPartial(stream, 1, "Actually, tell me about trains")))
        assert calls.barge_ins == 1
        next_reply[0]()
        await rig.wait_until_or_timeout(lambda: call.audible, 1)
        before = len(calls.frames)
        for frame in range(40):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE if frame < 15 else 0, np.float32))
            await asyncio.sleep(CHUNK / 16000)
            if frame == 25:
                for stream in [stream for stream in link._connection.hearing if stream not in streams]:
                    link._receive(link._connection, encode(HearPartial(stream, 2, "and boats too")))
        await room(rig, .5)
        heard.append(voiced_ms(calls.frames[before:]) >= 250)

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking"), timeout=30)
    assert heard == [True]


def test_an_afterthought_on_its_own_stream_reaches_muse_when_the_request_ended_before_muse_answered(monkeypatch):
    """As the branch's amended afterthought test, with Reachy's 1.2 s end of turn, but the request ends before Muse's
    first reply, so the afterthought's own words, on the stream opened after Muse's next reply starts, are the only
    words that can confirm it."""
    from reachy_muse.protocol import HearPartial, encode

    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    next_reply = []

    async def answer(sdk, emit, script):
        next_reply.append(lambda: reply_events(sdk, emit, "r2", REPLY, voiced(4)))
        await asyncio.sleep(1.6)
        reply_events(sdk, emit, "r1", REPLY, voiced(7.5))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, None, calls, wake_timeout_s=30)
    conversation.silence_s = 1.2
    link = companion_hearing(conversation)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        while not call.audible:
            await room(rig, .04)
        await room(rig, .5)
        streams = []
        for _ in range(40):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
            for stream in [stream for stream in link._connection.hearing if stream not in streams]:
                streams.append(stream)
                link._receive(link._connection, encode(HearPartial(stream, 1, "Actually, tell me about trains")))
        assert calls.barge_ins == 1
        next_reply[0]()
        await rig.wait_until_or_timeout(lambda: call.audible, 1)
        before = len(calls.frames)
        for frame in range(40):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE if frame < 15 else 0, np.float32))
            await asyncio.sleep(CHUNK / 16000)
            if frame == 25:
                for stream in [stream for stream in link._connection.hearing if stream not in streams]:
                    link._receive(link._connection, encode(HearPartial(stream, 2, "and boats too")))
        await room(rig, .5)
        heard.append((len(streams), calls.barge_ins, voiced_ms(calls.frames[before:]) >= 250))

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking"), timeout=30)
    assert heard == [(1, 2, True)]


def test_after_a_second_interruption_muse_s_third_reply_never_streams_back_into_muse(monkeypatch):
    """The user cuts Muse off, talks on over its next reply until that is cut too, stops, and Muse answers at once.
    Words said after a reply starts, and the echo of the third, come as partials of streams opened after it."""
    from reachy_muse.protocol import HearPartial, encode

    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    first = "Actually, tell me about trains."
    more = first + " No, wait, I mean boats, the big ones."
    replies = []

    async def answer(sdk, emit, script):
        replies.append(lambda: reply_events(sdk, emit, "r2", REPLY, voiced(4)))
        replies.append(lambda: reply_events(sdk, emit, "r3", REPLY, voiced(3)))
        reply_events(sdk, emit, "r1", REPLY, voiced(7.5))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, None, calls, wake_timeout_s=30)
    link = companion_hearing(conversation)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        await room(rig, .5)
        streams = []
        for _ in range(40):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
            for stream in [stream for stream in link._connection.hearing if stream not in streams]:
                streams.append(stream)
                link._receive(link._connection, encode(HearPartial(stream, 1, first)))
        replies[0]()
        for frame in range(100):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
            if frame == 50:
                for stream in [stream for stream in link._connection.hearing if stream not in streams]:
                    link._receive(link._connection, encode(HearPartial(stream, 2, more.removeprefix(first).lstrip())))
            if calls.barge_ins == 2 and frame > 60:
                break
        heard.append(calls.barge_ins)
        replies[1]()
        await rig.wait_until_or_timeout(lambda: call.audible, 1)
        before = len(calls.frames)
        echo = ECHO[:16000]
        for frame, start in enumerate(range(0, len(echo), CHUNK)):
            rig.hardware.samples.put(echo[start:start + CHUNK].copy())
            await asyncio.sleep(CHUNK / 16000)
            if frame == 10:
                for stream in [stream for stream in link._connection.hearing if stream not in streams]:
                    link._receive(link._connection, encode(HearPartial(
                        stream, 3, "Great news, it will be sunny tomorrow!")))
        await room(rig, .5)
        heard.append((calls.barge_ins, echo_frames(calls.frames[before:])))

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking"), timeout=30)
    assert heard == [2, (2, 0)]


@pytest.mark.parametrize("muse_says", ["Okay.", "Okay, I'll stop listening."])
def test_stop_listening_over_muse_is_obeyed_even_when_muse_s_next_reply_repeats_it(monkeypatch, muse_says):
    """As the command repro, with the companion answering the finish, but Muse's next reply, which starts before
    Reachy ends the turn, says the command's words back. The turn ended where that reply began."""
    from reachy_muse.device_command import WakeWindow
    from reachy_muse.protocol import HearPartial, encode

    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    next_reply = []

    async def answer(sdk, emit, script):
        next_reply.append(lambda: reply_events(sdk, emit, "r2", muse_says, voiced(2)))
        reply_events(sdk, emit, "r1", REPLY, voiced(7.5))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, None, calls, wake_timeout_s=30)
    link = companion_hearing(conversation, forced_endpoints=True)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        await rig.wait_until_or_timeout(lambda: call.audible, 5)
        await room(rig, .5)
        streams = []
        for _ in range(30):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
            for stream in [stream for stream in link._connection.hearing if stream not in streams]:
                streams.append(stream)
                link._receive(link._connection, encode(HearPartial(stream, 1, "Stop listening.")))
        assert calls.barge_ins == 1
        next_reply[0]()
        await room(rig, 2.0)
        heard.append(voice._wake_window is WakeWindow.DISMISSED)

    run(rig, conversation, user, rest=("listening", "idle", "thinking", "speaking", "sleeping"), timeout=30)
    assert heard == [True]


def test_room_noise_after_a_turn_the_user_talked_through_does_not_hold_the_window_open(monkeypatch):
    """The companion ends the user's turn on its last voiced frame. Afterwards the room has only a click every half
    second, never three voiced frames in a row, so no turn opens; once Muse has answered, the window still closes."""
    from reachy_muse.protocol import HearEndpoint, HearPartial, encode

    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    answers = []

    async def answer(sdk, emit, script):
        answers.append(lambda: reply_events(sdk, emit, "r2", REPLY, voiced(.3)))
        reply_events(sdk, emit, "r1", REPLY, voiced(.5))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, None, calls)
    link = companion_hearing(conversation)
    click = np.zeros(CHUNK, np.float32)
    click[140:180] = VOICE_SAMPLE
    heard = []

    async def clicks(seconds):
        for _ in range(round(seconds / .5)):
            rig.hardware.samples.put(click.copy())
            await room(rig, .48)

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        while not ("muse" in played(rig.recording.events) and not call.audible):
            await room(rig, .04)
        streams = []
        for _ in range(30):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
            for stream in [stream for stream in link._connection.hearing if stream not in streams]:
                streams.append(stream)
                link._receive(link._connection, encode(HearPartial(stream, 1, "And what about Sunday?")))
        link._receive(link._connection, encode(HearEndpoint(streams[0], "And what about Sunday?", False)))
        rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
        await asyncio.sleep(CHUNK / 16000)
        clicking = asyncio.create_task(clicks(8))
        while call.turns < 2:
            await asyncio.sleep(.01)
        answers[0]()
        await clicking
        closed = voice._wake_deadline is None
        before = len(calls.frames)
        await speak_live(rig)
        heard.append((closed, voiced_ms(calls.frames[before:])))

    run(rig, conversation, user, rest=("listening", "idle", "thinking"), timeout=40)
    assert heard == [(True, 0)]


def test_while_muse_works_past_the_follow_up_window_the_room_stops_reaching_muse_but_the_call_stays(monkeypatch):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(answer=keeps_working)
    conversation = realtime_service(rig, StreamingTranscripts("Book me a table for two."), calls)
    heard = {}

    async def user():
        await speak_live(rig, wake=True)
        await room(rig, 6.0)
        voice = conversation._voice[1]
        heard.update(until=time.monotonic(), call_open=voice.call is not None and not voice.call.done)

    run(rig, conversation, user, rest=("thinking",))
    assert [at for at in calls.frame_times if at > heard["until"] - 2.0] == []
    assert heard["call_open"] is True


def test_new_speech_after_the_wake_deadline_reaches_muse_without_a_wake_while_muse_works(monkeypatch):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(answer=keeps_working)
    transcripts = StreamingTranscripts("Book me a table for two.", "Make it for seven.")
    conversation = realtime_service(rig, transcripts, calls)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        await room(rig, 2.5)
        voice = conversation._voice[1]
        call = voice.call
        assert time.monotonic() > voice._wake_deadline
        before = len(calls.frames)
        await speak_live(rig)
        await rig.wait_until_or_timeout(lambda: len(call.pending) == 2)
        heard.append((voiced_ms(calls.frames[before:]), transcripts.texts, [turn.ordinal for turn in call.pending],
                      voice.call is call and not call.done))

    events = run(rig, conversation, user, rest=("thinking",))
    assert heard == [(500, [], [1, 2], True)]
    assert sent_to_muse(events) == []
    assert len(calls.opened) == 1


async def works_past_the_deadline_then_answers(sdk, emit, script):
    emit(sdk.Working(True))
    await asyncio.sleep(11)
    reply_events(sdk, emit, "r1", REPLY, voiced(1.0))
    emit(sdk.Working(False))


@pytest.mark.parametrize("after_s, reached", [(9, True), (10.5, False)])
def test_once_muse_finishes_speech_needs_no_wake_until_the_follow_up_timeout_ends(monkeypatch, after_s, reached):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(answer=works_past_the_deadline_then_answers)
    transcripts = StreamingTranscripts("Book me a table for two.", "Is there parking nearby?")
    conversation = realtime_service(rig, transcripts, calls, wake_timeout_s=10)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        deadline = voice._wake_deadline
        while not ("muse" in played(rig.recording.events) and not call.audible and not call.working):
            await room(rig, .04)
        finished = time.monotonic()
        await room(rig, finished + after_s - time.monotonic())
        before = calls.loud_after_reply
        await speak_live(rig)
        heard.append((finished > deadline, calls.loud_after_reply > before, transcripts.texts, voice.call is call))

    events = run(rig, conversation, user, rest=("thinking", "idle"), timeout=40)
    assert heard == ([(True, True, [], True)] if reached else [(True, False, ["Is there parking nearby?"], False)])
    assert sent_to_muse(events) == []


async def answers_and_leaves_working_on(sdk, emit, script):
    emit(sdk.Working(True))
    reply_events(sdk, emit, "r1", REPLY, voiced(1.0))


@pytest.mark.parametrize("after_s", [12, 20])
def test_working_left_on_holds_the_microphone_no_longer_than_muse_s_finish_and_the_follow_up_timeout(
        monkeypatch, after_s):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(answer=answers_and_leaves_working_on)
    transcripts = StreamingTranscripts("What's the weather?", "So, the neighbours are coming at six.")
    conversation = realtime_service(rig, transcripts, calls, wake_timeout_s=10)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        while not ("muse" in played(rig.recording.events) and not call.audible):
            await room(rig, .04)
        finished = time.monotonic()
        await room(rig, finished + after_s - time.monotonic())
        before = calls.loud_after_reply
        await speak_live(rig)
        heard.append((call.working, calls.loud_after_reply - before, transcripts.texts, voice.call is call))

    run(rig, conversation, user, rest=("thinking", "idle", "listening"), timeout=45)
    assert heard == [(True, 0, ["So, the neighbours are coming at six."], True)]


def test_speech_admitted_before_the_wake_deadline_can_finish_after_it(monkeypatch):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(answer=keeps_working)
    conversation = realtime_service(rig, StreamingTranscripts("Book a table.", "Make it for three."), calls)
    heard = []

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        voice._wake_deadline = time.monotonic() + .2
        before = calls.loud_after_reply
        await speak_live(rig)
        heard.append(calls.loud_after_reply - before)

    run(rig, conversation, user, rest=("thinking", "listening"))
    assert heard[0] >= 10


async def answer_then_hang_up(sdk, emit, script):
    await answers(REPLY, 1.0)(sdk, emit, script)
    await asyncio.sleep(3.5)
    script.hang_up(ConnectionError("media reset"))


def test_a_call_lost_after_muse_answered_does_not_ask_the_user_to_repeat(monkeypatch):
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    calls = ScriptedCalls(answer_early=True, answer=answer_then_hang_up)
    conversation = realtime_service(rig, StreamingTranscripts("Hey Muse, what's the weather tomorrow?"), calls)

    async def user():
        await speak_live(rig, wake=True)
        await rig.wait_until_or_timeout(lambda: calls.hung_up.done(), 8)
        await room(rig, 1.0)

    events = run(rig, conversation, user)
    assert set(played(events)) == {"muse"}


@pytest.mark.parametrize("transcriber", [StreamingTranscripts("And tomorrow?"), FailingStreamingTranscriber()],
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


@pytest.mark.parametrize("voiced_s", [.1, .22])
def test_a_short_sound_over_muse_s_reply_does_not_stop_it(monkeypatch, voiced_s):
    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    rig.hardware.chunk_s = .05
    calls = ScriptedCalls(reply_s=3.0)
    transcripts = StreamingTranscripts("Hey Muse, tell me a story.", "Mm-hm.")
    conversation = realtime_service(rig, transcripts, calls)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        await asyncio.sleep(.3)
        rig.say(voiced_s=voiced_s)
        await rig.wait_until_or_timeout(lambda: calls.barge_ins > 0, 2)

    events = run(rig, conversation, user)
    assert calls.barge_ins == 0
    assert events.count({"robot": "clear_audio"}) == 1


@pytest.mark.parametrize("voiced_s, partial", [(.1, "Mm-hm."), (.5, "Great news")])
def test_playback_noise_never_triggers_the_sdk_s_speech_started_gate(monkeypatch, voiced_s, partial):
    from musegadget import voice_call as sdk

    rig = rig_for(monkeypatch)
    rig.hardware.echo_cancelled_input = True
    flushed = []

    class Call(sdk.VoiceCall):
        async def _start(self):
            self.events = asyncio.Queue()
            self._stream = self.records()
            self.loud = self.paused = 0
            self.answered = False
            self.ready.set()

        async def records(self):
            while True:
                yield await self.events.get()

        def publish(self, event):
            self.events.put_nowait(json.dumps(event).encode() + b"\n")

        async def _event(self, event):
            if event["type"] != "input_audio_buffer.append":
                return
            pcm = base64.b64decode(event["audio"])
            loud = sdk.pcm_rms(pcm) >= 300
            if self.answered:
                calls.loud_after_reply += loud
                if loud:
                    self.publish({"type": "input_audio_buffer.speech_started", "item_id": "noise"})
            else:
                self.loud += loud
                self.paused += not loud and self.loud >= 10
                if self.paused == 2:
                    self.answered = True
                    self.publish({"type": "response.created", "response": {"id": "r1"}})
                    self.publish({"type": "response.output_audio_transcript.delta", "delta": REPLY})
                    self.publish({"type": "response.audio.delta", "delta": base64.b64encode(voiced(3)).decode()})

        async def _cleanup(self):
            self._flush()

    class SDKCalls(ScriptedCalls):
        def __call__(self, session, **options):
            original = options["on_event"]

            def observe(event):
                if isinstance(event, sdk.PlaybackFlushed):
                    flushed.append(event)
                original(event)

            return Call(session, **{**options, "on_event": observe})

    calls = SDKCalls()
    conversation = realtime_service(rig, StreamingTranscripts("Hey Muse, tell me a story.", partial), calls)
    heard = []

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        rig.say(voiced_s=voiced_s)
        await room(rig, .7)
        heard.append((calls.loud_after_reply, len(flushed), conversation._voice[1].call.audible))

    events = run(rig, conversation, user)
    assert heard == [(0, 0, True)]
    assert sent_to_muse(events) == []


def expressions(events):
    return [event["params"]["name"] for event in events
            if event.get("robot") == "command" and event["name"] == "reachy.expression"]


async def looks_it_up_then_answers(sdk, emit, script):
    emit(sdk.Working(True))
    reply_events(sdk, emit, "r1", "Great question, let me look that up.", voiced(1.2))
    await asyncio.sleep(2.5)
    reply_events(sdk, emit, "r2", REPLY, voiced(1.2) + b"\0\0" * int(.3 * RATE) + voiced(1.0))
    emit(sdk.Working(False))


def test_muse_s_aside_while_it_works_gets_no_closing_and_its_answer_gets_one(monkeypatch):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(answer=looks_it_up_then_answers)
    conversation = realtime_service(rig, StreamingTranscripts("Hey Muse, what's the weather tomorrow?"), calls)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "success1" in expressions(rig.recording.events), 10)

    events = run(rig, conversation, user)
    assert expressions(events) == ["success1"]
    assert events.index({"robot": "command", "name": "reachy.expression",
                         "params": {"name": "success1", "intensity": 0.6}}) > max(
        index for index, event in enumerate(events) if event.get("audio") == "muse")


def test_muse_moving_reachy_during_its_reply_replaces_reachy_s_own_closing(monkeypatch):
    rig = rig_for(monkeypatch)
    services = []

    async def answers_with_a_gesture(sdk, emit, script):
        await answers(REPLY, 1.0)(sdk, emit, script)
        await asyncio.sleep(1.5)
        await asyncio.to_thread(services[0].run_command, "reachy.expression", {"name": "happy"}, 5000)

    calls = ScriptedCalls(answer=answers_with_a_gesture)
    conversation = realtime_service(rig, StreamingTranscripts("Hey Muse, what's the weather tomorrow?"), calls)
    services.append(conversation)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "happy" in expressions(rig.recording.events), 10)
        await asyncio.sleep(2.0)

    events = run(rig, conversation, user)
    assert expressions(events) == ["happy"]


async def acknowledges_then_works(sdk, emit, script):
    emit(sdk.Working(True))
    reply_events(sdk, emit, "r1", "Sure, one moment.", voiced(1.0))
    await asyncio.sleep(3.0)
    emit(sdk.Working(False))


def test_after_muse_s_spoken_acknowledgement_reachy_looks_thoughtful_until_muse_s_work_settles(monkeypatch):
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(answer=acknowledges_then_works)
    conversation = realtime_service(rig, StreamingTranscripts("Hey Muse, book me a table for two."), calls)
    seen = []

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: "muse" in played(rig.recording.events), 5)
        await asyncio.sleep(2.0)
        seen.append(rig.states()[-1])
        await asyncio.sleep(1.5)
        seen.append(rig.states()[-1])

    run(rig, conversation, user)
    assert seen == ["thinking", "listening"]
