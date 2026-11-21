"""Unanswered calls exercised through microphone capture, admission and playback."""

import asyncio
import contextlib
import time
from types import SimpleNamespace

import pytest

from test_device_command_turns import played, run, sent_to_muse
from test_realtime_turns import (CHUNK, VOICE_SAMPLE, REPLY, ScriptedCalls, StreamingTranscripts,
                                 answers, keeps_working, np, pytestmark, realtime_service,
                                 reply_events, rig_for, room, speak_live, voiced)


LOST = "tts:I lost my line to Muse. Please say that again."


def turn_outcomes(caplog):
    return [dict(field.split("=", 1) for field in line.split()[2:])
            for line in caplog.messages if line.startswith("Realtime turn ") and "outcome=" in line]


def stall_cues(caplog):
    return [line for line in caplog.messages if "cue=still_waiting" in line]


def recoveries(caplog):
    return [line for line in caplog.messages if line.startswith("Realtime recovery ")]


async def wait_for(rig, predicate, seconds=6):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        await room(rig, .04)
    assert predicate()


def test_silent_call_acknowledges_retries_then_recovers_without_replaying(monkeypatch, caplog):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(reply_s=.1)
    transcripts = StreamingTranscripts("Tell me the weather.", "What about tomorrow?", "Are you there?")
    conversation = realtime_service(rig, transcripts, calls, wake_timeout_s=10)

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        await wait_for(rig, lambda: not call.audible)
        await room(rig, .5)
        for ordinal in (2, 3):
            before = len([state for state in rig.states() if state == "thinking"])
            await speak_live(rig)
            assert len([state for state in rig.states() if state == "thinking"]) == before + 1
            assert [turn.ordinal for turn in call.pending] == list(range(2, ordinal + 1))
            if ordinal == 2:
                await wait_for(rig, lambda: bool(stall_cues(caplog)))
                assert stall_cues(caplog) == ["Realtime turn ordinal=2 cue=still_waiting"]
                assert voice.call is call
                assert LOST not in played(rig.recording.events)
        await wait_for(rig, lambda: len(calls.opened) == 2)
        await wait_for(rig, lambda: LOST in played(rig.recording.events))
        assert call.done
        assert calls.closed_calls == 1
        assert calls.closed_links == 1
        assert calls.links == 2
        assert voice.call is not call
        assert voice._wake_deadline > time.monotonic()
        await room(rig, .5)
        assert all(not np.any(np.frombuffer(frame, "<i2")) for frame in calls.call_frames[1])
        assert transcripts.texts == []

    events = run(rig, conversation, user, rest=("listening", "idle"), timeout=30)
    assert played(events).count(LOST) == 1
    assert sent_to_muse(events) == []
    assert stall_cues(caplog) == ["Realtime turn ordinal=2 cue=still_waiting", "Realtime turn ordinal=3 cue=still_waiting"]
    assert recoveries(caplog) == ["Realtime recovery attempt=1 reason=retry"]
    assert [(row["ordinal"], row["outcome"]) for row in turn_outcomes(caplog)] == [
        ("1", "answered"), ("2", "recovered"), ("3", "recovered")]
    assert [(row["voice_to_reply_ms"], row["endpoint_to_reply_ms"]) for row in turn_outcomes(caplog)[1:]] == [
        ("none", "none"), ("none", "none")]
    assert not any(text in line for text in (REPLY, "What about tomorrow?", "Are you there?") for line in caplog.messages)


class SlowHangUpCalls(ScriptedCalls):
    """The first call hangs up like the SDK with a wedged server: its shielded end-call request and stream
    drain take ``hang_up_s``. ``hang_ups`` records how many links were closed when each hang-up finished."""

    def __init__(self, *, hang_up_s, **options):
        super().__init__(**options)
        self.hang_up_s = hang_up_s
        self.opened_at = []
        self.hang_ups = []

    def __call__(self, session, **options):
        call = super().__call__(session, **options)
        self.opened_at.append(time.monotonic())
        slow = len(self.opened) == 1
        run_call = call.run

        async def run(source, output, *, realtime_input=False):
            try:
                return await run_call(source, output, realtime_input=realtime_input)
            finally:
                if slow:
                    cleanup = asyncio.ensure_future(asyncio.sleep(self.hang_up_s))
                    while not cleanup.done():
                        with contextlib.suppress(asyncio.CancelledError):
                            await asyncio.shield(cleanup)
                    self.hang_ups.append(self.closed_links)

        call.run = run
        return call


def test_a_slow_hang_up_neither_delays_the_fresh_call_nor_loses_the_repeat(monkeypatch, caplog):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    calls = SlowHangUpCalls(hang_up_s=12, reply_s=.1)
    transcripts = StreamingTranscripts("Tell me the weather.", "What about tomorrow?", "Are you there?",
                                       "What about tomorrow?")
    conversation = realtime_service(rig, transcripts, calls, wake_timeout_s=10)
    during_hang_up = []

    async def user():
        await speak_live(rig, wake=True)
        call = conversation._voice[1].call
        await wait_for(rig, lambda: not call.audible)
        await room(rig, .5)
        await speak_live(rig)
        await wait_for(rig, lambda: bool(stall_cues(caplog)))
        await speak_live(rig)
        await wait_for(rig, lambda: bool(recoveries(caplog)), 8)
        escalated = time.monotonic()
        await wait_for(rig, lambda: LOST in played(rig.recording.events), 2)
        await room(rig, .6)
        await speak_live(rig)
        await wait_for(rig, lambda: len(turn_outcomes(caplog)) == 4, 5)
        during_hang_up.append((calls.opened_at[1] - escalated < .1, list(calls.hang_ups), calls.closed_links))
        await wait_for(rig, lambda: calls.closed_links == 1, 12)

    events = run(rig, conversation, user, rest=("listening", "idle"), timeout=50)
    assert during_hang_up == [(True, [], 0)]
    assert calls.hang_ups == [0]
    assert calls.links == 2
    assert played(events).count(LOST) == 1
    assert set(played(events)[played(events).index(LOST) + 1:]) == {"muse"}
    assert recoveries(caplog) == ["Realtime recovery attempt=1 reason=retry"]
    assert [(row["ordinal"], row["outcome"]) for row in turn_outcomes(caplog)] == [
        ("1", "answered"), ("2", "recovered"), ("3", "recovered"), ("1", "answered")]
    assert any(np.any(np.frombuffer(frame, "<i2")) for frame in calls.call_frames[1])


def test_answered_turns_have_no_waiting_or_spoken_recovery_cue(monkeypatch, caplog):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    emitters = []

    async def answer(sdk, emit, script):
        emitters.append(lambda: reply_events(sdk, emit, "r2", REPLY, voiced(.3)))
        await answers(REPLY, .1)(sdk, emit, script)

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, StreamingTranscripts("How is it today?", "And tomorrow?"), calls,
                                    wake_timeout_s=10)

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        await wait_for(rig, lambda: not voice.call.audible)
        await room(rig, .5)
        await speak_live(rig)
        emitters[0]()
        await wait_for(rig, lambda: len(turn_outcomes(caplog)) == 2)

    events = run(rig, conversation, user, rest=("listening", "idle"))
    assert stall_cues(caplog) == []
    assert recoveries(caplog) == []
    assert set(played(events)) == {"muse"}
    assert [row["outcome"] for row in turn_outcomes(caplog)] == ["answered", "answered"]
    assert len(calls.opened) == 1


@pytest.mark.parametrize("close_first", [False, True])
@pytest.mark.parametrize("earlier_pending, outcomes", [
    (False, [("1", "unanswered")]),
    (True, [("1", "unanswered"), ("2", "unanswered")]),
])
def test_call_failure_during_command_classification_announces_loss_once(
        monkeypatch, caplog, close_first, earlier_pending, outcomes):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(reply=None)
    classified = []
    announced = []

    class Model:
        available = True

        async def classify(self, text, allowed):
            classified.append(text)
            call = conversation._voice[1].call
            calls.hang_up(ConnectionError("media reset"))
            while not call.done:
                await asyncio.sleep(0)
            if close_first:
                await rig.wait_until_or_timeout(lambda: calls.closed_links == 1, 1)
                if earlier_pending:
                    await rig.wait_until_or_timeout(lambda: LOST in played(rig.recording.events), 1)
            announced.append(LOST in played(rig.recording.events))
            return None

    texts = ["First question."] if earlier_pending else []
    conversation = realtime_service(rig, StreamingTranscripts(*texts, "Tell me about your voice."), calls,
                                    model=Model(), wake_timeout_s=10)

    async def user():
        await speak_live(rig, wake=True)
        if earlier_pending:
            await speak_live(rig)
        await wait_for(rig, lambda: conversation._voice[1].call is None)
        await room(rig, .5)

    events = run(rig, conversation, user, rest=("listening", "idle"))
    assert classified == ["Tell me about your voice."]
    assert announced == [close_first and earlier_pending]
    assert played(events) == [LOST]
    assert sent_to_muse(events) == []
    assert calls.closed_calls == 1
    assert calls.closed_links == 1
    assert len(calls.opened) == 1
    assert [(row["ordinal"], row["outcome"]) for row in turn_outcomes(caplog)] == outcomes


def test_twenty_seconds_of_work_with_events_is_not_a_stall(monkeypatch, caplog):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)

    async def answer(sdk, emit, script):
        for _ in range(20):
            await keeps_working(sdk, emit, script)
            await asyncio.sleep(1)
        reply_events(sdk, emit, "r1", REPLY, voiced(.3))
        emit(sdk.Working(False))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, StreamingTranscripts("Find a table for dinner."), calls)

    async def user():
        await speak_live(rig, wake=True)
        await wait_for(rig, lambda: "muse" in played(rig.recording.events), 23)

    events = run(rig, conversation, user, timeout=30)
    assert stall_cues(caplog) == []
    assert recoveries(caplog) == []
    assert set(played(events)) == {"muse"}
    assert [row["outcome"] for row in turn_outcomes(caplog)] == ["answered"]
    assert [line for line in caplog.messages if "event=Working" in line] == [
        "Realtime event=Working working=on", "Realtime event=Working working=off"]


def tool_calls_then_reply(*tool_calls_at, reply_at):
    """Muse works on the request: each tool call is a response that completes at once without audio."""
    async def answer(sdk, emit, script):
        started = time.monotonic()
        emit(sdk.Working(True))
        for index, at in enumerate(tool_calls_at):
            await asyncio.sleep(started + at - time.monotonic())
            emit(sdk.ResponseStarted(f"tool-call-{index}"))
            emit(sdk.ResponseDone(f"tool-call-{index}", "completed"))
        await asyncio.sleep(started + reply_at - time.monotonic())
        reply_events(sdk, emit, "r1", REPLY, voiced(.3))
        emit(sdk.Working(False))
    return answer


@pytest.mark.parametrize("tool_calls_at", [(0,), (0, 12)], ids=["one tool call", "two tool calls"])
def test_a_tool_calls_silent_response_does_not_end_slow_work(monkeypatch, caplog, tool_calls_at):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(answer=tool_calls_then_reply(*tool_calls_at, reply_at=22))
    conversation = realtime_service(rig, StreamingTranscripts("Find a table for dinner."), calls, wake_timeout_s=10)

    async def user():
        await speak_live(rig, wake=True)
        await wait_for(rig, lambda: "muse" in played(rig.recording.events) or bool(recoveries(caplog)), 25)

    events = run(rig, conversation, user, rest=("listening", "idle"), timeout=40)
    assert stall_cues(caplog) == []
    assert recoveries(caplog) == []
    assert set(played(events)) == {"muse"}
    assert [row["outcome"] for row in turn_outcomes(caplog)] == ["answered"]
    assert len(calls.opened) == 1


@pytest.mark.parametrize("wake_timeout_s, outcomes", [
    (10, ["answered"]),  # the question comes after the window's deadline, so it needs the wake word
    (30, ["answered", "answered"]),
])
def test_asking_again_during_a_long_tool_call_does_not_hang_up_on_muse(monkeypatch, caplog, wake_timeout_s, outcomes):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(answer=tool_calls_then_reply(0, reply_at=22))
    conversation = realtime_service(rig, StreamingTranscripts("Book a table for two.", "Are you still there?"), calls,
                                    wake_timeout_s=wake_timeout_s)

    async def user():
        await speak_live(rig, wake=True)
        endpoint = conversation._voice[1].call.pending[0].endpoint_at
        await wait_for(rig, lambda: time.monotonic() >= endpoint + 13, 15)
        await speak_live(rig)
        await wait_for(rig, lambda: "muse" in played(rig.recording.events) or bool(recoveries(caplog)), 12)

    events = run(rig, conversation, user, rest=("listening", "idle"), timeout=45)
    assert recoveries(caplog) == []
    assert stall_cues(caplog) == []
    assert set(played(events)) == {"muse"}
    assert [row["outcome"] for row in turn_outcomes(caplog)] == outcomes
    assert len(calls.opened) == 1


@pytest.mark.parametrize("working, delay, reason", [(False, 15, "no_events"), (True, 30, "working_stale")])
def test_an_eventless_pending_turn_recovers_at_its_limit(monkeypatch, caplog, working, delay, reason):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(answer=keeps_working) if working else ScriptedCalls(reply=None)
    conversation = realtime_service(rig, StreamingTranscripts("Please find a table."), calls, wake_timeout_s=10)

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        call = voice.call
        endpoint = call.pending[0].endpoint_at
        await wait_for(rig, lambda: time.monotonic() >= endpoint + delay - .2, delay)
        assert voice.call is call
        assert recoveries(caplog) == []
        await wait_for(rig, lambda: len(calls.opened) == 2, 2)
        elapsed = time.monotonic() - endpoint
        assert delay <= elapsed < delay + 1
        assert voice._wake_deadline > time.monotonic()
        await wait_for(rig, lambda: LOST in played(rig.recording.events))

    events = run(rig, conversation, user, rest=("listening", "idle"), timeout=40)
    assert recoveries(caplog) == [f"Realtime recovery attempt=1 reason={reason}"]
    assert played(events) == [LOST]
    assert sent_to_muse(events) == []
    assert [row["outcome"] for row in turn_outcomes(caplog)] == ["recovered"]
    assert stall_cues(caplog) == ([] if working else ["Realtime turn ordinal=1 cue=still_waiting"])


@pytest.mark.parametrize("working", [True, False], ids=["Working around it", "no Working"])
def test_a_request_muse_answers_by_acting_is_answered_without_a_lost_line(monkeypatch, caplog, working):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)

    async def answer(sdk, emit, script):
        if working:
            emit(sdk.Working(True))
        emit(sdk.ResponseStarted("nod"))
        emit(sdk.ResponseDone("nod", "completed"))
        if working:
            emit(sdk.Working(False))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, StreamingTranscripts("Nod your head."), calls, wake_timeout_s=10)

    async def user():
        await speak_live(rig, wake=True)
        await wait_for(rig, lambda: bool(turn_outcomes(caplog)) or bool(recoveries(caplog)), 17)

    events = run(rig, conversation, user, rest=("listening",), timeout=30)
    assert [(row["ordinal"], row["outcome"], row["voice_to_reply_ms"]) for row in turn_outcomes(caplog)] == [
        ("1", "silent", "none")]
    assert 1000 <= int(turn_outcomes(caplog)[0]["last_event_age_ms"]) < 1300
    assert recoveries(caplog) == []
    assert stall_cues(caplog) == []
    assert played(events) == []
    assert rig.states()[-2:] == ["thinking", "listening"]
    assert len(calls.opened) == 1


@pytest.mark.parametrize("working_first", [True, False], ids=["Working first", "Working just after"])
def test_a_hung_tool_call_stays_pending_until_its_work_goes_stale(monkeypatch, caplog, working_first):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)

    async def answer(sdk, emit, script):
        if working_first:
            emit(sdk.Working(True))
        emit(sdk.ResponseStarted("tool-call"))
        emit(sdk.ResponseDone("tool-call", "completed"))
        if not working_first:
            await asyncio.sleep(.3)
            emit(sdk.Working(True))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, StreamingTranscripts("Book a table for two."), calls, wake_timeout_s=10)

    async def user():
        await speak_live(rig, wake=True)
        call = conversation._voice[1].call
        await wait_for(rig, lambda: len(calls.opened) == 2, 33)
        assert 30 <= time.monotonic() - call.last_event_at < 31
        await wait_for(rig, lambda: LOST in played(rig.recording.events))

    events = run(rig, conversation, user, rest=("listening", "idle"), timeout=45)
    assert recoveries(caplog) == ["Realtime recovery attempt=1 reason=working_stale"]
    assert [row["outcome"] for row in turn_outcomes(caplog)] == ["recovered"]
    assert stall_cues(caplog) == []
    assert played(events) == [LOST]


def test_second_recovery_in_one_wake_sends_only_the_next_request_by_chat(monkeypatch, caplog):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(reply=None)

    async def muse(session, request):
        await session.answer(request, "Here is the answer.")

    conversation = realtime_service(rig, StreamingTranscripts("One request.", "First retry.", "Another request.",
                                                              "Second retry.", "My new request."), calls, muse,
                                    wake_timeout_s=10)

    async def user():
        await speak_live(rig, wake=True)
        for attempt in (1, 2):
            voice = conversation._voice[1]
            call = voice.call
            await wait_for(rig, lambda: any(turn.cued_at is not None for turn in call.pending))
            await speak_live(rig)
            await wait_for(rig, lambda: len(recoveries(caplog)) == attempt)
            await wait_for(rig, lambda: played(rig.recording.events).count(LOST) == attempt)
            await room(rig, .6)
            if attempt == 1:
                assert voice.call is not call
                await speak_live(rig)
        assert voice.call is None
        assert not voice.realtime.usable
        assert len(calls.opened) == 2
        assert voice._wake_deadline > time.monotonic()
        await speak_live(rig)
        await wait_for(rig, lambda: "tts:Here is the answer." in played(rig.recording.events))

    events = run(rig, conversation, user, rest=("listening", "idle"), timeout=40)
    assert played(events) == [LOST, LOST, "tts:Here is the answer."]
    assert sent_to_muse(events) == ["My new request."]
    assert recoveries(caplog) == ["Realtime recovery attempt=1 reason=retry", "Realtime recovery attempt=2 reason=retry"]
    assert [row["outcome"] for row in turn_outcomes(caplog)] == ["recovered"] * 4


def test_reply_diagnostics_use_each_turns_last_voiced_sample(monkeypatch, caplog):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    clock = SimpleNamespace(samples=0, reply_delay=0)
    clock.monotonic = lambda: 100 + clock.samples / 16000 + clock.reply_delay
    monkeypatch.setattr("reachy_muse.conversation.time", clock)
    monkeypatch.setattr("reachy_muse.realtime_call.time", clock)
    emitters = []

    async def answer(sdk, emit, script):
        emitters.append(lambda: reply_events(sdk, emit, "r1", REPLY, voiced(.3)))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, StreamingTranscripts("First question.", "Second question."), calls,
                                    wake_timeout_s=10)
    is_speech = conversation.speech_gate.is_speech

    def timed_speech(pcm, rate):
        assert rate == 16000
        clock.samples += len(pcm) // 2
        return is_speech(pcm, rate)

    monkeypatch.setattr(conversation.speech_gate, "is_speech", timed_speech)

    async def user():
        await speak_live(rig, wake=True)
        await speak_live(rig)
        assert clock.samples == 32000
        assert len(calls.frames) == 49
        emitters[0]()
        clock.reply_delay = .125
        await rig.wait_until_or_timeout(lambda: len(turn_outcomes(caplog)) == 2)
        assert conversation._voice[1].call.pending == []

    events = run(rig, conversation, user, rest=("listening", "idle"))
    assert [line for line in caplog.messages if line.startswith("Realtime turn ")] == [
        "Realtime turn ordinal=1 outcome=answered words=2 voice_to_reply_ms=1625 endpoint_to_reply_ms=1425 "
        "held=no feed_sent_delta=32 last_event_age_ms=125",
        "Realtime turn ordinal=2 outcome=answered words=2 voice_to_reply_ms=625 endpoint_to_reply_ms=425 "
        "held=no feed_sent_delta=7 last_event_age_ms=125"]
    assert [line for line in caplog.messages if line.startswith("Muse's realtime reply started")] == [
        "Muse's realtime reply started 1.62 s after turn 1's last voiced sample",
        "Muse's realtime reply started 0.62 s after turn 2's last voiced sample"]
    assert [line for line in caplog.messages if line.startswith("Realtime event=")] == [
        "Realtime event=ResponseStarted ordinal=1",
        "Realtime event=ResponseDone ordinal=1 status=completed audio_bytes=14400"]
    assert [line for line in caplog.messages if line.startswith("Realtime call_ready ")] == [
        "Realtime call_ready elapsed_ms=0 link=new queued_feed_frames=0"]
    assert [line for line in caplog.messages if line.startswith("Realtime call_summary ")] == [
        "Realtime call_summary duration_ms=2125 turns=2 answered=2 silent=0 unanswered=0 recovered=0 "
        "responses=1 audio_bytes=14400 feed_sent=49 feed_evictions=0 max_queue_age_ms=0"]
    assert stall_cues(caplog) == []
    assert set(played(events)) == {"muse"}


@pytest.mark.parametrize("done, cues", [(False, ["Realtime turn ordinal=1 cue=still_waiting"]), (True, [])])
def test_only_an_open_response_without_audio_stops_suspending_after_eight_seconds(monkeypatch, caplog, done, cues):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)

    async def answer(sdk, emit, script):
        emit(sdk.Working(True))
        emit(sdk.ResponseStarted("private-response-id"))
        if done:
            emit(sdk.ResponseDone("private-response-id", "completed"))
        for _ in range(13):
            emit(sdk.TranscriptDelta("private-response-id", "private reply words"))
            await asyncio.sleep(1)

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, StreamingTranscripts("Please answer this."), calls, wake_timeout_s=10)

    async def user():
        await speak_live(rig, wake=True)
        call = conversation._voice[1].call
        await wait_for(rig, lambda: bool(stall_cues(caplog)) or time.monotonic() - call.opened_at >= 14, 15)
        if cues:
            assert 12 <= time.monotonic() - call.opened_at < 14
        assert stall_cues(caplog) == cues
        assert recoveries(caplog) == []
        assert played(rig.recording.events) == []

    run(rig, conversation, user, rest=("thinking",), timeout=20)
    assert [row["outcome"] for row in turn_outcomes(caplog)] == ["unanswered"]
    assert not any("private-response-id" in line or "private reply words" in line for line in caplog.messages)


def test_pending_turn_reaches_a_decision_within_ten_seconds_of_window_expiry(monkeypatch, caplog):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(reply=None)
    conversation = realtime_service(rig, StreamingTranscripts("Please answer this."), calls)

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        deadline = voice._wake_deadline
        call = voice.call
        await wait_for(rig, lambda: time.monotonic() >= deadline + 9.8, 12)
        assert voice.call is call
        assert voice._wake_deadline == deadline
        await wait_for(rig, lambda: len(calls.opened) == 2, 1)
        assert deadline + 10 <= time.monotonic() < deadline + 10.3
        assert voice._wake_deadline > time.monotonic()

    events = run(rig, conversation, user, rest=("listening", "idle"), timeout=20)
    assert recoveries(caplog) == ["Realtime recovery attempt=1 reason=window_expired"]
    assert played(events) == [LOST]


class SlowlyReadyCalls(ScriptedCalls):
    """Each call becomes ready ``ready_after_s`` after it starts, as over a slow connection."""

    def __init__(self, *, ready_after_s, **options):
        super().__init__(**options)
        self.ready_after_s = ready_after_s

    def __call__(self, session, **options):
        call = super().__call__(session, **options)
        run_call = call.run

        async def run(source, output, *, realtime_input=False):
            await asyncio.sleep(self.ready_after_s)
            return await run_call(source, output, realtime_input=realtime_input)

        call.run = run
        return call


def test_first_turn_wait_clock_starts_when_the_call_is_ready(monkeypatch, caplog):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    calls = SlowlyReadyCalls(ready_after_s=5, reply=None)
    conversation = realtime_service(rig, StreamingTranscripts("Please answer this."), calls, wake_timeout_s=10)

    async def user():
        await speak_live(rig, wake=True)
        call = conversation._voice[1].call
        await wait_for(rig, lambda: call.ready_at is not None)
        assert stall_cues(caplog) == []
        await wait_for(rig, lambda: time.monotonic() >= call.ready_at + 3.8, 5)
        assert stall_cues(caplog) == []
        await wait_for(rig, lambda: bool(stall_cues(caplog)), 1)
        assert 4 <= time.monotonic() - call.ready_at < 4.3

    events = run(rig, conversation, user, rest=("thinking",))
    assert stall_cues(caplog) == ["Realtime turn ordinal=1 cue=still_waiting"]
    assert played(events) == []
    assert recoveries(caplog) == []


def test_turns_spoken_while_the_call_connects_wait_from_its_readiness(monkeypatch, caplog):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    calls = SlowlyReadyCalls(ready_after_s=15, reply=None)
    conversation = realtime_service(rig, StreamingTranscripts("What's the weather?", "In Paris tomorrow."), calls)
    at_readiness = []

    async def user():
        await speak_live(rig, wake=True)
        await speak_live(rig)
        call = conversation._voice[1].call
        await wait_for(rig, lambda: call.ready_at is not None, 16)
        at_readiness.append((stall_cues(caplog), recoveries(caplog)))
        await wait_for(rig, lambda: bool(recoveries(caplog)), 5)
        assert 4 <= time.monotonic() - call.ready_at < 4.5
        await wait_for(rig, lambda: LOST in played(rig.recording.events))

    events = run(rig, conversation, user, rest=("listening", "idle"), timeout=40)
    assert at_readiness == [([], [])]
    assert stall_cues(caplog) == ["Realtime turn ordinal=1 cue=still_waiting", "Realtime turn ordinal=2 cue=still_waiting"]
    assert recoveries(caplog) == ["Realtime recovery attempt=1 reason=window_expired"]
    assert played(events) == [LOST]


def test_a_two_part_request_answered_by_one_slow_reply_is_not_a_retry(monkeypatch, caplog):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    emitters = []

    async def answer(sdk, emit, script):
        emitters.append(lambda: reply_events(sdk, emit, "r1", REPLY, voiced(.3)))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, StreamingTranscripts("What's the weather?", "In Paris tomorrow."), calls,
                                    wake_timeout_s=10)

    async def user():
        await speak_live(rig, wake=True)
        await speak_live(rig)
        call = conversation._voice[1].call
        await wait_for(rig, lambda: len(call.pending) == 2, 2)
        second = call.pending[1].endpoint_at
        await wait_for(rig, lambda: time.monotonic() >= second + 4.6, 6)
        assert recoveries(caplog) == []
        emitters[0]()
        await wait_for(rig, lambda: "muse" in played(rig.recording.events), 3)

    events = run(rig, conversation, user, rest=("listening", "idle"), timeout=30)
    assert recoveries(caplog) == []
    assert stall_cues(caplog) == ["Realtime turn ordinal=1 cue=still_waiting", "Realtime turn ordinal=2 cue=still_waiting"]
    assert set(played(events)) == {"muse"}
    assert [row["outcome"] for row in turn_outcomes(caplog)] == ["answered", "answered"]


def test_asking_again_before_the_waiting_nod_leaves_recovery_to_the_event_limit(monkeypatch, caplog):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(reply=None)
    conversation = realtime_service(rig, StreamingTranscripts("What's the weather?", "Hello?"), calls, wake_timeout_s=10)

    async def user():
        await speak_live(rig, wake=True)
        first = conversation._voice[1].call.pending[0].endpoint_at
        await speak_live(rig)
        await wait_for(rig, lambda: len(calls.opened) == 2, 17)
        assert 15 <= time.monotonic() - first < 16.5
        await wait_for(rig, lambda: LOST in played(rig.recording.events))

    events = run(rig, conversation, user, rest=("listening", "idle"), timeout=30)
    assert recoveries(caplog) == ["Realtime recovery attempt=1 reason=no_events"]
    assert stall_cues(caplog) == ["Realtime turn ordinal=1 cue=still_waiting", "Realtime turn ordinal=2 cue=still_waiting"]
    assert played(events) == [LOST]
    assert sent_to_muse(events) == []


def test_retry_clock_pauses_for_speech_and_recovery_waits_for_a_seven_tenths_pause(monkeypatch, caplog):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(reply=None)
    conversation = realtime_service(rig, StreamingTranscripts("First request.", "First retry.", "Please keep listening."),
                                    calls, wake_timeout_s=10)

    async def user():
        await speak_live(rig, wake=True)
        call = conversation._voice[1].call
        await wait_for(rig, lambda: bool(stall_cues(caplog)))
        await speak_live(rig)
        second = call.pending[1]
        await wait_for(rig, lambda: second.waiting_s >= 3.5)
        for _ in range(60):
            rig.hardware.samples.put(np.full(CHUNK, VOICE_SAMPLE, np.float32))
            await asyncio.sleep(CHUNK / 16000)
        assert second.waiting_s < 3.7
        assert recoveries(caplog) == []
        await room(rig, .5)
        assert conversation._voice[1].call is call
        assert recoveries(caplog) == []
        stopped = call.pending[-1].speech_end_at
        await wait_for(rig, lambda: len(calls.opened) == 2, 2)
        assert time.monotonic() - stopped >= .7

    events = run(rig, conversation, user, rest=("listening", "idle"), timeout=25)
    assert recoveries(caplog) == ["Realtime recovery attempt=1 reason=retry"]
    assert played(events) == [LOST]
    assert sent_to_muse(events) == []


def test_reply_arriving_during_the_waiting_gesture_cancels_recovery(monkeypatch, caplog):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)
    calls = ScriptedCalls(reply=None)
    conversation = realtime_service(rig, StreamingTranscripts("First question.", "Second question."), calls,
                                    wake_timeout_s=10)

    async def user():
        from musegadget import voice_call as sdk

        await speak_live(rig, wake=True)
        call = conversation._voice[1].call
        await wait_for(rig, lambda: bool(stall_cues(caplog)))
        loop = asyncio.get_running_loop()
        gesture = rig.hardware.gesture

        def answer_during_gesture(name):
            gesture(name)
            loop.call_soon_threadsafe(reply_events, sdk, call.on_event, "r1", REPLY, voiced(.3))
            time.sleep(.05)

        monkeypatch.setattr(rig.hardware, "gesture", answer_during_gesture)
        await speak_live(rig)
        await wait_for(rig, lambda: len(turn_outcomes(caplog)) == 2)
        assert conversation._voice[1].call is call
        assert not call.done

    events = run(rig, conversation, user, rest=("listening", "idle"), timeout=25)
    assert recoveries(caplog) == []
    assert set(played(events)) == {"muse"}
    assert [row["outcome"] for row in turn_outcomes(caplog)] == ["answered", "answered"]
