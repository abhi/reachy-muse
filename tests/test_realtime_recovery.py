"""Unanswered calls exercised through microphone capture, admission and playback."""

import asyncio
import re
import time

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
            await wait_for(rig, lambda: any(turn.cued for turn in call.pending))
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
    emitters = []

    async def answer(sdk, emit, script):
        emitters.append(lambda: reply_events(sdk, emit, "r1", REPLY, voiced(.3)))

    calls = ScriptedCalls(answer=answer)
    conversation = realtime_service(rig, StreamingTranscripts("First question.", "Second question."), calls,
                                    wake_timeout_s=10)
    marks = []
    expected = []
    latency_lines = []

    async def user():
        await speak_live(rig, wake=True)
        voice = conversation._voice[1]
        await speak_live(rig)
        assert [turn.ordinal for turn in voice.call.pending] == [1, 2]
        marks.extend((turn.speech_end_at, turn.endpoint_at) for turn in voice.call.pending)
        before = voice.call.feed.sent
        emitters[0]()
        await wait_for(rig, lambda: len(turn_outcomes(caplog)) == 2)
        replied = voice.call.speaker.reply_started
        for index, (speech, endpoint) in enumerate(marks, 1):
            expected.append((round((replied - speech) * 1000), round((replied - endpoint) * 1000)))
            latency_lines.append(f"Muse's realtime reply started {replied - speech:.2f} s "
                                 f"after turn {index}'s last voiced sample")
        assert voice.call.feed.sent >= before
        assert voice.call.pending == []

    events = run(rig, conversation, user, rest=("listening", "idle"))
    rows = turn_outcomes(caplog)
    assert [(row["ordinal"], row["outcome"], row["words"], row["held"]) for row in rows] == [
        ("1", "answered", "2", "no"), ("2", "answered", "2", "no")]
    assert [(int(row["voice_to_reply_ms"]), int(row["endpoint_to_reply_ms"])) for row in rows] == expected
    assert all(195 <= voice_ms - endpoint_ms <= 205 for voice_ms, endpoint_ms in expected)
    assert all(int(row["feed_sent_delta"]) >= 0 and int(row["last_event_age_ms"]) < 100 for row in rows)
    assert [line for line in caplog.messages if line.startswith("Muse's realtime reply started")] == latency_lines
    assert [line for line in caplog.messages if line.startswith("Realtime event=")] == [
        "Realtime event=ResponseStarted ordinal=1",
        "Realtime event=ResponseDone ordinal=1 status=completed audio_bytes=14400"]
    ready = [line for line in caplog.messages if line.startswith("Realtime call_ready ")]
    assert len(ready) == 1
    assert re.fullmatch(r"Realtime call_ready elapsed_ms=\d+ link=new queued_feed_frames=\d+", ready[0])
    summaries = [line for line in caplog.messages if line.startswith("Realtime call_summary ")]
    assert len(summaries) == 1
    assert re.fullmatch(r"Realtime call_summary duration_ms=\d+ turns=2 answered=2 unanswered=0 recovered=0 "
                        r"responses=1 audio_bytes=14400 feed_sent=\d+ feed_evictions=0 max_queue_age_ms=\d+", summaries[0])
    assert stall_cues(caplog) == []
    assert set(played(events)) == {"muse"}


@pytest.mark.parametrize("done", [False, True])
def test_silent_response_stops_suspending_after_eight_seconds(monkeypatch, caplog, done):
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
        await wait_for(rig, lambda: bool(stall_cues(caplog)), 13)
        call = conversation._voice[1].call
        assert 12 <= time.monotonic() - call.opened_at < 14
        assert stall_cues(caplog) == ["Realtime turn ordinal=1 cue=still_waiting"]
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


def test_first_turn_wait_clock_starts_when_the_call_is_ready(monkeypatch, caplog):
    caplog.set_level("INFO")
    rig = rig_for(monkeypatch)

    class DelayedCalls(ScriptedCalls):
        def __call__(self, session, **options):
            call = super().__call__(session, **options)
            run_call = call.run

            async def run(source, output, *, realtime_input=False):
                await asyncio.sleep(5)
                return await run_call(source, output, realtime_input=realtime_input)

            call.run = run
            return call

    calls = DelayedCalls(reply=None)
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
