"""Offline turn experiments using the suite's in-memory session and audio fakes.

    PYTHONPATH=src:tests python -m bench.latency_turn_replay dispatch --runs 7

No real Muse session, companion connection, or audio device is constructed.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import time
from types import SimpleNamespace

import numpy as np

from bench.latency_replay import summary
from reachy_muse.capabilities import Hold, Mode
from reachy_muse.contract.intent import Intent
from reachy_muse.contract.replies import ReplyStyle
from reachy_muse.conversation import VoiceConversation, _SpeechJob
from reachy_muse.failover import CompanionRoute, FailoverNarrator
from reachy_muse.local_backends import RuleNarrator, _LocalEndedTurn, backends_for
from reachy_muse.turn_timeline import TurnTimeline
from test_conversation import FakeHardware, FakeSession, chat_event


REQUEST = "What's the weather tomorrow?"


class ToneSpeech:
    async def stream(self, text, rate):
        await asyncio.sleep(.01)
        yield np.full(rate // 20, .1, dtype=np.float32)


class AudibleTimeline(TurnTimeline):
    def __init__(self):
        super().__init__()
        self.audible = asyncio.Event()

    def mark(self, stage, at=None):
        super().mark(stage, at)
        if stage == "first_audio":
            self.audible.set()


class DelayedOpener(RuleNarrator):
    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    async def acknowledge(self, heard):
        await asyncio.sleep(self.delay)
        return Hold(Intent.QUESTION, None)


def owned_session(**kwargs):
    return FakeSession(acknowledgement={
        "ok": True, "status": 200, "response": {"result": {
            "message_id": "user-1", "session_id": "robot-chat", "is_thread": True,
        }}}, **kwargs)


async def dispatch_once(opener_delay):
    answers = []
    sent = []

    async def send(session):
        sent.append(time.monotonic())

        async def answer():
            await asyncio.sleep(.02)
            await session.events.put(chat_event("delta.message_done", "reply-1",
                                                content="Sunny tomorrow. [reachy:happy]"))
        answers.append(asyncio.create_task(answer()))

    session = owned_session(send_hook=send)
    backends = replace(backends_for(Mode.ON_ROBOT, session, speech=ToneSpeech()),
                       narrator=DelayedOpener(opener_delay))
    conversation = VoiceConversation(session, FakeHardware(), backends=backends,
                                     owns_chat=True, session_id="robot-chat")
    timeline = AudibleTimeline()
    timeline.mark("speech_end")
    timeline.mark("endpoint")
    timeline.mark("transcript")
    subscriber = asyncio.create_task(conversation._subscribe())
    await session.chat_subscribed.wait()
    turning = asyncio.create_task(conversation.turn(b"\0" * 32044, recognized_text=REQUEST,
                                                    timeline=timeline))
    audible = asyncio.create_task(timeline.audible.wait())
    try:
        done, _ = await asyncio.wait({turning, audible}, timeout=2, return_when=asyncio.FIRST_COMPLETED)
        if turning in done:
            turning.result()
        assert audible in done, "the replay never produced answer audio"
        assert len(sent) == 1
        assert session.setup_messages == [(REQUEST, "robot-chat")]
        return (sent[0] - timeline.marks["transcript"],
                timeline.marks["first_audio"] - timeline.marks["speech_end"])
    finally:
        for task in (turning, subscriber, audible, *answers):
            task.cancel()
        await asyncio.gather(turning, subscriber, audible, *answers, return_exceptions=True)
        await conversation._moments.close()


async def dispatch_replay(runs):
    result = {}
    for delay in (.2, 0):
        measured = [await dispatch_once(delay) for _ in range(runs)]
        result[f"opener_delay_{delay * 1000:.0f}_ms"] = {
            "transcript_to_send": summary([row[0] for row in measured]),
            "speech_end_to_first_answer": summary([row[1] for row in measured]),
            "requests_per_turn": 1, "request_bytes": len(REQUEST.encode()),
        }
    return result


async def prompt_replay():
    session = owned_session()
    conversation = VoiceConversation(session, FakeHardware(),
                                     backends=backends_for(Mode.ON_ROBOT, session, speech=ToneSpeech()),
                                     owns_chat=True, session_id="robot-chat")
    result = []
    for style in (ReplyStyle.MARKER, ReplyStyle.MARKER, ReplyStyle.PLAIN_SHORT,
                  ReplyStyle.PLAIN_SHORT, ReplyStyle.MARKER):
        job = await conversation._dispatch(b"\0" * 32044, REQUEST, started=time.monotonic(),
                                           defer_playback=True, timeline=TurnTimeline(), style=style)
        message, chat = session.setup_messages[-1]
        assert chat == "robot-chat" and message.endswith(REQUEST)
        result.append({"style": style.value, "payload_bytes": len(message.encode()),
                       "context_included": message != REQUEST})
        job.turn.retire()
        conversation.job = None
    assert len(session.setup_messages) == 5
    assert [row["context_included"] for row in result] == [False, False, True, False, True]
    return {"requests": len(session.setup_messages), "turns": result}


async def endpoint_replay():
    from test_end_of_turn import RATE, silence, speech, turns

    reminder = speech("reminder")
    cut = round(2.04 * RATE)
    clips = {"unmodified": np.concatenate([reminder, silence(3)]),
             "one_second_hesitation": np.concatenate([
                 reminder[:cut], silence(1), reminder[cut:], silence(3)])}
    result = {}
    for name, audio in clips.items():
        result[name] = {str(threshold): turns(audio, threshold, 1.2) for threshold in (2.0, .8)}
    assert len(result["one_second_hesitation"]["2.0"]) == 1
    assert len(result["one_second_hesitation"]["0.8"]) == 2
    return {"units": "each pair is endpoint audio time and retained WAV duration, in seconds",
            "clips": result}


async def queue_once(interrupt):
    class QueueSpeech:
        async def stream(self, text, rate):
            yield np.full(round(rate * (.6 if text == "Briefing." else .04)),
                          .1 if text == "Briefing." else .2, dtype=np.float32)

    hardware = FakeHardware()
    conversation = VoiceConversation(None, hardware,
        backends=backends_for(Mode.ON_ROBOT, None, speech=QueueSpeech()), owns_chat=True)
    previous, answer = asyncio.Event(), asyncio.Event()
    times = []

    def audible():
        times.append(time.monotonic())
        answer.set()

    conversation._output_queue.put_nowait(_SpeechJob("briefing", "Briefing.", on_start=previous.set))
    speaker = asyncio.create_task(conversation._play_output())
    try:
        await asyncio.wait_for(previous.wait(), 1)
        while not hardware.played:
            await asyncio.sleep(.001)
        started = time.monotonic()
        if interrupt:
            conversation._hush()
        conversation._output_queue.put_nowait(_SpeechJob("answer", "Answer.", on_start=audible))
        await asyncio.wait_for(answer.wait(), 2)
        previous_frames = sum(len(chunk) for chunk in hardware.played if np.isclose(chunk[0], .1))
        assert previous_frames < 4800 if interrupt else previous_frames == 4800
        return times[0] - started, previous_frames / hardware.output_sample_rate
    finally:
        speaker.cancel()
        await asyncio.gather(speaker, return_exceptions=True)


async def queue_replay(runs):
    result = {}
    for interrupt in (False, True):
        measurements = [await queue_once(interrupt) for _ in range(runs)]
        result["interrupt" if interrupt else "ordered"] = {
            "queued_answer_to_first_audio": summary([item[0] for item in measurements]),
            "prior_speech_seconds_delivered": [item[1] for item in measurements],
        }
    return result


async def asr_replay(runs):
    result = {}
    for extra_start in (False, True):
        residual, total = [], []
        for _ in range(runs):
            calls = []

            async def transcribe(wav):
                calls.append(wav)
                await asyncio.sleep(.02)
                return "Remind me at eight, not at nine."

            turn = _LocalEndedTurn(b"recording", None, transcribe, False)
            started = time.monotonic()
            turn.start()
            if extra_start:
                turn.start()
            await asyncio.sleep(.06)
            waiting = time.monotonic()
            endpoint = await turn.endpoint()
            ended = time.monotonic()
            assert endpoint.text == "Remind me at eight, not at nine."
            assert calls == [b"recording"]
            residual.append(ended - waiting)
            total.append(ended - started)
        result["extra_early_start" if extra_start else "current_endpoint_start"] = {
            "remaining_asr_wait": summary(residual), "total_queue_and_asr": summary(total),
            "recognition_calls_per_turn": 1,
        }
    return result


async def narration_replay(runs):
    class CompanionNarrator:
        available = True

        def __init__(self, delay):
            self.delay = delay
            self.calls = 0

        async def lines(self, request, reply, style):
            self.calls += 1
            await asyncio.sleep(self.delay)
            return await RuleNarrator().lines(request, reply, style)

    result = {}
    for delay in (.2, 0):
        companion = CompanionNarrator(delay)
        route = CompanionRoute(SimpleNamespace(up=True))
        route.begin_turn()
        narrator = FailoverNarrator(companion, RuleNarrator(), route)
        elapsed = []
        for _ in range(runs):
            started = time.monotonic()
            lines = await narrator.lines(REQUEST, "Sunny tomorrow. Bring a hat.", ReplyStyle.PLAIN_SHORT)
            elapsed.append(time.monotonic() - started)
            assert [line.text for line in lines] == ["Sunny tomorrow.", "Bring a hat."]
        assert companion.calls == 0
        result[f"companion_delay_{delay * 1000:.0f}_ms"] = {
            "narration": summary(elapsed), "companion_requests": companion.calls,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("dispatch", "prompt", "endpoint", "queue", "asr", "narration"))
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()
    experiment = {"dispatch": lambda: dispatch_replay(args.runs), "prompt": prompt_replay,
                  "endpoint": endpoint_replay, "queue": lambda: queue_replay(args.runs),
                  "asr": lambda: asr_replay(args.runs),
                  "narration": lambda: narration_replay(args.runs)}[args.experiment]()
    print(json.dumps(asyncio.run(experiment), indent=2))


if __name__ == "__main__":
    main()
