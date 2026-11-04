"""Offline sentence playback comparison; no devices, network, or models.

Run this same file with PYTHONPATH=<revision>/src:<revision>/tests to compare
revisions. --baseline accepts a previous JSON report and checks PCM equality.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from reachy_muse.capabilities import Mode
from reachy_muse.contract.replies import ReplyStyle
from reachy_muse.conversation import VoiceConversation
from reachy_muse.local_backends import backends_for
from test_conversation import (FakeHardware, FakeSession, acknowledge, chat_event,
                               feed, reply_job, sentence_frame)

TEXTS = ("The report is ready.", "Would you like the details?", "Wonderful!")


def quantiles(values):
    return {name: float(np.percentile(values, percentile)) for name, percentile in
            (("p50_ms", 50), ("p95_ms", 95), ("max_ms", 100))}


class MeasuredHardware(FakeHardware):
    def __init__(self):
        super().__init__()
        self.push_times = []
        self.plans = []
        self.finishes = 0

    def play_audio(self, samples, *, sentence=None):
        self.push_times.append(time.perf_counter())
        self.plans.append(sentence)
        super().play_audio(samples)

    def finish_sentence(self):
        self.finishes += 1


class SyntheticSpeech:
    async def stream(self, text, rate):
        await asyncio.sleep(.005)
        async for chunk in prepared_audio():
            yield chunk


async def prepared_audio():
    # Different signs and lengths detect dropped, duplicated, or reordered PCM.
    yield np.full(80, .125, dtype=np.float32)
    yield np.full(40, -.25, dtype=np.float32)


async def once(prepared):
    hardware = MeasuredHardware()
    session = FakeSession()
    conversation = VoiceConversation(session, hardware,
        backends=backends_for(Mode.ON_ROBOT, session, speech=SyntheticSpeech()))
    job = reply_job("robot-chat", style=ReplyStyle.EXPRESSIVE_JSON)
    acknowledge(job, {"message_id": "user-1"})
    started = time.perf_counter()
    starts = []
    completion = None
    queued = asyncio.Queue()
    frames = [sentence_frame(text) for text in TEXTS]

    async def producer():
        nonlocal completion
        for index, frame in enumerate(frames):
            if index:
                await asyncio.sleep(.09)
            for segment in feed(job, chat_event("delta.text_append", "reply-1",
                                                seq=index + 1, text=frame)):
                await queued.put(segment)
        completion = time.perf_counter()
        feed(job, chat_event("delta.message_done", "reply-1", seq=10,
                            content="".join(frames)))
        await queued.put(None)

    task = asyncio.create_task(producer())
    try:
        while (segment := await queued.get()) is not None:
            # Omit neutral wire hints to exercise local inference on plain text.
            await conversation._speak(segment.message_id, text=segment.text,
                stream_segment=True, prepared_stream=prepared_audio() if prepared else None,
                on_start=lambda: starts.append(time.perf_counter()))
        await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await conversation._moments.close()
    expected = np.tile(np.concatenate((np.full(80, .125, dtype=np.float32),
                                      np.full(40, -.25, dtype=np.float32))), len(TEXTS))
    actual = np.concatenate(hardware.played)
    assert np.array_equal(actual, expected), "PCM samples changed or were reordered"
    assert hardware.push_times[0] < completion, "audio waited for full reply completion"
    expressions = Counter(plan.expression.value for plan in hardware.plans if plan is not None)
    return {"first_push_ms": (hardware.push_times[0] - started) * 1000,
            "on_start_ms": (starts[0] - started) * 1000,
            "first_push_before_completion": True,
            "pcm_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            "pcm_frames": len(actual), "chunk_frames": [len(chunk) for chunk in hardware.played],
            "sentence_count": len(starts), "classified_count": sum(expressions.values()),
            "expressions": {"neutral": 0, **expressions}, "finish_notifications": hardware.finishes}


def classifier_measurements(runs):
    try:
        from reachy_muse.motion.sentence import classify_sentence
    except ModuleNotFoundError as exc:
        if exc.name != "reachy_muse.motion.sentence":
            raise
        return {"available": False}
    elapsed = []
    coverage = Counter()
    corpus = (*TEXTS, "I'm sorry that failed.", "Wow!", "Yes, absolutely.",
              "not wonderful", "word " * 1000)
    for _ in range(runs * 100):
        for text in corpus:
            started = time.process_time_ns()
            cue = classify_sentence(text)
            elapsed.append((time.process_time_ns() - started) / 1e6)
            coverage[cue.expression.value] += 1
    return {"available": True, "cpu": quantiles(elapsed), "calls": len(elapsed),
            "expressions": {"neutral": 0, **coverage}}


async def replay(runs):
    result = {"runs": runs, "classifier": classifier_measurements(runs), "cases": {}}
    for prepared in (False, True):
        rows = [await once(prepared) for _ in range(runs)]
        invariant_keys = ("pcm_sha256", "pcm_frames", "chunk_frames", "sentence_count",
                          "classified_count", "expressions", "finish_notifications",
                          "first_push_before_completion")
        invariants = {key: rows[0][key] for key in invariant_keys}
        assert all(all(row[key] == value for key, value in invariants.items()) for row in rows)
        result["cases"]["prepared" if prepared else "streamed"] = {
            **invariants, "first_actual_sink_push": quantiles([r["first_push_ms"] for r in rows]),
            "on_start": quantiles([r["on_start_ms"] for r in rows])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    result = asyncio.run(replay(args.runs))
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        result["comparison"] = {}
        for name, case in result["cases"].items():
            before = baseline["cases"][name]
            for key in ("pcm_sha256", "pcm_frames", "chunk_frames", "sentence_count"):
                assert case[key] == before[key], f"{name}: changed {key}"
            result["comparison"][name] = {
                key: case["first_actual_sink_push"][key] - before["first_actual_sink_push"][key]
                for key in ("p50_ms", "p95_ms", "max_ms")}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
