"""Offline latency experiments through Reachy's production conversation path.

    PYTHONPATH=src python -m bench.latency_replay --runs 7

Piper inference, the microphone, and the Muse transport are in-memory fakes.
Each synthesized phrase costs 10 ms. These are replay times, not robot forecasts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import time

import numpy as np

from bench.latency_journal import percentile
from reachy_muse.capabilities import Mode, SpokenLine
from reachy_muse.conversation import VoiceConversation
from reachy_muse.local_backends import backends_for
from reachy_muse.local_speech import PiperSpeech


class ReplaySpeech(PiperSpeech):
    """Keep Piper's ownership and cache code; replace only inference and its pipe."""

    def __init__(self, model, delay_s=.01):
        super().__init__(model)
        self.delay_s = delay_s
        self.requests = []

    async def stream(self, text, output_rate):
        async with self._lock:
            self.requests.append((text, output_rate))
            await asyncio.sleep(self.delay_s)
            code = hashlib.sha256(text.encode()).digest()[0] / 256
            yield np.full(output_rate // 100, code, dtype=np.float32)


class SilentHardware:
    sample_rate = 16000
    output_sample_rate = 48000
    echo_cancelled_input = True

    def read_audio(self):
        return None

    def clear_audio(self):
        pass

    def set_state(self, state, *, expression=None):
        pass


class OfflineSession:
    def __init__(self):
        self.registered = asyncio.Event()
        self.registered.set()
        self.chat_subscribed = asyncio.Event()
        self.events = asyncio.Queue()

    async def subscribe_chat(self, session_id):
        self.chat_subscribed.set()
        while True:
            yield await self.events.get()


class ReadyLog(logging.Handler):
    def __init__(self):
        super().__init__()
        self.ready = asyncio.Event()

    def emit(self, record):
        if record.getMessage().startswith("Reachy is ready."):
            self.ready.set()


async def startup(speech, progress_speech):
    session = OfflineSession()
    backends = backends_for(Mode.ON_ROBOT, session, speech=speech, progress_speech=progress_speech, wake=True)
    conversation = VoiceConversation(session, SilentHardware(), backends=backends, owns_chat=True)
    logger = logging.getLogger("reachy_muse.conversation")
    handler, old_level = ReadyLog(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    started = time.perf_counter()
    running = asyncio.create_task(conversation.run())
    ready = asyncio.create_task(handler.ready.wait())
    try:
        done, _ = await asyncio.wait({running, ready}, timeout=10, return_when=asyncio.FIRST_COMPLETED)
        if running in done:
            running.result()
        if ready not in done:
            raise TimeoutError("offline conversation never became ready")
        elapsed = time.perf_counter() - started
    finally:
        running.cancel()
        ready.cancel()
        await asyncio.gather(running, ready, return_exceptions=True)
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    digest = hashlib.sha256()
    for voice in (backends.voice, backends.progress_voice):
        for text in voice.fixed_phrases:
            line = SpokenLine(text, None, "notice")
            async for chunk in voice.stream(line, SilentHardware.output_sample_rate):
                digest.update(chunk.tobytes())
    return elapsed, digest.hexdigest()


def summary(values):
    return {"n": len(values), "p50_ms": round(percentile(values, .5) * 1000, 3),
            "p95_ms": round(percentile(values, .95) * 1000, 3)}


async def warm_replay(runs):
    cold, reconnect = [], []
    calls = []
    with tempfile.TemporaryDirectory(prefix="reachy-latency-") as directory:
        model = Path(directory) / "fake.onnx"
        model.write_bytes(b"offline fixture, never loaded")
        Path(str(model) + ".json").write_text(json.dumps({"audio": {"sample_rate": 22050}}))
        for _ in range(runs):
            speech, progress = ReplaySpeech(model), ReplaySpeech(model)
            before, digest = await startup(speech, progress)
            cold.append(before)
            first_count = len(speech.requests) + len(progress.requests)
            after, replay_digest = await startup(speech, progress)
            reconnect.append(after)
            assert replay_digest == digest, "reconnect changed the phrase audio"
            calls.append([first_count, len(speech.requests) + len(progress.requests) - first_count])
    return {"experiment": "startup with 10 ms per phrase and a ready local subscription",
            "cold": summary(cold), "reconnect": summary(reconnect), "synthesis_calls": calls,
            "pcm_sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    print(json.dumps(asyncio.run(warm_replay(args.runs)), indent=2))


if __name__ == "__main__":
    main()
