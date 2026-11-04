"""Offline voice-turn measurements. Only fakes and ephemeral loopback sockets.

Run with PYTHONPATH=src:tests python -m bench.turn2 engine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from unittest.mock import patch

from websockets.asyncio.client import connect

from reachy_muse.companion_server import hearing_voxtral
from reachy_muse.companion_server.server import CompanionServer, Engines
from reachy_muse.protocol import AudioKind, HearOpen, Hello, Op, Sender, decode, encode, encode_audio


async def engine_replay(stall_s=.3):
    intervals, pongs, work = [], [], {}

    class Utterance:
        text = "Should I work out today?"

        def __init__(self, models, config):
            work["open_start"] = time.monotonic()
            time.sleep(stall_s)
            work["open_end"] = time.monotonic()

        def feed(self, samples):
            work["feed_start"] = time.monotonic()
            time.sleep(stall_s)
            work["feed_end"] = time.monotonic()
            return self.text

    hearing = hearing_voxtral.VoxtralHearing()
    hearing._models = object()
    try:
        with patch.object(hearing_voxtral, "_Utterance", Utterance):
            async with CompanionServer(Engines(hearing=hearing), "offline-token") as server:
                async with connect(server.url, proxy=None) as ws:
                    await ws.send(encode(Hello((1,), "offline-token", (Op.HEAR,), 16000, "replay")))
                    await ws.recv()

                    async def ping():
                        previous = time.monotonic()
                        while True:
                            pong = await ws.ping()
                            await pong
                            now = time.monotonic()
                            intervals.append(now - previous)
                            pongs.append(now)
                            previous = now
                            await asyncio.sleep(.01)

                    probing = asyncio.create_task(ping())
                    try:
                        await asyncio.sleep(.02)
                        await ws.send(encode(HearOpen(1, 0)))
                        await ws.send(encode_audio(AudioKind.MIC, 1, 0, b"\x01\0" * 320))
                        partial = decode(await ws.recv(), sender=Sender.COMPANION)
                        endpoint = decode(await ws.recv(), sender=Sender.COMPANION)
                        await asyncio.sleep(.02)
                    finally:
                        probing.cancel()
                        await asyncio.gather(probing, return_exceptions=True)
    finally:
        hearing._thread.shutdown()
    return {
        "injected_call_ms": round(stall_s * 1000),
        "max_ping_gap_ms": round(max(intervals) * 1000, 2),
        "pongs_during_open": sum(work["open_start"] < p < work["open_end"] for p in pongs),
        "pongs_during_feed": sum(work["feed_start"] < p < work["feed_end"] for p in pongs),
        "partial": partial.text, "endpoint": endpoint.text,
    }


async def jitter_replay(scale=.05):
    from websockets.asyncio.server import serve
    from reachy_muse import companion_client as client
    from reachy_muse.protocol import Welcome

    rows = []
    # Same burst lengths and ping phase on both versions; scale all network timers.
    for jitter_s in (0, .5, 1, 1.5, 2, 3, 4, 6, 8):
        accepted = asyncio.Queue()

        async def companion(ws):
            await ws.recv()
            await ws.send(encode(Welcome(1, (Op.HEAR, Op.SPEAK, Op.NARRATE), ())))
            accepted.put_nowait(ws)
            await ws.wait_closed()

        async with serve(companion, "127.0.0.1", 0, ping_interval=None, close_timeout=.01) as server:
            port = server.sockets[0].getsockname()[1]
            link = client.CompanionLink(f"ws://127.0.0.1:{port}", token="offline-token", out_rate=16000,
                                        insecure=True, backoff_s=(10, 10),
                                        ping_interval_s=client.PING_INTERVAL_S * scale,
                                        ping_timeout_s=client.PING_TIMEOUT_S * scale)
            running = asyncio.create_task(link.run())
            ws = None
            try:
                await asyncio.wait_for(link.settled(), 1)
                ws = await accepted.get()
                await asyncio.sleep(client.PING_INTERVAL_S * scale * .9)
                ws.transport.pause_reading()
                await asyncio.sleep(jitter_s * scale)
                ws.transport.resume_reading()
                await asyncio.sleep(.14)
                rows.append({"jitter_s": jitter_s, "drops": link.transitions.count(client.LinkState.DOWN)})
            finally:
                if ws is not None:
                    ws.transport.resume_reading()
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
    return {"time_scale": scale, "bursts": rows, "drops": sum(row["drops"] for row in rows)}


async def reconnect_replay():
    from reachy_muse.companion_client import CompanionLink, LinkState
    from reachy_muse.companion_server.engines import EchoNarrator
    from reachy_muse.protocol import NarrateOp

    async with CompanionServer(Engines(narrator=EchoNarrator()), "offline-token") as server:
        link = CompanionLink(server.url[:-3], token="offline-token", out_rate=16000, insecure=True)
        running = asyncio.create_task(link.run())
        try:
            await asyncio.wait_for(link.settled(), 1)
            started = time.monotonic()
            next(iter(server._server.connections)).transport.abort()
            async with asyncio.timeout(15):
                while link.state is not LinkState.DOWN:
                    await asyncio.sleep(.001)
                while not link.up:
                    await asyncio.sleep(.001)
            elapsed = time.monotonic() - started
            reply = await link.narrate(NarrateOp.ACKNOWLEDGE, "Should I work out today?")
            return {"idle_recovery_ms": round(elapsed * 1000, 2),
                    "reply": [line.text for line in reply.lines]}
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)


async def answer_replay(*, late=False, streamed=True):
    from dataclasses import replace
    from types import SimpleNamespace
    import numpy as np
    from bench.latency_turn_replay import AudibleTimeline, owned_session
    from reachy_muse.capabilities import Mode
    from reachy_muse.contract.replies import ReplyStyle
    from reachy_muse.conversation import VoiceConversation
    from reachy_muse.failover import FailoverVoice, _Rest
    from reachy_muse.local_backends import backends_for
    from test_conversation import FakeHardware, chat_event

    first = "A light workout is a good option."
    second = "Take a rest day if you feel worn out."
    requests, starts, generated, producers = [], {}, {}, []
    first_text = None

    class Voice:
        available = True

        def is_presynthesized(self, text):
            return False

        async def stream(self, line, rate):
            requests.append(line.text)
            await asyncio.sleep(.25)
            generated[line.text] = time.monotonic()
            yield np.full(rate // 2, .1 if line.text == first else .2 if line.text == second else .3,
                          dtype=np.float32)

    class Hardware(FakeHardware):
        def play_audio(self, samples):
            label = round(float(samples[0]), 1)
            starts.setdefault(label, time.monotonic())
            super().play_audio(samples)

    async def send(session):
        async def answer():
            nonlocal first_text
            await asyncio.sleep(.02)
            if late:
                conversation.job.dispatched_at -= 32
            first_text = time.monotonic()
            if streamed:
                await session.events.put(chat_event("delta.text_append", "reply-1", text=first + " "))
                await asyncio.sleep(2)
            await session.events.put(chat_event("delta.message_done", "reply-1", seq=2, content=first + " " + second))
            await session.events.put(chat_event("task.status", seq=200, status="completed"))
        producers.append(asyncio.create_task(answer()))

    session = owned_session(send_hook=send)
    route = SimpleNamespace(this_turn=True, speech=_Rest("speech"))
    backends = replace(backends_for(Mode.ON_ROBOT, session), local_style=ReplyStyle.PLAIN_SHORT,
                       voice=FailoverVoice(Voice(), Voice(), route))
    conversation = VoiceConversation(session, Hardware(), backends=backends,
                                     owns_chat=True, session_id="robot-chat")
    timeline = AudibleTimeline()
    subscriber = asyncio.create_task(conversation._subscribe())
    speaker = asyncio.create_task(conversation._play_output())
    try:
        await session.chat_subscribed.wait()
        async with asyncio.timeout(10):
            await conversation.turn(b"\0" * 32044, recognized_text="Should I work out today?",
                                    timeline=timeline, defer_playback=True)
            await conversation._output_queue.join()
        await asyncio.gather(*producers)
        return {"first_sentence_to_audio_ms": round((starts[.1] - first_text) * 1000, 2),
                "sentence_gap_ms": round((starts[.2] - starts[.1] - .5) * 1000, 2),
                "spoken": requests}
    finally:
        for task in (subscriber, speaker, *producers):
            task.cancel()
        await asyncio.gather(subscriber, speaker, *producers, return_exceptions=True)
        await conversation._moments.close()


async def endpoint_replay():
    import numpy as np
    from reachy_muse.capabilities import Mode
    from reachy_muse.cli import end_of_turn_silence
    from test_end_of_turn import RATE, silence, speech, turns

    threshold = end_of_turn_silence(Mode.COMPANION, None)
    clips = {name: speech(name) for name in ("reminder", "correction", "negation")}
    reminder = clips["reminder"]
    clips["one_second_correction"] = np.concatenate([
        reminder[:round(2.04 * RATE)], silence(1), reminder[round(2.04 * RATE):]])
    clips["short_fragment_hesitation"] = np.concatenate([
        reminder[:round(.7 * RATE)], silence(1), reminder[round(.7 * RATE):]])
    return {name: turns(np.concatenate([audio, silence(3)]), threshold, 1.2)
            for name, audio in clips.items()}


async def whisper_replay():
    import sys
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from reachy_muse.local_backends import LocalHearing
    from reachy_muse.local_transcription import WhisperTranscriber

    worker_code = """
import json, struct, sys, time
time.sleep(.3)
sys.stdout.buffer.write(struct.pack('<I', 0))
sys.stdout.buffer.flush()
while header := sys.stdin.buffer.read(4):
    size, = struct.unpack('<I', header)
    sys.stdin.buffer.read(size)
    time.sleep(.02)
    packet = json.dumps({'text': 'Should I work out today?'}).encode()
    sys.stdout.buffer.write(struct.pack('<I', len(packet)) + packet)
    sys.stdout.buffer.flush()
"""
    spawn = asyncio.create_subprocess_exec

    async def fake_worker(*args, **kwargs):
        return await spawn(sys.executable, "-c", worker_code, **kwargs)

    with TemporaryDirectory() as directory:
        model = Path(directory)
        for name in ("model.bin", "config.json", "tokenizer.json"):
            (model / name).touch()
        transcriber = WhisperTranscriber(model)
        hearing = LocalHearing(transcriber)
        wav = b"RIFF" + b"\0" * 4 + b"WAVE" + b"\0" * 32
        with patch("asyncio.create_subprocess_exec", fake_worker):
            try:
                before = time.monotonic()
                await hearing.start()
                startup = time.monotonic() - before
                before = time.monotonic()
                first = await hearing.transcribe(wav)
                first_s = time.monotonic() - before
                pid = transcriber._process.pid
                before = time.monotonic()
                second = await hearing.transcribe(wav)
                return {"startup_ms": round(startup * 1000, 2),
                        "first_fallback_ms": round(first_s * 1000, 2),
                        "second_fallback_ms": round((time.monotonic() - before) * 1000, 2),
                        "worker_reused": transcriber._process.pid == pid,
                        "transcripts": [first, second]}
            finally:
                await transcriber.close()


async def progress_replay():
    from reachy_muse.progress import ProgressPlan, backend_status_from_event

    plan = ProgressPlan("Should I work out today?", 0)
    plan.set_status(backend_status_from_event("working", "is working"), 1.7)
    spoken = []
    for tick in range(320):
        now = tick / 10
        if update := plan.take(now):
            spoken.append({"at_s": now, "text": update.text})
    return {"waiting_seconds": 32, "spoken": spoken, "line_count": len(spoken)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("engine", "jitter", "reconnect", "answer", "late-answer", "pipeline", "endpoint", "whisper", "progress"))
    args = parser.parse_args()
    experiments = {"engine": engine_replay, "jitter": jitter_replay, "reconnect": reconnect_replay,
                   "answer": answer_replay, "late-answer": lambda: answer_replay(late=True),
                   "pipeline": lambda: answer_replay(streamed=False), "endpoint": endpoint_replay,
                   "whisper": whisper_replay, "progress": progress_replay}
    print(json.dumps(asyncio.run(experiments[args.experiment]()), indent=2))
