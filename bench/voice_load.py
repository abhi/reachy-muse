"""Time companion speech end to end, alone and under the companion's real concurrent load.

    .venv/bin/python bench/voice_load.py --voice chatterbox-turbo [--runs N] [--out FILE]

It starts its own `reachy-companion serve --insecure --motion --voice VOICE` on a free port with a temporary
pairing, so a companion serving the robot keeps running. Then it acts as the robot, at the robot's 16 kHz:
- speak: each line is timed from sending speak_start to every speech frame received. First audio is the first
  frame; RTF is the time to the last frame over the audio's length. A playback model starts each line after the
  first frame plus a jitter buffer and counts any frame that arrives after the audio before it has finished.
- speak under load: the same lines while a second connection hears tests/companion_server/data/reminder.wav in
  real time over and over, a third asks for narrated lines back to back, and a fourth keeps two animate requests
  outstanding.
- paths: acknowledge, lines, classify and hear_final (as in bench/generated_motion.py), alone and while another
  connection speaks without pause.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generated_motion import CORPUS, REPLY, REQUEST, keep_animating, speech_paths  # noqa: E402
from latency import hear, memory, read_wav, robot, spread, start_server  # noqa: E402

from reachy_muse.companion_client import CompanionLink  # noqa: E402
from reachy_muse.protocol import (  # noqa: E402
    AudioKind, Hello, Narrate, NarrateOp, Op, Sender, SpeakDone, SpeakStart, VoiceRole, Welcome, decode,
    decode_audio, encode,
)

RATE = 16_000
LINES = (
    "Hi there! It's really good to see you again.",
    "Okay, checking your calendar for tomorrow.",
    "Still working on it. I'm almost there.",
    "Your flight to Lisbon leaves Tuesday at 7:45 am from gate 12, and the fare is 389 dollars.",
    "Sorry, I couldn't reach Muse just now. Please try again in a minute.",
    "The weather in Oslo tomorrow looks mild. Expect a high of 18 degrees with a 70 percent chance of rain in "
    "the afternoon. You might want an umbrella when you leave at 8.",
    "Got it, one moment.",
    "You have 3 unread messages and 1 missed call.",
)
JITTER_S = (0.0, 0.15, 0.3)
NARRATE_EVERY_S = 0.0


async def speak_timed(ws, stream: int, text: str) -> dict:
    await ws.send(encode(SpeakStart(stream, text, VoiceRole.MAIN)))
    started = time.perf_counter()
    arrivals: List[tuple] = []
    while True:
        frame = await ws.recv()
        if isinstance(frame, bytes):
            audio = decode_audio(frame)
            assert audio.kind is AudioKind.SPEECH and audio.stream == stream
            arrivals.append((time.perf_counter() - started, len(audio.pcm) / 2 / RATE))
            continue
        if isinstance(decode(frame, sender=Sender.COMPANION), SpeakDone):
            break
    audio_s = sum(d for _, d in arrivals)
    result = {"first_s": arrivals[0][0], "rtf": arrivals[-1][0] / audio_s, "audio_s": audio_s}
    for jitter in JITTER_S:
        cursor = arrivals[0][0] + jitter
        gaps = 0.0
        for arrived, length in arrivals:
            if arrived > cursor:
                gaps += arrived - cursor
                cursor = arrived
            cursor += length
        result[f"underrun_s_at_{int(jitter * 1000)}ms"] = gaps
    return result


async def speak_all(url: str, token: str, runs: int, streams) -> List[dict]:
    ws = await robot(url, token, out_rate=RATE)
    results = []
    for _ in range(runs):
        for text in LINES:
            results.append(await speak_timed(ws, next(streams), text))
    await ws.close()
    return results


def summarize(results: List[dict]) -> dict:
    summary = {"first_audio_s": spread([r["first_s"] for r in results]), "rtf": spread([r["rtf"] for r in results])}
    for jitter in JITTER_S:
        key = f"underrun_s_at_{int(jitter * 1000)}ms"
        hit = [r[key] for r in results if r[key] > 0.005]
        summary[f"jitter_{int(jitter * 1000)}ms"] = {"lines_with_underrun": f"{len(hit)}/{len(results)}",
                                                    "worst_gap_s": round(max(hit, default=0.0), 3),
                                                    "first_audio_p95_s": spread([r["first_s"] + jitter for r in results])["p95"]}
    return summary


async def background_load(url: str, token: str, stop: asyncio.Event) -> None:
    from websockets.asyncio.client import connect

    async def hearing() -> None:
        ws = await robot(url, token, out_rate=RATE)
        pcm = read_wav(CORPUS / "reminder.wav")
        for stream in itertools.count(1001, 2):
            if stop.is_set():
                break
            await hear(ws, stream, pcm, finish=True)
        await ws.close()

    async def narrating() -> None:
        ws = await connect(url, max_size=None)
        await ws.send(encode(Hello((1,), token, (Op.NARRATE,), 24_000, "bench-load")))
        assert isinstance(decode(await ws.recv(), sender=Sender.COMPANION), Welcome)
        for identity in itertools.count(1):
            if stop.is_set():
                break
            await ws.send(encode(Narrate(identity, NarrateOp.LINES, REQUEST, reply=REPLY)))
            await ws.recv()
            await asyncio.sleep(NARRATE_EVERY_S)
        await ws.close()

    link = CompanionLink(url.removesuffix("/v1"), token=token, insecure=True, out_rate=24_000)
    running = asyncio.create_task(link.run())
    await link.settled()
    made: List[float] = []
    try:
        await asyncio.gather(hearing(), narrating(), keep_animating(link, stop, made))
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


async def keep_speaking(url: str, token: str, stop: asyncio.Event) -> None:
    ws = await robot(url, token, out_rate=RATE)
    for stream, text in zip(itertools.count(5000, 2), itertools.cycle(LINES)):
        if stop.is_set():
            break
        await speak_timed(ws, stream, text)
    await ws.close()


async def under(load, url: str, token: str, work):
    stop = asyncio.Event()
    loading = asyncio.create_task(load(url, token, stop))
    await asyncio.sleep(2.0)
    try:
        return await work()
    finally:
        stop.set()
        await asyncio.gather(loading, return_exceptions=True)


async def measure(url: str, token: str, runs: int) -> dict:
    streams = itertools.count(2, 2)
    alone = await speak_all(url, token, runs, streams)
    loaded = await under(background_load, url, token, lambda: speak_all(url, token, runs, streams))
    paths_alone = await speech_paths(url, token, runs)
    paths_speaking = await under(keep_speaking, url, token, lambda: speech_paths(url, token, runs))
    return {"speak_alone": summarize(alone), "speak_under_load": summarize(loaded),
            "paths_alone_s": {k: spread(v) for k, v in paths_alone.items()},
            "paths_while_speaking_s": {k: spread(v) for k, v in paths_speaking.items()}}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--narrate-every", type=float, default=0.0, metavar="S",
                        help="pause between the load's narrations; 0 asks back to back")
    args = parser.parse_args()
    global NARRATE_EVERY_S
    NARRATE_EVERY_S = args.narrate_every
    started = time.perf_counter()
    server, url, token = start_server(["--motion", "--voice", args.voice])
    threading.Thread(target=lambda: [print(line, end="", file=sys.stderr, flush=True) for line in server.stderr],
                     daemon=True).start()
    try:
        result = {"voice": args.voice, "narrate_every_s": args.narrate_every, "gpu_env": {k: v for k, v in os.environ.items() if k.startswith("MLX_")}, "load_s": round(time.perf_counter() - started, 1), "memory_loaded": memory(server.pid)}
        result.update(await measure(url, token, args.runs))
        result["memory_after"] = memory(server.pid)
    finally:
        server.terminate()
        server.wait(30)
    text = json.dumps(result, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    asyncio.run(main())
