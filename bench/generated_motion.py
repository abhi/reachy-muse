"""Measure generated motion on a real companion, and what it costs hearing, speech, narration and classify.

    .venv/bin/python bench/generated_motion.py [--port 8799] [--runs N] [--out FILE]

It pairs into a temporary state directory and starts its own `reachy-companion serve --insecure --motion`,
so a companion already serving the robot keeps running untouched. Then it acts as the robot:
- cold: one animate request per moment intent, timed from request to frames, as a cue that found nothing ready.
- prefetched: the robot's MomentDirector over a real CompanionLink, timed from cue to play_motion.
- contention: narrate, classify, speak and hear, each timed alone and again while a second connection keeps
  two animate requests outstanding, the most the robot ever has.
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
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latency import hear, memory, read_wav, robot, speak, speech_end_s, spread, start_server  # noqa: E402

from reachy_muse.companion_client import CompanionLink, CompanionMotion  # noqa: E402
from reachy_muse.device_command import DeviceCommand  # noqa: E402
from reachy_muse.motion.moments import MOMENTS, Moment, MomentDirector  # noqa: E402
from reachy_muse.protocol import (  # noqa: E402
    Classify, ClassifyResult, Narrate, NarrateOp, NarrateResult, Op, Sender, decode, encode,
)

CORPUS = Path(__file__).resolve().parent.parent / "tests" / "companion_server" / "data"
REQUEST = "What's the weather in San Jose tomorrow morning?"
REPLY = ("Tomorrow morning in San Jose starts cool and cloudy at about 11 degrees. The clouds clear by noon "
         "and it reaches 19 degrees. There is no rain in the forecast, but the wind picks up after 4 pm.")
COMMAND = "could you turn it down a little"


class Asker:
    """One narrate and classify connection; requests run one at a time so each timing is clean."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.ids = itertools.count(1)

    async def ask(self, build: Callable[[int], object], answer: type) -> float:
        identity = next(self.ids)
        started = time.perf_counter()
        await self.ws.send(encode(build(identity)))
        while True:
            message = decode(await self.ws.recv(), sender=Sender.COMPANION)
            if isinstance(message, answer) and message.id == identity:
                return time.perf_counter() - started
            raise RuntimeError(f"unexpected {message}")


async def speech_paths(url: str, token: str, runs: int) -> Dict[str, List[float]]:
    from websockets.asyncio.client import connect
    from reachy_muse.protocol import Hello, Welcome

    times: Dict[str, List[float]] = {"acknowledge": [], "lines": [], "classify": [], "speak_first_audio": [],
                                     "hear_final": []}
    # hear_final runs from the end of speech in the clip to the companion's endpoint.
    ws = await connect(url, max_size=None)
    await ws.send(encode(Hello((1,), token, (Op.NARRATE, Op.CLASSIFY), 24_000, "bench")))
    assert isinstance(decode(await ws.recv(), sender=Sender.COMPANION), Welcome)
    asker = Asker(ws)
    voice = await robot(url, token)
    pcm = read_wav(CORPUS / "reminder.wav")
    stream = itertools.count(1, 2)
    speak_stream = itertools.count(2, 2)
    for _ in range(runs):
        times["acknowledge"].append(await asker.ask(lambda i: Narrate(i, NarrateOp.ACKNOWLEDGE, REQUEST), NarrateResult))
        times["lines"].append(await asker.ask(lambda i: Narrate(i, NarrateOp.LINES, REQUEST, reply=REPLY), NarrateResult))
        times["classify"].append(await asker.ask(
            lambda i: Classify(i, COMMAND, (DeviceCommand.VOLUME_DOWN, DeviceCommand.SLEEP)), ClassifyResult))
        times["speak_first_audio"].append((await speak(voice, next(speak_stream), "Sure, I can help with that.")).first_frame_s)
        heard = await hear(voice, next(stream), pcm, finish=True)
        times["hear_final"].append(heard.endpoint_s - speech_end_s(pcm))
    await ws.close()
    await voice.close()
    return times


async def keep_animating(link: CompanionLink, stop: asyncio.Event, made: List[float]) -> None:
    intents = itertools.cycle(intent for spec in MOMENTS.values() for intent in spec.intents)

    async def one() -> None:
        while not stop.is_set():
            started = time.perf_counter()
            await link.animate(next(intents), 4.0)
            made.append(time.perf_counter() - started)

    await asyncio.gather(one(), one())


class Recorder:
    def __init__(self) -> None:
        self.played: List[float] = []

    def play_motion(self, frames, fps, **options) -> None:
        self.played.append(time.perf_counter())

    def stop_motion(self) -> None:
        pass


async def motion(url: str, token: str, runs: int) -> dict:
    link = CompanionLink(url.removesuffix("/v1"), token=token, insecure=True, out_rate=24_000)
    running = asyncio.create_task(link.run())
    await link.settled()
    assert link.can(Op.ANIMATE), "the companion does not offer animate"

    cold: Dict[str, List[float]] = {moment.value: [] for moment in MOMENTS}
    frames: Dict[str, List[float]] = {moment.value: [] for moment in MOMENTS}
    for _ in range(runs):
        for moment, spec in MOMENTS.items():
            for intent in spec.intents:
                started = time.perf_counter()
                result = await link.animate(intent, spec.duration_s)
                cold[moment.value].append(time.perf_counter() - started)
                frames[moment.value].append(len(result.frames) / result.fps)

    hardware = Recorder()
    director = MomentDirector(CompanionMotion(link), hardware)
    prefetched: Dict[str, List[float]] = {moment.value: [] for moment in MOMENTS}
    connected = time.perf_counter()
    keeping = asyncio.create_task(director.keep_ready(every_s=.05))
    while len(director._ready) < len(MOMENTS):
        await asyncio.sleep(.01)
    all_ready_s = time.perf_counter() - connected
    for _ in range(runs):
        for moment in (Moment.GREETING, Moment.THINKING):
            while moment not in director._ready:
                await asyncio.sleep(.01)
            cued = time.perf_counter()
            director.cue(moment)
            while len(hardware.played) == sum(map(len, prefetched.values())):
                await asyncio.sleep(.001)
            prefetched[moment.value].append((hardware.played[-1] - cued) * 1000)
        for expression, moment in (("happy", Moment.CELEBRATION), ("sad", Moment.APOLOGY)):
            while moment not in director._ready:
                await asyncio.sleep(.01)
            from reachy_muse.contract.expressions import Expression
            director.prepare_closing(Expression(expression))
            spoken = time.perf_counter()
            director.finish_turn(lambda: True)
            while len(hardware.played) == sum(map(len, prefetched.values())):
                await asyncio.sleep(.001)
            prefetched[moment.value].append((hardware.played[-1] - spoken) * 1000)
    keeping.cancel()
    director.stop()
    running.cancel()
    await asyncio.gather(running, return_exceptions=True)
    return {"cold_s": {k: spread(v) for k, v in cold.items()}, "motion_s": {k: spread(v) for k, v in frames.items()},
            "prefetched_ms": {k: spread(v) for k, v in prefetched.items()}, "all_moments_ready_after_connect_s":
            round(all_ready_s, 2)}


async def contention(url: str, token: str, runs: int) -> dict:
    alone = await speech_paths(url, token, runs)
    link = CompanionLink(url.removesuffix("/v1"), token=token, insecure=True, out_rate=24_000)
    running = asyncio.create_task(link.run())
    await link.settled()
    stop = asyncio.Event()
    made: List[float] = []
    animating = asyncio.create_task(keep_animating(link, stop, made))
    try:
        contended = await speech_paths(url, token, runs)
    finally:
        stop.set()
        await asyncio.gather(animating, return_exceptions=True)
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
    return {"alone_s": {k: spread(v) for k, v in alone.items()},
            "with_motion_s": {k: spread(v) for k, v in contended.items()},
            "motion_while_contended_s": spread(made)}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--only", choices=("motion", "contention"))
    args = parser.parse_args()
    started = time.perf_counter()
    server, url, token = start_server(["--port", str(args.port), "--motion"])
    # Keep reading the server's log so a full pipe never blocks it.
    threading.Thread(target=lambda: [print(line, end="", file=sys.stderr, flush=True) for line in server.stderr],
                     daemon=True).start()
    try:
        result = {"load_s": round(time.perf_counter() - started, 1), "memory_loaded": memory(server.pid)}
        if args.only != "contention":
            result["motion"] = await motion(url, token, args.runs)
        if args.only != "motion":
            result["contention"] = await contention(url, token, args.runs)
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
