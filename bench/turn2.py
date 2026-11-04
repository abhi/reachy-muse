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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("engine",))
    args = parser.parse_args()
    print(json.dumps(asyncio.run(engine_replay()), indent=2))
