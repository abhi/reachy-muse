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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("engine", "jitter"))
    args = parser.parse_args()
    experiments = {"engine": engine_replay, "jitter": jitter_replay}
    print(json.dumps(asyncio.run(experiments[args.experiment]()), indent=2))
