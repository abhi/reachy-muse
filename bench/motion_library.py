"""Measure library serving and storage with synthetic trajectories; never load ML models."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from websockets.asyncio.client import connect

from reachy_muse.companion_server.motion_library import INTENTS, LibraryAnimator, build_library
from reachy_muse.companion_server.server import CompanionServer, Engines
from reachy_muse.contract.commands import POSE_LIMITS
from reachy_muse.protocol import Animate, AnimateResult, Hello, Op, Sender, decode, encode


class SyntheticEngine:
    def generate(self, intent, duration_hint_s, *, seed):
        phase = np.linspace(0, 2 * np.pi, round(25 * duration_hint_s))
        frames = np.zeros((len(phase), 9))
        for index, (low, high) in enumerate(POSE_LIMITS.values()):
            frames[:, index] = np.sin(phase * (1 + seed % 3)) * min(abs(low), abs(high)) * 0.2
        return SimpleNamespace(fps=25, recipe="synthetic sine", frames=frames)


def percentiles(values):
    return {label: round(float(np.percentile(values, percentile)), 3)
            for label, percentile in (("p50_ms", 50), ("p95_ms", 95), ("max_ms", 100))}


async def measure(requests=1000, variants=12):
    with tempfile.TemporaryDirectory(prefix="motion-library-bench-") as directory:
        path = Path(directory) / "motion-library"
        count = build_library(path, SyntheticEngine(), {"fixture": "synthetic-v1"}, variants=variants)
        files = [p for p in path.rglob("*") if p.is_file()]
        animator = LibraryAnimator(path)
        await animator.load()
        keys = tuple(INTENTS)
        direct = []
        for index in range(requests):
            key = keys[index % len(keys)]
            started = time.perf_counter_ns()
            result = await animator.animate(key, INTENTS[key].duration_s)
            direct.append((time.perf_counter_ns() - started) / 1e6)
            assert result.frames

        wire = []
        async with CompanionServer(Engines(motion=animator), "bench-library-token", port=0) as server:
            async with connect(server.url) as ws:
                await ws.send(encode(Hello((1,), "bench-library-token", (Op.ANIMATE,), 24000, "offline-bench")))
                await ws.recv()
                for index in range(requests):
                    key = keys[index % len(keys)]
                    request = encode(Animate(index, key, round(INTENTS[key].duration_s * 1000)))
                    started = time.perf_counter_ns()
                    await ws.send(request)
                    result = decode(await ws.recv(), sender=Sender.COMPANION)
                    wire.append((time.perf_counter_ns() - started) / 1e6)
                    assert isinstance(result, AnimateResult) and result.frames
        return {"fixture": "synthetic trajectories, no model execution", "intents": len(keys),
                "variants_per_intent": variants, "motions": count, "requests": requests,
                "files": len(files), "disk_bytes": sum(p.stat().st_size for p in files),
                "allocated_bytes": sum(p.stat().st_blocks * 512 for p in files),
                "animate": percentiles(direct), "websocket_roundtrip": percentiles(wire)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--variants", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(measure(args.requests, args.variants)), indent=2))
