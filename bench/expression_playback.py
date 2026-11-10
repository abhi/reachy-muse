"""Replay chat and realtime first audio through the actual controller and a memory-only sink.

Run the same script against both source trees. No devices, network, or models.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from types import SimpleNamespace
import time

import numpy as np

from reachy_muse.capabilities import Mode
from reachy_muse.conversation import VoiceConversation
from reachy_muse.hardware import ReachyController
from reachy_muse.local_backends import backends_for
from reachy_muse.motion import MotionStyle
from reachy_muse.motion.layers import MotionLayers
from reachy_muse.motion.reply_motion import ReplyMotion
from reachy_muse.realtime_call import RealtimeSpeaker
from test_conversation import FakeHardware

TEXT = "The report contains three sections."


class Hardware(FakeHardware):
    output_sample_rate = 48000

    def __init__(self):
        super().__init__()
        self.pushes = []
        self.pcm = []
        self.controller = ReachyController()
        self.controller._np = np
        self.controller._output_sample_rate = self.output_sample_rate
        self.controller._layers = MotionLayers(MotionStyle(sway="sdk"))
        self.controller._require_running = lambda: None
        self.controller._robot = SimpleNamespace(media=SimpleNamespace(push_audio_sample=self.accept))
        self.controller._start_music()

    def accept(self, samples):
        self.pushes.append(time.perf_counter())
        self.pcm.append(samples.copy())

    def play_audio(self, samples, **kwargs):
        return self.controller.play_audio(samples, **kwargs)

    def finish_sentence(self):
        self.controller.finish_sentence()

    def clear_audio(self):
        pass


def pcm(rate, seconds):
    t = np.arange(int(rate * seconds)) / rate
    return (.15 * np.sin(2 * np.pi * np.cumsum(150 + 100 * t) / rate)).astype(np.float32)


async def once(path):
    hardware = Hardware()
    ready = None
    if path == "realtime":
        async def nothing():
            pass
        motion = ReplyMotion()
        speaker = RealtimeSpeaker(hardware, motion, on_audible=lambda *args: None,
                                  on_start=nothing, on_idle=nothing)
        motion.words("reply", TEXT)
        audio = (pcm(24000, .04) * 32767).astype("<i2").tobytes()
        began = time.perf_counter()
        ready = began
        motion.audio("reply", 0, audio)
        speaker.chunk("reply", 0, audio)
        while not hardware.pushes:
            await asyncio.sleep(0)
        await speaker.close()
    else:
        async def stream():
            nonlocal ready
            if path == "streamed":
                await asyncio.sleep(.005)
            # A large chunk exercises post-push analysis too.
            audio = pcm(hardware.output_sample_rate, .8)
            ready = time.perf_counter()
            yield audio
        conversation = VoiceConversation(None, hardware, backends=backends_for(Mode.MUSE_VOICE, None))
        began = time.perf_counter()
        await conversation._speak_owned("reply", text=TEXT, prepared_stream=stream(),
                                       stream_segment=True, drain=False)
        await conversation._moments.close()
    return {"first_ms": (hardware.pushes[0] - began) * 1000,
            "ready_to_sink_ms": (hardware.pushes[0] - ready) * 1000,
            "sha256": hashlib.sha256(np.concatenate(hardware.pcm).tobytes()).hexdigest(),
            "frames": sum(map(len, hardware.pcm))}


async def run(trials):
    report = {}
    for path in ("prepared", "streamed", "realtime"):
        await once(path)  # warm imports and thread creation
        rows = [await once(path) for _ in range(trials)]
        assert len({row["sha256"] for row in rows}) == 1
        report[path] = {"first_push_ms": {key: round(float(np.percentile([r["first_ms"] for r in rows], p)), 4)
                                         for key, p in (("p50", 50), ("p95", 95), ("max", 100))},
                        "ready_to_sink_ms": {key: round(float(np.percentile([r["ready_to_sink_ms"] for r in rows], p)), 4)
                                             for key, p in (("p50", 50), ("p95", 95), ("max", 100))},
                        "pcm_sha256": rows[0]["sha256"], "pcm_frames": rows[0]["frames"]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--baseline")
    parser.add_argument("--json")
    args = parser.parse_args()
    report = asyncio.run(run(args.runs))
    if args.baseline:
        with open(args.baseline) as source:
            before = json.load(source)
        for path, result in report.items():
            assert result["pcm_sha256"] == before[path]["pcm_sha256"]
            assert result["pcm_frames"] == before[path]["pcm_frames"]
            result["delta_ms"] = {k: round(result["first_push_ms"][k] - before[path]["first_push_ms"][k], 4)
                                  for k in ("p50", "p95", "max")}
    output = json.dumps(report, indent=2) + "\n"
    if args.json:
        with open(args.json, "w") as destination:
            destination.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
