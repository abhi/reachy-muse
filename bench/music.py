"""Offline music CPU and wake replay. No robot, speaker, services or Muse calls."""

import argparse
import asyncio
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tempfile
import threading
import time
import wave

import numpy as np

from reachy_muse.motion.dance import BeatTracker, DanceTimeline
from reachy_muse.music_audio import decode


def synthetic_music(seconds, rate=16000):
    t = np.arange(round(seconds * rate)) / rate
    pulse = np.exp(-((t % .5) / .025))
    chord = .12 * np.sin(2 * np.pi * 220 * t) + .08 * np.sin(2 * np.pi * 330 * t)
    return (chord + .65 * pulse * np.sin(2 * np.pi * 80 * t)).astype(np.float32)


def cpu_trial(rate, seconds=600):
    chunk = synthetic_music(1, rate)
    tracker, dance = BeatTracker(rate), DanceTimeline()
    wall, cpu = time.perf_counter(), time.process_time()
    beats = ticks = 0
    for second in range(seconds):
        for offset in range(0, rate, rate // 50):
            part = chunk[offset:offset + rate // 50]
            found = tracker.feed(part)
            beats += len(found)
            start = second + offset / rate
            dance.offer(found, second * rate + offset, start + .06, rate)
            dance.sample(start, eligible=True)
            ticks += 1
    cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
    return {"rate": rate, "audio_s": seconds, "cpu_s": round(cpu, 4), "wall_s": round(wall, 4),
            "one_core_percent": round(100 * cpu / seconds, 4), "beats": beats, "motion_ticks": ticks}


def decode_trial():
    import av

    with tempfile.TemporaryDirectory(prefix="reachy-music-bench-") as folder:
        path = Path(folder) / "synthetic.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            wav.writeframes((synthetic_music(60, 48000) * 32767).astype("<i2").tobytes())
        cpu, wall = time.process_time(), time.perf_counter()
        samples = sum(len(block.pcm) for block in decode(path, 16000, threading.Event()))
        cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
    return {"audio_s": 60, "output_samples": samples, "cpu_s": round(cpu, 4),
            "wall_s": round(wall, 4), "one_core_percent": round(100 * cpu / 60, 4)}


async def wake_trial(model, clip):
    from reachy_muse.wake_policy import WakeContext, WakePolicy
    from reachy_muse.wake_word import WakeWordDetector

    with wave.open(str(clip), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    detector = WakeWordDetector(model, threshold=.1, candidate_replay=True)
    policy = WakePolicy()
    results = []
    try:
        music = synthetic_music(len(audio) / 16000)
        speech_rms = np.sqrt(np.mean(audio ** 2))
        music_rms = np.sqrt(np.mean(music ** 2))
        trials = [("clean", audio, False)]
        for snr in (20, 10, 0):
            mixed = audio + music * speech_rms / music_rms / 10 ** (snr / 20)
            trials.append((f"music_{snr}_db", np.clip(mixed, -1, 1), True))
        trials.append(("music_only", synthetic_music(30), True))
        for name, pcm, playing in trials:
            detector.reset()
            policy.reset(new_trial=True)
            decisions, hits = [], 0
            cpu = time.process_time()
            for offset in range(0, len(pcm), 256):
                hits += int(await policy.feed(detector, pcm[offset:offset + 256], context=WakeContext(music=playing)))
                if policy.last_decision is not None:
                    decisions.append({"accepted": policy.last_decision.accepted,
                                      "reason": policy.last_decision.reason,
                                      "replay_floor": policy.last_decision.replay_floor})
            results.append({"condition": name, "hits": hits, "audio_s": len(pcm) / 16000,
                            "cpu_s": round(time.process_time() - cpu, 4), "decisions": decisions})
    finally:
        detector.close()
    return {"clip_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
            "verifier": "unavailable; production conservative replay", "trials": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wake-model", type=Path)
    parser.add_argument("--wake-clip", type=Path)
    args = parser.parse_args()
    result = {"machine": platform.machine(), "system": platform.platform(), "python": platform.python_version(),
              "versions": {name: version(name) for name in ("numpy", "av")},
              "tracker_and_motion": [cpu_trial(16000), cpu_trial(48000)], "decode_and_tracker": decode_trial()}
    if args.wake_model and args.wake_clip:
        result["wake"] = asyncio.run(wake_trial(args.wake_model, args.wake_clip))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
