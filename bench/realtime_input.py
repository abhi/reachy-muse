"""Offline CPU replay of the robot's call input path. No devices or network.

PYTHONPATH=src python bench/realtime_input.py --streaming-model /path/to/model --json report.json

The pipeline run includes VAD, recording, echo matching, buffering and 24 kHz
resampling. The optional native run measures the configured single-thread Sherpa
decoder over the same fixtures, including end-of-turn padding. Model loading is
reported separately. CPU seconds per audio second are fractions of one host core,
not measurements of the robot's CM4. Inference and pipeline costs are additive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
import wave

import numpy as np

from reachy_muse.local_backends import LocalHearing
from reachy_muse.realtime_call import CallInput, MicrophoneFeed, _Resampler

RATE = 16000
CHUNK = 256
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "companion_server" / "data"


def recordings():
    audio = []
    for path in sorted(FIXTURES.glob("*.wav")):
        with wave.open(str(path), "rb") as source:
            assert source.getsampwidth() == 2 and source.getnchannels() == 1
            samples = np.frombuffer(source.readframes(source.getnframes()), "<i2").astype(np.float32) / 32768
            samples = _Resampler(source.getframerate(), RATE, output="flt")(samples)
        audio.append(np.concatenate((samples, np.zeros(RATE, np.float32))))
    return audio


def measurements(cpu, wall, seconds, latencies):
    return {"audio_s": round(seconds, 3), "cpu_s": round(cpu, 4), "wall_s": round(wall, 4),
            "cpu_fraction_of_one_core": round(cpu / seconds, 5),
            "feed_p95_ms": round(float(np.percentile(latencies, 95)) * 1000, 3),
            "feed_max_ms": round(max(latencies) * 1000, 3)}


def pipeline(audio, repeats, vad):
    # Warm PyAV and VAD imports before measuring the steady-state microphone work.
    LocalHearing().open_turn(RATE, silence_s=.2, commands_only=True, vad=vad)
    latencies = []
    endpoints = 0
    seconds = 0
    cpu, wall = time.process_time(), time.perf_counter()
    for _ in range(repeats):
        for samples in audio:
            turn = LocalHearing().open_turn(RATE, silence_s=.2, commands_only=True, vad=vad)
            feed = MicrophoneFeed(RATE)
            gate = CallInput(feed, RATE)
            for offset in range(0, len(samples), CHUNK):
                chunk = samples[offset:offset + CHUNK]
                start = time.perf_counter()
                turn.feed(chunk)
                heard = turn.take()
                gate.push(chunk, speaking=turn.speech_active and turn.quiet_s == 0,
                          text="sunny tomorrow", playback=True,
                          heard_over=("It will be sunny tomorrow. Would you like a reminder?",),
                          quiet_s=turn.quiet_s)
                if heard.ended is not None:
                    heard.ended.cancel()
                    gate.reset()
                    endpoints += 1
                latencies.append(time.perf_counter() - start)
            turn.abort()
            seconds += len(samples) / RATE
    result = measurements(time.process_time() - cpu, time.perf_counter() - wall, seconds, latencies)
    result["endpoints"] = endpoints
    return result


def native(audio, repeats, model_path):
    from reachy_muse.streaming_transcription import _load_model, _SherpaTurn

    start = time.perf_counter()
    model = _load_model(model_path)
    load_s = time.perf_counter() - start
    latencies = []
    final_latencies = []
    seconds = 0
    cpu, wall = time.process_time(), time.perf_counter()
    for _ in range(repeats):
        for samples in audio:
            turn = _SherpaTurn(model)
            for offset in range(0, len(samples), CHUNK):
                pcm = np.rint(samples[offset:offset + CHUNK] * 32767).astype("<i2").tobytes()
                start = time.perf_counter()
                turn.feed(pcm)
                latencies.append(time.perf_counter() - start)
            start = time.perf_counter()
            turn.finish()
            final_latencies.append(time.perf_counter() - start)
            turn.close()
            seconds += len(samples) / RATE
    result = measurements(time.process_time() - cpu, time.perf_counter() - wall, seconds, latencies)
    result.update(model_load_s=round(load_s, 3), final_max_ms=round(max(final_latencies) * 1000, 3))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streaming-model", type=Path)
    parser.add_argument("--vad", choices=("webrtc", "silero"), default="webrtc")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    audio = recordings()
    vad = None
    if args.vad == "silero":
        from reachy_muse.speech_gate import SileroSpeechGate
        vad = SileroSpeechGate()
    report = {"host": f"{platform.system()} {platform.machine()}", "chunk_ms": 16,
              "vad": args.vad, "pipeline": pipeline(audio, args.repeats, vad)}
    if args.streaming_model is not None:
        report["sherpa_native"] = native(audio, args.repeats, args.streaming_model)
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.json is not None:
        args.json.write_text(rendered)


if __name__ == "__main__":
    main()
