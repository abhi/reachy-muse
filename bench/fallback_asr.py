"""CPU-only Whisper through the production worker. No robot, network, or companion."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import math
from pathlib import Path
import platform
import time
import wave

import numpy as np

from reachy_muse.local_transcription import WhisperTranscriber


DATA = Path(__file__).resolve().parents[1] / "tests/companion_server/data"


def wav_bytes(audio):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes((audio * 32768).astype("<i2").tobytes())
    return output.getvalue()


async def measure(model, runs):
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    report = {"machine": platform.machine(), "runs": runs, "cases": []}
    transcriber = WhisperTranscriber(model)
    try:
        started = time.perf_counter()
        await transcriber.start()
        report["startup_s"] = time.perf_counter() - started
        for case in json.loads((DATA / "manifest.json").read_text())["cases"]:
            with wave.open(str(DATA / case["file"])) as wav:
                assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
                audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
            speech = get_speech_timestamps(audio, VadOptions(speech_pad_ms=250))
            assert speech, case["file"]
            # Only this benchmark removes corpus padding. Recognition always gets the
            # complete selected WAV, including every speech region and its margins.
            short = audio[speech[0]["start"]:speech[-1]["end"]]
            for name, samples in (("short", short), ("original", audio)):
                durations, texts = [], set()
                wav = wav_bytes(samples)
                for _ in range(runs):
                    started = time.perf_counter()
                    texts.add(await transcriber.transcribe(wav))
                    durations.append(time.perf_counter() - started)
                durations.sort()
                report["cases"].append({
                    "name": case["name"], "window": name, "audio_s": len(samples) / 16000,
                    "reference": case["reference"], "recognized": sorted(texts),
                    "p50_s": durations[len(durations) // 2],
                    "p95_s": durations[math.ceil(len(durations) * .95) - 1],
                })
    finally:
        await transcriber.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(asyncio.run(measure(args.model, args.runs)), indent=2))
