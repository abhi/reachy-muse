"""Offline CPU speaker evaluation. Synthetic WAVs stay in the supplied directory.

Run with --synthesize once on macOS; the saved WAVs can then be evaluated on
another CPU without TTS. No network, robot, companion, or Muse connections.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import wave

import numpy as np

from reachy_muse.voice_verify import SpeakerEncoder, enroll

VOICES = ("Samantha", "Daniel", "Fred", "Kathy", "Karen", "Junior", "Ralph")
PHRASES = (
    "Hey Muse, please listen to my voice.",
    "Hey Muse, what is on my calendar?",
    "Hey Muse, tell me about the weather.",
    "Hey Muse.",
    "Hey Muse, could you help me?",
    "Hey Muse, what time is it?",
    "Hey Muse, tell me something interesting.",
)


def rates(genuine, impostor, threshold):
    if not genuine or not impostor:
        raise ValueError("Evaluation needs genuine and impostor scores")
    return {"far": sum(v >= threshold for v in impostor) / len(impostor),
            "frr": sum(v < threshold for v in genuine) / len(genuine)}


def eer(genuine, impostor):
    candidates = sorted(set([-1., 1.000001, *genuine, *impostor]))
    points = [rates(genuine, impostor, threshold) for threshold in candidates]
    closest = min(points, key=lambda point: abs(point["far"] - point["frr"]))
    return (closest["far"] + closest["frr"]) / 2


def calibrate(genuine, impostor):
    # Choose the lowest hundredth with a 0.02 margin over calibration impostors.
    # This target is empirical zero FAR on calibration, not a population claim.
    threshold = math.ceil((max(impostor) + .02) * 100) / 100
    if not -1 < threshold < 1:
        raise ValueError("No usable calibrated threshold")
    return threshold


def load(path):
    with wave.open(str(path)) as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
            raise ValueError("Benchmark WAVs must be mono PCM16 at 16 kHz")
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if len(audio) < 6400:
        raise ValueError("TTS returned no usable audio")
    return audio


def conditions(audio, seed):
    rng = np.random.default_rng(seed)
    yield "clean", audio
    distant = audio * .25
    delayed = np.pad(distant[:-640], (640, 0))
    yield "distance_proxy", distant + .25 * delayed
    rms = float(np.sqrt(np.mean(audio ** 2)))
    for snr in (20, 10):
        noise = rng.normal(0, rms / 10 ** (snr / 20), len(audio)).astype(np.float32)
        yield f"snr_{snr}", np.clip(audio + noise, -1, 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--synthesize", action="store_true")
    args = parser.parse_args(argv)
    if args.synthesize:
        args.audio_dir.mkdir(parents=True, exist_ok=True)
        for voice in VOICES:
            for index, phrase in enumerate(PHRASES):
                path = args.audio_dir / f"{voice}-{index}.wav"
                subprocess.run(["say", "-v", voice, "-r", "170", "-o", str(path),
                                "--file-format=WAVE", "--data-format=LEI16@16000", phrase], check=True)
                load(path)
    encoder = SpeakerEncoder(args.model)
    profile = enroll(encoder, [(load(args.audio_dir / f"Samantha-{i}.wav"), 16000)
                               for i in range(3)], threshold=.3)
    reference = np.asarray(profile.embedding)
    rows = []
    latency = []
    for voice_index, voice in enumerate(VOICES):
        for phrase in range(3, len(PHRASES)):
            original = load(args.audio_dir / f"{voice}-{phrase}.wav")
            for condition, audio in conditions(original, 20261106 + voice_index * 100 + phrase):
                # Every phrase starts with Hey Muse. Approximate the KWS trigger
                # at 1.5s; do not score the unrelated end of a long TTS sentence.
                for window, candidate in (("wake", audio[:24000]), ("segment", audio[:16000])):
                    started = time.perf_counter()
                    try:
                        score = float(np.dot(reference, encoder.embed(candidate, 16000)))
                    except ValueError:
                        score = -1.  # Insufficient speech is a rejection in the real verifier.
                    latency.append((time.perf_counter() - started) * 1000)
                    split = "calibration" if phrase in (3, 4) and voice in VOICES[:4] else "evaluation"
                    # Impostor voices used for calibration must not enter held-out evaluation.
                    if split == "evaluation" and voice in ("Daniel", "Fred", "Kathy"):
                        continue
                    if split == "evaluation" and voice == "Samantha" and phrase in (3, 4):
                        continue
                    rows.append({"split": split, "genuine": voice == "Samantha", "condition": condition,
                                 "window": window, "score": round(score, 6)})
    def scores(split, genuine):
        return [row["score"] for row in rows if row["split"] == split and row["genuine"] == genuine]
    threshold = calibrate(scores("calibration", True), scores("calibration", False))
    results = {}
    for split in ("calibration", "evaluation"):
        genuine, impostor = scores(split, True), scores(split, False)
        results[split] = {"genuine": len(genuine), "impostor": len(impostor),
                          "eer": eer(genuine, impostor), **rates(genuine, impostor, threshold)}
    report = {"model_sha256": encoder.model_id, "model_bytes": args.model.stat().st_size,
              "platform": platform.system(), "machine": platform.machine(),
              "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
                     if platform.system() == "Darwin" else platform.processor(),
              "provider": "cpu", "threads": 1, "threshold": threshold, "results": results,
              "latency_ms": {"p50": float(np.percentile(latency, 50)),
                             "p95": float(np.percentile(latency, 95)), "max": max(latency)},
              "trials": rows}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "trials"}, indent=2))


if __name__ == "__main__":
    main()
