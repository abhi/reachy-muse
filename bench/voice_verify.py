"""Offline CPU speaker evaluation. Synthetic WAVs stay in the supplied directory.

Run with --synthesize once on macOS; the saved WAVs can then be evaluated on
another CPU without TTS. No network, robot, companion, or Muse connections.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import wave

import numpy as np

from reachy_muse.voice_verify import RequestGate, SpeakerEncoder, VoiceVerifier, enroll

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


class MeasuredVerifier(VoiceVerifier):
    def __init__(self, encoder, profile):
        super().__init__(encoder, profile, "strict")
        self.scores = []
        self.latency = []

    def verify(self, audio, rate):
        started = time.perf_counter()
        decision = super().verify(audio, rate)
        self.latency.append((time.perf_counter() - started) * 1000)
        self.scores.append(round(decision.score, 6) if decision.score is not None else -1.)
        return decision


def evaluate(encoder, profile, audio, window):
    verifier = MeasuredVerifier(encoder, profile)
    retention = {}
    if window == "wake":
        verifier.verify(audio[:24000], 16000)
    else:
        gate = RequestGate(verifier, 16000)
        padded = np.pad(audio, (0, 16000 + (-len(audio) % 320)))
        admitted = np.concatenate([gate.feed(padded[start:start + 320])
                                   for start in range(0, len(padded), 320)])
        incoming = int(np.count_nonzero(audio))
        retained = int(np.count_nonzero(admitted))
        retention = {"input_nonzero": incoming, "retained_nonzero": retained,
                     "complete": incoming > 0 and incoming == retained}
    return {"score": min(verifier.scores, default=-1.), "segment_scores": verifier.scores,
            **retention}, verifier.latency


def summarize(rows, threshold):
    genuine = [row for row in rows if row["genuine"]]
    impostor = [row for row in rows if not row["genuine"]]
    positive, negative = [row["score"] for row in genuine], [row["score"] for row in impostor]
    result = {"genuine": len(genuine), "impostor": len(impostor),
              "eer": eer(positive, negative), **rates(positive, negative, threshold)}
    if rows[0]["window"] == "request":
        result.update(genuine_complete=sum(row["complete"] for row in genuine),
                      complete_request_retention=sum(row["complete"] for row in genuine) / len(genuine),
                      impostor_with_audio=sum(row["retained_nonzero"] > 0 for row in impostor))
    return result


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
                utterances = [(f"{voice}-{index}.wav", phrase)]
                if index > 3:
                    utterances.append((f"{voice}-{index}-request.wav", phrase.partition(", ")[2]))
                for filename, text in utterances:
                    path = args.audio_dir / filename
                    subprocess.run(["say", "-v", voice, "-r", "170", "-o", str(path),
                                    "--file-format=WAVE", "--data-format=LEI16@16000", text], check=True)
                    load(path)
    encoder = SpeakerEncoder(args.model)
    profile = enroll(encoder, [(load(args.audio_dir / f"Samantha-{i}.wav"), 16000)
                               for i in range(3)], threshold=.3)
    trials = []
    for voice_index, voice in enumerate(VOICES):
        for phrase in range(3, len(PHRASES)):
            split = "calibration" if phrase in (3, 4) and voice in VOICES[:4] else "evaluation"
            if split == "evaluation" and (voice in ("Daniel", "Fred", "Kathy")
                                           or voice == "Samantha" and phrase in (3, 4)):
                continue
            for window in (("wake",) if phrase == 3 else ("wake", "request")):
                suffix = "-request" if window == "request" else ""
                original = load(args.audio_dir / f"{voice}-{phrase}{suffix}.wav")
                for condition, audio in conditions(original, 20261106 + voice_index * 100 + phrase):
                    trials.append(({"split": split, "genuine": voice == "Samantha", "voice": voice,
                                    "phrase": phrase, "condition": condition, "window": window}, audio))
    calibration = {True: [], False: []}
    for trial, audio in trials:
        if trial["split"] == "calibration":
            measured, _ = evaluate(encoder, profile, audio, trial["window"])
            calibration[trial["genuine"]].extend(measured["segment_scores"])
    threshold = calibrate(calibration[True], calibration[False])
    profile = replace(profile, threshold=threshold)
    rows, latency = [], []
    for trial, audio in trials:
        measured, elapsed = evaluate(encoder, profile, audio, trial["window"])
        rows.append({**trial, **measured})
        latency.extend(elapsed)
    results = {}
    for split in ("calibration", "evaluation"):
        results[split] = {window: summarize([row for row in rows if row["split"] == split
                                             and row["window"] == window], threshold)
                          for window in ("wake", "request")}
        results[split]["conditions"] = {
            condition: {window: summarize([row for row in rows if row["split"] == split
                                            and row["window"] == window and row["condition"] == condition], threshold)
                        for window in ("wake", "request")}
            for condition in sorted({row["condition"] for row in rows})}
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
