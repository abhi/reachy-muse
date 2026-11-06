"""Operator wake trials. Audio and recognition stay local unless explicitly saved."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

from reachy_muse.wake_policy import RATE, WakePolicy, WakeVerifier, pcm_wav


def positive_int(value):
    number = int(value)
    if not 1 <= number <= 1000:
        raise argparse.ArgumentTypeError("count must be between 1 and 1000")
    return number


def add_arguments(parser):
    parser.add_argument("--count", type=positive_int, default=10)
    parser.add_argument("--wake-model", type=Path)
    parser.add_argument("--verify-model", type=Path, help="Optional local streaming STT model")
    parser.add_argument("--seconds", type=float, default=5, help="Observed seconds per trial (default: 5)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--media-backend", choices=("local", "webrtc", "default"), default="local")
    parser.add_argument("--candidate-threshold", type=float, default=.1)
    parser.add_argument("--threshold", type=float, default=.25)
    parser.add_argument("--score", type=float, default=1.0, help="Keyword boost, not confidence")
    parser.add_argument("--report", type=Path, help="Save content-free trial measurements as JSON")
    parser.add_argument("--record-dir", type=Path, help="Opt in to saving labelled microphone WAVs for replay")


@dataclass(frozen=True)
class Trial:
    positive: bool
    audio_s: float
    triggers: int
    valid: bool
    candidates: tuple[dict, ...]


def summarize(trials: list[Trial]) -> dict:
    positives = [trial for trial in trials if trial.positive and trial.valid]
    negatives = [trial for trial in trials if not trial.positive and trial.valid]
    exposure = sum(trial.audio_s for trial in negatives)
    false_accepts = sum(trial.triggers for trial in negatives)
    return {
        "positive_trials": len(positives),
        "positive_hits": sum(trial.triggers > 0 for trial in positives),
        "recall": sum(trial.triggers > 0 for trial in positives) / len(positives) if positives else None,
        "negative_audio_s": exposure,
        "false_triggers": false_accepts,
        "false_triggers_per_hour": false_accepts * 3600 / exposure if exposure else None,
        "zero_event_95pct_upper_per_hour": -math.log(.05) * 3600 / exposure if exposure and not false_accepts else None,
        "invalid_trials": sum(not trial.valid for trial in trials),
        "score_note": "Native posterior unavailable; threshold is a trigger floor, replay_floor is a replay bound.",
        "trials": [asdict(trial) for trial in trials],
    }


async def measure(detector, policy, chunks, *, positive: bool, recorded=None) -> Trial:
    """Count audio exposure, never prompt time or missing microphone samples."""
    await asyncio.to_thread(detector.reset)
    policy.reset(new_trial=True)
    samples = triggers = 0
    valid = True
    candidates = []
    async for audio, gap in chunks:
        if gap:
            valid = False
            await asyncio.to_thread(detector.reset)
            policy.reset()
        samples += len(audio)
        if recorded is not None:
            recorded.append(audio.copy())
        triggers += int(await policy.feed(detector, audio))
        if policy.last_decision is not None:
            candidates.append(asdict(policy.last_decision))
    return Trial(positive, samples / RATE, triggers, valid and samples > 0, tuple(candidates))


async def microphone_chunks(hardware, duration):
    """Drain audio independently of inference; mark overflow and read gaps."""
    import av
    import numpy as np

    queue = asyncio.Queue(maxsize=64)
    overflow = False
    stop_at = time.monotonic() + duration
    resampler = av.AudioResampler(format="flt", layout="mono", rate=RATE)

    async def produce():
        nonlocal overflow
        last = time.monotonic()
        try:
            while time.monotonic() < stop_at:
                read = asyncio.create_task(asyncio.to_thread(hardware.read_audio))
                try:
                    audio = await asyncio.shield(read)
                except asyncio.CancelledError:
                    await read
                    raise
                if audio is None or not len(audio):
                    await asyncio.sleep(.005)
                    continue
                now = time.monotonic()
                gap = now - last > max(.5, len(audio) / hardware.sample_rate * 3)
                last = now
                frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(audio[None, :]), format="flt", layout="mono")
                frame.sample_rate = hardware.sample_rate
                for converted in resampler.resample(frame):
                    if queue.full():
                        queue.get_nowait()
                        overflow = True
                    queue.put_nowait((converted.to_ndarray().reshape(-1), gap or overflow))
                    overflow = False
        finally:
            await queue.put(None)

    producer = asyncio.create_task(produce())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
        await producer
    finally:
        producer.cancel()
        # Drain so a producer in its final sentinel write can exit.
        while not queue.empty():
            queue.get_nowait()
        await asyncio.gather(producer, return_exceptions=True)


async def prompt_ready(hardware, prompt):
    """Keep draining the SDK while the operator positions themselves."""
    finished = False

    async def discard():
        while not finished:
            await asyncio.to_thread(hardware.read_audio)
            await asyncio.sleep(.002)

    draining = asyncio.create_task(discard())
    try:
        await asyncio.to_thread(input, prompt)
    finally:
        finished = True
        await draining
    print("Listening now.", flush=True)


async def run(args) -> int:
    import numpy as np
    from reachy_muse.hardware import ReachyController
    from reachy_muse.wake_word import WakeWordDetector

    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 300:
        raise ValueError("Trial duration must be between 1 and 300 seconds")
    if not 0 < args.candidate_threshold <= args.threshold <= 1:
        raise ValueError("Need 0 < candidate threshold <= threshold <= 1")
    model = args.wake_model or args.state_dir / "wake-hey-muse"
    detector = WakeWordDetector(model, threshold=args.candidate_threshold, score=args.score, candidate_replay=True)
    recognizer = None
    hardware = None
    try:
        if args.verify_model:
            from reachy_muse.streaming_transcription import SherpaStreamingTranscriber
            recognizer = SherpaStreamingTranscriber(args.verify_model)
            await recognizer.start()
        policy = WakePolicy(WakeVerifier(local=recognizer), fallback_threshold=args.threshold)
        hardware = ReachyController(host=args.host, port=args.port, media_backend=args.media_backend,
                                 enable_motion=False, sleep_on_close=False)
        await asyncio.to_thread(hardware.start)
        trials, manifest = [], []
        if args.record_dir:
            args.record_dir.mkdir(parents=True, exist_ok=True)
        print("Local wake test. The robot stays still. Each trial starts after Enter.")
        for positive in (True, False):
            for index in range(args.count):
                prompt = (f'Say "hey muse" once at {("0.5 m", "1.5 m", "3 m")[index % 3]}.' if positive else
                          ("Say a sentence without the wake phrase.", "Stay silent.",
                           'Say a confusable phrase such as "hey music".')[index % 3])
                await prompt_ready(hardware, f"{index + 1}/{args.count}: {prompt} Press Enter when ready. ")
                recording = [] if args.record_dir else None
                trial = await measure(detector, policy, microphone_chunks(hardware, args.seconds),
                                      positive=positive, recorded=recording)
                # Missing trailing audio also invalidates the trial.
                if trial.audio_s < args.seconds * .8:
                    trial = Trial(trial.positive, trial.audio_s, trial.triggers, False, trial.candidates)
                trials.append(trial)
                print(json.dumps(asdict(trial), sort_keys=True))
                if recording:
                    filename = f'{"positive" if positive else "negative"}-{index + 1:03d}.wav'
                    pcm = np.rint(np.concatenate(recording) * 32767).astype("<i2").tobytes()
                    (args.record_dir / filename).write_bytes(pcm_wav(pcm))
                    manifest.append({"path": filename, "positive": positive, "kind": "operator",
                                     "split": "holdout", "valid": trial.valid})
        report = summarize(trials)
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.report:
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if args.record_dir:
            (args.record_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return 0 if not report["invalid_trials"] else 1
    finally:
        if hardware is not None:
            await asyncio.to_thread(hardware.close)
        if recognizer is not None:
            await recognizer.close()
        await asyncio.to_thread(detector.close)
