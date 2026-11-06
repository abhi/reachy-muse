"""CPU-only wake corpus generation, tuning, and production-policy replay.

Generate into bench/out; commit the manifest/hashes and numeric report, not audio.
All model paths are command arguments. No robot or companion connection is made.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
import time
import types

import numpy as np

from reachy_muse.wake_policy import RATE, WakeContext, WakePolicy, WakeVerifier, pcm_wav
from reachy_muse.wake_word import MODEL_FILES, WakeWordDetector


def digest(path):
    with Path(path).open("rb") as source:
        checksum = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
        return checksum.hexdigest()


def read_audio(path):
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != RATE:
        audio = np.interp(np.arange(round(len(audio) * RATE / rate)) * rate / RATE,
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


def build(args):
    from kokoro_onnx import Kokoro
    import onnxruntime as ort

    args.output.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.tts_model), providers=["CPUExecutionProvider"], sess_options=options)
    assert session.get_providers() == ["CPUExecutionProvider"]
    tts = Kokoro.from_session(session, str(args.voices))
    rng = np.random.default_rng(20261106)
    manifest = []
    ordinary = []

    def save(audio, *, positive, kind, split, group, snr=None, playback=(), phrase_end=None):
        filename = f"clip-{len(manifest):04d}.wav"
        path = args.output / filename
        pcm = np.rint(np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        path.write_bytes(pcm_wav(pcm))
        manifest.append({"path": filename, "sha256": digest(path), "positive": positive,
                         "kind": kind, "split": split, "group": group, "snr_db": snr,
                         "playback": playback, "phrase_end_s": phrase_end,
                         "audio_s": len(audio) / RATE})

    def render(text, voice, speed):
        audio, rate = tts.create(text, voice=voice, speed=speed,
                                 lang="en-gb" if voice.startswith("b") else "en-us")
        audio = np.interp(np.arange(round(len(audio) * RATE / rate)) * rate / RATE,
                          np.arange(len(audio)), audio).astype(np.float32)
        return audio * (.08 / max(.0001, float(np.sqrt(np.mean(audio ** 2)))))

    for voice in ("af_heart", "am_michael", "bf_emma", "bm_george", "af_sky", "bm_lewis"):
        split = ("tune" if voice in ("af_heart", "am_michael") else
                 "development" if voice in ("bf_emma", "bm_george") else "holdout")
        for speed in (.85, 1.15):
            for phrase in ("Hey Muse.", "Hey, Muse!"):
                audio = render(phrase, voice, speed)
                group = f"{voice}-{speed}-{len(manifest)}"
                for snr in (None, 20, 10, 0):
                    prefix = .24 + float(rng.choice((0, .08, .16, .24)))
                    clip = np.pad(audio, (round(prefix * RATE), int(.8 * RATE)))
                    if snr is not None:
                        noise = rng.normal(size=len(clip))
                        # A reproducible noise proxy, not a recorded robot motor.
                        noise += .3 * np.sin(np.arange(len(clip)) * 2 * np.pi * 120 / RATE)
                        noise *= .08 / (10 ** (snr / 20) * np.sqrt(np.mean(noise ** 2)))
                        clip = clip + noise
                    save(clip, positive=True, kind="synthetic_positive", split=split,
                         group=group, snr=snr, phrase_end=prefix + len(audio) / RATE)
        for index, phrase in enumerate((
            "Hey music, turn it up.", "Hey news, what happened today?", "Hey moose, come here.",
            "The museum opens tomorrow.", "Hi Muse.", "The forecast is sunny with some clouds.",
            "Our reporter has the latest results from the station.",
        )):
            audio = render(phrase, voice, 1.0)
            ordinary.append(audio)
            for snr in (None, 10, 0):
                clip = np.pad(audio, (int(.32 * RATE), int(.8 * RATE)))
                if snr is not None:
                    clip = clip + rng.normal(0, .08 / 10 ** (snr / 20), len(clip))
                save(clip, positive=False, kind="confusable" if index < 5 else "ordinary_speech",
                     split=split, group=f"{voice}-negative-{index}", snr=snr)
        for phrase in ("To speak to me, say Hey Muse.", "Hey Muse."):
            audio = render(phrase, voice, 1.0)
            for gain in (.15, .5, 1):
                save(np.pad(audio * gain, (int(.32 * RATE), int(.8 * RATE))), positive=False,
                     kind="own_voice", split=split, group=f"{voice}-output-{phrase}", playback=(phrase,))
        print(f"Generated {voice}: {len(manifest)} clips", flush=True)
    for index in range(8):
        left, right = ordinary[index], ordinary[-index - 1]
        length = max(len(left), len(right))
        mixed = np.pad(left, (0, length - len(left))) + .45 * np.pad(right, (0, length - len(right)))
        save(np.pad(mixed, (int(.32 * RATE), int(.8 * RATE))), positive=False, kind="tv_like",
             split="holdout", group=f"tv-{index}")
    for index in range(4):
        save(rng.normal(0, index * .015, 10 * RATE), positive=False, kind="silence_noise",
             split="holdout", group=f"noise-{index}")
    for filename in ("correction.wav", "negation.wav", "reminder.wav"):
        source = Path("tests/companion_server/data") / filename
        if source.exists():
            save(read_audio(source), positive=False, kind="recorded_negative", split="holdout", group=filename)
    if args.recordings:
        recorded = json.loads((args.recordings / "manifest.json").read_text())
        for index, case in enumerate(recorded):
            path = Path(case["path"])
            if path.is_absolute() or ".." in path.parts or type(case["positive"]) is not bool:
                raise ValueError("Recorded manifests need relative paths and boolean labels")
            if not case.get("valid", True):
                continue
            save(read_audio(args.recordings / path), positive=case["positive"], kind="operator",
                 split="holdout", group=f"operator-{index}")
    # An actual continuous stream test, not many resets pretending to be idle.
    phrase = render("Hey Muse.", "bf_emma", 1.0)
    save(np.concatenate((np.zeros(120 * RATE), phrase, np.zeros(RATE))), positive=True,
         kind="long_idle", split="holdout", group="long-idle", phrase_end=120 + len(phrase) / RATE)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "provenance.json").write_text(json.dumps({
        "generator": "Kokoro ONNX, CPUExecutionProvider, two threads", "seed": 20261106,
        "tts_sha256": digest(args.tts_model), "voices_sha256": digest(args.voices),
        "limitations": "Generated voices and noise proxies; no recorded human positives or physical robot echo.",
    }, indent=2) + "\n")


def metrics(rows):
    positives = [row for row in rows if row["positive"]]
    negatives = [row for row in rows if not row["positive"]]
    exposure = sum(row["audio_s"] for row in negatives)
    false = sum(row["triggers"] for row in negatives)
    latency = [row["latency_ms"] for row in positives if row["latency_ms"] is not None]
    return {"positives": len(positives), "hits": sum(row["triggers"] > 0 for row in positives),
            "recall": sum(row["triggers"] > 0 for row in positives) / len(positives) if positives else None,
            "negative_audio_s": round(exposure, 3), "false_accepts": false,
            "false_accepts_per_hour": round(false * 3600 / exposure, 3) if exposure else None,
            "latency_ms_p50": round(float(np.median(latency)), 2) if latency else None,
            "latency_ms_p95": round(float(np.percentile(latency, 95)), 2) if latency else None}


def compact_report(result):
    def compact(run):
        result = {key: value for key, value in run.items() if key != "rows"}
        events = [event for row in run["rows"] for event in row["candidates"]]
        result["decisions"] = dict(Counter(event["reason"] for event in events))
        result["verification"] = dict(Counter(f'{event["verifier"]}:{event["verification"]}' for event in events))
        result["audio_s"] = round(sum(row["audio_s"] for row in run["rows"]), 3)
        result["alignment_available_accepts"] = sum(event["alignment_available"] for event in events)
        return result
    return {key: [{name: item for name, item in compact(run).items()
                   if name not in ("by_kind", "by_snr", "by_split")} for run in value] if key == "sweep" else
            compact(value) if key in ("baseline", "after", "companion_down_local_down") else value
            for key, value in result.items()}


def probe(args):
    """Compare raw KWS alternatives without changing the production policy."""
    import sentencepiece

    manifest = json.loads((args.corpus / "manifest.json").read_text())
    detector = WakeWordDetector(args.model, threshold=.1, candidate_replay=True)
    spotter = detector._spotter
    runs = []
    try:
        for count in (1, 2, 4):
            hits = positives = false = negatives = 0
            duration = 0.0
            started = time.process_time()
            for case in manifest:
                if case["split"] != "tune" and case["kind"] != "long_idle":
                    continue
                audio = read_audio(args.corpus / case["path"])
                duration += len(audio) / RATE
                streams = [detector._new_stream() for _ in range(count)]
                for index, stream in enumerate(streams):
                    if index:
                        stream.accept_waveform(RATE, np.zeros(index * 5120 // count, np.float32))
                found = False
                for offset in range(0, len(audio), 256):
                    for stream in streams:
                        stream.accept_waveform(RATE, audio[offset:offset + 256])
                    while True:
                        ready = [stream for stream in streams if spotter.is_ready(stream)]
                        if not ready:
                            break
                        spotter.decode_streams(ready)
                        for stream in ready:
                            found = bool(detector._result(stream)[0]) or found
                if case["positive"]:
                    positives += 1
                    hits += found
                else:
                    negatives += 1
                    false += found
            runs.append({"streams": count, "hits": hits, "positives": positives,
                         "false_clips": false, "negative_clips": negatives, "audio_s": round(duration, 3),
                         "host_cpu_s": round(time.process_time() - started, 3)})
        case = next(case for case in manifest if case["kind"] == "long_idle")
        phrase = read_audio(args.corpus / case["path"])[120 * RATE:]
        offsets = []
        for prefix in (0, .08, .16, .24, .32, .56, .88, 1.20, 2, 10):
            detector.reset()
            audio = np.concatenate((np.zeros(round(prefix * RATE), np.float32), phrase))
            hits = 0
            for offset in range(0, len(audio), 256):
                hits += detector.feed(audio[offset:offset + 256], align=False)
            offsets.append({"prefix_s": prefix, "triggers": hits})
        aliases = []
        tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.model / "bpe.model"))
        canonical = detector._keyword_tokens
        for alias in ("HEY MEWS", "HAY MUSE"):
            detector._keyword_tokens = canonical + (" ".join(tokenizer.encode(alias, out_type=str)),)
            hits = false = 0
            for case in manifest:
                if case["split"] != "tune":
                    continue
                detector.reset()
                audio = read_audio(args.corpus / case["path"])
                found = False
                for offset in range(0, len(audio), 256):
                    found = detector.feed(audio[offset:offset + 256], align=False) or found
                if case["positive"]:
                    hits += found
                else:
                    false += found
            aliases.append({"alias": alias, "hits": hits, "false_clips": false})
        result = {"threshold": .1, "provider": "cpu", "streams": runs, "same_waveform_offsets": offsets,
                  "keyword_aliases": aliases,
                  "note": "Raw candidate observations, not verified wake accepts. No batching wait is introduced."}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        detector.close()


async def replay(args):
    from reachy_muse.streaming_transcription import SherpaStreamingTranscriber

    manifest = json.loads((args.corpus / "manifest.json").read_text())
    baseline_source = subprocess.check_output(["git", "show", "dd1ed17:src/reachy_muse/wake_word.py"], text=True)
    baseline_module = types.ModuleType("wake_baseline_dd1ed17")
    sys.modules[baseline_module.__name__] = baseline_module
    exec(compile(baseline_source, "wake_baseline_dd1ed17.py", "exec"), baseline_module.__dict__)
    for case in manifest:
        path = Path(case["path"])
        if path.is_absolute() or ".." in path.parts or digest(args.corpus / path) != case["sha256"]:
            raise ValueError("Invalid corpus path or audio hash")
    recognizer = SherpaStreamingTranscriber(args.verify_model) if args.verify_model else None
    if recognizer:
        await recognizer.start()

    async def evaluate(threshold, boost, *, baseline=False, split=None, offline=False, variants=1, window=1.5):
        detector = (baseline_module.WakeWordDetector(args.model) if baseline else
                    WakeWordDetector(args.model, threshold=threshold, score=boost, candidate_replay=True,
                                     token_variants=variants))
        policy = WakePolicy(WakeVerifier(local=None if offline else recognizer), verification_window_s=window)
        rows = []
        cpu = time.process_time()
        wall = time.perf_counter()
        try:
            for case in manifest:
                if split and case["split"] != split:
                    continue
                audio = read_audio(args.corpus / case["path"])
                detector.reset()
                policy.reset(new_trial=True)
                events, triggers, first = [], 0, None
                context = WakeContext(speaking=bool(case["playback"]), playback=tuple(case["playback"]))
                for offset in range(0, len(audio), 256):
                    chunk = audio[offset:offset + 256]
                    started = time.perf_counter()
                    found = detector.feed(chunk) if baseline else await policy.feed(detector, chunk, context=context)
                    elapsed = time.perf_counter() - started
                    if found:
                        triggers += 1
                        if first is None:
                            first = (offset + len(chunk)) / RATE + elapsed
                    if not baseline and policy.last_decision is not None:
                        events.append(asdict(policy.last_decision))
                end = case["phrase_end_s"]
                rows.append({key: case[key] for key in ("path", "positive", "kind", "split", "audio_s", "snr_db")} |
                            {"triggers": triggers, "candidates": events,
                             "latency_ms": round((first - end) * 1000, 2) if first is not None and end is not None else None})
            return {"threshold": threshold, "boost": boost, "variants": variants,
                    "verification_window_s": window,
                    "baseline": baseline, "offline": offline,
                    "summary": metrics(rows),
                    "by_split": {name: metrics([row for row in rows if row["split"] == name]) for name in ("tune", "development", "holdout")},
                    "by_kind": {name: metrics([row for row in rows if row["kind"] == name]) for name in sorted({r["kind"] for r in rows})},
                    "by_snr": {str(snr): metrics([row for row in rows if row["kind"] == "synthetic_positive" and row["snr_db"] == snr])
                               for snr in (None, 20, 10, 0)},
                    "host_parent_cpu_s": round(time.process_time() - cpu, 3),
                    "host_wall_s": round(time.perf_counter() - wall, 3), "rows": rows}
        finally:
            detector.close()

    try:
        baseline = await evaluate(.25, 1, baseline=True)
        print("Baseline", baseline["summary"], flush=True)
        sweep = []
        for threshold in (.05, .1, .15, .25):
            for boost in (1.0, 1.5):
                result = await evaluate(threshold, boost, split="tune")
                sweep.append(result)
                print("Tune", threshold, boost, result["summary"], flush=True)
        for threshold in (.05, .1, .25):
            result = await evaluate(threshold, 1.0, variants=3, split="tune")
            sweep.append(result)
            print("Token variants", threshold, result["summary"], flush=True)
        for threshold, boost in ((.05, 3.0), (.1, 3.0), (.1, 5.0)):
            result = await evaluate(threshold, boost, split="tune")
            sweep.append(result)
            print("High boost", threshold, boost, result["summary"], flush=True)
        result = await evaluate(.1, 1.0, split="tune", window=2.0)
        sweep.append(result)
        print("Two-second window", result["summary"], flush=True)
        selected = max(sweep, key=lambda result: (-result["summary"]["false_accepts"], result["summary"]["hits"],
                                                 -result["variants"], -result["boost"], result["threshold"],
                                                 -result["verification_window_s"]))
        after = await evaluate(selected["threshold"], selected["boost"], variants=selected["variants"],
                               window=selected["verification_window_s"])
        offline = await evaluate(selected["threshold"], selected["boost"], variants=selected["variants"], offline=True,
                                 window=selected["verification_window_s"])
        result = {"platform": platform.platform(), "provider": "cpu", "kws_threads": 1,
                  "versions": {name: version(name) for name in ("sherpa-onnx", "sentencepiece", "kokoro-onnx", "soundfile", "numpy")},
                  "baseline_ref": "dd1ed17", "baseline_source_sha256": hashlib.sha256(baseline_source.encode()).hexdigest(),
                  "model_sha256": {key: digest(args.model / name) for key, name in MODEL_FILES.items()},
                  "baseline": baseline, "sweep": sweep, "after": after, "companion_down_local_down": offline,
                  "latency_note": "Audio-time detection lag plus final call wall time; synthetic phrase end is rendered audio end, not phonetic annotation. Unpaced replay.",
                  "cpu_note": "Parent process includes KWS/replay, excludes local recognizer child CPU. Not a CM4 measurement."}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if args.summary:
            args.summary.write_text(json.dumps(compact_report(result), indent=2) + "\n")
        print("Selected", selected["threshold"], selected["boost"], after["summary"], flush=True)
        print("Standalone", offline["summary"], flush=True)
    finally:
        if recognizer:
            await recognizer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("build")
    generate.add_argument("--tts-model", type=Path, required=True)
    generate.add_argument("--voices", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--recordings", type=Path, help="Optional labelled wake-test recording directory")
    run = sub.add_parser("replay")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--verify-model", type=Path)
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--summary", type=Path, help="Compact report suitable for source control")
    probes = sub.add_parser("probe")
    probes.add_argument("--model", type=Path, required=True)
    probes.add_argument("--corpus", type=Path, required=True)
    probes.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.command == "build":
        build(args)
    elif args.command == "probe":
        probe(args)
    else:
        asyncio.run(replay(args))


if __name__ == "__main__":
    main()
