"""Render the same test sentences in several Kokoro voices, for playing and recording on the robot.

    .venv/bin/python bench/voices.py [--out DIR] [--voice NAME ...] [--sentence TEXT ...] [--list]

Every sentence is spoken as the companion speaks it: Kokoro at 24 kHz, resampled to 48 kHz, then
raised to the serve loudness. The folder gets <voice>/<nn>.wav and a manifest.json that names each
file with its text, the words that must survive (critical), and its level before and after the gain.
Record each file from the robot's microphone under the same relative path, then run bench/score_voices.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from reachy_muse.companion_server.voice_kokoro import REPO, Loudness, load_model, pcm16, render

RATE = 48_000
HERE = Path(__file__).resolve().parent
CANDIDATES = ("af_heart", "af_bella", "af_nicole", "am_michael", "bf_emma", "bm_george")
SENTENCES = (
    {"text": "Your appointment with Doctor Patel is on Tuesday at 3:15 pm.", "critical": ["Patel", "Tuesday", "3:15 pm"]},
    {"text": "I did not find any flights to Lisbon under 400 dollars.", "critical": ["did not", "Lisbon", "400 dollars"]},
    {"text": "Maya called twice and left a message about the 12 tickets.", "critical": ["Maya", "twice", "12 tickets"]},
    {"text": "The oven is not on, so dinner will take another 25 minutes.", "critical": ["not on", "25 minutes"]},
    {"text": "Remind Rohan to water the plants on Friday morning.", "critical": ["Rohan", "Friday morning"]},
    {"text": "It is 18 degrees in Oslo with a 70 percent chance of rain.", "critical": ["18 degrees", "Oslo", "70 percent"]},
)


def installed_voices() -> List[str]:
    from huggingface_hub import try_to_load_from_cache

    config = try_to_load_from_cache(REPO, "config.json")
    if not isinstance(config, str):
        raise SystemExit(f"{REPO} is not in the Hugging Face cache; run serve once to download it")
    return sorted(path.stem for path in (Path(config).parent / "voices").glob("*.safetensors"))


def level(audio: np.ndarray) -> Dict[str, float]:
    samples = np.frombuffer(pcm16(audio), "<i2").astype(np.float64) / 32767
    return {"dbfs": round(20 * np.log10(np.sqrt(np.mean(samples ** 2)) + 1e-12), 2), "peak": round(float(np.abs(samples).max()), 3)}


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm)


def render_all(out: Path, voices: Sequence[str], sentences: Sequence[Dict], loudness: Loudness) -> Dict:
    model = load_model(voices[0])
    files = []
    for voice in voices:
        for index, sentence in enumerate(sentences, 1):
            raw = render(model, sentence["text"], voice, RATE)
            loud = loudness.apply(raw)
            name = f"{voice}/{index:02d}.wav"
            write_wav(out / name, pcm16(loud))
            files.append({"file": name, "voice": voice, "sentence": index, "seconds": round(len(raw) / RATE, 2),
                          "raw": level(raw), "normalized": level(loud)})
            print(f"{name:<18} raw {files[-1]['raw']['dbfs']:6.2f} dBFS peak {files[-1]['raw']['peak']:.3f}"
                  f"  ->  {files[-1]['normalized']['dbfs']:6.2f} dBFS peak {files[-1]['normalized']['peak']:.3f}", file=sys.stderr)
    manifest = {"rate": RATE, "repo": REPO, "loudness": {"target_dbfs": loudness.target_dbfs, "ceiling_dbfs": loudness.ceiling_dbfs},
                "voices": list(voices), "sentences": [{"id": i, **s} for i, s in enumerate(sentences, 1)], "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "out" / "voices")
    parser.add_argument("--voice", action="append", help=f"a Kokoro voice to render; repeat for more (default: {', '.join(CANDIDATES)})")
    parser.add_argument("--sentence", action="append", help="a sentence to render instead of the six test sentences; repeat for more")
    parser.add_argument("--loudness", type=float, default=Loudness().target_dbfs, metavar="DBFS")
    parser.add_argument("--list", action="store_true", help="print the voices this Kokoro has and exit")
    args = parser.parse_args()
    available = installed_voices()
    if args.list:
        print(" ".join(available))
        return
    voices = args.voice or list(CANDIDATES)
    unknown = sorted(set(voices) - set(available))
    if unknown:
        parser.error(f"no such voice: {', '.join(unknown)}; --list prints the {len(available)} there are")
    sentences = [{"text": text, "critical": []} for text in args.sentence] if args.sentence else list(SENTENCES)
    manifest = render_all(args.out, voices, sentences, Loudness(target_dbfs=args.loudness))
    print(f"{len(manifest['files'])} files and manifest.json in {args.out}")


if __name__ == "__main__":
    main()
