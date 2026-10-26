"""Score recordings of the bench/voices.py sentences: how well Parakeet hears each voice.

    .venv/bin/python bench/score_voices.py MANIFEST [--recordings DIR] [--out FILE]

MANIFEST is the manifest.json bench/voices.py wrote. DIR holds one WAV per rendered file under the same
relative path (<voice>/<nn>.wav), recorded from the robot's microphone; it defaults to the manifest's own
folder, which scores the rendered files themselves. Per voice it reports:
- WER: word edits over reference words, after both sides are spelled out ("3:15 pm" is "three fifteen pm");
- critical: the share of names, numbers and negations heard intact;
- clarity: the share of spectral energy between 1 and 4 kHz, where consonants carry intelligibility.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PARAKEET = "mlx-community/parakeet-tdt-0.6b-v2"
PARAKEET_RATE = 16_000


def words(text: str) -> List[str]:
    """Lowercase words with every number, time and amount spelled out, so "$400" and "400 dollars" agree."""
    from num2words import num2words

    text = text.lower().replace("p.m.", "pm").replace("a.m.", "am").replace("n't", " not").replace("%", " percent")
    text = re.sub(r"\bdr\.", "doctor", text).replace("per cent", "percent")
    text = re.sub(r"\$(\d+)", r"\1 dollars", text)
    text = re.sub(r"\b(\d{1,2})[.:](\d{2})(?=\s*[ap]m\b)", r"\1:\2", text)  # Parakeet writes 3:15 pm as 3.15 p.m.
    text = re.sub(r"\b(\d{1,2}):00\b", r"\1", text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", r"\1 \2", text)
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)?\b", lambda m: num2words(int(m[1])), text)
    return re.findall(r"[a-z]+", text.replace("-", " "))


def word_errors(reference: Sequence[str], heard: Sequence[str]) -> int:
    row = list(range(len(heard) + 1))
    for i, ref in enumerate(reference, 1):
        previous, row[0] = row[0], i
        for j, word in enumerate(heard, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (ref != word))
    return row[-1]


def intact(text: str, phrases: Sequence[str]) -> List[bool]:
    padded = f" {' '.join(words(text))} "
    return [f" {' '.join(words(p))} " in padded for p in phrases]


def clarity(samples: np.ndarray, rate: int) -> float:
    power = np.abs(np.fft.rfft(samples)) ** 2
    hz = np.fft.rfftfreq(len(samples), 1 / rate)
    total = power.sum()
    return float(power[(hz >= 1000) & (hz <= 4000)].sum() / total) if total > 0 else 0.0


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16-bit PCM")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").astype(np.float32) / 32768
        return samples.reshape(-1, wav.getnchannels()).mean(axis=1), wav.getframerate()


def score(manifest: Dict, recordings: Path) -> Dict:
    import soxr
    from mlx.core import array
    from parakeet_mlx import from_pretrained
    from parakeet_mlx.audio import get_logmel

    parakeet = from_pretrained(PARAKEET)
    texts = {s["id"]: s for s in manifest["sentences"]}
    files = []
    for entry in manifest["files"]:
        samples, rate = read_wav(recordings / entry["file"])
        heard = parakeet.generate(get_logmel(array(soxr.resample(samples, rate, PARAKEET_RATE)), parakeet.preprocessor_config))[0].text.strip()
        sentence = texts[entry["sentence"]]
        files.append({"file": entry["file"], "voice": entry["voice"], "heard": heard, "words": len(words(sentence["text"])),
                      "errors": word_errors(words(sentence["text"]), words(heard)),
                      "critical": intact(heard, sentence["critical"]), "clarity": round(clarity(samples, rate), 4)})
        print(f"{entry['file']:<18} {files[-1]['errors']} errors  {heard}", file=sys.stderr)
    return {"recordings": recordings.name, "voices": [summarize(voice, [f for f in files if f["voice"] == voice]) for voice in manifest["voices"]],
            "files": files}


def summarize(voice: str, files: Sequence[Dict]) -> Dict:
    critical = [ok for f in files for ok in f["critical"]]
    return {"voice": voice, "wer": round(sum(f["errors"] for f in files) / max(1, sum(f["words"] for f in files)), 4),
            "critical": f"{sum(critical)}/{len(critical)}", "clarity": round(float(np.mean([f["clarity"] for f in files])), 4)}


def table(voices: Sequence[Dict]) -> str:
    rows = ["| Voice | WER | Critical intact | Clarity (1-4 kHz share) |", "|---|---|---|---|"]
    rows += [f"| {v['voice']} | {v['wer']:.1%} | {v['critical']} | {v['clarity']:.3f} |" for v in voices]
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="the manifest.json bench/voices.py wrote")
    parser.add_argument("--recordings", type=Path, help="the folder of robot recordings (default: the manifest's folder)")
    parser.add_argument("--out", type=Path, help="write every transcript and score as JSON (default: scores.json in the recordings folder)")
    args = parser.parse_args()
    recordings = args.recordings or args.manifest.parent
    manifest = json.loads(args.manifest.read_text())
    missing = [f["file"] for f in manifest["files"] if not (recordings / f["file"]).exists()]
    if missing:
        parser.error(f"{len(missing)} recordings are missing from {recordings}, such as {missing[0]}")
    result = score(manifest, recordings)
    (args.out or recordings / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print(table(result["voices"]))


if __name__ == "__main__":
    main()
