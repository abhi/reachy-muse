"""Measure hearing and speech latency over the companion's WebSocket, against the C2 and C3 targets.

    .venv/bin/python bench/latency.py [--corpus DIR] [--runs N] [--no-final-pass] [--delay-ms MS]

It pairs into a temporary state directory, starts `reachy-companion serve --insecure`,
and acts as the robot. Every WAV in the corpus manifest is sent twice in real time:
- forced: speech plus 0.2 s of silence, then hear_close{finish}. Measures end of audio to hear_endpoint.
- self: the whole clip, trailing silence included. Measures end of speech to the companion's own hear_endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from websockets.asyncio.client import ClientConnection, connect

from musegadget.reachy_companion_protocol import (
    AudioKind,
    CloseHow,
    HearClose,
    HearEndpoint,
    HearOpen,
    HearPartial,
    Hello,
    Op,
    Sender,
    SpeakDone,
    SpeakStart,
    VoiceRole,
    Welcome,
    decode,
    decode_audio,
    encode,
    encode_audio,
)

RATE = 16_000
FRAME = 320  # 20 ms of microphone audio
HERE = Path(__file__).resolve().parent
SENTENCES = (
    "Sure, I can help with that.",
    "Your reminder is set for eight tonight, and I will call Maya at nine.",
    "The weather in San Jose tomorrow morning looks cool and cloudy, so bring a light jacket.",
    "I added seven apples, two cartons of oat milk, and one bag of rice to your grocery list.",
    "Before you leave, remember that the violet notebook is on the kitchen table, next to the keys.",
)
TARGETS = {"final_p50_s": 0.300, "partial_gap_s": 0.600, "first_audio_p50_s": 0.250}


@dataclass
class Heard:
    text: str
    forced: bool
    endpoint_s: float  # from the first frame sent
    audio_end_s: float  # when the last frame was due
    partial_times_s: List[float] = field(default_factory=list)


@dataclass
class Spoken:
    first_frame_s: Optional[float]
    frames: int
    done_frames: int


def read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, RATE), path
        return wav.readframes(wav.getnframes())


def speech_end_s(pcm: bytes) -> float:
    """The end of the last 20 ms frame louder than -34 dBFS."""
    import numpy as np

    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    frames = samples[: len(samples) // FRAME * FRAME].reshape(-1, FRAME)
    loud = np.flatnonzero(np.sqrt((frames**2).mean(axis=1)) > 0.02)
    return (int(loud[-1]) + 1) * FRAME / RATE if loud.size else 0.0


def words(text: str) -> List[str]:
    text = text.lower().replace("don't", "do not")
    numbers = {"1": "one", "2": "two", "7": "seven", "8": "eight", "9": "nine"}
    text = re.sub(r"\b\d+\b", lambda match: numbers.get(match[0], match[0]), text)
    return re.findall(r"[a-z0-9]+", text)


def missing(text: str, phrases: Sequence[str]) -> List[str]:
    padded = f" {' '.join(words(text))} "
    return [p for p in phrases if f" {' '.join(words(p))} " not in padded]


async def robot(url: str, token: str, out_rate: int = 24_000) -> ClientConnection:
    ws = await connect(url, max_size=None)
    await ws.send(encode(Hello((1,), token, (Op.HEAR, Op.SPEAK), out_rate, "bench")))
    welcome = decode(await ws.recv(), sender=Sender.COMPANION)
    assert isinstance(welcome, Welcome), welcome
    return ws


async def hear(ws: ClientConnection, stream: int, pcm: bytes, *, finish: bool) -> Heard:
    """Send pcm in real time, then hear_close{finish} if asked and the companion has not ended the turn."""
    await ws.send(encode(HearOpen(stream, 0)))
    started = time.perf_counter()
    ended = asyncio.Event()

    async def send() -> None:
        for seq, offset in enumerate(range(0, len(pcm), FRAME * 2)):
            delay = started + offset / 2 / RATE - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            if ended.is_set():
                return
            await ws.send(encode_audio(AudioKind.MIC, stream, seq, pcm[offset : offset + FRAME * 2]))
        await asyncio.sleep(max(0.0, started + len(pcm) / 2 / RATE - time.perf_counter()))
        if finish and not ended.is_set():
            await ws.send(encode(HearClose(stream, CloseHow.FINISH)))

    sender = asyncio.create_task(send())
    partials: List[float] = []
    try:
        while True:
            message = decode(await ws.recv(), sender=Sender.COMPANION)
            now = time.perf_counter() - started
            if isinstance(message, HearPartial) and message.stream == stream:
                partials.append(now)
            elif isinstance(message, HearEndpoint) and message.stream == stream:
                ended.set()
                return Heard(message.text, message.forced, now, len(pcm) / 2 / RATE, partials)
    finally:
        await sender


async def speak(ws: ClientConnection, stream: int, text: str) -> Spoken:
    await ws.send(encode(SpeakStart(stream, text, VoiceRole.MAIN)))
    started = time.perf_counter()
    first, frames = None, 0
    while True:
        frame = await ws.recv()
        if isinstance(frame, bytes):
            audio = decode_audio(frame)
            assert audio.kind is AudioKind.SPEECH and audio.stream == stream and audio.seq == frames
            first = first if first is not None else time.perf_counter() - started
            frames += 1
            continue
        message = decode(frame, sender=Sender.COMPANION)
        if isinstance(message, SpeakDone) and message.stream == stream:
            return Spoken(first, frames, message.frames)


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def spread(values: Sequence[float]) -> Dict[str, float]:
    return {"p50": round(statistics.median(values), 3), "p95": round(percentile(values, 0.95), 3), "max": round(max(values), 3), "n": len(values)}


def start_server(serve_args: Sequence[str]) -> Tuple[subprocess.Popen, str, str]:
    state = Path(tempfile.mkdtemp(prefix="reachy-bench-"))
    base = [sys.executable, "-m", "reachy_companion", "--state-dir", str(state)]
    subprocess.run([*base, "pair", "--host", "127.0.0.1"], check=True, capture_output=True)
    server = subprocess.Popen([*base, "serve", "--insecure", "--port", "0", *serve_args], stderr=subprocess.PIPE, text=True)
    assert server.stderr is not None
    for line in server.stderr:
        sys.stderr.write(line)
        found = re.search(r"serving (ws://\S+)", line)
        if found:
            return server, found.group(1), (state / "token").read_text()
    raise SystemExit(f"serve exited with {server.wait()}")


def memory(pid: int) -> Dict[str, str]:
    summary = subprocess.run(["vmmap", "--summary", str(pid)], capture_output=True, text=True).stdout
    return {key: value.strip() for key, value in re.findall(r"^(Physical footprint(?: \(peak\))?):\s+(\S+)", summary, re.M)}


async def measure(url: str, token: str, corpus: Path, runs: int) -> dict:
    cases = json.loads((corpus / "manifest.json").read_text())["cases"]
    ws = await robot(url, token)
    stream = 1
    forced, own, gaps, transcripts = [], [], [], []
    for _ in range(runs):
        for case in cases:
            pcm = read_wav(corpus / case["file"])
            end = speech_end_s(pcm)
            trimmed = pcm[: int((end + 0.2) * RATE) * 2]
            heard = await hear(ws, stream, trimmed, finish=True)
            stream += 2
            forced.append(heard.endpoint_s - heard.audio_end_s)
            transcripts.append({"case": case["name"], "path": "forced", "forced": heard.forced, "text": heard.text, "missing": missing(heard.text, case["critical"])})
            heard = await hear(ws, stream, pcm, finish=True)
            stream += 2
            if not heard.forced:
                own.append(heard.endpoint_s - end)
            gaps += [b - a for a, b in zip(heard.partial_times_s, heard.partial_times_s[1:])]
            transcripts.append({"case": case["name"], "path": "self", "forced": heard.forced, "text": heard.text, "missing": missing(heard.text, case["critical"])})
    first_audio, frames_ok = [], True
    for _ in range(runs):
        for text in SENTENCES:
            spoken = await speak(ws, stream + 1, text)
            stream += 2
            frames_ok &= spoken.frames > 0 and spoken.frames == spoken.done_frames
            if spoken.first_frame_s is not None:
                first_audio.append(spoken.first_frame_s)
    await ws.close()
    return {
        "forced_final_after_end_of_audio_s": spread(forced),
        "self_endpoint_after_end_of_speech_s": spread(own) if own else None,
        "self_endpoints": f"{len(own)} of {runs * len(cases)}",
        "partial_gap_s": spread(gaps) if gaps else None,
        "speak_start_to_first_frame_s": spread(first_audio),
        "speak_frames_match_done": frames_ok,
        "transcripts": transcripts,
    }


def verdicts(report: dict) -> List[str]:
    lines = []
    final = report["forced_final_after_end_of_audio_s"]["p50"]
    lines.append(f"C2 final, forced: p50 {final * 1000:.0f} ms vs target {TARGETS['final_p50_s'] * 1000:.0f} ms: {'pass' if final <= TARGETS['final_p50_s'] else 'FAIL'}")
    own = report["self_endpoint_after_end_of_speech_s"]
    if own:
        lines.append(f"C2 final, companion's own endpoint: p50 {own['p50'] * 1000:.0f} ms after end of speech (includes the Smart Turn wait)")
    gap = report["partial_gap_s"]
    if gap:
        lines.append(f"C2 partial gap: p50 {gap['p50'] * 1000:.0f} / p95 {gap['p95'] * 1000:.0f} / max {gap['max'] * 1000:.0f} ms vs {TARGETS['partial_gap_s'] * 1000:.0f} ms: {'pass' if gap['p95'] <= TARGETS['partial_gap_s'] else 'FAIL'} at p95")
    first = report["speak_start_to_first_frame_s"]["p50"]
    lines.append(f"C3 first audio: p50 {first * 1000:.0f} ms vs target {TARGETS['first_audio_p50_s'] * 1000:.0f} ms: {'pass' if first <= TARGETS['first_audio_p50_s'] else 'FAIL'}")
    bad = [t for t in report["transcripts"] if t["missing"]]
    lines.append(f"critical phrases: {'all kept' if not bad else bad}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=HERE.parent / "tests" / "data", help="a directory with manifest.json and 16 kHz mono WAVs")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--out", type=Path, help="write the full report as JSON")
    args, serve_args = parser.parse_known_args()
    server, url, token = start_server(serve_args)
    try:
        report = asyncio.run(measure(url, token, args.corpus, args.runs))
        report["serve_args"] = serve_args
        report["memory"] = memory(server.pid)
    finally:
        server.send_signal(2)
        server.wait(10)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "transcripts"}, indent=1))
    print("\n".join(verdicts(report)))


if __name__ == "__main__":
    main()
