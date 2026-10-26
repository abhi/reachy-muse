from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import Dict, List, Optional, Sequence, Tuple

from reachy_muse.device_command import COMMANDS, Commands, DeviceCommand, executable, parse_command, worth_asking

D = DeviceCommand
ALLOWED = executable(wake=True, output=False, face=True, volume=True)
MODEL_LABELS = frozenset(command for command in ALLOWED if COMMANDS[command].evidence)

GRAMMAR_PHRASES = (
    "go to sleep", "okay reachy, go back to sleep", "goodnight reachy", "wake up", "stop listening",
    "that's all, thanks", "louder", "could you speak a little louder?", "you're too loud", "lower your voice",
    "look at me", "stop looking at me", "turn off face tracking", "how do I sleep better", "stop the music",
    "what's the volume of the moon", "tell me about wake-up routines")

PARAPHRASES: Tuple[Tuple[str, DeviceCommand], ...] = (
    ("time to hit the hay", D.SLEEP), ("let's call it a night", D.SLEEP), ("you can rest now", D.SLEEP),
    ("power down for the night", D.SLEEP), ("nap time for you", D.SLEEP), ("you can go back to sleep now", D.SLEEP),
    ("rise and shine little robot", D.WAKE_UP), ("come on, wake yourself up", D.WAKE_UP),
    ("okay I'm done talking to you", D.STOP_LISTENING), ("we're finished, you can stop listening for now", D.STOP_LISTENING),
    ("turn off your microphone", D.STOP_LISTENING),
    ("could you please quiet down a bit", D.VOLUME_DOWN), ("I need you to pipe down", D.VOLUME_DOWN),
    ("can you be a bit quieter", D.VOLUME_DOWN), ("you're kind of loud", D.VOLUME_DOWN),
    ("make your voice softer", D.VOLUME_DOWN), ("speak a tad more quietly", D.VOLUME_DOWN),
    ("crank up your volume", D.VOLUME_UP), ("I can barely hear you", D.VOLUME_UP),
    ("speak a bit more loudly", D.VOLUME_UP), ("your voice is too soft", D.VOLUME_UP),
    ("make yourself louder", D.VOLUME_UP),
    ("keep your eyes on me", D.FACE_TRACKING_ON), ("look over at me", D.FACE_TRACKING_ON),
    ("follow me with your eyes", D.FACE_TRACKING_ON), ("turn toward my face", D.FACE_TRACKING_ON),
    ("quit staring at me", D.FACE_TRACKING_OFF), ("stop following my face around", D.FACE_TRACKING_OFF),
    ("quit tracking me", D.FACE_TRACKING_OFF), ("stop watching me", D.FACE_TRACKING_OFF),
)

QUESTIONS = (
    "how do I sleep better", "stop the music", "what's the volume of the moon", "tell me about wake-up routines",
    "what time should I go to bed tonight", "set an alarm to wake me up at seven", "turn up the thermostat",
    "how loud is a jet engine", "look up the weather in Paris", "who sang stop in the name of love",
    "can you remind me to call mom", "what's a good bedtime story for kids", "how do I stop snoring",
    "turn down the lights", "is it bad to sleep with the tv on", "what does REM sleep mean",
    "follow up on my email to Sam", "find a quiet restaurant nearby", "what are the face tracking features on iphone",
    "play some relaxing sleep music", "how many hours of sleep do teenagers need", "listen to this joke",
    "tell me a story about a sleepy dragon", "what's louder a lion or a tiger", "cancel my dentist appointment",
    "stop my timer", "how do I mute my microphone in zoom", "my neighbor is too loud, what can I do",
    "why do cats sleep so much", "keep an eye on the weather for me",
)


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))]


def grammar(runs: int) -> List[str]:
    commands = Commands()
    async def time_all() -> Dict[str, List[float]]:
        times: Dict[str, List[float]] = {phrase: [] for phrase in GRAMMAR_PHRASES}
        for _ in range(runs):
            for phrase in GRAMMAR_PHRASES:
                began = time.perf_counter()
                await commands.recognize(phrase, ALLOWED)
                times[phrase].append(time.perf_counter() - began)
        return times
    times = asyncio.run(time_all())
    every = [value for values in times.values() for value in values]
    return [f"Grammar over {len(GRAMMAR_PHRASES)} phrases x {runs} runs: p50 {statistics.median(every) * 1e6:.0f} us, "
            f"p95 {percentile(every, .95) * 1e6:.0f} us, max {max(every) * 1e6:.0f} us."]


async def model() -> List[str]:
    from reachy_muse.companion_client import CompanionCommandModel, CompanionLink
    from reachy_muse.companion_server.narrator_mlx import MlxNarrator
    from reachy_muse.companion_server.server import CompanionServer, Engines

    narrator = MlxNarrator()
    await narrator.load()
    token = "bench-token"
    async with CompanionServer(Engines(narrator=narrator, classifier=narrator), token, host="127.0.0.1", port=0) as server:
        link = CompanionLink(server.url[:-3], token=token, out_rate=16000, insecure=True)
        running = asyncio.ensure_future(link.run())
        try:
            await asyncio.wait_for(link.settled(), 10)
            commands = Commands(CompanionCommandModel(link))
            await commands.recognize("could you nap for a bit", ALLOWED)
            rows = []
            for kind, cases in (("paraphrase", PARAPHRASES), ("question", tuple((q, None) for q in QUESTIONS))):
                for text, expected in cases:
                    route = ("grammar" if parse_command(text) is not None
                             else "model" if worth_asking(text, MODEL_LABELS) else "prefilter")
                    began = time.perf_counter()
                    got = await commands.recognize(text, ALLOWED)
                    seconds = time.perf_counter() - began
                    raw = await narrator.classify(text, tuple(MODEL_LABELS))
                    rows.append((kind, text, expected, got, seconds, route, raw))
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
    return report(rows)


def report(rows) -> List[str]:
    lines = []
    for kind in ("paraphrase", "question"):
        mine = [row for row in rows if row[0] == kind]
        times = [row[4] for row in mine]
        right = sum(row[3] is row[2] for row in mine)
        lines.append(f"{kind}s: {len(mine)}, right {right}, p50 {statistics.median(times) * 1000:.0f} ms, "
                     f"p95 {percentile(times, .95) * 1000:.0f} ms, max {max(times) * 1000:.0f} ms")
    asked = [row[4] for row in rows if row[5] == "model"]
    lines.append(f"model round trips: {len(asked)}, p50 {statistics.median(asked) * 1000:.0f} ms, "
                 f"p95 {percentile(asked, .95) * 1000:.0f} ms, max {max(asked) * 1000:.0f} ms")
    raw_hits = [row for row in rows if row[0] == "question" and row[6] is not None]
    lines.append(f"model alone, asked every question with no prefilter or evidence check: {len(raw_hits)} of 30 labelled"
                 + "".join(f"\n  {row[1]!r} -> {row[6].name}" for row in raw_hits))
    false_positives = [row for row in rows if row[0] == "question" and row[3] is not None]
    lines.append(f"false positives: {len(false_positives)} of {sum(row[0] == 'question' for row in rows)}"
                 + "".join(f"\n  {row[1]!r} -> {row[3].name}" for row in false_positives))
    lines.append("")
    lines.append("| kind | phrase | expected | got | ms | decided by | model alone |")
    lines.append("|---|---|---|---|---|---|---|")
    for kind, text, expected, got, seconds, route, raw in rows:
        name = lambda command: command.name if command is not None else "none"  # noqa: E731
        lines.append(f"| {kind} | {text} | {name(expected)} | {name(got)} | {seconds * 1000:.0f} | {route} | {name(raw)} |")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar-only", action="store_true")
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args(argv)
    print("\n".join(grammar(args.runs)))
    if not args.grammar_only:
        print("\n".join(asyncio.run(model())))


if __name__ == "__main__":
    main()
