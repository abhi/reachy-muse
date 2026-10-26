"""Summarize an offline journal using tools.turn_timeline's parser.

    python -m bench.latency_journal JOURNAL --since 2026-10-12

No network, model, robot, or service access. Missing marks remain missing.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
from pathlib import Path
import re

from tools.turn_timeline import parse


KINDS = ("device command", "local chat", "Muse question", "morning briefing", "empty transcript", "unknown")
PAIRS = (
    ("wake to speech end", "wake", "speech_end"),
    ("endpoint silence", "speech_end", "endpoint"),
    ("transcription", "endpoint", "transcript"),
    ("dispatch preparation", "transcript", "dispatch"),
    ("Muse acceptance", "dispatch", "muse_ack"),
    ("first status", "muse_ack", "first_status"),
    ("Muse first text", "first_status", "first_text"),
    ("reply completion", "first_text", "reply_done"),
    ("narration", "reply_done", "narration_done"),
    ("answer audio preparation / queue", "narration_done", "first_audio"),
    ("acknowledgement audio", "dispatch", "ack_audio"),
    ("local chat audio", "transcript", "chat_audio"),
    ("wake to answer audio", "wake", "answer_audio"),
    ("speech end to answer audio", "speech_end", "answer_audio"),
)
TIMESTAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})")


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (index - lower)


def turns(lines, since):
    context = []
    result = []
    for number, line in enumerate(lines, 1):
        context.append(line)
        parsed = parse([line])
        if not parsed:
            continue
        when, stages, facts = parsed[0]
        if "command" in facts:
            kind = "device command"
        elif "chat_audio" in stages:
            kind = "local chat"
        elif "speech_end" in stages and "dispatch" in stages:
            kind = "Muse question"
        elif any("Speech recognition outcome empty" in item for item in context):
            kind = "empty transcript"
        elif "dispatch" in stages and any("Muse accepted Reachy's morning briefing request" in item for item in context):
            kind = "morning briefing"
        else:
            kind = "unknown"
        context.clear()
        if when >= since:
            result.append((number, when, kind, stages))
    return result


def duration(stages, start, end):
    if end == "answer_audio":
        end = "chat_audio" if "chat_audio" in stages else "first_audio"
    if start in stages and end in stages:
        return stages[end] - stages[start]
    return None


def summarize(lines, since):
    rows = turns(lines, since)
    counts = Counter(row[2] for row in rows)
    print("Turn counts: " + ", ".join(f"{kind}={counts[kind]}" for kind in KINDS))
    print("Cells are n: p50 / p95 seconds. Percentiles use linear interpolation.")
    print("| Stage | All | " + " | ".join(KINDS) + " |")
    print("|---|" + "---|" * (len(KINDS) + 1))
    for label, start, end in PAIRS:
        cells = []
        for kind in (None, *KINDS):
            values = [value for _, _, turn_kind, stages in rows if kind is None or kind == turn_kind
                      if (value := duration(stages, start, end)) is not None]
            cells.append(f"{len(values)}: {percentile(values, .5):.3f} / {percentile(values, .95):.3f}"
                         if values else "missing")
        print("| " + " | ".join((label, *cells)) + " |")
    print("\nStage sums, not forecasts of achievable savings:")
    for label, start, end in PAIRS[:10]:
        values = [value for _, _, _, stages in rows if (value := duration(stages, start, end)) is not None]
        print(f"{label}: {sum(values):.3f} s, n={len(values)}")
    print("\nSource line, date, kind:")
    for number, when, kind, _ in rows:
        print(number, when, kind)
    registered = None
    gaps = []
    for line in lines:
        match = TIMESTAMP.match(line)
        if not match or match[1] < since:
            continue
        when = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S,%f")
        if "registered with the Muse" in line:
            registered = when
        elif "Reachy is ready" in line and registered is not None:
            gaps.append((when - registered).total_seconds())
            registered = None
    if gaps:
        print(f"\nRegistration to ready: n={len(gaps)}, p50={percentile(gaps, .5):.3f}, "
              f"p95={percentile(gaps, .95):.3f}, sum={sum(gaps):.3f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--since", default="2026-10-12")
    args = parser.parse_args()
    data = args.journal.read_bytes()
    lines = data.decode().splitlines()
    timestamps = [match[1] for line in lines if (match := TIMESTAMP.match(line))]
    print(f"Journal SHA256: {hashlib.sha256(data).hexdigest()}")
    print(f"Timestamp range: {min(timestamps)} to {max(timestamps)}")
    summarize(lines, args.since)


if __name__ == "__main__":
    main()
