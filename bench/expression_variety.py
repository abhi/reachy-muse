"""Offline semantic coverage, gesture variety, recorded-log inventory, and CPU timing.

PYTHONPATH=src python bench/expression_variety.py --morning-log PATH --json report.json
The log is read locally; no text is sent anywhere. Its text is not copied into reports.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
import time

import numpy as np

from reachy_muse.contract.expressions import Expression
from reachy_muse.motion.sentence import SentenceMotion, classify_sentence
from reachy_muse.motion.prosody import ProsodyTracker
try:
    from reachy_muse.motion.sentence_delivery import SentenceSelector
except ModuleNotFoundError:
    SentenceSelector = None


def quantiles(values):
    return {key: round(float(np.percentile(values, percent)), 6)
            for key, percent in (("p50_ms", 50), ("p95_ms", 95), ("p99_ms", 99), ("max_ms", 100))}


def select(rows):
    selector = SentenceSelector() if SentenceSelector else None
    return [selector.choose(row) if selector else row for row in rows]


def identity(plan):
    return getattr(plan, "variant", None) or plan.expression.value


def longest_run(names):
    longest = run = 0
    previous = None
    for name in names:
        run = run + 1 if name == previous else 1
        previous = name
        longest = max(longest, run)
    return longest


def inventory(path):
    raw = path.read_bytes()
    lines = raw.decode().splitlines()
    briefing = []
    replies = []
    selector = SentenceSelector() if SentenceSelector else None
    for previous, line in zip(lines, lines[1:]):
        if "Muse speech started" in previous and "spoke with expression" in line:
            briefing.append(line.rsplit(" ", 1)[-1])
        if "motion for Muse's realtime reply:" in line:
            replies.append(re.search(r"expressions (.*?),", line)[1].split())
    def sequence(names):
        plans = [SentenceMotion(Expression(name)) for name in names]
        chosen = [selector.choose(plan) if selector else plan for plan in plans]
        return {"recorded": names, "recorded_counts": dict(Counter(names)),
                "gesture_replay": [identity(p) for p in chosen],
                "replay_counts": dict(Counter(identity(p) for p in chosen))}
    return {"sha256": hashlib.sha256(raw).hexdigest(), "transcript_available": False,
            "initial_history": "empty; retained across briefing and realtime replies",
            "lexical_agreement": None, "briefing": sequence(briefing),
            "realtime": [sequence(reply) for reply in replies]}


def prosody_profiles(classified):
    if SentenceSelector is None:
        return None
    profiles = {}
    t = np.arange(19200) / 24000
    for name, amplitude, frequency in (("quiet", 900, np.full(len(t), 160.)),
                                       ("animated", 9000, np.full(len(t), 160.)),
                                       ("rising", 6000, 140 + 140 * t / .8)):
        pcm = amplitude * np.sin(2 * np.pi * np.cumsum(frequency) / 24000)
        pcm *= .6 + .4 * np.sin(2 * np.pi * 4 * t) ** 2
        tracker = ProsodyTracker()
        began = time.process_time_ns()
        tracker.feed(pcm)
        delivery = tracker.delivery()
        cpu = (time.process_time_ns() - began) / 1e6
        assert delivery is not None
        selector = SentenceSelector()
        chosen = [selector.choose(plan, delivery) for plan in classified]
        assert [p.expression for p in chosen] == [p.expression for p in classified]
        profiles[name] = {"pitch_range_semitones": delivery.pitch_range, "energy_dbfs": delivery.energy_dbfs,
                          "syllables_per_s": delivery.syllables_per_s, "rise_semitones": delivery.rise,
                          "analysis_cpu_ms": cpu, "semantic_changes": 0,
                          "gestures": dict(Counter(identity(p) for p in chosen))}
    return profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morning-log", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--rounds", type=int, default=100)
    args = parser.parse_args()
    corpus = Path(__file__).with_name("expression_sentences.json")
    rows = json.loads(corpus.read_text())
    assert len(rows) == 200
    classified = [classify_sentence(row["text"]) for row in rows]
    chosen = select(classified)
    mismatches = [{"id": row["id"], "expected": row["label"], "actual": plan.expression.value}
                  for row, plan in zip(rows, classified) if row["label"] != plan.expression.value]
    elapsed, selection = [], []
    for _ in range(args.rounds):
        selector = SentenceSelector() if SentenceSelector else None
        for row in rows:
            started = time.process_time_ns()
            plan = classify_sentence(row["text"])
            stopped = time.process_time_ns()
            if selector:
                selector.choose(plan)
            ended = time.process_time_ns()
            elapsed.append((stopped - started) / 1e6)
            selection.append((ended - started) / 1e6)
    bounds = []
    for _ in range(1000):
        started = time.process_time_ns()
        classify_sentence("word " * 1000 + "wonderful!")
        bounds.append((time.process_time_ns() - started) / 1e6)
    factual = select([classify_sentence("The report contains three sections.") for _ in range(30)])
    report = {"host": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()},
              "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(), "sentences": len(rows),
              "labels": dict(Counter(row["label"] for row in rows)),
              "expressions": dict(Counter(p.expression.value for p in classified)),
              "gestures": dict(Counter(identity(p) for p in chosen)),
              "agreement": (len(rows) - len(mismatches)) / len(rows), "mismatches": mismatches,
              "classifier_cpu": quantiles(elapsed), "classifier_and_selection_cpu": quantiles(selection),
              "bounded_long_input_cpu": quantiles(bounds), "calls": len(elapsed),
              "prosody_profiles": prosody_profiles(classified),
              "cm4_estimate_multiplier": 10, "cm4_estimated_classifier_cpu": quantiles([v * 10 for v in elapsed]),
              "factual_run": {"gestures": dict(Counter(identity(p) for p in factual)),
                              "max_identical_run": longest_run([identity(p) for p in factual])}}
    if args.morning_log:
        report["morning_log"] = inventory(args.morning_log)
    output = json.dumps(report, indent=2) + "\n"
    if args.json:
        args.json.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
