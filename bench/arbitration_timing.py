"""Offline virtual-clock replay using measured delays. No models, sockets, or sleeps."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from bench.arbitration_eval import (
    HERE, ROUTES, SPLITS, RecordedModel, load_cases, score_decisions, select, snapshot_for, spread,
)
from bench.latency_journal import turns
from reachy_muse.arbitration import InvalidArbitration

JOURNAL_PROFILE = HERE / "arbitration_timing.json"
COMPANION_PROFILE = HERE / "generated_motion_runs/with-speech-first.json"
STAGES = {
    "hearing_ms": ("speech_end", "transcript"),
    "dispatch_ms": ("transcript", "dispatch"),
    "first_text_ms": ("dispatch", "first_text"),
    "first_audio_ms": ("dispatch", "first_audio"),
}


def journal_profile(path: Path) -> dict:
    raw = path.read_bytes()
    samples = []
    for line, _, kind, stages in turns(raw.decode().splitlines(), "2026-10-26"):
        if kind not in ("Muse question", "morning briefing"):
            continue
        delays = {name: round((stages[end] - stages[start]) * 1000, 3)
                  for name, (start, end) in STAGES.items() if start in stages and end in stages}
        samples.append({"line": line, "kind": kind, **delays})
    return {"source": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "samples": samples}


@dataclass
class FakeClock:
    ms: float = 0

    def advance(self, ms: float) -> None:
        self.ms += ms


class TimedModel:
    def __init__(self, model, clock: FakeClock, elapsed_ms: float):
        self.model, self.clock, self.elapsed_ms = model, clock, elapsed_ms

    def __call__(self, snapshot):
        output = self.model(snapshot)
        self.clock.advance(self.elapsed_ms)
        return replace(output, elapsed_ms=self.elapsed_ms)


def simulate(defaults: dict, cases: list[dict], delays: dict, remote: list[dict], model=None) -> dict:
    metrics = defaultdict(lambda: defaultdict(list))
    statuses = Counter()
    decisions = []
    for case in cases:
        snapshot = snapshot_for(case, defaults)
        clock = FakeClock()
        clock.advance(delays["hearing_ms"])
        began = clock.ms
        timed_model = TimedModel(model, clock, delays["classify_ms"]) if model is not None else None
        decision, _, status = select(snapshot, timed_model)
        decisions.append(decision)
        statuses[status] += 1
        route = decision.action.kind
        times = metrics[route]
        times["after_final_ms"].append(clock.ms - began)
        times["after_speech_end_ms"].append(clock.ms)
        if route == "LOCAL_REPLY":
            clock.advance(delays["voice_ms"])
            times["first_answer_audio_ms"].append(clock.ms)
        elif route in ("MUSE", "MUSE_FOLLOW_UP") and snapshot.muse_ready:
            for sample in remote:
                sent = FakeClock(clock.ms)
                sent.advance(sample["dispatch_ms"])
                times["dispatch_ms"].append(sent.ms)
                for source, metric in (("first_text_ms", "first_text_ms"),
                                       ("first_audio_ms", "first_answer_audio_ms")):
                    if source in sample:
                        times[metric].append(sent.ms + sample[source])
    return {**score_decisions(defaults, cases, decisions), "model_outcomes": dict(statuses),
            "fake_stage_ms": delays,
            "paths": {route: {name: spread(metrics[route][name]) for name in (
                "after_final_ms", "after_speech_end_ms", "dispatch_ms", "first_text_ms", "first_answer_audio_ms"
            )} for route in ROUTES}}


def benchmark(defaults: dict, cases: list[dict], journal: dict, model=None) -> dict:
    companion = json.loads(COMPANION_PROFILE.read_text())["contention"]["alone_s"]
    remote = [sample for sample in journal["samples"]
              if sample["kind"] == "Muse question" and "dispatch_ms" in sample]
    if not remote:
        raise InvalidArbitration("missing_timing_samples")
    profiles = {}
    for quantile in ("p50", "p95"):
        delays = {name: companion[source][quantile] * 1000 for name, source in (
            ("hearing_ms", "hear_final"), ("classify_ms", "classify"), ("voice_ms", "speak_first_audio"))}
        profiles[quantile] = simulate(defaults, cases, delays, remote, model)
    return {"clock": "simulated_ms_excluding_python_cpu", "journal": journal,
            "companion_source": str(COMPANION_PROFILE.relative_to(HERE.parent)),
            "remote_samples": len(remote), "profiles": profiles,
            "background_briefing_first_text_ms": spread([
                sample["first_text_ms"] for sample in journal["samples"]
                if sample["kind"] == "morning briefing" and "first_text_ms" in sample])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("rules", "recorded"), default="recorded")
    parser.add_argument("--split", choices=SPLITS, default="all")
    parser.add_argument("--journal", type=Path, help="rebuild numeric timing evidence from a local journal")
    args = parser.parse_args()
    defaults, cases = load_cases()
    if args.split != "all":
        cases = [case for case in cases if case["split"] == args.split]
    journal = journal_profile(args.journal) if args.journal else json.loads(JOURNAL_PROFILE.read_text())
    model = RecordedModel() if args.engine == "recorded" else None
    print(json.dumps({"engine": args.engine, "split": args.split,
                      **benchmark(defaults, cases, journal, model)}, indent=2))


if __name__ == "__main__":
    main()
