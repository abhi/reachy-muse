"""Offline route scoring. No network, hardware, model imports, or transcript logging."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from reachy_muse.arbitration import (
    Code, Decision, Evidence, Ignore, InvalidArbitration, LocalDevice, LocalReply, Muse,
    MuseCancel, MuseFollowUp, Snapshot, Target, TextRef, decide, muse_text, template_speech,
    validate_decision, validate_snapshot,
)
from reachy_muse.device_command import DeviceCommand

HERE = Path(__file__).resolve().parent
CASES = HERE / "arbitration_cases.json"
RECORDINGS = HERE / "arbitration_fake_model.json"
ROUTES = ("LOCAL_DEVICE", "LOCAL_REPLY", "MUSE", "MUSE_FOLLOW_UP", "MUSE_CANCEL", "IGNORE")
SPLITS = ("all", "development", "holdout", "adversarial", "audit")


def load_cases(path: Path = CASES) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["version"] != 1 or data["synthetic"] is not True:
        raise InvalidArbitration("fixture_version_or_provenance")
    return data["defaults"], data["cases"]


def snapshot_for(case: dict, defaults: dict) -> Snapshot:
    def target(raw):
        if raw is None:
            return None
        return Target(raw["id"], TextRef(**raw["request"]),
                      tuple(TextRef(**item) for item in raw.get("amendments", ())),
                      TextRef(**raw["clarification"]) if raw.get("clarification") else None,
                      raw.get("state", "running"), raw.get("omitted", False))

    try:
        state = {**defaults, **case["state"]}
        state["capabilities"] = frozenset(DeviceCommand(value) for value in state["capabilities"])
        state["active"] = target(state["active"])
        state["held"] = target(state["held"])
        state["recent"] = tuple(TextRef(**item) for item in state["recent"])
        snapshot = Snapshot(case["id"], case["text"], **state)
        validate_snapshot(snapshot)
        return snapshot
    except (KeyError, ValueError, TypeError, AttributeError):
        raise InvalidArbitration("fixture_schema") from None


@dataclass(frozen=True)
class ModelOutput:
    text: str
    elapsed_ms: float


class RecordedModel:
    def __init__(self, path: Path = RECORDINGS):
        self.data = json.loads(path.read_text(encoding="utf-8"))
        if self.data["kind"] != "handwritten_fake":
            raise InvalidArbitration("recording_provenance")

    def __call__(self, snapshot: Snapshot) -> ModelOutput:
        raw = self.data["responses"].get(snapshot.turn_id, self.data["default"])
        return ModelOutput(raw["text"], raw["elapsed_ms"])


def parse_proposal(snapshot: Snapshot, output: str) -> Decision:
    """Parse the bounded model vocabulary; acceptance is a separate rule check."""
    if not isinstance(output, str) or len(output.encode("utf-8")) > 240:
        raise InvalidArbitration("model_shape")
    tokens = output.strip().split()
    try:
        if tokens == ["MUSE"]:
            action = Muse()
        elif len(tokens) == 2 and tokens[0] == "LOCAL_DEVICE":
            action = LocalDevice(DeviceCommand(tokens[1].lower()))
        elif len(tokens) == 2 and tokens[0] == "LOCAL_REPLY":
            speech = template_speech(tokens[1], snapshot.text)
            if speech is None:
                raise ValueError
            action = LocalReply(speech)
        elif len(tokens) == 2 and tokens[0] in ("MUSE_FOLLOW_UP", "MUSE_CANCEL"):
            if not any(target.id == tokens[1] for target in snapshot.targets):
                raise ValueError
            action = MuseFollowUp(tokens[1]) if tokens[0] == "MUSE_FOLLOW_UP" else MuseCancel(tokens[1])
        elif len(tokens) == 2 and tokens[0] == "IGNORE" and tokens[1] in ("empty", "noise", "echo", "not_addressed"):
            action = Ignore(tokens[1])
        else:
            raise ValueError
    except (KeyError, ValueError):
        raise InvalidArbitration("model_shape") from None
    return Decision(snapshot.turn_id, snapshot.revision, snapshot.context_epoch, snapshot.generation,
                    snapshot.phase, 1.0, (Evidence("model", Code.MODEL),), action)


def select(snapshot: Snapshot, model: Callable[[Snapshot], ModelOutput] | None = None
           ) -> tuple[Decision, Decision | None, str]:
    fallback = decide(snapshot)
    if model is None or fallback.evidence[0].code is not Code.UNRESOLVED:
        return fallback, fallback, "rule"
    if not snapshot.model_eligible:
        return fallback, None, "bypass"
    proposal = None
    try:
        output = model(snapshot)
        proposal = parse_proposal(snapshot, output.text)
        limit = min(snapshot.budget_ms, 50 if snapshot.phase == "final" else 200)
        if not math.isfinite(output.elapsed_ms) or output.elapsed_ms < 0:
            raise InvalidArbitration("model_duration")
        if output.elapsed_ms > limit:
            return replace(fallback, evidence=(Evidence("deadline", Code.MODEL_DEADLINE),)), proposal, "deadline"
        validate_decision(snapshot, proposal)
        return proposal, proposal, "accepted"
    except Exception:
        return replace(fallback, evidence=(Evidence("rule", Code.MODEL_INVALID),)), proposal, "invalid"


def decision_fields(decision: Decision) -> dict:
    action = decision.action
    fields = {"kind": action.kind}
    if isinstance(action, LocalDevice):
        fields["command"] = action.command.value
    elif isinstance(action, LocalReply):
        fields["template"] = action.speech.template_id
    elif isinstance(action, (MuseFollowUp, MuseCancel)):
        fields["target"] = action.target_id
    elif isinstance(action, Ignore):
        fields["reason"] = action.reason
    return fields


def matches(expected: dict, actual: dict) -> bool:
    return (all(actual.get(key) == value for key, value in expected.items() if key not in ("templates", "muse_text"))
            and ("templates" not in expected or actual.get("template") in expected["templates"]))


def score(cases: list[dict], decisions: list[Decision | None]) -> dict:
    if len(decisions) != len(cases):
        raise InvalidArbitration("decision_count")
    confusion = {route: {other: 0 for other in (*ROUTES, "INVALID")} for route in ROUTES}
    correct = Counter()
    count = Counter()
    false_local = local_predicted = local_correct = local_expected = negatives = unsafe = 0
    errors = []
    for case, decision in zip(cases, decisions):
        expected = case["expected"]
        actual = decision_fields(decision) if decision else {"kind": "INVALID"}
        route = actual["kind"]
        good = matches(expected, actual)
        count[expected["kind"]] += 1
        correct[expected["kind"]] += good
        confusion[expected["kind"]][route] += 1
        negatives += case["fact_or_action"]
        false_local += case["fact_or_action"] and route == "LOCAL_REPLY"
        local_predicted += route == "LOCAL_REPLY"
        local_expected += expected["kind"] == "LOCAL_REPLY"
        local_correct += expected["kind"] == "LOCAL_REPLY" and good
        effect = {"LOCAL_DEVICE": "device", "LOCAL_REPLY": "speak", "MUSE": "send",
                  "MUSE_FOLLOW_UP": "send", "MUSE_CANCEL": "cancel"}.get(route)
        unsafe += (effect in case["forbidden_effects"]
                   or (route in ("LOCAL_DEVICE", "MUSE_FOLLOW_UP", "MUSE_CANCEL", "IGNORE") and not good))
        if not good:
            errors.append({"id": case["id"], "expected": {k: v for k, v in expected.items() if k != "muse_text"},
                           "actual": actual})

    def ratio(n, total):
        return {"correct": n, "total": total, "rate": round(n / total, 6) if total else None}

    per_class = {route: ratio(correct[route], count[route]) for route in ROUTES}
    total_correct = sum(correct.values())
    release = (bool(cases) and total_correct / len(cases) >= .98 and negatives > 0
               and all(item["rate"] is not None and item["rate"] >= .95 for item in per_class.values())
               and false_local == unsafe == 0)
    return {"accuracy": ratio(total_correct, len(cases)), "per_class": per_class, "confusion": confusion,
            "false_local_reply": {"errors": false_local, "total": negatives,
                                  "rate": round(false_local / negatives, 6) if negatives else None},
            "local_reply_precision": ratio(local_correct, local_predicted),
            "local_reply_recall": ratio(local_correct, local_expected),
            "unsafe_effects": unsafe, "release_gate": release, "errors": errors}


def spread(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}
    return {"n": len(ordered), **{name: round(ordered[math.ceil(len(ordered) * percentile) - 1], 4)
                                for name, percentile in (("p50", .5), ("p95", .95), ("p99", .99), ("max", 1))}}


def score_decisions(defaults: dict, cases: list[dict], decisions: list[Decision]) -> dict:
    if len(decisions) != len(cases):
        raise InvalidArbitration("decision_count")
    fidelity = fidelity_total = 0
    for case, decision in zip(cases, decisions):
        snapshot = snapshot_for(case, defaults)
        validate_decision(snapshot, decision)
        muse_bound = decision.action.kind in ("MUSE", "MUSE_FOLLOW_UP")
        if ((muse_bound or case["expected"]["kind"] in ("MUSE", "MUSE_FOLLOW_UP"))
                and not isinstance(case["expected"].get("muse_text"), str)):
            raise InvalidArbitration("fixture_muse_text")
        if muse_bound:
            fidelity_total += 1
            fidelity += muse_text(snapshot, decision).encode("utf-8") == case["expected"]["muse_text"].encode("utf-8")
    scored = score(cases, decisions)
    scored["release_gate"] = scored["release_gate"] and fidelity == fidelity_total
    return {"decisions": scored, "transcript_fidelity": {"correct": fidelity, "total": fidelity_total}}


def evaluate(defaults: dict, cases: list[dict], model=None) -> dict:
    decisions, proposals = [], []
    timings = defaultdict(list)
    statuses = Counter()
    for case in cases:
        snapshot = snapshot_for(case, defaults)
        began = time.perf_counter()
        decision, proposal, status = select(snapshot, model)
        timings[decision.action.kind].append((time.perf_counter() - began) * 1000)
        decisions.append(decision)
        proposals.append(proposal)
        statuses[status] += 1
    return {**score_decisions(defaults, cases, decisions), "proposals_before_validation": score(cases, proposals),
            "model_outcomes": dict(statuses),
            "offline_cpu_ms": {route: spread(timings[route]) for route in ROUTES}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("rules", "recorded"), default="rules")
    parser.add_argument("--split", choices=SPLITS, default="all")
    args = parser.parse_args()
    defaults, cases = load_cases()
    if args.split != "all":
        cases = [case for case in cases if case["split"] == args.split]
    model = RecordedModel() if args.engine == "recorded" else None
    print(json.dumps({"engine": args.engine, "split": args.split, **evaluate(defaults, cases, model)}, indent=2))


if __name__ == "__main__":
    main()
