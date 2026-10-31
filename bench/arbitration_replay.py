"""Replay synthetic transcript revisions through the shadow contract, without effects."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from bench.arbitration_eval import RecordedModel, decision_fields, load_cases, select, snapshot_for
from reachy_muse.arbitration import InvalidArbitration, muse_text, validate_decision


def replay(defaults: dict, cases: list[dict], model=None) -> dict:
    traces = []
    partials = changed = revoked = 0
    for case in cases:
        final = replace(snapshot_for(case, defaults), revision=len(case["partials"]))
        proposals = []
        stages = []
        previous_at = -1
        for revision, partial in enumerate(case["partials"]):
            if not previous_at <= partial["at_ms"] <= case["final_at_ms"]:
                raise InvalidArbitration("fixture_timing")
            previous_at = partial["at_ms"]
            request = replace(final, text=partial["text"], revision=revision, phase="provisional", budget_ms=200)
            decision, _, _ = select(request, model)
            validate_decision(request, decision)
            try:
                muse_text(request, decision)
            except InvalidArbitration:
                pass
            else:
                raise InvalidArbitration("partial_dispatch")
            proposals.append(decision)
            stages.append({"at_ms": partial["at_ms"], "phase": "provisional", **decision_fields(decision)})
            partials += 1
        decision, _, status = select(final, model)
        validate_decision(final, decision)
        for proposal in proposals:
            changed += proposal.action != decision.action
            try:
                validate_decision(final, proposal)
            except InvalidArbitration:
                revoked += 1
            else:
                raise InvalidArbitration("partial_still_valid")
        stages.append({"at_ms": case["final_at_ms"], "phase": "final", **decision_fields(decision)})
        traces.append({"id": case["id"], "stages": stages, "model_status": status})
    return {"cases": len(cases), "partial_revisions": partials, "route_changes_at_final": changed,
            "revoked_partial_decisions": revoked, "traces": traces}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("rules", "recorded"), default="rules")
    args = parser.parse_args()
    defaults, cases = load_cases()
    print(json.dumps(replay(defaults, cases, RecordedModel() if args.engine == "recorded" else None), indent=2))


if __name__ == "__main__":
    main()
