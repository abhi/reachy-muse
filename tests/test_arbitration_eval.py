import json
import re
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from bench.arbitration_eval import (
    ModelOutput, RecordedModel, evaluate, load_cases, parse_proposal, score, score_decisions, select, snapshot_for,
)
from bench.arbitration_models import measure
from reachy_muse.arbitration import Code, InvalidArbitration, Muse, Snapshot, decide, muse_text


def test_all_240_design_rows_are_transcribed_and_families_do_not_cross_the_split():
    defaults, cases = load_cases()
    seeds = [case for case in cases if case["split"] in ("development", "holdout")]
    assert Counter(case["split"] for case in seeds) == {"development": 160, "holdout": 80}
    assert Counter(case["expected"]["kind"] for case in seeds) == {
        "LOCAL_DEVICE": 36, "LOCAL_REPLY": 40, "MUSE": 92, "MUSE_FOLLOW_UP": 32,
        "MUSE_CANCEL": 16, "IGNORE": 24,
    }
    rows = re.findall(r'^\| ([DLMFCI]\d\d) \| (.*?) \|',
                      Path("docs/design/companion-arbitrator.md").read_text(), re.M)
    assert {id: {"Empty string": "", "Whitespace only": "   "}.get(text, text) for id, text in rows} == {
        case["id"]: case["text"] for case in seeds}
    splits = defaultdict(set)
    for case in seeds:
        splits[case["family"]].add(case["split"])
    assert all(len(assigned) == 1 for assigned in splits.values())
    assert len({case["id"] for case in cases}) == 324
    for case in cases:
        snapshot_for(case, defaults)


def test_frozen_scores_count_semantic_misses_and_literal_targets():
    defaults, cases = load_cases()
    cases = [case for case in cases if case["split"] != "audit"]
    report = evaluate(defaults, cases)
    assert report["decisions"]["accuracy"] == {"correct": 235, "total": 264, "rate": .890152}
    assert report["decisions"]["false_local_reply"] == {"errors": 0, "total": 199, "rate": 0.0}
    assert report["decisions"]["local_reply_precision"] == {"correct": 40, "total": 40, "rate": 1.0}
    assert report["decisions"]["local_reply_recall"] == {"correct": 40, "total": 40, "rate": 1.0}
    assert report["decisions"]["per_class"]["MUSE_FOLLOW_UP"] == {"correct": 7, "total": 36, "rate": .194444}
    assert report["transcript_fidelity"] == {"correct": 147, "total": 147}
    assert report["decisions"]["unsafe_effects"] == 0
    assert report["decisions"]["release_gate"] is False
    holdout = evaluate(defaults, [case for case in cases if case["split"] == "holdout"])
    assert holdout["decisions"]["accuracy"] == {"correct": 74, "total": 80, "rate": .925}


@pytest.mark.parametrize("kind,correct", [("MUSE", 1), ("MUSE_FOLLOW_UP", 1), (None, 0)])
@pytest.mark.parametrize("runner", [evaluate, measure])
def test_changed_muse_text_fails_release_gate_with_correct_routes(monkeypatch, kind, correct, runner):
    defaults, cases = load_cases()
    cases = [case for case in cases if case["id"] in ("D01", "L01", "M01", "F21", "C01", "I01")]

    def run():
        report = runner(defaults, cases, RecordedModel())
        return report["rounds"][0] if runner is measure else report

    original = run()
    assert original["decisions"]["release_gate"] is True

    def changed_text(snapshot, decision):
        if kind is None or decision.action.kind == kind:
            return "corrupted literal"
        return muse_text(snapshot, decision)

    monkeypatch.setattr("bench.arbitration_eval.muse_text", changed_text)
    report = run()
    assert report["decisions"]["accuracy"] == {"correct": 6, "total": 6, "rate": 1.0}
    assert report["decisions"]["release_gate"] is False
    assert original["transcript_fidelity"] == {"correct": 2, "total": 2}
    assert report["transcript_fidelity"] == {"correct": correct, "total": 2}
    assert "corrupted literal" not in json.dumps(report)


@pytest.mark.parametrize("case_id", ["M01", "F21"])
@pytest.mark.parametrize("expected_text", [{}, {"muse_text": None}, {"muse_text": 7}])
@pytest.mark.parametrize("runner", [evaluate, measure])
def test_muse_bound_cases_require_literal_expected_text(case_id, expected_text, runner):
    defaults, cases = load_cases()
    case = next(case for case in cases if case["id"] == case_id)
    case["expected"] = {key: value for key, value in case["expected"].items() if key != "muse_text"} | expected_text
    with pytest.raises(InvalidArbitration, match="^fixture_muse_text$"):
        runner(defaults, [case], RecordedModel())


def test_fake_outputs_exercise_invalid_and_late_proposals_without_authorizing_effects():
    defaults, cases = load_cases()
    cases = [case for case in cases if case["split"] != "audit"]
    report = evaluate(defaults, cases, RecordedModel())
    assert report["decisions"]["accuracy"]["correct"] == 235
    assert report["model_outcomes"] == {"rule": 124, "invalid": 12, "accepted": 127, "deadline": 1}
    assert report["proposals_before_validation"]["false_local_reply"]["errors"] == 1
    assert report["decisions"]["false_local_reply"]["errors"] == 0


def test_wrong_target_counts_as_an_error_even_when_route_matches():
    defaults, cases = load_cases()
    case = next(case for case in cases if case["id"] == "T05")
    decision = decide(snapshot_for(case, defaults))
    assert decision.action.target_id == "recipe-1"
    wrong = replace(decision, action=replace(decision.action, target_id="weather-1"))
    assert score([case], [wrong])["accuracy"] == {"correct": 0, "total": 1, "rate": 0.0}
    assert score([case], [wrong])["unsafe_effects"] == 1


@pytest.mark.parametrize("case_id", ["T01", "T06"])
def test_incorrect_follow_up_fails_release_gate_even_at_high_accuracy(case_id):
    defaults, cases = load_cases()
    decisions = []
    for case in cases:
        expected = case["expected"]
        argument = next((expected[key] for key in ("command", "target", "reason") if key in expected), "")
        if "templates" in expected:
            argument = expected["templates"][0]
        decisions.append(parse_proposal(snapshot_for(case, defaults), f"{expected['kind']} {argument}"))
    assert score(cases, decisions)["release_gate"] is True

    index = next(index for index, case in enumerate(cases) if case["id"] == case_id)
    decisions[index] = parse_proposal(snapshot_for(cases[index], defaults), "MUSE_FOLLOW_UP weather-1")
    report = score(cases, decisions)
    assert report["accuracy"] == {"correct": 323, "total": 324, "rate": .996914}
    assert report["unsafe_effects"] == 1
    assert report["release_gate"] is False
    assert report["errors"] == [{
        "id": case_id,
        "expected": {"kind": "MUSE_FOLLOW_UP", "target": "recipe-1"} if case_id == "T01" else {"kind": "MUSE"},
        "actual": {"kind": "MUSE_FOLLOW_UP", "target": "weather-1"},
    }]


def test_zero_negative_denominator_is_missing_coverage():
    defaults, cases = load_cases()
    case = next(case for case in cases if case["id"] == "L01")
    result = score([case], [decide(snapshot_for(case, defaults))])
    assert result["false_local_reply"]["rate"] is None
    assert result["release_gate"] is False


@pytest.mark.parametrize("count", [0, 59, 61])
@pytest.mark.parametrize("scorer", [score, score_decisions])
def test_scoring_requires_one_decision_per_fixture(count, scorer):
    defaults, cases = load_cases()
    cases = [case for case in cases if case["split"] == "audit"]
    decisions = [decide(snapshot_for(case, defaults)) for case in cases]
    assert score_decisions(defaults, cases, decisions)["decisions"]["release_gate"] is True
    decisions = (decisions + [decisions[-1]])[:count]
    with pytest.raises(InvalidArbitration, match="^decision_count$"):
        if scorer is score:
            scorer(cases, decisions)
        else:
            scorer(defaults, cases, decisions)


@pytest.mark.parametrize("raw", ["MUSE extra", "MUSE_CANCEL invented", "IGNORE unknown", "LOCAL_REPLY invented",
                                 '{"route":"MUSE"}', "private sentinel" * 50])
def test_proposal_parser_rejects_unbounded_or_invented_fields_without_echoing_them(raw):
    with pytest.raises(InvalidArbitration, match="^model_shape$"):
        parse_proposal(Snapshot("t", "What's the weather?"), raw)


@pytest.mark.parametrize("phase,budget,elapsed,status", [
    ("final", 1000, 51, "deadline"), ("final", 20, 21, "deadline"),
    ("provisional", 1000, 201, "deadline"), ("final", 50, 50, "accepted"),
])
def test_deadline_includes_recorded_queue_and_generation_time(phase, budget, elapsed, status):
    request = Snapshot("t", "What's the weather?", phase=phase, budget_ms=budget)
    result, _, observed = select(request, lambda _: ModelOutput("MUSE", elapsed))
    assert result.action == Muse()
    assert observed == status


def test_model_exceptions_and_private_output_are_not_logged(caplog):
    def fail(_):
        raise ValueError("private sentinel")
    result, proposal, status = select(Snapshot("t", "private request"), fail)
    assert result.evidence[0].code is Code.MODEL_INVALID
    assert proposal is None
    assert status == "invalid"
    assert caplog.text == ""


def test_report_contains_no_user_or_model_text():
    defaults, cases = load_cases()
    report = json.dumps(evaluate(defaults, cases, RecordedModel()))
    assert "Good morning" not in report
    assert "What's the weather" not in report
    assert "not a route" not in report


@pytest.mark.parametrize("model", [None, RecordedModel()])
def test_independent_audit_coverage_and_decisions(model):
    defaults, cases = load_cases()
    audit = [case for case in cases if case["split"] == "audit"]
    assert len(audit) == 60
    assert Counter(case["family"] for case in audit) == {
        "audit_mixed": 10, "audit_asr": 10, "audit_partial": 10, "audit_echo": 10,
        "audit_side_conversation": 10, "audit_morning_briefing": 10,
    }
    existing_text = {case["text"] for case in cases if case["split"] != "audit"}
    assert len({case["text"] for case in audit} - existing_text) == 60
    assert all(case["rationale"] for case in audit)
    report = evaluate(defaults, audit, model)
    assert report["decisions"]["accuracy"] == {"correct": 60, "total": 60, "rate": 1.0}
    assert report["decisions"]["false_local_reply"] == {"errors": 0, "total": 42, "rate": 0.0}
    assert report["transcript_fidelity"] == {"correct": 31, "total": 31}
    assert report["decisions"]["unsafe_effects"] == 0
    combined = evaluate(defaults, cases, model)
    assert combined["decisions"]["accuracy"] == {"correct": 295, "total": 324, "rate": .910494}
    assert combined["decisions"]["false_local_reply"] == {"errors": 0, "total": 241, "rate": 0.0}
    assert combined["transcript_fidelity"] == {"correct": 178, "total": 178}
    assert combined["decisions"]["release_gate"] is False


def test_closed_windows_and_oversized_inputs_never_reach_the_model():
    def forbidden(_):
        pytest.fail("model called")
    for request in (Snapshot("t", "Hello", admission="closed"), Snapshot("t", "é" * 3000)):
        select(request, forbidden)
