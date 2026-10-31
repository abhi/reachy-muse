import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from bench.arbitration_eval import ModelOutput, RecordedModel, load_cases, parse_proposal, snapshot_for
from bench.arbitration_models import load_model, main, measure
from bench.arbitration_replay import replay
from reachy_muse.arbitration import (
    InvalidArbitration, Muse, MuseCancel, MuseFollowUp, Snapshot, Target, TextRef, decide, validate_decision,
)


@pytest.mark.parametrize("model", [None, RecordedModel()])
def test_pauses_and_final_suffixes_revoke_every_provisional_decision(model):
    defaults, cases = load_cases()
    cases = [case for case in cases if case["split"] != "audit"]
    result = replay(defaults, cases, model)
    assert {key: value for key, value in result.items() if key != "traces"} == {
        "cases": 264, "partial_revisions": 10, "route_changes_at_final": 3, "revoked_partial_decisions": 10,
    }
    greeting = next(trace for trace in result["traces"] if trace["id"] == "M01")
    assert [(stage["at_ms"], stage["kind"]) for stage in greeting["stages"]] == [
        (200, "LOCAL_REPLY"), (750, "MUSE"), (1000, "MUSE")]
    command = next(trace for trace in result["traces"] if trace["id"] == "M89")
    assert [(stage["at_ms"], stage["kind"]) for stage in command["stages"]] == [
        (200, "LOCAL_DEVICE"), (800, "MUSE"), (1000, "MUSE")]
    assert "Good morning" not in json.dumps(result)


@pytest.mark.parametrize("model", [None, RecordedModel()])
def test_audit_partial_commands_and_greetings_are_revoked(model):
    defaults, cases = load_cases()
    result = replay(defaults, [case for case in cases if case["split"] == "audit"], model)
    assert {key: value for key, value in result.items() if key != "traces"} == {
        "cases": 60, "partial_revisions": 13, "route_changes_at_final": 10, "revoked_partial_decisions": 13,
    }
    face = next(trace for trace in result["traces"] if trace["id"] == "E30")
    assert [stage["command"] for stage in face["stages"]] == ["stop_talking", "face_tracking_off"]


def test_held_cancellation_is_invalid_after_either_target_changes_epoch():
    defaults, cases = load_cases()
    request = snapshot_for(next(case for case in cases if case["id"] == "T05"), defaults)
    decision = decide(request)
    assert decision.action == MuseCancel("recipe-1")
    for field in ("active", "held"):
        target = getattr(request, field)
        amended = replace(target, amendments=(*target.amendments, TextRef("new", "user", "Actually, for two.")))
        current = replace(request, context_epoch=1, **{field: amended})
        with pytest.raises(InvalidArbitration, match="^stale_decision$"):
            validate_decision(current, decision)
    assert decide(replace(request, held=None, context_epoch=1)).action == Muse()


def test_completion_and_reconnect_do_not_retarget_prior_follow_up():
    target = Target("explain-1", TextRef("request-1", "user", "Explain thunderstorms."))
    request = Snapshot("t", "Use simpler words.", active=target)
    decision = decide(request)
    assert decision.action == MuseFollowUp("explain-1")
    completed = replace(request, active=replace(target, state="completed"), context_epoch=1)
    assert decide(completed).action == Muse()
    for changed in (completed, replace(request, generation=1), replace(request, revision=1)):
        with pytest.raises(InvalidArbitration, match="^stale_decision$"):
            validate_decision(changed, decision)


def test_model_entrypoint_requires_flag_before_import_or_loading(monkeypatch):
    def forbidden(_):
        pytest.fail("loaded a real engine")
    monkeypatch.setattr("bench.arbitration_models.load_model", forbidden)
    with pytest.raises(SystemExit) as result:
        main([])
    assert result.value.code == 2
    with pytest.raises(SystemExit) as result:
        main(["--real-models", "--rounds", "0"])
    assert result.value.code == 2


def test_real_model_report_can_be_exercised_entirely_with_a_fake():
    defaults, cases = load_cases()
    chosen = [case for case in cases if case["id"] in ("L04", "M01", "I13")]
    seen = []

    def model(snapshot):
        seen.append(snapshot.turn_id)
        return ModelOutput("LOCAL_REPLY morning", 10)

    report = measure(defaults, chosen, model)
    assert seen == ["L04", "M01"]
    assert report["first_model_request_ms"] == 10
    [round] = report["rounds"]
    assert round["raw_model"]["accuracy"] == {"correct": 1, "total": 2, "rate": .5}
    assert round["raw_model"]["false_local_reply"] == {"errors": 1, "total": 1, "rate": 1.0}
    assert round["decisions"]["accuracy"] == {"correct": 3, "total": 3, "rate": 1.0}
    assert round["outcomes"] == {"rule": 1, "invalid": 1, "not_uploaded": 1}


@pytest.fixture
def fake_classifier(monkeypatch):
    fake = SimpleNamespace(prompts=[], output="MUSE")

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages, ensure_ascii=False)

    def stream_generate(model, tokenizer, prompt, **kwargs):
        fake.prompts.append(prompt)
        yield SimpleNamespace(text=fake.output)

    mx = SimpleNamespace(synchronize=lambda: None)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(
        load=lambda repo: (object(), Tokenizer()), stream_generate=stream_generate))
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", SimpleNamespace(make_sampler=lambda **kwargs: None))
    fake.classify = load_model("fake")
    return fake


def test_model_prompt_uses_opaque_ids_but_keeps_text_and_report_ids(fake_classifier):
    defaults, cases = load_cases()
    case = next(case for case in cases if case["id"] == "T01")
    request = snapshot_for(case, defaults)
    request = replace(request, active=replace(request.active, clarification=TextRef(
        "weather-1:question", "muse", "Should I use tomorrow?")),
        recent=(TextRef("recipe-1:echo", "local", "I'm still working."),), echo_ref="recipe-1:echo")
    fake_classifier.classify(request)
    fake_classifier.classify(replace(request, turn_id="D01"))

    prompts = fake_classifier.prompts
    data = json.loads(json.loads(prompts[0])[1]["content"])
    assert "turn_id" not in data
    assert data["text"] == "And make it vegetarian."
    assert data["active"]["id"] == "target-1"
    assert data["held"]["id"] == "target-2"
    assert data["active"]["request"] == {
        "id": "ref-1", "author": "user", "text": "What's the weather in Boston?"}
    assert data["active"]["amendments"] == [
        {"id": "ref-2", "author": "user", "text": "Tomorrow, please."}]
    assert data["active"]["clarification"] == {
        "id": "ref-3", "author": "muse", "text": "Should I use tomorrow?"}
    assert data["held"]["request"] == {
        "id": "ref-4", "author": "user", "text": "Find a dinner recipe."}
    assert data["held"]["amendments"] == [
        {"id": "ref-5", "author": "user", "text": "For four people."}]
    assert data["recent"] == [{"id": "ref-6", "author": "local", "text": "I'm still working."}]
    assert data["echo_ref"] == "ref-6"
    assert all(item.id not in prompts[0] for item in (*request.targets, *request.refs))
    assert prompts[0] == prompts[1]
    assert request.active.id == "weather-1"
    assert request.echo_ref == "recipe-1:echo"
    report = measure(defaults, [case], fake_classifier.classify)
    assert report["rounds"][0]["raw_model"]["errors"][0]["id"] == "T01"


def test_model_identifiers_do_not_reveal_target_topics(fake_classifier):
    defaults, cases = load_cases()
    for case_id in ("T01", "T03"):
        request = snapshot_for(next(case for case in cases if case["id"] == case_id), defaults)
        fake_classifier.classify(request)
    data = [json.loads(json.loads(prompt)[1]["content"]) for prompt in fake_classifier.prompts]
    identifiers = [
        [(item[slot]["id"], item[slot]["request"]["id"], item[slot]["amendments"][0]["id"])
         for slot in ("active", "held")]
        for item in data
    ]
    assert identifiers[0] == identifiers[1] == [
        ("target-1", "ref-1", "ref-2"), ("target-2", "ref-3", "ref-4")]
    assert data[0]["active"]["request"]["text"] == "What's the weather in Boston?"
    assert data[1]["active"]["request"]["text"] == "Find a dinner recipe."


@pytest.mark.parametrize("case_id,output,action", [
    ("T01", "MUSE_FOLLOW_UP target-2", MuseFollowUp("recipe-1")),
    ("T02", "MUSE_FOLLOW_UP target-1", MuseFollowUp("weather-1")),
    ("T03", "MUSE_FOLLOW_UP target-1", MuseFollowUp("recipe-1")),
    ("T04", "MUSE_FOLLOW_UP target-2", MuseFollowUp("weather-1")),
    ("T05", "MUSE_CANCEL target-2", MuseCancel("recipe-1")),
])
def test_model_target_aliases_are_restored_for_scoring(fake_classifier, case_id, output, action):
    defaults, cases = load_cases()
    case = next(case for case in cases if case["id"] == case_id)
    request = snapshot_for(case, defaults)
    fake_classifier.output = output
    proposal = parse_proposal(request, fake_classifier.classify(request).text)
    assert proposal.action == action
    report = measure(defaults, [case], fake_classifier.classify)
    assert report["rounds"][0]["raw_model"]["accuracy"] == {"correct": 1, "total": 1, "rate": 1.0}


@pytest.mark.parametrize("output", [
    "MUSE_CANCEL recipe-1", "MUSE_FOLLOW_UP weather-1", "MUSE_CANCEL target-3",
    "MUSE_CANCEL " + " " * 240 + "target-2",
])
def test_model_targets_must_use_a_supplied_alias(fake_classifier, output):
    defaults, cases = load_cases()
    case = next(case for case in cases if case["id"] == "T05")
    fake_classifier.output = output
    report = measure(defaults, [case], fake_classifier.classify)
    assert report["rounds"][0]["raw_model"]["errors"] == [{
        "id": "T05", "expected": {"kind": "MUSE_CANCEL", "target": "recipe-1"},
        "actual": {"kind": "INVALID"},
    }]


def test_operator_entrypoint_prints_no_model_text_on_failure(monkeypatch, capsys):
    def failed(_):
        raise RuntimeError("private sentinel")
    monkeypatch.setattr("bench.arbitration_models.load_model", failed)
    with pytest.raises(SystemExit) as result:
        main(["--real-models"])
    assert str(result.value) == "model evaluation failed; no model output or exception content recorded"
    assert "sentinel" not in str(capsys.readouterr())
