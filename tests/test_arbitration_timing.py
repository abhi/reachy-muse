import json

from bench.arbitration_eval import RecordedModel, load_cases
from bench.arbitration_timing import journal_profile, simulate


def test_timing_replay_waits_for_late_models_and_preserves_route_effects():
    defaults, cases = load_cases()
    cases = [case for case in cases if case["id"] in ("E10", "E17", "E21", "E31", "E55", "E56")]
    report = simulate(defaults, cases, {"hearing_ms": 600, "classify_ms": 70, "voice_ms": 80},
                      [{"dispatch_ms": 32, "first_text_ms": 4000, "first_audio_ms": 5000}], RecordedModel())
    assert report["decisions"]["accuracy"] == {"correct": 6, "total": 6, "rate": 1.0}
    assert report["transcript_fidelity"] == {"correct": 2, "total": 2}
    assert report["model_outcomes"] == {"rule": 5, "deadline": 1}
    paths = report["paths"]
    assert paths["MUSE"]["after_final_ms"]["p50"] == 70
    assert paths["MUSE"]["after_speech_end_ms"]["p50"] == 670
    assert paths["MUSE"]["dispatch_ms"]["p50"] == 702
    assert paths["MUSE"]["first_text_ms"]["p50"] == 4702
    assert paths["MUSE"]["first_answer_audio_ms"]["p50"] == 5702
    assert paths["LOCAL_REPLY"]["after_final_ms"]["p50"] == 0
    assert paths["LOCAL_REPLY"]["first_answer_audio_ms"]["p50"] == 680
    assert paths["MUSE_FOLLOW_UP"]["first_answer_audio_ms"]["p50"] == 5632
    for route in ("LOCAL_DEVICE", "MUSE_CANCEL", "IGNORE"):
        assert paths[route]["after_speech_end_ms"]["p50"] == 600
        assert paths[route]["first_answer_audio_ms"] == {"n": 0}
    assert "Good morning" not in json.dumps(report)


def test_missing_remote_audio_and_disconnected_muse_do_not_report_zero_latency():
    defaults, cases = load_cases()
    case = next(case for case in cases if case["id"] == "E21")
    delays = {"hearing_ms": 600, "classify_ms": 70, "voice_ms": 80}
    report = simulate(defaults, [case], delays, [{"dispatch_ms": 32, "first_text_ms": 4000}])
    assert report["paths"]["MUSE"]["first_answer_audio_ms"] == {"n": 0}
    case["state"]["muse_ready"] = False
    report = simulate(defaults, [case], delays, [{"dispatch_ms": 32, "first_audio_ms": 5000}])
    assert report["paths"]["MUSE"]["dispatch_ms"] == {"n": 0}
    assert report["paths"]["MUSE"]["first_answer_audio_ms"] == {"n": 0}
    assert report["paths"]["MUSE"]["after_speech_end_ms"]["p50"] == 600


def test_journal_extraction_keeps_only_numeric_timings_and_source_references(tmp_path):
    path = tmp_path / "journal.txt"
    path.write_text(
        "2026-10-30 07:04:00,000 private user message\n"
        "2026-10-30 07:04:11,000 INFO Reachy turn timeline speech_end=1 endpoint=2 "
        "transcript=3 dispatch=3.032 first_text=7 first_audio=8 secret=private-sentinel\n")
    result = journal_profile(path)
    assert result["samples"] == [{"line": 2, "kind": "Muse question", "hearing_ms": 2000,
                                  "dispatch_ms": 32, "first_text_ms": 3968, "first_audio_ms": 4968}]
    assert "private" not in json.dumps(result)
    assert "secret" not in json.dumps(result)
