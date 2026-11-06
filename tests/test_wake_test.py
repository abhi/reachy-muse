import asyncio

import numpy as np
import pytest

from reachy_muse.wake_policy import WakePolicy
from reachy_muse.wake_test import Trial, measure, summarize
from test_wake_policy import Detector


def test_trial_denominators_and_scores_have_literal_expectations():
    trials = [Trial(True, 5, 1, True, ()), Trial(True, 5, 0, True, ()),
              Trial(False, 10, 1, True, ()), Trial(False, 10, 0, True, ()),
              Trial(False, 100, 50, False, ())]
    report = summarize(trials)
    assert (report["recall"], report["negative_audio_s"], report["false_triggers_per_hour"],
            report["invalid_trials"]) == (.5, 20, 180, 1)
    assert summarize([])["false_triggers_per_hour"] is None
    assert summarize([Trial(False, 3600, 0, True, ())])["zero_event_95pct_upper_per_hour"] == pytest.approx(2.995732)


def test_trial_with_capture_gap_is_invalid_and_counts_only_audio():
    async def run():
        async def chunks():
            yield np.zeros(16000, np.float32), False
            yield np.zeros(16000, np.float32), True
        detector = Detector()
        result = await measure(detector, WakePolicy(), chunks(), positive=False)
        assert (result.valid, result.audio_s, result.triggers, detector.resets) == (False, 2.0, 0, 2)
    asyncio.run(run())


def test_independent_trials_do_not_inherit_cooldown():
    async def run():
        async def chunks():
            yield np.full(160, .1), False
        policy, detector = WakePolicy(), Detector()
        first = await measure(detector, policy, chunks(), positive=True)
        second = await measure(detector, policy, chunks(), positive=True)
        assert (first.triggers, second.triggers, first.audio_s, second.audio_s) == (1, 1, .01, .01)
    asyncio.run(run())


def test_operator_cli_runs_labelled_trials_without_identity_muse_or_motion(tmp_path, monkeypatch, capsys):
    import json
    from reachy_muse import cli, wake_test

    events, prompts, options = [], [], {}
    detector = Detector()
    detector.close = lambda: events.append("detector_closed")

    class Hardware:
        def __init__(self, **kwargs):
            options.update(kwargs)

        def start(self):
            events.append("hardware_started")

        def close(self):
            events.append("hardware_closed")

        def read_audio(self):
            return None

    values = iter((.1, 0, .1, 0))

    async def chunks(hardware, seconds):
        assert seconds == 1
        yield np.full(16000, next(values), np.float32), False

    monkeypatch.setattr("reachy_muse.hardware.ReachyController", Hardware)
    monkeypatch.setattr("reachy_muse.wake_word.WakeWordDetector", lambda *args, **kwargs: detector)
    monkeypatch.setattr(wake_test, "microphone_chunks", chunks)
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "")
    monkeypatch.setattr(cli.identity, "load_or_create", lambda: pytest.fail("wake-test must not pair or connect to Muse"))
    report_path = tmp_path / "report.json"
    assert cli.main(["--state-dir", str(tmp_path), "wake-test", "--count", "2", "--seconds", "1",
                     "--report", str(report_path), "--record-dir", str(tmp_path / "recordings")]) == 0
    report = json.loads(report_path.read_text())
    assert (report["recall"], report["false_triggers_per_hour"], report["negative_audio_s"]) == (.5, 1800, 2)
    assert len(prompts) == 4
    assert "0.5 m" in prompts[0] and "1.5 m" in prompts[1]
    assert "without the wake phrase" in prompts[2] and "silent" in prompts[3]
    assert (options["enable_motion"], options["sleep_on_close"]) == (False, False)
    assert events == ["hardware_started", "hardware_closed", "detector_closed"]
    assert len(json.loads((tmp_path / "recordings/manifest.json").read_text())) == 4
    assert '"score": null' in capsys.readouterr().out
