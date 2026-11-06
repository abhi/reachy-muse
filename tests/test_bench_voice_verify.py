import pytest

from bench.voice_verify import calibrate, eer, rates


def test_error_rates_count_boundary_acceptances():
    assert rates([.8, .6, .4, .2], [.7, .5, .3, .1], .5) == {"far": .5, "frr": .5}
    assert eer([.8, .6, .4, .2], [.7, .5, .3, .1]) == .5
    assert eer([.9, .8], [.2, .1]) == 0.


def test_calibration_uses_only_calibration_impostors_and_margin():
    assert calibrate([.8, .7], [.21, .35]) == .37
    assert rates([.8, .7], [.21, .35], .37) == {"far": 0., "frr": 0.}


def test_empty_evaluation_is_an_error():
    with pytest.raises(ValueError, match="needs genuine and impostor"):
        rates([], [.2], .5)


def test_benchmark_reports_rejected_request_tail_after_matching_wake(tmp_path, monkeypatch):
    import json
    import numpy as np

    from bench import voice_verify
    from reachy_muse.voice_verify import Profile
    from test_voice_verify import Vad

    class Encoder:
        model_id = "a" * 64
        dimension = 2

        def __init__(self, path):
            pass

        def embed(self, audio, rate):
            score = (.6 if np.any(audio == .375) else .1 if np.any(audio == .25)
                     else .2 if np.any(audio == .5) else 1.)
            return np.array([score, np.sqrt(1 - score ** 2)], dtype=np.float32)

    def load(path):
        prefix = 16000 if "request" in path.stem else 40000
        if not path.name.startswith("Samantha"):
            return np.concatenate((np.full(prefix, .5), np.full(8000, .375))).astype(np.float32)
        return np.concatenate((np.full(prefix, .125), np.full(8000, .25))).astype(np.float32)

    monkeypatch.setattr(voice_verify, "SpeakerEncoder", Encoder)
    monkeypatch.setattr(voice_verify, "load", load)
    monkeypatch.setattr(voice_verify, "enroll", lambda encoder, recordings, threshold:
                        Profile("a" * 64, (1., 0.), threshold))
    monkeypatch.setattr(voice_verify, "conditions", lambda audio, seed: [("clean", audio)])
    monkeypatch.setattr(voice_verify.subprocess, "check_output", lambda *args, **kwargs: "test CPU")
    monkeypatch.setattr("webrtcvad.Vad", lambda mode: Vad())
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    output = tmp_path / "report.json"
    voice_verify.main(["--model", str(model), "--audio-dir", str(tmp_path), "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["threshold"] == .62
    requests = [row for row in report["trials"] if row["split"] == "evaluation"
                and row["genuine"] and row["window"] != "wake"]
    assert [(row["window"], row.get("input_nonzero"), row.get("retained_nonzero"), row.get("complete"))
            for row in requests] == [("request", 24000, 9600, False)] * 2
