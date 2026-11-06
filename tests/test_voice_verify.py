import asyncio
import json
import logging
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from reachy_muse import cli
from reachy_muse.voice_verify import Decision, Profile, RequestGate, SpeakerEncoder, VoiceVerifier, enroll, rejection_cue


class Encoder:
    model_id = "a" * 64
    dimension = 2

    def embed(self, audio, rate):
        return np.array([0., 1.] if np.max(audio) > .4 else [1., 0.], dtype=np.float32)


class Vad:
    def is_speech(self, pcm, rate):
        return bool(np.any(np.frombuffer(pcm, dtype="<i2")))


def verifier(mode="strict", realtime=False):
    return VoiceVerifier(Encoder(), Profile("a" * 64, (1., 0.), .7), mode, realtime)


@pytest.mark.parametrize("mode,realtime,expected", [
    ("strict", False, Decision(0., False, False)),
    ("soft", False, Decision(0., False, True)),
    ("soft", True, Decision(0., False, False)),
])
def test_policy_has_literal_match_and_impostor_decisions(mode, realtime, expected, caplog):
    subject = verifier(mode, realtime)
    with caplog.at_level(logging.INFO, logger="reachy_muse.voice_verify"):
        assert subject.verify(np.array([.5]), 16000) == expected
        assert subject.verify(np.array([.25]), 16000) == Decision(1., True, True)
    assert [record.getMessage() for record in caplog.records] == [
        "voice_verify score=0.0000 decision=" + ("observe" if mode == "soft" and not realtime else "reject"),
        "voice_verify score=1.0000 decision=accept",
    ]


@pytest.mark.parametrize("output", [[float("nan"), 0], [0, 0], [1, 2, 3]])
def test_invalid_model_output_fails_closed(output, caplog):
    subject = verifier()
    subject.encoder.embed = lambda *args: output
    with caplog.at_level(logging.INFO):
        assert subject.verify(np.ones(16000), 16000) == Decision(None, False, False)
    assert caplog.messages == ["voice_verify score=none decision=reject"]


def test_inference_exception_does_not_log_content(caplog):
    subject = verifier()
    def broken(*args):
        raise RuntimeError("private waveform details")
    subject.encoder.embed = broken
    with caplog.at_level(logging.INFO):
        assert subject.verify(np.ones(16000), 16000) == Decision(None, False, False)
    assert caplog.messages == ["voice_verify score=none decision=reject"]


def test_cosine_threshold_boundary():
    subject = verifier()
    subject.encoder.embed = lambda *args: [.6, .8]
    assert subject.verify([.1], 16000) == Decision(pytest.approx(.6), False, False)
    subject.profile = Profile("a" * 64, (1., 0.), .6)
    assert subject.verify([.1], 16000) == Decision(pytest.approx(.6), True, True)


def test_enrollment_replaces_private_file_and_forget_needs_no_hardware(tmp_path, monkeypatch, capsys):
    profile = enroll(Encoder(), [(np.full(16000, .25), 16000)] * 3, .7)
    path = tmp_path / "voice-profile.json"
    path.write_text("old")
    path.chmod(0o644)
    profile.save(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == {
        "version": 1, "model_sha256": "a" * 64, "embedding": [1., 0.], "threshold": .7}
    assert Profile.load(path) == profile
    monkeypatch.setattr(cli, "_hardware", lambda *a, **k: pytest.fail("no hardware for forgetting"))
    assert cli.main(["--state-dir", str(tmp_path), "enroll-voice", "--forget"]) == 0
    assert not path.exists()
    assert cli.main(["--state-dir", str(tmp_path), "enroll-voice", "--forget"]) == 0
    assert capsys.readouterr().out == "Voice enrollment deleted.\nVoice enrollment deleted.\n"


def test_enrollment_rejects_other_speaker_without_replacing_profile(tmp_path):
    path = tmp_path / "voice-profile.json"
    original = Profile("a" * 64, (1., 0.), .7)
    original.save(path)
    with pytest.raises(ValueError, match="phrases did not match"):
        enroll(Encoder(), [(np.array([v]), 16000) for v in (.25, .25, .5)], .7).save(path)
    assert Profile.load(path) == original


@pytest.mark.parametrize("corruption", ["mode", "symlink", "nan", "version", "empty"])
def test_invalid_profile_rejected(tmp_path, corruption):
    path = tmp_path / "voice-profile.json"
    original = Profile("a" * 64, (1., 0.), .7)
    original.save(path)
    if corruption == "mode":
        path.chmod(0o644)
    elif corruption == "symlink":
        target = tmp_path / "target"
        path.rename(target)
        path.symlink_to(target)
    else:
        data = json.loads(path.read_text())
        data.update({"threshold": float("nan")} if corruption == "nan" else
                    {"version": 2} if corruption == "version" else {"embedding": []})
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Cannot load voice enrollment"):
        Profile.load(path)


def test_changed_model_cannot_reuse_enrollment():
    with pytest.raises(ValueError, match="Voice model changed"):
        VoiceVerifier(Encoder(), Profile("b" * 64, (1., 0.), .7), "strict")


@pytest.mark.parametrize("rate", [16000, 48000])
def test_request_gate_holds_genuine_and_masks_next_speaker_at_input_rate(rate):
    gate = RequestGate(verifier(), rate, vad=Vad())
    held = gate.feed(np.full(rate // 2, .25, np.float32))
    assert set(held) == {0.}
    accepted = gate.feed(np.zeros(rate // 2, np.float32))
    # Resampling filters introduce a few edge samples at 48 kHz.
    assert accepted[rate // 4] == pytest.approx(.25, abs=1e-4)
    assert len(accepted) == pytest.approx(rate, abs=rate // 50)
    rejected = gate.feed(np.concatenate((np.full(rate // 2, .5), np.zeros(rate // 2))).astype(np.float32))
    assert set(rejected) == {0.}
    assert gate.rejected is True


def test_gap_discards_pending_speech_and_soft_chat_passes_original_audio():
    gate = RequestGate(verifier(), 16000, vad=Vad())
    gate.feed(np.full(6400, .25, np.float32))
    gate.reset()
    assert set(gate.feed(np.zeros(6400, np.float32))) == {0.}
    soft = RequestGate(verifier("soft"), 16000, vad=Vad())
    assert soft.feed(np.full(9600, .5, np.float32)).tolist() == [.5] * 9600


def test_policy_changes_release_held_samples_once_without_replaying_observed_audio():
    subject = verifier("soft", True)
    gate = RequestGate(subject, 16000, vad=Vad())
    assert set(gate.feed(np.full(4817, .25, np.float32))) == {0.}
    subject.realtime = False
    assert gate.feed(np.full(320, .5, np.float32)).tolist() == [.25] * 4817 + [.5] * 320
    subject.realtime = True
    audio = np.concatenate((np.full(8000, .125), np.zeros(4800))).astype(np.float32)
    admitted = gate.feed(audio)
    assert admitted[admitted != 0].tolist() == [.125] * 8000


def test_long_speech_rechecks_every_segment_without_losing_first_samples():
    gate = RequestGate(verifier(), 16000, vad=Vad())
    audio = np.full(320, .25, np.float32)
    released = [gate.feed(audio) for _ in range(500)]
    released.append(gate.feed(np.zeros(4800, np.float32)))
    joined = np.concatenate(released)
    assert joined[joined != 0].tolist() == [.25] * 160000
    assert joined[:640].tolist() == [0.] * 640


@pytest.mark.parametrize("frames", [51, 55, 60, 101, 105])
def test_short_final_words_use_uncommitted_audio_not_a_prior_grant(frames):
    subject = verifier()
    original = subject.encoder.embed
    def require_evidence(audio, rate):
        if len(audio) < 6400 or np.count_nonzero(audio) < 4800:
            raise ValueError("too short")
        return original(audio, rate)
    subject.encoder.embed = require_evidence
    gate = RequestGate(subject, 16000, vad=Vad())
    pieces = [gate.feed(np.full(320, .25, np.float32)) for _ in range(frames)]
    pieces.append(gate.feed(np.zeros(6400, np.float32)))
    audio = np.concatenate(pieces)
    assert audio[audio != 0].tolist() == [.25] * (frames * 320)


def test_pause_near_cut_keeps_enough_voiced_evidence_for_final_word():
    subject = verifier()
    original = subject.encoder.embed
    def require_evidence(audio, rate):
        if np.count_nonzero(audio) < 4800:
            raise ValueError("too little voiced evidence")
        return original(audio, rate)
    subject.encoder.embed = require_evidence
    gate = RequestGate(subject, 16000, vad=Vad())
    chunks = [np.full(320, value, np.float32) for value in
              ([.25] * 36 + [0.] * 13 + [.25] * 2 + [0.] * 15)]
    admitted = np.concatenate([gate.feed(chunk) for chunk in chunks])
    assert admitted[admitted != 0].tolist() == [.25] * 12160


def test_cli_off_needs_no_profile_or_model_and_enabled_requires_wake():
    assert cli.voice_verifier(cli.parser().parse_args(["run"])) is None
    with pytest.raises(ValueError, match="requires --wake-model and --voice-model"):
        cli.voice_verifier(cli.parser().parse_args(["run", "--voice-verify", "strict"]))


def test_enrollment_cli_records_three_phrases_and_closes(tmp_path, monkeypatch, capsys):
    from reachy_muse import voice_verify
    states = []
    robot = SimpleNamespace(sample_rate=16000, start=lambda: states.append("start"),
                            close=lambda: states.append("close"))
    monkeypatch.setattr(cli, "_hardware", lambda *a, **k: robot)
    monkeypatch.setattr(cli, "_record_voice_phrase", lambda r: np.full(16000, .25, np.float32))
    monkeypatch.setattr(voice_verify, "SpeakerEncoder", lambda p: Encoder())
    prompts = []
    monkeypatch.setattr("builtins.input", lambda p: prompts.append(p))
    assert cli.main(["--state-dir", str(tmp_path), "enroll-voice", "--voice-model", "speaker.onnx",
                     "--voice-threshold", ".7"]) == 0
    assert states == ["start", "close"]
    assert len(prompts) == 3
    assert Profile.load(tmp_path / "voice-profile.json").embedding == (1., 0.)
    assert capsys.readouterr().out == "Voice enrolled locally. Recordings were not saved. Enable --voice-verify soft or strict.\n"


def test_rejection_chime_is_local_short_and_quiet():
    audio = rejection_cue(16000)
    assert (len(audio), audio.dtype.name, float(audio[0])) == (2560, "float32", 0.)
    assert float(np.max(np.abs(audio))) == pytest.approx(.04, abs=.001)


def test_encoder_uses_cpu_and_validates_speech_before_inference(tmp_path, monkeypatch):
    configurations = []
    waveforms = []
    class Config:
        def __init__(self, **options):
            configurations.append(options)
        def validate(self):
            return True
    class Extractor:
        dim = 2
        def __init__(self, config):
            pass
        def create_stream(self):
            return SimpleNamespace(accept_waveform=lambda **kw: waveforms.append(kw["waveform"]),
                                   input_finished=lambda: None)
        def is_ready(self, stream):
            return True
        def compute(self, stream):
            return [3., 4.]
    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(
        SpeakerEmbeddingExtractorConfig=Config, SpeakerEmbeddingExtractor=Extractor))
    monkeypatch.setattr("webrtcvad.Vad", lambda mode: Vad())
    model = tmp_path / "speaker.onnx"
    model.write_bytes(b"test model")
    encoder = SpeakerEncoder(model)
    assert encoder.embed(np.full(16000, .25), 16000).tolist() == pytest.approx([.6, .8])
    assert configurations == [{"model": str(model), "num_threads": 1, "provider": "cpu", "debug": False}]
    assert [len(audio) for audio in waveforms] == [16000, 16000]
    with pytest.raises(ValueError, match="at least 0.4"):
        encoder.embed(np.ones(320), 16000)
    with pytest.raises(ValueError, match="0.3 seconds of speech"):
        encoder.embed(np.zeros(16000), 16000)
