"""The companion labels each request's voice against the owner's print. Fake encoder, no models."""

import asyncio
import json
import logging
import os
import shutil
import stat
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from websockets.asyncio.client import connect

from reachy_muse.companion_server import cli, hearing_voxtral, owner_voice as owner_voice_module, voice_control
from reachy_muse.companion_server.engines import EchoNarrator, ScriptedHearing
from reachy_muse.companion_server.owner_voice import PRINT_FILE, OwnerVoice, VoicePrint
from reachy_muse.companion_server.server import CompanionServer, Engines
from reachy_muse.companion_server.watchdog import EngineExecutor
from reachy_muse.protocol import (
    AudioKind, CloseHow, Error, ErrorCode, HearClose, HearEndpoint, HearOpen, HearPartial, Hello, Narrate,
    NarratedLine, NarrateOp, NarrateResult, Op, Sender, SpeakerLabel, SpeakerObservation, Welcome, decode, encode,
    encode_audio,
)

TOKEN = "test"
MODEL = "a" * 64
# Each fake voice is one constant sample level, and the encoder turns a window's level into its embedding.
OWNER, OWNER_LIKE, OTHER, SECRET = 1000, 1200, 3000, 2718
VOICES = {OWNER: (1.0, 0.0, 0.0), OWNER_LIKE: (0.9, 0.43589, 0.0), OTHER: (0.2, 0.0, 0.9798),
          SECRET: (0.6180339887, 0.7861513778, 0.0)}
FRAME = b"\0\0" * 320
WORDS = "my secret pin is 4417"


class Encoder:
    """Stands in for voice_verify.SpeakerEncoder. ``gate`` holds every embedding until it is set."""

    model_id = MODEL
    dimension = 3

    def __init__(self, gate=None, failure=None):
        self.gate = gate
        self.failure = failure
        self.windows = []
        self.entered = threading.Event()

    def embed(self, samples, rate):
        self.windows.append(len(samples))
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(5)
        if self.failure is not None:
            raise self.failure
        return np.array(VOICES[round(float(np.mean(np.abs(samples))) * 32768)])


class Vad:
    def is_speech(self, pcm, rate):
        return any(pcm)


def voice(level, seconds):
    return [np.full(320, level, dtype="<i2").tobytes()] * round(seconds * 50)


def silence(seconds):
    return [FRAME] * round(seconds * 50)


@pytest.fixture
def state(tmp_path):
    return tmp_path


def enrolled(state, embedding=VOICES[OWNER]):
    VoicePrint(MODEL, embedding).save(state / PRINT_FILE)


@pytest.fixture
def speaker_thread():
    with EngineExecutor("speaker") as thread:
        yield thread


def owner_voice(state, thread, encoder=None, threshold=0.74):
    return OwnerVoice(encoder or Encoder(), state / PRINT_FILE, threshold, name="fake-speaker", vad=Vad, thread=thread)


def serve(script, speaker, *, hearing=None, control=None):
    """The real server, and with ``control`` the owner's enrollment socket in that state directory."""
    async def run():
        engines = Engines(hearing=hearing or ScriptedHearing((WORDS,)), narrator=EchoNarrator(), speaker=speaker)
        async with CompanionServer(engines, TOKEN) as companion:
            if control is None:
                await script(companion)
                return
            async with voice_control.serving(speaker, control):
                await script(companion)

    asyncio.run(asyncio.wait_for(run(), 10))


async def robot(companion, ops=(Op.HEAR, Op.NARRATE, Op.SPEAKER)):
    ws = await connect(companion.url, proxy=None)
    await ws.send(encode(Hello((1,), TOKEN, ops, 16000, "test")))
    welcome = await recv(ws)
    assert isinstance(welcome, Welcome), welcome
    return ws


async def recv(ws, timeout=2):
    return decode(await asyncio.wait_for(ws.recv(), timeout), sender=Sender.COMPANION)


async def say(ws, stream, frames, how=CloseHow.FINISH):
    await ws.send(encode(HearOpen(stream, 0)))
    for seq, pcm in enumerate(frames):
        await ws.send(encode_audio(AudioKind.MIC, stream, seq, pcm))
    await ws.send(encode(HearClose(stream, how)))


async def heard(ws, stream):
    """Every message under ``stream``, through its endpoint and the one message after it."""
    messages = []
    while HearEndpoint not in map(type, messages[:-1]):
        message = await recv(ws)
        if getattr(message, "stream", None) == stream:
            messages.append(message)
    return messages


async def nothing_else_arrived(ws):
    await ws.send(encode(Narrate(id=999, op=NarrateOp.ACKNOWLEDGE, request="Barrier.")))
    assert await recv(ws) == NarrateResult(999, (NarratedLine("Barrier.", None),))


def speaker_lines(caplog):
    return [message for message in caplog.messages if message.startswith("speaker ")]


def test_an_owner_like_voice_is_owner_and_an_other_like_voice_is_other(state, speaker_thread, caplog):
    caplog.set_level(logging.INFO, logger="reachy_muse.companion_server")
    enrolled(state)
    encoder = Encoder()

    async def script(companion):
        ws = await robot(companion)
        await say(ws, 1, silence(.5) + voice(OWNER_LIKE, 2) + silence(.3))
        assert await heard(ws, 1) == [HearPartial(1, 0, WORDS), HearEndpoint(1, WORDS, True),
                                      SpeakerObservation(1, 8000, 40000, SpeakerLabel.OWNER)]
        await say(ws, 3, voice(OTHER, 1.2))
        assert (await heard(ws, 3))[-2:] == [HearEndpoint(3, WORDS, True), SpeakerObservation(3, 0, 19200, SpeakerLabel.OTHER)]
        await nothing_else_arrived(ws)

    serve(script, owner_voice(state, speaker_thread, encoder))
    assert speaker_lines(caplog) == ["speaker stream=1 voiced_s=2.00 score=0.9000 label=owner",
                                     "speaker stream=3 voiced_s=1.20 score=0.2000 label=other"]
    assert encoder.windows == [16000, 16000, 19200]


@pytest.mark.parametrize("seconds, windows", [(1.0, [16000]), (1.6, [24000]), (2.5, [20000, 20000]),
                                              (7.0, [24000] * 4)])
def test_a_request_is_judged_on_up_to_four_windows_of_its_speech(state, speaker_thread, seconds, windows):
    enrolled(state)
    encoder = Encoder()

    async def script(companion):
        ws = await robot(companion)
        await say(ws, 1, voice(OWNER, seconds))
        assert (await heard(ws, 1))[-1] == SpeakerObservation(1, 0, sum(windows), SpeakerLabel.OWNER)

    serve(script, owner_voice(state, speaker_thread, encoder))
    assert encoder.windows == windows


def test_without_a_print_every_voice_is_unknown_and_no_embedding_is_made(state, speaker_thread, caplog):
    caplog.set_level(logging.INFO, logger="reachy_muse.companion_server")
    encoder = Encoder()

    async def script(companion):
        ws = await robot(companion)
        await say(ws, 1, voice(OWNER, 2))
        assert (await heard(ws, 1))[-1] == SpeakerObservation(1, 0, 32000, SpeakerLabel.UNKNOWN)

    serve(script, owner_voice(state, speaker_thread, encoder))
    assert speaker_lines(caplog) == ["speaker stream=1 voiced_s=2.00 score=none label=unknown"]
    assert encoder.windows == []


def test_less_than_a_second_of_speech_is_unknown(state, speaker_thread, caplog):
    caplog.set_level(logging.INFO, logger="reachy_muse.companion_server")
    enrolled(state)
    encoder = Encoder()

    async def script(companion):
        ws = await robot(companion)
        await say(ws, 1, silence(1) + voice(OWNER, .98) + silence(1))
        assert (await heard(ws, 1))[-1] == SpeakerObservation(1, 0, 47680, SpeakerLabel.UNKNOWN)

    serve(script, owner_voice(state, speaker_thread, encoder))
    assert speaker_lines(caplog) == ["speaker stream=1 voiced_s=0.98 score=none label=unknown"]
    assert encoder.windows == []


def test_an_old_robot_that_never_asks_gets_no_label_and_costs_no_check(state, speaker_thread):
    enrolled(state)
    encoder = Encoder()

    async def script(companion):
        ws = await connect(companion.url, proxy=None)
        await ws.send(encode(Hello((1,), TOKEN, (Op.HEAR, Op.NARRATE), 16000, "older robot")))
        assert await recv(ws) == Welcome(1, (Op.HEAR, Op.NARRATE), (("hear", "scripted"), ("narrate", "echo")))
        await say(ws, 1, voice(OWNER, 2))
        assert [await recv(ws), await recv(ws)] == [HearPartial(1, 0, WORDS), HearEndpoint(1, WORDS, True)]
        await asyncio.sleep(.1)
        await nothing_else_arrived(ws)

    serve(script, owner_voice(state, speaker_thread, encoder))
    assert encoder.windows == []


def test_a_robot_that_asks_is_granted_speaker_with_the_model_name(state, speaker_thread):
    async def script(companion):
        ws = await connect(companion.url, proxy=None)
        await ws.send(encode(Hello((1,), TOKEN, (Op.HEAR, Op.SPEAKER), 16000, "test")))
        assert await recv(ws) == Welcome(1, (Op.HEAR, Op.SPEAKER), (("hear", "scripted"), ("speaker", "fake-speaker")))

    serve(script, owner_voice(state, speaker_thread))


def test_hearing_runs_on_while_a_slow_speaker_check_holds_its_own_thread(state, speaker_thread):
    """The real hearing worker transcribes the next request while the speaker thread is still busy."""
    enrolled(state)
    gate = threading.Event()
    encoder = Encoder(gate)

    class Stream:
        done = False

        def feed(self, samples):
            pass

        def step(self, *, max_decode_tokens):
            return []

        def close(self):
            self.done = True

    hearing = hearing_voxtral.VoxtralHearing(hearing_voxtral.HearingConfig(final_pass=False))
    hearing._models = SimpleNamespace(
        voxtral=SimpleNamespace(create_streaming_session=lambda **kwargs: Stream()),
        vad=SimpleNamespace(run=lambda *args: (np.array([[0.0]]), np.zeros((2, 1, 128), np.float32))))

    async def script(companion):
        ws = await robot(companion)
        await say(ws, 1, voice(OWNER, 2))
        assert await recv(ws) == HearEndpoint(1, "", True)
        assert await asyncio.to_thread(encoder.entered.wait, 2)
        await say(ws, 3, voice(OTHER, 2))
        assert await recv(ws) == HearEndpoint(3, "", True)
        assert not gate.is_set()
        gate.set()
        assert [await recv(ws), await recv(ws)] == [SpeakerObservation(1, 0, 32000, SpeakerLabel.OWNER),
                                                    SpeakerObservation(3, 0, 32000, SpeakerLabel.OTHER)]

    try:
        serve(script, owner_voice(state, speaker_thread, encoder), hearing=hearing)
    finally:
        gate.set()
        hearing._thread.shutdown()


@pytest.mark.parametrize("finished", [True, False], ids=["after_its_endpoint", "before_its_endpoint"])
def test_an_aborted_stream_gets_no_label(state, speaker_thread, finished):
    enrolled(state)
    gate = threading.Event()
    encoder = Encoder(gate)

    async def script(companion):
        ws = await robot(companion)
        await say(ws, 1, voice(OWNER, 2), how=CloseHow.FINISH if finished else CloseHow.ABORT)
        if finished:
            assert await recv(ws) == HearEndpoint(1, "", True)
            assert await asyncio.to_thread(encoder.entered.wait, 2)
            await ws.send(encode(HearClose(1, CloseHow.ABORT)))
        await nothing_else_arrived(ws)
        gate.set()
        await asyncio.sleep(.1)
        await nothing_else_arrived(ws)

    try:
        serve(script, owner_voice(state, speaker_thread, encoder), hearing=ScriptedHearing(()))
    finally:
        gate.set()
    assert len(encoder.windows) == (2 if finished else 0)


def test_a_reopened_stream_id_never_gets_the_old_streams_label(state, speaker_thread):
    enrolled(state)
    gate = threading.Event()
    encoder = Encoder(gate)

    async def script(companion):
        ws = await robot(companion)
        await say(ws, 1, voice(OTHER, 2))
        assert (await recv(ws), await recv(ws)) == (HearPartial(1, 0, WORDS), HearEndpoint(1, WORDS, True))
        assert await asyncio.to_thread(encoder.entered.wait, 2)
        await say(ws, 1, voice(OWNER, 2))
        assert await recv(ws) == HearPartial(1, 0, WORDS)
        gate.set()
        assert await heard(ws, 1) == [HearEndpoint(1, WORDS, True), SpeakerObservation(1, 0, 32000, SpeakerLabel.OWNER)]
        await nothing_else_arrived(ws)

    try:
        serve(script, owner_voice(state, speaker_thread, encoder))
    finally:
        gate.set()


def test_a_print_forgotten_while_a_request_is_scored_gives_unknown(state, speaker_thread):
    enrolled(state)
    gate = threading.Event()
    encoder = Encoder(gate)

    async def script(companion):
        ws = await robot(companion)
        await say(ws, 1, voice(OWNER, 2))
        assert await asyncio.to_thread(encoder.entered.wait, 2)
        (state / PRINT_FILE).unlink()
        gate.set()
        assert (await heard(ws, 1))[-1] == SpeakerObservation(1, 0, 32000, SpeakerLabel.UNKNOWN)

    try:
        serve(script, owner_voice(state, speaker_thread, encoder))
    finally:
        gate.set()


def test_a_failing_check_is_unknown_and_keeps_the_link_without_logging_its_detail(state, speaker_thread, caplog):
    enrolled(state)
    encoder = Encoder(failure=RuntimeError(f"tensor detail {WORDS}"))

    async def script(companion):
        ws = await robot(companion)
        await say(ws, 1, voice(OWNER, 2))
        assert (await heard(ws, 1))[-1] == SpeakerObservation(1, 0, 32000, SpeakerLabel.UNKNOWN)
        await nothing_else_arrived(ws)

    serve(script, owner_voice(state, speaker_thread, encoder))
    assert "speaker check failed (stream=1): RuntimeError" in caplog.messages
    assert WORDS not in caplog.text


def test_no_log_holds_the_words_the_audio_or_an_embedding(state, speaker_thread, caplog, capfd):
    caplog.set_level(logging.INFO)
    enrolled(state, VOICES[SECRET])

    async def script(companion):
        ws = await robot(companion)
        await say(ws, 1, voice(SECRET, 2))
        assert (await heard(ws, 1))[-1] == SpeakerObservation(1, 0, 32000, SpeakerLabel.OWNER)

    serve(script, owner_voice(state, speaker_thread))
    logged = caplog.text + "".join(capfd.readouterr())
    assert "speaker stream=1 voiced_s=2.00 score=1.0000 label=owner" in caplog.messages
    for private in (WORDS, "6180339887", "7861513778", repr(np.full(320, SECRET, dtype="<i2").tobytes()[:8])[2:-1]):
        assert private not in logged


@pytest.mark.parametrize("damage, warning", [
    ("mode", "ignoring the voice print: it must be a private mode-600 file; run reachy-companion enroll-voice again"),
    ("symlink", "ignoring the voice print: it cannot be opened; run reachy-companion enroll-voice again"),
    ("corrupt", "ignoring the voice print: it is unreadable; run reachy-companion enroll-voice again"),
    ("model", "ignoring the voice print: another speaker model made it; run reachy-companion enroll-voice again"),
])
def test_a_print_that_cannot_be_trusted_is_ignored_once_and_labels_unknown(state, speaker_thread, caplog, damage, warning):
    path = state / PRINT_FILE
    if damage == "model":
        VoicePrint("b" * 64, VOICES[OWNER]).save(path)
    else:
        enrolled(state)
    if damage == "mode":
        path.chmod(0o644)
    elif damage == "symlink":
        path.rename(state / "elsewhere.json")
        path.symlink_to(state / "elsewhere.json")
    elif damage == "corrupt":
        path.write_text('{"version": 1, "model_sha256": "' + MODEL + '", "embedding": [0, 0, 0]}')

    async def script(companion):
        ws = await robot(companion)
        for stream in (1, 3):
            await say(ws, stream, voice(OWNER, 2))
            assert (await heard(ws, stream))[-1] == SpeakerObservation(stream, 0, 32000, SpeakerLabel.UNKNOWN)

    serve(script, owner_voice(state, speaker_thread))
    assert [message for message in caplog.messages if message.startswith("ignoring")] == [warning]


def test_a_print_that_appears_or_is_repaired_while_serving_is_used_from_the_next_request(state, speaker_thread):
    path = state / PRINT_FILE
    before = [lambda: None, lambda: enrolled(state), lambda: path.chmod(0o644), lambda: path.chmod(0o600)]

    async def script(companion):
        ws = await robot(companion)
        labels = []
        for stream, change in zip((1, 3, 5, 7), before):
            change()
            await say(ws, stream, voice(OWNER, 2))
            labels.append((await heard(ws, stream))[-1].label)
        assert labels == [SpeakerLabel.UNKNOWN, SpeakerLabel.OWNER, SpeakerLabel.UNKNOWN, SpeakerLabel.OWNER]

    serve(script, owner_voice(state, speaker_thread))


def test_the_print_is_private_json_with_the_model_and_its_unit_embedding(state):
    enrolled(state)
    path = state / PRINT_FILE
    assert (os.stat(path).st_mode & 0o777, json.loads(path.read_text())) == (
        0o600, {"version": 1, "model_sha256": MODEL, "embedding": [1.0, 0.0, 0.0]})
    assert VoicePrint.load(path) == VoicePrint(MODEL, (1.0, 0.0, 0.0))


def fake_sherpa(monkeypatch):
    """The robot's SpeakerEncoder, loaded through a sherpa_onnx stand-in."""

    class Config:
        def __init__(self, **options):
            pass

        def validate(self):
            return True

    class Extractor:
        dim = 3

        def __init__(self, config):
            pass

        def create_stream(self):
            return SimpleNamespace(accept_waveform=lambda **kw: None, input_finished=lambda: None)

        def is_ready(self, stream):
            return True

        def compute(self, stream):
            return [0.0, 3.0, 4.0]

    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(
        SpeakerEmbeddingExtractorConfig=Config, SpeakerEmbeddingExtractor=Extractor))


def test_serve_loads_the_speaker_model_with_the_calibrated_threshold_only_for_its_model(tmp_path, monkeypatch, caplog):
    from reachy_muse import voice_verify

    caplog.set_level(logging.INFO, logger="reachy_muse.companion_server")
    fake_sherpa(monkeypatch)
    model = tmp_path / "wespeaker_en_voxceleb_resnet34.onnx"
    model.write_bytes(b"test model")
    monkeypatch.setattr(voice_verify, "CALIBRATED_MODEL_SHA256", voice_verify.hashlib.sha256(b"test model").hexdigest())
    args = SimpleNamespace(speaker_model=model, speaker_threshold=None, state_dir=tmp_path)

    async def loaded():
        voice = await cli._speaker(args)
        voice._thread.shutdown()
        return voice.name, voice.threshold, voice.encoder.dimension

    assert asyncio.run(loaded()) == ("wespeaker_en_voxceleb_resnet34", 0.74, 3)
    assert caplog.messages == [
        "speaker recognition ready: model wespeaker_en_voxceleb_resnet34, threshold 0.74, no voice print yet"]


def test_a_speaker_model_that_will_not_load_leaves_the_companion_serving_without_it(tmp_path, monkeypatch, caplog):
    fake_sherpa(monkeypatch)
    model = tmp_path / "other.onnx"
    model.write_bytes(b"an uncalibrated model")
    args = SimpleNamespace(speaker_model=model, speaker_threshold=None, state_dir=tmp_path)
    assert asyncio.run(cli._speaker(args)) is None
    assert caplog.messages == [
        "speaker recognition is off: ValueError: this speaker model needs --speaker-threshold; see bench/voice_verify.md"]
    missing = SimpleNamespace(speaker_model=tmp_path / "missing.onnx", speaker_threshold=.6, state_dir=tmp_path)
    assert asyncio.run(cli._speaker(missing)) is None
    assert caplog.messages[-1].startswith("speaker recognition is off: FileNotFoundError")


@pytest.mark.parametrize("threshold", ["1", "-1", "nan", "high"])
def test_serve_refuses_a_threshold_outside_cosine_range(threshold, capsys):
    with pytest.raises(SystemExit):
        cli.main(["serve", "--speaker-threshold", threshold])
    assert "--speaker-threshold" in capsys.readouterr().err



# ---- enrolling and forgetting, as the owner does on the Mac


LISTENING = ("Listening for your voice. Say 5 full sentences to Reachy within 3 minutes, each with at least 2 seconds "
             "of speech. Only you should speak. Reachy answers them as usual. Press Ctrl-C to stop.")
NOT_LISTENING = ("No companion is listening for enrollment. Start reachy-companion serve with --speaker-model, "
                 "then try again.")


@pytest.fixture
def private_state():
    """A state directory only this user can enter, with a path short enough for a Unix socket."""
    path = Path(tempfile.mkdtemp(prefix="rc-"))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def command(state, *argv):
    """``reachy-companion --state-dir STATE ARGV``, returning its exit status."""
    try:
        cli.main(["--state-dir", str(state), *argv])
    except SystemExit as exit:
        return exit.code
    return 0


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(.01)


async def enroll_in_background(engine, state):
    enrolling = asyncio.create_task(asyncio.to_thread(command, state, "enroll-voice"))
    await until(lambda: engine.enrolling or enrolling.done())
    return enrolling


async def label(ws, stream, frames):
    await say(ws, stream, frames)
    return (await heard(ws, stream))[-1].label


def test_enroll_voice_takes_five_long_sentences_then_tells_the_owner_from_others(private_state, speaker_thread, capsys,
                                                                                caplog):
    caplog.set_level(logging.INFO, logger="reachy_muse.companion_server")
    engine = owner_voice(private_state, speaker_thread)

    async def script(companion):
        ws = await robot(companion)
        enrolling = await enroll_in_background(engine, private_state)
        assert await label(ws, 1, voice(OWNER, 1.5)) is SpeakerLabel.UNKNOWN
        for stream in (3, 5, 7, 9, 11):
            assert await label(ws, stream, silence(.4) + voice(OWNER, 2.4)) is SpeakerLabel.UNKNOWN
        assert await enrolling == 0
        assert not engine.enrolling
        assert [await label(ws, 13, voice(OWNER_LIKE, 2)), await label(ws, 15, voice(OTHER, 2))] == [
            SpeakerLabel.OWNER, SpeakerLabel.OTHER]

    serve(script, engine, control=private_state)
    assert capsys.readouterr().out.splitlines() == [
        LISTENING,
        "Skipped a sentence with 1.5 s of speech; each needs 2 s.",
        *(f"Heard sentence {n} of 5: 2.4 s of speech." for n in range(1, 6)),
        f"Enrolled your voice from 5 sentences, 12.0 s of speech. The voice print stays on this Mac in "
        f"{private_state / PRINT_FILE}.",
    ]
    assert VoicePrint.load(private_state / PRINT_FILE) == VoicePrint(MODEL, VOICES[OWNER])
    assert "voice print enrolled from 5 utterances with 12.0 s of speech" in caplog.messages


def test_sentences_that_do_not_sound_alike_change_nothing(private_state, speaker_thread, capsys):
    enrolled(private_state, VOICES[OTHER])
    engine = owner_voice(private_state, speaker_thread)

    async def script(companion):
        ws = await robot(companion)
        enrolling = await enroll_in_background(engine, private_state)
        for stream, level in zip((1, 3, 5, 7, 9), (OWNER, OWNER, OTHER, OWNER, OWNER)):
            await label(ws, stream, voice(level, 2.4))
        assert await enrolling == 1

    serve(script, engine, control=private_state)
    assert capsys.readouterr().out.splitlines()[-1] == (
        "Sentence 3 does not sound like the others, so nothing changed. Run enroll-voice again with only you speaking.")
    assert VoicePrint.load(private_state / PRINT_FILE) == VoicePrint(MODEL, VOICES[OTHER])


def test_an_enrollment_that_runs_out_of_time_changes_nothing(private_state, speaker_thread, capsys, monkeypatch):
    monkeypatch.setattr(owner_voice_module, "ENROLL_WITHIN_S", 1.2)
    engine = owner_voice(private_state, speaker_thread)

    async def script(companion):
        ws = await robot(companion)
        enrolling = await enroll_in_background(engine, private_state)
        await label(ws, 1, voice(OWNER, 2.4))
        assert await enrolling == 1
        assert not engine.enrolling
        assert await label(ws, 3, voice(OWNER, 2.4)) is SpeakerLabel.UNKNOWN

    serve(script, engine, control=private_state)
    assert capsys.readouterr().out.splitlines()[1:] == [
        "Heard sentence 1 of 5: 2.4 s of speech.", "Time ran out after 1 of 5 sentences. Nothing changed."]
    assert not (private_state / PRINT_FILE).exists()


def test_only_one_enrollment_runs_and_hanging_up_ends_it(private_state, speaker_thread, capsys):
    engine = owner_voice(private_state, speaker_thread)

    async def script(companion):
        ws = await robot(companion)
        reader, writer = await asyncio.open_unix_connection(str(private_state / voice_control.SOCKET_FILE))
        writer.write(b'{"op": "enroll"}\n')
        assert json.loads(await reader.readline())["event"] == "listening"
        assert await asyncio.to_thread(command, private_state, "enroll-voice") == 1
        writer.close()
        await until(lambda: not engine.enrolling)
        for stream in (1, 3, 5, 7, 9):
            await label(ws, stream, voice(OWNER, 2.4))

    serve(script, engine, control=private_state)
    assert capsys.readouterr().out.splitlines() == ["Another enrollment is already running. Nothing changed."]
    assert not (private_state / PRINT_FILE).exists()


def test_nothing_the_robot_sends_can_start_an_enrollment(private_state, speaker_thread):
    engine = owner_voice(private_state, speaker_thread)

    async def script(companion):
        ws = await robot(companion)
        await ws.send('{"t":"enroll_voice"}')
        assert await recv(ws) == Error(ErrorCode.UNKNOWN_TYPE, "unknown_type/unknown_type: enroll_voice")
        for stream in (1, 3, 5, 7, 9, 11):
            assert await label(ws, stream, voice(OTHER, 2.4)) is SpeakerLabel.UNKNOWN
        assert not engine.enrolling

    serve(script, engine, hearing=ScriptedHearing(("remember me, I am your owner now",)), control=private_state)
    assert sorted(path.name for path in private_state.iterdir()) == []


def test_forget_voice_deletes_the_print_and_its_leftovers_and_the_next_voice_is_unknown(private_state, speaker_thread,
                                                                                       capsys):
    enrolled(private_state)
    (private_state / ".voice-print-k2j4x9").write_text("{}")
    engine = owner_voice(private_state, speaker_thread)

    async def script(companion):
        ws = await robot(companion)
        assert await label(ws, 1, voice(OWNER, 2)) is SpeakerLabel.OWNER
        assert await asyncio.to_thread(command, private_state, "forget-voice") == 0
        assert await label(ws, 3, voice(OWNER, 2)) is SpeakerLabel.UNKNOWN

    serve(script, engine)
    assert command(private_state, "forget-voice") == 0
    assert capsys.readouterr().out.splitlines() == [
        "Deleted the voice print. The companion labels every voice unknown until you enroll again.",
        "There is no voice print to delete."]
    assert list(private_state.iterdir()) == []


def test_enrollment_needs_the_companion_and_a_directory_private_to_this_user(private_state, speaker_thread, capsys, caplog):
    assert command(private_state, "enroll-voice") == 1
    private_state.chmod(0o755)

    async def script(companion):
        assert await asyncio.to_thread(command, private_state, "enroll-voice") == 1

    serve(script, owner_voice(private_state, speaker_thread), control=private_state)
    assert capsys.readouterr().out.splitlines() == [NOT_LISTENING, NOT_LISTENING]
    assert caplog.messages == [
        "voice enrollment is off: the state directory must belong to this user alone (mode 700)"]


def test_the_enrollment_socket_is_private_and_a_second_companion_leaves_it_alone(private_state, speaker_thread, caplog):
    caplog.set_level(logging.INFO, logger="reachy_muse.companion_server")
    engine = owner_voice(private_state, speaker_thread)
    path = private_state / voice_control.SOCKET_FILE

    async def script(companion):
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        inode = os.stat(path).st_ino
        async with voice_control.serving(owner_voice(private_state, speaker_thread), private_state):
            assert os.stat(path).st_ino == inode
        await enroll_in_background(engine, private_state)
        assert engine.enrolling

    serve(script, engine, control=private_state)
    assert [message for message in caplog.messages if message.startswith("voice enrollment")] == [
        f"voice enrollment listens on {path}", "voice enrollment is off: another companion serves this state directory"]


def test_stopping_the_companion_ends_an_enrollment_at_once(private_state, speaker_thread, capsys):
    engine = owner_voice(private_state, speaker_thread)

    async def script(companion):
        await enroll_in_background(engine, private_state)

    serve(script, engine, control=private_state)
    assert not engine.enrolling
    assert capsys.readouterr().out.splitlines() == [
        LISTENING, "The companion stopped before the enrollment finished. Nothing changed."]
