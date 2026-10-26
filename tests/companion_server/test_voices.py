import asyncio
import logging

import pytest

from reachy_muse.protocol import VoiceRole
from reachy_muse.companion_server.voice_chatterbox import Chunk
from reachy_muse.companion_server.voice_fallback import FallbackVoice
from reachy_muse.companion_server.voices import parse_voice

RATE = 16_000
TENTH = RATE // 10


@pytest.mark.parametrize("spec, parsed", [
    ("chatterbox-turbo", ("chatterbox-turbo", "builtin")),
    ("chatterbox-turbo:builtin", ("chatterbox-turbo", "builtin")),
    ("kokoro", ("kokoro", "af_heart")),
    ("kokoro:af_heart", ("kokoro", "af_heart")),
    ("kokoro:bm_george", ("kokoro", "bm_george")),
])
def test_a_voice_is_an_engine_and_optionally_one_of_its_voices(spec, parsed):
    assert parse_voice(spec) == parsed


@pytest.mark.parametrize("spec, message", [
    ("af_heart", "unknown voice engine 'af_heart'; choose from chatterbox-turbo, kokoro"),
    ("piper:amy", "unknown voice engine 'piper'; choose from chatterbox-turbo, kokoro"),
    ("chatterbox-turbo:af_heart", "chatterbox-turbo has only the voice builtin, not 'af_heart'"),
    ("chatterbox-turbo:my_recording", "chatterbox-turbo has only the voice builtin, not 'my_recording'"),
])
def test_unknown_engines_and_any_chatterbox_voice_but_its_own_are_refused(spec, message):
    with pytest.raises(ValueError) as refused:
        parse_voice(spec)
    assert str(refused.value) == message


class StubKokoro:
    name = "kokoro-82m"

    def __init__(self):
        self.spoken = []

    async def load(self):
        pass

    async def speak(self, text, role, out_rate, expression=None):
        self.spoken.append(text)
        for sentence in text.split(". "):
            yield b"\x02\x00" * TENTH


class StubChatterbox:
    name = "chatterbox-turbo"

    def __init__(self, *, fail_load=False, stall_at=None, pause_s=0.0):
        self.fail_load = fail_load
        self.stall_at = stall_at
        self.pause_s = pause_s

    async def load(self):
        if self.fail_load:
            raise RuntimeError("no weights")

    async def stream(self, lines, out_rate, expression=None):
        for index, _ in enumerate(lines):
            if index == self.stall_at:
                await asyncio.Event().wait()
            yield Chunk(index, b"\x01\x00" * TENTH, False)
            await asyncio.sleep(self.pause_s)
            yield Chunk(index, b"\x01\x00" * TENTH, True)


def spoken(voice, *texts):
    async def run():
        await voice.load()
        return [b"".join([chunk async for chunk in voice.speak(text, VoiceRole.MAIN, RATE)]) for text in texts]

    return asyncio.run(asyncio.wait_for(run(), 5))


def warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.parametrize("problem", ["timeout", "failure", "underrun", "load_failure"])
def test_voice_warnings_never_include_private_speech_or_exception_text(caplog, problem):
    class Primary(StubChatterbox):
        async def load(self):
            if problem == "load_failure":
                raise RuntimeError("SENSITIVE_SENTINEL")

        async def stream(self, lines, out_rate, expression=None):
            if problem == "timeout":
                await asyncio.Event().wait()
            if problem == "failure":
                raise RuntimeError("SENSITIVE_SENTINEL")
            async for chunk in super().stream(lines, out_rate, expression):
                yield chunk

    voice = FallbackVoice(Primary(pause_s=.3), StubKokoro())
    if problem == "timeout":
        voice.deadline_s = .01
    spoken(voice, "SENSITIVE_SENTINEL is private.")
    assert warnings(caplog)
    assert "SENSITIVE_SENTINEL" not in caplog.text


@pytest.mark.parametrize("ending", ["stall", "eof"])
def test_a_primary_that_stops_mid_sentence_finishes_with_fallback(ending):
    closed = []

    class Primary(StubChatterbox):
        async def stream(self, lines, out_rate, expression=None):
            try:
                yield Chunk(0, b"\x01\x00" * TENTH, False)
                if ending == "stall":
                    await asyncio.Event().wait()
            finally:
                closed.append(True)

    kokoro = StubKokoro()
    voice = FallbackVoice(Primary(), kokoro)
    voice.deadline_s = .03
    voice.prebuffer_s = .005

    async def run():
        return b"".join([chunk async for chunk in voice.speak("Finish this sentence.", VoiceRole.MAIN, RATE)])

    assert asyncio.run(asyncio.wait_for(run(), .5)) == b"\x01\x00" * TENTH + b"\x02\x00" * TENTH
    assert kokoro.spoken == ["Finish this sentence."]
    assert closed == [True]


def test_slow_playback_does_not_timeout_audio_already_in_the_queue():
    kokoro = StubKokoro()
    voice = FallbackVoice(StubChatterbox(), kokoro)
    voice.deadline_s = .01
    voice.prebuffer_s = 0

    async def run():
        chunks = []
        async for chunk in voice.speak("One sentence.", VoiceRole.MAIN, RATE):
            chunks.append(chunk)
            await asyncio.sleep(.03)
        return b"".join(chunks)

    assert asyncio.run(run()) == b"\x01\x00" * (2 * TENTH)
    assert kokoro.spoken == []


def test_when_chatterbox_fails_to_load_kokoro_speaks_every_line(caplog):
    kokoro = StubKokoro()
    voice = FallbackVoice(StubChatterbox(fail_load=True), kokoro)
    assert spoken(voice, "First. Second.", "Third.") == [b"\x02\x00" * (2 * TENTH), b"\x02\x00" * TENTH]
    assert voice.name == "kokoro-82m"
    assert kokoro.spoken == ["First. Second.", "Third."]
    assert warnings(caplog) == ["voice chatterbox-turbo failed to load (RuntimeError); kokoro-82m speaks every line"]


def test_a_sentence_chatterbox_cannot_start_in_time_and_the_rest_of_the_line_go_to_kokoro(caplog):
    kokoro = StubKokoro()
    voice = FallbackVoice(StubChatterbox(stall_at=1), kokoro)
    voice.deadline_s = 0.1
    assert spoken(voice, "First. Second. Third.") == [b"\x01\x00" * (2 * TENTH) + b"\x02\x00" * (2 * TENTH)]
    assert kokoro.spoken == ["Second. Third."]
    assert warnings(caplog) == ["voice chatterbox-turbo: sentence 2 had no audio after 0.1 s; kokoro-82m speaks the rest"]


def test_a_chunk_that_arrives_after_the_robot_ran_out_is_logged_as_an_underrun(caplog):
    voice = FallbackVoice(StubChatterbox(pause_s=0.3), StubKokoro())
    assert spoken(voice, "Only one sentence here.") == [b"\x01\x00" * (2 * TENTH)]
    [message] = warnings(caplog)
    gap_ms, rest = message.removeprefix("voice underrun of ").split(" ", 1)
    assert 45 <= int(gap_ms) <= 90 and rest == "ms in sentence 1"


def test_holding_the_first_audio_covers_a_short_wait_for_the_next_chunk(caplog):
    voice = FallbackVoice(StubChatterbox(pause_s=0.2), StubKokoro())
    assert spoken(voice, "Only one sentence here.") == [b"\x01\x00" * (2 * TENTH)]
    assert warnings(caplog) == []


def test_serve_refuses_a_voice_it_does_not_have(capsys):
    from reachy_muse.companion_server import cli

    with pytest.raises(SystemExit) as refused:
        cli.main(["serve", "--voice", "chatterbox-turbo:someone"])
    assert refused.value.code == 2
    assert "argument --voice: chatterbox-turbo has only the voice builtin, not 'someone'" in capsys.readouterr().err
