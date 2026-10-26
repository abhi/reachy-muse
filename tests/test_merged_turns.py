"""Whole turns where device commands meet the turn timeline, echo dropping, and one-word asides."""

import logging

import pytest

from reachy_muse.contract.prompts import STOP_REQUEST
from test_conversation import sentence_frame
from test_device_command_turns import LongVoice, never_asked, played, rig_with, run, sent_to_muse, service
from test_pins import Transcripts, Wake

np = pytest.importorskip("numpy")


def timeline_lines(caplog):
    return [record.getMessage() for record in caplog.records
            if record.getMessage().startswith("Reachy turn timeline")]


def test_a_device_command_logs_its_timeline_without_any_muse_stage(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="reachy_muse")
    rig = rig_with(monkeypatch)
    conversation = service(rig, never_asked, Transcripts("Hey Muse, go to sleep."),
                           wake_detector=Wake(), wake_timeout_s=30)

    async def user():
        rig.say(wake=True)

    run(rig, conversation, user, rest=("sleeping",))
    [line] = timeline_lines(caplog)
    stages = [field.partition("=")[0] for field in line.removeprefix("Reachy turn timeline ").split()]
    assert stages == ["wake", "speech_end", "endpoint", "transcript", "forced", "command"]
    assert line.endswith(" command=sleep")


def test_a_one_word_goodnight_while_muse_works_is_obeyed_not_dropped_as_an_aside(monkeypatch):
    rig = rig_with(monkeypatch)
    transcripts = Transcripts("Hey Muse, search for the latest robot news.", "Goodnight.")

    async def muse(session, request):
        if request == 1:
            await rig.wait_until_or_timeout(lambda: session.requests == 2, 10)
            await session.answer(request, "Robots learned to fold laundry. [reachy:happy]")
        else:
            await session.answer(request, "Okay, I stopped. [reachy:nod]")

    conversation = service(rig, muse, transcripts, wake_detector=Wake(), wake_timeout_s=30)

    async def user():
        rig.say(wake=True)
        await rig.quiet("thinking")
        rig.say()

    events = run(rig, conversation, user, rest=("sleeping",))
    assert sent_to_muse(events) == ["search for the latest robot news.", STOP_REQUEST]
    assert played(events)[-1] == "tts:Good night."
    assert rig.states()[-1] == "sleeping"


def test_stop_talking_over_a_line_that_says_stop_talking_is_not_taken_for_its_echo(monkeypatch):
    rig = rig_with(monkeypatch, chunk_s=.1)
    rig.hardware.echo_cancelled_input = True
    lines = ["tts:Tell me to stop talking anytime.", "tts:Robots began as toys.", "tts:Now they fold laundry."]

    async def muse(session, request):
        await session.answer(request, "".join(sentence_frame(line[4:]) for line in lines))

    conversation = service(rig, muse, Transcripts("Tell me about robots.", "Stop talking."),
                           voice=LongVoice(rig.recording, chunks=20), stream_replies=True)

    async def user():
        rig.say()
        await rig.wait_until_or_timeout(lambda: {"robot": "play", "audio": lines[0]} in rig.recording.events, 5)
        rig.say()

    events = run(rig, conversation, user)
    assert sent_to_muse(events) == ["Tell me about robots."]
    assert played(events) == lines[:1]
    assert {"robot": "gesture", "name": "nod"} in events
