"""Every turn logs one timeline line, and tools/turn_timeline.py turns the journal into a table."""

import logging

import pytest

import reachy_muse.turn_timeline
from tools import turn_timeline as tool

np = pytest.importorskip("numpy")
pytest.importorskip("av")


def timeline_lines(caplog):
    return [record.getMessage() for record in caplog.records
            if record.getMessage().startswith("Reachy turn timeline ")]


async def speak_in_real_time(rig, *, wake=False):
    """Microphone chunks at the pace a real microphone delivers them, so wall-clock stage times are real."""
    import asyncio
    from test_pins import CHUNK, VOICE_SAMPLE, WAKE_SAMPLE

    chunks = [np.full(CHUNK, WAKE_SAMPLE, dtype=np.float32)] if wake else []
    chunks += [np.full(CHUNK, VOICE_SAMPLE, dtype=np.float32)] * round(.5 * 16000 / CHUNK)
    chunks += [np.zeros(CHUNK, dtype=np.float32)] * round(.5 * 16000 / CHUNK)
    for chunk in chunks:
        rig.hardware.samples.put(chunk)
        await asyncio.sleep(CHUNK / 16000)


def stages_of(line):
    return [field.split("=")[0] for field in line.split()[3:]]


def test_a_wake_turn_logs_one_line_with_each_stage_it_reached_in_order(monkeypatch, caplog):
    from test_pins import Rig, Transcriber, Wake, run_scenario

    caplog.set_level(logging.INFO)
    rig = Rig(monkeypatch)

    async def muse(session, request):
        await session.answer(request, "Sunny and warm all day. [reachy:happy]")

    service = rig.service(muse, owned=False, speech=rig.voice(), progress_speech=rig.voice("progress_tts"),
                          transcriber=Transcriber("Hey Muse, what's the weather tomorrow?"),
                          wake_detector=Wake(), wake_timeout_s=.3)

    async def user():
        await speak_in_real_time(rig, wake=True)
    run_scenario(rig, service, user)
    lines = timeline_lines(caplog)
    assert len(lines) == 1
    assert stages_of(lines[0]) == ["wake", "speech_end", "endpoint", "transcript", "dispatch", "muse_ack",
                                   "first_text", "reply_done", "narration_done", "first_audio", "forced"]
    values = dict(field.split("=") for field in lines[0].split()[3:])
    assert values["wake"] == "0.000"
    assert values["forced"] == "no"
    seconds = [float(values[stage]) for stage in stages_of(lines[0])[:-1]]
    assert seconds == sorted(seconds)


def test_a_turn_that_recognizes_no_words_still_logs_its_timeline(monkeypatch, caplog):
    from test_pins import Rig, Transcriber, run_scenario

    caplog.set_level(logging.INFO)
    rig = Rig(monkeypatch)

    async def muse(session, request):
        raise AssertionError("an empty transcript must not reach Muse")

    service = rig.service(muse, speech=rig.voice(), progress_speech=rig.voice("progress_tts"),
                          transcriber=Transcriber(""))

    async def user():
        rig.say()
    run_scenario(rig, service, user)
    assert [stages_of(line) for line in timeline_lines(caplog)] == [
        ["speech_end", "endpoint", "transcript", "forced"]]


JOURNAL = """\
2026-10-08 02:32:40,847 INFO reachy_muse.conversation: Reachy heard its wake phrase; microphone open
2026-10-08 02:32:58,195 INFO reachy_muse.turn_timeline: Reachy turn timeline wake=0.000 speech_end=2.950 \
endpoint=3.682 transcript=3.715 dispatch=3.719 muse_ack=4.743 ack_audio=4.721 first_text=15.106 \
reply_done=16.205 narration_done=17.150 first_audio=17.347 forced=no
2026-10-08 02:33:04,608 INFO reachy_muse.turn_timeline: Reachy turn timeline speech_end=0.000 endpoint=0.750 \
transcript=0.760 dispatch=0.770 muse_ack=1.010 first_text=4.140 reply_done=4.670 narration_done=4.700 \
first_audio=5.210 forced=yes
"""


def test_the_tool_tabulates_seconds_since_each_turns_first_stage():
    assert tool.table(tool.parse(JOURNAL.splitlines())) == (
        "    time  wake  speech_end  endpoint  transcript  dispatch  muse_ack  ack_audio  first_text  reply_done"
        "  narration_done  first_audio  total  forced\n"
        "02:32:58  0.00        2.95      3.68        3.71      3.72      4.74       4.72       15.11       16.20"
        "           17.15        17.35  17.35      no\n"
        "02:33:04     -        0.00      0.75        0.76      0.77      1.01          -        4.14        4.67"
        "            4.70         5.21   5.21     yes\n")


def test_the_tool_shows_each_stage_as_time_since_the_previous_one_with_steps():
    rows = tool.table(tool.parse(JOURNAL.splitlines()), steps=True).splitlines()
    assert rows[2].split() == ["02:33:04", "-", "0.00", "0.75", "0.01", "0.01", "0.24", "-", "3.13", "0.53",
                               "0.03", "0.51", "5.21", "yes"]


def test_the_tool_and_the_robot_name_the_same_stages_in_the_same_order():
    assert tool.STAGES == reachy_muse.turn_timeline.STAGES
