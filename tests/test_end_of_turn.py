"""Where the robot's voice detector ends a turn, measured on recorded speech."""

import wave
from pathlib import Path

import pytest

from reachy_muse.capabilities import Mode
from reachy_muse.cli import end_of_turn_silence

np = pytest.importorskip("numpy")
pytest.importorskip("av")
webrtcvad = pytest.importorskip("webrtcvad")

SPEECH = Path(__file__).parent / "companion_server" / "data"
RATE = 16000


def speech(name):
    """The recording cut 20 ms after its last voiced frame, so silence can be added exactly."""
    with wave.open(str(SPEECH / f"{name}.wav")) as recording:
        pcm = np.frombuffer(recording.readframes(recording.getnframes()), dtype="<i2")
    vad = webrtcvad.Vad(2)
    voiced = [vad.is_speech(pcm[start:start + 320].tobytes(), RATE) for start in range(0, len(pcm) - 320, 320)]
    last = len(voiced) - voiced[::-1].index(True)
    return pcm[:last * 320].astype(np.float32) / 32768


def silence(seconds):
    return np.zeros(round(seconds * RATE), np.float32)


def turns(audio, silence_s, short_utterance_silence_s=None):
    """Each ended turn as (seconds of audio fed when it ended, seconds of WAV)."""
    from reachy_muse.voice_audio import TurnRecorder

    recorder = TurnRecorder(RATE, silence_s=silence_s, max_s=60.0,
                            short_utterance_silence_s=short_utterance_silence_s)
    ended = []
    for start in range(0, len(audio), 320):
        wav = recorder.feed(audio[start:start + 320])
        if wav is not None:
            ended.append((round((start + 320) / RATE, 2), round((len(wav) - 44) / 2 / RATE, 2)))
    return ended


def test_the_turn_ends_point_eight_seconds_after_the_last_word_instead_of_two():
    reminder = speech("reminder")
    assert len(reminder) / RATE == 3.0
    assert turns(np.concatenate([reminder, silence(3)]), 2.0, 1.2) == [(5.0, 4.88)]
    assert turns(np.concatenate([reminder, silence(3)]), .8, 1.2) == [(3.8, 3.68)]


def test_a_mid_sentence_pause_of_point_four_seconds_keeps_remind_me_at_eight_not_at_nine_one_turn():
    reminder = speech("reminder")
    # "Remind me at eight to call Maya, | not at nine." The comma is a 0.14 s gap 2.04 s in; this makes it 0.54 s.
    paused = np.concatenate([reminder[:round(2.04 * RATE)], silence(.4), reminder[round(2.04 * RATE):], silence(2)])
    assert turns(paused, .8, 1.2) == [(4.2, 4.08)]


def test_a_hesitation_after_a_short_fragment_waits_for_the_rest_of_the_request():
    reminder = speech("reminder")
    hesitant = np.concatenate([reminder[:round(.7 * RATE)], silence(1), reminder[round(.7 * RATE):], silence(2)])
    assert turns(hesitant, .8, 1.2) == [(4.8, 4.68)]
    assert turns(hesitant, .8) == [(1.62, 1.5), (4.8, 3.18)]


@pytest.mark.parametrize("mode, given, expected", [
    (Mode.ON_ROBOT, None, .8),
    (Mode.MUSE_VOICE, None, .8),
    (Mode.COMPANION, None, 1.2),
    (Mode.ON_ROBOT, 1.5, 1.5),
    (Mode.COMPANION, .8, .8),
])
def test_the_default_end_of_turn_silence_depends_on_who_decides_the_turn_is_over(mode, given, expected):
    assert end_of_turn_silence(mode, given) == expected


def test_companion_vad_fallback_keeps_recorded_corrections_and_one_second_pauses():
    import asyncio
    from bench.turn2 import endpoint_replay

    assert asyncio.run(endpoint_replay()) == {
        "reminder": [(4.2, 4.08)],
        "correction": [(5.4, 5.28)],
        "negation": [(4.46, 4.34)],
        "one_second_correction": [(5.2, 5.08)],
        "short_fragment_hesitation": [(5.2, 5.08)],
    }


def test_smart_turn_endpoint_is_admitted_before_the_robot_silence_limit():
    import asyncio
    from reachy_muse import companion_client as client
    from reachy_muse.protocol import HearEndpoint, Op, encode

    link = client.CompanionLink("ws://offline.invalid", token="offline", out_rate=RATE, insecure=True)
    link._connection = client._Connection(None, (Op.HEAR,))
    link._set(client.LinkState.UP)
    turn = client.CompanionHearing(link).open_turn(RATE, silence_s=2)
    audio = np.concatenate([speech("reminder"), silence(.52)])
    for start in range(0, len(audio), 320):
        turn.feed(audio[start:start + 320])
        assert turn.take().ended is None
    assert turn.quiet_s == .52
    link._receive(link._connection, encode(HearEndpoint(1, "Remind me at eight to call Maya, not at nine.", False)))
    result = asyncio.run(turn.take().ended.endpoint())
    assert result.text == "Remind me at eight to call Maya, not at nine."
    assert result.forced is False
    assert turn.active is False
