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
    (Mode.COMPANION, None, 2.0),
    (Mode.ON_ROBOT, 1.5, 1.5),
    (Mode.COMPANION, .8, .8),
])
def test_the_default_end_of_turn_silence_depends_on_who_decides_the_turn_is_over(mode, given, expected):
    assert end_of_turn_silence(mode, given) == expected
