import json

import numpy as np
import pytest

from bench.score_voices import clarity, intact, word_errors, words

RATE = 48_000
TIME = np.arange(RATE) / RATE


def test_numbers_times_and_amounts_are_spelled_out_the_same_way_on_both_sides():
    assert words("Your appointment with Dr. Patel is on Tuesday at 3.15 p.m.") == words("Your appointment with Doctor Patel is on Tuesday at 3:15 pm.")
    assert words("I didn't find flights under $400, a 70% chance.") == ["i", "did", "not", "find", "flights", "under", "four", "hundred", "dollars",
                                                                         "a", "seventy", "percent", "chance"]
    assert words("another twenty-five minutes, a seventy per cent chance") == words("another 25 minutes, a 70 percent chance")


def test_word_errors_count_substitutions_insertions_and_deletions():
    assert word_errors(words("remind Rohan to water the plants"), words("remind Rohen to water plants")) == 2
    assert word_errors(words("it is 18 degrees"), words("it is eighteen degrees")) == 0


def test_a_misheard_name_is_the_critical_token_that_fails():
    assert intact("Remind Rohen to water the plants on Friday morning.", ["Rohan", "Friday morning"]) == [False, True]


def test_clarity_is_the_share_of_energy_between_1_and_4_khz():
    assert round(clarity(np.sin(2 * np.pi * 2000 * TIME), RATE), 3) == 1.0
    assert round(clarity(np.sin(2 * np.pi * 300 * TIME), RATE), 3) == 0.0
    assert round(clarity(np.sin(2 * np.pi * 300 * TIME) + np.sin(2 * np.pi * 2000 * TIME), RATE), 3) == 0.5


@pytest.mark.real
def test_parakeet_hears_the_rendered_test_sentences_almost_word_for_word(tmp_path):
    from bench.score_voices import score
    from bench.voices import SENTENCES, render_all
    from reachy_companion.voice_kokoro import Loudness

    render_all(tmp_path, ["af_heart"], SENTENCES, Loudness())
    result = score(json.loads((tmp_path / "manifest.json").read_text()), tmp_path)
    print(result["voices"], [f["heard"] for f in result["files"]])
    (voice,) = result["voices"]
    assert voice["wer"] <= 0.03  # at most 2 of the 70 words; Parakeet sometimes hears "Rohan" as "Rohen"
    assert int(voice["critical"].split("/")[0]) >= 15 and voice["critical"].endswith("/16")
