import pytest

from reachy_muse.contract.expressions import Expression
from reachy_muse.motion.sentence import SentenceMotion, classify_sentence


@pytest.mark.parametrize("text, expected", [
    ("The train leaves at six.", SentenceMotion(Expression.NEUTRAL)),
    ("Congratulations, you did it!", SentenceMotion(Expression.HAPPY, "exclamation")),
    ("Good news, your booking is confirmed.", SentenceMotion(Expression.HAPPY)),
    ("I'm sorry, the booking failed.", SentenceMotion(Expression.SAD)),
    ("Unfortunately, that didn't work!", SentenceMotion(Expression.SAD, "exclamation")),
    ("Would you like another option?", SentenceMotion(Expression.CURIOUS, "question")),
    ('She asked, "Ready?"', SentenceMotion(Expression.CURIOUS, "question")),
    ("Wow, that's surprising!", SentenceMotion(Expression.SURPRISED, "exclamation")),
    ("Yes, I can help.", SentenceMotion(Expression.NOD)),
    ("That is not wonderful.", SentenceMotion(Expression.NEUTRAL)),
    ("The test never failed.", SentenceMotion(Expression.NEUTRAL)),
    ("It isn't good news.", SentenceMotion(Expression.NEUTRAL)),
    ("It isn't bad news.", SentenceMotion(Expression.NEUTRAL)),
    ("Thanks for waiting!", SentenceMotion(Expression.HAPPY, "exclamation")),
    ("I'm not happy about it.", SentenceMotion(Expression.NEUTRAL)),
    ("The outcome is not surprising.", SentenceMotion(Expression.NEUTRAL)),
    ("", SentenceMotion(Expression.NEUTRAL)),
    ("Enjoy the concert tonight.", SentenceMotion(Expression.HAPPY)),
    ("Your flight is delayed by an hour.", SentenceMotion(Expression.SAD)),
    ("Surprisingly, the museum is free today.", SentenceMotion(Expression.SURPRISED)),
    ("Sure, I'll remind you at six.", SentenceMotion(Expression.NOD)),
    ("The game wasn't cancelled.", SentenceMotion(Expression.NEUTRAL)),
])
def test_local_expression_and_punctuation(text, expected):
    assert classify_sentence(text) == expected


def test_explicit_expression_keeps_independent_question_accent():
    assert classify_sentence("Is that good news?", Expression.SAD) == SentenceMotion(Expression.SAD, "question")
    assert classify_sentence("Sadly, yes!", Expression.HAPPY) == SentenceMotion(Expression.HAPPY, "exclamation")


@pytest.mark.parametrize("text, expected", [
    ("Congratulations!", SentenceMotion(Expression.HAPPY, "exclamation")),
    ("Sadly the moat is closed.", SentenceMotion(Expression.SAD)),
    ("Would you like directions?", SentenceMotion(Expression.CURIOUS, "question")),
    ("The train leaves at six.", SentenceMotion(Expression.NEUTRAL)),
])
def test_a_neutral_hint_lets_the_text_show_its_feeling(text, expected):
    assert classify_sentence(text, Expression.NEUTRAL) == expected


def test_long_text_bounds_lexical_scan_but_keeps_terminal_punctuation():
    assert classify_sentence("a " * 10000 + "wonderful!") == SentenceMotion(Expression.NEUTRAL, "exclamation")
