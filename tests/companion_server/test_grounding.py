import pytest

from musegadget.reachy_capabilities import grounded
from reachy_companion.grounding import missing_facts, ungrounded

REPLY = ("Tomorrow in Lyon it will be 18 degrees with a 40% chance of rain. "
         "Your dentist appointment with Dr. Okafor is at 7:45 pm on October 12th, and the copay is $25.50.")


@pytest.mark.parametrize("line", [
    "Lyon will reach 18 degrees tomorrow.",
    "There's a forty percent chance of rain.",
    "Your appointment with Dr. Okafor is at 7:45 p.m.",
    "It's at 7:45 on the twelfth of October.",
    "The copay is $25.50.",
    "Okafor's office expects you at 7:45 PM.",
    "Sure, it's on October 12.",
    "Checking Lyon now, and I'm on it.",
])
def test_lines_that_restate_the_reply_are_grounded(line):
    assert missing_facts(line, "what's the weather in lyon", REPLY) == []


@pytest.mark.parametrize("line, problem", [
    ("Lyon will reach 19 degrees tomorrow.", "the number 19"),
    ("Your appointment is at 7:45 am.", "the time 7:45 a.m."),
    ("Your appointment is at 8 pm.", "the time 8:00 p.m."),
    ("The copay is $26.", "the amount 26 usd"),
    ("The copay is 25.50 euros.", "the amount 25.50 eur"),
    ("There's a fifty percent chance of rain.", "the amount 50 %"),
    ("Your appointment with Dr. Patel is soon.", "the name 'Patel'"),
    ("Paris will be warmer.", "the name 'Paris'"),
    ("It will rain tonight.", "the date word 'tonight'"),
    ("Your appointment is on Friday.", "the date word 'friday'"),
    ("It's your third visit.", "the number 3"),
    ("Maya says it will rain.", "the name 'Maya'"),
    ("Bring an umbrella.", "the name 'Bring'"),
])
def test_lines_that_add_a_fact_name_it(line, problem):
    assert problem in missing_facts(line, "what's the weather in lyon", REPLY)


def test_a_time_may_not_gain_am_or_pm_but_may_restate_a_bare_hour():
    assert missing_facts("The reminder is set for 8:00.", "remind me at eight") == []
    assert missing_facts("The reminder is set for 8 pm.", "remind me at eight") == ["the time 8:00 p.m."]
    assert missing_facts("Lunch is at noon.", "lunch at 12 pm") == []


def test_number_words_and_digits_are_the_same_fact():
    assert missing_facts("You have 3 meetings and twenty-one emails.", "three meetings, 21 emails") == []
    assert missing_facts("That's two hundred and five steps.", "205 steps") == []
    assert missing_facts("That's 1,200 steps.", "twelve hundred steps") == []
    assert missing_facts("That's 1,300 steps.", "twelve hundred steps") == ["the number 1300"]


@pytest.mark.parametrize("line", [
    "Your appointment with Dr. Okafor is at 7:45 pm on October 12th.",
    "Lyon will reach 18 degrees tomorrow.",
    "The copay is $25.50.",
])
def test_lines_this_check_accepts_the_robot_accepts_too(line):
    assert ungrounded(line, REPLY) == []
    assert grounded(line, REPLY)

