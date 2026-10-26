import pytest

from reachy_muse.contract.intent import Intent, guess_intent, maybe_chat


@pytest.mark.parametrize("text, gated, intent", [
    ("how are you doing today", True, Intent.CHAT),
    ("I wonder if it'll rain", False, Intent.QUESTION),
    ("remind me to call mom", False, Intent.ACTION),
    ("book a table for two", False, Intent.ACTION),
    ("ugh what a day", True, Intent.CHAT),
    ("what did you just say", False, Intent.FOLLOW_UP),
    ("why is the sky blue", False, Intent.QUESTION),
    ("thanks, can you also check the traffic", False, Intent.QUESTION),
    ("I love this song", True, Intent.CHAT),
    ("I just got back from a long run", True, Intent.CHAT),
    ("thanks what is the capital of France", False, Intent.QUESTION),
    ("haha explain quantum physics", False, Intent.QUESTION),
    ("can you please send an email", False, Intent.ACTION),
    ("I love my calendar", False, Intent.QUESTION),
    ("I slept 3 hours", False, Intent.QUESTION),
    ("do you ever get bored", True, Intent.CHAT),
    ("good morning Reachy", True, Intent.CHAT),
    ("you're a pretty cool little robot", True, Intent.CHAT),
    ("I'm hungry", False, Intent.QUESTION),
])
def test_social_gate_and_fallback(text, gated, intent):
    assert maybe_chat(text) is gated
    assert guess_intent(text, "Tell me about rainbows") is intent


def test_a_reference_requires_a_previous_exchange():
    assert guess_intent("why is that", "") is Intent.QUESTION
    assert guess_intent("why is that", "Rainbows bend sunlight") is Intent.FOLLOW_UP
    assert Intent.parse("future") is None
    assert Intent.parse(3) is None
