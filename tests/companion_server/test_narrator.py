import asyncio
import json

import pytest

from reachy_muse.contract.expressions import Expression
from reachy_muse.protocol import NarratedLine, NarrateOp
from reachy_muse.companion_server.narrator_mlx import Ask, MlxNarrator, fallback, parse

REMINDER = Ask(NarrateOp.LINES, "remind me to call Maya at eight",
               reply="**Done.** I set a reminder to call Maya at 8 pm tonight.\n\n- It repeats daily.")


def answer(*lines):
    return json.dumps({"lines": [{"text": text, "expression": expression} for text, expression in lines]})


def test_a_grounded_answer_parses_into_lines_with_expressions():
    lines, problems = parse(REMINDER, answer(("I set a reminder to call Maya at 8 pm tonight.", "nod"), ("It repeats daily.", "neutral")))
    assert problems == ()
    assert lines == (NarratedLine("I set a reminder to call Maya at 8 pm tonight.", Expression.NOD),
                     NarratedLine("It repeats daily.", Expression.NEUTRAL))


@pytest.mark.parametrize("raw, problem", [
    (answer(("I set a reminder to call Maya at 9 pm.", "nod")), "the line 'I set a reminder to call Maya at 9 pm.' says the time 9:00 p.m., which the input does not"),
    (answer(("I set a reminder to call Sam at 8 pm.", "nod")), "the line 'I set a reminder to call Sam at 8 pm.' says the name 'Sam', which the input does not"),
    (answer(("Reminder set.", "wink")), "the expression 'wink' is not one of neutral, happy, sad, surprised, curious, nod, shake, listening, thinking"),
    (answer(("**Reminder** set.", "nod")), "the line '**Reminder** set.' has markup, a link or an emoji"),
    (answer(*[("Reminder set.", "nod")] * 4), "there are 4 lines; at most 3 are allowed"),
    ('{"lines": [{"text": "Reminder set.", "expression": "nod"}', "the answer is not valid JSON (Expecting ',' delimiter)"),
    ('{"lines": "Reminder set."}', 'the answer must be {"lines": [{"text": ..., "expression": ...}]}'),
])
def test_an_answer_that_breaks_a_rule_names_the_problem(raw, problem):
    assert problem in parse(REMINDER, raw)[1]


def test_an_acknowledgement_may_not_claim_a_result():
    ask = Ask(NarrateOp.ACKNOWLEDGE, "book a table for two at seven")
    assert parse(ask, answer(("Sure, a table for two at seven.", "nod")))[1] == ()
    assert parse(ask, answer(("Okay, I've booked it.", "nod")))[1] == ("the line \"Okay, I've booked it.\" claims a result before the work has started",)


def test_a_progress_line_may_not_repeat_one_already_said():
    ask = Ask(NarrateOp.PROGRESS, "find flights to Lisbon", status="Looking at flights now.", already_said=("Looking at flights now.",))
    assert parse(ask, answer(("Looking at flights now!", "thinking")))[1] == ("the line 'Looking at flights now!' was already said",)
    assert parse(ask, answer(("Still checking flights to Lisbon.", "thinking")))[1] == ()


def test_the_fallback_speaks_the_replys_first_sentences_without_markup():
    assert fallback(REMINDER) == (NarratedLine("Done.", None),
                                  NarratedLine("I set a reminder to call Maya at 8 pm tonight.", None),
                                  NarratedLine("It repeats daily.", None))
    assert fallback(Ask(NarrateOp.ACKNOWLEDGE, "what time is it")) == ()


class Scripted(MlxNarrator):
    """The real narrator with its model replaced by a list of answers."""

    def __init__(self, *answers):
        super().__init__()
        self.answers = list(answers)

    def _generate(self, conversation, max_tokens):
        return self.answers.pop(0)


def test_a_rejected_answer_is_retried_once():
    narrator = Scripted(answer(("Call Sam at 9 pm.", "nod")), answer(("I set a reminder to call Maya at 8 pm tonight.", "nod")))
    lines = asyncio.run(narrator.lines(REMINDER.request, REMINDER.reply))
    assert lines == (NarratedLine("I set a reminder to call Maya at 8 pm tonight.", Expression.NOD),)


def test_two_rejected_answers_fall_back():
    narrator = Scripted(answer(("Call Sam at 9 pm.", "nod")), "not json")
    narration = narrator.narrate(REMINDER)
    assert (narration.fell_back, len(narration.attempts)) == (True, 2)
    assert narration.lines == fallback(REMINDER)
    assert asyncio.run(Scripted("not json", "not json").acknowledge("what time is it")) == ()


@pytest.mark.parametrize("request_", ["how are you", "Thanks!", "good morning", "hi Reachy", "what's up", "Hi Reachy, how are you doing today?",
                                      "hey reachy how's your day been", "thank you, that was really helpful", "good night reachy, sleep well", "stop"])
def test_small_talk_and_very_short_requests_get_no_acknowledgement(request_):
    restating = answer(("Okay, let me show how are you.", "nod"))
    assert asyncio.run(Scripted(restating).acknowledge(request_)) == ()


def test_a_task_after_a_greeting_is_still_acknowledged():
    narrator = Scripted(answer(("Sure, a timer for ten minutes.", "nod")))
    assert asyncio.run(narrator.acknowledge("good morning, set a timer for ten minutes")) == (NarratedLine("Sure, a timer for ten minutes.", Expression.NOD),)
