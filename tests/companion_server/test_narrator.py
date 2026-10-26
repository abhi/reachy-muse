import asyncio
import json

import pytest

from reachy_muse.companion_server.engines import Narrated
from reachy_muse.contract.intent import Intent
from reachy_muse.contract.expressions import Expression
from reachy_muse.protocol import NarratedLine, NarrateOp
from reachy_muse.companion_server.narrator_mlx import OPENING_RULES, Ask, MlxNarrator, fallback, messages, parse
from reachy_muse.contract.persona import REACHY_PERSONA

REMINDER = Ask(NarrateOp.LINES, "remind me to call Maya at eight",
               reply="**Done.** I set a reminder to call Maya at 8 pm tonight.\n\n- It repeats daily.")


def answer(*lines):
    return json.dumps({"lines": [{"text": text, "expression": expression} for text, expression in lines]})


def test_a_grounded_answer_parses_into_lines_with_expressions():
    lines, problems = parse(REMINDER, answer(("I set a reminder to call Maya at 8 pm tonight.", "nod"), ("It repeats daily.", "neutral")))
    assert problems == ()
    assert lines == Narrated(None, (NarratedLine("I set a reminder to call Maya at 8 pm tonight.", Expression.NOD),
                                      NarratedLine("It repeats daily.", Expression.NEUTRAL)))


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


def opening(intent, text=None):
    return json.dumps({"intent": intent, "lines": [] if text is None else [{"text": text, "expression": "nod"}]})


def test_an_acknowledgement_may_not_claim_a_result():
    ask = Ask(NarrateOp.ACKNOWLEDGE, "book a table for two at seven")
    assert parse(ask, opening("action", "Sure, I'll get that going."))[1] == ()
    assert parse(ask, opening("action", "Okay, I've got it."))[1] == (
        "the line \"Okay, I've got it.\" claims a result before the work has started",)
    assert parse(ask, opening("action", "Okay, the table is booked."))[1] == (
        "the line 'Okay, the table is booked.' answers or reports instead of only reacting",)
    assert parse(Ask(NarrateOp.ACKNOWLEDGE, "set a timer for ten minutes"), opening("action", "Ten minutes started."))[1] == (
        "the line 'Ten minutes started.' answers or reports instead of only reacting",)


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

    def _generate(self, conversation, max_tokens, prefix):
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
    assert asyncio.run(Scripted("not json", "not json").acknowledge("what time is it")) == Narrated(
        Intent.QUESTION, (NarratedLine("Hmm, good question.", None),))
    assert asyncio.run(Scripted(opening("follow_up", "Seven it is."), "not json").acknowledge(
        "make it seven", recently_said=("Right, about that.",))) == Narrated(Intent.FOLLOW_UP, (NarratedLine("Mm, okay.", None),))


def test_restatement_fallback_does_not_log_private_text(caplog):
    private = "Call Alice about her diagnosis."
    narrator = Scripted(opening("action", private), opening("action", private))
    narration = narrator.narrate(Ask(NarrateOp.ACKNOWLEDGE, private))

    assert narration.fell_back and len(narration.attempts) == 2
    assert narration.attempts[-1].problems == (
        "the line 'Call Alice about her diagnosis.' restates the request",)
    assert caplog.messages
    assert private not in caplog.text
    assert "Alice" not in caplog.text and "diagnosis" not in caplog.text


@pytest.mark.parametrize("request_", ["how are you", "Thanks!", "good morning", "hi Reachy", "what's up"])
def test_chat_without_local_permission_is_silent(request_):
    narrator = Scripted(opening("chat"))
    assert asyncio.run(narrator.acknowledge(request_)) == Narrated(Intent.CHAT, ())


def test_a_task_after_a_greeting_is_still_acknowledged():
    narrator = Scripted(opening("action", "Sure, I'll get that going."))
    assert asyncio.run(narrator.acknowledge("good morning, set a timer for ten minutes")) == Narrated(
        Intent.ACTION, (NarratedLine("Sure, I'll get that going.", Expression.NOD),))


@pytest.mark.parametrize("intent, request_, previous, may_chat, text", [
    (Intent.CHAT, "I'm so tired today", "", True, "Aw, rough day. Can I keep you company?"),
    (Intent.QUESTION, "Why is the sky blue", "", False, "Ooh, good question."),
    (Intent.ACTION, "remind me to call mom", "", False, "Sure, I'll get that going."),
    (Intent.FOLLOW_UP, "and tomorrow", "what is the forecast", False, "Right, about that."),
])
def test_each_intent_parses_and_returns_through_the_real_narrator(intent, request_, previous, may_chat, text):
    ask = Ask(NarrateOp.ACKNOWLEDGE, request_, previous=previous, may_chat=may_chat)
    expected = Narrated(intent, (NarratedLine(text, Expression.NOD),))
    assert parse(ask, opening(intent.value, text)) == (expected, ())
    assert asyncio.run(Scripted(opening(intent.value, text)).acknowledge(
        request_, previous=previous, may_chat=may_chat)) == expected


@pytest.mark.parametrize("request_, intent, text, problem", [
    ("book a table for two", "action", "Sure, a table for two.", "the line 'Sure, a table for two.' restates the request"),
    ("book a table", "action", "I've sent it", 'the line "I\'ve sent it" claims a result before the work has started'),
    ("tell me about Lisbon", "question", "Ooh, Lisbon in May!", "the line 'Ooh, Lisbon in May!' says the name 'May', which the input does not"),
    ("hi", "future", None, "the intent is missing or unknown"),
])
def test_bad_openings_name_the_literal_problem(request_, intent, text, problem):
    assert problem in parse(Ask(NarrateOp.ACKNOWLEDGE, request_), opening(intent, text))[1]


def test_acknowledgement_retry_preserves_intent_and_recent_lines():
    narrator = Scripted(opening("action", "I've sent it"), opening("action", "On it."))
    assert asyncio.run(narrator.acknowledge("send an email")) == Narrated(
        Intent.ACTION, (NarratedLine("On it.", Expression.NOD),))
    ask = Ask(NarrateOp.ACKNOWLEDGE, "why?", already_said=("Ooh, good question.",))
    assert parse(ask, opening("question", "Ooh, good question!"))[1] == (
        "the line 'Ooh, good question!' was already said",)


@pytest.mark.parametrize("raw", ["Sure!", "[]", '{"intent":"chat","lines":"Hi"}',
                                 '{"intent":4,"lines":[]}', '{"intent":"action","lines":[{"text":null}]}'])
def test_real_checks_reject_bad_model_shapes(raw):
    assert parse(Ask(NarrateOp.ACKNOWLEDGE, "hi"), raw)[1]


def test_companion_rechecks_local_chat_permission_against_the_request():
    ask = Ask(NarrateOp.ACKNOWLEDGE, "can you check the weather", may_chat=True)
    assert parse(ask, opening("chat", "Hello.")) == (Narrated(Intent.CHAT, ()), ())
    assert parse(Ask(NarrateOp.ACKNOWLEDGE, "hi"), opening("chat", "Hello.")) == (Narrated(Intent.CHAT, ()), ())


def test_the_opening_prompt_speaks_as_reachy_and_shows_what_it_said_lately():
    assert OPENING_RULES.startswith("You speak as Reachy. " + REACHY_PERSONA)
    system, user = messages(Ask(NarrateOp.ACKNOWLEDGE, "I'm so tired today", previous="how are you",
                                already_said=("Doing great, you?",), may_chat=True))
    assert system == {"role": "system", "content": OPENING_RULES}
    assert user == {"role": "user", "content": (
        "Request: I'm so tired today\nPrevious request: how are you\nReachy said lately:\n- Doing great, you?"
        "\nReachy may answer: yes")}
    assert messages(Ask(NarrateOp.ACKNOWLEDGE, "what's the weather", may_chat=True))[1]["content"].endswith(
        "Reachy said lately:\n(nothing yet)\nReachy may answer: no")
