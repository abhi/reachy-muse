import pytest

from reachy_muse.capabilities import Heard, Hold, LocalReply, SpokenLine, grounded
from reachy_muse.companion_client import CompanionOpening
from reachy_muse.companion_server.narrator_mlx import _CLAIM
from reachy_muse.contract.intent import FALLBACK_OPENERS, Intent, pick
from reachy_muse.failover import LATE, settle
from reachy_muse.local_backends import ACKNOWLEDGEMENTS, RuleNarrator


@pytest.mark.parametrize("phrase", ACKNOWLEDGEMENTS)
def test_presynthesized_openers_do_not_claim_or_invent_facts(phrase):
    assert grounded(phrase, "") is True
    assert _CLAIM.search(phrase) is None


@pytest.mark.parametrize("got, heard, expected", [
    (LATE, Heard("book a table"), Hold(Intent.ACTION, None)),
    (None, Heard("book a table"), Hold(Intent.ACTION, SpokenLine("On it.", None, "ack"))),
    (CompanionOpening(None, (SpokenLine("Sure.", None, "ack"),)), Heard("book a table"),
     Hold(Intent.ACTION, SpokenLine("Sure.", None, "ack"))),
    (CompanionOpening(Intent.CHAT, (SpokenLine("Aw, rough day.", None, "ack"),)), Heard("I'm tired", may_chat=True),
     LocalReply((SpokenLine("Aw, rough day.", None, "answer"),))),
    (CompanionOpening(Intent.CHAT, (SpokenLine("Aw, rough day.", None, "ack"),)), Heard("I'm tired"), Hold(Intent.CHAT, None)),
    (CompanionOpening(Intent.CHAT, ()), Heard("I'm tired", may_chat=True), Hold(Intent.CHAT, None)),
    (CompanionOpening(None, (SpokenLine("Hello.", None, "ack"),)), Heard("I'm tired", may_chat=True), Hold(Intent.ACTION, SpokenLine("Hello.", None, "ack"))),
    (CompanionOpening(Intent.QUESTION, ()), Heard("why?"), Hold(Intent.QUESTION, None)),
    (CompanionOpening(Intent.FOLLOW_UP, (SpokenLine("Right, about that.", None, "ack"),)), Heard("and?"),
     Hold(Intent.FOLLOW_UP, SpokenLine("Right, about that.", None, "ack"))),
])
def test_settle_rows(got, heard, expected):
    assert settle(heard, Intent.ACTION, got, RuleNarrator(presynthesized=lambda text: True)) == expected


def test_an_old_companion_never_answers_chat_locally():
    got = CompanionOpening(None, (SpokenLine("Hi.", None, "ack"),))
    assert settle(Heard("hi", may_chat=True), Intent.CHAT, got, RuleNarrator()) == Hold(Intent.CHAT, None)


def test_rotation_skips_recent_phrases_and_is_scoped_per_intent():
    local = RuleNarrator(presynthesized=lambda text: True)
    assert local.hold(Intent.QUESTION, Heard("why?")).line.text == "Hmm, good question."
    assert local.hold(Intent.ACTION, Heard("send it")).line.text == "On it."
    assert local.hold(Intent.QUESTION, Heard("why?")).line.text == "Ooh, let me think."
    assert pick(FALLBACK_OPENERS[Intent.ACTION], 0, ("On it.",)) == "Sure, I'll get that going."
    assert pick(FALLBACK_OPENERS[Intent.ACTION], 0, FALLBACK_OPENERS[Intent.ACTION]) is None
