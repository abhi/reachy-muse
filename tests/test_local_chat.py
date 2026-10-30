import asyncio
import dataclasses

import pytest

from reachy_muse.capabilities import Heard, Hold, Mode, SpokenLine
from reachy_muse.companion_client import CompanionOpening
from reachy_muse.contract.intent import CHAT_PHRASES, Intent
from reachy_muse.conversation import TurnOutcome, VoiceConversation
from reachy_muse.failover import CompanionRoute, FailoverNarrator
from reachy_muse.local_backends import backends_for
from reachy_muse.turn_timeline import TurnTimeline
from test_conversation import FakeHardware, FakeSession, cancel_task

np = pytest.importorskip("numpy")


class Speech:
    def __init__(self):
        self.spoken = []

    async def stream(self, text, rate):
        self.spoken.append(text)
        yield np.full(80, .1, dtype=np.float32)


class Companion:
    available = True
    up = True

    def __init__(self, intent=Intent.CHAT, delay=None):
        self.intent = intent
        self.delay = delay
        self.heard = []
        self.returned = False

    async def acknowledge(self, heard):
        self.heard.append(heard)
        if self.delay is not None:
            await self.delay.wait()
        self.returned = True
        return CompanionOpening(self.intent, (SpokenLine("Aw, rough day.", None, "ack"),))


class DispatchOnly(VoiceConversation):
    async def _deliver(self, job):
        await asyncio.sleep(.001)
        return TurnOutcome.ACCEPTED


def robot(session, hardware, speech, companion, *, local_chat=True):
    local = backends_for(Mode.ON_ROBOT, session, speech=speech)
    local.voice.fixed_phrases += CHAT_PHRASES
    route = CompanionRoute(companion)
    route.begin_turn()
    narrator = FailoverNarrator(companion, local.narrator, route)
    return DispatchOnly(session, hardware, backends=dataclasses.replace(local, narrator=narrator),
                        local_chat=local_chat)


def test_chat_is_audible_without_dispatch_and_next_request_carries_the_note():
    async def scenario():
        session, hardware, speech = FakeSession(), FakeHardware(), Speech()
        companion = Companion()
        conversation = robot(session, hardware, speech, companion)
        timeline = TurnTimeline()
        assert await conversation.turn(b"voice", recognized_text="I'm tired", timeline=timeline) is TurnOutcome.ACCEPTED
        assert session.setup_messages == []
        assert speech.spoken == ["Aw, rough day."]
        assert len(hardware.played) == 1
        assert list(conversation._recent.said) == ["Aw, rough day."]
        assert "chat_audio" in timeline.marks
        companion.intent = Intent.QUESTION
        await conversation.turn(b"voice", recognized_text="Why is the sky blue?")
        assert session.setup_messages == [(
            "Earlier, said to Reachy only: user: I'm tired; Reachy: Aw, rough day.\n"
            "The user's spoken request is: Why is the sky blue?", None)]
        assert companion.heard[-1] == Heard("Why is the sky blue?", "I'm tired", ("Aw, rough day.",), False)
        assert conversation._recent.chat_note() == ("", 0)
    asyncio.run(scenario())


def test_local_chat_off_keeps_chat_muse_bound():
    async def scenario():
        session, hardware, speech = FakeSession(), FakeHardware(), Speech()
        companion = Companion()
        conversation = robot(session, hardware, speech, companion, local_chat=False)
        await conversation.turn(b"voice", recognized_text="I'm tired")
        assert session.setup_messages == [("I'm tired", None)]
        assert conversation._recent.chat_note() == ("", 0)
        assert speech.spoken == []
    asyncio.run(scenario())


def test_an_ungated_question_is_sent_before_slow_narration_returns():
    async def scenario():
        companion = Companion(Intent.QUESTION, asyncio.Event())
        session, hardware, speech = FakeSession(), FakeHardware(), Speech()
        conversation = robot(session, hardware, speech, companion)
        await conversation.turn(b"voice", recognized_text="Why is the sky blue?")
        assert session.setup_messages == [("Why is the sky blue?", None)]
        assert companion.returned is False
    asyncio.run(scenario())


def test_a_gated_utterance_that_the_companion_calls_question_goes_to_muse():
    async def scenario():
        companion = Companion(Intent.QUESTION)
        session, hardware, speech = FakeSession(), FakeHardware(), Speech()
        conversation = robot(session, hardware, speech, companion)
        await conversation.turn(b"voice", recognized_text="I'm tired")
        assert companion.returned is True
        assert companion.heard == [Heard("I'm tired", may_chat=True)]
        assert session.setup_messages == [("I'm tired", None)]
        assert conversation._recent.chat_note() == ("", 0)
    asyncio.run(scenario())


def test_companion_down_uses_rotated_rules_and_never_answers_chat():
    async def scenario():
        companion = Companion()
        companion.up = False
        session, hardware, speech = FakeSession(), FakeHardware(), Speech()
        conversation = robot(session, hardware, speech, companion)
        await conversation.backends.voice.warm(hardware.output_sample_rate)
        narrator = conversation.backends.narrator
        assert await narrator.acknowledge(Heard("why is the sky blue")) == Hold(
            Intent.QUESTION, SpokenLine("Hmm, good question.", None, "ack"))
        assert await narrator.acknowledge(Heard("why is grass green")) == Hold(
            Intent.QUESTION, SpokenLine("Ooh, let me think.", None, "ack"))
        assert await narrator.acknowledge(Heard("send an email")) == Hold(
            Intent.ACTION, SpokenLine("On it.", None, "ack"))
        assert await narrator.acknowledge(Heard("I'm tired", may_chat=True)) == Hold(Intent.CHAT, None)
        assert companion.heard == []
    asyncio.run(scenario())


@pytest.mark.parametrize("text, reply", [
    ("good morning", "Good morning!"),
    ("Thanks!", "You're welcome!"),
    ("How are you?", "I'm here and ready to chat."),
])
def test_companion_down_answers_simple_chat_from_cached_audio_without_muse(text, reply):
    async def scenario():
        companion = Companion()
        companion.up = False
        session, hardware, speech = FakeSession(), FakeHardware(), Speech()
        conversation = robot(session, hardware, speech, companion)
        await conversation.backends.voice.warm(hardware.output_sample_rate)
        speech.spoken.clear()
        assert await conversation.turn(b"voice", recognized_text=text) is TurnOutcome.ACCEPTED
        assert session.setup_messages == []
        assert companion.heard == []
        assert speech.spoken == []  # no inference at reply time
        assert len(hardware.played) == 1
        assert list(conversation._recent.said) == [reply]
    asyncio.run(scenario())


@pytest.mark.parametrize("text, enabled", [
    ("Good morning, what's on my calendar?", True),
    ("Thanks, remind me at eight", True),
    ("How are you going to fix my reminder?", True),
    ("good morning", False),
])
def test_local_grammar_never_swallows_requests_or_bypasses_chat_off(text, enabled):
    async def scenario():
        companion = Companion()
        companion.up = False
        session = FakeSession()
        conversation = robot(session, FakeHardware(), Speech(), companion, local_chat=enabled)
        await conversation.turn(b"voice", recognized_text=text)
        assert session.setup_messages == [(text, None)]
        assert companion.heard == []
    asyncio.run(scenario())


def test_unsent_chat_survives_a_rejected_muse_request():
    async def scenario():
        session = FakeSession(acknowledgement={"ok": False, "status": 403})
        conversation = robot(session, FakeHardware(), Speech(), Companion(Intent.QUESTION))
        conversation._recent.chatted("thanks", "Anytime.")
        with pytest.raises(ConnectionError, match="rejected"):
            await conversation.turn(b"voice", recognized_text="Why?")
        assert conversation._recent.chat_note() == ("Earlier, said to Reachy only: user: thanks; Reachy: Anytime.", 1)
    asyncio.run(scenario())


def test_deferred_chat_records_speech_only_when_the_queue_plays():
    async def scenario():
        session, hardware, speech = FakeSession(), FakeHardware(), Speech()
        conversation = robot(session, hardware, speech, Companion())
        timeline = TurnTimeline()
        await conversation.turn(b"voice", recognized_text="I'm tired", defer_playback=True, timeline=timeline)
        assert list(conversation._recent.said) == []
        assert "chat_audio" not in timeline.marks
        speaker = asyncio.create_task(conversation._play_output())
        try:
            await asyncio.wait_for(conversation._output_queue.join(), 2)
            assert list(conversation._recent.said) == ["Aw, rough day."]
            assert "chat_audio" in timeline.marks
        finally:
            await cancel_task(speaker)
    asyncio.run(scenario())
