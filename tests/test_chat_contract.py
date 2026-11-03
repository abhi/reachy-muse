# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Every user turn relies on registration, without a voice preamble."""

import asyncio

import pytest

from musegadget import muse_turn
import reachy_muse.conversation
from reachy_muse.capabilities import Mode
from reachy_muse.contract.replies import ReplyStyle
from reachy_muse.local_backends import backends_for
from reachy_muse.conversation import TurnOutcome, VoiceConversation


@pytest.fixture(autouse=True)
def no_reply_wait(monkeypatch):
    monkeypatch.setattr(muse_turn, "REPLY_QUIET_S", 0)


class Session:
    def __init__(self):
        self.registered = asyncio.Event()
        self.registered.set()
        self.chat_subscribed = asyncio.Event()
        self.chat_subscribed.set()
        self.sent = []
        self.conversation = None

    async def send_chat(self, text, session_id, **options):
        self.sent.append((text, session_id, options))
        serial = len(self.sent)
        for status in ("running", "completed"):
            self.conversation.job.turn.feed({
                "type": "event",
                "event": "task.status",
                "payload": {
                    "session_id": session_id,
                    "task_id": f"task-{serial}",
                    "reply_to_message_id": f"user-{serial}",
                    "status": status,
                },
            })
        return {"ok": True, "response": {"result": {
            "session_id": session_id,
            "is_thread": True,
            "message_id": f"user-{serial}",
        }}}


class Hardware:
    sample_rate = 16000
    output_sample_rate = 16000
    motion_enabled = True
    antenna_mode = "both"
    face_tracking_enabled = False

    def __init__(self):
        self.states = []
        self.cleared = 0

    def set_state(self, state, **options):
        self.states.append(state)

    def clear_audio(self):
        self.cleared += 1

    def read_audio(self):
        return None


async def owned_turn(text):
    session = Session()
    conversation = VoiceConversation(
        session,
        Hardware(), backends=backends_for(Mode.ON_ROBOT, session, speech=object(), transcriber=object()),
        session_id="owned-chat",
        owns_chat=True,
        wake_detector=type("Wake", (), {"phrase": "hey muse"})(),
    )
    session.conversation = conversation
    outcome = await conversation.turn(b"wav", recognized_text=text)
    return outcome, session.sent


def test_owned_chat_sends_only_the_recognized_user_transcript(monkeypatch):
    monkeypatch.setattr(muse_turn, "EMPTY_REPLY_GRACE_S", 0)
    outcome, sent = asyncio.run(owned_turn("What should I cook for lunch?"))
    assert outcome is TurnOutcome.ACCEPTED
    assert sent == [("What should I cook for lunch?", "owned-chat", {})]


def test_whole_turn_adds_only_the_unsent_local_chat_note():
    async def check():
        session = Session()
        conversation = VoiceConversation(session, Hardware(), session_id="owned-chat", owns_chat=True,
            backends=backends_for(Mode.ON_ROBOT, session, speech=object(), transcriber=object()))
        session.conversation = conversation
        conversation._recent.chatted("I'm tired.", "Long day?")
        assert await conversation.turn(b"wav", recognized_text="What should I cook for lunch?") is TurnOutcome.ACCEPTED
        assert session.sent == [(
            "Earlier, said to Reachy only: user: I'm tired.; Reachy: Long day?\n"
            "The user's spoken request is: What should I cook for lunch?", "owned-chat", {})]
    asyncio.run(check())


def test_new_owned_conversation_cannot_reinject_context_after_reconnect(monkeypatch):
    monkeypatch.setattr(muse_turn, "EMPTY_REPLY_GRACE_S", 0)

    async def check():
        first = await owned_turn("First request")
        reconnected = await owned_turn("Second request")
        return first, reconnected

    (first_outcome, first_sent), (second_outcome, second_sent) = asyncio.run(check())
    assert first_outcome is second_outcome is TurnOutcome.ACCEPTED
    assert first_sent[0][0] == "First request"
    assert second_sent[0][0] == "Second request"
    assert "The user's spoken request is" not in first_sent[0][0] + second_sent[0][0]


@pytest.mark.parametrize("owns_chat", [False, True])
@pytest.mark.parametrize("style", list(ReplyStyle))
def test_whole_turns_across_companion_changes_and_reconnect_send_literal_words(monkeypatch, owns_chat, style):
    from dataclasses import replace

    monkeypatch.setattr(muse_turn, "EMPTY_REPLY_GRACE_S", 0)

    class Companion:
        online = False

        def begin_turn(self):
            return self.online

    async def check():
        sent = []
        companion = Companion()
        for _ in range(2):
            session = Session()
            bundle = backends_for(Mode.ON_ROBOT, session, speech=object(), transcriber=object())
            conversation = VoiceConversation(session, Hardware(), owns_chat=owns_chat,
                session_id="owned-chat" if owns_chat else None,
                backends=replace(bundle, local_style=style, companion=companion))
            session.conversation = conversation
            for online in (False, True, False, True):
                companion.online = online
                assert await conversation.turn(b"wav", recognized_text="What should I cook for lunch?") is TurnOutcome.ACCEPTED
            sent.extend(session.sent)
        return sent

    sent = asyncio.run(check())
    options = {"output_modality": "voice"} if style is ReplyStyle.MUSE_VOICE else {}
    session_id = "owned-chat" if owns_chat else None
    assert sent == [
        ("What should I cook for lunch?", session_id, options),
        ("What should I cook for lunch?", session_id, {}),
        ("What should I cook for lunch?", session_id, options),
        ("What should I cook for lunch?", session_id, {}),
        ("What should I cook for lunch?", session_id, options),
        ("What should I cook for lunch?", session_id, {}),
        ("What should I cook for lunch?", session_id, options),
        ("What should I cook for lunch?", session_id, {}),
    ]


@pytest.mark.parametrize("owns_chat, expected", [(True, []), (False, [])])
def test_owned_microphone_does_not_send_expression_bootstrap(owns_chat, expected):
    pytest.importorskip("webrtcvad")
    pytest.importorskip("av")
    async def check():
        calls = []
        session = Session()
        conversation = VoiceConversation(
            session, Hardware(),
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object()),
                                         session_id="owned-chat" if owns_chat else None,
            owns_chat=owns_chat,
        )

        async def turn(wav, **options):
            calls.append(wav)
            await asyncio.Event().wait()

        conversation.turn = turn
        microphone = asyncio.create_task(conversation._microphone())
        await asyncio.sleep(0.03)
        microphone.cancel()
        with pytest.raises(asyncio.CancelledError):
            await microphone
        assert calls == expected

    asyncio.run(check())


@pytest.mark.parametrize("style", list(ReplyStyle))
def test_briefing_dispatch_sends_its_own_length_and_collects_every_sentence(monkeypatch, style):
    from dataclasses import replace
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from reachy_muse.contract.prompts import MORNING_BRIEFING_REQUEST
    from reachy_muse.conversation import _BriefingSlot
    from reachy_muse.morning_briefing import briefing_request
    from test_conversation import chat_event, sentence_frame

    sentences = ["Good morning.", "Finish one task first.", "Markets rose.",
                 "Chip stocks led the gains.", "A new phone shipped.", "Have a good day."]
    reply = ("".join(sentence_frame(text) for text in sentences)
             if style is ReplyStyle.EXPRESSIVE_JSON else " ".join(sentences))

    class BriefingSession(Session):
        async def send_chat(self, text, session_id, **options):
            response = await super().send_chat(text, session_id, **options)
            job = self.conversation.job
            await self.conversation._queue_replies(job.reader.read(job.turn.feed(
                chat_event("delta.message_done", "reply-1", session_id=session_id, content=reply))), job)
            return response

    class NoCondensing:
        async def lines(self, *args, **kwargs):
            raise AssertionError("briefings must bypass narration and condensation")

    async def check():
        session = BriefingSession()
        bundle = backends_for(Mode.ON_ROBOT, session, speech=object(), transcriber=object())
        conversation = VoiceConversation(session, Hardware(), session_id="owned-chat", owns_chat=True,
                                        backends=replace(bundle, local_style=style, narrator=NoCondensing()))
        session.conversation = conversation
        request = briefing_request(MORNING_BRIEFING_REQUEST,
                                   datetime(2026, 10, 24, 5, 20, tzinfo=ZoneInfo("America/Los_Angeles")))
        slot = _BriefingSlot(request, lambda: False)
        await conversation._collect_briefing(slot)
        return session.sent[0][0], slot.result.result()

    monkeypatch.setattr(muse_turn, "EMPTY_REPLY_GRACE_S", 0)
    sent, lines = asyncio.run(check())
    assert [line.text for line in lines] == sentences
    assert sent == (
        "This is Reachy's scheduled morning briefing, not a user speech transcript. "
        "It will be spoken aloud to the user as they walk into the room. It is Saturday, October 24, 05:20 local time. "
        "For this reply, speak for about 60 to 90 seconds, roughly 150 to 220 words. "
        "In this order: wish them a warm good morning; give one "
        "concrete piece of advice to help them keep their focus today; give the latest market news, with the major "
        "stock indices' latest moves and any notable stocks; then the latest technology news, two or three stories. "
        "Use plain, listenable sentences with no lists, headings, links or symbols, and say numbers the way a person "
        "would say them aloud. Check every market and news fact with your tools, and say plainly what you could not "
        "verify.")
