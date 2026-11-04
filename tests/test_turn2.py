import asyncio

import pytest

from bench.turn2 import answer_replay
from reachy_muse.contract.replies import ReplyStyle
from reachy_muse.conversation import NarrationSegment, SpeechSegment
from test_conversation import acknowledge, chat_event, feed, reply_job


def plain_job():
    job = reply_job("robot-chat", style=ReplyStyle.PLAIN_SHORT)
    acknowledge(job, {"message_id": "user-1"})
    return job


def test_plain_opening_plays_before_final_text_without_repeating_it():
    result = asyncio.run(answer_replay())
    assert result["spoken"] == [
        "A light workout is a good option.", "Take a rest day if you feel worn out.",
    ]
    assert result["first_sentence_to_audio_ms"] < 600


def test_companion_synthesizes_the_next_sentence_during_playback():
    result = asyncio.run(answer_replay(streamed=False))
    assert result["spoken"] == [
        "A light workout is a good option.", "Take a rest day if you feel worn out.",
    ]
    assert result["first_sentence_to_audio_ms"] < 600
    assert result["sentence_gap_ms"] < 150


def test_a_slow_answer_does_not_queue_a_redundant_introduction():
    result = asyncio.run(answer_replay(late=True))
    assert result["spoken"] == [
        "A light workout is a good option.", "Take a rest day if you feel worn out.",
    ]
    assert result["first_sentence_to_audio_ms"] < 600


def test_late_answer_still_names_its_topic_after_intervening_talk():
    from reachy_muse.capabilities import Mode
    from reachy_muse.conversation import VoiceConversation
    from reachy_muse.local_backends import backends_for
    from test_conversation import FakeHardware

    async def scenario():
        conversation = VoiceConversation(None, FakeHardware(), backends=backends_for(Mode.ON_ROBOT, None))
        job = plain_job()
        job.request = "Should I work out today?"
        job.dispatched_at = 0
        job.defer_playback = True
        job.context_changed = True
        await conversation._answer_begins(job)
        assert conversation._output_queue.get_nowait().text == "About Should I work out today:"
        await conversation._moments.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("style", [ReplyStyle.PLAIN_SHORT, ReplyStyle.EXPRESSIVE_JSON])
def test_inline_late_introduction_precedes_buffered_companion_answers(monkeypatch, style):
    from dataclasses import replace
    from types import SimpleNamespace

    from musegadget import muse_turn
    from reachy_muse.capabilities import Mode
    from reachy_muse.conversation import TurnOutcome, VoiceConversation
    from reachy_muse.failover import CompanionRoute, FailoverVoice
    from reachy_muse.local_backends import backends_for
    from test_conversation import sentence_frame
    from test_turn_pacing import Labels, PacedHardware

    monkeypatch.setattr(muse_turn, "REPLY_QUIET_S", 0)

    async def scenario():
        buffered, introducing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        labels = Labels()
        hardware = PacedHardware(labels)
        intro = "About Should I work out today:"

        class CompanionSpeech:
            available = True

            async def stream(self, line, rate):
                if line.text == intro:
                    introducing.set()
                    await release.wait()
                    yield labels.samples(line.text)
                else:
                    for index in range(129):
                        if index == 128:
                            buffered.set()
                        yield labels.samples(line.text, 1)

        route = CompanionRoute(SimpleNamespace(up=True))
        route.this_turn = True
        backends = backends_for(Mode.ON_ROBOT, None)
        voice = FailoverVoice(CompanionSpeech(), backends.voice, route)
        conversation = VoiceConversation(None, hardware, backends=replace(backends, voice=voice))
        job = reply_job("robot-chat", style=style)
        job.request = "Should I work out today?"
        acknowledge(job, {"message_id": "user-1"})

        def text(line):
            return sentence_frame(line) if style is ReplyStyle.EXPRESSIVE_JSON else line + " "

        async def feed_reply(event):
            await conversation._queue_replies(feed(job, event), job)

        async def finish_reply():
            await introducing.wait()
            await feed_reply(chat_event("delta.text_append", "reply-2", seq=3, text=text("Walk tomorrow.")))
            await feed_reply(chat_event("delta.message_done", "reply-2", seq=4, content=text("Walk tomorrow.")))
            release.set()

        speak = conversation._speech_with_timeout

        async def receive_during_handoff(speaking):
            if hardware.spoken() == [intro]:
                await feed_reply(chat_event("delta.text_append", "reply-3", seq=5, text=text("Sleep well.")))
                await feed_reply(chat_event("delta.message_done", "reply-3", seq=6, content=text("Sleep well.")))
                await feed_reply(chat_event("task.status", seq=7, status="completed"))
            await speak(speaking)
        monkeypatch.setattr(conversation, "_speech_with_timeout", receive_during_handoff)

        try:
            await feed_reply(chat_event("delta.text_append", "reply-1", text=text("Rest today.")))
            await asyncio.wait_for(buffered.wait(), 1)
            job.context_changed = True
            job.dispatched_at -= 21
            await feed_reply(chat_event("delta.message_done", "reply-1", seq=2, content=text("Rest today.")))
            outcome, _ = await asyncio.wait_for(asyncio.gather(conversation._deliver(job), finish_reply()), 2)
            assert outcome is TurnOutcome.ACCEPTED
            assert hardware.spoken() == [intro, "Rest today.", "Walk tomorrow.", "Sleep well."]
            assert sum(len(chunk) for chunk in hardware.played) == 467
            assert not conversation._prepared_speech and not route.speech_lock.locked()
        finally:
            await asyncio.gather(*(audio.aclose() for audio in conversation._prepared_speech.values()))
            await asyncio.gather(*conversation._closing_speech)
            await conversation._moments.close()
    asyncio.run(scenario())


def test_startup_warms_the_persistent_whisper_worker_before_fallback():
    from bench.turn2 import whisper_replay

    result = asyncio.run(whisper_replay())
    assert result["transcripts"] == ["Should I work out today?", "Should I work out today?"]
    assert result["worker_reused"] is True
    assert result["startup_ms"] >= 300
    assert result["first_fallback_ms"] < 150
    assert result["second_fallback_ms"] < 150


def test_plain_prefix_keeps_final_remainder_for_narration_once():
    job = plain_job()
    assert feed(job, chat_event("delta.text_append", "reply-1", text="Rest today. You")) == [
        SpeechSegment("reply-1", 0, "Rest today.")]
    assert feed(job, chat_event("delta.text_append", "reply-1", seq=2, text=" earned it. ")) == []
    assert feed(job, chat_event("delta.message_done", "reply-1", seq=3,
                               content="Rest today. You earned it.")) == [NarrationSegment("reply-1", 1)]
    assert job.reader.narration_text("reply-1") == "You earned it."
    assert feed(job, chat_event("delta.message_done", "reply-1", seq=4,
                               content="Rest today. You earned it.")) == []


@pytest.mark.parametrize("spoken", [False, True])
def test_final_replacement_discards_an_unheard_prefix_or_drops_a_spoken_messages_tail(spoken):
    job = plain_job()
    assert feed(job, chat_event("delta.text_append", "reply-1", text="Work out today. Draft tail. ")) == [
        SpeechSegment("reply-1", 0, "Work out today.")]
    stream = job.reader.streams["reply-1"]
    stream.spoken = int(spoken)
    result = feed(job, chat_event("delta.message_done", "reply-1", seq=2, content="Take a rest day."))
    if spoken:
        assert result == [] and stream.revised
    else:
        assert result == [NarrationSegment("reply-1", 1)] and stream.skip_before == 1
        assert job.reader.narration_text("reply-1") == "Take a rest day."


@pytest.mark.parametrize("draft", ["Dr. Smith says", "**Rest** today. Then walk. ",
                                   "Rest today", "```json\n", "[A source](https://example.com). "])
def test_incomplete_or_formatted_openings_keep_final_narration(draft):
    job = plain_job()
    assert feed(job, chat_event("delta.text_append", "reply-1", text=draft)) == []
    assert feed(job, chat_event("delta.message_done", "reply-1", seq=2, content="Take a rest day.")) == [
        NarrationSegment("reply-1", 1)]
    assert job.reader.narration_text("reply-1") == "Take a rest day."


def test_final_that_only_extends_an_uncommitted_fragment_is_allowed():
    job = plain_job()
    assert feed(job, chat_event("delta.text_append", "reply-1", text="Rest today. Tomorrow go")) == [
        SpeechSegment("reply-1", 0, "Rest today.")]
    job.reader.streams["reply-1"].spoken = 1
    assert feed(job, chat_event("delta.message_done", "reply-1", seq=2,
                               content="Rest today. Tomorrow take it easy.")) == [NarrationSegment("reply-1", 1)]
    assert job.reader.narration_text("reply-1") == "Tomorrow take it easy."


def test_leading_expression_marker_does_not_make_an_unchanged_final_look_revised():
    job = plain_job()
    assert feed(job, chat_event("delta.text_append", "reply-1",
                               text="[reachy:happy] Rest today. Tomorrow")) == [
        SpeechSegment("reply-1", 0, "Rest today.")]
    job.reader.streams["reply-1"].spoken = 1
    assert feed(job, chat_event("delta.message_done", "reply-1", seq=2,
                               content="[reachy:happy] Rest today. Walk tomorrow.")) == [NarrationSegment("reply-1", 1)]
    assert job.reader.narration_text("reply-1") == "Walk tomorrow."
