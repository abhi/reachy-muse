import asyncio

import pytest

from bench.turn2 import answer_replay
from reachy_muse.contract.replies import ReplyStyle
from reachy_muse.conversation import SpeechSegment
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
                               content="Rest today. You earned it.")) == ["reply-1"]
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
        assert result == ["reply-1"] and stream.skip_before == 1
        assert job.reader.narration_text("reply-1") == "Take a rest day."


@pytest.mark.parametrize("draft", ["Dr. Smith says", "**Rest** today. Then walk. ",
                                   "Rest today", "```json\n", "[A source](https://example.com). "])
def test_incomplete_or_formatted_openings_keep_final_narration(draft):
    job = plain_job()
    assert feed(job, chat_event("delta.text_append", "reply-1", text=draft)) == []
    assert feed(job, chat_event("delta.message_done", "reply-1", seq=2, content="Take a rest day.")) == ["reply-1"]
    assert job.reader.narration_text("reply-1") == "Take a rest day."


def test_final_that_only_extends_an_uncommitted_fragment_is_allowed():
    job = plain_job()
    assert feed(job, chat_event("delta.text_append", "reply-1", text="Rest today. Tomorrow go")) == [
        SpeechSegment("reply-1", 0, "Rest today.")]
    job.reader.streams["reply-1"].spoken = 1
    assert feed(job, chat_event("delta.message_done", "reply-1", seq=2,
                               content="Rest today. Tomorrow take it easy.")) == ["reply-1"]
    assert job.reader.narration_text("reply-1") == "Tomorrow take it easy."


def test_leading_expression_marker_does_not_make_an_unchanged_final_look_revised():
    job = plain_job()
    assert feed(job, chat_event("delta.text_append", "reply-1",
                               text="[reachy:happy] Rest today. Tomorrow")) == [
        SpeechSegment("reply-1", 0, "Rest today.")]
    job.reader.streams["reply-1"].spoken = 1
    assert feed(job, chat_event("delta.message_done", "reply-1", seq=2,
                               content="[reachy:happy] Rest today. Walk tomorrow.")) == ["reply-1"]
    assert job.reader.narration_text("reply-1") == "Walk tomorrow."
