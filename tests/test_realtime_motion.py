"""Reachy's own motion for Muse's realtime replies: sentences, their audio starts, and prosody accents."""

import asyncio

import numpy as np

from reachy_muse.contract.expressions import Expression
from reachy_muse.motion.prosody import Accent, ProsodyTracker
from reachy_muse.motion.reply_alignment import ReplyAligner, Sentence, SentenceCutter
from reachy_muse.motion.reply_motion import ReplyMotion

RATE = 24000


def voiced(seconds, *, f0=150, amp=4000):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * f0 * t) * (.6 + .4 * np.sin(2 * np.pi * 4 * t) ** 2)).astype(np.int16)


def quiet(seconds):
    return np.zeros(int(seconds * RATE), np.int16)


def chunks(pcm, size=4800):
    for offset in range(0, len(pcm), size):
        yield offset, pcm[offset:offset + size].astype("<i2").tobytes()


def test_sentences_are_cut_from_deltas_with_their_character_offsets_and_expected_pauses():
    cutter = SentenceCutter()
    assert cutter.feed("Sure. The flight") == [Sentence(0, "Sure.", 0)]
    assert cutter.feed(" leaves at 9 a.m. on Monday! Ok") == [
        Sentence(1, "The flight leaves at 9 a.m. on Monday!", 6, .35)]
    assert cutter.pending() == Sentence(2, "Ok", 45, .7)
    assert cutter.finish("Sure. The flight leaves at 9 a.m. on Monday! Ok then") == [Sentence(2, "Ok then", 45, .7)]
    assert cutter.pause_s(len(cutter.text), 2) == .7


def test_a_sentence_start_is_estimated_then_snapped_to_the_pause_once_its_audio_arrives():
    pcm = np.concatenate([voiced(1.0), quiet(.4), voiced(1.0)])
    aligner = ReplyAligner()
    second = Sentence(1, "Next one here.", 18, .35)
    aligner.audio(0, pcm[:RATE // 2])
    assert aligner.start(Sentence(0, "Hello there now.", 0)) == (0, True)
    assert aligner.start(second) == (37200, False)
    aligner.audio(RATE // 2, pcm[RATE // 2:])
    assert aligner.start(second) == (33600, True)


def test_stress_and_a_rising_question_are_found_in_the_pcm():
    stressed = voiced(3.0, amp=2000)
    stressed[int(1.5 * RATE):int(1.6 * RATE)] = voiced(.1, amp=12000)
    assert ProsodyTracker().feed(stressed) == [Accent(37440, "exclamation")]

    t = np.arange(int(.8 * RATE)) / RATE
    rise = (3000 * np.sin(2 * np.pi * np.cumsum(150 + 100 * t / .8) / RATE)).astype(np.int16)
    tracker = ProsodyTracker()
    found = []
    for _, pcm in chunks(np.concatenate([voiced(1.0, amp=3000), rise, quiet(.5)]), 960):
        found += tracker.feed(np.frombuffer(pcm, "<i2"))
    assert found == [Accent(40560, "question")]


def reply_pcm():
    return np.concatenate([voiced(2.0), quiet(.35), voiced(1.2), quiet(.2)])


def test_stress_detected_behind_playback_is_delivered_once_at_the_current_chunk():
    motion = ReplyMotion()
    stressed = voiced(3.0, amp=2000)
    stressed[int(1.5 * RATE):int(1.6 * RATE)] = voiced(.1, amp=12000)
    marks = play_from(motion, "r1", stressed, 0)
    assert [(at, mark.accent) for at, mark in marks if mark.accent] == [(38400, "exclamation")]


def test_accents_from_40ms_audio_chunks_are_delivered_once_and_stale_accents_are_dropped():
    motion = ReplyMotion()
    stressed = voiced(3.0, amp=2000)
    stressed[int(1.5 * RATE):int(1.6 * RATE)] = voiced(.1, amp=12000)
    marks = []
    for offset, data in chunks(stressed, 960):
        motion.audio("r1", offset, data)
        marks += [(offset + mark.sample, mark.accent) for mark in motion.marks("r1", offset, 960) if mark.accent]
    assert marks == [(40320, "exclamation")]
    motion.audio("r2", 0, stressed.astype("<i2").tobytes())
    assert motion.marks("r2", 60000, 960) == []


def play(motion, response_id, pcm):
    """Muse's audio arrives ahead of playback, as it does on a call; marks are asked chunk by chunk."""
    for offset, data in chunks(pcm):
        motion.audio(response_id, offset, data)
    marks = []
    for offset, data in chunks(pcm):
        marks += [(offset + mark.sample, mark) for mark in motion.marks(response_id, offset, len(data) // 2)]
    motion.audio_done(response_id)
    return marks


def test_each_sentence_gets_its_expression_at_the_sample_its_speech_starts():
    async def scenario():
        motion = ReplyMotion()
        motion.started("r1")
        motion.words("r1", "Great news, your flight is booked! ")
        motion.words("r1", "Want a window seat?")
        motion.transcript("r1", "Great news, your flight is booked! Want a window seat?")
        marks = play(motion, "r1", reply_pcm())
        sentences = [(at, mark.text, mark.sentence.expression, mark.sentence.terminal)
                     for at, mark in marks if mark.text is not None]
        assert sentences == [
            (0, "Great news, your flight is booked!", Expression.HAPPY, "exclamation"),
            (56400, "Want a window seat?", Expression.CURIOUS, "question"),
        ]
        assert [(item.index, item.sample, item.confirmed, item.late) for item in motion.scheduled("r1")] == [
            (0, 0, True, False), (1, 56400, True, False)]
        assert motion.closing("r1") is Expression.HAPPY

    asyncio.run(scenario())


def test_a_complete_transcript_before_streamed_pcm_keeps_the_later_sentence_at_its_speech_start():
    motion = ReplyMotion()
    motion.transcript("r1", "Great news, your flight is booked! Want a window seat?")
    pcm = reply_pcm()
    motion.audio("r1", 0, pcm[:18000].astype("<i2").tobytes())
    marks = [(offset + mark.sample, mark.text)
             for offset in (0, 4800, 9600, 14400)
             for mark in motion.marks("r1", offset, min(4800, 18000 - offset)) if mark.text is not None]
    motion.audio("r1", 18000, pcm[18000:].astype("<i2").tobytes())
    marks += [(offset + mark.sample, mark.text)
              for offset in range(18000, len(pcm), 4800)
              for mark in motion.marks("r1", offset, min(4800, len(pcm) - offset)) if mark.text is not None]
    assert marks == [(0, "Great news, your flight is booked!"), (56400, "Want a window seat?")]


def test_a_sentence_whose_words_come_late_starts_where_playback_is():
    async def scenario():
        motion = ReplyMotion()
        pcm = voiced(2.0)
        marks = play(motion, "r1", pcm[:24000])
        motion.words("r1", "Sorry, that train was cancelled.")
        marks += play_from(motion, "r1", pcm, 24000)
        sentences = [(at, mark.sentence.expression) for at, mark in marks if mark.text is not None]
        assert sentences == [(24000, Expression.SAD)]
        assert [(item.sample, item.late, item.shown) for item in motion.scheduled("r1")] == [(24000, True, True)]

    asyncio.run(scenario())


def play_from(motion, response_id, pcm, start):
    marks = []
    for offset, data in chunks(pcm[start:]):
        motion.audio(response_id, start + offset, data)
        marks += [(start + offset + mark.sample, mark)
                  for mark in motion.marks(response_id, start + offset, len(data) // 2)]
    return marks


def test_while_muse_moves_reachy_itself_local_expressions_and_accents_hold_back():
    async def scenario():
        moving = True
        motion = ReplyMotion(muse_moving=lambda: moving)
        stressed = voiced(3.0, amp=2000)
        stressed[int(1.5 * RATE):int(1.6 * RATE)] = voiced(.1, amp=12000)
        motion.words("r1", "Yes, I can do that.")
        for offset, data in chunks(stressed):
            motion.audio("r1", offset, data)
        marks = []
        for offset, data in chunks(stressed):
            moving = offset < 24000 or 48000 <= offset
            marks += [(offset + mark.sample, mark.text, mark.sentence, mark.accent)
                      for mark in motion.marks("r1", offset, len(data) // 2)]
        assert marks == [(0, "Yes, I can do that.", None, None), (37440, None, None, "exclamation")]

    asyncio.run(scenario())


def test_a_flushed_reply_forgets_its_sentences():
    async def scenario():
        motion = ReplyMotion()
        motion.words("r1", "Happy birthday! ")
        play(motion, "r1", voiced(1.0))
        motion.flushed(("r1",))
        assert (motion.scheduled("r1"), motion.closing("r1")) == ([], None)

    asyncio.run(scenario())


def test_a_sentence_started_from_its_first_words_takes_the_feeling_its_rest_reveals():
    async def scenario():
        motion = ReplyMotion()
        pcm = voiced(2.0)
        motion.words("r1", "I'm ")
        for offset, data in chunks(pcm):
            motion.audio("r1", offset, data)
        marks = [(mark.sample, mark.text, mark.sentence.expression, mark.revised)
                 for mark in motion.marks("r1", 0, 4800)]
        motion.words("r1", "sorry, that train was cancelled. ")
        marks += [(4800 + mark.sample, mark.text, mark.sentence.expression, mark.revised)
                  for mark in motion.marks("r1", 4800, 4800)]
        assert marks == [(0, "I'm", Expression.NEUTRAL, False), (4800, None, Expression.SAD, True)]
        assert motion.closing("r1") is Expression.SAD

    asyncio.run(scenario())


def test_a_reply_s_motion_is_summarized_in_one_line():
    async def scenario():
        motion = ReplyMotion()
        motion.words("r1", "Great news, your flight is booked! Want a window seat?")
        motion.transcript("r1", "Great news, your flight is booked! Want a window seat?")
        play(motion, "r1", reply_pcm())
        return motion.report("r1"), motion.report("r2")

    assert asyncio.run(scenario()) == (
        "2 sentences at 0.00 2.35 s (2 pause-confirmed, 0 late, 0 held for Muse), expressions happy curious, "
        "0 accents", None)
