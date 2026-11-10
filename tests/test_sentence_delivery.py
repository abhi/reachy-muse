"""Literal expression and motion expectations through the shared playback owner."""

from dataclasses import replace

import numpy as np
import pytest

from reachy_muse.capabilities import Motion
from reachy_muse.contract.expressions import Expression
from reachy_muse.motion.prosody import Delivery, ProsodyTracker
from reachy_muse.motion.sentence import SentenceMotion, classify_sentence
from reachy_muse.motion.sentence_delivery import SentenceDelivery, SentenceSelector, neutral_motion
from reachy_muse.realtime_call import realtime_supported
from test_sentence_motion import controller


@pytest.mark.parametrize("text,expected", [
    ("Stocks gained two percent.", "happy"),
    ("Shares rallied to a record high.", "happy"),
    ("Stocks fell after the inflation report.", "sad"),
    ("Stock prices fell after the report.", "sad"),
    ("Share prices rose in afternoon trading.", "happy"),
    ("The company announced layoffs.", "sad"),
    ("Inflation fell again.", "happy"),
    ("Inflation rose again.", "sad"),
    ("Traffic deaths dropped this year.", "happy"),
    ("The temperature falls tonight.", "neutral"),
    ("Shares rose, but profits fell.", "neutral"),
    ("The company denied that profits fell.", "neutral"),
    ("No layoffs were announced.", "neutral"),
    ("The Great Depression lasted years.", "neutral"),
    ("Niagara Falls is on the border.", "neutral"),
    ("You've got this.", "happy"),
    ("Try saving a copy first.", "nod"),
    ("I'm not sure about the time.", "thinking"),
    ("Maybe we should wait.", "thinking"),
    ("Tell me more about your day.", "curious"),
    ("Shall I keep going?", "curious"),
    ("The discovery was astonishing!", "surprised"),
    ("There are two tabs in the file.", "neutral"),
])
def test_spoken_and_news_cues(text, expected):
    assert classify_sentence(text).expression.value == expected


def test_facts_cycle_without_changing_their_meaning_and_greetings_are_warm():
    selector = SentenceSelector()
    choices = [selector.choose(classify_sentence("The train leaves at six.")) for _ in range(6)]
    assert [(p.expression.value, p.variant) for p in choices] == [
        ("neutral", "attentive"), ("neutral", "curious_tilt"), ("neutral", "warm"),
        ("neutral", "attentive"), ("neutral", "curious_tilt"), ("neutral", "warm")]
    assert classify_sentence("Hello, here's your briefing.").variant == "warm"
    assert classify_sentence("Take care, see you later.").variant == "warm"


def test_delivered_prosody_can_engage_or_soften_but_cannot_invent_valence():
    sentence = classify_sentence("The next item is about the launch.")
    assert SentenceSelector().choose(sentence, Delivery(7, -18, 4, 3)).variant == "curious_tilt"
    assert SentenceSelector().choose(sentence, Delivery(2, -18, 4, 0)).variant == "warm"
    assert SentenceSelector().choose(sentence, Delivery(2, -32, 2, 0)).variant == "attentive"
    assert SentenceSelector().choose(sentence, Delivery(2, -18, 1, 0)).variant == "attentive"
    for text, expected in [("The flight was cancelled.", "sad"), ("You've got this.", "happy"),
                           ("Is that the right file?", "curious"), ("Maybe tomorrow.", "thinking")]:
        choice = SentenceSelector().choose(classify_sentence(text), Delivery(8, -10, 5, 4))
        assert choice.expression.value == expected
        assert choice.variant is None


def test_delivery_and_revisions_cannot_force_three_identical_neutral_gestures():
    selector = SentenceSelector()
    base = classify_sentence("The report is on the desk.")
    delivery = Delivery(2, -18, 4, 0)
    result = []
    for _ in range(6):
        selector.choose(base)
        result.append(selector.choose(base, delivery, revise=True).variant)
    assert result == ["warm", "warm", "attentive", "warm", "warm", "attentive"]
    assert [selector.choose(classify_sentence("I'm sorry for your loss.")).expression.value
            for _ in range(4)] == ["sad", "sad", "sad", "sad"]


def rising_pcm(rate=24000):
    t = np.arange(int(.8 * rate)) / rate
    return (.18 * np.sin(2 * np.pi * np.cumsum(140 + 140 * t / .8) / rate)
            * (.6 + .4 * np.sin(2 * np.pi * 4 * t) ** 2)).astype(np.float32)


def test_features_come_from_pcm_and_short_silent_or_unvoiced_chunks_do_not_vote():
    tracker = ProsodyTracker()
    pcm = rising_pcm()
    tracker.feed(pcm[:960] * 32767)
    assert tracker.delivery() is None
    tracker.feed(pcm[960:] * 32767)
    delivery = tracker.delivery()
    assert delivery.pitch_range == pytest.approx(9.0, abs=1)
    assert delivery.energy_dbfs == pytest.approx(-19.7, abs=.3)
    assert delivery.rise > 5
    assert 3 <= delivery.syllables_per_s <= 9
    for audio in [np.zeros(19200), np.full(19200, 4000), np.random.default_rng(7).normal(0, 2000, 19200)]:
        noise = ProsodyTracker()
        noise.feed(audio)
        assert noise.delivery() is None


def test_shared_delivery_refines_once_and_clear_drops_old_audio():
    delivery = SentenceDelivery()
    pcm = rising_pcm()
    base = classify_sentence("The report has three sections.")
    initial = delivery.accept(pcm[:960], 24000, base)
    assert (initial.expression.value, initial.variant, initial.revised) == ("neutral", "attentive", False)
    refined = delivery.accept(pcm[960:], 24000, None)
    assert (refined.expression.value, refined.variant, refined.revised) == ("neutral", "curious_tilt", True)
    assert delivery.accept(pcm, 24000, None) is None
    delivery.clear()
    assert delivery.accept(pcm, 24000, None) is None
    assert delivery.accept(pcm, 24000, classify_sentence("The market slumped.")).expression.value == "sad"


def test_late_lexical_correction_wins_over_active_prosody_and_preserves_sentence_count():
    delivery = SentenceDelivery()
    pcm = rising_pcm()
    delivery.accept(pcm[:960], 24000, classify_sentence("The news"))
    corrected = delivery.accept(pcm[960:], 24000, replace(classify_sentence("The news is terrible."), revised=True))
    assert (corrected.expression.value, corrected.variant, corrected.revised) == ("sad", None, True)
    assert delivery.selector.recent == ["sad"]


def test_subtle_fallback_and_library_variants_stay_within_calibrated_bounds():
    selector = SentenceSelector()
    plans = [selector.choose(classify_sentence("The file is ready.")) for _ in range(3)]
    peaks = [np.max(np.abs(neutral_motion(p.variant).frames), axis=0).tolist() for p in plans]
    assert peaks == [[0, 0, 0, 0, 3, 0, 2, 2, 0], [0, 0, 0, 5, 2, 0, 6, 3, 0],
                     [0, 0, 2, 3, 0, 0, 8, 8, 0]]
    for plan in plans:
        frames = neutral_motion(plan.variant).frames
        np.testing.assert_allclose(frames[0], [0] * 9, atol=1e-10)
        np.testing.assert_allclose(frames[-1], [0] * 9, atol=1e-10)
    loud = replace(plans[0], motion=Motion(25, ((50.,) * 9, (-50.,) * 9)))
    hardware = controller()
    hardware.play_audio(np.zeros(10), sentence=loud)
    frames = hardware._sentences.due(100000000)[0].move.frames
    assert frames.tolist() == [[1, 1, 1, 1, 3, 1, 3, 3, 0], [-1, -1, -1, -1, -3, -1, -3, -3, 0]]


def test_actual_speaker_push_precedes_analysis_and_refinement_keeps_the_sentence_serial(monkeypatch):
    hardware = controller()
    hardware._output_sample_rate = 24000
    hardware._start_music()
    observed = []
    hardware._robot.media.push_audio_sample = lambda audio: observed.append(hardware._sentence_delivery.samples)
    monkeypatch.setattr("reachy_muse.hardware.time.monotonic", lambda: 10.)
    pcm = rising_pcm()
    hardware.play_audio(pcm[:960], sentence=classify_sentence("The report is ready."))
    hardware.play_audio(pcm[960:])
    hardware.finish_sentence()
    events = hardware._sentences.due(20)
    assert observed == [0, 960]
    assert [(e.kind, e.value) for e in events] == [
        ("start", ("neutral", 1)), ("start", ("neutral", 1)), ("end", ("plain", 1))]
    assert np.max(events[0].move.frames[:, 4]) == 3
    assert np.max(events[1].move.frames[:, 3]) == 5


def test_command_preemption_cannot_restart_from_old_prosody():
    hardware = controller()
    hardware._output_sample_rate = 24000
    hardware._start_music()
    pcm = rising_pcm()
    hardware.play_audio(pcm[:960], sentence=classify_sentence("The report is ready."))
    hardware._sentences.cancel()
    hardware.play_audio(pcm[960:])
    assert hardware._sentences.due(100000000) == []
    hardware.play_audio(pcm[:960], sentence=classify_sentence("I'm sorry."))
    assert [(e.kind, e.value) for e in hardware._sentences.due(100000000)] == [("start", ("sad", 2))]


@pytest.mark.parametrize("path", ["chat", "realtime"])
def test_chat_and_transcript_delta_replies_share_the_same_neutral_rotation(path):
    if path == "realtime" and not realtime_supported():
        pytest.skip("the Muse SDK has no voice call events")
    import asyncio
    from musegadget import voice_call as sdk
    from bench.expression_playback import Hardware
    from reachy_muse.capabilities import Mode
    from reachy_muse.conversation import VoiceConversation
    from reachy_muse.local_backends import backends_for
    from reachy_muse.motion.reply_motion import ReplyMotion
    from reachy_muse.realtime_call import MicrophoneFeed, RealtimeCall, RealtimeSpeaker

    async def scenario():
        hardware = Hardware()
        texts = ["The file is ready.", "The train leaves at six.", "The report has two sections."]
        if path == "chat":
            conversation = VoiceConversation(None, hardware, backends=backends_for(Mode.MUSE_VOICE, None))
            async def speech():
                yield np.zeros(320, dtype=np.float32)
            try:
                for text in texts:
                    await conversation._speak_owned("briefing", text=text, prepared_stream=speech(),
                                                   stream_segment=True, drain=False)
            finally:
                await conversation._moments.close()
        else:
            async def nothing():
                pass
            motion = ReplyMotion()
            speaker = RealtimeSpeaker(hardware, motion, on_audible=lambda *args: None,
                                      on_start=nothing, on_idle=nothing)
            call = RealtimeCall(MicrophoneFeed(16000), speaker, motion, on_working=lambda working: None,
                                on_reply_done=lambda reply: None)
            try:
                for i, text in enumerate(texts):
                    name = str(i)
                    call.on_event(sdk.ResponseStarted(name))
                    call.on_event(sdk.TranscriptDelta(name, text + " "))
                    call.on_event(sdk.AudioChunk(name, 0, bytes(4800 * 2)))
                    call.on_event(sdk.TranscriptDone(name, text + " "))
                    call.on_event(sdk.ResponseDone(name, "completed"))
                    await asyncio.wait_for(speaker.drain(), 2)
            finally:
                await call.close()
        events = hardware.controller._sentences.due(100000000)
        return [(e.value, np.max(np.abs(e.move.frames), axis=0).tolist())
                for e in events if e.kind == "start"]

    assert asyncio.run(scenario()) == [
        (("neutral", 1), [0, 0, 0, 0, 3, 0, 2, 2, 0]),
        (("neutral", 2), [0, 0, 0, 5, 2, 0, 6, 3, 0]),
        (("neutral", 3), [0, 0, 2, 3, 0, 0, 8, 8, 0])]


def test_flat_delivery_does_not_restart_a_factual_gesture():
    selector = SentenceSelector()
    sentence = classify_sentence("The file is on the desk.")
    assert selector.choose(sentence).variant == "attentive"
    assert selector.choose(sentence, Delivery(1, -23, 3, 0), revise=True).variant == "attentive"


def test_a_greeting_split_across_realtime_deltas_corrects_the_playing_variant():
    if not realtime_supported():
        pytest.skip("the Muse SDK has no voice call events")
    import asyncio
    from musegadget import voice_call as sdk
    from bench.expression_playback import Hardware
    from reachy_muse.motion.reply_motion import ReplyMotion
    from reachy_muse.realtime_call import MicrophoneFeed, RealtimeCall, RealtimeSpeaker

    async def scenario():
        hardware = Hardware()
        motion = ReplyMotion()

        async def nothing():
            pass

        speaker = RealtimeSpeaker(hardware, motion, on_audible=lambda *args: None,
                                  on_start=nothing, on_idle=nothing)
        call = RealtimeCall(MicrophoneFeed(16000), speaker, motion, on_working=lambda working: None,
                            on_reply_done=lambda reply: None)
        try:
            call.on_event(sdk.ResponseStarted("r1"))
            call.on_event(sdk.TranscriptDelta("r1", "Good "))
            call.on_event(sdk.AudioChunk("r1", 0, bytes(4800 * 2)))
            while not hardware.pushes:
                await asyncio.sleep(0)
            call.on_event(sdk.TranscriptDelta("r1", "morning. "))
            call.on_event(sdk.AudioChunk("r1", 4800, bytes(4800 * 2)))
            call.on_event(sdk.TranscriptDone("r1", "Good morning. "))
            call.on_event(sdk.ResponseDone("r1", "completed"))
            await asyncio.wait_for(speaker.drain(), 2)
            events = hardware.controller._sentences.due(100000000)
            return ([(e.value, np.max(np.abs(e.move.frames), axis=0).tolist())
                     for e in events if e.kind == "start"],
                    hardware.controller._sentence_delivery.selector.recent)
        finally:
            await call.close()

    assert asyncio.run(asyncio.wait_for(scenario(), 3)) == ([
        (("neutral", 1), [0, 0, 0, 0, 3, 0, 2, 2, 0]),
        (("neutral", 1), [0, 0, 2, 3, 0, 0, 8, 8, 0]),
    ], ["warm"])


def test_an_unplayed_or_empty_chunk_does_not_consume_a_variant():
    hardware = controller()
    hardware._output_sample_rate = 24000
    hardware._start_music()
    plan = classify_sentence("The report is on the desk.")
    hardware.play_audio(np.array([], dtype=np.float32), sentence=plan)
    hardware._robot.media.fail = True
    with pytest.raises(Exception, match="push failed"):
        hardware.play_audio(np.ones(960, dtype=np.float32), sentence=plan)
    hardware._robot.media.fail = False
    hardware.play_audio(np.zeros(960, dtype=np.float32), sentence=plan)
    event = hardware._sentences.due(100000000)[0]
    assert event.value == ("neutral", 1)
    assert np.max(event.move.frames[:, 4]) == 3


def test_a_late_transcript_revision_cannot_revive_a_cleared_sentence():
    delivery = SentenceDelivery()
    pcm = rising_pcm()
    delivery.accept(pcm[:960], 24000, classify_sentence("The report"))
    delivery.clear()
    revision = replace(classify_sentence("The report is surprising!"), revised=True)
    assert delivery.accept(pcm[960:], 24000, revision) is None
    assert delivery.accept(pcm, 24000, classify_sentence("Sorry, the train is cancelled.")).expression.value == "sad"
