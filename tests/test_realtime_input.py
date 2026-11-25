"""The samples Muse receives when the speaker leaks into Reachy's microphone."""

import pytest

np = pytest.importorskip("numpy")

from reachy_muse.follow_up import Spoken
from reachy_muse.realtime_call import CallInput


class Feed:
    def __init__(self):
        self.samples = []

    def push(self, samples, *, silent=False):
        self.samples.extend(samples.tolist())


@pytest.mark.parametrize("text", ["", "sunny", "It will be sunny tomorrow", "stop talking"])
def test_even_loud_sustained_echo_and_unknown_noise_stay_out_of_muse_s_input(text):
    feed = Feed()
    gate = CallInput(feed, 1000)
    for _ in range(60):
        assert gate.push(np.ones(100), speaking=True, text=text, playback=True,
                         heard_over=("It will be sunny tomorrow. You can ask me to stop talking.",)) is False
    assert feed.samples == [0.0] * 6000
    assert gate.confirmed is False


def test_distinct_sustained_user_words_release_the_opening_audio_before_barge_in():
    feed = Feed()
    gate = CallInput(feed, 1000)
    seen = []
    for text in ("", "Actually", "Actually tell me about Friday", "Actually tell me about Friday"):
        seen.append(gate.push(np.full(100, .5), speaking=True, text=text, playback=True,
                              heard_over=("It will be sunny tomorrow.",)))
    assert seen == [False, False, False, True]
    assert feed.samples == [0.0] * 300 + [.5] * 400
    assert gate.confirmed is True
    assert gate.release() is False
    gate.discard()
    assert gate.confirmed is True
    gate.reset()
    assert gate.push(np.ones(100), speaking=True, text="sunny", playback=True,
                     heard_over=("It will be sunny tomorrow.",)) is False
    assert feed.samples[-100:] == [0.0] * 100


def test_audio_without_reply_words_cannot_authorize_an_interruption():
    feed = Feed()
    gate = CallInput(feed, 1000)
    for _ in range(10):
        gate.push(np.ones(100), speaking=True, text="good question", playback=True, heard_over=())
    assert feed.samples == [0.0] * 1000
    assert gate.confirmed is False


def test_an_idle_microphone_has_no_recognition_delay():
    feed = Feed()
    gate = CallInput(feed, 1000)
    assert gate.push(np.full(100, .5), speaking=True, text="", playback=False, heard_over=()) is True
    assert feed.samples == [.5] * 100


def test_a_transient_distinct_hypothesis_does_not_interrupt_echo():
    feed = Feed()
    gate = CallInput(feed, 1000)
    for text in ("sunny tomorrow", "sunny tomorrow", "turn around", "sunny tomorrow"):
        assert gate.push(np.ones(100), speaking=True, text=text, playback=True,
                         heard_over=("It will be sunny tomorrow.",)) is False
    assert feed.samples == [0.0] * 400


@pytest.mark.parametrize("text, playback", [
    ("Привет мир", "Привет мир!"),
    ("你好 世界", "你好 世界！"),
    ("' ' ___", "It will be sunny tomorrow."),
])
def test_multilingual_echo_and_empty_normalized_words_never_release_audio(text, playback):
    feed = Feed()
    gate = CallInput(feed, 1000)
    for _ in range(4):
        gate.push(np.ones(100), speaking=True, text=text, playback=True, heard_over=(playback,))
    assert feed.samples == [0.0] * 400
    assert gate.confirmed is False


@pytest.mark.parametrize("muse_words, held", [("What", True), ("sunny tomorrow", False)],
                         ids=["one word", "two words in a row"])
def test_one_of_muse_s_words_keeps_the_voice_held_under_it_but_two_in_a_row_erase_it(muse_words, held):
    """A request often opens with a word Muse's reply also holds, which could be its echo: the voice heard under that
    word stays held, though it counts toward no confirmation, and reaches Muse with the request. Two of Muse's words
    in a row are its echo, and erase what is held."""
    feed = Feed()
    gate = CallInput(feed, 1000)
    reply = ("What a day: it will be sunny tomorrow.",)
    for text, value in (("", .25), (muse_words, .25), (muse_words, .25),
                        ("What about Friday", .5), ("What about Friday", .5), ("What about Friday", .5)):
        gate.push(np.full(100, value), speaking=True, text=text, playback=True, heard_over=reply)
    assert gate.confirmed is True
    assert feed.samples == [0.0] * 500 + ([.25] * 300 if held else []) + [.5] * 300


@pytest.mark.parametrize("user_samples", [2048, 6144])
def test_echo_before_user_speech_neither_counts_toward_interruption_nor_reaches_muse(user_samples):
    feed = Feed()
    gate = CallInput(feed, 16000)
    for _ in range(31):
        gate.push(np.ones(512), speaking=True, text="sunny tomorrow", playback=True,
                  heard_over=("It will be sunny tomorrow.",))
    for _ in range(user_samples // 512):
        gate.push(np.full(512, .5), speaking=True, text="stop talking", playback=True,
                  heard_over=("It will be sunny tomorrow.",))
    if user_samples == 2048:
        assert feed.samples == [0.0] * 17920
        assert gate.confirmed is False
    else:
        assert feed.samples == [0.0] * 20480 + [.5] * 6144
        assert gate.confirmed is True


@pytest.mark.parametrize("quiet_frames, interrupted", [(0, True), (10, False)])
def test_delayed_words_can_confirm_speech_only_within_the_recognition_window(quiet_frames, interrupted):
    feed = Feed()
    gate = CallInput(feed, 1000)
    for _ in range(5):
        gate.push(np.full(100, .5), speaking=True, text="", playback=True, heard_over=("Sunny tomorrow.",))
    for _ in range(quiet_frames):
        gate.push(np.zeros(100), speaking=False, text="", playback=True, heard_over=("Sunny tomorrow.",))
    for _ in range(2):
        gate.push(np.zeros(100), speaking=False, text="stop talking", playback=True,
                  heard_over=("Sunny tomorrow.",))
    assert gate.confirmed is interrupted
    assert feed.samples == ([0.0] * 600 + [.5] * 500 + [0.0] * 200 if interrupted else [0.0] * 1700)


def held_turn(gate, *, voice, partial):
    """Four words in 200 ms voiced stretches 100 ms apart, from time 0, then the robot's 2 s of silence before its
    endpoint."""
    reply = ("It will be sunny tomorrow.",)
    for index, voiced in enumerate([voice, voice, False] * 4 + [False] * 19, 1):
        gate.push(np.full(100, .5) if voiced else np.zeros(100), speaking=voiced, text=partial, playback=True,
                  heard_over=reply, at=index / 10)


# Muse's reply, as Reachy's speaker played it while the voice of held_turn was heard.
PLAYED = (Spoken("It will be sunny tomorrow.", 0.0, 1.5),)


def test_a_turn_s_final_words_release_all_its_voice_after_the_recognition_window():
    """No stretch lasts 300 ms, so no partial confirms the words, and the recognition window closes long before the
    endpoint. The turn's final words still confirm, but the audio waits until the turn is Muse's: Muse then gets
    every word with the pauses the gate held, and after them the speech that ran on past the endpoint."""
    feed = Feed()
    gate = CallInput(feed, 1000)
    reply = ("It will be sunny tomorrow.",)
    held_turn(gate, voice=True, partial="What about")
    assert (gate.confirmed, feed.samples) == (False, [0.0] * 3100)
    assert gate.finish("What about Friday then?", reply, PLAYED) is True
    gate.push(np.full(100, .25), speaking=True, text="", playback=True, heard_over=reply)
    assert feed.samples == [0.0] * 3200
    assert gate.release() is True
    gate.push(np.full(100, .75), speaking=True, text="", playback=True, heard_over=reply)
    assert feed.samples == ([0.0] * 3200 + ([.5] * 200 + [0.0] * 100) * 3 + [.5] * 200 + [0.0] * 700
                            + [.25] * 100 + [.75] * 100)


def test_a_turn_reachy_takes_at_its_endpoint_never_reaches_muse_and_later_speech_must_confirm_afresh():
    """The final words confirm a device command, which Reachy takes itself: its audio, and the speech that ran on past
    its endpoint, never reach Muse, and the speech after that must confirm on its own."""
    feed = Feed()
    gate = CallInput(feed, 1000)
    reply = ("It will be sunny tomorrow.",)
    held_turn(gate, voice=True, partial="Turn the")
    assert gate.finish("Turn the volume down.", reply, PLAYED) is True
    gate.push(np.full(100, .25), speaking=True, text="", playback=True, heard_over=reply)
    gate.discard()
    assert (gate.confirmed, gate.release()) == (False, False)
    for _ in range(4):
        gate.push(np.full(100, .75), speaking=True, text="", playback=True, heard_over=reply)
    assert (gate.confirmed, feed.samples) == (False, [0.0] * 3600)


@pytest.mark.parametrize("voice, partial, final", [
    (True, "", "sunny tomorrow"), (True, "", "It'll be sunny tomorrow."), (True, "", "Okay."),
    (False, "", "What about Friday then?"), (True, "sunny tomorrow", "What about Friday then?")],
    ids=["echo words", "echo with a word written differently", "one word", "no voice",
         "voice held only under echo words"])
def test_a_turn_s_final_words_release_nothing_without_distinct_words_over_held_voice(voice, partial, final):
    feed = Feed()
    gate = CallInput(feed, 1000)
    held_turn(gate, voice=voice, partial=partial)
    assert gate.finish(final, ("It will be sunny tomorrow.",), PLAYED) is False
    assert feed.samples == [0.0] * 3100


@pytest.mark.parametrize("offered_at, confirmed", [(-4.0, True), (0.0, False)],
                         ids=["Muse made the offer before", "Muse made the offer as the voice was heard"])
def test_final_words_are_muse_s_echo_only_if_muse_played_them_while_their_voice_was_heard(offered_at, confirmed):
    """The final words take up Muse's offer in Muse's own words, as its echo would hold them. Echo holds only words
    Muse played while the voice was heard: if Muse made the offer before then, the user is taking it up."""
    gate = CallInput(Feed(), 1000)
    offer = "Or I can set a reminder for you."
    held_turn(gate, voice=True, partial="Yes")
    played = (Spoken(offer, offered_at, offered_at + 1.5),)
    assert gate.finish("Yes, set a reminder for me.", (offer,), played) is confirmed


def test_final_words_are_weighed_against_what_muse_played_since_their_voice_began_though_its_echo_reset_the_gate():
    """Muse's echo, written at first as Muse's very words, erases what the gate holds until the companion gets a word
    wrong. The turn's final words still cover the echo from its start, so they are weighed against what Muse played
    from then: they are its echo, and confirm nothing."""
    gate = CallInput(Feed(), 1000)
    line = "Tell me what you're wearing and I'll suggest something for tonight."
    for index in range(1, 41):
        partial = "Tell me what you're wearing" if index < 20 else "Tell me what you're wearing and I just"
        gate.push(np.full(100, .5) if index % 3 else np.zeros(100), speaking=bool(index % 3), text=partial,
                  playback=True, heard_over=(line,), at=index / 10)
    assert gate.finish("Tell me what you're wearing and I just suggest some four tonight", (line,),
                       (Spoken(line, 0.0, 4.5),)) is False


@pytest.mark.parametrize("words, kept", [("", False), ("Reachy,", True)], ids=["no words", "a word"])
def test_voice_held_past_the_recognition_window_stays_only_if_words_were_heard_in_it(words, kept):
    """A sound, then a pause past the recognition window, then a request that a partial confirms. Voice the recognizer
    heard a word in waits with the request for Muse, though the partial is gone by then, as at a turn's end. Sound it
    heard no word in, such as Muse's echo, is dropped."""
    feed = Feed()
    gate = CallInput(feed, 1000)
    reply = ("It will be sunny tomorrow.",)
    for _ in range(3):
        gate.push(np.full(100, .25), speaking=True, text=words, playback=True, heard_over=reply)
    for _ in range(10):
        gate.push(np.zeros(100), speaking=False, text="", playback=True, heard_over=reply)
    for _ in range(3):
        gate.push(np.full(100, .5), speaking=True, text=f"{words} what about Friday".strip(), playback=True,
                  heard_over=reply)
    assert gate.confirmed is True
    assert feed.samples == [0.0] * 1500 + ([.25] * 300 + [0.0] * 700 if kept else [0.0] * 300) + [.5] * 300


def test_separate_short_sounds_do_not_combine_into_sustained_speech():
    feed = Feed()
    gate = CallInput(feed, 1000)
    for _ in range(2):
        for _ in range(2):
            gate.push(np.ones(100), speaking=True, text="stop talking", playback=True,
                      heard_over=("Sunny tomorrow.",))
        gate.push(np.zeros(100), speaking=False, text="stop talking", playback=True,
                  heard_over=("Sunny tomorrow.",))
    assert feed.samples == [0.0] * 600
    assert gate.confirmed is False


def test_a_separate_streaming_command_model_preserves_whisper_for_chat_only():
    import asyncio

    from reachy_muse.local_backends import LocalHearing
    from test_pins import Vad
    from test_realtime_turns import StreamingTranscripts

    class Batch:
        calls = 0

        async def transcribe(self, wav):
            self.calls += 1
            return "Tell me a story."

    async def scenario():
        batch = Batch()
        hearing = LocalHearing(batch, command_transcriber=StreamingTranscripts("volume down"))
        command = hearing.open_turn(16000, silence_s=.1, vad=Vad(), commands_only=True)
        command.feed(np.full(8000, .2, np.float32))
        command.take()
        command.feed(np.zeros(1600, np.float32))
        endpoint = await command.take().ended.endpoint()
        assert endpoint.text == "volume down"
        assert batch.calls == 0
        chat = hearing.open_turn(16000, silence_s=.1, vad=Vad())
        chat.feed(np.full(8000, .2, np.float32))
        chat.take()
        chat.feed(np.zeros(1600, np.float32))
        endpoint = await chat.take().ended.endpoint()
        assert endpoint.text == "Tell me a story."
        assert batch.calls == 1

    asyncio.run(scenario())
