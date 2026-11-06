"""The samples Muse receives when the speaker leaks into Reachy's microphone."""

import pytest

np = pytest.importorskip("numpy")

from reachy_muse.realtime_call import CallInput


class Feed:
    def __init__(self):
        self.samples = []

    def push(self, samples):
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
