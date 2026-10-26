import asyncio
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_muse.contract.expressions import Expression
from reachy_muse.protocol import VoiceRole
from reachy_muse.companion_server.voice_chatterbox import CALIBRATION, ChatterboxVoice, Seams, _SteadyRandom
from reachy_muse.companion_server.voice_kokoro import _voiced_rms

MODEL_RATE = 24_000
SAMPLES_PER_TOKEN = 960


@pytest.fixture
def steady_random(monkeypatch):
    def normal(shape, *, key):
        return np.random.default_rng(key).standard_normal(shape, dtype=np.float32)

    mx = SimpleNamespace(random=SimpleNamespace(key=lambda seed: seed, normal=normal), moveaxis=np.moveaxis)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    return _SteadyRandom(7, 24_000)


def test_growing_mel_decodes_keep_the_same_noise_prefix(steady_random):
    draws = []
    for shape in ((1, 80, 40), (1, 80, 80), (1, 80, 120)):
        steady_random.calls = 0
        noise = steady_random.normal(shape)
        assert noise.shape == shape
        draws.append(noise)
    np.testing.assert_array_equal(draws[0], draws[1][:, :, :40])
    np.testing.assert_array_equal(draws[1], draws[2][:, :, :80])


@pytest.mark.parametrize("shape", [(1, 480001, 1), (1, 965760, 1), (1, 9, 965760)])
def test_noise_covers_long_sentences_through_closing_silence(steady_random, shape):
    time_axis = 1 if shape[-1] == 1 else 2
    short_shape = list(shape)
    short_shape[time_axis] = 480000
    first = steady_random.normal(short_shape)
    steady_random.calls = 0
    longer = steady_random.normal(shape)
    assert longer.shape == shape
    np.testing.assert_array_equal(first, np.take(longer, range(480000), axis=time_axis))


class SineTurbo:
    sample_rate = MODEL_RATE

    def __init__(self, tokens=50, per_batch=20, decode_s=0.0, lengths=()):
        self.count = tokens
        self.lengths = dict(lengths)
        self.per_batch = per_batch
        self.decode_s = decode_s
        self.texts = []
        self.decodes = 0

    def tokens(self, text):
        self.texts.append(text)
        count = self.lengths.get(text, self.count)
        for start in range(0, count, self.per_batch):
            yield list(range(start, min(start + self.per_batch, count))), start + self.per_batch >= count

    def decode(self, tokens, final, seed):
        self.decodes += 1
        time.sleep(self.decode_s)
        n = np.arange((len(tokens) + 5 * final) * SAMPLES_PER_TOKEN)
        return (0.1 * np.sin(2 * np.pi * 200 * n / MODEL_RATE)).astype(np.float32)


def loaded(model):
    voice = ChatterboxVoice(open_model=lambda: model)
    asyncio.run(voice.load())
    model.texts.clear()
    return voice


def spoken(voice, text, rate, expression=None):
    async def run():
        return b"".join([chunk async for chunk in voice.speak(text, VoiceRole.MAIN, rate, expression)])

    return np.frombuffer(asyncio.run(run()), "<i2")


def level(samples):
    x = samples.astype(np.float64) / 32767
    return round(20 * np.log10(np.sqrt(np.mean(x ** 2))), 2), int(np.abs(samples).max())


def test_a_tone_decoded_in_pieces_comes_out_whole_at_the_calibrated_level():
    voice = loaded(SineTurbo())
    at_model_rate = spoken(voice, "One.", MODEL_RATE)
    gain = 10 ** (-12 / 20) / (0.1 / np.sqrt(2))
    whole = np.clip(0.1 * gain * np.sin(2 * np.pi * 200 * np.arange(55 * SAMPLES_PER_TOKEN) / MODEL_RATE), -1, 1) * 32767
    assert at_model_rate.size == 52_800
    assert int(np.abs(at_model_rate - whole.astype("<i2")).max()) <= 1

    at_robot_rate = spoken(voice, "One. Two.", 16_000)
    assert at_robot_rate.size == 2 * 35_200
    assert level(at_robot_rate) == (-12.0, 11640)


@pytest.mark.parametrize("expression, first", [
    (Expression.HAPPY, "[chuckle] Good news."),
    (Expression.SAD, "[sigh] Good news."),
    (Expression.SURPRISED, "[gasp] Good news."),
    (Expression.CURIOUS, "Good news."),
    (None, "Good news."),
])
def test_an_expression_becomes_a_sound_tag_on_the_first_sentence_only(expression, first):
    model = SineTurbo(tokens=5)
    spoken(loaded(model), "Good news. It rained.", 16_000, expression)
    assert model.texts == [first, "It rained."]


def test_load_calibrates_on_three_fixed_sentences():
    model = SineTurbo(tokens=5)
    asyncio.run(ChatterboxVoice(open_model=lambda: model).load())
    assert model.texts == list(CALIBRATION)


def test_the_next_sentence_is_synthesized_without_waiting_for_the_robot():
    model = SineTurbo(tokens=5)
    voice = loaded(model)

    async def run():
        chunks = voice.stream(["First.", "Second."], 16_000)
        first = await anext(chunks)
        for _ in range(100):
            if len(model.texts) == 2:
                break
            await asyncio.sleep(0.01)
        await chunks.aclose()
        return first.sentence, model.texts

    assert asyncio.run(run()) == (0, ["First.", "Second."])


def test_closing_the_stream_stops_the_model_between_token_batches():
    model = SineTurbo(tokens=5, decode_s=0.005, lengths={"A very long sentence.": 2000})
    voice = loaded(model)
    model.decodes = 0

    async def run():
        chunks = voice.stream(["A very long sentence."], 16_000)
        await anext(chunks)
        await chunks.aclose()
        spoken_after = b"".join([c async for c in voice.speak("Next.", VoiceRole.MAIN, 16_000)])
        return len(spoken_after)

    assert asyncio.run(run()) == 2 * 6400
    assert model.decodes < 10


def test_where_two_decodes_disagree_the_held_tail_fades_into_the_newer_one():
    seams = Seams(1000)
    first = seams.feed(np.ones(800, np.float32), False)
    second = seams.feed(np.full(1000, 2.0, np.float32), True)
    assert (first.size, second.size) == (500, 500)
    assert np.array_equal(first, np.ones(500))
    assert np.allclose(second[:60], np.linspace(1, 2, 60)) and np.array_equal(second[60:], np.full(440, 2.0))


@pytest.mark.real
def test_real_chatterbox_starts_within_a_second_near_the_target_level_without_clipping():
    voice = ChatterboxVoice()

    async def run():
        await voice.load()
        started = time.perf_counter()
        first_s, chunks = None, []
        async for chunk in voice.speak("Your reminder is set for eight tonight. I will call Maya at nine.", VoiceRole.MAIN, 16_000):
            first_s = first_s or time.perf_counter() - started
            chunks.append(chunk)
        return first_s, time.perf_counter() - started, np.frombuffer(b"".join(chunks), "<i2")

    first_s, total_s, samples = asyncio.run(run())
    voiced = 20 * np.log10(_voiced_rms(samples.astype(np.float64) / 32767, 16_000))
    print(f"first chunk {first_s:.3f} s, {samples.size / 16_000:.2f} s of speech in {total_s:.2f} s, voiced {voiced:.2f} dBFS, peak {np.abs(samples).max()}")
    assert first_s < 1.0
    assert -16.0 <= voiced <= -12.0
    assert np.abs(samples).max() <= 20675
