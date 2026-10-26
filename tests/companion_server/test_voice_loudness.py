import asyncio
import hashlib

import numpy as np
import pytest

from reachy_muse.protocol import VoiceRole
from reachy_muse.companion_server.voice_kokoro import Limiter, Loudness, _limit, pcm16

RATE = 16_000
TIME = np.arange(RATE) / RATE


def tone(hz: float, amplitude: float = 0.1) -> np.ndarray:
    return (amplitude * np.sin(2 * np.pi * hz * TIME)).astype(np.float32)


def measured(pcm: bytes):
    samples = np.frombuffer(pcm, "<i2").astype(np.float64) / 32767
    return round(20 * np.log10(np.sqrt(np.mean(samples ** 2))), 2), int(np.abs(np.frombuffer(pcm, "<i2")).max())


def test_a_quiet_tone_is_raised_to_the_target_level_unchanged_in_shape():
    assert measured(pcm16(Loudness().apply(tone(1000), RATE))) == (-12.0, 11639)
    assert measured(pcm16(Loudness(target_dbfs=-18.0).apply(tone(1000), RATE))) == (-18.0, 5833)


def test_a_pause_does_not_make_the_sentence_louder():
    half = tone(1000)
    half[RATE // 2:] = 0
    assert measured(pcm16(Loudness().apply(half, RATE))) == (-15.01, 11639)


def test_a_peak_the_gain_would_push_past_full_scale_is_held_under_the_ceiling():
    burst = tone(1000, 0.02)
    burst[:160] = np.where(np.sin(2 * np.pi * 1000 * TIME[:160]) >= 0, 0.5, -0.5)
    assert measured(pcm16(Loudness().apply(burst, RATE))) == (-20.66, 20674)


def test_silence_stays_silent_and_the_same_sentence_always_comes_out_the_same():
    assert pcm16(Loudness().apply(np.zeros(480, np.float32), RATE)) == bytes(960)
    noise = np.random.default_rng(7).normal(0, 0.05, RATE).astype(np.float32)
    assert pcm16(Loudness().apply(noise, RATE)) == pcm16(Loudness().apply(noise, RATE))


@pytest.mark.real
def test_a_real_kokoro_sentence_reaches_the_target_without_clipping():
    from reachy_muse.companion_server.voice_kokoro import KokoroVoice

    async def run():
        voice = KokoroVoice()
        await voice.load()
        return b"".join([chunk async for chunk in voice.speak("Your reminder is set for eight tonight, and I will call Maya at nine.", VoiceRole.MAIN, RATE)])

    level, peak = measured(asyncio.run(run()))
    print(f"level {level} dBFS, peak {peak}")
    assert -15.0 <= level <= -12.0
    assert peak <= 20675


def burst_noise() -> np.ndarray:
    noise = np.random.default_rng(7).normal(0, 0.05, RATE).astype(np.float32)
    noise[4000:4100] *= 12
    return noise


def test_kokoro_speech_is_byte_for_byte_what_it_was_before_streaming_limits():
    assert hashlib.sha256(pcm16(Loudness().apply(burst_noise(), RATE))).hexdigest() == "15446e1cc4c834a713d0d25925486ef0c90fa6c5591e747ce850b4693d2d4855"


@pytest.mark.parametrize("cuts", [[], [1], [64, 65, 66], [4033, 4090, 4101], list(range(0, RATE, 1000)), list(range(7, RATE, 331))])
def test_the_streaming_limiter_matches_the_whole_sentence_limiter_for_any_chunking(cuts):
    louder = burst_noise() * 4
    limiter = Limiter(ceiling=0.63, reach=32)
    pieces = [limiter.feed(piece) for piece in np.split(louder, cuts)] + [limiter.flush()]
    streamed = np.concatenate(pieces)
    assert np.array_equal(streamed, _limit(louder, 0.63, 32))
    assert round(float(np.abs(streamed).max()), 4) == 0.63
