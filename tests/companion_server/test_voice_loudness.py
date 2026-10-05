import asyncio

import numpy as np
import pytest

from musegadget.reachy_companion_protocol import VoiceRole
from reachy_companion.voice_kokoro import Loudness, pcm16

RATE = 48_000
TIME = np.arange(RATE) / RATE


def measured(pcm: bytes):
    samples = np.frombuffer(pcm, "<i2").astype(np.float64) / 32767
    return round(20 * np.log10(np.sqrt(np.mean(samples ** 2))), 2), int(np.abs(np.frombuffer(pcm, "<i2")).max())


def test_a_quiet_tone_is_raised_to_the_target_level_unchanged_in_shape():
    tone = (0.1 * np.sin(2 * np.pi * 1000 * TIME)).astype(np.float32)  # -23 dBFS RMS
    assert measured(pcm16(Loudness().apply(tone))) == (-20.0, 4633)
    assert measured(pcm16(Loudness(target_dbfs=-26.0).apply(tone))) == (-26.0, 2322)


def test_a_peak_the_gain_would_push_past_full_scale_is_held_under_the_ceiling():
    tone = (0.02 * np.sin(2 * np.pi * 1000 * TIME)).astype(np.float32)
    tone[:480] = np.where(np.sin(2 * np.pi * 1000 * TIME[:480]) >= 0, 0.5, -0.5)  # a 10 ms burst; plain gain makes it 0.96
    assert measured(pcm16(Loudness().apply(tone))) == (-21.35, 26599)  # -1 dBFS is 29204


def test_silence_stays_silent_and_the_same_sentence_always_comes_out_the_same():
    assert pcm16(Loudness().apply(np.zeros(480, np.float32))) == bytes(960)
    noise = np.random.default_rng(7).normal(0, 0.05, RATE).astype(np.float32)
    assert pcm16(Loudness().apply(noise)) == pcm16(Loudness().apply(noise))


@pytest.mark.real
def test_a_real_kokoro_sentence_reaches_the_target_without_clipping():
    from reachy_companion.voice_kokoro import KokoroVoice

    async def run():
        voice = KokoroVoice()
        await voice.load()
        return b"".join([chunk async for chunk in voice.speak("Your reminder is set for eight tonight, and I will call Maya at nine.", VoiceRole.MAIN, RATE)])

    level, peak = measured(asyncio.run(run()))
    print(f"level {level} dBFS, peak {peak}")
    assert -20.5 <= level <= -19.5
    assert peak <= 29204
