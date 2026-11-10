from pathlib import Path
import time
import wave

import numpy as np
import pytest

from reachy_muse.music import MusicController
from reachy_muse.music_analysis import AudioAnalyzer
from test_music import Sink, until
from test_music_analysis import stem


def admitted(pcm):
    return [frame.music for frame in AudioAnalyzer(16000).feed(pcm)]


def speech(name):
    import av
    path = Path(__file__).parent / "companion_server" / "data" / (name + ".wav")
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        pcm = [f.to_ndarray().reshape(-1) for frame in container.decode(audio=0) for f in resampler.resample(frame)]
        pcm.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    return np.concatenate(pcm)


@pytest.mark.parametrize("name", ["correction", "reminder", "negation"])
def test_recorded_conversation_and_tv_speech_do_not_admit_music(name):
    voice = speech(name)
    voice = np.tile(voice, max(1, int(12 * 16000 / len(voice)) + 1))
    assert not any(admitted(voice))
    noise = np.random.default_rng(24).normal(0, .01, len(voice)).astype(np.float32)
    assert not any(admitted(voice * .6 + noise))


@pytest.mark.parametrize("kind", ["kick", "hat", "snare"])
def test_single_periodic_tapping_source_is_not_music(kind):
    assert not any(admitted(stem(kind, seconds=12)))


def test_noise_and_silence_are_rejected():
    assert not any(admitted(np.zeros(16000 * 8, np.float32)))
    assert not any(admitted(np.random.default_rng(8).normal(0, .2, 16000 * 8).astype(np.float32)))


def test_music_is_admitted_then_conversation_closes_it():
    music = stem("music", seconds=8)
    voice = np.tile(speech("correction"), 3)
    result = admitted(np.concatenate((music, voice)))
    assert not any(result[:60])
    assert all(result[100:195])
    assert not any(result[250:])


def test_room_request_cannot_claim_dancing_and_speech_cannot_keep_it_alive():
    sink = Sink()
    controller = MusicController(sink.output(), 16000)
    try:
        assert controller.listen()["dancing"] is False
        voice = np.tile(speech("reminder"), 5)
        for offset in range(0, 16000 * 7, 640):
            sink.now = 10 + (offset + 640) / 16000
            controller.tap(voice[offset:offset + 640], sink.now)
            time.sleep(.001)
            assert controller.dancing is False
        sink.now = 18
        controller.tick()
        assert controller.active is False
    finally:
        controller.close()


def test_room_music_starts_and_conversation_stops_motion():
    sink = Sink()
    controller = MusicController(sink.output(), 16000)
    try:
        controller.listen()
        for pcm in (stem("music", seconds=5), np.tile(speech("correction"), 2)[:64000]):
            for offset in range(0, len(pcm), 640):
                sink.now += len(pcm[offset:offset + 640]) / 16000
                controller.tap(pcm[offset:offset + 640], sink.now)
                until(lambda: not controller._session.rooms)
            if len(pcm) == 80000:
                until(lambda: controller.dancing)
            else:
                until(lambda: not controller.dancing)
    finally:
        controller.close()
