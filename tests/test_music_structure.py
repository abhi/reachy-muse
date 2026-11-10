import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from reachy_muse.music_analysis import AudioAnalyzer
from reachy_muse.music_structure import StructureTracker
from reachy_muse.motion.dance import DanceTimeline
from reachy_muse.music import MusicController
from reachy_muse.music_audio import AudioBlock
from test_music import Sink, until
from test_music_analysis import stem


def song():
    rate = 16000
    t = np.arange(6 * rate) / rate
    def section(notes, gain):
        chord = sum(np.sin(2 * np.pi * note * t) for note in notes) / 3
        kick = .2 * np.sin(2 * np.pi * 70 * t) * np.exp(-(t % .5) / .03)
        return (gain * chord + kick).astype(np.float32)
    a = section((261.63, 329.63, 392), .18)
    b = section((311.13, 369.99, 466.16), .38)
    return np.concatenate((a, b, a, b))


def analyze(pcm):
    tracker = StructureTracker(16000, rolling=False)
    frames = AudioAnalyzer(16000).feed(pcm)
    tracker.feed(frames)
    return frames, tracker.plan(complete=True)


def test_chroma_self_similarity_recovers_a_b_a_b_and_first_chorus():
    _, plan = analyze(song())
    assert [(s.start, s.end, s.family, s.occurrence, s.chorus) for s in plan.sections] == [
        (0, 6, 0, 0, False), (6, 12, 1, 0, True),
        (12, 18, 0, 1, False), (18, 24, 1, 1, True)]
    assert plan.at(24) is None
    assert plan.gain(23.5) < .25


def test_repeated_chorus_reuses_shape_at_larger_gain():
    frames, plan = analyze(song())
    plan = replace(plan, complete=False)
    first, repeat = DanceTimeline(), DanceTimeline()
    for tick in range(95):
        seconds = tick * .04
        f = frames[150 + tick]
        r = frames[450 + tick]
        first.offer_features((f,), 6 * 16000, 0., 16000, plan)
        repeat.offer_features((r,), 18 * 16000, 0., 16000, plan)
        a, b = first.sample(seconds, eligible=True), repeat.sample(seconds, eligible=True)
    assert b == pytest.approx(a * 1.12, abs=.02)
    assert np.linalg.norm(a) > .1


def test_build_and_drop_are_detected_from_literal_energy_rise():
    base = stem("music", seconds=1, gain=.5)
    pcm = np.concatenate([base * gain for gain in (.1, .15, .25, .4, .9, .9, .9, .9)])
    _, plan = analyze(pcm)
    assert any(s.build > .1 for s in plan.sections)
    assert [(s.start, s.drop) for s in plan.sections] == [(0, False), (4, True)]
    assert plan.gain(4.2) > plan.gain(2.2)


def test_full_analysis_does_not_block_playback_priming_or_survive_stop(tmp_path):
    source = tmp_path / "track.wav"
    source.touch()
    blocked, release = threading.Event(), threading.Event()
    def decoder(path, rate, cancelled):
        if threading.current_thread().name == "reachy-music-analysis":
            blocked.set()
            release.wait(2)
        for i in range(40):
            if cancelled.is_set():
                return
            yield AudioBlock(i * 640, np.full(640, .1, dtype=np.float32), ())
    sink = Sink()
    music = MusicController(sink.output(), 16000, folder=tmp_path, decoder=decoder)
    try:
        started = time.monotonic()
        music.play({"query": "track"})
        until(lambda: bool(sink.frames))
        assert time.monotonic() - started < .48
        assert blocked.wait(1)
        music.stop()
        release.set()
        assert music.status()["playing"] is False
    finally:
        release.set()
        music.close()


def test_drop_preview_precedes_480_ms_live_playback_lead():
    tracker = StructureTracker(16000)
    analyzer = AudioAnalyzer(16000)
    pcm = song()
    first_seen = None
    for offset in range(0, 7 * 16000, 640):
        tracker.feed(analyzer.feed(pcm[offset:offset + 640]))
        if any(s.drop and s.start >= 6 for s in tracker.plan().sections):
            first_seen = (offset + 640) / 16000
            break
    assert first_seen == pytest.approx(6.16, abs=.04)
    assert first_seen - .48 < 6
