import asyncio
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from reachy_muse.music import MusicController, select_track
from reachy_muse.music_audio import AudioBlock, MusicOutput, decode


def until(condition):
    deadline = time.monotonic() + 2
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(.005)


class Sink:
    def __init__(self):
        self.now = 10.
        self.frames = []
        self.clears = 0

    def push(self, pcm):
        self.frames.append(pcm.copy())

    def clear(self):
        self.clears += 1

    def output(self):
        return MusicOutput(16000, self.push, self.clear, clock=lambda: self.now)


def constant(source, rate, cancelled):
    for i in range(1000):
        if cancelled.is_set():
            return
        yield AudioBlock(i * 640, np.full(640, .8, dtype=np.float32), ())


def test_music_and_speech_overlay_on_one_sample_clock():
    sink = Sink()
    output = sink.output()
    output.music = lambda count, at, speech: np.full(count, .1, dtype=np.float32)
    output.pump()
    start = output.push_speech(np.full(640, .2, dtype=np.float32))
    assert start == pytest.approx(10.02)
    assert output.tail == pytest.approx(10.06)
    assert [len(frame) for frame in sink.frames] == [320, 640]
    assert sink.frames[0].tolist() == pytest.approx([.1] * 320)
    assert sink.frames[1].tolist() == pytest.approx([.3] * 640)
    assert not output.clear(speech=False)
    assert sink.clears == 0
    output.clear()
    assert sink.clears == 1
    assert output.tail == 0


def test_output_pump_does_not_build_an_unbounded_sink_queue():
    sink = Sink()
    output = sink.output()
    output.music = lambda n, at, speech: np.full(n, .1)
    for _ in range(100):
        output.pump()
    assert output.tail == pytest.approx(10.04)
    assert sum(map(len, sink.frames)) == 640


def test_duck_volume_cap_resume_and_stop_preserve_speech():
    sink = Sink()
    music = MusicController(sink.output(), 16000, decoder=constant, echo_cancelled=True)
    try:
        music.play({"url": "https://private.invalid/song", "volume": .9})
        until(lambda: bool(sink.frames))
        for _ in range(150):
            time.sleep(.001)
            sink.now += .02
            music.tick()
        assert max(sink.frames[-1]) == pytest.approx(.4, abs=.001)
        music.hold("wake", True)
        music.hold("notification", True)
        for _ in range(30):
            time.sleep(.001)
            sink.now += .02
            music.tick()
        assert max(sink.frames[-1]) == pytest.approx(.0503570, abs=.0001)
        assert music.dancing is False
        music.hold("wake", False)
        assert music.dancing is False
        music.hold("notification", False)
        assert music.dancing is True
        music.output.push_speech(np.full(640, .1, dtype=np.float32))
        before = sink.clears
        music.stop()
        assert sink.clears == before
        assert music.status() == {"playing": False, "elapsed": 0., "dancing": False}
        tail = len(sink.frames)
        sink.now += 1
        music.tick()
        assert len(sink.frames) == tail
    finally:
        music.close()


def test_room_dance_stops_after_eight_seconds_or_deadline_without_playback():
    sink = Sink()
    music = MusicController(sink.output(), 16000)
    try:
        music.listen(60)
        sink.now = 17.99
        music.tick()
        assert music.active
        sink.now = 18
        music.tick()
        assert not music.active
        music.listen(2)
        sink.now = 20
        music.tick()
        assert not music.active
        assert sink.frames == []
    finally:
        music.close()


def test_capture_tap_does_not_wait_for_a_speaker_write():
    sink = Sink()
    music = MusicController(sink.output(), 16000)
    entered = threading.Event()
    release = threading.Event()
    try:
        music.listen()
        def hold():
            with music.output.lock:
                entered.set()
                release.wait(2)
        worker = threading.Thread(target=hold)
        worker.start()
        assert entered.wait(1)
        started = time.monotonic()
        music.tap(np.zeros(320, dtype=np.float32), sink.now)
        assert time.monotonic() - started < .05
        release.set()
        worker.join()
    finally:
        release.set()
        music.close()


def test_stop_rejects_late_decoder_output_and_new_session_replaces_it():
    sink = Sink()
    entered, release = threading.Event(), threading.Event()
    def delayed(source, rate, cancelled):
        if source.endswith("old"):
            entered.set()
            release.wait(2)
            yield AudioBlock(0, np.full(640, -.8, dtype=np.float32), ())
        else:
            yield from constant(source, rate, cancelled)
    music = MusicController(sink.output(), 16000, decoder=delayed)
    try:
        music.play({"url": "https://private.invalid/old"})
        assert entered.wait(1)
        music.stop()
        music.play({"url": "https://private.invalid/new"})
        release.set()
        until(lambda: bool(sink.frames))
        assert all(np.min(frame) >= 0 for frame in sink.frames)
        assert music.status()["source"] == "https://private.invalid/new"
    finally:
        release.set()
        music.close()


def test_decode_error_logs_no_source_or_exception_text(caplog):
    sink = Sink()
    def fail(*args):
        raise ValueError("https://private.invalid/secret-track")
    music = MusicController(sink.output(), 16000, decoder=fail)
    try:
        music.play({"url": "https://private.invalid/secret-track"})
        until(lambda: not music.active)
        assert "Music decode failed" in caplog.text
        assert "private.invalid" not in caplog.text
        assert "secret-track" not in caplog.text
    finally:
        music.close()


def test_local_search_is_case_insensitive_and_confined(tmp_path):
    folder = tmp_path / "music"
    folder.mkdir()
    (folder / "Sunny.MP3").touch()
    (folder / "notes.txt").touch()
    secret = tmp_path / "private.flac"
    secret.touch()
    (folder / "outside.flac").symlink_to(secret)
    assert select_track(folder, "sUnN") == folder / "Sunny.MP3"
    assert select_track(folder, "shuffle") == folder / "Sunny.MP3"
    with pytest.raises(ValueError, match="No matching music file"):
        select_track(folder, "outside")


def test_real_pyav_decode_resamples_and_detects_beats(tmp_path):
    import wave
    from test_dance import clicks
    source = tmp_path / "private.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        wav.writeframes((clicks(120, seconds=3, rate=48000) * 32767).astype("<i2").tobytes())
    blocks = list(decode(source, 16000, threading.Event()))
    assert sum(len(block.pcm) for block in blocks) == 48000
    assert [beat.sample for block in blocks for beat in block.beats] == [3200, 11200, 19200, 27200, 35200, 43200]
    assert blocks[0].pcm.dtype == np.float32
