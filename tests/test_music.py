import threading
import time

import numpy as np
import pytest

from reachy_muse.music import MusicController, select_track
from reachy_muse.music_audio import AudioBlock, MusicMix, MusicOutput, decode


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
    output.music = lambda count, at, speech: MusicMix(np.full(count, .1, dtype=np.float32), lambda: None)
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
    output.music = lambda n, at, speech: MusicMix(np.full(n, .1), lambda: None)
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
    assert select_track(folder, "sunny.mp3") == folder / "Sunny.MP3"
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


@pytest.mark.parametrize("suffix,codec", [(".mp3", "mp3"), (".ogg", "libopus"), (".flac", "flac")])
def test_real_compressed_music_formats(tmp_path, suffix, codec):
    import av
    source = tmp_path / ("synthetic" + suffix)
    audio = np.sin(np.arange(16000) * 2 * np.pi * 440 / 16000).astype(np.float32) * .2
    with av.open(str(source), "w") as container:
        stream = container.add_stream(codec, rate=16000)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(audio[None, :], format="fltp", layout="mono")
        frame.sample_rate = 16000
        for packet in (*stream.encode(frame), *stream.encode(None)):
            container.mux(packet)
    blocks = list(decode(source, 16000, threading.Event()))
    pcm = np.concatenate([block.pcm for block in blocks])
    assert len(pcm) == pytest.approx(16000, abs=2048)
    assert float(np.sqrt(np.mean(pcm ** 2))) == pytest.approx(.1414, abs=.02)


def test_http_audio_uses_the_same_decoder_without_a_second_player(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import io
    import wave
    content = io.BytesIO()
    with wave.open(content, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(np.full(1600, 8192, dtype="<i2").tobytes())
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(content.getvalue())))
            self.end_headers()
            self.wfile.write(content.getvalue())

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/audio"
        pcm = np.concatenate([block.pcm for block in decode(url, 16000, threading.Event())])
        assert pcm.tolist() == [.25] * 1600
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_failed_sink_never_publishes_accepted_music_timing():
    sink = Sink()
    def fail(pcm):
        raise RuntimeError("closed")
    output = MusicOutput(16000, fail, sink.clear, clock=lambda: sink.now)
    accepted = []
    output.music = lambda n, at, speech: MusicMix(np.full(n, .1), lambda: accepted.append(at))
    with pytest.raises(RuntimeError, match="closed"):
        output.pump()
    assert accepted == []
    assert output.tail == 0


@pytest.mark.parametrize("ending", ["sleep", "close"])
def test_hardware_sleep_and_shutdown_cancel_music_workers(ending):
    from test_hardware import FakeRobot, start_controller
    robot = FakeRobot()
    hardware = start_controller(robot, enable_motion=False)
    music = hardware.music
    music._decode = constant
    try:
        music.play({"url": "https://example.com/audio", "dance": False})
        until(lambda: bool(robot.media.output))
        if ending == "sleep":
            hardware.set_state("sleeping")
        else:
            hardware.close()
        assert not music.active
        assert robot.media.clear_count == 1
        count = len(robot.media.output)
        time.sleep(.06)
        assert len(robot.media.output) == count
    finally:
        hardware.close()
    assert not music._worker.is_alive()
    assert not music._pump.is_alive()


def test_music_dance_uses_the_existing_motor_writer_and_yields_to_listening():
    from reachy_muse.motion.dance import BeatTracker
    from reachy_muse.motion import MotionStyle
    from test_dance import clicks
    from test_hardware import FakeRobot, start_controller
    def track(source, rate, cancelled):
        pcm = clicks(120, seconds=4, rate=rate)
        tracker = BeatTracker(rate)
        for offset in range(0, len(pcm), 640):
            part = pcm[offset:offset + 640]
            yield AudioBlock(offset, part, tracker.feed(part))
    robot = FakeRobot()
    hardware = start_controller(robot, style=MotionStyle(breathing=False, expression_clips=False))
    hardware.music._decode = track
    try:
        hardware.music.play({"url": "https://example.com/audio"})
        start = len(robot.frames)
        until(lambda: any(abs(hardware._pose_parameters(frame[1])["pitch"]) > 1.5
                          for frame in robot.frames[start:]))
        assert len(robot.writers) == 1
        assert all(abs(np.degrees(frame[3])) <= 7 for frame in robot.frames[start:])
        hardware.set_state("listening")
        time.sleep(.5)
        assert hardware.music.dance.sample(time.monotonic(), eligible=False).tolist() == [0.] * 9
        assert len(robot.writers) == 1
    finally:
        hardware.close()
