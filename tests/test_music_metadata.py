import io
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from reachy_muse.music_audio import decode, AudioBlock
from reachy_muse.music_metadata import TrackMetadata
from reachy_muse.music import MusicController
from test_music import Sink, until


@pytest.mark.parametrize("suffix,codec", [(".mp3", "mp3"), (".flac", "flac"), (".ogg", "libopus")])
def test_actual_file_tags(tmp_path, suffix, codec):
    import av
    source = tmp_path / ("private" + suffix)
    tags = {"title": "Test title", "artist": "Test artist", "album": "Test album", "genre": "Jazz"}
    with av.open(str(source), "w") as output:
        output.metadata.update(tags)
        stream = output.add_stream(codec, rate=16000)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(np.zeros((1, 16000), dtype=np.float32), format="fltp", layout="mono")
        frame.sample_rate = 16000
        for packet in (*stream.encode(frame), *stream.encode(None)):
            output.mux(packet)
    blocks = list(decode(source, 16000, threading.Event()))
    assert blocks[0].metadata.payload() == tags


def test_icy_title_changes_survive_actual_decode_and_do_not_enter_logs(caplog):
    content = io.BytesIO()
    with wave.open(content, "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(np.full(16000 * 2, 4096, dtype="<i2").tobytes())
    audio = content.getvalue()
    interval = 8000
    wire = bytearray()
    for i, start in enumerate(range(0, len(audio), interval)):
        part = audio[start:start + interval]
        wire.extend(part)
        if len(part) == interval:
            label = "Secret artist - First private title" if i < 3 else "Other artist - Second private title"
            metadata = f"StreamTitle='{label}';".encode()
            size = (len(metadata) + 15) // 16
            wire.extend(bytes([size]) + metadata.ljust(size * 16, b"\0"))
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.headers["Icy-MetaData"] == "1"
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("icy-metaint", str(interval))
            self.end_headers()
            for start in range(0, len(wire), 137):
                self.wfile.write(wire[start:start + 137])
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        blocks = list(decode(f"http://127.0.0.1:{server.server_port}/private", 16000, threading.Event()))
        identities = {b.metadata.title: b.metadata.artist for b in blocks if b.metadata.title}
        assert identities == {"First private title": "Secret artist", "Second private title": "Other artist"}
        assert np.concatenate([b.pcm for b in blocks]).tolist() == [.125] * 32000
        assert "private" not in caplog.text
        assert "Secret artist" not in caplog.text
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_status_changes_only_with_accepted_blocks():
    sink = Sink()
    def decoder(*args):
        for i in range(40):
            yield AudioBlock(i * 640, np.zeros(640, np.float32), (),
                             metadata=TrackMetadata("One" if i < 20 else "Two", "Artist", "Album", "Genre"))
    music = MusicController(sink.output(), 16000, decoder=decoder)
    try:
        music.play({"url": "https://example.invalid/audio"})
        until(lambda: bool(sink.frames))
        assert music.status()["title"] == "One"
        for _ in range(45):
            sink.now += .02
            music.tick()
        assert {k: music.status()[k] for k in ("title", "artist", "album", "genre")} == {
            "title": "Two", "artist": "Artist", "album": "Album", "genre": "Genre"}
    finally:
        music.close()


def test_slow_http_body_returns_control_for_cancellation(monkeypatch):
    import http.client
    import socket
    import time
    from types import SimpleNamespace
    from reachy_muse.music_stream import HTTPAudio
    client, peer = socket.socketpair()
    client.settimeout(.03)
    done = threading.Event()
    def send():
        try:
            peer.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n")
            while not done.wait(.01):
                peer.sendall(b"x")
        except OSError:
            pass
    worker = threading.Thread(target=send)
    worker.start()
    response = http.client.HTTPResponse(client)
    response.begin()
    monkeypatch.setattr("reachy_muse.music_stream.open_http", lambda *a, **kw: response)
    cancelled = threading.Event()
    try:
        with HTTPAudio("http://example.invalid/audio", cancelled) as source:
            started = time.monotonic()
            assert source.read(4096) == b"x"
            assert time.monotonic() - started < .1
            cancelled.set()
            assert source.read(4096) == b""
    finally:
        done.set()
        client.close()
        peer.close()
        worker.join()


@pytest.mark.parametrize("phase", ["headers", "chunks"])
def test_cancellation_interrupts_slow_headers_and_chunk_framing(phase):
    import time
    from reachy_muse.music_stream import HTTPAudio
    cancelled, done = threading.Event(), threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if phase == "headers":
                    wire = b"HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\n\r\n"
                else:
                    self.wfile.write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
                    self.wfile.flush()
                    wire = b"000000000000000000000000000001\r\nx\r\n"
                for byte in wire:
                    if done.wait(.02):
                        return
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
            except OSError:
                pass
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    timer = threading.Timer(.15, cancelled.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(Exception):
            with HTTPAudio(f"http://127.0.0.1:{server.server_port}/audio", cancelled) as source:
                source.read(4096)
        assert time.monotonic() - started < .6
    finally:
        cancelled.set()
        done.set()
        timer.cancel()
        server.shutdown()
        server.server_close()
        worker.join()
