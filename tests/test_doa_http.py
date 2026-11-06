"""Bound direction reads and microphone cleanup against an offline HTTP daemon."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from types import SimpleNamespace

import pytest

from reachy_muse.hardware import ReachyController
from reachy_muse.speaker_direction import WakeDirection
from test_conversation import FakeHardware, bounded
from test_hardware import FakeRobot
from test_wake_handoff import handoff


@pytest.fixture
def doa_server():
    servers = []

    def serve(*payloads, slow=None):
        state = SimpleNamespace(paths=[], started=threading.Event(), stopped=threading.Event())

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                payload = payloads[min(len(state.paths), len(payloads) - 1)].encode()
                state.paths.append(self.path)
                headers = f"HTTP/1.0 200 OK\r\nContent-Length: {len(payload)}\r\n\r\n".encode()
                state.started.set()
                try:
                    for name, data in (("headers", headers), ("body", payload)):
                        if slow == name:
                            for byte in data:
                                self.wfile.write(bytes([byte]))
                                self.wfile.flush()
                                if state.stopped.wait(.03):
                                    return
                        else:
                            self.wfile.write(data)
                            self.wfile.flush()
                except OSError:
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
        thread.start()
        servers.append((server, thread, state))
        controller = ReachyController(host="127.0.0.1", port=server.server_port)
        controller._robot = FakeRobot()
        return controller, state

    yield serve
    for server, thread, state in servers:
        state.stopped.set()
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.fixture(params=["getaddrinfo", "create_connection"], ids=["resolver", "connect"])
def stalled_doa_setup(monkeypatch, request):
    state = SimpleNamespace(entered=threading.Event(), release=threading.Event(),
                            stopped=threading.Event(), calls=0)

    def stalled(*args, **kwargs):
        state.calls += 1
        state.entered.set()
        try:
            state.release.wait(1)
            raise OSError("connection setup failed")
        finally:
            state.stopped.set()

    monkeypatch.setattr(f"socket.{request.param}", stalled)
    controller = ReachyController(host="daemon.invalid", port=8123)
    controller._robot = FakeRobot()
    yield controller, state
    state.release.set()
    assert state.stopped.wait(2)


def test_doa_deadline_covers_connection_setup_without_starting_more_lookups(stalled_doa_setup):
    controller, state = stalled_doa_setup
    started = time.monotonic()
    assert controller.read_doa() is None
    elapsed = time.monotonic() - started
    assert elapsed < .35
    assert not state.stopped.is_set()
    for _ in range(3):
        assert controller.read_doa() is None
    assert state.calls == 1


def test_sampler_shutdown_finishes_during_stalled_connection_setup(stalled_doa_setup):
    controller, state = stalled_doa_setup

    async def run():
        direction = WakeDirection(controller.read_doa)
        task = asyncio.create_task(direction.sample_while(lambda: True))
        try:
            assert await asyncio.to_thread(state.entered.wait, 2)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    started = time.monotonic()
    asyncio.run(run())
    elapsed = time.monotonic() - started
    assert elapsed < .35
    assert not state.stopped.is_set()


@pytest.mark.parametrize("slow", ["headers", "body"])
def test_doa_read_has_a_total_deadline_for_trickled_responses(doa_server, slow):
    controller, _ = doa_server('{"angle": 1, "speech_detected": true}', slow=slow)
    started = time.monotonic()
    reading = controller.read_doa()
    elapsed = time.monotonic() - started
    assert reading is None
    assert elapsed < .35


def test_doa_read_rejects_an_oversized_response(doa_server):
    controller, _ = doa_server('{"angle": 1, "speech_detected": true, "unused": "' + "x" * 5000 + '"}')
    assert controller.read_doa() is None


def test_sampler_cancellation_does_not_wait_for_an_unfinished_source():
    entered, release = threading.Event(), threading.Event()

    def read():
        entered.set()
        release.wait(2)
        return None

    async def run():
        direction = WakeDirection(read)
        task = asyncio.create_task(direction.sample_while(lambda: True))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=.1)
            assert task in done, "cancellation waited for the unfinished direction source"
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_microphone_cleanup_finishes_during_a_trickled_response(doa_server, handoff):
    controller, state = doa_server('{"angle": 1, "speech_detected": true}', slow="body")

    class Hardware(FakeHardware):
        motion_enabled = True

        def read_doa(self):
            return controller.read_doa()

        def look_at_speaker(self, angle):
            controller.look_at_speaker(angle)

    async def listening(*args):
        await asyncio.Event().wait()

    async def run():
        task = asyncio.create_task(handoff(listening, hardware=Hardware()))
        try:
            assert await asyncio.to_thread(state.started.wait, 2)
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=.35)
            assert task in done, "microphone cleanup waited for the trickled HTTP response"
        finally:
            state.stopped.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(bounded(run()))
