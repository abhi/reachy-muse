import asyncio
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from websockets.asyncio.client import connect

from reachy_muse.companion_server import server as serving
from reachy_muse.companion_server import hearing_voxtral as hearing_module
from reachy_muse.companion_server.engines import NodAnimator, Partial
from reachy_muse.companion_server.watchdog import EngineExecutor, Watchdog
from reachy_muse.protocol import (
    Animate, AnimateResult, AudioKind, CloseHow, Error, ErrorCode, HearClose, HearEndpoint, HearOpen, HearPartial,
    Hello, Op, Sender, decode, encode, encode_audio,
)


@pytest.mark.parametrize("decoding_at", ["feed", "finish"])
def test_synchronous_voxtral_decoding_keeps_a_slow_stream_alive(monkeypatch, decoding_at):
    now, exits = [100.0], []
    monitor = Watchdog(clock=lambda: now[0], loop_s=1000, exit_process=exits.append)
    monkeypatch.setattr(serving, "watchdog", monitor)
    monkeypatch.setattr(serving, "HEARING_CHECK_S", .001)
    monkeypatch.setattr(hearing_module, "watchdog", monitor)

    async def scenario():
        loop = asyncio.get_running_loop()

        class Stream:
            closed = False
            done = False
            tokens = 0

            def feed(self, samples):
                pass

            def close(self):
                self.closed = True
                self.done = self.tokens == 8

            def step(self, *, max_decode_tokens):
                deltas = []
                if decoding_at == "finish" and not self.closed:
                    return deltas
                for _ in range(min(max_decode_tokens, 8 - self.tokens)):
                    now[0] += .5
                    self.tokens += 1
                    deltas.append("test ")
                    asyncio.run_coroutine_threadsafe(asyncio.sleep(.01), loop).result(2)
                self.done = self.closed and self.tokens == 8
                return deltas

        hearing = hearing_module.VoxtralHearing(hearing_module.HearingConfig(final_pass=False))
        hearing._thread.shutdown()
        hearing._thread = EngineExecutor("hearing", monitor=monitor)
        hearing._models = SimpleNamespace(voxtral=SimpleNamespace(create_streaming_session=lambda **kwargs: Stream()))
        monitor.start()
        try:
            async with serving.CompanionServer(serving.Engines(hearing=hearing), "test") as server:
                async with connect(server.url, proxy=None) as ws:
                    await ws.send(encode(Hello((1,), "test", (Op.HEAR,), 16000, "test")))
                    await ws.recv()
                    await ws.send(encode(HearOpen(1, 0)))
                    await ws.send(encode_audio(AudioKind.MIC, 1, 0, b"\0\0" * 160))
                    await ws.send(encode(HearClose(1, CloseHow.FINISH)))
                    messages = []
                    while True:
                        message = decode(await asyncio.wait_for(ws.recv(), 2), sender=Sender.COMPANION)
                        messages.append(message)
                        if isinstance(message, (HearEndpoint, Error)):
                            break
                    expected = ([HearPartial(1, 0, "test test test test test test test test")]
                                if decoding_at == "feed" else [])
                    assert messages == [*expected, HearEndpoint(1, "test test test test test test test test", True)]
        finally:
            await hearing.run(lambda: None)
            hearing._thread.shutdown()
            monitor.stop()

    asyncio.run(scenario())
    assert exits == []


def test_stuck_hearing_fails_streams_and_refuses_opens_until_worker_progress(monkeypatch):
    now = [100.0]
    entered, progress, progressed, release = (threading.Event() for _ in range(4))
    exits, opened = [], []
    monitor = Watchdog(clock=lambda: now[0], loop_s=1000, exit_process=exits.append)
    monkeypatch.setattr(serving, "watchdog", monitor, raising=False)

    class Hearing:
        name = "fake"

        def open(self, pre_roll_frames):
            opened.append(pre_roll_frames)
            return self

        async def feed(self, pcm):
            return await asyncio.wrap_future(worker.submit(blocked))

        def cancel(self):
            pass

        async def close(self):
            pass

    def blocked():
        entered.set()
        assert progress.wait(5)
        monitor.progress()
        progressed.set()
        assert release.wait(5)
        return [Partial("test")]

    async def receive(ws):
        try:
            return decode(await asyncio.wait_for(ws.recv(), .25), sender=Sender.COMPANION)
        except asyncio.TimeoutError:
            return None

    async def scenario():
        monitor.start()
        try:
            engines = serving.Engines(hearing=Hearing(), motion=NodAnimator())
            async with serving.CompanionServer(engines, "test") as server:
                async with connect(server.url, proxy=None) as ws:
                    await ws.send(encode(Hello((1,), "test", (Op.HEAR, Op.ANIMATE), 16000, "test")))
                    await ws.recv()
                    await ws.send(encode(HearOpen(1, 0)))
                    await ws.send(encode_audio(AudioKind.MIC, 1, 0, b"\0\0" * 320))
                    assert await asyncio.to_thread(entered.wait, 2)
                    now[0] = 102.9
                    await ws.send(encode_audio(AudioKind.MIC, 1, 1, b"\0\0" * 320))
                    assert await receive(ws) is None
                    now[0] = 103.0
                    await ws.send(encode_audio(AudioKind.MIC, 1, 2, b"\0\0" * 320))
                    error = await receive(ws)
                    assert isinstance(error, Error), error
                    assert (error.code, error.stream) == (ErrorCode.OVERLOADED, 1)
                    await ws.send(encode(HearOpen(3, 0)))
                    error = await receive(ws)
                    assert (error.code, error.stream) == (ErrorCode.OVERLOADED, 3)
                    assert opened == [0]
                    await ws.send(encode(Animate(1, "nod", 1000)))
                    assert isinstance(await receive(ws), AnimateResult)
                    now[0] = 131.0
                    await asyncio.sleep(.15)
                    assert exits == []
                    progress.set()
                    assert await asyncio.to_thread(progressed.wait, 2)
                    await ws.send(encode(HearOpen(5, 0)))
                    await ws.send(encode_audio(AudioKind.MIC, 5, 0, b"\0\0" * 320))
                    # The worker is still in the original call, but its progress reopens admission.
                    assert await receive(ws) is None
                    assert opened == [0, 0]
                    release.set()
                    assert await receive(ws) == HearPartial(5, 0, "test")
                    assert await receive(ws) is None
        finally:
            progress.set()
            release.set()
            monitor.stop()

    with EngineExecutor("hearing", monitor=monitor) as worker:
        asyncio.run(scenario())


def test_a_long_final_pass_is_progress_within_its_budget(monkeypatch):
    """Parakeet's final pass is one native call without progress; over 8.2 s of audio it may take 8.2 s."""
    final_pass_s = 3.5  # past the 3 s streaming deadline and the measured 2.77 s p95
    now, exits = [100.0], []
    entered, release = threading.Event(), threading.Event()
    monitor = Watchdog(clock=lambda: now[0], loop_s=1000, exit_process=exits.append)
    monkeypatch.setattr(serving, "watchdog", monitor)
    monkeypatch.setattr(hearing_module, "watchdog", monitor)
    monkeypatch.setattr(serving, "HEARING_CHECK_S", .01)
    monkeypatch.setitem(sys.modules, "mlx.core", SimpleNamespace(array=lambda value: value))
    monkeypatch.setitem(sys.modules, "parakeet_mlx.audio", SimpleNamespace(get_logmel=lambda audio, config: audio))

    class Stream:
        def feed(self, samples):
            pass

        def step(self, *, max_decode_tokens):
            return []

    def generate(mel):
        now[0] += final_pass_s
        entered.set()
        assert release.wait(5)
        return [SimpleNamespace(text="the final")]

    async def scenario():
        hearing = hearing_module.VoxtralHearing(hearing_module.HearingConfig())
        hearing._thread.shutdown()
        hearing._thread = EngineExecutor("hearing", monitor=monitor)
        hearing._models = SimpleNamespace(
            voxtral=SimpleNamespace(create_streaming_session=lambda **kwargs: Stream()),
            parakeet=SimpleNamespace(generate=generate, preprocessor_config=None),
            vad=SimpleNamespace(run=lambda *args: (np.array([[1.0]]), None)),
        )
        monitor.start()
        try:
            async with serving.CompanionServer(serving.Engines(hearing=hearing), "test") as server:
                async with connect(server.url, proxy=None) as ws:
                    await ws.send(encode(Hello((1,), "test", (Op.HEAR,), 16000, "test")))
                    await ws.recv()
                    speech = b"\1\0" * 320
                    await ws.send(encode(HearOpen(1, 0)))
                    for seq in range(410):  # 8.2 s, as long as the benchmark's reminder.wav
                        await ws.send(encode_audio(AudioKind.MIC, 1, seq, speech))
                    await ws.send(encode(HearOpen(3, 0)))
                    await ws.send(encode_audio(AudioKind.MIC, 3, 0, speech))
                    await ws.send(encode(HearClose(1, CloseHow.FINISH)))
                    assert await asyncio.to_thread(entered.wait, 2)
                    await ws.send(encode_audio(AudioKind.MIC, 3, 1, speech))
                    await asyncio.sleep(.2)
                    await ws.send(encode(HearOpen(5, 0)))
                    errors = []
                    try:
                        while True:
                            message = decode(await asyncio.wait_for(ws.recv(), .25), sender=Sender.COMPANION)
                            if isinstance(message, Error):
                                errors.append((message.code.value, message.stream))
                    except asyncio.TimeoutError:
                        pass
                    release.set()
                    assert errors == []
                    async with asyncio.timeout(2):
                        while not isinstance(message := decode(await ws.recv(), sender=Sender.COMPANION), HearEndpoint):
                            pass
                    assert message == HearEndpoint(1, "the final", True)
        finally:
            release.set()
            await hearing.run(lambda: None)
            hearing._thread.shutdown()
            monitor.stop()

    asyncio.run(scenario())
    assert exits == []
