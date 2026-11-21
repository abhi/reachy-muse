import asyncio
import threading

from websockets.asyncio.client import connect

from reachy_muse.companion_server import server as serving
from reachy_muse.companion_server.engines import NodAnimator, Partial
from reachy_muse.companion_server.watchdog import EngineExecutor, Watchdog
from reachy_muse.protocol import (
    Animate, AnimateResult, AudioKind, Error, ErrorCode, HearOpen, HearPartial,
    Hello, Op, Sender, decode, encode, encode_audio,
)


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
