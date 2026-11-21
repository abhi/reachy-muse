import asyncio
import gc
import sys
import threading
from types import SimpleNamespace

import pytest
import numpy as np

from reachy_muse.companion_client import CompanionLink
from reachy_muse.companion_server.hearing_voxtral import HearingConfig, VoxtralHearing
from reachy_muse.companion_server.server import CompanionServer, Engines
from reachy_muse.protocol import CloseHow


@pytest.mark.parametrize("kind", ["wake", "stream", "disconnect"])
@pytest.mark.parametrize("blocked_at", ["queued", "open", "feed", "step", "vad", "mel"])
def test_abandoned_hearing_skips_remaining_model_work(monkeypatch, kind, blocked_at):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def stage(name):
        calls.append(name)
        if name == blocked_at:
            entered.set()
            assert release.wait(5)

    class Stream:
        pending = False

        def feed(self, samples):
            self.pending = True
            stage("feed")

        def step(self, **kwargs):
            if not self.pending:
                return []
            self.pending = False
            stage("step")
            return ["private transcript"]

    def open_stream(**kwargs):
        stage("open")
        return Stream()

    def final(*args):
        stage("final")
        return [SimpleNamespace(text="private final")]

    def vad(*args):
        stage("vad")
        return np.array([[0.0]]), None

    def mel(*args):
        stage("mel")

    monkeypatch.setitem(sys.modules, "mlx.core", SimpleNamespace(array=lambda value: value))
    monkeypatch.setitem(sys.modules, "parakeet_mlx.audio", SimpleNamespace(get_logmel=mel))

    async def scenario():
        opened, closed = asyncio.Event(), asyncio.Event()

        class Hearing(VoxtralHearing):
            def open(self, pre_roll_frames):
                session = super().open(pre_roll_frames)
                opened.set()
                close = session.close

                async def observed_close():
                    closed.set()
                    await close()

                session.close = observed_close
                return session

        hearing = Hearing(HearingConfig(speech_threshold=1, silence_endpoint_s=0))
        hearing._models = SimpleNamespace(
            voxtral=SimpleNamespace(create_streaming_session=open_stream),
            parakeet=SimpleNamespace(generate=final, preprocessor_config=None),
            vad=SimpleNamespace(run=vad),
        )
        occupying = None
        if blocked_at == "queued":
            occupying = hearing._thread.submit(stage, "queued")
        try:
            async with CompanionServer(Engines(hearing=hearing), "test") as server:
                link = CompanionLink(server.url.removesuffix("/v1"), token="test", out_rate=16000, insecure=True)
                running = asyncio.create_task(link.run())
                wake = None
                try:
                    await asyncio.wait_for(link.settled(), 2)
                    if kind == "wake":
                        wake = asyncio.create_task(link.transcribe_wake(b"\1\0" * 512))
                    else:
                        stream = link.open_hear()
                        stream.send(b"\1\0" * 512)
                        stream.finish()
                    await asyncio.wait_for(opened.wait(), 2)
                    assert await asyncio.to_thread(entered.wait, 2)
                    if kind == "wake":
                        wake.cancel()
                        await asyncio.gather(wake, return_exceptions=True)
                    elif kind == "stream":
                        stream.close(CloseHow.ABORT)
                    else:
                        running.cancel()
                    await asyncio.wait_for(closed.wait(), 2)
                    release.set()
                    await hearing.run(lambda: None)
                finally:
                    release.set()
                    running.cancel()
                    await asyncio.gather(running, return_exceptions=True)
        finally:
            release.set()
            hearing._thread.shutdown()
            if occupying is not None:
                occupying.result()

    asyncio.run(scenario())
    assert calls == {
        "queued": ["queued"], "open": ["open"],
        "feed": ["open", "feed"], "step": ["open", "feed", "step"],
        "vad": ["open", "feed", "step", "feed", "step", "vad"],
        "mel": ["open", "feed", "step", "feed", "step", "vad", "mel"],
    }[blocked_at]


def test_closed_session_never_reopens_or_runs_a_final_pass(monkeypatch):
    calls = []
    hearing = VoxtralHearing()
    hearing._models = SimpleNamespace(voxtral=SimpleNamespace(
        create_streaming_session=lambda **kwargs: calls.append("open")))

    async def scenario():
        session = hearing.open(0)
        await session.close()
        assert await session.feed(b"\0\0") == []
        assert await session.finish() == ""

    try:
        asyncio.run(scenario())
    finally:
        hearing._thread.shutdown()
    assert calls == []


def test_abandoned_native_work_releases_arrays_on_the_hearing_worker():
    entered, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    created, destroyed = [], []

    class Stream:
        def feed(self, samples):
            entered.set()
            assert release.wait(5)

        def __del__(self):
            destroyed.append(threading.get_ident())

    def create_stream(**kwargs):
        created.append(threading.get_ident())
        return Stream()

    hearing = VoxtralHearing()
    hearing._models = SimpleNamespace(voxtral=SimpleNamespace(create_streaming_session=create_stream))

    async def scenario():
        session = hearing.open(0)
        feeding = asyncio.create_task(session.feed(b"\0\0" * 160))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            feeding.cancel()
            await asyncio.gather(feeding, return_exceptions=True)
            await session.close()
            hearing._thread.submit(cleaned.set)
            release.set()
            # Hold the asyncio callbacks until the worker has cleared the session's reference.
            assert cleaned.wait(2)
        finally:
            release.set()

    try:
        asyncio.run(scenario())
    finally:
        hearing._thread.shutdown()
    gc.collect()
    assert len(created) == 1
    assert destroyed == created
