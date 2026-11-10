import asyncio
from dataclasses import replace

import numpy as np
import pytest

from reachy_muse.capabilities import Motion
from reachy_muse.motion.dance import DanceTimeline
from reachy_muse.motion.dance_signature import DanceSignature
from reachy_muse.music import MusicController
from reachy_muse.music_structure import Section, SongPlan
from test_music import Sink, constant
from test_music_analysis import features, stem


def test_signature_beat_warp_and_late_result_only_at_next_chorus():
    motion = Motion(25, ((0.,) * 9, (2.,) * 9, (0.,) * 9))
    signature = DanceSignature.from_motion(motion, 3.)
    section = Section(4., 12., 1, 0, True, .8, 0., False, .5, "groove")
    f = replace(features(stem("music", seconds=.2))[-1], harmony=1., bass=1., loudness=1.)
    assert signature.sample(section, 4., f).tolist() == [0.] * 9
    assert signature.sample(section, 5., f)[8] == pytest.approx(.7)
    assert signature.sample(section, 6., f).tolist() == [0.] * 9
    late = DanceSignature.from_motion(motion, 4.1)
    assert late.sample(section, 5., f).tolist() == [0.] * 9
    assert late.sample(replace(section, start=12.), 13., f)[8] == pytest.approx(.7)


@pytest.mark.parametrize("frames", [(), ((0.,) * 8,) * 2, ((float("nan"),) * 9,) * 2])
def test_invalid_signature_is_absent(frames):
    assert DanceSignature.from_motion(Motion(25, frames), 0) is None


def test_absent_and_failed_companion_leave_audio_driven_motion_running():
    async def run(source):
        sink = Sink()
        music = MusicController(sink.output(), 16000, decoder=constant)
        try:
            music.play({"url": "https://example.invalid/audio"})
            music._session.plan = SongPlan((Section(8, 16, 1, 0, True, .7, 0, False, .5, "groove"),))
            task = asyncio.create_task(music.signatures(source))
            for f in features(stem("music", seconds=.4)):
                music.dance.offer_features((f,), 0, 0, 16000)
            music.dance.sample(0, eligible=True)
            assert np.linalg.norm(music.dance.sample(.4, eligible=True)) > .1
            await asyncio.sleep(.25)
            assert music.active
            assert music._session.signatures == {}
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        finally:
            music.close()
    class Broken:
        library_available = True
        async def generate(self, intent, duration):
            assert (intent, duration) == ("dance_groove", 2.)
            raise RuntimeError("no companion")
    asyncio.run(run(None))
    asyncio.run(run(Broken()))


def test_one_slow_request_survives_replacement_without_retry_storm():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        class Slow:
            library_available = True
            calls = 0
            async def generate(self, intent, duration):
                self.calls += 1
                entered.set()
                await release.wait()
                return Motion(25, ((0.,) * 9, (1.,) * 9))
        source, sink = Slow(), Sink()
        music = MusicController(sink.output(), 16000, decoder=constant)
        task = asyncio.create_task(music.signatures(source))
        try:
            music.play({"url": "https://example.invalid/one"})
            old = music._session
            old.plan = SongPlan((Section(4, 8, 1, 0, True, .5, 0, False, .5, "calm"),))
            await asyncio.wait_for(entered.wait(), 1)
            music.stop()
            music.listen()
            await asyncio.sleep(.2)
            assert source.calls == 1
            release.set()
            await asyncio.sleep(.05)
            assert old.signatures == {}
            assert music._session.signatures == {}
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            music.close()
    asyncio.run(run())


def test_signature_aligns_to_measured_offbeat_origin():
    signature = DanceSignature.from_motion(Motion(25, ((0.,) * 9, (2.,) * 9, (0.,) * 9)), 1.)
    section = Section(6., 10., 0, 0, True, .8, 0, False, .5, "groove", beat_origin=.25)
    frame = replace(features(stem("music", seconds=.2))[-1], bass=1, loudness=1)
    assert signature.sample(section, 6.1, frame).tolist() == [0.] * 9
    assert signature.sample(section, 7.25, frame)[8] == pytest.approx(.7)


def test_stream_lookahead_requests_first_peak_before_playback_reaches_it():
    from reachy_muse.music_analysis import AudioAnalyzer
    from reachy_muse.music_audio import AudioBlock
    from reachy_muse.music_structure import StructureTracker
    from test_music_structure import song
    async def run():
        sink = Sink()
        called = asyncio.Event()
        def decoder(source, rate, cancelled):
            analyzer, structure = AudioAnalyzer(rate), StructureTracker(rate)
            pcm = song()
            for offset in range(0, len(pcm), 640):
                if cancelled.is_set():
                    return
                part = pcm[offset:offset + 640]
                frames = analyzer.feed(part)
                structure.feed(frames)
                yield AudioBlock(offset, part, (), frames, structure.plan())
        music = MusicController(sink.output(), 16000, decoder=decoder)
        class Source:
            library_available = True
            async def generate(self, intent, duration):
                assert music._session.consumed / 16000 < 6
                called.set()
                return Motion(25, ((0.,) * 9, (1.,) * 9))
        task = asyncio.create_task(music.signatures(Source()))
        try:
            music.play({"url": "https://example.invalid/live"})
            for _ in range(350):
                sink.now += .02
                music.tick()
                await asyncio.sleep(.005)
                if called.is_set():
                    break
            assert called.is_set()
            assert music._session.signatures
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            music.close()
    asyncio.run(run())


def test_paced_live_stream_prepares_peak_character_before_first_drop():
    import time
    from reachy_muse.music_analysis import AudioAnalyzer
    from reachy_muse.music_audio import AudioBlock
    from reachy_muse.music_structure import StructureTracker
    from test_music_structure import song
    async def run():
        sink = Sink()
        def decoder(source, rate, cancelled):
            analyzer, structure = AudioAnalyzer(rate), StructureTracker(rate)
            pcm = song()
            began = time.monotonic()
            for offset in range(0, len(pcm), 640):
                if cancelled.wait(max(0., began + (offset + 640) / rate - time.monotonic())):
                    return
                part = pcm[offset:offset + 640]
                frames = analyzer.feed(part)
                structure.feed(frames)
                yield AudioBlock(offset, part, (), frames, structure.plan())
        controller = MusicController(sink.output(), 16000, decoder=decoder)
        class Source:
            library_available = True
            async def generate(self, intent, duration):
                return Motion(25, ((0.,) * 9, (1.,) * 9, (0.,) * 9))
        task = asyncio.create_task(controller.signatures(Source()))
        try:
            began = time.monotonic()
            controller.play({"url": "https://example.invalid/live"})
            while time.monotonic() - began < 8:
                sink.now = 10 + time.monotonic() - began
                controller.tick()
                await asyncio.sleep(.02)
                peak = next((s for s in controller._session.plan.sections if s.drop and s.start >= 6), None)
                if peak is not None:
                    signature = controller._session.signatures[0, peak.character]
                    assert signature.ready_after < 6
                    assert controller._session.consumed / 16000 < 6
                    break
            else:
                pytest.fail("No live drop preview")
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            controller.close()
    asyncio.run(run())
