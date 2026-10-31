"""Fixed recordings survive restarts without trusting stale or partial PCM."""

import asyncio
import json
import os

import pytest

np = pytest.importorskip("numpy")

from reachy_muse import local_speech, phrase_cache
from reachy_muse.capabilities import SpokenLine
from reachy_muse.local_backends import LocalVoice
from reachy_muse.local_speech import PiperSpeech
from reachy_muse.phrase_cache import PhraseCache
from test_local_speech import model, worker  # noqa: F401


class SyntheticPiper(PiperSpeech):
    def __init__(self, model, cache_dir, samples=(.25, -.5)):
        super().__init__(model, cache_dir=cache_dir)
        self.samples = samples
        self.requests = []

    async def stream(self, text, rate):
        self.requests.append((text, rate))
        yield np.array(self.samples, dtype=np.float32)


async def spoken(voice, text="Yes?", rate=16000, *, prepare=False):
    line = SpokenLine(text, None, "notice")
    stream = voice.prepare(line, rate) if prepare else voice.stream(line, rate)
    return np.concatenate([chunk async for chunk in stream]).tolist()


def test_restart_loads_completed_phrases_and_only_synthesizes_new_text(model, tmp_path):
    async def scenario():
        directory = tmp_path / "cache"
        first = SyntheticPiper(model, directory)
        await LocalVoice(first, ("Yes?", "On it.")).warm(16000)
        restarted = SyntheticPiper(model, directory, samples=(-.75, .5))
        voice = LocalVoice(restarted, ("Yes?", "On it.", "New phrase."))
        await voice.warm(16000)
        assert await spoken(voice) == [.25, -.5]
        assert await spoken(voice, "On it.", prepare=True) == [.25, -.5]
        assert await spoken(voice, "New phrase.") == [-.75, .5]
        assert restarted.requests == [("New phrase.", 16000)]
        assert len(list(directory.glob("*.pcm"))) == 3
        # Arbitrary replies take the live path and never become disk recordings.
        assert await spoken(voice, "A private reply.") == [-.75, .5]
        assert len(list(directory.glob("*.pcm"))) == 3
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["model", "text", "rate", "inference", "phonemes", "source_rate",
                                   "processing", "library"])
def test_changed_audio_inputs_invalidate_the_recording(model, tmp_path, monkeypatch, change):
    async def scenario():
        directory = tmp_path / "cache"
        await LocalVoice(SyntheticPiper(model, directory), ("Yes?",)).warm(16000)
        text, rate = "Yes?", 16000
        if change == "model":
            stat = model.stat()
            model.write_bytes(b"X" * stat.st_size)
            os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        elif change == "text":
            text = "Yes!"
        elif change == "rate":
            rate = 48000
        elif change in ("inference", "phonemes", "source_rate"):
            sidecar = model.with_suffix(".onnx.json")
            config = json.loads(sidecar.read_text())
            if change == "inference":
                config["inference"] = {"length_scale": 1.2, "noise_scale": .5}
            elif change == "phonemes":
                config["phoneme_id_map"] = {"a": [12]}
            else:
                config["audio"]["sample_rate"] = 24000
            sidecar.write_text(json.dumps(config))
        elif change == "processing":
            monkeypatch.setattr(local_speech, "PCM_PROCESSING_VERSION", 2)
        else:
            monkeypatch.setattr(local_speech, "version", lambda package: "changed")
        speech = SyntheticPiper(model, directory, samples=(-.75, .5))
        voice = LocalVoice(speech, (text,))
        await voice.warm(rate)
        assert await spoken(voice, text, rate) == [-.75, .5]
        assert speech.requests == [(text, rate)]
    asyncio.run(scenario())


def test_model_location_and_config_formatting_do_not_invalidate_audio(model, tmp_path):
    async def scenario():
        directory = tmp_path / "cache"
        await LocalVoice(SyntheticPiper(model, directory), ("Yes?",)).warm(16000)
        moved = tmp_path / "moved.onnx"
        moved.write_bytes(model.read_bytes())
        moved.with_suffix(".onnx.json").write_text('{"audio": { "sample_rate": 22050 }}\n')
        speech = SyntheticPiper(moved, directory, samples=(-.75, .5))
        voice = LocalVoice(speech, ("Yes?",))
        await voice.warm(16000)
        assert await spoken(voice) == [.25, -.5]
        assert speech.requests == []
        # A warm recording at another output rate must never be replayed.
        assert await spoken(voice, rate=48000) == [-.75, .5]
        assert speech.requests == [("Yes?", 48000)]
    asyncio.run(scenario())


@pytest.mark.parametrize("corruption", ["empty", "truncated", "bit_flip", "wrong_key"])
def test_corrupt_file_is_ignored_and_replaced_with_complete_audio(model, tmp_path, corruption):
    async def scenario():
        directory = tmp_path / "cache"
        await LocalVoice(SyntheticPiper(model, directory), ("Yes?",)).warm(16000)
        path, = directory.glob("*.pcm")
        data = bytearray(path.read_bytes())
        if corruption == "empty":
            data = b""
        elif corruption == "truncated":
            data = data[:-1]
        else:
            data[10 if corruption == "wrong_key" else -1] ^= 1
        path.write_bytes(data)
        speech = SyntheticPiper(model, directory, samples=(-.75, .5))
        voice = LocalVoice(speech, ("Yes?",))
        await voice.warm(16000)
        assert await spoken(voice) == [-.75, .5]
        assert speech.requests == [("Yes?", 16000)]
        restarted = SyntheticPiper(model, directory)
        voice = LocalVoice(restarted, ("Yes?",))
        await voice.warm(16000)
        assert await spoken(voice) == [-.75, .5]
        assert restarted.requests == []
    asyncio.run(scenario())


@pytest.mark.parametrize("fail", [False, True])
def test_atomic_replace_exposes_old_or_new_pcm_and_cleans_temporary_files(model, tmp_path, monkeypatch, fail):
    cache = PhraseCache(tmp_path / "cache", model, synthesis={}, processing_version=1)
    cache.save("Yes?", 16000, (np.array([.25, -.5]),))
    observed = []
    replace = os.replace

    def commit(source, target):
        observed.append(np.concatenate(cache.load("Yes?", 16000)).tolist())
        assert source.parent == target.parent
        if fail:
            raise OSError("disk full")
        replace(source, target)
        observed.append(np.concatenate(cache.load("Yes?", 16000)).tolist())

    monkeypatch.setattr(phrase_cache.os, "replace", commit)
    cache.save("Yes?", 16000, (np.array([-.75, .5]),))
    assert observed == ([[.25, -.5]] if fail else [[.25, -.5], [-.75, .5]])
    assert np.concatenate(cache.load("Yes?", 16000)).tolist() == ([.25, -.5] if fail else [-.75, .5])
    assert [path.suffix for path in cache.directory.iterdir()] == [".pcm"]


def test_unwritable_cache_retains_audio_in_memory(model, tmp_path):
    directory = tmp_path / "not-a-directory"
    directory.write_text("occupied")

    async def scenario():
        speech = SyntheticPiper(model, directory)
        voice = LocalVoice(speech, ("Yes?",))
        await voice.warm(16000)
        await voice.warm(16000)
        assert await spoken(voice) == [.25, -.5]
        assert speech.requests == [("Yes?", 16000)]
    asyncio.run(scenario())


def test_real_piper_restart_replays_disk_without_starting_a_worker(model, tmp_path, worker):
    async def scenario():
        speech = PiperSpeech(model, cache_dir=tmp_path / "cache")
        try:
            await LocalVoice(speech, ("Yes?",)).warm(22050)
        finally:
            await speech.close()
        restarted = PiperSpeech(model, cache_dir=tmp_path / "cache")
        voice = LocalVoice(restarted, ("Yes?",))
        await voice.warm(22050)
        samples = await spoken(voice, rate=22050)
        assert len(samples) == 2205
        assert samples[:3] == [0.0, 0.045806884765625, 0.09088134765625]
        assert len(worker[1]) == 1
        assert worker[1][0].returncode == -9
    asyncio.run(scenario())


@pytest.mark.parametrize("text", ["Yes?", "Live reply."])
def test_live_speech_preempts_stalled_cache_synthesis_and_warming_resumes(model, tmp_path, worker, monkeypatch, text):
    async def scenario():
        monkeypatch.setenv("PIPER_TEST_MODE", "stall")
        speech = PiperSpeech(model, cache_dir=tmp_path / "cache")
        voice = LocalVoice(speech, ("Yes?",))
        warming = asyncio.create_task(voice.warm(22050))
        try:
            while not worker[2].exists():
                await asyncio.sleep(.005)
            monkeypatch.setenv("PIPER_TEST_MODE", "normal")
            samples = await asyncio.wait_for(spoken(voice, text, 22050), 3)
            assert len(samples) == 2205
            assert samples[:3] == [0.0, 0.045806884765625, 0.09088134765625]
            await asyncio.wait_for(warming, 3)
            assert worker[1][0].returncode == -9
            assert worker[2].read_text() == "Yes?"
            assert len(list((tmp_path / "cache").glob("*.pcm"))) == 1
        finally:
            warming.cancel()
            await asyncio.gather(warming, return_exceptions=True)
            await speech.close()
    asyncio.run(scenario())


def test_cancelled_warmup_persists_only_complete_phrases(model, tmp_path):
    async def scenario():
        partial = asyncio.Event()

        class InterruptedPiper(SyntheticPiper):
            async def stream(self, text, rate):
                async for chunk in super().stream(text, rate):
                    yield chunk
                if text == "On it.":
                    partial.set()
                    await asyncio.Future()

        directory = tmp_path / "cache"
        first = InterruptedPiper(model, directory)
        warming = asyncio.create_task(LocalVoice(first, ("Yes?", "On it.")).warm(16000))
        await asyncio.wait_for(partial.wait(), 1)
        warming.cancel()
        with pytest.raises(asyncio.CancelledError):
            await warming
        assert len(list(directory.glob("*.pcm"))) == 1
        restarted = SyntheticPiper(model, directory, samples=(-.75, .5))
        voice = LocalVoice(restarted, ("Yes?", "On it."))
        await voice.warm(16000)
        assert await spoken(voice) == [.25, -.5]
        assert await spoken(voice, "On it.") == [-.75, .5]
        assert restarted.requests == [("On it.", 16000)]
    asyncio.run(scenario())
