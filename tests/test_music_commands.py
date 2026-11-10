import asyncio
import math
from pathlib import Path
import threading
import time

import pytest

from reachy_muse.contract.commands import parse_command
from reachy_muse.conversation import _CommandParams
from reachy_muse.device_command import WakeWindow
from reachy_muse.music import MusicController
from test_music import Sink, constant
from test_registered_commands import command_device, wait_until


def test_music_payload_defaults_and_limits():
    assert parse_command("reachy.music.play", {"query": "shuffle"}).params == {
        "query": "shuffle", "dance": True, "volume": .35}
    assert parse_command("reachy.music.play", {"url": "https://example.com/a.mp3", "dance": False, "volume": .5}).params == {
        "url": "https://example.com/a.mp3", "dance": False, "volume": .5}
    assert parse_command("reachy.dance", {}).params == {"seconds": 60}
    assert parse_command("reachy.dance", {"seconds": 600}).params == {"seconds": 600}
    assert parse_command("reachy.music.stop", {}).params == {}
    assert parse_command("reachy.music.status", {}).params == {}


@pytest.mark.parametrize("params", [
    {}, {"query": "x", "url": "https://example.com/a"}, {"query": ""}, {"query": " "},
    {"query": "x" * 241}, {"query": "secret\ntrack"}, {"query": True}, {"url": "file:///tmp/audio.wav"},
    {"url": "ftp://example.com/a"}, {"url": "https://"}, {"url": "https://u:p@example.com/a"},
    {"url": "https://example.com:abc/a"}, {"url": "https://example.com/a#secret"},
    {"url": "https://example.com/a b"}, {"query": "x", "dance": 1}, {"query": "x", "volume": True},
    {"query": "x", "volume": -.01}, {"query": "x", "volume": .51}, {"query": "x", "volume": math.nan},
    {"query": "x", "volume": math.inf}, {"query": "x", "unknown": "private"},
])
def test_invalid_music_sources_and_options(params):
    with pytest.raises(ValueError):
        parse_command("reachy.music.play", params)


@pytest.mark.parametrize("seconds", [True, 0, -1, 601, math.inf, math.nan, "60"])
def test_invalid_room_duration(seconds):
    with pytest.raises(ValueError):
        parse_command("reachy.dance", {"seconds": seconds})


def test_registered_music_commands_and_stale_play_after_stop_or_hush():
    async def run():
        service, voice, _ = command_device()
        sink = Sink()
        music = voice.music = MusicController(sink.output(), 16000, decoder=constant)
        async def invoke(name, params={}):
            return await asyncio.to_thread(service.run_command, name, params)
        try:
            voice._turn_active = True
            result = await invoke("reachy.music.play", {"url": "https://example.com/private.mp3"})
            assert result == {"ok": True, "payload": {"playing": True, "elapsed": 0., "dancing": False,
                                                       "source": "https://example.com/private.mp3",
                                                       "music_detected": False, "energy": 0., "character": "calm"}}
            status = (await invoke("reachy.music.status"))["payload"]
            assert {k: status[k] for k in ("playing", "dancing", "source")} == {
                "playing": True, "dancing": False, "source": "https://example.com/private.mp3"}
            old = _CommandParams("reachy.music.play", {"url": "https://example.com/old"}, None)
            assert await invoke("reachy.music.stop") == {"ok": True, "payload": {"playing": False, "elapsed": 0., "dancing": False}}
            assert await invoke("reachy.music.play", old) == {"ok": False, "error": "interrupted"}
            old = _CommandParams("reachy.music.play", {"url": "https://example.com/old"}, None)
            voice._hush()
            assert await invoke("reachy.music.play", old) == {"ok": False, "error": "unavailable"}
            assert not music.active
            voice._say_hushed = False
            voice._turn_active = False
            result = await invoke("reachy.dance", {"seconds": 2})
            assert result == {"ok": True, "payload": {"playing": False, "elapsed": 0., "dancing": False,
                                                        "music_detected": False, "energy": 0., "character": "calm"}}
            assert await asyncio.to_thread(service.run_command, "reachy.music.stop", {}, session=object()) == {
                "ok": False, "error": "disconnected"}
            assert music.active
        finally:
            music.close()
    asyncio.run(run())


def test_slow_folder_lookup_cannot_restart_music_after_stop(monkeypatch, tmp_path):
    entered, release = threading.Event(), threading.Event()
    def find(*args):
        entered.set()
        release.wait(2)
        return tmp_path / "music.wav"
    monkeypatch.setattr("reachy_muse.music.select_track", find)
    async def run():
        service, voice, _ = command_device()
        music = voice.music = MusicController(Sink().output(), 16000, decoder=constant)
        play = asyncio.create_task(asyncio.to_thread(service.run_command, "reachy.music.play", {"query": "private"}))
        try:
            await wait_until(entered.is_set)
            assert (await asyncio.to_thread(service.run_command, "reachy.music.stop", {}))["ok"]
            release.set()
            assert await play == {"ok": False, "error": "music_unavailable"}
            assert not music.active
        finally:
            release.set()
            await play
            music.close()
    asyncio.run(run())


@pytest.mark.parametrize("state", ["asleep", "hushed", "not_ready", "expired"])
def test_unavailable_or_expired_commands_cannot_start_music(state):
    async def run():
        service, voice, _ = command_device()
        music = voice.music = MusicController(Sink().output(), 16000, decoder=constant)
        params = _CommandParams("reachy.dance", {}, None)
        if state == "asleep":
            voice._wake_window = WakeWindow.ASLEEP
        elif state == "hushed":
            voice._hush()
        elif state == "not_ready":
            voice._ready.clear()
        else:
            params.deadline = time.monotonic() - 1
        try:
            result = await asyncio.to_thread(service.run_command, "reachy.dance", params)
            assert result == {"ok": False, "error": "expired" if state == "expired" else "unavailable"}
            assert not music.active
        finally:
            music.close()
    asyncio.run(run())


def test_cli_music_folder_default_and_override():
    from reachy_muse.cli import parser
    assert parser().parse_args(["run"]).music_dir == Path("~/Music")
    assert parser().parse_args(["run", "--music-dir", "/srv/music"]).music_dir == Path("/srv/music")
