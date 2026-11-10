import io
import json
import threading
from urllib.parse import parse_qs

import numpy as np
import pytest

from reachy_muse import music_identity
from reachy_muse.music_identity import RoomSongID
from reachy_muse.music import MusicController
from test_music import Sink


def test_missing_chromaprint_and_key_skip_without_network(monkeypatch):
    monkeypatch.setattr(music_identity.shutil, "which", lambda name: None)
    monkeypatch.setattr(music_identity, "urlopen", lambda *a, **kw: pytest.fail("network"))
    for key in (None, "secret-key"):
        identifier = RoomSongID(16000, key)
        assert identifier.feed(np.ones(320000, np.float32), True) is None
        assert identifier.lookup(np.ones(320000, np.float32), threading.Event()) is None


def test_only_fingerprint_and_protocol_fields_leave_robot(monkeypatch, caplog):
    monkeypatch.setattr(music_identity.shutil, "which", lambda name: "/local/fpcalc")
    class Process:
        returncode = 0
        def __init__(self, args, **kwargs):
            assert args == ["/local/fpcalc", "-format", "s16le", "-rate", "16000", "-channels", "1", "-length", "20", "-json", "-"]
        def communicate(self, input=None, timeout=None):
            assert len(input) == 640000
            return b'{"duration":20,"fingerprint":"local-fingerprint"}', None
        def poll(self):
            return 0
    monkeypatch.setattr(music_identity.subprocess, "Popen", Process)
    calls = []
    def lookup(request, timeout):
        calls.append(request)
        assert request.full_url == "https://api.acoustid.org/v2/lookup"
        assert request.method == "POST"
        assert parse_qs(request.data.decode()) == {"client": ["owner-key"], "duration": ["20"],
                "fingerprint": ["local-fingerprint"], "meta": ["recordings releasegroups"]}
        return io.BytesIO(json.dumps({"status": "ok", "results": [{"score": .95, "recordings": [
            {"title": "Private title", "artists": [{"name": "Private artist"}], "releasegroups": [{"title": "Private album"}]}]}]}).encode())
    monkeypatch.setattr(music_identity, "urlopen", lookup)
    identifier = RoomSongID(16000, "owner-key")
    pcm = identifier.feed(np.ones(320000, np.float32) * .1, True)
    result = identifier.lookup(pcm, threading.Event())
    assert result.payload() == {"title": "Private title", "artist": "Private artist", "album": "Private album"}
    assert len(calls) == 1
    assert identifier.feed(np.ones(320000, np.float32), True) is None
    assert caplog.text == ""


def test_speech_resets_collection_and_cancellation_prevents_lookup(monkeypatch):
    monkeypatch.setattr(music_identity.shutil, "which", lambda name: "/local/fpcalc")
    monkeypatch.setattr(music_identity.subprocess, "Popen", lambda *a, **kw: pytest.fail("fingerprinting after stop"))
    identifier = RoomSongID(16000, "key")
    assert identifier.feed(np.ones(160000, np.float32), True) is None
    assert identifier.feed(np.zeros(100, np.float32), False) is None
    assert identifier.feed(np.ones(160000, np.float32), True) is None
    pcm = identifier.feed(np.ones(160000, np.float32), True)
    assert len(pcm) == 320000
    stopped = threading.Event()
    stopped.set()
    assert identifier.lookup(pcm, stopped) is None


@pytest.mark.parametrize("enabled,key", [(False, "present"), (True, None)])
def test_disabled_identification_never_collects_room_audio(monkeypatch, enabled, key):
    monkeypatch.setattr("reachy_muse.music.RoomSongID", lambda *a: pytest.fail("identity was enabled"))
    controller = MusicController(Sink().output(), 16000, room_song_id=enabled, acoustid_api_key=key)
    try:
        controller.listen()
        controller.tap(np.ones(640, np.float32), 10.)
        assert controller._identity_worker is None
    finally:
        controller.close()


def test_opt_in_flag_and_environment_key(monkeypatch):
    from reachy_muse.cli import parser
    monkeypatch.setenv("ACOUSTID_API_KEY", "environment-key")
    off = parser().parse_args(["run"])
    assert off.room_song_id is False
    assert off.acoustid_api_key == "environment-key"
    on = parser().parse_args(["run", "--room-song-id", "--acoustid-api-key", "flag-key"])
    assert on.room_song_id is True
    assert on.acoustid_api_key == "flag-key"
