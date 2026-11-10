"""Opt-in room fingerprinting. Audio stays local; lookup sends a fingerprint."""

import json
import shutil
import subprocess
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from reachy_muse.music_metadata import TrackMetadata


class RoomSongID:
    def __init__(self, rate, key):
        self.rate = rate
        self._key = key
        self._program = shutil.which("fpcalc") if key else None
        self._parts = []
        self._count = 0
        self._attempted = False

    def feed(self, pcm, music):
        if self._attempted or self._program is None:
            return None
        if not music:
            self._parts.clear()
            self._count = 0
            return None
        remaining = self.rate * 20 - self._count
        self._parts.append(pcm[:remaining].copy())
        self._count += min(len(pcm), remaining)
        if self._count < self.rate * 20:
            return None
        audio = np.concatenate(self._parts)
        self._parts.clear()
        self._attempted = True
        return audio

    def lookup(self, pcm, cancelled):
        if cancelled.is_set() or self._program is None:
            return None
        process = None
        try:
            audio = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
            process = subprocess.Popen([self._program, "-format", "s16le", "-rate", str(self.rate),
                                        "-channels", "1", "-length", "20", "-json", "-"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + 5
            while not cancelled.is_set() and time.monotonic() < deadline:
                try:
                    output, _ = process.communicate(input=audio, timeout=.1)
                    break
                except subprocess.TimeoutExpired:
                    audio = None
            else:
                return None
            if process.returncode or cancelled.is_set():
                return None
            result = json.loads(output)
            fingerprint = result.get("fingerprint")
            if not isinstance(fingerprint, str) or not 1 <= len(fingerprint) <= 20000:
                return None
            body = urlencode({"client": self._key, "duration": round(len(pcm) / self.rate),
                              "fingerprint": fingerprint, "meta": "recordings releasegroups"}).encode()
            request = Request("https://api.acoustid.org/v2/lookup", data=body,
                              headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
            if cancelled.is_set():
                return None
            with urlopen(request, timeout=4.) as response:
                payload = json.loads(response.read(65537))
            if cancelled.is_set() or payload.get("status") != "ok":
                return None
            matches = sorted(payload.get("results", []), key=lambda item: item.get("score", 0), reverse=True)
            if not matches or matches[0].get("score", 0) < .8:
                return None
            recordings = matches[0].get("recordings", [])
            if not recordings:
                return None
            recording = recordings[0]
            artists = recording.get("artists", [])
            albums = recording.get("releasegroups", [])
            return TrackMetadata.from_tags({"title": recording.get("title"),
                                             "artist": ", ".join(a["name"] for a in artists),
                                             "album": albums[0].get("title") if albums else None})
        except Exception:
            return None
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()
