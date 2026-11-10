"""HTTP audio reader with live ICY metadata stripped before PyAV decoding."""

from collections import deque
import io
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from reachy_muse.music_metadata import TrackMetadata


class _Redirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme not in {"http", "https"}:
            raise ValueError("Unsupported audio redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HTTPAudio(io.RawIOBase):
    def __init__(self, source, cancelled):
        if urlsplit(source).scheme not in {"http", "https"}:
            raise ValueError("Unsupported audio URL")
        self._response = build_opener(_Redirects()).open(Request(source, headers={"Icy-MetaData": "1"}), timeout=2.)
        self._cancelled = cancelled
        try:
            self._interval = int(self._response.headers.get("icy-metaint", "0"))
            if not 0 <= self._interval <= 16 * 1024 * 1024:
                raise ValueError("Invalid metadata interval")
        except Exception:
            self._response.close()
            raise
        self._remaining = self._interval
        self._position = 0
        self._events = deque(maxlen=256)
        self._metadata = None

    def readable(self):
        return True

    def seekable(self):
        return False

    def _read_exact(self, size):
        parts = []
        while size and not self._cancelled.is_set():
            part = self._response.read(size)
            if not part:
                break
            parts.append(part)
            size -= len(part)
        return b"".join(parts)

    def read(self, size=-1):
        if self._cancelled.is_set():
            return b""
        size = min(4096, size) if size >= 0 else 4096
        if not size:
            return b""
        if self._interval and self._remaining == 0:
            length = self._read_exact(1)
            if not length:
                return b""
            packet = self._read_exact(length[0] * 16)
            if len(packet) != length[0] * 16:
                return b""
            metadata = TrackMetadata.from_icy(packet.rstrip(b"\0").decode("utf-8", errors="replace"))
            if metadata is not None:
                self._events.append((self._position, metadata))
            self._remaining = self._interval
        audio = self._response.read(min(size, self._remaining) if self._interval else size)
        self._position += len(audio)
        if self._interval:
            self._remaining -= len(audio)
        return audio

    def metadata_at(self, position):
        while self._events and self._events[0][0] <= position:
            _, self._metadata = self._events.popleft()
        return self._metadata

    def close(self):
        self._response.close()
        super().close()
