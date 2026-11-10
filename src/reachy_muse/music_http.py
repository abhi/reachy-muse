"""HTTP reads with cancellation and a deadline spanning header/chunk parsing."""

import http.client
import io
import socket
import time
from urllib.parse import urlsplit
from urllib.request import HTTPHandler, HTTPSHandler, HTTPRedirectHandler, build_opener


class _Budget:
    def __init__(self, cancelled, seconds):
        self.cancelled = cancelled
        self.reset(seconds)

    def reset(self, seconds):
        self.deadline = time.monotonic() + seconds

    def receive(self, sock, buffer):
        while True:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0 or self.cancelled.is_set():
                raise TimeoutError("Audio request ended")
            sock.settimeout(min(.2, remaining))
            try:
                return sock.recv_into(buffer)
            except socket.timeout:
                continue


class _Reader(io.RawIOBase):
    def __init__(self, sock, budget):
        self.sock, self.budget = sock, budget

    def readable(self):
        return True

    def readinto(self, buffer):
        return self.budget.receive(self.sock, buffer)

    def close(self):
        self.sock.close()
        super().close()


class _Socket:
    def __init__(self, sock, budget):
        self.sock, self.budget = sock, budget
        self.reader = None

    def __getattr__(self, name):
        return getattr(self.sock, name)

    def makefile(self, mode, *args, **kwargs):
        self.reader = _Reader(self.sock, self.budget)
        return io.BufferedReader(self.reader)

    def close(self):
        if self.reader is None or self.reader.closed:
            self.sock.close()


class _Redirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme not in {"http", "https"}:
            raise ValueError("Unsupported audio redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_http(request, *, timeout, cancelled):
    budget = _Budget(cancelled, timeout)

    class Connection(http.client.HTTPConnection):
        def connect(self):
            super().connect()
            self.sock = _Socket(self.sock, budget)

    class SecureConnection(http.client.HTTPSConnection):
        def connect(self):
            super().connect()
            self.sock = _Socket(self.sock, budget)

    class HTTP(HTTPHandler):
        def http_open(self, req):
            return self.do_open(Connection, req)

    class HTTPS(HTTPSHandler):
        def https_open(self, req):
            return self.do_open(SecureConnection, req, context=self._context)

    if cancelled.is_set():
        raise TimeoutError("Audio request ended")
    response = build_opener(_Redirects(), HTTP(), HTTPS()).open(request, timeout=timeout)
    response.music_budget = budget
    return response


def read_budget(response, seconds):
    budget = getattr(response, "music_budget", None)
    if budget is not None:
        budget.reset(seconds)
