"""Restart a serving companion whose loop or model worker stops making progress."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


LOOP_STALL_S = 10.0
ENGINE_STALL_S = 30.0
CHECK_S = 0.5
EXIT_STATUS = 70


class Watchdog:
    """One process supervisor, armed only after all model loads and warm-ups finish.

    Track work on the worker itself: cancelling its asyncio waiter cannot hide a
    native call that is still running. Idle workers and queued calls have no timer.
    """

    def __init__(self, *, loop_s=LOOP_STALL_S, engine_s=ENGINE_STALL_S,
                 check_s=CHECK_S, clock=time.monotonic, exit_process=os._exit):
        self.loop_s, self.engine_s, self.check_s = loop_s, engine_s, check_s
        self._clock, self._exit = clock, exit_process
        self._lock = threading.Lock()
        self._workers = {}
        self._stopped = threading.Event()
        self._thread = None
        self._heartbeat = None
        self._tick = 0.0

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._stopped.clear()

        def tick():
            with self._lock:
                self._tick = self._clock()
            self._heartbeat = loop.call_later(self.check_s, tick)

        tick()
        self._thread = threading.Thread(target=self._watch, name="companion-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._heartbeat is not None:
            self._heartbeat.cancel()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    @contextmanager
    def working(self, name):
        if self._thread is None:
            yield
            return
        identity = threading.get_ident()
        with self._lock:
            self._workers[identity] = (name, self._clock())
        try:
            yield
        finally:
            with self._lock:
                self._workers.pop(identity, None)

    def progress(self) -> None:
        """A token or audio chunk completed on the calling model thread."""
        identity = threading.get_ident()
        with self._lock:
            if identity in self._workers:
                name, _ = self._workers[identity]
                self._workers[identity] = (name, self._clock())

    def _watch(self) -> None:
        while not self._stopped.wait(self.check_s):
            now = self._clock()
            with self._lock:
                stalled = [("event loop", now - self._tick, self.loop_s)]
                stalled.extend((name, now - last, self.engine_s) for name, last in self._workers.values())
            for name, age, limit in stalled:
                if age >= limit:
                    # A wedged worker may own a logging lock. Write directly to launchd's stderr.
                    message = f"companion watchdog: {name} made no progress for {age:.1f}s (limit {limit:g}s); exiting {EXIT_STATUS} for restart\n"
                    try:
                        os.write(2, message.encode())
                    finally:
                        self._exit(EXIT_STATUS)
                    return


# serve is the only owner that arms this supervisor. Engine imports and model loads are inert.
watchdog = Watchdog()


class EngineExecutor(ThreadPoolExecutor):
    def __init__(self, name, *, monitor=watchdog):
        super().__init__(1, thread_name_prefix=name)
        self._name, self._monitor = name, monitor

    def submit(self, fn, /, *args, **kwargs):
        def work():
            with self._monitor.working(self._name):
                return fn(*args, **kwargs)
        return super().submit(work)
