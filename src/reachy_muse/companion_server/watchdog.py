"""Restart a serving companion whose loop or model worker stops making progress."""

from __future__ import annotations

import asyncio
import faulthandler
import os
import resource
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Optional


LOOP_STALL_S = 10.0
ENGINE_STALL_S = 30.0
HEARING_STALL_S = 120.0
CHECK_S = 0.5
EXIT_STATUS = 70


class Watchdog:
    """One process supervisor, armed only after all model loads and warm-ups finish.

    Track work on the worker itself: cancelling its asyncio waiter cannot hide a
    native call that is still running. Idle workers and queued calls have no timer.
    """

    def __init__(self, *, loop_s=LOOP_STALL_S, engine_s=ENGINE_STALL_S, hearing_s=HEARING_STALL_S,
                 check_s=CHECK_S, clock=time.monotonic, exit_process=os._exit):
        self.loop_s, self.engine_s, self.check_s = loop_s, engine_s, check_s
        self.hearing_s = hearing_s
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
            self._workers[identity] = (name, self._clock(), 0.0)
        try:
            yield
        finally:
            with self._lock:
                self._workers.pop(identity, None)

    def progress(self, quiet_s: float = 0.0) -> None:
        """A token or audio chunk completed on the calling model thread.

        Its next stage may then run ``quiet_s`` without progress before ``stalled`` names the
        thread. The hard limit is unchanged.
        """
        identity = threading.get_ident()
        with self._lock:
            if identity in self._workers:
                name = self._workers[identity][0]
                self._workers[identity] = (name, self._clock(), quiet_s)

    def stalled(self, name: str, limit_s: float) -> Optional[float]:
        """Seconds a ``name`` worker has gone without progress, once past ``limit_s`` and its stage's allowance."""
        with self._lock:
            now = self._clock()
            return max((now - last for worker, last, quiet_s in self._workers.values()
                        if worker == name and now - last >= max(limit_s, quiet_s)), default=None)

    def _watch(self) -> None:
        while not self._stopped.wait(self.check_s):
            now = self._clock()
            with self._lock:
                stalled = [("event loop", now - self._tick, self.loop_s)]
                stalled.extend((name, now - last, self.hearing_s if name == "hearing" else self.engine_s)
                               for name, last, _ in self._workers.values())
            for name, age, limit in stalled:
                if age >= limit:
                    # A wedged worker may own a logging lock. Write directly to launchd's stderr.
                    message = f"companion watchdog: {name} made no progress for {age:.1f}s (limit {limit:g}s); exiting {EXIT_STATUS} for restart\n"
                    try:
                        os.write(2, message.encode())
                    finally:
                        try:
                            faulthandler.dump_traceback(file=2, all_threads=True)
                        finally:
                            try:
                                usage = resource.getrusage(resource.RUSAGE_SELF)
                                os.write(2, f"companion watchdog: page faults: minor={usage.ru_minflt} major={usage.ru_majflt}\n".encode())
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
