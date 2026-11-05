"""Robot activity and a killable, low-priority offline motion builder.

The companion writes only at operation boundaries. The builder holds one advisory
lock for its lifetime and watches that status while its own child does model work.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
from typing import Iterator, Sequence
import uuid

log = logging.getLogger("reachy_muse.companion_server")

ACTIVITY_FILE = "robot-activity.json"
BUILD_LOCK = "motion-library-build.lock"
PARENT_FD = "REACHY_MOTION_BUILD_PARENT_FD"
POLL_S = 0.025
ABORTED = 75


class ActivityStatus:
    """One companion's activity across all authenticated connections."""

    def __init__(self, state_dir: Path) -> None:
        self.path = state_dir / ACTIVITY_FILE
        state_dir.mkdir(parents=True, exist_ok=True)
        self._session = uuid.uuid4().hex
        self._revision = 0
        self._active = 0
        self.touch()

    def touch(self) -> None:
        """Record receipt before scheduling an op, and restart the quiet interval."""
        self._revision += 1
        status = {"version": 1, "session": self._session, "revision": self._revision,
                  "active": self._active, "wall": time.time(), "monotonic": time.monotonic()}
        fd, name = tempfile.mkstemp(prefix=".robot-activity-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as file:
                json.dump(status, file, separators=(",", ":"), allow_nan=False)
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)

    @contextlib.contextmanager
    def busy(self) -> Iterator[None]:
        self._active += 1
        try:
            self.touch()
            yield
        finally:
            self._active -= 1
            self.touch()


def _read_activity(state_dir: Path) -> dict | None:
    try:
        with (state_dir / ACTIVITY_FILE).open() as file:
            value = json.loads(file.read(4096))
    except FileNotFoundError:
        return None
    if (not isinstance(value, dict) or value.get("version") != 1
            or not isinstance(value.get("session"), str) or not value["session"]
            or type(value.get("revision")) is not int or value["revision"] < 1
            or type(value.get("active")) is not int or value["active"] < 0):
        raise ValueError("invalid robot activity status")
    for field in ("wall", "monotonic"):
        if type(value.get(field)) not in (int, float) or not math.isfinite(value[field]) or value[field] < 0:
            raise ValueError("invalid robot activity timestamp")
    return value


def _quiet(status: dict | None, idle_seconds: float, require_status: bool) -> bool:
    if status is None:
        return not require_status
    # Both clocks must agree that the interval elapsed. A backward clock change
    # or a future monotonic timestamp from another boot fails closed.
    return (status["active"] == 0 and time.time() - status["wall"] >= idle_seconds
            and time.monotonic() - status["monotonic"] >= idle_seconds)


def run_guarded_build(state_dir: Path, command: Sequence[str], *, idle_seconds: float = 1200,
                      require_status: bool = True) -> int:
    """Run one owned child, or skip if activity is unknown, recent, or another build owns the lock.

    The child must call watch_build_parent() before importing or loading models.
    Status changes abort with exit code 75. A skipped build returns zero so the
    nightly launchd job does not retry while the robot is busy.
    """
    if not math.isfinite(idle_seconds) or idle_seconds < 0:
        raise ValueError("idle_seconds must be finite and nonnegative")
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / BUILD_LOCK).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("motion library build skipped: another build is running")
            return 0
        try:
            before = _read_activity(state_dir)
        except (OSError, ValueError):
            log.warning("motion library build skipped: robot activity is unreadable")
            return 0
        if not _quiet(before, idle_seconds, require_status):
            log.info("motion library build skipped: robot has not been quiet for %.0f minutes", idle_seconds / 60)
            return 0

        read_fd, write_fd = os.pipe()
        process = None
        try:
            env = dict(os.environ, **{PARENT_FD: str(read_fd)})
            process = subprocess.Popen(command, env=env, pass_fds=(read_fd,), start_new_session=True)
            os.close(read_fd)
            read_fd = -1

            def unchanged() -> bool:
                try:
                    status = _read_activity(state_dir)
                    return status == before and _quiet(status, idle_seconds, require_status)
                except (OSError, ValueError):
                    return False

            # The child waits for this byte before it can load a model. Recheck
            # after spawn, because an op may have arrived during process creation.
            if not unchanged():
                log.info("motion library build aborted: robot activity changed")
                return ABORTED
            try:
                os.write(write_fd, b"G")
            except BrokenPipeError:
                return process.wait()
            while True:
                if not unchanged():
                    log.info("motion library build aborted: robot activity changed")
                    return ABORTED
                result = process.poll()
                if result is not None:
                    return result
                time.sleep(POLL_S)
        finally:
            try:
                if process is not None:
                    if process.poll() is None:
                        # This is our live process group, never a PID read from disk.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            if process.poll() is None:
                                raise
                    process.wait()
            finally:
                if read_fd >= 0:
                    os.close(read_fd)
                os.close(write_fd)


def watch_build_parent() -> None:
    """Child entry gate and parent-death watcher; call before any model work."""
    inherited = os.environ.pop(PARENT_FD, None)
    if inherited is None:
        raise RuntimeError("motion build worker must be started by its supervisor")
    fd = int(inherited)
    if os.read(fd, 1) != b"G":
        os._exit(ABORTED)
    os.nice(10)

    def parent_closed() -> None:
        try:
            while os.read(fd, 1):
                pass
        finally:
            os._exit(ABORTED)

    threading.Thread(target=parent_closed, name="motion-build-parent", daemon=True).start()
