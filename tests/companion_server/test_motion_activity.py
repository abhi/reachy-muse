import asyncio
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest
from websockets.asyncio.client import connect

from reachy_muse.companion_server import motion_activity as activity
from reachy_muse.companion_server.engines import ScriptedHearing
from reachy_muse.companion_server.server import CompanionServer, Engines
from reachy_muse.protocol import HearOpen, Hello, Op, SpeakStart, VoiceRole, encode


def status(path):
    return json.loads((path / activity.ACTIVITY_FILE).read_text())


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition did not become true"
        time.sleep(.005)


def child_command(marker, *, hang=True):
    code = ("from reachy_muse.companion_server.motion_activity import watch_build_parent; "
            "watch_build_parent(); from pathlib import Path; import time; "
            f"Path({str(marker)!r}).write_text('started'); "
            + ("time.sleep(30)" if hang else ""))
    return [sys.executable, "-c", code]


def test_activity_covers_nested_ops_and_finishing_restarts_quiet_time(tmp_path, monkeypatch):
    now = [2000.0]
    monkeypatch.setattr(activity.time, "time", lambda: now[0])
    monkeypatch.setattr(activity.time, "monotonic", lambda: now[0] / 2)
    tracker = activity.ActivityStatus(tmp_path)
    initial = status(tmp_path)
    assert {key: initial[key] for key in ("version", "revision", "active", "wall", "monotonic")} == {
        "version": 1, "revision": 1, "active": 0, "wall": 2000.0, "monotonic": 1000.0}
    with tracker.busy():
        with tracker.busy():
            assert status(tmp_path)["active"] == 2
            now[0] = 8000.0
            assert activity._quiet(status(tmp_path), 1200, True) is False
        assert status(tmp_path)["active"] == 1
    assert {key: status(tmp_path)[key] for key in ("revision", "active", "wall", "monotonic")} == {
        "revision": 5, "active": 0, "wall": 8000.0, "monotonic": 4000.0}
    assert activity._quiet(status(tmp_path), 1200, True) is False
    now[0] = 10400.0
    assert activity._quiet(status(tmp_path), 1200, True) is True


def test_activity_writes_are_atomic_and_cleanup_temp_files(tmp_path, monkeypatch):
    tracker = activity.ActivityStatus(tmp_path)
    original = (tmp_path / activity.ACTIVITY_FILE).read_text()
    replace = os.replace
    seen = []

    def inspect(source, target):
        seen.append(json.loads(Path(source).read_text())["revision"])
        assert Path(target).read_text() == original
        replace(source, target)

    monkeypatch.setattr(activity.os, "replace", inspect)
    tracker.touch()
    assert seen == [2]
    assert status(tmp_path)["revision"] == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == ["robot-activity.json"]


@pytest.mark.parametrize("contents", ["{", "{}", '{"version":1,"session":"a","revision":1,"active":0,"wall":NaN,"monotonic":0}'])
def test_invalid_status_never_starts_a_child(tmp_path, contents):
    (tmp_path / activity.ACTIVITY_FILE).write_text(contents)
    marker = tmp_path / "child"
    assert activity.run_guarded_build(tmp_path, child_command(marker), idle_seconds=0, require_status=False) == 0
    assert marker.exists() is False


def test_missing_status_skips_nightly_but_allows_first_manual_build(tmp_path):
    marker = tmp_path / "child"
    assert activity.run_guarded_build(tmp_path, child_command(marker, hang=False)) == 0
    assert marker.exists() is False
    assert activity.run_guarded_build(tmp_path, child_command(marker, hang=False), require_status=False) == 0
    assert marker.read_text() == "started"


def test_recent_or_active_status_never_starts_even_a_manual_child(tmp_path):
    tracker = activity.ActivityStatus(tmp_path)
    marker = tmp_path / "child"
    assert activity.run_guarded_build(tmp_path, child_command(marker), require_status=False) == 0
    with tracker.busy():
        assert activity.run_guarded_build(tmp_path, child_command(marker), idle_seconds=0, require_status=False) == 0
    assert marker.exists() is False


def test_backward_wall_or_monotonic_clock_fails_closed(tmp_path, monkeypatch):
    activity.ActivityStatus(tmp_path)
    recorded = status(tmp_path)
    monkeypatch.setattr(activity.time, "time", lambda: recorded["wall"] + 1300)
    monkeypatch.setattr(activity.time, "monotonic", lambda: recorded["monotonic"] - 1)
    assert activity._quiet(recorded, 1200, True) is False
    monkeypatch.setattr(activity.time, "time", lambda: recorded["wall"] - 1)
    monkeypatch.setattr(activity.time, "monotonic", lambda: recorded["monotonic"] + 1300)
    assert activity._quiet(recorded, 1200, True) is False


def test_owned_child_is_killed_promptly_when_activity_changes(tmp_path):
    tracker = activity.ActivityStatus(tmp_path)
    marker = tmp_path / "child"
    result = []
    runner = threading.Thread(target=lambda: result.append(activity.run_guarded_build(
        tmp_path, child_command(marker), idle_seconds=0)))
    runner.start()
    try:
        wait_for(marker.exists)
        changed = time.monotonic()
        tracker.touch()
        runner.join(timeout=1)
        assert result == [75]
        assert time.monotonic() - changed < .2
    finally:
        tracker.touch()
        runner.join(timeout=3)
    assert runner.is_alive() is False


def test_activity_during_spawn_aborts_before_child_starts_work(tmp_path, monkeypatch):
    tracker = activity.ActivityStatus(tmp_path)
    marker = tmp_path / "child"
    popen = subprocess.Popen

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        tracker.touch()
        return child

    monkeypatch.setattr(activity.subprocess, "Popen", spawn)
    assert activity.run_guarded_build(tmp_path, child_command(marker), idle_seconds=0) == 75
    assert marker.exists() is False


def test_child_startup_failure_preserves_its_exit_code(tmp_path, monkeypatch):
    popen = subprocess.Popen

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        child.wait(timeout=2)
        return child

    monkeypatch.setattr(activity.subprocess, "Popen", spawn)
    assert activity.run_guarded_build(tmp_path, [sys.executable, "-c", "raise SystemExit(7)"],
                                      require_status=False) == 7


def test_a_live_lock_skips_but_a_leftover_lock_file_does_not(tmp_path):
    marker = tmp_path / "child"
    lock_path = tmp_path / activity.BUILD_LOCK
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert activity.run_guarded_build(tmp_path, child_command(marker), require_status=False) == 0
        assert marker.exists() is False
    assert activity.run_guarded_build(tmp_path, child_command(marker, hang=False), require_status=False) == 0
    assert marker.read_text() == "started"


def test_child_exits_when_supervisor_pipe_closes(tmp_path):
    marker = tmp_path / "child"
    read_fd, write_fd = os.pipe()
    child = subprocess.Popen(child_command(marker), pass_fds=(read_fd,),
                             env=dict(os.environ, **{activity.PARENT_FD: str(read_fd)}))
    os.close(read_fd)
    try:
        os.write(write_fd, b"G")
        wait_for(marker.exists)
        os.close(write_fd)
        write_fd = -1
        assert child.wait(timeout=2) == 75
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        if child.poll() is None:
            child.kill()
        child.wait()


def test_worker_cannot_run_without_supervisor(monkeypatch):
    monkeypatch.delenv(activity.PARENT_FD, raising=False)
    with pytest.raises(RuntimeError, match="must be started by its supervisor"):
        activity.watch_build_parent()


def test_authenticated_hear_and_speak_record_activity_until_disconnect(tmp_path):
    tracker = activity.ActivityStatus(tmp_path)
    observed = []

    class HoldingVoice:
        name = "holding"

        async def speak(self, *args):
            observed.append(status(tmp_path)["active"])
            yield b"\x00\x00" * 2400
            await asyncio.Event().wait()

    async def run():
        async with CompanionServer(Engines(hearing=ScriptedHearing(), voice=HoldingVoice()), "token",
                                   activity=tracker) as server:
            async with connect(server.url) as ws:
                await ws.send(encode(Hello((1,), "token", (Op.HEAR, Op.SPEAK), 24000, "test")))
                await ws.recv()
                await ws.send(encode(HearOpen(1, 0)))
                await ws.send(encode(SpeakStart(2, "hello", VoiceRole.MAIN)))
                await ws.recv()
                assert status(tmp_path)["active"] == 2
                assert status(tmp_path)["revision"] == 5
        assert status(tmp_path)["active"] == 0
        assert status(tmp_path)["revision"] == 7

    asyncio.run(run())
    assert observed == [2]
