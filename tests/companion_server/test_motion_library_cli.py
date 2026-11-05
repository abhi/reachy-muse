import argparse
import asyncio
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_muse.companion_server import cli


def test_explicit_voice_selection_is_preserved():
    assert cli._voice("kokoro:af_heart") == "kokoro:af_heart"


@pytest.mark.parametrize("outcome", ["loaded", "failed", "cancelled"])
def test_engine_startup_blocks_builds_and_restarts_quiet_time(tmp_path, monkeypatch, outcome):
    from reachy_muse.companion_server import motion_activity

    now = [2000.0]
    observed = []
    monkeypatch.setattr(motion_activity.time, "time", lambda: now[0])
    monkeypatch.setattr(motion_activity.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(cli.pairing, "load", lambda directory: SimpleNamespace(token="test"))

    async def engines(args):
        observed.append(motion_activity._read_activity(tmp_path)["active"])
        now[0] += 1300
        observed.append(motion_activity._quiet(motion_activity._read_activity(tmp_path), 1200, True))
        if outcome == "failed":
            raise RuntimeError("failed")
        if outcome == "cancelled":
            raise asyncio.CancelledError()
        return cli.Engines()

    def loaded():
        raise RuntimeError("loaded")

    monkeypatch.setattr(cli, "_engines", engines)
    monkeypatch.setattr(cli.watchdog, "start", loaded)
    error = asyncio.CancelledError if outcome == "cancelled" else RuntimeError
    with pytest.raises(error):
        asyncio.run(cli._serve(argparse.Namespace(state_dir=tmp_path)))

    assert observed == [1, False]
    finished = motion_activity._read_activity(tmp_path)
    assert {key: finished[key] for key in ("active", "wall", "monotonic")} == {
        "active": 0, "wall": 3300.0, "monotonic": 3300.0}
    assert motion_activity._quiet(finished, 1200, True) is False
    now[0] += 1200
    assert motion_activity._quiet(finished, 1200, True) is True


def test_worker_builds_then_resumes_without_generating_again(tmp_path, monkeypatch, capsys):
    from reachy_muse.companion_server import motion, motion_activity, motion_library

    calls = []

    class FakeEngine:
        def __init__(self, directory):
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
            calls.append("engine")

        def generate(self, intent, duration, *, seed):
            calls.append("generate")
            return SimpleNamespace(fps=25, recipe="fake", frames=np.zeros((round(25 * duration), 9)))

        def close(self):
            calls.append("close")

    monkeypatch.setattr(motion_activity, "watch_build_parent", lambda: calls.append("gate"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setattr(motion_library, "model_identity", lambda directory: {"fake": "weights"})
    monkeypatch.setattr(motion, "MotionEngine", FakeEngine)
    args = argparse.Namespace(state_dir=tmp_path, motion_dir=None, variants=2, only=["happy"])
    cli._build_library(args)
    cli._build_library(args)

    assert calls == ["gate", "engine", "generate", "generate", "close", "gate", "engine", "close"]
    assert capsys.readouterr().out.splitlines() == [
        f"built 2 motions in {tmp_path / 'motion-library'}", f"built 0 motions in {tmp_path / 'motion-library'}"]


def test_live_planner_and_library_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit) as error:
        cli.main(["serve", "--motion", "--motion-library"])
    assert error.value.code == 2


@pytest.mark.parametrize("option", ["--variants", "--if-idle"])
@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_build_rejects_invalid_counts_before_starting_a_process(option, value):
    with pytest.raises(SystemExit) as error:
        cli.main(["motion-library", "build", option, value])
    assert error.value.code == 2


def test_build_default_and_nightly_options_reach_the_supervisor(tmp_path, monkeypatch):
    from reachy_muse.companion_server import motion_activity

    calls = []
    monkeypatch.setattr(motion_activity, "run_guarded_build", lambda *a, **kw: calls.append((a, kw)) or 0)
    for extra in ([], ["--variants", "3", "--only", "happy", "nod", "--only", "wake", "--if-idle", "20"]):
        with pytest.raises(SystemExit) as error:
            cli.main(["--state-dir", str(tmp_path), "motion-library", "build", *extra])
        assert error.value.code == 0

    prefix = [sys.executable, "-m", "reachy_muse.companion_server", "--state-dir", str(tmp_path),
              "motion-library", "build", "--_worker", "--variants"]
    assert calls == [
        ((tmp_path, prefix + ["12"]), {"idle_seconds": 1200, "require_status": False}),
        ((tmp_path, prefix + ["3", "--only", "happy", "nod", "wake"]),
         {"idle_seconds": 1200, "require_status": True}),
    ]


def test_library_serve_never_imports_the_motion_engine_or_ml_runtimes(tmp_path, monkeypatch):
    for name in ("reachy_muse.companion_server.motion", "torch", "mlx", "mlx_lm", "generator", "planner"):
        monkeypatch.setitem(sys.modules, name, None)

    async def scenario():
        engines = await cli._engines(argparse.Namespace(engines="stub", state_dir=tmp_path,
                                                        motion=False, motion_library=True))
        assert engines.motion.name == "reachy-motion-library"
        missing = await engines.motion.animate("happy", 2.0)
        assert (missing.fps, missing.frames) == (25, ())

    asyncio.run(scenario())


def test_library_animate_preserves_wire_frames_without_a_planner(tmp_path, monkeypatch):
    from websockets.asyncio.client import connect
    from reachy_muse.companion_server.motion_library import build_library
    from reachy_muse.companion_server.server import CompanionServer
    from reachy_muse.protocol import Animate, AnimateResult, Hello, Op, Sender, Welcome, decode, encode

    class FakeEngine:
        def generate(self, intent, duration, *, seed):
            return SimpleNamespace(fps=25, recipe="fake", frames=np.tile(
                [1, -2, 3, 4, 5, 6, 7, 8, 9], (round(25 * duration), 1)))

    build_library(tmp_path / "motion-library", FakeEngine(), {}, variants=1, only=["greeting"])
    monkeypatch.setitem(sys.modules, "reachy_muse.companion_server.motion", None)

    async def scenario():
        engines = await cli._engines(argparse.Namespace(engines="stub", state_dir=tmp_path,
                                                        motion=False, motion_library=True))
        async with CompanionServer(engines, "test-library", port=0) as server:
            async with connect(server.url) as ws:
                await ws.send(encode(Hello((1,), "test-library", (Op.ANIMATE,), 24000, "test")))
                welcome = decode(await ws.recv(), sender=Sender.COMPANION)
                assert welcome == Welcome(1, (Op.ANIMATE,), (("animate", "reachy-motion-library"),))
                await ws.send(encode(Animate(1, "a cheerful greeting", 2500)))
                result = decode(await ws.recv(), sender=Sender.COMPANION)
                assert result == AnimateResult(1, 25, ((1., -2., 3., 4., 5., 6., 8., 9., 7.),) * 62)
                await ws.send(encode(Animate(2, "not in the catalogue", 2500)))
                assert decode(await ws.recv(), sender=Sender.COMPANION) == AnimateResult(2, 25, ())

    asyncio.run(scenario())
