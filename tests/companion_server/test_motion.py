import os
import threading

import numpy as np
import pytest

from reachy_muse.contract.commands import POSE_LIMITS

from reachy_muse.companion_server.motion import CHANNELS, WARM_INTENT, MotionEngine, Result, Trajectory


class StubPipeline:
    def __init__(self):
        self.calls = []

    def run(self, intent, duration_hint_s=None):
        self.calls.append((intent, duration_hint_s, threading.current_thread().name))
        frames = np.zeros((int(25 * (duration_hint_s or 2.0)), len(CHANNELS)))
        return Result("", "hold 2", 9, 1, frames, 0.0, 0, 0, {"planner": 1.0, "generator": 1.0, "reach": 1.0, "total": 3.0})


def test_nothing_loads_until_the_first_request_which_runs_on_the_motion_thread(tmp_path):
    built = []
    pipeline = StubPipeline()
    engine = MotionEngine(tmp_path, build=lambda: built.append(1) or pipeline)
    assert built == []

    trajectory = engine.generate("nodding yes", 3.0)

    assert built == [1]
    assert [call[:2] for call in pipeline.calls] == [(WARM_INTENT, None), ("nodding yes", 3.0)]
    assert pipeline.calls[-1][2].startswith("motion")
    assert (trajectory.intent, trajectory.recipe, trajectory.frames.shape, trajectory.duration_s) == ("nodding yes", "hold 2", (75, 9), 3.0)
    engine.close()


def test_load_warms_once_and_later_requests_reuse_the_models(tmp_path):
    import asyncio

    built = []
    pipeline = StubPipeline()
    engine = MotionEngine(tmp_path, build=lambda: built.append(1) or pipeline)
    asyncio.run(engine.load())
    asyncio.run(engine.load())
    engine.generate("thinking hard")

    assert built == [1]
    assert [call[0] for call in pipeline.calls] == [WARM_INTENT, "thinking hard"]
    engine.close()


def test_a_request_with_no_valid_recipe_raises_to_the_caller(tmp_path):
    class Refusing(StubPipeline):
        def run(self, intent, duration_hint_s=None):
            if intent != WARM_INTENT:
                raise ValueError(f"the planner wrote no valid recipe for {intent!r}")
            return super().run(intent, duration_hint_s)

    engine = MotionEngine(tmp_path, build=Refusing)
    with pytest.raises(ValueError, match="no valid recipe for 'gibberish'"):
        engine.generate("gibberish")
    engine.close()


def test_animate_sends_frames_in_wire_order_rounded_to_tenths(tmp_path):
    import asyncio

    class Posed(StubPipeline):
        def run(self, intent, duration_hint_s=None):
            result = super().run(intent, duration_hint_s)
            frames = np.tile([1.04, -2.0, 8.0, 4.0, 6.0, -10.0, 12.0, 30.0, -40.0], (len(result.frames), 1))
            return Result(*[frames if name == "frames" else getattr(result, name) for name in Result.__dataclass_fields__])

    engine = MotionEngine(tmp_path, build=Posed)
    animation = asyncio.run(engine.animate("thinking hard", 0.2))
    engine.close()

    assert (animation.fps, len(animation.frames)) == (25, 5)
    assert animation.frames[0] == (1.0, -2.0, 8.0, 4.0, 6.0, -10.0, 30.0, -40.0, 12.0)


def test_animate_keeps_at_most_fifteen_seconds_of_frames(tmp_path):
    import asyncio

    engine = MotionEngine(tmp_path, build=StubPipeline)
    animation = asyncio.run(engine.animate("thinking hard", 20.0))
    engine.close()

    assert len(animation.frames) == 375


def test_prepared_models_load_without_the_network(tmp_path, monkeypatch):
    from reachy_muse.companion_server.motion import prepare

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", None)
    bundle = tmp_path / "binhpham--reachy-mini-motion-planner-4b@1b60d8f16b85"
    converted = tmp_path / "binhpham--reachy-mini-motion-planner-4b@1b60d8f16b85-mlx-8bit"
    bundle.mkdir()
    converted.mkdir()
    (bundle / "generator.pt").write_bytes(b"")
    (converted / "config.json").write_text("{}")

    assert prepare(tmp_path) == (bundle, converted)


def _upstream(rows):
    """Rows of x y z (mm), roll pitch yaw, antenna_right antenna_left, body_yaw (degrees) in upstream units."""
    rows = np.array(rows, dtype=float)
    return np.concatenate([rows[:, :3] / 1000, np.radians(rows[:, 3:])], axis=1)


def test_frames_come_back_in_pose_limits_order_and_units():
    pytest.importorskip("reachy_mini_rust_kinematics")
    from common.reach import Reach
    from reachy_muse.companion_server.motion import reachable

    upstream = _upstream([[1, -2, 8, 4, 6, -10, 30, -40, 12]] * 5)
    frames, projected, clamped, unreachable = reachable(upstream, Reach())

    assert CHANNELS == ("x", "y", "z", "roll", "pitch", "yaw", "body_yaw", "right_antenna", "left_antenna")
    assert np.round(frames[0], 3).tolist() == [1.0, -2.0, 8.0, 4.0, 6.0, -10.0, 12.0, 30.0, -40.0]
    assert (projected, clamped, unreachable) == (0.0, 0, 0)


def test_out_of_range_frames_are_clamped_into_pose_limits_and_stay_reachable():
    pytest.importorskip("reachy_mini_rust_kinematics")
    from common.reach import Reach
    from reachy_muse.companion_server.motion import reachable

    upstream = _upstream([[0, 0, 3, 0, 0, 0, 0, 0, 0], [0, 0, 30, 0, 35, 60, 150, -150, 80]])
    frames, _, clamped, unreachable = reachable(upstream, Reach())

    low, high = np.array([POSE_LIMITS[c] for c in CHANNELS]).T
    assert (clamped, unreachable) == (1, 0)
    assert np.all((frames >= low) & (frames <= high))
    assert np.round(frames[1, 7:], 3).tolist() == [90.0, -90.0]


def _swings(series, travel):
    count, direction, extreme = 0, 0, series[0]
    for value in series:
        if direction == 0 and abs(value - extreme) >= travel:
            count, direction, extreme = 1, 1 if value > extreme else -1, value
        elif direction and (value - extreme) * direction > 0:
            extreme = value
        elif direction and (extreme - value) * direction >= travel:
            count, direction, extreme = count + 1, -direction, value
    return count


@pytest.mark.real
def test_the_real_engine_nods_for_nodding_yes():
    engine = MotionEngine()
    trajectory = engine.generate("nodding yes", 4.0)
    engine.close()

    pitch = trajectory.frames[:, CHANNELS.index("pitch")]
    print(trajectory.recipe, round(trajectory.duration_s, 2))
    assert 3.0 <= trajectory.duration_s <= 5.5
    assert _swings(pitch, 5.0) >= 4, trajectory.recipe
    assert np.all(np.abs(pitch) <= 20.0)


@pytest.mark.parametrize("loaded", [False, True])
def test_serve_loads_motion_only_with_the_flag(monkeypatch, tmp_path, loaded):
    import argparse
    import asyncio

    pytest.importorskip("mlx_audio")
    from reachy_muse.companion_server import cli, hearing_voxtral, motion, narrator_mlx, voice_kokoro

    class Loads:
        name = "stub"

        def __init__(self, *args, **kwargs):
            pass

        async def load(self):
            pass

    pipeline = StubPipeline()
    for module, name in ((hearing_voxtral, "VoxtralHearing"), (voice_kokoro, "KokoroVoice"), (narrator_mlx, "MlxNarrator")):
        monkeypatch.setattr(module, name, Loads)
    monkeypatch.setattr(motion.MotionEngine, "_real_pipeline", lambda self: pipeline)
    args = argparse.Namespace(engines="real", delay_ms=480, final_pass=True, voice="kokoro:af_heart", loudness=-20.0,
                              narrator_model="stub", state_dir=tmp_path, motion=loaded, motion_dir=tmp_path)

    engines = asyncio.run(cli._engines(args))

    assert (engines.motion is not None) == loaded
    assert [call[0] for call in pipeline.calls] == ([WARM_INTENT] if loaded else [])
    if loaded:
        engines.motion.close()


@pytest.mark.parametrize("preset, expected", [(None, "16"), ("4", "4")])
def test_serve_caps_gpu_command_buffers_unless_the_operator_set_them(monkeypatch, tmp_path, preset, expected):
    import argparse
    import asyncio

    pytest.importorskip("mlx_audio")
    from reachy_muse.companion_server import cli, hearing_voxtral, narrator_mlx, voice_kokoro

    class Loads:
        name = "stub"

        def __init__(self, *args, **kwargs):
            pass

        async def load(self):
            pass

    for module, name in ((hearing_voxtral, "VoxtralHearing"), (voice_kokoro, "KokoroVoice"), (narrator_mlx, "MlxNarrator")):
        monkeypatch.setattr(module, name, Loads)
    if preset is None:
        monkeypatch.delenv("MLX_MAX_OPS_PER_BUFFER", raising=False)
    else:
        monkeypatch.setenv("MLX_MAX_OPS_PER_BUFFER", preset)
    args = argparse.Namespace(engines="real", delay_ms=480, final_pass=True, voice="kokoro:af_heart", loudness=-12.0,
                              narrator_model="stub", state_dir=tmp_path, motion=False, motion_dir=tmp_path)

    asyncio.run(cli._engines(args))

    assert os.environ["MLX_MAX_OPS_PER_BUFFER"] == expected
