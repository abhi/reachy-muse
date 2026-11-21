import ast
import asyncio
import contextlib
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from reachy_muse.companion_server import cli, hearing_voxtral, narrator_mlx
from reachy_muse.device_command import DeviceCommand


def _installed_wired_limit(mx):
    """Execute the installed dependency's limit context with no MLX imports or GPU work."""
    spec = importlib.util.find_spec("mlx_lm")
    if spec is None:
        pytest.skip("mlx-lm is not installed")
    package = Path(spec.origin).parent
    namespace = dict(mx=mx, contextlib=contextlib, tree_reduce=lambda f, model, initial: initial)
    for filename, name in (("utils.py", "maybe_set_recommended_wired_limit"), ("generate.py", "wired_limit")):
        source = ast.parse((package / filename).read_text())
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == name)
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                 function], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    return namespace["wired_limit"]


@pytest.mark.parametrize("motion", [False, True])
def test_limits_are_set_after_all_loads_and_survive_narrator_toggles(monkeypatch, caplog, motion):
    calls = []
    limits = dict(wired=0, cache=0)

    def limit(kind, size):
        old, limits[kind] = limits[kind], size
        calls.append((kind, size))
        return old

    mx = SimpleNamespace(
        device_info=lambda: {"max_recommended_working_set_size": 8 * 1024**3},
        set_wired_limit=lambda size: limit("wired", size),
        set_cache_limit=lambda size: limit("cache", size),
        clear_cache=lambda: calls.append("clear cache"),
        synchronize=lambda *args: None,
    )
    wired_limit = _installed_wired_limit(mx)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    class Engine:
        name = "fake"

        def __init__(self, name):
            self.name = name

        async def load(self):
            calls.append(self.name)

    monkeypatch.setattr(hearing_voxtral, "VoxtralHearing", lambda *args: Engine("hearing"))
    narrator_type = narrator_mlx.MlxNarrator
    monkeypatch.setattr(narrator_mlx, "MlxNarrator", lambda *args: Engine("narrator"))
    monkeypatch.setattr(cli, "build_voice", lambda *args: Engine("voice"))
    monkeypatch.setitem(sys.modules, "reachy_muse.companion_server.motion", SimpleNamespace(
        DEFAULT_DIR="fake", MotionEngine=lambda *args, **kwargs: Engine("motion")))
    args = SimpleNamespace(engines="real", motion=motion, motion_dir=None, delay_ms=480, final_pass=True,
                           voice="kokoro", loudness=-20, narrator_model="fake")
    caplog.set_level(logging.INFO)
    asyncio.run(cli._engines(args))
    assert calls == ["hearing", "voice", "narrator", *(["motion"] if motion else []), "clear cache",
                     ("wired", 8589934592), ("cache", 268435456)]
    assert caplog.text.count("MLX memory limits: wired=8589934592 cache=268435456 bytes") == 1
    calls.clear()

    def generate(*args, **kwargs):
        with wired_limit(None):
            assert limits == {"wired": 8589934592, "cache": 268435456}
            yield SimpleNamespace(text="NONE")

    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(stream_generate=generate))
    narrator = narrator_type()
    narrator._tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1], apply_chat_template=lambda *a, **kw: "")
    try:
        assert narrator._generate([], 1, "") == "NONE"
        assert narrator.classify_now("private input", tuple(DeviceCommand)) is None
    finally:
        narrator._thread.shutdown()
    assert calls == [("wired", 8589934592)] * 4
    assert limits == {"wired": 8589934592, "cache": 268435456}
    assert "private input" not in caplog.text
