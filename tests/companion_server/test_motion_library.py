import asyncio
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_muse.companion_server import motion_library as library
from reachy_muse.companion_server.motion import CHANNELS, DEFAULT_BITS, DEFAULT_REPO, DEFAULT_REVISION
from reachy_muse.contract.commands import POSE_LIMITS
from reachy_muse.protocol import ANIMATION_CHANNELS


class Engine:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def generate(self, prompt, duration, *, seed):
        if len(self.calls) == self.fail_at:
            raise RuntimeError("interrupted")
        self.calls.append((prompt, duration, seed))
        row = [float(i + len(self.calls)) for i in range(9)]
        return SimpleNamespace(recipe="hold", fps=25, frames=np.array([row] * round(25 * duration)))


def load(root):
    engine = library.LibraryAnimator(root)
    asyncio.run(engine.load())
    return engine


def animate(engine, intent="happy", duration=2.5):
    return asyncio.run(engine.animate(intent, duration))


def test_resume_each_completed_variant_without_loading_complete_engine(tmp_path):
    engine = Engine(fail_at=2)
    with pytest.raises(RuntimeError, match="interrupted"):
        library.build_library(tmp_path, engine, {"weights": "a"}, variants=4, only=["happy"])
    assert len(json.loads((tmp_path / "manifest.json").read_text())["intents"]["happy"]["variants"]) == 2
    resumed = Engine()
    assert library.build_library(tmp_path, resumed, {"weights": "a"}, variants=4, only=["happy"]) == 2
    assert len(resumed.calls) == 2
    assert library.build_library(tmp_path, Engine(fail_at=0), {"weights": "a"}, variants=4, only=["happy"]) == 0


def test_oversized_integer_poses_are_clamped_before_float_conversion(tmp_path):
    class IntegerEngine:
        def generate(self, prompt, duration, *, seed):
            return SimpleNamespace(recipe="hold", fps=25,
                                   frames=np.full((round(25 * duration), 9), 10 ** 1000, dtype=object))

    assert library.build_library(tmp_path, IntegerEngine(), {}, variants=1, only=["happy"]) == 1
    result = animate(load(tmp_path))
    assert result.frames[0] == (15, 15, 20, 20, 20, 35, 90, 90, 35)


def test_publication_interrupted_before_manifest_keeps_old_snapshot(tmp_path, monkeypatch):
    library.build_library(tmp_path, Engine(), {}, variants=1, only=["happy"])
    previous = (tmp_path / "manifest.json").read_bytes()
    original = library._atomic

    def fail_manifest(path, data):
        if path.name == "manifest.json":
            raise RuntimeError("killed before manifest")
        original(path, data)

    monkeypatch.setattr(library, "_atomic", fail_manifest)
    with pytest.raises(RuntimeError, match="before manifest"):
        library.build_library(tmp_path, Engine(), {}, variants=2, only=["happy"])
    assert (tmp_path / "manifest.json").read_bytes() == previous
    assert len(animate(load(tmp_path)).frames) == 62
    monkeypatch.setattr(library, "_atomic", original)
    assert library.build_library(tmp_path, Engine(), {}, variants=2, only=["happy"]) == 1


def test_invalidation_model_and_only_changed_intent(tmp_path, monkeypatch):
    assert library.build_library(tmp_path, Engine(), {"weights": "a"}, variants=2, only=["happy", "sad"]) == 4
    monkeypatch.setitem(library.INTENTS, "happy", replace(library.INTENTS["happy"], prompts=("new happy prompt",)))
    assert library.build_library(tmp_path, Engine(), {"weights": "a"}, variants=2, only=["happy", "sad"]) == 2
    assert library.build_library(tmp_path, Engine(), {"weights": "b"}, variants=2, only=["happy", "sad"]) == 4


def test_wire_order_variant_rotation_and_bounds(tmp_path):
    engine = Engine()
    library.build_library(tmp_path, engine, {}, variants=4, only=["happy"])
    assert len({call[2] for call in engine.calls}) == 4
    assert len({call[1] for call in engine.calls}) == 3
    animator = load(tmp_path)
    samples = [animate(animator).frames[0] for _ in range(5)]
    assert samples[0] == (1, 2, 3, 4, 5, 6, 8, 9, 7)
    assert len(set(samples[:4])) == 4
    assert samples[4] == samples[0]
    assert len(animate(animator).frames) == 62
    assert animate(animator, duration=15).frames == ()
    assert animate(animator, duration=float("nan")).frames == ()
    assert animate(animator, intent="unknown").frames == ()


def test_aliases_cover_existing_prompts(tmp_path):
    library.build_library(tmp_path, Engine(), {}, variants=1, only=["thinking"])
    animator = load(tmp_path)
    for prompt in library.INTENTS["thinking"].prompts:
        assert len(animate(animator, prompt, 4).frames) == 100


def test_missing_corrupt_and_clamped_artifacts(tmp_path):
    assert animate(load(tmp_path)).frames == ()
    library.build_library(tmp_path, Engine(), {}, variants=1, only=["happy"])
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["intents"]["happy"]["variants"]["0"]
    path = tmp_path / entry["path"]
    data = json.loads(path.read_text())
    data["frames"] = [[999999] * 9] * 50
    payload = json.dumps(data).encode()
    path.write_bytes(payload)
    assert animate(load(tmp_path)).frames == ()
    entry["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    assert animate(load(tmp_path)).frames[0] == tuple(POSE_LIMITS[c][1] for c in ANIMATION_CHANNELS)
    assert library.build_library(tmp_path, Engine(), {}, variants=1, only=["happy"]) == 0
    path.write_text("broken")
    assert library.build_library(tmp_path, Engine(), {}, variants=1, only=["happy"]) == 1


def test_snapshot_does_not_read_files_during_animate(tmp_path, monkeypatch):
    library.build_library(tmp_path, Engine(), {}, variants=1, only=["happy"])
    animator = load(tmp_path)
    monkeypatch.setattr(library.Path, "read_bytes", lambda self: pytest.fail("file read during animate"))
    assert animate(animator).frames


def test_model_identity_checks_weights_and_hashes_changes(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        library.model_identity(tmp_path)
    name = f"{DEFAULT_REPO.replace('/', '--')}@{DEFAULT_REVISION[:12]}"
    bundle = tmp_path / name
    planner = tmp_path / f"{name}-mlx-{DEFAULT_BITS}bit"
    bundle.mkdir()
    planner.mkdir()
    (bundle / "generator.pt").write_bytes(b"generator")
    (planner / "config.json").write_text("{}")
    (planner / "model.safetensors").write_bytes(b"model-a")
    def missing(name):
        raise library.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(library.metadata, "distribution", missing)
    first = library.model_identity(tmp_path)
    (planner / "model.safetensors").write_bytes(b"model-b")
    assert first != library.model_identity(tmp_path)
    second = library.model_identity(tmp_path)
    (bundle / "generator.pt").write_bytes(b"generator-b")
    assert second != library.model_identity(tmp_path)
    (planner / "tokenizer.json").write_text("{}")
    assert len(library.model_identity(tmp_path)["weights"]) == 4


@pytest.mark.parametrize("variants,only", [(0, None), (True, None), (1, ["bogus"])])
def test_bad_selection_does_not_call_engine(tmp_path, variants, only):
    with pytest.raises(ValueError):
        library.build_library(tmp_path, Engine(fail_at=0), {}, variants=variants, only=only)


def test_no_repeat_when_duration_changes_eligible_variants(tmp_path):
    library.build_library(tmp_path, Engine(), {}, variants=4, only=["happy"])
    animator = load(tmp_path)
    first = animate(animator, duration=1.2).frames[0]
    second = animate(animator, duration=2.5).frames[0]
    assert first != second


def test_seed_reaches_existing_motion_pipeline():
    from reachy_muse.companion_server.motion import MotionEngine, Result

    seeds = []

    class Pipeline:
        def run(self, intent, duration_hint_s=None, seed=0):
            seeds.append(seed)
            return Result("", "hold", 1, 1, np.zeros((25, 9)), 0, 0, 0, {"total": 1, "planner": 1})

    engine = MotionEngine(build=Pipeline)
    try:
        engine.generate("hello", 1, seed=123)
        engine.generate("hello", 1, seed=456)
        assert seeds == [0, 123, 456]
    finally:
        engine.close()


def test_rotation_visits_all_twelve_variants(tmp_path):
    library.build_library(tmp_path, Engine(), {}, variants=12, only=["happy"])
    animator = load(tmp_path)
    samples = [animate(animator).frames[0][0] for _ in range(13)]
    assert set(samples[:12]) == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    assert samples[12] == samples[0]
    assert samples[12] not in samples[9:12]
    short = animate(animator, duration=1.2).frames[0][0]
    normal = animate(animator).frames[0][0]
    assert short != normal


@pytest.mark.parametrize("malformed", [None, [], 5, {"variants": []}, {"variants": None}])
def test_builder_repairs_malformed_manifest_entries(tmp_path, malformed):
    library.build_library(tmp_path, Engine(), {}, variants=1, only=["happy"])
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    if isinstance(malformed, dict):
        malformed["fingerprint"] = manifest["intents"]["happy"]["fingerprint"]
    manifest["intents"]["happy"] = malformed
    path.write_text(json.dumps(manifest))
    assert library.build_library(tmp_path, Engine(), {}, variants=1, only=["happy"]) == 1
    assert animate(load(tmp_path)).frames


def test_dance_intents_use_existing_library_and_keep_expression_lookup(tmp_path):
    library.build_library(tmp_path, Engine(), {}, variants=1, only=["nod"])
    animator = load(tmp_path)
    dance = animate(animator, intent="dance_groove", duration=2.)
    speech = animate(animator, intent="nod", duration=2.5)
    assert len(dance.frames) == 50
    assert len(speech.frames) == 62
    assert dance.frames[0] == speech.frames[0]
