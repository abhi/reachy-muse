"""Offline, resumable motion generation and model-free animation lookup."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict, dataclass
import fcntl
import hashlib
from importlib import metadata
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence

from reachy_muse.companion_server.engines import Animation
from reachy_muse.companion_server.pair import default_state_dir
from reachy_muse.contract.commands import POSE_LIMITS
from reachy_muse.contract.expressions import Expression
from reachy_muse.motion.moments import MOMENTS
from reachy_muse.protocol import ANIMATION_CHANNELS, MAX_ANIMATION_FRAMES

log = logging.getLogger(__name__)
SCHEMA = 1
FPS = 25
DEFAULT_LIBRARY_DIR = default_state_dir() / "motion-library"


@dataclass(frozen=True)
class IntentSpec:
    prompts: tuple[str, ...]
    duration_s: float = 2.5


_EXPRESSION_PROMPTS = {
    Expression.NEUTRAL: ("gentle neutral emphasis", "a calm attentive head movement", "a small relaxed conversational gesture"),
    Expression.HAPPY: ("a gentle happy expression", "a cheerful head tilt", "quietly delighted and smiling"),
    Expression.SAD: ("a gentle sad expression", "lowering the head with sympathy", "a subdued sympathetic gesture"),
    Expression.SURPRISED: ("a gentle surprised expression", "a small astonished head lift", "perking up in mild surprise"),
    Expression.CURIOUS: ("a gentle curious expression", "tilting the head inquisitively", "leaning in with curiosity"),
    Expression.NOD: ("a gentle nod of agreement", "nodding yes", "a small reassuring nod"),
    Expression.SHAKE: ("a gentle head shake", "shaking the head no", "a small hesitant head shake"),
    Expression.LISTENING: ("listening attentively", "a quiet attentive head tilt", "leaning slightly forward to listen"),
    Expression.THINKING: ("a thoughtful pause", "pondering quietly"),
}
INTENTS = {expression.value: IntentSpec(prompts) for expression, prompts in _EXPRESSION_PROMPTS.items()}
INTENTS.update({moment.value: IntentSpec(spec.intents, spec.duration_s) for moment, spec in MOMENTS.items()})
INTENTS["wake"] = IntentSpec(("waking up with a gentle stretch", "perking up after a quiet rest"))


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_identity(motion_dir: Path) -> dict:
    """Hash local weights and generation code without importing or loading models.

    Missing converted planner weights fail here, before MotionEngine can download.
    """
    from reachy_muse.companion_server.motion import DEFAULT_BITS, DEFAULT_REPO, DEFAULT_REVISION

    name = f"{DEFAULT_REPO.replace('/', '--')}@{DEFAULT_REVISION[:12]}"
    motion_dir = Path(motion_dir)
    generator = motion_dir / name / "generator.pt"
    planner = motion_dir / f"{name}-mlx-{DEFAULT_BITS}bit"
    if not generator.is_file() or not (planner / "config.json").is_file() or not list(planner.glob("*.safetensors")):
        raise FileNotFoundError("offline motion build requires local generator.pt and a converted MLX planner")
    files = [generator, *(p for p in planner.rglob("*") if p.is_file() and ".cache" not in p.parts)]
    result = {"weights": {str(p.relative_to(motion_dir)): _file_hash(p) for p in sorted(files)},
              "pipeline": _file_hash(Path(__file__).with_name("motion.py")), "packages": {}}
    for name in ("reachy-motion-generator", "reachy-mini-rust-kinematics", "torch", "numpy", "scipy", "mlx", "mlx-lm"):
        try:
            package = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            result["packages"][name] = {"version": "not-installed"}
            continue
        code = {str(p): _file_hash(Path(package.locate_file(p))) for p in package.files or ()
                if str(p).endswith((".py", "direct_url.json")) and Path(package.locate_file(p)).is_file()}
        result["packages"][name] = {"version": package.version, "code_sha256": _digest(_json(code))}
    return result


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _manifest(root: Path) -> dict:
    try:
        value = json.loads((root / "manifest.json").read_bytes())
        if value["schema"] == SCHEMA and isinstance(value["intents"], dict):
            return value
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {"schema": SCHEMA, "intents": {}}


def _frames(value: Any) -> tuple:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= MAX_ANIMATION_FRAMES:
        raise ValueError("motion must contain 1..375 frames")
    frames = []
    for frame in value:
        if not isinstance(frame, (list, tuple)) or len(frame) != len(ANIMATION_CHANNELS):
            raise ValueError("motion frame must have nine channels")
        row = []
        for channel, item in zip(ANIMATION_CHANNELS, frame):
            if isinstance(item, bool) or not isinstance(item, (float, int)) or not math.isfinite(item):
                raise ValueError("motion frame contains a nonfinite or nonnumeric value")
            low, high = POSE_LIMITS[channel]
            row.append(round(max(low, min(high, item)), 1))
        frames.append(tuple(row))
    return tuple(frames)


def _read_variant(root: Path, entry: Mapping) -> Animation:
    path = Path(entry["path"])
    if path.is_absolute() or ".." in path.parts or not (root / path).resolve().is_relative_to(root.resolve()):
        raise ValueError("motion path must stay inside the library")
    payload = (root / path).read_bytes()
    if _digest(payload) != entry["sha256"]:
        raise ValueError("motion artifact hash mismatch")
    value = json.loads(payload)
    if value["schema"] != SCHEMA or value["channels"] != list(ANIMATION_CHANNELS) or value["fps"] != FPS:
        raise ValueError("motion artifact schema, channel order or FPS mismatch")
    return Animation(FPS, _frames(value["frames"]))


def build_library(library_dir: Path, engine: Any, model_identity: Mapping, *, variants: int = 12,
                  only: Optional[Sequence[str]] = None) -> int:
    """Generate missing variants, committing each independently. Return the number built.

    The supplied engine is lazy. A complete build never calls it. The caller owns
    engine shutdown and runs this operation in a killable process.
    """
    if isinstance(variants, bool) or not isinstance(variants, int) or variants < 1:
        raise ValueError("variants must be a positive integer")
    keys = tuple(dict.fromkeys(only if only is not None else INTENTS))
    unknown = set(keys) - INTENTS.keys()
    if unknown:
        raise ValueError(f"unknown motion intents: {', '.join(sorted(unknown))}")
    root = Path(library_dir)
    root.mkdir(parents=True, exist_ok=True)
    built = 0
    with (root / ".build.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another motion library build is already running") from None
        manifest = _manifest(root)
        for key in keys:
            spec = INTENTS[key]
            identity = {"model": dict(model_identity), "intent": asdict(spec),
                        "builder": _file_hash(Path(__file__)), "schema": SCHEMA}
            fingerprint = _digest(_json(identity))
            entry = manifest["intents"].get(key, {})
            if not isinstance(entry, dict) or entry.get("fingerprint") != fingerprint or not isinstance(entry.get("variants"), dict):
                entry = {"fingerprint": fingerprint, "identity": identity, "variants": {}}
                manifest["intents"][key] = entry
            for index in range(variants):
                existing = entry["variants"].get(str(index))
                if existing is not None:
                    try:
                        _read_variant(root, existing)
                        continue
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                prompt = spec.prompts[index % len(spec.prompts)]
                duration = spec.duration_s * (0.9, 1.0, 1.1)[index % 3]
                seed = int(_digest(f"{key}:{index}".encode())[:8], 16)
                trajectory = engine.generate(prompt, duration, seed=seed)
                from reachy_muse.companion_server.motion import CHANNELS

                if trajectory.fps != FPS:
                    raise ValueError("builder requires a 25 FPS trajectory")
                raw = trajectory.frames.tolist()
                frames = _frames([[row[CHANNELS.index(channel)] for channel in ANIMATION_CHANNELS] for row in raw])
                if not 0.5 <= len(frames) / FPS / duration <= 2:
                    raise ValueError("generated motion duration is outside the safe retiming range")
                payload = _json({"schema": SCHEMA, "intent": key, "prompt": prompt, "seed": seed,
                                 "duration_hint_s": duration, "recipe": trajectory.recipe,
                                 "fps": FPS, "channels": ANIMATION_CHANNELS, "frames": frames})
                digest = _digest(payload)
                path = Path(key) / f"{index}-{digest}.json"
                _atomic(root / path, payload)
                entry["variants"][str(index)] = {"path": str(path), "sha256": digest}
                _atomic(root / "manifest.json", _json(manifest))
                built += 1
    return built


class LibraryAnimator:
    """An immutable validated snapshot; animate does no file or model I/O."""

    name = "reachy-motion-library"

    def __init__(self, library_dir: Path = DEFAULT_LIBRARY_DIR) -> None:
        self.library_dir = Path(library_dir)
        self._animations: dict[str, tuple[tuple[str, Animation], ...]] = {}
        self._recent: dict[str, deque[str]] = {}
        self._next: dict[str, int] = {}
        self._aliases = {prompt: key for key, spec in INTENTS.items() for prompt in spec.prompts}

    async def load(self) -> None:
        self._animations = await asyncio.to_thread(self._snapshot)
        self._recent.clear()
        self._next.clear()

    def _snapshot(self) -> dict[str, tuple[tuple[str, Animation], ...]]:
        result = {}
        for key, entry in _manifest(self.library_dir)["intents"].items():
            if key not in INTENTS or not isinstance(entry, dict) or not isinstance(entry.get("variants"), dict):
                continue
            variants = []
            for variant_id, item in entry["variants"].items():
                try:
                    variants.append((variant_id, _read_variant(self.library_dir, item)))
                except (OSError, ValueError, KeyError, TypeError):
                    log.warning("ignoring invalid motion library variant for %s", key)
            if variants:
                result[key] = tuple(variants)
        return result

    async def animate(self, intent: str, duration_s: float) -> Animation:
        empty = Animation(FPS, ())
        if not math.isfinite(duration_s) or not 0.5 <= duration_s <= 15:
            return empty
        key = self._aliases.get(intent, intent)
        choices = self._animations.get(key, ())
        cursor = self._next.get(key, 0)
        ordered = choices[cursor:] + choices[:cursor]
        eligible = {variant_id: animation for variant_id, animation in ordered
                    if 0.5 <= duration_s / (len(animation.frames) / FPS) <= 2}
        if not eligible:
            return empty
        recent = self._recent.setdefault(key, deque(maxlen=3))
        available = [index for index in eligible if index not in recent]
        while not available:
            recent.popleft()
            available = [index for index in eligible if index not in recent]
        index = available[0]
        recent.append(index)
        position = next(i for i, (variant_id, _) in enumerate(choices) if variant_id == index)
        self._next[key] = (position + 1) % len(choices)
        animation = eligible[index]
        log.debug("motion library intent=%s variant=%s", key, index)
        count = min(MAX_ANIMATION_FRAMES, round(duration_s * FPS))
        source = animation.frames
        frames = []
        for i in range(count):
            at = i * (len(source) - 1) / max(1, count - 1)
            lower = int(at)
            upper = min(lower + 1, len(source) - 1)
            weight = at - lower
            frames.append(tuple(round(a + (b - a) * weight, 1) for a, b in zip(source[lower], source[upper])))
        return Animation(FPS, tuple(frames))
