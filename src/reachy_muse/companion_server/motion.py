"""Generated motion: an intent such as "nodding yes" becomes 25 Hz head, antenna and body frames.

It runs reachy-motion-generator (Apache-2.0): a fine-tuned Qwen3.5 planner writes a short recipe,
a 21.8M-parameter flow-matching generator turns the recipe into a 9-channel trajectory, and the
SDK's inverse kinematics pulls every frame back to a pose the head can reach. Here the planner runs
on MLX and the generator on torch MPS; bench/motion.md has the measurements behind that choice.

Every frame leaves inside POSE_LIMITS, in its units: millimeters for x, y, z and degrees for the rest.
One thread owns the models. Hearing, voice and the narrator never wait on it: with a SpeechFirst, the
planner pauses before each token while one of them is working.

The planner and generator weights live in a motion directory, by default under the user's
Application Support folder, outside any checkout. prepare() fills it on first use.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import numpy as np

from reachy_muse.companion_server.watchdog import EngineExecutor, watchdog
from reachy_muse.companion_server.engines import Animation, SpeechFirst
from reachy_muse.contract.commands import POSE_LIMITS
from reachy_muse.protocol import ANIMATION_CHANNELS, MAX_ANIMATION_FRAMES

log = logging.getLogger("reachy_muse.companion_server")

FPS = 25
CHANNELS: Tuple[str, ...] = tuple(POSE_LIMITS)
_LOW = np.array([POSE_LIMITS[c][0] for c in CHANNELS])
_HIGH = np.array([POSE_LIMITS[c][1] for c in CHANNELS])
# Upstream trajectories are x y z (m), roll pitch yaw (rad), antenna_right antenna_left (rad), body_yaw (rad).
_UPSTREAM = np.array([0, 1, 2, 3, 4, 5, 8, 6, 7])
_SCALE = np.array([1000.0] * 3 + [np.degrees(1.0)] * 6)
_WIRE = np.array([CHANNELS.index(channel) for channel in ANIMATION_CHANNELS])

DEFAULT_REPO = "binhpham/reachy-mini-motion-planner-4b"
DEFAULT_REVISION = "1b60d8f16b85473ceb20e3b62e464780c532b519"
DEFAULT_BITS = 8
WARM_INTENT = "warm-up. You stretch after waking."
# The longest the planner waits for speech to finish before each token.
YIELD_LIMIT_S = 2.0
DEFAULT_DIR = Path("~/Library/Application Support/reachy-companion/motion").expanduser()


@dataclass(frozen=True)
class Trajectory:
    """frames is (T, 9) in CHANNELS order, at FPS, already reachable and inside POSE_LIMITS."""

    intent: str
    recipe: str
    frames: np.ndarray
    fps: int = FPS

    @property
    def duration_s(self) -> float:
        return len(self.frames) / self.fps


@dataclass(frozen=True)
class Result:
    """One pipeline run, with what the bench and the logs need beyond the Trajectory."""

    idea: str
    recipe: str
    recipe_tokens: int
    attempts: int
    frames: np.ndarray
    projected_fraction: float
    clamped_frames: int
    unreachable: int
    timing_ms: Dict[str, float]


class Planner(Protocol):
    tok: Any

    def text(self, prompt: str) -> str: ...

    def complete(self, texts: list, temperature: float, seed: int) -> list: ...


class MlxPlanner:
    """A planner converted with mlx_lm.convert, answering with upstream's prompt and parser."""

    def __init__(self, path: Path, max_tokens: int = 600, yield_to: Optional[SpeechFirst] = None) -> None:
        from mlx_lm import load

        self.model, self.tok = load(str(path))
        self.max_tokens = max_tokens
        self.yield_to = yield_to

    def text(self, prompt: str) -> str:
        from planner.prompt import student_messages

        return self.tok.apply_chat_template(student_messages(prompt), tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def complete(self, texts: list, temperature: float, seed: int) -> list:
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(seed)
        sampler = make_sampler(temp=temperature, top_p=0.95 if temperature else 0.0)
        return [self._generate(text, sampler) for text in texts]

    def _generate(self, text: str, sampler: Any) -> str:
        from mlx_lm import stream_generate

        pieces = []
        for response in stream_generate(self.model, self.tok, text, max_tokens=self.max_tokens, sampler=sampler):
            watchdog.progress()
            pieces.append(response.text)
            if self.yield_to is not None:
                self.yield_to.wait_quiet(YIELD_LIMIT_S)
        return "".join(pieces)


class Pipeline:
    """Planner, generator and reach projection, run in turn on the caller's thread.

    planner defaults to upstream's transformers planner on ``device``; pass an MlxPlanner to plan on MLX.
    """

    def __init__(self, bundle: Path, *, device: str = "mps", planner: Optional[Planner] = None, retries: int = 2) -> None:
        from common.reach import Reach
        from generator.sample import load

        if planner is None:
            import generator.model
            from inference.engine import Planner as HfPlanner

            chosen = generator.model.device
            generator.model.device = lambda: device
            try:
                planner = HfPlanner(str(bundle), backend="hf")
            finally:
                generator.model.device = chosen
        self.planner = planner
        self.retries = retries
        self.net, self.stats, self.device = load(str(Path(bundle) / "generator.pt"), device)
        self.reach = Reach()

    def plan(self, intent: str, seed: int = 0) -> Tuple[str, str, str, int]:
        """Greedy first, then up to ``retries`` samples at temperature 0.7, as upstream serves it."""
        from inference.engine import Planner as HfPlanner

        text = self.planner.text(intent)
        for attempt in range(1 + self.retries):
            raw = self.planner.complete([text], 0.0 if attempt == 0 else 0.7, seed + attempt)[0]
            idea, recipe = HfPlanner.parse(raw)
            if recipe:
                return idea, recipe, raw, attempt + 1
        raise ValueError(f"the planner wrote no valid recipe for {intent!r}")

    def run(self, intent: str, duration_hint_s: Optional[float] = None, seed: int = 0) -> Result:
        from generator.sample import generate_batch
        from planner.dsl import expand, to_plan

        started = time.perf_counter()
        idea, recipe, raw, attempts = self.plan(intent, seed)
        natural_s = len(expand(recipe)) / FPS
        tempo = 1.0 if duration_hint_s is None else float(np.clip(duration_hint_s / natural_s, 0.5, 2.0))
        rng = np.random.default_rng(seed)
        plan = to_plan(expand(recipe, rng, tsc=tempo), rng, fc=2.0, kdt=0.25)
        planned = time.perf_counter()

        upstream = generate_batch(self.net, self.stats, [plan], self.device, seeds=[seed], steps=8, cfg=1.5)[0]
        generated = time.perf_counter()

        frames, projected_fraction, clamped, unreachable = reachable(upstream, self.reach)
        reached = time.perf_counter()

        timing = {"planner": (planned - started) * 1e3, "generator": (generated - planned) * 1e3,
                  "reach": (reached - generated) * 1e3, "total": (reached - started) * 1e3}
        return Result(idea, recipe, len(self.planner.tok.encode(raw)), attempts, frames, projected_fraction,
                      clamped, unreachable, {k: round(v, 1) for k, v in timing.items()})


def reachable(upstream: np.ndarray, reach: Any) -> Tuple[np.ndarray, float, int, int]:
    """An upstream (T, 9) trajectory -> (frames in CHANNELS order and units, fraction projected, frames clamped, frames unreachable).

    Clamps to POSE_LIMITS, projects onto the reachable set, and clamps again, because projection moves
    a frame toward the last reachable one along a rotation path that can leave the limits box.
    """
    from common.motion import rpy, to_move

    upstream = upstream.copy()
    frames = upstream[:, _UPSTREAM] * _SCALE
    clamped = np.clip(frames, _LOW, _HIGH)
    touched = np.any(clamped != frames, axis=1)
    upstream[:, _UPSTREAM] = clamped / _SCALE
    move, projected_fraction = reach.project(to_move(upstream))
    data = move["set_target_data"]
    heads = np.array([f["head"] for f in data])
    projected = np.concatenate([heads[:, :3, 3], rpy(heads[:, :3, :3]), np.array([[*f["antennas"], f["body_yaw"]] for f in data])], axis=1)
    frames = projected[:, _UPSTREAM] * _SCALE
    final = np.clip(frames, _LOW, _HIGH)
    touched |= np.any(final != frames, axis=1)
    unreachable = sum(not reach.ok(_head(row), np.radians(row[6])) for row in final)
    return final, projected_fraction, int(touched.sum()), unreachable


def _head(row: np.ndarray) -> np.ndarray:
    from common.motion import rpy_to_mat

    head = np.eye(4)
    head[:3, :3] = rpy_to_mat(np.radians(row[None, 3:6]))[0]
    head[:3, 3] = row[:3] / 1000
    return head


def prepare(motion_dir: Path, repo: str = DEFAULT_REPO, revision: str = DEFAULT_REVISION, bits: int = DEFAULT_BITS) -> Tuple[Path, Path]:
    """Download the bundle and convert its planner to MLX, once, under motion_dir. Returns (bundle, mlx planner).

    At run time only bundle/generator.pt and the converted planner are read, so once both exist the
    full-precision planner download can be deleted and nothing touches the network.
    """
    name = f"{repo.replace('/', '--')}@{revision[:12]}"
    bundle = motion_dir / name
    converted = motion_dir / f"{name}-mlx-{bits}bit"
    if (bundle / "generator.pt").exists() and (converted / "config.json").exists():
        return bundle, converted
    from huggingface_hub import snapshot_download

    snapshot_download(repo, revision=revision, local_dir=bundle, allow_patterns=["generator.pt", "planner/*"],
                      ignore_patterns=["planner/model-mtp.safetensors"])
    if not (converted / "config.json").exists():
        import shutil

        from mlx_lm import convert

        partial = converted.with_name(converted.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        convert(str(bundle / "planner"), mlx_path=str(partial), quantize=bits < 16, q_bits=bits, q_group_size=64, dtype="bfloat16")
        partial.rename(converted)
    return bundle, converted


class MotionEngine:
    """Turns an intent into a Trajectory on its own thread. Nothing loads until load() or the first request."""

    name = "reachy-motion"

    def __init__(self, motion_dir: Path = DEFAULT_DIR, *, build: Optional[Callable[[], Any]] = None,
                 yield_to: Optional[SpeechFirst] = None) -> None:
        self.motion_dir = motion_dir
        self.yield_to = yield_to
        self._build = build or self._real_pipeline
        self._thread = EngineExecutor("motion")
        self._pipeline: Any = None

    def _real_pipeline(self) -> Pipeline:
        bundle, planner = prepare(self.motion_dir)
        return Pipeline(bundle, device="mps", planner=MlxPlanner(planner, yield_to=self.yield_to))

    async def load(self) -> None:
        """Load and warm the models on the motion thread. serve awaits this at startup when --motion is on."""
        await asyncio.wrap_future(self._thread.submit(self._load))

    def _load(self) -> Any:
        if self._pipeline is None:
            started = time.perf_counter()
            self._pipeline = self._build()
            self._pipeline.run(WARM_INTENT)
            log.info("motion ready in %.1f s", time.perf_counter() - started)
        return self._pipeline

    def submit(self, intent: str, duration_hint_s: Optional[float] = None) -> "Future[Trajectory]":
        """Queue one request on the motion thread. From asyncio, await asyncio.wrap_future(engine.submit(...))."""
        return self._thread.submit(self._generate, intent, duration_hint_s)

    def generate(self, intent: str, duration_hint_s: Optional[float] = None) -> Trajectory:
        """Block until the motion thread returns this intent's Trajectory. Raises ValueError if no valid recipe."""
        return self.submit(intent, duration_hint_s).result()

    async def animate(self, intent: str, duration_s: float) -> Animation:
        trajectory = await asyncio.wrap_future(self.submit(intent, duration_s))
        frames = np.round(trajectory.frames[:MAX_ANIMATION_FRAMES, _WIRE], 1)
        return Animation(trajectory.fps, tuple(tuple(float(value) for value in frame) for frame in frames))

    def _generate(self, intent: str, duration_hint_s: Optional[float]) -> Trajectory:
        result = self._load().run(intent, duration_hint_s)
        log.info("motion %r: %.1f s of frames in %.0f ms (planner %.0f ms)", intent, len(result.frames) / FPS,
                 result.timing_ms["total"], result.timing_ms["planner"])
        return Trajectory(intent, result.recipe, result.frames)

    def close(self) -> None:
        self._thread.shutdown(wait=False, cancel_futures=True)
