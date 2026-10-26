"""Measure text-to-motion on this Mac: planner, generator and reach projection, per configuration.

    python bench/motion.py --bundle DIR --planner hf --device mps --out runs/hf-mps-0.8b.json
    python bench/motion.py --bundle DIR --planner mlx --mlx-model DIR --device mps --out runs/mlx-4bit-0.8b.json
    python bench/motion.py ... --probes                # also score upstream's 16 behavioral probes (greedy)
    python bench/motion.py ... --contend MLX_REPO      # decode with another MLX model in a second process meanwhile

--bundle is a reachy-motion-generator bundle (planner/, generator.pt), such as a local download of
binhpham/reachy-mini-motion-planner-0.8b. --device places the generator, and the planner too for --planner hf.
It needs the companion-motion extra. Every prompt runs --runs times after one warm-up request. The planner
decodes greedily, as upstream serves it, so repeats measure timing, not new recipes. The JSON keeps every
run, the recipes, trajectory statistics, the quality checks and peak memory. Use --raw-out bench/out/raw.json
to save raw trajectories separately in the ignored output directory.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from motion_report import compact_report, model_name
from planner.distill.probes import _osc_cycles as oscillations
from reachy_muse.companion_server.motion import CHANNELS, MlxPlanner, Pipeline
from reachy_muse.contract.commands import POSE_LIMITS

PROMPTS = (
    "happy, answering a question",
    "curious, listening",
    "thinking hard",
    "surprised by good news",
    "sad, apologizing",
    "nodding yes",
    "shaking head no",
    "excited greeting",
)
# The same intents in the shape the planners were trained on: "word. one sentence of context."
FORMATTED = (
    "happy. You are answering a question cheerfully.",
    "curious. You are listening closely to someone.",
    "thinking hard. You are working on a difficult problem.",
    "surprised. You just heard good news.",
    "sad. You are apologizing for a mistake.",
    "nodding yes. You agree with what was just said.",
    "shaking your head no. You disagree.",
    "excited. You greet someone you are happy to see.",
)
OSCILLATION_CHECKS = {"nodding yes": ("pitch", 5.0), "shaking head no": ("yaw", 8.0),
                      FORMATTED[5]: ("pitch", 5.0), FORMATTED[6]: ("yaw", 8.0)}
CONTENDER = """
import sys, time
from mlx_lm import load, generate
model, tok = load(sys.argv[1])
prompt = tok.apply_chat_template([{"role": "user", "content": "Tell me a long story about a robot."}], add_generation_prompt=True, tokenize=False)
print("ready", flush=True)
while True:
    generate(model, tok, prompt, max_tokens=64)
"""


def outside_limits(frames: np.ndarray) -> int:
    """Frames with any channel outside POSE_LIMITS."""
    low = np.array([POSE_LIMITS[c][0] for c in CHANNELS])
    high = np.array([POSE_LIMITS[c][1] for c in CHANNELS])
    return int(np.any((frames < low) | (frames > high), axis=1).sum())


def peak_footprint_bytes() -> int:
    """The process's lifetime peak physical footprint, which on Apple Silicon includes Metal allocations."""
    buffer = (ctypes.c_uint64 * 64)()
    libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
    if libc.proc_pid_rusage(ctypes.c_int(__import__("os").getpid()), 4, ctypes.byref(buffer)) != 0:
        return -1
    return int(buffer[2 + 28])  # 16-byte uuid, then ri_lifetime_max_phys_footprint is field 28 of rusage_info_v4


def live_server() -> dict:
    """CPU and footprint of the launchd companion, read only."""
    pid = subprocess.run(["pgrep", "-f", "reachy-companion serve"], capture_output=True, text=True).stdout.split()
    if not pid:
        return {"running": False}
    cpu = subprocess.run(["ps", "-o", "%cpu=", "-p", pid[0]], capture_output=True, text=True).stdout.strip()
    footprint = subprocess.run(["footprint", pid[0]], capture_output=True, text=True).stdout
    line = next((l for l in footprint.splitlines() if "Footprint:" in l), "")
    return {"running": True, "pid": int(pid[0]), "cpu_percent": float(cpu or 0), "footprint": line.split("Footprint:")[-1].strip()}


def percentiles(values: list) -> dict:
    ordered = sorted(values)
    return {"p50": round(statistics.median(ordered), 1),
            "p95": round(ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))], 1),
            "max": round(ordered[-1], 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--planner", choices=("hf", "mlx"), required=True)
    parser.add_argument("--mlx-model", type=Path, help="an mlx_lm.convert output of the bundle's planner")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--formatted", action="store_true", help="use the trained prompt shape for the 8 intents")
    parser.add_argument("--limit", type=int, default=8, help="time only the first N intents")
    parser.add_argument("--probes", action="store_true")
    parser.add_argument("--contend", help="an MLX model to decode in a second process during the timed runs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-out", type=Path, help="also save raw trajectories locally (use ignored bench/out/)")
    args = parser.parse_args()

    started = time.perf_counter()
    baseline = peak_footprint_bytes()
    planner = MlxPlanner(args.mlx_model) if args.planner == "mlx" else None
    pipeline = Pipeline(args.bundle, device=args.device, planner=planner)
    load_s = time.perf_counter() - started
    pipeline.run("warm-up. You stretch after waking.")
    server_before = live_server()

    contender = None
    if args.contend:
        contender = subprocess.Popen([sys.executable, "-c", CONTENDER, args.contend], stdout=subprocess.PIPE, text=True)
        contender.stdout.readline()
    try:
        runs, examples = [], {}
        for repeat in range(args.runs):
            for prompt in (FORMATTED if args.formatted else PROMPTS)[:args.limit]:
                result = pipeline.run(prompt)
                runs.append({"prompt": prompt, "repeat": repeat, "idea": result.idea, "recipe": result.recipe,
                             "recipe_tokens": result.recipe_tokens, "attempts": result.attempts,
                             "duration_s": round(len(result.frames) / 25, 2), **result.timing_ms})
                if repeat == 0:
                    examples[prompt] = result
    finally:
        if contender is not None:
            contender.kill()
            contender.wait()
    server_after = live_server()

    quality = {}
    for prompt, result in examples.items():
        check = {"frames": len(result.frames), "outside_limits": outside_limits(result.frames),
                 "unreachable_after_clamp": result.unreachable, "clamped_frames": result.clamped_frames, "projected_fraction": round(result.projected_fraction, 3),
                 "range_deg_or_mm": {c: round(float(np.ptp(result.frames[:, i])), 1) for i, c in enumerate(CHANNELS)}}
        if prompt in OSCILLATION_CHECKS:
            channel, amplitude = OSCILLATION_CHECKS[prompt]
            other = "yaw" if channel == "pitch" else "pitch"
            check["oscillation"] = {
                "channel": channel,
                "half_cycles": oscillations(result.frames[:, CHANNELS.index(channel)], amplitude),
                f"{other}_half_cycles": oscillations(result.frames[:, CHANNELS.index(other)], amplitude),
                "min_amplitude_deg": amplitude}
        quality[prompt] = check

    probes = None
    if args.probes:
        from planner.distill.probes import PROBES, score, split
        recipes = {}
        for prompt, _, _ in PROBES:
            try:
                recipes[prompt] = [pipeline.plan(prompt)[1]]
            except ValueError:
                recipes[prompt] = [None]
        overall, per = score(recipes)
        core, skill = split(per)
        probes = {"overall": overall, "ood_core": core, "skill": skill, "per_prompt": per, "recipes": recipes}

    keys = ("planner", "generator", "reach", "total")
    report = {
        "config": {"planner": args.planner, "mlx_model": model_name(args.mlx_model),
                   "bundle": model_name(args.bundle), "device": args.device, "runs": args.runs, "formatted": args.formatted, "contend": model_name(args.contend),
                   "machine": platform.platform(), "python": platform.python_version()},
        "load_s": round(load_s, 1),
        "latency_ms": {k: percentiles([r[k] for r in runs]) for k in keys},
        "recipe_tokens": percentiles([r["recipe_tokens"] for r in runs]),
        "duration_s": percentiles([r["duration_s"] for r in runs]),
        "peak_footprint_gb": round(peak_footprint_bytes() / 1e9, 2),
        "baseline_footprint_gb": round(baseline / 1e9, 2),
        "live_server": {"before": server_before, "after": server_after},
        "quality": quality,
        "probes": probes,
        "runs": runs,
        "trajectories": {p: {"channels": CHANNELS, "fps": 25, "frames": np.round(r.frames, 2).tolist()} for p, r in examples.items()},
    }
    if args.raw_out:
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        args.raw_out.write_text(json.dumps(report, indent=1) + "\n")
    compact_report(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("config", "load_s", "latency_ms", "recipe_tokens", "duration_s", "peak_footprint_gb")}, indent=1))
    for prompt, check in quality.items():
        print(f"{prompt:32s} {check['frames']:4d} frames  outside={check['outside_limits']}  unreachable={check['unreachable_after_clamp']}"
              + (f"  {check['oscillation']}" if "oscillation" in check else ""))
    if probes:
        print(f"probes: overall {probes['overall']:.2f}  ood_core {probes['ood_core']:.2f}  skill {probes['skill']:.2f}")


if __name__ == "__main__":
    main()
