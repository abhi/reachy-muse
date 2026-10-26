"""Print the bench/motion.md tables from bench/motion_runs/*.json.

    python bench/motion_table.py
"""

import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "motion_runs"
NOD = ("nodding yes", "nodding yes. You agree with what was just said.")
SHAKE = ("shaking head no", "shaking your head no. You disagree.")


def at_limit(report):
    """Frames sitting exactly on a POSE_LIMITS bound, per channel: what the clamp cut off."""
    out = {}
    for quality in report["quality"].values():
        for channel, count in quality["frames_at_limit"].items():
            out[channel] = out.get(channel, 0) + count
    return out


def main() -> None:
    reports = {path.stem: json.loads(path.read_text()) for path in sorted(RUNS.glob("*.json"))}
    print("| config | n | planner p50 / p95 | generator p50 / p95 | reach p50 / p95 | total p50 / p95 | tokens p50 | motion s p50 | peak GB | load s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, r in reports.items():
        lat = r["latency_ms"]
        cell = lambda k: f"{lat[k]['p50']:.0f} / {lat[k]['p95']:.0f}"
        print(f"| {name} | {len(r['runs'])} | {cell('planner')} | {cell('generator')} | {cell('reach')} | {cell('total')} "
              f"| {r['recipe_tokens']['p50']:.0f} | {r['duration_s']['p50']:.1f} | {r['peak_footprint_gb']:.1f} | {r['load_s']:.0f} |")
    print()
    print("| config | probes (16, greedy) | OOD core | skill | frames outside POSE_LIMITS | unreachable | clamped frames | nod pitch swings | shake yaw swings |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, r in reports.items():
        q = r["quality"].values()
        probes = r["probes"]
        quality = r["quality"]
        nod = next((quality[p]["pitch_swings"] for p in NOD if p in quality), "-")
        shake = next((quality[p]["yaw_swings"] for p in SHAKE if p in quality), "-")
        p = (f"{probes['overall']:.2f}", f"{probes['ood_core']:.2f}", f"{probes['skill']:.2f}") if probes else ("-",) * 3
        print(f"| {name} | {p[0]} | {p[1]} | {p[2]} | {sum(c['outside_limits'] for c in q)} | {sum(c['unreachable_after_clamp'] for c in q)} "
              f"| {sum(c.get('clamped_frames', 0) for c in q)} / {sum(c['frames'] for c in q)} | {nod} | {shake} |")
    print()
    for name, r in reports.items():
        print(f"{name}: frames on a limit {at_limit(r)}")
    print()
    for name, r in reports.items():
        before, after = r["live_server"]["before"], r["live_server"]["after"]
        print(f"{name}: live server cpu {before.get('cpu_percent')} -> {after.get('cpu_percent')}%, footprint {before.get('footprint')} -> {after.get('footprint')}")


if __name__ == "__main__":
    main()
