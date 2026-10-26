"""Compact motion reports without loading the motion models.

    python bench/motion_report.py bench/motion_runs/*.json

Keep the recipes, timings and trajectory statistics used by motion_table.py.
The conversion is safe to repeat on reports that already contain only summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LIMITS = {"x": (-15, 15), "y": (-15, 15), "z": (-10, 20), "roll": (-20, 20), "pitch": (-20, 20), "yaw": (-35, 35),
          "body_yaw": (-35, 35), "right_antenna": (-90, 90), "left_antenna": (-90, 90)}


def model_name(value: str | Path | None) -> str | None:
    """Use a local directory's basename, or keep a Hugging Face repo ID."""
    if value is None:
        return None
    path = Path(value)
    if isinstance(value, Path) or path.is_absolute() or str(value).startswith((".", "~")) or len(path.parts) > 2 or path.exists():
        return path.name
    return value


def swings(series, travel):
    """Moves of at least `travel` between turning points. Two full nods are 4."""
    count, direction, extreme = 0, 0, series[0]
    for value in series:
        if direction == 0 and abs(value - extreme) >= travel:
            count, direction, extreme = 1, 1 if value > extreme else -1, value
        elif direction and (value - extreme) * direction > 0:
            extreme = value
        elif direction and (extreme - value) * direction >= travel:
            count, direction, extreme = count + 1, -direction, value
    return count


def compact_report(report: dict) -> None:
    """Replace raw trajectories in place with the statistics the tables need."""
    for key in ("bundle", "mlx_model", "contend"):
        report["config"][key] = model_name(report["config"][key])
    for prompt, trajectory in report.pop("trajectories", {}).items():
        frames, channels = trajectory["frames"], trajectory["channels"]
        at_limit = {}
        for row in frames:
            for channel, value in zip(channels, row):
                if value in LIMITS[channel]:
                    at_limit[channel] = at_limit.get(channel, 0) + 1
        report["quality"][prompt].update({
            "frames_at_limit": at_limit,
            "pitch_swings": swings([row[channels.index("pitch")] for row in frames], 5.0),
            "yaw_swings": swings([row[channels.index("yaw")] for row in frames], 8.0),
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.reports:
        report = json.loads(path.read_text())
        compact_report(report)
        path.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
