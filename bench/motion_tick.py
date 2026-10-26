"""Time one motion-loop tick, before and after the motion layers, with the SDK's own pose math.

    ~/muse-reachy-venv/bin/python bench/motion_tick.py [--emotions DIR]

Run it on the robot to see the cost on its CPU; no motor or daemon is used. Each row
is CPU time per tick, averaged over 4000 ticks, and its share of the 25 ms tick period.
The clip row plays success1, read from DIR (the emotion library's JSON files) or,
without DIR, from the cached library that RecordedMoves loads.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from reachy_mini.motion.recorded_move import RecordedMove
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import linear_pose_interpolation

from reachy_muse.hardware import EMOTIONS_DATASET, ReachyController, _Clip
from reachy_muse.motion import MotionStyle
from reachy_muse.motion.layers import ClipTimeline, MotionLayers

TICKS = 4000


class Robot:
    def set_target(self, head, antennas, body_yaw):
        pass


def controller(style):
    c = ReachyController(pose_factory=create_head_pose, pose_interpolator=linear_pose_interpolation, style=style)
    c._np = np
    c._robot = Robot()
    c._frozen_antennas = np.deg2rad([-10, 10])
    c._last_pose, c._last_antennas, c._last_body_yaw = create_head_pose(), np.deg2rad([-10, 10]), 0.0
    c._rest = (c._last_pose, c._last_antennas, c._last_body_yaw)
    c._layers = MotionLayers(style)
    return c


def main_tick(c, now, state, expression):
    pose, antennas = c._state_pose(state, now % 3, expression)
    c._write(c._interpolate_pose(c._last_pose, pose, 0.16),
             c._last_antennas * 0.78 + antennas * 0.22, c._last_body_yaw * 0.92)


def layered_tick(c, now, state, expression):
    pose, antennas = c._state_pose(state, now % 3, expression)
    rest_pose, rest_antennas, rest_body_yaw = c._rest
    c._rest = (c._interpolate_pose(rest_pose, pose, 0.16), rest_antennas * 0.78 + antennas * 0.22,
               rest_body_yaw * 0.92)
    resting = c._pose_vector(*c._rest) + c._layers.breathing_offset(now, active=state in ("idle", "listening"))
    c._write_vector(c._show_clip(now, resting) + c._layers.sway_offset(now, active=state == "speaking"))


def time_ticks(tick, c, state, expression=None):
    t0 = time.thread_time_ns()
    for i in range(TICKS):
        tick(c, 1000.0 + i * 0.025, state, expression)
    return (time.thread_time_ns() - t0) / TICKS / 1000


def speech(seconds, rate=16000):
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)


def success1(emotions: Path | None) -> RecordedMove:
    if emotions is not None:
        return RecordedMove(json.loads((emotions / "success1.json").read_text()))
    from reachy_mini.motion.recorded_move import RecordedMoves
    return RecordedMoves(EMOTIONS_DATASET).get("success1")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emotions", type=Path, help="folder holding the emotion library's JSON files")
    move = success1(parser.parse_args().emotions)
    rows = []
    rows.append(("main: speaking, SDK wobbler on the daemon", time_ticks(main_tick, controller(MotionStyle(sway="sdk")), "speaking", "happy")))
    rows.append(("layered, REACHY_MUSE_SWAY=sdk, idle breathing", time_ticks(layered_tick, controller(MotionStyle(sway="sdk")), "idle")))
    c = controller(MotionStyle())
    c._layers.feed_speech(speech(TICKS * 0.025 + 5), 16000, 1000.0)
    rows.append(("layered: speaking with sway", time_ticks(layered_tick, c, "speaking", "happy")))
    c = controller(MotionStyle())
    c._layers.feed_speech(speech(TICKS * 0.025 + 5), 16000, 1000.0)

    def clip_tick(c, now, state, expression):
        if c._clip is None:
            c._clip = _Clip(move, ClipTimeline(now, move.duration), 0.6)
        layered_tick(c, now, state, expression)

    rows.append(("layered: speaking with sway and success1 looping", time_ticks(clip_tick, c, "speaking", "happy")))
    feed = MotionLayers(MotionStyle())
    chunk = speech(0.1)
    t0 = time.thread_time_ns()
    for i in range(1000):
        feed.feed_speech(chunk, 16000, i * 0.1)
    rows.append(("feed_speech, per 100 ms of 16 kHz audio", (time.thread_time_ns() - t0) / 1000 / 1000))
    for name, micros in rows:
        print(f"{name:55s} {micros:8.1f} us  {micros / 25000:6.2%} of a 25 ms tick")


if __name__ == "__main__":
    main()
