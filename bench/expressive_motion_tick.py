"""CPU-only motion-loop benchmark; no robot, services, network or model needed.

Run with PYTHONPATH=src python bench/expressive_motion_tick.py. A synthetic clock
runs the real controller loop at 40 Hz; results describe this machine only, not
the robot CPU. Robot I/O, real scheduling jitter, and audio feeding are excluded.
"""
import math
import time
from types import SimpleNamespace

import numpy as np

import reachy_muse.hardware as hardware
from reachy_muse.contract.expressions import Expression
from reachy_muse.motion import MotionStyle
from reachy_muse.motion.layers import AXES, MotionLayers, Ramp
from reachy_muse.motion.sentence import SentenceMotion

PERIOD = .025
TICKS = 2400


def pose_factory(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, degrees=True, mm=True):
    roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    pose = np.eye(4)
    pose[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                    [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
    pose[:3, 3] = np.array([x, y, z]) / 1000
    return pose


class Robot:
    def __init__(self):
        self.media = self
        self.audio = self
        self.pose = pose_factory()
        self.antennas = np.deg2rad([-10, 10])
        self.body_yaw = 0.0
        self.frames = []
        self.record = False

    def enable_motors(self): pass
    def disable_motors(self): pass
    def get_current_head_pose(self): return self.pose
    def get_current_joint_positions(self): return [self.body_yaw], self.antennas
    def push_audio_sample(self, audio): pass
    def clear_player(self): pass
    def set_target(self, head, antennas, body_yaw):
        self.pose, self.antennas, self.body_yaw = head.copy(), antennas.copy(), body_yaw
        if self.record:
            self.frames.append((head.copy(), antennas.copy(), body_yaw))


class Move:
    duration = 1.0
    def evaluate(self, elapsed):
        phase = math.sin(math.pi * min(elapsed, self.duration) / self.duration)
        return pose_factory(roll=12 * phase, pitch=-5 * phase), np.deg2rad([-20, 20]), 0.0


class Library:
    def list_moves(self): return ('success1', 'surprised1', 'inquiring2')
    def get(self, name): return Move()


class ClockStop:
    def __init__(self, c, robot, expressive):
        self.c, self.robot, self.expressive = c, robot, expressive
        self.now, self.count, self.cpu, self.last = 1000.0, 0, [], None
        self.plans = [SentenceMotion(Expression.HAPPY, 'exclamation'),
                      SentenceMotion(Expression.CURIOUS, 'question'),
                      SentenceMotion(Expression.HAPPY, 'plain'),
                      SentenceMotion(Expression.HAPPY, 'question')]
        t = np.arange(16000) / 16000
        self.pcm = (.3 * np.sin(2*np.pi*220*t) * (.6+.4*np.sin(2*np.pi*4*t))).astype(np.float32)

    def is_set(self): return self.count >= TICKS
    def wait(self, duration):
        if self.c._started.is_set():
            if self.last is not None:
                self.cpu.append(time.thread_time_ns() - self.last)
            self.robot.record = True
            if self.count % 40 == 0:
                plan = self.plans[(self.count // 40) % len(self.plans)]
                self.c.play_audio(self.pcm, sentence=plan if self.expressive else None)
                if self.expressive:
                    self.c.finish_sentence()
            self.count += 1
            self.last = time.thread_time_ns()
        self.now += duration
        return self.is_set()


def run(expressive):
    robot = Robot()
    c = hardware.ReachyController(pose_factory=pose_factory, pose_interpolator=lambda a,b,t: a*(1-t)+b*t,
                                  sleep_on_close=False, style=MotionStyle())
    c._np, c._robot = np, robot
    c._output_sample_rate = 16000
    c._start_music()
    c._layers, c._generated_gain = MotionLayers(MotionStyle()), Ramp(1)
    c._moves = Library()
    c._require_running = lambda: None
    c._state = 'speaking'
    stop = c._stop = ClockStop(c, robot, expressive)
    original = hardware.time
    hardware.time = SimpleNamespace(monotonic=lambda: stop.now)
    try:
        c._motion_loop()
    finally:
        hardware.time = original
    if c._failure:
        raise c._failure
    frames = np.array([c._pose_vector(head, antennas, body) for head, antennas, body in robot.frames])
    differences = np.abs(np.diff(frames, axis=0))
    return np.array(stop.cpu), np.max(np.abs(frames), axis=0), np.max(differences, axis=0)


def main():
    print('CPU-only synthetic 40 Hz controller loop on this host; fake robot and clip.')
    for label, expressive in [('layered sway', False), ('expressive sentences + sway', True)]:
        cpu, pose, step = run(expressive)
        avg, p99 = np.mean(cpu) / 1000, np.percentile(cpu, 99) / 1000
        print(f'{label}: mean {avg:.1f} us/tick ({avg/25000:.2%} of 25 ms), p99 {p99:.1f} us/tick')
        print(' axis                   ' + ' '.join(f'{axis:>13s}' for axis in AXES))
        print(' max absolute pose      ' + ' '.join(f'{value:13.2f}' for value in pose))
        print(' max step per 25 ms     ' + ' '.join(f'{value:13.2f}' for value in step))
        print(' max speed per second   ' + ' '.join(f'{value/PERIOD:13.2f}' for value in step))
    print('Translations mm; angles degrees. Speed is peak adjacent target difference / 25 ms, not measured motor speed.')


if __name__ == '__main__':
    main()
