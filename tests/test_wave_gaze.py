"""Literal wave admission through the motion owner, with no camera or network."""

from dataclasses import replace
import math

import numpy as np
import pytest

from bench.wave import BOX, CAMERA_MATRIX, clip, scheduled
from reachy_muse.camera_gaze import CameraGaze, GazeProjection, GazeSettler
from reachy_muse.wave_wake import WaveWatcher
from test_face_tracking_motion import drive


@pytest.mark.parametrize("step", [.025, .05])
def test_settling_measures_velocity_and_time_instead_of_tick_count(step):
    gaze = GazeSettler()
    moving = []
    for tick in range(round(1 / step) + 1):
        now = tick * step
        pose = [0, 0, 0, 0, 0, 2 * now, 0]
        gaze.update(now, pose, pose, np.eye(3), eligible=True)
        moving.append(gaze.gaze.settled)
    assert set(moving) == {False}
    flags = []
    for tick in range(1, round(.5 / step) + 1):
        gaze.update(1 + tick * step, pose, pose, np.eye(3), eligible=True)
        if tick in (round(.25 / step), round(.5 / step)):
            flags.append(gaze.gaze.settled)
    assert flags == [False, True]


@pytest.mark.parametrize("source", ["command", "feedback"])
@pytest.mark.parametrize("speed, settled", [(.99, True), (1.0, False), (1.01, False)])
def test_settling_uses_a_strict_one_degree_per_second_bound(source, speed, settled):
    gaze = GazeSettler()
    for tick in range(21):
        now = tick * .025
        commanded, observed = np.zeros(7), np.zeros(7)
        (commanded if source == "command" else observed)[5] = speed * now
        gaze.update(now, commanded, observed, np.eye(3), eligible=True)
    assert gaze.gaze.settled is settled


@pytest.mark.parametrize("phase", [0, .007])
@pytest.mark.parametrize("axis, bound", [(0, .8), (5, 1), (6, 1)])
@pytest.mark.parametrize("speed, settled", [(.8, True), (1.2, False)])
def test_settling_with_50_hz_feedback_consumed_at_40_hz(phase, axis, bound, speed, settled):
    gaze = GazeSettler()
    flags = []
    for tick in range(401):
        now = tick / 40
        sampled_at = math.floor((now + phase) * 50) / 50 - phase
        observed = np.zeros(7)
        observed[axis] = bound * speed * sampled_at
        gaze.update(now, np.zeros(7), observed, np.eye(3), eligible=True)
        if tick > 40:
            flags.append(gaze.gaze.settled)
    assert sum(flags) == (360 if settled else 0)


@pytest.mark.parametrize("source", ["command", "feedback"])
@pytest.mark.parametrize("axis", [0, 3, 4, 5, 6])
def test_either_commanded_or_observed_motion_invalidates_evidence(source, axis):
    gaze = GazeSettler()
    for tick in range(20):
        gaze.update(tick * .025, np.zeros(7), np.zeros(7), np.eye(3), eligible=True)
    assert gaze.gaze.settled is True
    previous = gaze.gaze.generation
    commanded, observed = np.zeros(7), np.zeros(7)
    (commanded if source == "command" else observed)[axis] = .1
    gaze.update(.5, commanded, observed, np.eye(3), eligible=True)
    assert gaze.gaze.settled is False
    assert gaze.gaze.generation == previous + 1


@pytest.mark.parametrize("path", [
    [(0, 0), (.1, 0)],
    [(0, 0), (.03, 0), (.03, .03), (0, .03)],
])
def test_observed_direction_changes_do_not_cancel_out_when_settling(path):
    gaze = GazeSettler()
    flags = []
    for tick in range(41):
        observed = [0, 0, 0, *path[tick % len(path)], 0, 0]
        gaze.update(tick / 40, np.zeros(7), observed, np.eye(3), eligible=True)
        flags.append(gaze.gaze.settled)
    assert set(flags) == {False}


def gaze_sequence(monkeypatch, *, during_motion=False, after_settle=True, arrival=False):
    still = next(clip("still", 1001))
    gesture = list(clip("wave", 1001, seconds=3))
    now, next_sample = [0.0], [0.0]
    seen, reads, cues, following = [], [], [], []
    watcher = None

    def read():
        elapsed = now[0]
        frame = still
        if during_motion and elapsed < 3:
            frame = gesture[min(len(gesture) - 1, int(elapsed * 8))]
        if after_settle and 5 <= elapsed < 8:
            frame = gesture[min(len(gesture) - 1, int((elapsed - 5) * 8))]
        seen[:] = frame[2]
        reads.append(round(elapsed, 3))
        return frame[1]

    class People:
        def detect(self, frame):
            return list(seen)

    def action(controller, follower, tick):
        nonlocal watcher
        now[0] = tick * .025
        if tick == 1:
            controller._wave_watching = True
            watcher = controller._wave_watcher = WaveWatcher(
                read, controller._wave_still, clock=lambda: now[0], read_gaze=controller._read_wave_gaze)
            watcher.set_active(True)
        if during_motion:
            follower.yaw = 12 * math.sin(now[0] * 6) if tick < 120 else 8
        if tick == 180 and arrival:
            controller.start_watching(None)
        if tick in (100, 175, 250):
            following.append(follower.active)
        if now[0] >= next_sample[0]:
            next_sample[0] = now[0] + watcher._sample(People())
        if controller.take_wave():
            cues.append((round(now[0], 3), controller.claim_wave(), follower.active))

    controller, _, samples = drive(monkeypatch, action, ticks=360,
                                    face_follow_idle=True, wave_wake=True)
    watcher.close()
    return cues, reads, following, samples


def test_idle_person_is_followed_then_a_real_wave_is_admitted(monkeypatch, caplog):
    caplog.set_level("INFO", logger="reachy_muse.hardware")
    cues, reads, following, samples = gaze_sequence(monkeypatch)
    assert following == [True, True, True]
    assert samples[100]["yaw"] == pytest.approx(8, abs=.01)
    assert samples[100]["pitch"] == pytest.approx(-4, abs=.01)
    assert cues == [(6.425, True, True)]
    assert min(reads) >= 1
    assert [r.message for r in caplog.records if r.message.startswith("Wave gaze:")] == [
        "Wave gaze: follow", "Wave gaze: settled", "Wave gaze: wave watching"]


def test_wave_during_face_motion_is_ignored(monkeypatch):
    cues, reads, following, _ = gaze_sequence(monkeypatch, during_motion=True, after_settle=False)
    assert cues == []
    assert all(at > 3 for at in reads)
    assert following == [True, True, True]


def test_new_wave_after_face_motion_and_settle_is_admitted(monkeypatch):
    cues, reads, _, _ = gaze_sequence(monkeypatch, during_motion=True)
    assert all(at > 3 for at in reads)
    assert cues == [(6.85, True, True)]


def test_arrival_watching_takes_priority_over_follow_and_wave(monkeypatch):
    cues, reads, following, _ = gaze_sequence(monkeypatch, arrival=True)
    assert cues == []
    assert max(reads) < 4.5
    assert following == [True, True, False]


@pytest.mark.parametrize("when", ["none", "read", "inference", "between_samples", "pending"])
def test_motion_generation_discards_evidence_even_when_already_settled_again(when):
    now = [0.0]
    gaze = [CameraGaze(0, True, 0, np.eye(3))]
    gesture = list(clip("wave", 1001, seconds=3))
    cues = []
    changed = False

    def move():
        nonlocal changed
        gaze[0] = replace(gaze[0], generation=1)
        changed = True

    def read():
        if when == "read" and now[0] >= 2.25 and not changed:
            move()
        return gesture[min(len(gesture) - 1, int(now[0] * 8))][1]

    class People:
        def detect(self, frame):
            if when == "inference" and now[0] >= 2.25 and not changed:
                move()
            return [BOX]

    watcher = WaveWatcher(read, lambda: True, clock=lambda: now[0], read_gaze=lambda: gaze[0])
    watcher.set_active(True)
    # Seed the gate with a quiet period so the whole short wave is observed.
    quiet = next(clip("still", 1001))[1]
    watcher._gate.update(quiet, -2)
    watcher._gate.update(quiet, -1)
    for tick in range(24):
        now[0] = tick * .125
        if when == "between_samples" and now[0] >= 1 and not changed:
            move()
        watcher._sample(People())
        if when == "pending" and watcher._pending is not None and not changed:
            move()
        if watcher.take_wave():
            cues.append(now[0])
    assert changed is (when != "none")
    assert cues == ([2.25] if when == "none" else [])
    watcher.close()


def test_compensation_projects_to_fixed_pixels_and_returns_native_cue_coordinates():
    pytest.importorskip("cv2")
    from test_hardware import pose_factory

    frame = np.zeros((120, 160, 3), np.uint8)
    frame[60, 80] = 255
    projection = GazeProjection(CAMERA_MATRIX)
    projection.project(frame, np.eye(3))
    result, to_camera = projection.project(frame, pose_factory(yaw=1)[:3, :3])
    # Positive robot yaw moves the current central ray left in the reference view.
    assert np.unravel_index(result[:, :, 0].argmax(), (120, 160)) == (60, 77)
    pixel = to_camera @ [80, 60, 1]
    assert (pixel[:2] / pixel[2]).tolist() == pytest.approx([82.792810, 60], abs=.000001)


def test_projection_reduces_native_frames_and_restores_native_pixel_coordinates():
    pytest.importorskip("cv2")
    from test_hardware import pose_factory

    frame = np.zeros((1080, 1920, 3), np.uint8)
    projection = GazeProjection(np.diag([12., 9., 1.]) @ CAMERA_MATRIX)
    projection.project(frame, np.eye(3))
    result, to_camera = projection.project(frame, pose_factory(yaw=1)[:3, :3])
    assert result.shape == (234, 416, 3)
    pixel = to_camera @ [208, 117, 1]
    assert (pixel[:2] / pixel[2]).tolist() == pytest.approx([993.513725, 540], abs=.00001)


@pytest.mark.parametrize("delay", [0, .32])
def test_wave_watcher_compensates_residual_gaze_drift_with_cached_person_boxes(delay):
    pytest.importorskip("cv2")
    assert bool(scheduled("wave", delay, 1001, head_motion=True)["cues_at"]) is True
    assert scheduled("still", delay, 1001, head_motion=True)["cues_at"] == []
    assert scheduled("pan", delay, 1001, head_motion=True)["cues_at"] == []
