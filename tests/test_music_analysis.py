from dataclasses import replace

import numpy as np
import pytest

from reachy_muse.music_analysis import AudioAnalyzer
from reachy_muse.motion.dance import DANCE_CAP, DanceTimeline, feature_pose
from reachy_muse.motion.layers import clamp_to_limits


def stem(kind, seconds=8, rate=16000, gain=.5):
    t = np.arange(round(seconds * rate)) / rate
    pulse = np.exp(-(t % .5) / .04)
    if kind == "kick":
        return (gain * np.sin(2 * np.pi * 70 * t) * pulse).astype(np.float32)
    if kind == "hat":
        return (gain * np.sin(2 * np.pi * 6000 * t) * pulse).astype(np.float32)
    if kind == "snare":
        return (gain * np.sin(2 * np.pi * 1100 * t) * pulse).astype(np.float32)
    chord = sum(np.sin(2 * np.pi * hz * t) for hz in (261.63, 329.63, 392)) / 3
    return (gain * (.55 * chord + pulse * np.sin(2 * np.pi * 70 * t))).astype(np.float32)


def features(pcm, rate=16000):
    return tuple(replace(f, music=True) for f in AudioAnalyzer(rate).feed(pcm))


def test_synthetic_stems_drive_literal_axes():
    kick = np.array([feature_pose(f) for f in features(stem("kick"))])
    hat = np.array([feature_pose(f) for f in features(stem("hat"))])
    assert np.max(np.abs(kick[:, 8])) > 1
    assert np.max(np.abs(kick[:, 6:8])) < .01
    assert np.max(np.abs(hat[:, 6:8])) > 1
    assert np.max(np.abs(hat[:, 8])) < .01
    snare = np.array([feature_pose(f) for f in features(stem("snare"))])
    assert np.min(snare[:, 2]) < -.1
    assert np.max(snare[:, 4]) > .3


def test_quiet_intro_and_different_songs_have_different_motion():
    loud = np.array([feature_pose(f) for f in features(stem("music"))])
    quiet = np.array([feature_pose(f) for f in features(stem("music", gain=.005))])
    other = np.array([feature_pose(f) for f in features(stem("hat"))])
    assert np.linalg.norm(quiet) < np.linalg.norm(loud) * .02
    assert np.sqrt(np.mean((loud - other) ** 2)) > .25
    assert np.max(np.abs(loud[:, 3])) > .1


def test_fragmented_audio_keeps_exact_feature_positions():
    pcm = stem("music", rate=22050)
    expected = features(pcm, 22050)
    analyzer = AudioAnalyzer(22050)
    actual = tuple(replace(f, music=True) for start in range(0, len(pcm), 257) for f in analyzer.feed(pcm[start:start + 257]))
    assert actual == expected
    assert [f.sample for f in actual[:3]] == [881, 1763, 2645]


def test_acceptance_hold_retarget_and_gap_preserve_pose_and_jerk_bounds():
    frames = features(stem("music", seconds=12))
    timeline = DanceTimeline()
    poses = []
    for tick in range(600):
        at = tick * .025
        if tick % 2 == 0 and tick // 2 < len(frames):
            f = frames[tick // 2]
            timeline.offer_features((replace(f, sample=0),), 0, at + .06, 16000)
        poses.append(timeline.sample(at, eligible=not 150 <= tick < 200))
        assert np.all(np.abs(timeline.jerk) <= np.array([9260, 9260, 13889, 23149, 32408, 18519, 64815, 64815, 41667]))
    poses = np.array(poses)
    assert np.all(np.max(np.abs(poses), axis=0) <= DANCE_CAP)
    assert np.array_equal(clamp_to_limits(poses), poses)
    jerk = np.diff(poses, n=3, axis=0) / .025 ** 3
    assert np.max(np.abs(jerk[:, 8])) < 41667
    assert np.linalg.norm(poses[195]) < np.linalg.norm(poses[149]) * .03
    assert np.max(np.abs(poses)) > .2
