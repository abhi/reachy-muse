import numpy as np
import pytest

from reachy_muse.motion.dance import Beat, BeatTracker, DanceTimeline, dance_pose
from reachy_muse.motion.layers import clamp_to_limits


def clicks(bpm, seconds=12, rate=16000, gain=.6):
    samples = np.zeros(round(seconds * rate), dtype=np.float32)
    for at in np.arange(.2, seconds, 60 / bpm):
        start = round(at * rate)
        n = min(round(.02 * rate), len(samples) - start)
        samples[start:start + n] = gain * np.hanning(2 * n)[n:]
    return samples


@pytest.mark.parametrize("bpm,expected", [(90, .6666667), (120, .5), (150, .4)])
def test_streaming_click_tempo_and_absolute_positions(bpm, expected):
    tracker = BeatTracker(16000)
    audio = clicks(bpm)
    beats = []
    for i in range(0, len(audio), 257):
        beats.extend(tracker.feed(audio[i:i + 257]))
    assert len(beats) == len(np.arange(.2, 12, expected))
    assert tracker.period == pytest.approx(expected, abs=.011)
    assert [beat.sample / 16000 for beat in beats[:4]] == pytest.approx(
        [.2 + i * expected for i in range(4)], abs=.011)


def test_packet_boundaries_do_not_change_beats_and_silence_has_none():
    audio = clicks(120)
    expected = BeatTracker(16000).feed(audio)
    tracker = BeatTracker(16000)
    actual = tuple(beat for part in np.array_split(audio, 73) for beat in tracker.feed(part))
    assert actual == expected
    assert BeatTracker(16000).feed(np.zeros(160000, dtype=np.float32)) == ()


def test_energy_scales_motion_and_all_patterns_stay_safe():
    full = dance_pose(4, 1)
    assert full.tolist() == pytest.approx([0, 0, -2.5, 0, 5, 0, -16, 16, 0])
    assert dance_pose(4, .25) == pytest.approx(full / 4)
    poses = np.array([dance_pose(t, 1) for t in np.linspace(0, 64, 6401)])
    assert np.max(np.abs(poses), axis=0).tolist() == pytest.approx(
        [2, 0, 2.5, 4, 5, 3, 16, 16, 7], abs=.03)
    assert np.array_equal(clamp_to_limits(poses), poses)
    assert not np.allclose(dance_pose(5.25, 1), dance_pose(21.25, 1))
    assert np.max(np.abs(dance_pose(16 - 1e-5, 1) - dance_pose(16 + 1e-5, 1))) < .01


def test_accepted_clock_does_not_drive_before_audible_time():
    dance = DanceTimeline()
    dance.offer((Beat(8000, .5, 0, 1),), 0, 10.06, 16000)
    assert dance.sample(10, eligible=True).tolist() == [0.] * 9
    assert dance.sample(10.55, eligible=True).tolist() == [0.] * 9
    dance.sample(10.6, eligible=True)
    assert np.linalg.norm(dance.sample(10.625, eligible=True)) > 0


def test_pause_releases_without_new_beat_drive():
    dance = DanceTimeline()
    dance.offer(tuple(Beat(i * 8000, .5, i, 1) for i in range(8)), 0, 0, 16000)
    for now in np.arange(0, 1, .025):
        before = dance.sample(now, eligible=True)
    after = dance.sample(1, eligible=False)
    assert np.max(np.abs(after - before)) < .5
    for now in np.arange(1.025, 2.5, .025):
        released = dance.sample(now, eligible=False)
    assert np.max(np.abs(released)) < .003
    dance.clear()
    assert np.max(np.abs(dance.sample(3, eligible=True))) < .0001
