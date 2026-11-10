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


def test_accepted_clock_sets_bob_peak_within_one_motion_tick():
    dance = DanceTimeline()
    dance.offer(tuple(Beat(i * 8000, .5, i, 1) for i in range(8)), 0, 10.06, 16000)
    times = np.arange(10.06, 12, .025)
    poses = np.array([dance.sample(t, eligible=True) for t in times])
    window = (times >= 11.45) & (times <= 11.7)
    peak = times[window][np.argmax(poses[window, 4])]
    assert peak == pytest.approx(11.56, abs=.025)


def test_pause_freezes_the_offset_then_releases_without_more_beats():
    dance = DanceTimeline()
    dance.offer(tuple(Beat(i * 8000, .5, i, 1) for i in range(8)), 0, 0, 16000)
    dance.sample(0, eligible=True)
    before = dance.sample(.5, eligible=True)
    assert dance.sample(.51, eligible=False) == pytest.approx(before)
    assert dance.sample(.71, eligible=False) == pytest.approx(before * .5)
    assert dance.sample(.92, eligible=False).tolist() == [0.] * 9
    assert dance.sample(1, eligible=True).tolist() == [0.] * 9
    assert np.any(dance.sample(1.5, eligible=True))
    dance.clear()
    dance.sample(1.51, eligible=False)
    assert dance.sample(2, eligible=False).tolist() == [0.] * 9
