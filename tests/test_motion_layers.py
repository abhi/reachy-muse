# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from reachy_muse.motion import MotionStyle
from reachy_muse.motion.layers import ClipTimeline, MotionLayers, breathing, clamp_to_limits, seamless_loop

RATE = 16000
YAW = 5
# Far below what the head motors resolve: one encoder step is 0.088 degrees.
BELOW_ONE_ENCODER_STEP = 0.05


def speech(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    syllables = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)
    return (amplitude * np.sin(2 * np.pi * 220 * t) * syllables).astype(np.float32)


def offsets(layers: MotionLayers, start: float, end: float) -> np.ndarray:
    return np.array([layers.sway_offset(t, active=True) for t in np.arange(start, end, 0.025)])


def test_sway_is_zero_in_silence_and_moves_the_head_while_speech_plays():
    layers = MotionLayers(MotionStyle())
    layers.feed_speech(np.zeros(RATE, dtype=np.float32), RATE, at=0.0)
    np.testing.assert_array_equal(offsets(layers, 0.0, 1.0), np.zeros((40, 9)))
    layers.feed_speech(speech(2.0), RATE, at=1.0)
    spoken = offsets(layers, 1.3, 3.0)
    assert np.ptp(spoken[:, YAW]) > 15
    assert np.abs(spoken).max() <= 12
    np.testing.assert_array_equal(spoken[:, 6:], 0)


def test_sway_starts_when_the_speaker_does_not_when_audio_is_queued():
    layers = MotionLayers(MotionStyle())
    layers.feed_speech(speech(1.0), RATE, at=0.0)
    np.testing.assert_array_equal(layers.sway_offset(0.05, active=True), np.zeros(9))
    assert np.abs(layers.sway_offset(0.2, active=True)).max() > 1


def test_flushed_speech_decays_below_one_encoder_step_within_1500_ms():
    layers = MotionLayers(MotionStyle())
    layers.feed_speech(speech(3.0), RATE, at=0.0)
    assert np.abs(offsets(layers, 0.0, 1.0)).max() > 1
    layers.interrupt_speech(at=1.0)
    assert np.abs(layers.sway_offset(1.05, active=True)).max() > 1
    assert np.abs(offsets(layers, 2.5, 4.0)).max() < BELOW_ONE_ENCODER_STEP


def test_sway_eases_out_when_reachy_stops_speaking():
    layers = MotionLayers(MotionStyle())
    layers.feed_speech(speech(3.0), RATE, at=0.0)
    assert np.abs(offsets(layers, 0.0, 1.0)).max() > 1
    assert np.abs(layers.sway_offset(1.2, active=False)).max() > 0
    np.testing.assert_array_equal(layers.sway_offset(1.6, active=False), np.zeros(9))


def test_sdk_style_leaves_sway_to_the_sdk_wobbler():
    animation, sdk = MotionLayers(MotionStyle()), MotionLayers(MotionStyle(sway="sdk"))
    for layers in (animation, sdk):
        layers.feed_speech(speech(1.0), RATE, at=0.0)
        layers.sway_offset(0.0, active=True)
    assert np.abs(animation.sway_offset(0.5, active=True)).max() > 1
    np.testing.assert_array_equal(sdk.sway_offset(0.5, active=True), np.zeros(9))


def test_suspended_sway_eases_back_over_four_tenths_of_a_second():
    suspended, steady = MotionLayers(MotionStyle()), MotionLayers(MotionStyle())
    for layers in (suspended, steady):
        layers.feed_speech(speech(2.0), RATE, at=0.0)
        layers.sway_offset(0.0, active=True)
    suspended.suspend(1.0)
    np.testing.assert_array_equal(suspended.sway_offset(1.0, active=True), np.zeros(9))
    np.testing.assert_allclose(suspended.sway_offset(1.2, active=True), 0.5 * steady.sway_offset(1.2, active=True))
    np.testing.assert_allclose(suspended.sway_offset(1.4, active=True), steady.sway_offset(1.4, active=True))


def test_clamp_holds_every_axis_inside_the_pose_limits():
    pose = np.array([20, -20, 25, 30, -30, 50, 100, -100, -40], dtype=float)
    np.testing.assert_array_equal(clamp_to_limits(pose), [15, -15, 20, 20, -20, 35, 90, -90, -35])
    inside = np.array([1, 2, -3, 4, 5, -6, 7, 8, 9], dtype=float)
    np.testing.assert_array_equal(clamp_to_limits(inside), inside)


def test_breathing_rises_5_mm_every_10_seconds_and_sways_the_antennas_every_2():
    np.testing.assert_allclose(breathing(0.0), np.zeros(9), atol=1e-12)
    np.testing.assert_allclose(breathing(2.5), [0, 0, 5, 0, -1, 0, 15, -15, 0], atol=1e-12)
    np.testing.assert_allclose(breathing(7.5), [0, 0, -5, 0, 1, 0, -15, 15, 0], atol=1e-12)
    np.testing.assert_allclose(breathing(3.0), breathing(13.0), atol=1e-12)
    np.testing.assert_allclose(breathing(0.5)[6:8], [15, -15], atol=1e-12)
    np.testing.assert_allclose(breathing(1.5)[6:8], [-15, 15], atol=1e-12)
    cycle = np.array([breathing(t) for t in np.arange(0, 10, 0.025)])
    np.testing.assert_allclose(np.abs(cycle).max(axis=0), [0, 0, 5, 0, 1, 0, 15, 15, 0], atol=1e-3)


def test_breathing_fades_out_over_four_tenths_of_a_second_when_its_state_ends():
    layers = MotionLayers(MotionStyle())
    layers.breathing_offset(0.0, active=True)
    np.testing.assert_allclose(layers.breathing_offset(2.5, active=True), breathing(2.5))
    layers.breathing_offset(10.0, active=False)
    np.testing.assert_allclose(layers.breathing_offset(10.2, active=False), 0.5 * breathing(10.2))
    np.testing.assert_array_equal(layers.breathing_offset(10.4, active=False), np.zeros(9))


def test_breathing_flag_off_keeps_the_rest_pose_still():
    layers = MotionLayers(MotionStyle(breathing=False))
    layers.breathing_offset(0.0, active=True)
    np.testing.assert_array_equal(layers.breathing_offset(2.5, active=True), np.zeros(9))


@pytest.mark.parametrize("environ, style", [
    ({}, MotionStyle(sway="animation", breathing=True, expression_clips=True)),
    ({"REACHY_MUSE_SWAY": "sdk", "REACHY_MUSE_BREATHING": "0", "REACHY_MUSE_EXPRESSION_CLIPS": "0"},
     MotionStyle(sway="sdk", breathing=False, expression_clips=False)),
])
def test_motion_style_reads_the_operator_flags(environ, style):
    assert MotionStyle.from_env(environ) == style


@pytest.mark.parametrize("environ, error", [
    ({"REACHY_MUSE_SWAY": "wobble"}, "REACHY_MUSE_SWAY must be animation or sdk, not 'wobble'"),
    ({"REACHY_MUSE_BREATHING": "yes"}, "REACHY_MUSE_BREATHING must be 1 or 0, not 'yes'"),
    ({"REACHY_MUSE_EXPRESSION_CLIPS": "true"}, "REACHY_MUSE_EXPRESSION_CLIPS must be 1 or 0, not 'true'"),
])
def test_unknown_flag_values_are_rejected(environ, error):
    with pytest.raises(ValueError, match=error):
        MotionStyle.from_env(environ)


def test_a_clip_crossfades_in_over_four_tenths_of_a_second_and_back_out_after_it_ends():
    timeline = ClipTimeline(started=10.0, duration=2.0)
    assert [timeline.weight(10.0 + t) for t in (0.0, 0.2, 0.4, 1.0)] == pytest.approx([0.0, 0.5, 1.0, 1.0])
    assert [timeline.weight(12.0 + t) for t in (0.0, 0.2, 0.4)] == pytest.approx([1.0, 0.5, 0.0])
    assert [timeline.clip_time(10.0 + t) for t in (-1.0, 0.5, 2.0, 2.3)] == pytest.approx([0.0, 0.5, 2.0, 2.0])
    assert [timeline.finished(12.0 + t) for t in (0.39, 0.41)] == [False, True]


def test_a_stopped_clip_holds_its_frame_and_fades_out_from_where_it_was():
    timeline = ClipTimeline(started=0.0, duration=2.0)
    timeline.stop(0.2)
    assert [timeline.weight(t) for t in (0.2, 0.4, 0.6)] == pytest.approx([0.5, 0.25, 0.0])
    assert timeline.clip_time(0.5) == pytest.approx(0.2)
    assert [timeline.finished(t) for t in (0.59, 0.61)] == [False, True]


def test_a_clip_shorter_than_two_fades_spends_half_of_itself_fading_in():
    timeline = ClipTimeline(started=0.0, duration=0.4)
    assert [timeline.weight(t) for t in (0.0, 0.1, 0.2, 0.4, 0.6, 0.8)] == pytest.approx([0.0, 0.5, 1.0, 1.0, 0.5, 0.0])


def test_a_seamless_loop_blends_its_tail_into_its_head():
    frames = np.repeat(np.arange(10.0)[:, None], 9, axis=1)
    looped = seamless_loop(frames, 2)
    np.testing.assert_allclose(looped[:, 0], [2, 3, 4, 5, 6, 7, 160 / 27, 83 / 27])
    cycle = np.concatenate([looped[:, 0], looped[:1, 0]])
    assert np.abs(np.diff(cycle)).max() <= 3.0


def test_a_loop_too_short_to_overlap_plays_as_it_is():
    frames = np.ones((1, 9))
    np.testing.assert_array_equal(seamless_loop(frames, 10), frames)
