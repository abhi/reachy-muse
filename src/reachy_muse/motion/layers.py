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

"""Motion layers added on top of Reachy's resting pose, and the final pose clamp.

A pose here is one 9-value vector in the units and order of ``POSE_LIMITS``:
x, y, z in millimetres, then roll, pitch, yaw, right antenna, left antenna and
body yaw in degrees. Breathing belongs to the resting pose. A recorded clip
replaces both through a ``ClipTimeline`` crossfade, and speech sway goes on top
of the result. The motion loop passes the sum through ``clamp_to_limits``
before every write. The crossfade and the gain ramp follow reachy-animation's
animator (https://github.com/pham-tuan-binh/reachy-animation, Apache-2.0).
"""

from __future__ import annotations

import math
import threading

import numpy as np

from reachy_muse.contract.commands import POSE_LIMITS
from reachy_muse.motion import MotionStyle
from reachy_muse.motion.speech_sway import SpeechSway

AXES = ("x", "y", "z", "roll", "pitch", "yaw", "right_antenna", "left_antenna", "body_yaw")
_LOW = np.array([POSE_LIMITS[axis][0] for axis in AXES])
_HIGH = np.array([POSE_LIMITS[axis][1] for axis in AXES])
# reachy-animation poses use metres and radians; ours use millimetres and degrees.
_FROM_ANIMATION_UNITS = np.array([1000.0] * 3 + [math.degrees(1.0)] * 6)

# The SDK's playback appsrc feeds an audiomixer (10 ms output buffers) and an
# audio sink with a 50 ms buffer-time, so sound leaves the speaker about 60 ms
# after push_audio_sample returns.
SWAY_LATENCY_S = 0.06
# Upper bound per axis for the sway offset. Upstream peaks are 6.75 mm on x,
# 5.6 mm on y, 3.4 mm on z, 3.4 degrees of roll, 9.75 degrees of pitch with the
# syllable accent and 11.25 degrees of yaw.
SWAY_CAP = np.array([7.0, 6.0, 4.0, 4.0, 10.0, 12.0, 0.0, 0.0, 0.0])
# Upper bound per axis for the breathing offset, equal to its amplitudes.
BREATHING_CAP = np.array([0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 15.0, 15.0, 0.0])
CROSSFADE_S = 0.4


def clamp_to_limits(pose: np.ndarray) -> np.ndarray:
    """Clip a pose vector to ``POSE_LIMITS``, the last step before any motor target."""
    return np.clip(pose, _LOW, _HIGH)


def crossfade_weight(elapsed_s: float, duration_s: float = CROSSFADE_S) -> float:
    """Smoothstep from 0 at the start of a crossfade to 1 at its end."""
    a = 1.0 if duration_s <= 0 else min(max(elapsed_s / duration_s, 0.0), 1.0)
    return a * a * (3.0 - 2.0 * a)


class ClipTimeline:
    """When a recorded clip shows over the resting pose, and how strongly.

    The clip fades in from the resting pose while it plays, holds its last
    frame when it ends or is stopped, and fades back out over ``fade_s``. A clip
    shorter than twice ``fade_s`` spends half of its length fading in, unless
    ``fade_in_s`` says how long.
    """

    def __init__(self, started: float, duration: float, fade_s: float = CROSSFADE_S,
                 fade_in_s: float | None = None) -> None:
        self.started = started
        self.duration = duration
        self.fade_s = fade_s
        self.fade_in_s = min(fade_s, duration / 2) if fade_in_s is None else fade_in_s
        self.end = started + duration

    def stop(self, t: float) -> None:
        """Hold the clip at ``t`` and start fading it out."""
        self.end = min(self.end, t)

    def weight(self, t: float) -> float:
        """0 shows only the resting pose, 1 only the clip."""
        fade_in = crossfade_weight(min(t, self.end) - self.started, self.fade_in_s)
        return fade_in * (1.0 - crossfade_weight(t - self.end, self.fade_s))

    def clip_time(self, t: float) -> float:
        """Seconds into the clip to show at ``t``."""
        return min(max(t, self.started), self.end) - self.started

    def finished(self, t: float) -> bool:
        return t >= self.end + self.fade_s


def seamless_loop(frames: np.ndarray, overlap: int) -> np.ndarray:
    """Frames that repeat without a jump: the last ``overlap`` frames blend into the first ones.

    Playing the result back to back runs ..., frames[overlap - 1], frames[overlap], ... across
    the seam, so it is ``overlap`` frames shorter than ``frames``.
    """
    overlap = min(overlap, len(frames) // 2)
    if overlap == 0:
        return frames
    body = frames[overlap:].copy()
    weights = np.array([crossfade_weight((i + 1) / (overlap + 1), 1.0) for i in range(overlap)])[:, None]
    body[-overlap:] = frames[-overlap:] * (1 - weights) + frames[:overlap] * weights
    return body


class Ramp:
    """A gain that eases to each new target from wherever it is, so retargeting never jumps."""

    def __init__(self, value: float) -> None:
        self.source = self.target = value
        self.start = self.duration = 0.0

    def at(self, t: float) -> float:
        return self.source + (self.target - self.source) * crossfade_weight(t - self.start, self.duration)

    def to(self, value: float, t: float, duration: float = CROSSFADE_S) -> None:
        self.source = self.at(t)
        self.target = value
        self.start = t
        self.duration = duration


def breathing(t: float) -> np.ndarray:
    """Idle breathing at ``t``: reachy-animation's head rise and antenna sway, plus a pitch nod.

    The head rises 5 mm and falls every 10 seconds, tipping up by one degree as
    it rises; the antennas sway 15 degrees apart and back every 2 seconds.
    """
    pose = np.zeros(len(AXES))
    rise = math.sin(2 * math.pi * 0.1 * t)
    sway = 15.0 * math.sin(2 * math.pi * 0.5 * t)
    pose[2], pose[4] = 5.0 * rise, -1.0 * rise
    pose[6], pose[7] = sway, -sway
    return np.clip(pose, -BREATHING_CAP, BREATHING_CAP)


class MotionLayers:
    """Idle breathing, and speech sway driven by the audio sent to the speaker.

    Each layer shows only while the caller asks for it, so spoken progress
    updates during thinking leave the head still, as the SDK wobbler did.
    Audio threads call ``feed_speech`` and ``interrupt_speech``; the motion
    thread calls the ``*_offset`` methods and ``suspend``. Every call takes its
    time from the caller.
    """

    def __init__(self, style: MotionStyle, *, latency_s: float = SWAY_LATENCY_S) -> None:
        self._lock = threading.Lock()
        self._speech = SpeechSway(latency_s=latency_s) if style.sway == "animation" else None
        self._sway_gain = Ramp(0.0)
        self._breathing = style.breathing
        self._breathing_gain = Ramp(0.0)

    def feed_speech(self, pcm: np.ndarray, sample_rate: int, at: float) -> None:
        """Schedule audio just handed to the speaker; chunks play back to back."""
        if self._speech is not None:
            with self._lock:
                self._speech.feed(pcm, sample_rate, at)

    def interrupt_speech(self, at: float) -> None:
        """Drop audio flushed from the speaker so the sway decays from its current level."""
        if self._speech is not None:
            with self._lock:
                self._speech.interrupt(at)

    def suspend(self, t: float) -> None:
        """Drop the offsets now, for a command that starts from the pose last written.

        The next offset call eases them back in over ``CROSSFADE_S``.
        """
        self._sway_gain.to(0.0, t, 0.0)
        self._breathing_gain.to(0.0, t, 0.0)

    def breathing_offset(self, t: float, *, active: bool) -> np.ndarray:
        """Breathing to add to the resting pose at ``t``, easing in or out when ``active`` changes."""
        if not self._breathing:
            return np.zeros(len(AXES))
        if self._breathing_gain.target != float(active):
            self._breathing_gain.to(float(active), t)
        return self._breathing_gain.at(t) * breathing(t)

    def sway_offset(self, t: float, *, active: bool) -> np.ndarray:
        """Speech sway to add on top of everything else at ``t``, easing in or out when ``active`` changes."""
        if self._speech is None:
            return np.zeros(len(AXES))
        if self._sway_gain.target != float(active):
            self._sway_gain.to(float(active), t)
        with self._lock:
            speech = self._speech.sample(t)
        return self._sway_gain.at(t) * np.clip(speech * _FROM_ANIMATION_UNITS, -SWAY_CAP, SWAY_CAP)
