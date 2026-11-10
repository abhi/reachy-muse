"""A library motion retimed to four source beats, used only at future peaks."""

from dataclasses import dataclass
import math

import numpy as np

from reachy_muse.motion.dance import DANCE_CAP


@dataclass(frozen=True)
class DanceSignature:
    frames: np.ndarray
    ready_after: float

    @classmethod
    def from_motion(cls, motion, ready_after):
        if motion is None or not 1 <= motion.fps <= 100:
            return None
        frames = np.asarray(motion.frames, dtype=float)
        if frames.ndim != 2 or frames.shape[1] != 9 or not 2 <= len(frames) <= 375 or not np.isfinite(frames).all():
            return None
        frames = np.clip(frames - frames[0], -DANCE_CAP, DANCE_CAP)
        frames.flags.writeable = False
        return cls(frames, ready_after)

    def sample(self, section, seconds, frame):
        start = math.ceil(section.start / section.period - 1e-8) * section.period
        if self.ready_after >= start or not (section.chorus or section.drop):
            return np.zeros(9)
        progress = (seconds - start) / (4 * section.period)
        if not 0 < progress < 1:
            return np.zeros(9)
        index = progress * (len(self.frames) - 1)
        low = int(index)
        pose = self.frames[low] * (1 - (index - low)) + self.frames[low + 1] * (index - low)
        bands = np.array([frame.harmony] * 2 + [frame.snare, frame.harmony, frame.snare,
                                              frame.harmony, frame.treble, frame.treble, frame.bass])
        return pose * bands * frame.loudness * math.sin(math.pi * progress) ** 2 * .35
