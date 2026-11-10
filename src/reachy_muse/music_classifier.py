"""Conservative music admission from sustained harmony and rhythmic evidence."""

from collections import deque

import numpy as np


class MusicClassifier:
    def __init__(self):
        self._history = deque(maxlen=75)
        self._good = self._bad = 0
        self.music = False
        self.confidence = 0.

    def feed(self, frame) -> bool:
        self._history.append(frame)
        if len(self._history) < 60:
            return False
        recent = tuple(self._history)
        chroma = np.array([f.chroma for f in recent])
        active = np.array([f.loudness > .035 and f.tonal > .55 and f.flatness < .12 for f in recent])
        coverage = float(np.mean(active))
        stability = float(np.mean(np.sum(chroma[1:] * chroma[:-1], axis=1)))
        lasting = float(np.mean(np.sum(chroma[10:] * chroma[:-10], axis=1)))
        polyphony = float(np.mean(np.sum(chroma > .22, axis=1)))
        rhythm = float(np.mean([f.regularity for f in recent]))
        live = float(np.mean(active[-15:]))
        self.confidence = min(1., coverage, stability, polyphony / 3, max(rhythm, lasting))
        candidate = (coverage > .7 and live > .65 and stability > .9 and polyphony >= 2.5
                     and (rhythm > .65 or lasting > .94))
        self._good = self._good + 1 if candidate else 0
        self._bad = 0 if candidate else self._bad + 1
        if self._good >= 10:
            self.music = True
        if self._bad >= 8 or frame.loudness < .008:
            self.music = False
        return self.music
