"""Streaming spectral features in source-sample coordinates."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from reachy_muse.motion.dance import BeatTracker
from reachy_muse.music_classifier import MusicClassifier


@dataclass(frozen=True, slots=True)
class AudioFeatures:
    sample: int
    loudness: float
    bass: float
    snare: float
    treble: float
    harmony: float
    chroma: tuple[float, ...]
    timbre: tuple[float, ...]
    flatness: float
    tonal: float
    period: float
    phase: float
    regularity: float
    music: bool = True
    character: str = "groove"


class AudioAnalyzer:
    def __init__(self, rate: int):
        self.rate = rate
        self.hop = max(1, round(rate * .04))
        self.window = max(self.hop, round(rate * .064))
        self._hann = np.hanning(self.window)
        self._scale = 2 / (self.window * np.sum(self._hann ** 2))
        frequencies = np.fft.rfftfreq(self.window, 1 / rate)
        edges = (35, 180, 500, 1200, 2500, 5000, rate / 2 + 1)
        self._bands = tuple((frequencies >= a) & (frequencies < b) for a, b in zip(edges, edges[1:]))
        self._pitched = (frequencies >= 180) & (frequencies < 2500)
        self._pitch_bins = np.round(69 + 12 * np.log2(np.maximum(frequencies, 1) / 440)).astype(int) % 12
        self._pending = np.empty(0, dtype=np.float32)
        self._history = np.zeros(self.window - self.hop, dtype=np.float32)
        self._sample = 0
        self._previous = np.zeros(6)
        self._snare = self._treble = 0.
        self._tracker = BeatTracker(rate)
        self._last_beat = 0
        self._beat_index = 0
        self._classifier = MusicClassifier()

    def feed(self, pcm) -> tuple[AudioFeatures, ...]:
        audio = np.concatenate((self._pending, pcm))
        count = len(audio) // self.hop
        self._pending = audio[count * self.hop:].copy()
        result = []
        for index in range(count):
            part = audio[index * self.hop:(index + 1) * self.hop]
            beats = self._tracker.feed(part)
            if beats:
                self._last_beat, self._beat_index = beats[-1].sample, beats[-1].index
            window = np.concatenate((self._history, part))
            self._history = window[self.hop:].copy()
            power = np.abs(np.fft.rfft(window * self._hann)) ** 2 * self._scale
            levels = np.array([np.sqrt(np.sum(power[band])) for band in self._bands])
            levels[levels < np.linalg.norm(levels) * .02] = 0
            flux = np.maximum(0, levels - self._previous)
            self._previous = levels
            self._snare = max(float(np.linalg.norm(flux[1:4])) / .18, self._snare * .65)
            self._treble = max(float(np.linalg.norm(flux[4:])) / .12, self._treble * .45)
            total = float(np.sum(power))
            rms = float(np.sqrt(np.mean(part * part)))
            peaks = np.zeros(len(power), dtype=bool)
            peaks[1:-1] = (power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])
            peaks &= self._pitched & (power > max(total * .018, 1e-10))
            chroma = np.bincount(self._pitch_bins[peaks], weights=power[peaks], minlength=12).astype(float)
            chroma /= max(float(np.linalg.norm(chroma)), 1e-10)
            pitched_power = float(np.sum(power[self._pitched]))
            tonal = min(1., float(np.sum(power[peaks])) * 1.7 / max(pitched_power, 1e-10))
            flatness = float(np.exp(np.mean(np.log(power[1:] + 1e-12))) / max(np.mean(power[1:]), 1e-12))
            period = self._tracker.period
            intervals = tuple(self._tracker._intervals)
            regularity = (max(0., 1 - float(np.std(intervals)) / period * 4)
                          if len(intervals) >= 3 else 0.)
            loudness = min(1., rms / .25)
            character = "calm" if loudness < .3 else "upbeat" if period < .48 and regularity > .5 else "groove"
            self._sample += self.hop
            feature = AudioFeatures(
                self._sample - 1, loudness, min(1., float(levels[0]) / .22),
                min(1., self._snare), min(1., self._treble),
                min(1., math.sqrt(pitched_power) / .22) * tonal,
                tuple(map(float, chroma)), tuple(map(float, levels / max(float(np.linalg.norm(levels)), 1e-10))),
                flatness, tonal, period,
                self._beat_index + (self._sample - 1 - self._last_beat) / self.rate / period,
                regularity, character=character)
            result.append(replace(feature, music=self._classifier.feed(feature)))
        return tuple(result)

    def finish(self) -> tuple[AudioFeatures, ...]:
        if not len(self._pending):
            return ()
        end = self._sample + len(self._pending) - 1
        frames = self.feed(np.zeros(self.hop - len(self._pending), dtype=np.float32))
        return tuple(replace(frame, sample=end) for frame in frames)
