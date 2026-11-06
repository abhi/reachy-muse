# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Emphasis and question rises found in a reply's speech PCM, ahead of its words.

Loudness is measured every 10 ms and pitch every 20 ms by autocorrelation. A stressed syllable
is a loudness peak well above the reply's recent level; a rise is a pitch climb into a pause.
Each accent names the sentence-accent shape the motion loop already plays: a small nod for
emphasis, an antenna flick for strong emphasis, a head tilt for a rise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np

SAMPLE_RATE = 24000
SPEECH_FLOOR = 400.0
EMPHASIS_DB = 5.0
STRONG_DB = 8.0
PEAK_RADIUS = 15
MIN_GAP_S = 1.0
RISE_RATIO = 1.15
RISE_SPAN_S = 0.4
PAUSE_HOPS = 15
VOICED = 0.45

Kind = Literal["plain", "exclamation", "question"]


@dataclass(frozen=True)
class Accent:
    sample: int
    kind: Kind


@dataclass(frozen=True)
class Delivery:
    """Measured delivery, not emotional valence; rate is envelope peaks per second."""

    pitch_range: float  # semitones, 10th to 90th percentile
    energy_dbfs: float
    syllables_per_s: float
    rise: float  # semitones, last third versus first third


class ProsodyTracker:
    """Accents for one reply, in samples from its start, found as its int16 PCM arrives in order."""

    def __init__(self, *, rate: int = SAMPLE_RATE) -> None:
        self.rate = rate
        self._hop = rate // 100
        self._pitch_hop = rate // 50
        self._window = rate // 25
        self._audio = np.zeros(0, dtype=np.float64)
        self._base = 0
        self._levels: List[float] = []
        self._pitch: List[Optional[float]] = []
        self._scanned = 0
        self._mean_db: Optional[float] = None
        self._last_accent = -math.inf
        self._quiet_run = 0
        self._rise_checked = -1

    def feed(self, pcm: np.ndarray) -> List[Accent]:
        self._audio = np.concatenate([self._audio, np.asarray(pcm, dtype=np.float64)])
        end = self._base + len(self._audio)
        while (len(self._levels) + 1) * self._hop <= end:
            start = len(self._levels) * self._hop - self._base
            frame = self._audio[start:start + self._hop]
            self._levels.append(float(np.sqrt(np.mean(frame * frame))))
        while len(self._pitch) * self._pitch_hop + self._window <= end:
            start = len(self._pitch) * self._pitch_hop - self._base
            self._pitch.append(self._f0(self._audio[start:start + self._window]))
        keep = min(len(self._levels) * self._hop, len(self._pitch) * self._pitch_hop) - self._base
        if keep:
            self._audio = self._audio[keep:]
            self._base += keep
        accents = []
        while self._scanned + PEAK_RADIUS < len(self._levels):
            accent = self._scan(self._scanned)
            if accent is not None:
                accents.append(accent)
            self._scanned += 1
        return accents

    def delivery(self) -> Optional[Delivery]:
        """Summarize at most the first 0.8 s already received, never wait for PCM.

        Require 0.4 s and voiced pitch. Silence, DC and unvoiced noise cannot
        supply an emotional signal. Envelope peaks estimate pace without ASR.
        """
        levels = np.asarray(self._levels[:80])
        pitch = np.asarray([p for p in self._pitch[:40] if p is not None])
        voiced = levels[levels >= SPEECH_FLOOR]
        if len(levels) < 40 or len(pitch) < 10 or len(voiced) < 20:
            return None
        low, high = np.percentile(pitch, [10, 90])
        third = len(pitch) // 3
        rise = 12 * math.log2(float(np.median(pitch[-third:]) / np.median(pitch[:third])))
        smooth = np.convolve(levels, np.ones(5) / 5, mode="valid")
        threshold = float(np.median(voiced))
        peaks, last = 0, -10
        for i in range(1, len(smooth) - 1):
            if smooth[i] > threshold and smooth[i - 1] < smooth[i] >= smooth[i + 1] and i - last >= 10:
                peaks, last = peaks + 1, i
        return Delivery(12 * math.log2(float(high / low)),
                        20 * math.log10(float(np.sqrt(np.mean(voiced ** 2))) / 32768),
                        peaks / (len(levels) / 100), rise)

    def _f0(self, window: np.ndarray) -> Optional[float]:
        if float(np.sqrt(np.mean(window * window))) < SPEECH_FLOOR:
            return None
        signal = (window - window.mean())[::2]
        rate = self.rate / 2
        size = 1 << (2 * len(signal) - 1).bit_length()
        spectrum = np.fft.rfft(signal, size)
        correlation = np.fft.irfft(spectrum * np.conj(spectrum), size)[:len(signal)]
        if correlation[0] <= 0:
            return None
        low, high = int(rate / 400), int(rate / 60)
        lag = low + int(np.argmax(correlation[low:high]))
        if correlation[lag] / correlation[0] < VOICED:
            return None
        return rate / lag

    def _scan(self, index: int) -> Optional[Accent]:
        level = self._levels[index]
        if level < SPEECH_FLOOR:
            self._quiet_run += 1
            if self._quiet_run == PAUSE_HOPS:
                return self._rise(index - PAUSE_HOPS)
            return None
        self._quiet_run = 0
        db = 20 * math.log10(level)
        self._mean_db = db if self._mean_db is None else self._mean_db + (db - self._mean_db) * 0.01
        neighbourhood = self._levels[max(0, index - PEAK_RADIUS):index + PEAK_RADIUS + 1]
        if level < max(neighbourhood) or not self._spaced(index):
            return None
        lift = db - self._mean_db
        if lift < EMPHASIS_DB:
            return None
        self._last_accent = index * self._hop
        return Accent(index * self._hop, "exclamation" if lift >= STRONG_DB else "plain")

    def _rise(self, pause_start: int) -> Optional[Accent]:
        """A pitch climb over the voiced stretch that ends where the pause begins."""
        end = pause_start * self._hop
        first = max(0, (end - int(RISE_SPAN_S * self.rate)) // self._pitch_hop)
        last = min(len(self._pitch), max(0, end - self._window) // self._pitch_hop + 1)
        voiced = [value for value in self._pitch[first:last] if value is not None]
        if len(voiced) < 6 or not self._spaced(pause_start):
            return None
        third = len(voiced) // 3
        before, after = float(np.median(voiced[:third])), float(np.median(voiced[-third:]))
        if after < before * RISE_RATIO:
            return None
        sample = max(0, end - int(0.1 * self.rate))
        self._last_accent = sample
        return Accent(sample, "question")

    def _spaced(self, index: int) -> bool:
        return index * self._hop - self._last_accent >= MIN_GAP_S * self.rate
