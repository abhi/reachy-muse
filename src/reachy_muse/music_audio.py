# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Bounded PyAV PCM decoding and the shared accepted speaker clock."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable

import numpy as np

from reachy_muse.motion.dance import Beat, BeatTracker
from reachy_muse.music_analysis import AudioAnalyzer, AudioFeatures


@dataclass(frozen=True)
class AudioBlock:
    offset: int
    pcm: np.ndarray
    beats: tuple[Beat, ...]
    features: tuple[AudioFeatures, ...] = ()


@dataclass(frozen=True)
class MusicMix:
    pcm: np.ndarray
    accepted: Callable[[], None]


def decode(source, rate: int, cancelled: threading.Event):
    """Yield 40 ms mono blocks; FFmpeg diagnostics never leave this worker."""
    import av

    options = {"protocol_whitelist": "http,https,tcp,tls,crypto"} if isinstance(source, str) else {}
    with av.logging.Capture(local=True) as diagnostics:
        with av.open(str(source), timeout=(3., 2.), options=options) as container:
            if not container.streams.audio:
                raise ValueError("No audio stream")
            stream = container.streams.audio[0]
            stream.codec_context.thread_count = 1
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
            tracker = BeatTracker(rate)
            analyzer = AudioAnalyzer(rate)
            pending = np.empty(0, dtype=np.float32)
            offset = 0
            size = max(1, round(rate * .04))

            def frames():
                for frame in container.decode(stream):
                    if cancelled.is_set():
                        return
                    yield from resampler.resample(frame)
                if not cancelled.is_set():
                    yield from resampler.resample(None)

            for frame in frames():
                diagnostics.clear()
                pcm = np.nan_to_num(frame.to_ndarray().reshape(-1), nan=0., posinf=0., neginf=0.)
                pending = np.concatenate((pending, np.clip(pcm, -1, 1)))
                while len(pending) >= size and not cancelled.is_set():
                    part, pending = pending[:size].copy(), pending[size:]
                    yield AudioBlock(offset, part, tracker.feed(part), analyzer.feed(part))
                    offset += len(part)
            if len(pending) and not cancelled.is_set():
                yield AudioBlock(offset, pending.copy(), tracker.feed(pending), analyzer.feed(pending) + analyzer.finish())


class MusicOutput:
    """Serialize mixed PCM and its timestamps, including music between replies.

    ``music`` runs under ``lock`` and returns the next music samples or None.
    The pump and synchronous speech callers share this owner, never independent
    SDK timelines. No speech queue or extra playback backend is introduced.
    """

    def __init__(self, rate, push, clear, *, clock=time.monotonic):
        self.rate, self._push, self._clear, self.clock = rate, push, clear, clock
        self.lock = threading.RLock()
        self.tail = self.speech_until = 0.
        self.music = None

    def push_speech(self, pcm) -> float:
        with self.lock:
            return self._write(pcm, speech=True)

    def pump(self) -> None:
        with self.lock:
            if self.clock() + .02 >= self.tail:
                self._write(np.zeros(max(1, round(self.rate * .02)), dtype=np.float32), speech=False)

    def _write(self, pcm, *, speech):
        start = max(self.clock(), self.tail)
        music = self.music(len(pcm), start, speech) if self.music is not None else None
        if music is None and not speech:
            return start
        mixed = pcm if music is None else np.clip(pcm + music.pcm, -1, 1).astype(np.float32)
        self._push(mixed)
        if music is not None:
            music.accepted()
        self.tail = start + len(pcm) / self.rate
        if speech:
            self.speech_until = self.tail + .06
        return start

    def clear(self, *, speech=True) -> bool:
        with self.lock:
            if not speech and self.clock() < self.speech_until:
                return False
            self._clear()
            self.tail = self.speech_until = 0.
            return True
