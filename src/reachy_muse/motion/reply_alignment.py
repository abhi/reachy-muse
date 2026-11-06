# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Where each sentence of a streamed reply starts in its audio, before that audio plays.

A realtime reply streams its transcript and its 24 kHz speech side by side, with no timing
between them. A sentence's start is estimated from its character position at the reply's
speaking rate plus typical pauses for the sentence breaks and commas before it, then snapped to the end of the pause
before it once that stretch of audio has arrived; longer pauses win over nearer ones, since
sentence breaks pause longer than commas. Each snapped sentence, and the finished reply,
corrects the rate for the sentences after it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

SAMPLE_RATE = 24000
FRAME = SAMPLE_RATE // 100
PRIOR_CHARS_PER_S = 15.0
BREAK_PAUSE_S = 0.35
COMMA_PAUSE_S = 0.15
SPEECH_FLOOR = 400.0
MIN_PAUSE_FRAMES = 8
MIN_WINDOW_S = 0.5
MAX_WINDOW_S = 1.0
LONG_PAUSE_FRAMES = 40
MAX_SENTENCE_CHARS = 240

_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s)|\n")
_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e", "a.m", "p.m", "no"})


@dataclass(frozen=True)
class Sentence:
    """``pause_s`` is the pause a speaker typically leaves before it, at sentence breaks and commas."""

    index: int
    text: str
    start_char: int
    pause_s: float = 0.0


class SentenceCutter:
    """Complete sentences from transcript deltas, each with its character offset in the reply."""

    def __init__(self) -> None:
        self.text = ""
        self._cut = 0
        self._count = 0

    def feed(self, delta: str) -> List[Sentence]:
        self.text += delta
        sentences = []
        while True:
            end = self._next_end()
            if end is None:
                return sentences
            sentence = self._emit(end)
            if sentence is not None:
                sentences.append(sentence)

    def finish(self, transcript: Optional[str] = None) -> List[Sentence]:
        """The rest of the reply as its last sentence; a full transcript replaces the deltas' text."""
        if transcript is not None and transcript.startswith(self.text[:self._cut]):
            self.text = transcript
        sentences = self.feed("")
        last = self._emit(len(self.text))
        return sentences + ([last] if last is not None else [])

    def pause_s(self, char: int, breaks: int) -> float:
        """The pause expected before ``char``, after ``breaks`` sentence breaks."""
        return breaks * BREAK_PAUSE_S + self.text.count(",", 0, char) * COMMA_PAUSE_S

    def pending(self) -> Optional[Sentence]:
        """The sentence still arriving, with the words it has so far."""
        raw = self.text[self._cut:]
        text = raw.strip()
        if not text:
            return None
        start = self._cut + len(raw) - len(raw.lstrip())
        return Sentence(self._count, text, start, self.pause_s(start, self._count))

    def _next_end(self) -> Optional[int]:
        position = self._cut
        while True:
            match = _END.search(self.text, position)
            if match is None:
                break
            words = self.text[self._cut:match.start()].split()
            if match.group() != "\n" and words and words[-1].lower().rstrip(".") in _ABBREVIATIONS:
                position = match.end()
                continue
            return match.end()
        if len(self.text) - self._cut > MAX_SENTENCE_CHARS:
            space = self.text.rfind(" ", self._cut, self._cut + MAX_SENTENCE_CHARS)
            return space if space > self._cut else self._cut + MAX_SENTENCE_CHARS
        return None

    def _emit(self, end: int) -> Optional[Sentence]:
        raw = self.text[self._cut:end]
        start = self._cut + len(raw) - len(raw.lstrip())
        self._cut = end
        text = raw.strip()
        if not text:
            return None
        self._count += 1
        return Sentence(self._count - 1, text, start, self.pause_s(start, self._count - 1))


class ReplyAligner:
    """Sentence starts for one reply, in samples from the start of its speech."""

    def __init__(self, *, rate: int = SAMPLE_RATE, chars_per_s: float = PRIOR_CHARS_PER_S) -> None:
        self.rate = rate
        self._frame = rate // 100
        self._prior = rate / chars_per_s
        self._levels: List[float] = []
        self._pending = np.zeros(0, dtype=np.float64)
        self.received = 0
        self._anchors: List[Tuple[int, int, int]] = []
        self._finished: Optional[Tuple[int, float]] = None
        self._audio_done = False

    def audio(self, offset: int, pcm: np.ndarray) -> None:
        """Record the loudness of int16 samples that start ``offset`` samples into the reply."""
        samples = np.asarray(pcm, dtype=np.float64)
        if offset > self.received:
            samples = np.concatenate([np.zeros(offset - self.received), samples])
        elif offset < self.received:
            samples = samples[self.received - offset:]
        if len(samples):
            self._audio_done = False
        data = np.concatenate([self._pending, samples])
        whole = len(data) // self._frame * self._frame
        if whole:
            frames = data[:whole].reshape(-1, self._frame)
            self._levels.extend(np.sqrt(np.mean(frames * frames, axis=1)).tolist())
        self._pending = data[whole:]
        self.received = len(self._levels) * self._frame + len(self._pending)

    def finish(self, total_chars: int, pause_s: float) -> None:
        """The whole transcript, which can arrive before the rest of its speech."""
        self._finished = (total_chars, pause_s)

    def audio_done(self) -> None:
        self._audio_done = True

    @property
    def onset(self) -> Optional[int]:
        for index, level in enumerate(self._levels):
            if level >= SPEECH_FLOOR:
                return index * self._frame
        return None

    def start(self, sentence: Sentence) -> Tuple[int, bool]:
        """The best estimate of ``sentence``'s first sample now, and whether a pause confirmed it."""
        sample, pause = self._place(sentence)
        return sample, pause is not None

    def pause_before(self, sentence: Sentence) -> Optional[int]:
        """Where the pause that confirms ``sentence``'s start begins, so the speech before it ends."""
        return self._place(sentence)[1]

    def _place(self, sentence: Sentence) -> Tuple[int, Optional[int]]:
        onset = self.onset
        if sentence.start_char == 0 or sentence.index == 0:
            return (onset, onset) if onset is not None else (0, None)
        base_char, base_sample, base_pause = 0, onset if onset is not None else 0, 0.0
        for char, sample, pause_s in self._anchors:
            if char < sentence.start_char:
                base_char, base_sample, base_pause = char, sample, pause_s
        estimate = base_sample + (sentence.pause_s - base_pause) * self.rate
        estimate += (sentence.start_char - base_char) * self._samples_per_char(onset)
        estimate = int(round(estimate))
        snapped = self._snap(estimate, estimate - base_sample, base_sample)
        if snapped is None and not self._audio_done:
            rate = self._transcript_rate(onset)
            if rate is not None:
                calibrated = int(round(base_sample + (sentence.pause_s - base_pause) * self.rate
                                       + (sentence.start_char - base_char) * rate))
                snapped = self._snap(calibrated, calibrated - base_sample, base_sample)
        return (snapped[1], snapped[0]) if snapped is not None else (estimate, None)

    def confirm(self, sentence: Sentence, sample: int) -> None:
        """A pause-confirmed start becomes evidence for the speaking rate."""
        if sentence.start_char > 0:
            self._anchors = sorted({*self._anchors, (sentence.start_char, sample, sentence.pause_s)})

    def _samples_per_char(self, onset: Optional[int]) -> float:
        if onset is None:
            return self._prior
        if self._audio_done:
            rate = self._transcript_rate(onset)
            if rate is not None:
                return rate
        for char, sample, pause_s in reversed(self._anchors):
            speaking = sample - onset - pause_s * self.rate
            if char > 0 and speaking > 0:
                return speaking / char
        return self._prior

    def _transcript_rate(self, onset: Optional[int]) -> Optional[float]:
        if onset is not None and self._finished is not None and self._finished[0]:
            chars, pause_s = self._finished
            end = self.speech_end
            speaking = (end or 0) - onset - pause_s * self.rate
            if end is not None and speaking > 0:
                return speaking / chars
        return None

    @property
    def speech_end(self) -> Optional[int]:
        """The sample after the last loud frame received so far."""
        for index in range(len(self._levels) - 1, -1, -1):
            if self._levels[index] >= SPEECH_FLOOR:
                return (index + 1) * self._frame
        return None

    def _snap(self, estimate: int, span: int, after: int) -> Optional[Tuple[int, int]]:
        """The pause whose end is nearest ``estimate``, favouring longer pauses, once its window has arrived."""
        window = max(int(MIN_WINDOW_S * self.rate), min(int(MAX_WINDOW_S * self.rate), span // 3))
        if self.received < estimate + window and self._finished is None:
            return None
        loud = [level for level in self._levels if level >= SPEECH_FLOOR]
        if not loud:
            return None
        quiet = max(SPEECH_FLOOR * 0.6, 0.1 * float(np.percentile(loud, 75)))
        low = max(after // self._frame + 1, (estimate - window) // self._frame - LONG_PAUSE_FRAMES)
        high = min(len(self._levels), (estimate + window) // self._frame + 1)
        best = None
        run = 0
        for index in range(low, high):
            if self._levels[index] < quiet:
                run += 1
                continue
            if run >= MIN_PAUSE_FRAMES:
                sample = index * self._frame
                distance = abs(sample - estimate) / window
                if distance <= 1:
                    score = distance - 0.8 * min(run / LONG_PAUSE_FRAMES, 1.0)
                    if best is None or score < best[0]:
                        best = (score, (index - run) * self._frame, sample)
            run = 0
        return None if best is None else best[1:]

