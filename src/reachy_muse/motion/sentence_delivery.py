# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Subtle sentence variants, selected only after the speaker accepts audio."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import math

import numpy as np

from reachy_muse.capabilities import Motion
from reachy_muse.contract.expressions import Expression
from reachy_muse.motion.prosody import Delivery, ProsodyTracker
from reachy_muse.motion.sentence import SentenceMotion

VARIANTS = ("attentive", "curious_tilt", "warm")
# Millimetres, head degrees, antenna degrees, body degrees. No body turn.
NEUTRAL_LIMITS = {
    "attentive": (1, 1, 1, 1, 3, 1, 3, 3, 0),
    "curious_tilt": (1, 1, 1, 5, 2, 2, 6, 3, 0),
    "warm": (1, 1, 2, 3, 1, 1, 8, 8, 0),
}


class SentenceSelector:
    """Keep semantic expression and gesture identity separate; never invent valence."""

    def __init__(self):
        self.recent = []
        self.next = 0

    def choose(self, sentence: SentenceMotion, delivery: Delivery | None = None,
               *, revise: bool = False) -> SentenceMotion:
        previous = self.recent[:-1] if revise else self.recent
        if sentence.expression is not Expression.NEUTRAL:
            self.recent = (previous + [sentence.expression.value])[-3:]
            return sentence
        preferred = sentence.variant
        if preferred is None and delivery is not None:
            if delivery.energy_dbfs < -29 or delivery.syllables_per_s < 1.5:
                preferred = "attentive"
            elif delivery.rise >= 2 or delivery.pitch_range >= 5:
                preferred = "curious_tilt"
            elif delivery.energy_dbfs > -20 and delivery.syllables_per_s >= 3:
                preferred = "warm"
        current = self.recent[-1] if revise and self.recent and self.recent[-1] in VARIANTS else None
        variant = preferred or current or VARIANTS[self.next % len(VARIANTS)]
        if previous[-2:] == [variant, variant]:
            variant = VARIANTS[(VARIANTS.index(variant) + 1) % len(VARIANTS)]
        if not revise:
            self.next += 1
        self.recent = (previous + [variant])[-3:]
        return replace(sentence, variant=variant)


@lru_cache(maxsize=3)
def neutral_motion(variant: str) -> Motion:
    """A 1.2 s gesture with zero endpoints, bounded before the usual crossfade."""
    frames = []
    for i in range(31):
        weight = math.sin(math.pi * i / 30) ** 2
        row = [0.] * 9
        if variant == "curious_tilt":
            row[3], row[4] = 5 * weight, -2 * weight
            row[6], row[7] = -6 * weight, 3 * weight
        elif variant == "warm":
            row[2], row[3] = 2 * weight, -3 * weight
            row[6], row[7] = -8 * weight, 8 * weight
        else:
            row[4] = 3 * weight
            row[6], row[7] = -2 * weight, 2 * weight
        frames.append(tuple(row))
    return Motion(25, tuple(frames))


class SentenceDelivery:
    """One playback owner's history and a bounded sample of the current sentence.

    Call only after a successful speaker push. At most 0.8 s is analysed once
    per sentence. Clear cancels its pending refinement, but keeps recent choices.
    """

    def __init__(self):
        self.selector = SentenceSelector()
        self.clear()

    def clear(self):
        self.current = None
        self.lexical = None
        self.tracker = None
        self.samples = 0
        self.refined = False

    def accept(self, pcm, rate: int, sentence: SentenceMotion | None) -> SentenceMotion | None:
        fresh = sentence is not None
        if fresh:
            if sentence.revised and self.current is None:
                return None
            revised = sentence.revised and self.current is not None
            if not revised:
                self.clear()
            self.lexical = sentence
            self.current = self.selector.choose(sentence, revise=revised)
            if self.current.expression is not Expression.NEUTRAL:
                self.tracker = None
            if self.current.expression is Expression.NEUTRAL and not revised and rate >= 8000:
                self.tracker = ProsodyTracker(rate=rate)
        if self.current is None:
            return None
        changed = False
        if self.tracker is not None and not self.refined:
            count = min(len(pcm), max(0, int(.8 * rate) - self.samples))
            if count:
                self.tracker.feed(np.asarray(pcm[:count]) * 32767)
                self.samples += count
            if self.samples >= int(.8 * rate):
                self.refined = True
                # Keep the lexical greeting/sign-off preference if present.
                delivery = self.tracker.delivery()
                if delivery is not None and self.current.expression is Expression.NEUTRAL:
                    choice = self.selector.choose(self.lexical, delivery, revise=True)
                    changed = choice.variant != self.current.variant
                    self.current = choice
        return replace(self.current, revised=not fresh or sentence.revised) if fresh or changed else None
