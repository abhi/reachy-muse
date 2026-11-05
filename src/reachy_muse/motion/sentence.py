# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Local expression choices for speech whose text is already ready to play."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from reachy_muse.contract.expressions import Expression
from reachy_muse.capabilities import Motion


@dataclass(frozen=True)
class SentenceMotion:
    expression: Expression
    terminal: Literal["plain", "question", "exclamation"] = "plain"
    motion: Motion | None = None


_WORDS = re.compile(r"[a-z]+(?:'[a-z]+)?")
_SAD = frozenset({"sorry", "unfortunately", "sad", "sadly", "failed", "failure", "condolences"})
_HAPPY = frozenset({"congratulations", "congrats", "hooray", "wonderful", "fantastic", "excellent", "delighted",
                    "great", "awesome", "lovely", "glad", "thanks", "happy", "excited"})
_SURPRISED = frozenset({"wow", "whoa", "surprising", "incredible", "unbelievable"})
_NEGATIONS = frozenset({"not", "no", "never", "isn't", "wasn't", "aren't", "don't", "didn't"})


def classify_sentence(text: str, hint: Expression | None = None) -> SentenceMotion:
    """Preserve explicit expressions; infer unmarked English text without I/O.

    Only the first 2048 characters and final punctuation are examined. Ambiguous
    statements stay neutral; punctuation accents do not depend on the emotion.
    """
    ending = text[-32:].rstrip().rstrip('\"\'”’)]}').rstrip()
    terminal = "question" if ending.endswith("?") else "exclamation" if ending.endswith("!") else "plain"
    if hint is not None:
        return SentenceMotion(hint, terminal)
    words = _WORDS.findall(text[:2048].lower().replace("’", "'"))
    positive = set()
    news = set()
    for index, word in enumerate(words):
        if not _NEGATIONS.intersection(words[max(0, index - 3):index]):
            positive.add(word)
            if words[index + 1:index + 2] == ["news"]:
                news.add(word)
    if positive & _SAD or "bad" in news:
        expression = Expression.SAD
    elif positive & _HAPPY or "good" in news:
        expression = Expression.HAPPY
    elif terminal == "question":
        expression = Expression.CURIOUS
    elif positive & _SURPRISED:
        expression = Expression.SURPRISED
    elif words and words[0] in {"yes", "absolutely", "certainly", "agreed"}:
        expression = Expression.NOD
    else:
        expression = Expression.NEUTRAL
    return SentenceMotion(expression, terminal)
