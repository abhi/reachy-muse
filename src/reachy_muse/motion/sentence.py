# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Bounded, local expression choices for text that is already ready to speak."""

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
    variant: Literal["attentive", "curious_tilt", "warm"] | None = None
    revised: bool = False


_WORDS = re.compile(r"[a-z]+(?:'[a-z]+)?|[,;.!?]")
_BOUNDARIES = (",", ";", ".", "!", "?", "but", "while")
_SAD = frozenset("sorry unfortunately unfortunate sad sadly failed failure condolences cancelled canceled "
                 "delayed worse worst layoffs layoff slump slumped slumps selloff plunged plummeted "
                 "killed injured deaths tragedy destroyed painful disappointing terrible bankruptcy "
                 "breach setback drought collapsed losses lost loss grief death tragic disaster crisis recession".split())
_HAPPY = frozenset("congratulations congrats hooray wonderful fantastic excellent delighted great awesome "
                   "lovely glad thanks happy excited enjoy perfect nice fun amazing beautiful yay "
                   "rebounded reopened restored won recover improves".split())
_SURPRISED = frozenset("wow whoa surprising surprisingly incredible incredibly unbelievable remarkable "
                       "astonishing shocking stunned stunning unexpectedly surprise".split())
_AGREEING = frozenset("yes absolutely certainly agreed sure definitely exactly correct yep yeah".split())
_NEGATIONS = frozenset("not no never isn't wasn't aren't don't didn't doesn't won't wouldn't cannot can't "
                       "denied without".split())
_UP = frozenset("gain gains gained rally rallied rose rise rises rising grew growth climbed climbs "
                "surge surged high higher".split())
_DOWN = frozenset("fall falls fell falling drop drops dropped decline declined declining eased lower".split())
_MARKET = frozenset("stocks stock shares share market markets index equities earnings profits profit "
                    "revenue hiring jobs wages sales".split())
_HARM = frozenset("unemployment inflation deaths infections crime pollution poverty costs".split())
_DIRECTION = _UP | _DOWN
_SUBJECTS = _HARM | _MARKET
_PHRASES = {
    Expression.SAD: ("bad news", "terrible news", "sold out", "close permanently", "without power"),
    Expression.HAPPY: ("good news", "well done", "you did it", "you've got this", "keep going",
                       "making progress", "paid off", "home safely", "found safe"),
    Expression.SURPRISED: ("who would have expected", "than anyone expected", "unexpected discovery"),
    Expression.THINKING: ("not sure", "don't know", "uncertain", "unclear", "perhaps", "maybe",
                          "unconfirmed", "too early to tell", "can't verify", "it depends",
                          "not enough information", "isn't enough information", "don't have enough information",
                          "not have enough information", "tentative", "unsure", "inconclusive", "missing some context"),
    Expression.CURIOUS: ("tell me more", "like to hear how"),
    Expression.NOD: ("try", "you could", "i'd suggest", "remember to", "make sure", "my advice",
                     "consider", "it may help", "take your time", "let's break", "you should",
                     "i recommend", "a useful next step", "please save", "you can begin", "start by", "it helps"),
}
_PATTERNS = {key: re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b")
             for key, phrases in _PHRASES.items()}
_WARM = re.compile(r"^(?:good morning|good afternoon|good evening|hello|hi\b|hey\b|welcome back|see you|"
                   r"take care|goodbye|have a good|talk to you)")


def classify_sentence(text: str, hint: Expression | None = None) -> SentenceMotion:
    """Lexical cues win over delivery. Mixed or negated sentiment stays factual.

    Inspect at most 2048 characters. Direction words need a nearby financial or
    adverse subject: falling inflation is welcome; falling temperatures are facts.
    This is English phrase matching, not a general sentiment model.
    """
    ending = text[-32:].rstrip().rstrip('\"\'”’)]}').rstrip()
    terminal = "question" if ending.endswith("?") else "exclamation" if ending.endswith("!") else "plain"
    if hint is not None and hint is not Expression.NEUTRAL:
        return SentenceMotion(hint, terminal)
    folded = text[:2048].lower().replace("’", "'").strip()
    token_matches = list(_WORDS.finditer(folded))
    tokens = [match[0] for match in token_matches]
    active = []
    clause = []
    for word in tokens:
        if word in _BOUNDARIES:
            clause = []
        negated = bool(_NEGATIONS.intersection(clause[-4:]))
        active.append(not negated)
        clause.append(word)
    words = {word for word, enabled in zip(tokens, active) if enabled}
    active_starts = {match.start() for match, enabled in zip(token_matches, active) if enabled}
    phrases = {expression: [match for match in pattern.finditer(folded) if match.start() in active_starts]
               for expression, pattern in _PATTERNS.items()}
    # Idioms, quoted vocabulary, and hypothetical markets are not a sentiment claim.
    literal = ("great depression" in folded or "great recession" in folded
               or folded.startswith(("the word ", "word ")) and " is " in folded
               or (" whether " in folded or folded.startswith("if ")) and " would " in folded)
    sad_indices = {index for index, word in enumerate(tokens) if active[index] and word in _SAD}
    sad = False
    happy = bool(words & _HAPPY) and not literal
    for matches, target in ((phrases[Expression.SAD], "sad"), (phrases[Expression.HAPPY], "happy")):
        for match in matches:
            if target == "happy" and terminal == "question" and match[0] == "keep going":
                continue
            if target == "sad":
                sad = True
            else:
                happy = True
    for index, word in enumerate(tokens):
        if not active[index] or word not in _DIRECTION or literal:
            continue
        # Do not borrow the subject across a comma or contrast clause.
        subject_index = None
        for neighbors in (range(index - 1, max(-1, index - 6), -1),
                          range(index + 1, min(len(tokens), index + 4))):
            for candidate in neighbors:
                if tokens[candidate] in _BOUNDARIES:
                    break
                if tokens[candidate] in _SUBJECTS:
                    subject_index = candidate
                    break
            if subject_index is not None:
                break
        subject = tokens[subject_index] if subject_index is not None else None
        if subject in _HARM:
            sad |= word in _UP
            happy |= word in _DOWN
            if word in _DOWN:
                sad_indices.discard(subject_index)
        elif subject in _MARKET:
            happy |= word in _UP
            sad |= word in _DOWN
    sad |= bool(sad_indices) and not literal
    if sad and happy:
        expression = Expression.NEUTRAL
    elif sad:
        expression = Expression.SAD
    elif happy:
        expression = Expression.HAPPY
    elif terminal == "question":
        expression = Expression.CURIOUS
    elif words & _SURPRISED or phrases[Expression.SURPRISED]:
        expression = Expression.SURPRISED
    elif phrases[Expression.THINKING]:
        expression = Expression.THINKING
    elif phrases[Expression.CURIOUS]:
        expression = Expression.CURIOUS
    elif tokens and tokens[0] in _AGREEING or any(match.start() == 0 for match in phrases[Expression.NOD]):
        expression = Expression.NOD
    else:
        expression = Expression.NEUTRAL
    variant = "warm" if expression is Expression.NEUTRAL and _WARM.match(folded) else None
    return SentenceMotion(expression, terminal, variant=variant)
