# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Reachy's motion for Muse's realtime replies: per-sentence expressions and prosody accents.

Transcript deltas are cut into sentences and each gets an expression from the local classifier
as soon as its words arrive. The speaker asks for the marks inside every chunk it is about to
play, so a sentence's motion starts at the sample the aligner places it at, and a stressed
syllable or a rising question gets its accent where the PCM shows it, whether or not its words
have arrived. A sentence whose speech starts before all its words have arrived starts with the
expression its first words suggest; once the rest arrive, a different feeling replaces it if the
sentence is still playing. A sentence ends where the pause before the next one begins, or, for a
reply's last sentence, where its speech ends once the reply is done; its terminal gesture plays
there. While Muse moves Reachy itself, local marks are withheld.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from reachy_muse.capabilities import CapabilityUnavailable, MotionSource
from reachy_muse.contract.expressions import Expression
from reachy_muse.motion.moments import CLOSINGS
from reachy_muse.motion.prosody import Accent, ProsodyTracker
from reachy_muse.motion.reply_alignment import SAMPLE_RATE, ReplyAligner, Sentence, SentenceCutter
from reachy_muse.motion.sentence import SentenceMotion, classify_sentence

log = logging.getLogger(__name__)

ACCENT_CLEARANCE_S = 0.5
LATE_LIMIT_S = 1.5


@dataclass(frozen=True)
class Mark:
    """Motion due ``sample`` samples into the chunk being played; ``ends`` closes the shown sentence."""

    sample: int
    sentence: Optional[SentenceMotion] = None
    text: Optional[str] = None
    accent: Optional[str] = None
    revised: bool = False
    ends: bool = False


@dataclass(frozen=True)
class Scheduled:
    """Where one sentence's motion was placed: its 24 kHz sample in the reply, and how it was found."""

    index: int
    sample: int
    confirmed: bool
    late: bool
    shown: bool


@dataclass
class _Reply:
    cutter: SentenceCutter = field(default_factory=SentenceCutter)
    aligner: ReplyAligner = field(default_factory=ReplyAligner)
    prosody: ProsodyTracker = field(default_factory=ProsodyTracker)
    sentences: List[Sentence] = field(default_factory=list)
    committed: Set[int] = field(default_factory=set)
    motions: Dict[int, SentenceMotion] = field(default_factory=dict)
    partial: Set[int] = field(default_factory=set)
    revision: Optional[Tuple[int, SentenceMotion]] = None
    library: Dict[int, asyncio.Task] = field(default_factory=dict)
    accents: List[Accent] = field(default_factory=list)
    scheduled: List[Scheduled] = field(default_factory=list)
    expressions: List[Expression] = field(default_factory=list)
    terminal: Optional[str] = None
    accented: int = 0
    unended: bool = False
    done: bool = False


class ReplyMotion:
    """``muse_moving`` says whether a Muse-invoked expression or move is running right now."""

    def __init__(self, *, library: Optional[MotionSource] = None,
                 muse_moving: Callable[[], bool] = lambda: False) -> None:
        self._library = library
        self._replies: Dict[Optional[str], _Reply] = {}
        self._muse_moving = muse_moving

    def reply(self, response_id: Optional[str]) -> _Reply:
        reply = self._replies.get(response_id)
        if reply is None:
            reply = self._replies[response_id] = _Reply()
            while len(self._replies) > 8:
                self._forget(next(iter(self._replies)))
        return reply

    def started(self, response_id: Optional[str]) -> None:
        self.reply(response_id)

    def words(self, response_id: Optional[str], delta: str) -> None:
        self._add(response_id, self.reply(response_id).cutter.feed(delta))

    def transcript(self, response_id: Optional[str], text: str) -> None:
        reply = self.reply(response_id)
        self._add(response_id, reply.cutter.finish(text))
        cutter = reply.cutter
        reply.aligner.finish(len(cutter.text), cutter.pause_s(len(cutter.text), max(0, len(reply.sentences) - 1)))

    def audio(self, response_id: Optional[str], offset: int, pcm: bytes) -> None:
        reply = self.reply(response_id)
        samples = np.frombuffer(pcm, dtype="<i2")
        reply.aligner.audio(offset, samples)
        reply.accents.extend(reply.prosody.feed(samples))

    def done(self, response_id: Optional[str]) -> None:
        """Muse finished the reply: the end of its speech received so far ends its last sentence."""
        reply = self._replies.get(response_id)
        if reply is not None:
            reply.done = True

    def flushed(self, response_ids) -> None:
        for response_id in response_ids:
            self._forget(response_id)

    def close(self) -> None:
        for response_id in list(self._replies):
            self._forget(response_id)

    def closing(self, response_id: Optional[str]) -> Optional[Expression]:
        """The feeling a reply's closing answers: the later of its last two sentences that has a closing."""
        reply = self._replies.get(response_id)
        if reply is None:
            return None
        return next((expression for expression in reversed(reply.expressions[-2:]) if expression in CLOSINGS), None)

    def report(self, response_id: Optional[str]) -> Optional[str]:
        """One log line on the motion a reply got so far."""
        reply = self._replies.get(response_id)
        if reply is None or not reply.scheduled:
            return None
        starts = " ".join(f"{item.sample / SAMPLE_RATE:.2f}" for item in reply.scheduled)
        return (f"{len(reply.scheduled)} sentences at {starts} s "
                f"({sum(item.confirmed for item in reply.scheduled)} pause-confirmed, "
                f"{sum(item.late for item in reply.scheduled)} late, "
                f"{sum(not item.shown for item in reply.scheduled)} held for Muse), "
                f"expressions {' '.join(expression.value for expression in reply.expressions)}, "
                f"{reply.accented} accents")

    def scheduled(self, response_id: Optional[str]) -> List[Scheduled]:
        reply = self._replies.get(response_id)
        return list(reply.scheduled) if reply is not None else []

    def marks(self, response_id: Optional[str], offset: int, count: int) -> List[Mark]:
        """Commit the motion that falls in the chunk ``[offset, offset + count)`` of a reply's speech."""
        reply = self.reply(response_id)
        end = offset + count
        moving = self._muse_moving()
        marks: List[Mark] = []
        if reply.revision is not None:
            (index, motion), reply.revision = reply.revision, None
            latest = reply.scheduled[-1]
            if (latest.index == index and latest.shown and not moving
                    and offset - latest.sample <= LATE_LIMIT_S * SAMPLE_RATE):
                reply.expressions[-1] = motion.expression
                marks.append(Mark(0, sentence=motion, revised=True))
        if reply.unended:
            following = next(self._due(reply), None)
            if following is not None:
                ends = reply.aligner.pause_before(following)
            else:
                ends = reply.aligner.speech_end if reply.done else None
            if ends is not None and ends < end:
                marks.append(Mark(max(ends, offset) - offset, ends=True))
                reply.unended = False
        for sentence in self._due(reply):
            sample, confirmed = reply.aligner.start(sentence)
            if sample >= end:
                break
            if confirmed:
                reply.aligner.confirm(sentence, sample)
            reply.committed.add(sentence.index)
            if sentence.index not in reply.motions:
                reply.partial.add(sentence.index)
            at = max(sample, offset)
            shown = not moving and offset - sample <= LATE_LIMIT_S * SAMPLE_RATE
            motion = self._motion(reply, sentence)
            reply.expressions.append(motion.expression)
            reply.terminal = motion.terminal
            reply.scheduled.append(Scheduled(sentence.index, at, confirmed, sample < offset, shown))
            if reply.unended:
                marks.append(Mark(at - offset, ends=True))
            reply.unended = shown
            marks.append(Mark(at - offset, sentence=motion if shown else None, text=sentence.text))
        clearance = ACCENT_CLEARANCE_S * SAMPLE_RATE
        remaining = []
        for accent in reply.accents:
            if accent.sample >= end:
                remaining.append(accent)
            elif (accent.sample >= offset and not moving
                  and not (accent.kind == "question" and reply.terminal == "question")
                  and all(abs(accent.sample - item.sample) >= clearance for item in reply.scheduled[-3:])):
                marks.append(Mark(accent.sample - offset, accent=accent.kind))
                reply.accented += 1
        reply.accents = remaining
        return sorted(marks, key=lambda mark: (mark.sample, not mark.ends, mark.accent is not None))

    def _due(self, reply: _Reply):
        for sentence in reply.sentences:
            if sentence.index not in reply.committed:
                yield sentence
        pending = reply.cutter.pending()
        if pending is not None and pending.index not in reply.committed:
            yield pending

    def _add(self, response_id: Optional[str], sentences: List[Sentence]) -> None:
        reply = self.reply(response_id)
        for sentence in sentences:
            reply.sentences.append(sentence)
            motion = classify_sentence(sentence.text)
            if sentence.index in reply.committed:
                if (sentence.index in reply.partial and reply.scheduled[-1].index == sentence.index
                        and motion.expression is not reply.expressions[-1]):
                    reply.revision = (sentence.index, motion)
                continue
            reply.motions[sentence.index] = motion
            if (self._library is not None and getattr(self._library, "library_available", False)
                    and motion.expression is not Expression.NEUTRAL):
                reply.library[sentence.index] = asyncio.create_task(self._generate(motion.expression))

    async def _generate(self, expression: Expression):
        try:
            return await self._library.generate(expression.value, 2.5)
        except CapabilityUnavailable:
            return None
        except Exception:
            log.exception("Realtime sentence motion failed")
            return None

    def _motion(self, reply: _Reply, sentence: Sentence) -> SentenceMotion:
        motion = reply.motions.get(sentence.index) or classify_sentence(sentence.text)
        task = reply.library.pop(sentence.index, None)
        if task is not None:
            if task.done() and not task.cancelled() and task.result() is not None:
                motion = replace(motion, motion=task.result())
            else:
                task.cancel()
        return motion

    def _forget(self, response_id: Optional[str]) -> None:
        reply = self._replies.pop(response_id, None)
        if reply is not None:
            for task in reply.library.values():
                task.cancel()
