# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Hands-free Muse conversations through Reachy Mini's own audio hardware."""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import itertools
import logging
import math
import re
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Awaitable, Callable

from reachy_muse import __version__
from reachy_muse import expression_plan as plan
from musegadget.link_client import DeviceDescription, LinkSession, Outcome, acknowledgement
from musegadget.muse_turn import AgentStatus, MuseTurn, ReplayScope, Reply, TaskFinished, TurnEvent
from reachy_muse.contract.commands import COMMAND_SPECS
from reachy_muse.contract.expressions import Expression
from reachy_muse.contract.prompts import SETUP_REQUEST, STOP_REQUEST, muse_request, voice_context
from reachy_muse.contract.replies import ReplyProtocolError, ReplyRevisionError, ReplyStyle, SentenceStream
from reachy_muse.capabilities import WAKE_CUE, Backends, EndedTurn, Endpoint, Heard, LocalReply, SpokenLine
from reachy_muse.device_command import COMMANDS, VOLUME_FAILED, DeviceCommand, WakeWindow, executable
from reachy_muse.follow_up import (FollowUp, classify_follow_up, is_echo, request_topic,
                                   too_short_for_follow_up)
from reachy_muse.contract.intent import Intent, maybe_chat
from reachy_muse.morning_briefing import MorningRoutine, briefing_lines
from reachy_muse.recent import Recent
from reachy_muse.motion.moments import Moment, MomentDirector
from reachy_muse.progress import BackendStatus, ProgressPlan, ProgressUpdate, backend_status_from_event
from musegadget.service import Service
from reachy_muse.speech_playback import PCMPlayer, SpeechPlayback
from reachy_muse.turn_timeline import TurnTimeline

log = logging.getLogger(__name__)
REPLY_TIMEOUT_S = 180.0
SPEECH_TIMEOUT_S = 180.0
# An acknowledgement starts only if Muse has sent no reply text this long after dispatch.
ACK_GRACE_S = 1.0
ECHO_TAIL_S = 0.35
PENDING_TURN_LIMIT = 8
INPUT_OVERFLOW_CUE = "My question queue is full. Please repeat that after I finish."
TRANSCRIPTION_RETRY_CUE = "I missed part of that. Please say it again."
LATE_ANSWER_S = 20.0
ANSWER_HOLD_LIMIT_S = 8.0
CANCEL_ACKNOWLEDGEMENT = "Okay, I'll drop that."
# Muse may answer Reachy's stop request without naming it as the parent; such messages
# this soon after a stop are taken as that answer, not the next request's.
STOP_REPLY_WINDOW_S = 5.0
BUSY_CUE = "One at a time, please."
# The daemon plays a sound at a new volume, outside Reachy's own echo gate.
VOLUME_SOUND_S = 1.0
_SPEECH_SERIALS = itertools.count(1)
# What Reachy says when the wake phrase alone wakes it, the same words as "wake up".
WAKE_GREETING = COMMANDS[DeviceCommand.WAKE_UP].reply


def strip_wake_prefix(text: str, phrase: str) -> str:
    """Strip one exact leading wake phrase, allowing ASR punctuation between words."""
    tokens = phrase.split()
    if not tokens:
        return text.strip()
    prefix = r"[\s,!.?:;-]*".join(re.escape(token) for token in tokens)
    return re.sub(r"^\s*" + prefix + r"\b[\s,!.?:;-]*", "", text,
                  count=1, flags=re.IGNORECASE).strip()


def question_after_wake(text: str, phrase: str) -> str | None:
    """Discard pre-wake words; None means ASR did not confirm the invocation."""
    prefix = r"[\s,!.?:;-]*".join(re.escape(token) for token in phrase.split())
    match = re.search(r"\b" + prefix + r"\b[\s,!.?:;-]*", text, flags=re.IGNORECASE)
    return text[match.end():].strip() if match is not None else None


@dataclass(frozen=True)
class SpeechSegment:
    message_id: str
    index: int
    text: str
    expression: str | None = None


@dataclass(frozen=True)
class ProgressSegment:
    text: str
    source: str = "frame"


@dataclass(frozen=True)
class BackendStatusSegment:
    status: BackendStatus


class TurnOutcome(Enum):
    EMPTY = "empty"
    WAKE_CUE = "wake_cue"
    ACCEPTED = "accepted"


class _SpeechSuperseded(Exception):
    """An unheard sentence was replaced before its audio could be queued."""


@dataclass(frozen=True)
class _CapturedAudio:
    samples: object
    discard: bool = False
    gap: bool = False
    gap_reason: str | None = None


@dataclass(frozen=True)
class _RecordedTurn:
    ended: EndedTurn
    endpoint_at: float
    speech_end_at: float
    heard_over: tuple[str, ...]    # Reachy's own lines that were audible while the user spoke


@dataclass(frozen=True)
class _QueuedTurn:
    ended: EndedTurn
    wake_strip_required: bool
    wake_epoch: int
    timeline: TurnTimeline
    heard_over: tuple[str, ...]


@dataclass(frozen=True)
class _SpeechJob:
    message_id: str | None
    text: str
    expression: str | None = None
    prepared_stream: object = None
    is_current: Callable[[], bool] | None = None
    on_start: Callable[[], None] | None = None
    started: float | None = None
    state: str = "speaking"
    role: str | None = None
    serial: int = field(default_factory=lambda: next(_SPEECH_SERIALS))


_SPEECH_LABELS = {
    "ack": ("Reachy acknowledgement", "acknowledgement audio"),
    "progress": ("Reachy progress", "progress audio"),
    "answer": ("Muse speech", "Muse audio"),
    "notice": ("Reachy notice", "notice audio"),
}


def _line(message_id: str | None, text: str, expression: str | None, *,
          progress: bool = False, role: str | None = None) -> SpokenLine:
    role = role or ("progress" if progress else "answer" if message_id is not None else "notice")
    return SpokenLine(text, Expression.parse(expression), role, message_id)


class _CaptureBuffer:
    """Bounded microphone audio, with explicit discontinuities on loss."""

    def __init__(self, sample_rate: int, *, capacity_s: float = 1.0):
        self.limit = int(sample_rate * capacity_s)
        self.batch_limit = int(sample_rate * .1)
        self.samples = 0
        self.chunks = deque()
        self.available = asyncio.Event()
        self.error = None
        self.gap = False
        self.gap_reason = None

    def push(self, samples, *, discard: bool = False, gap: bool = False,
             gap_reason: str | None = None) -> None:
        if not len(samples):
            return
        if gap:
            self.chunks.clear()
            self.samples = 0
            self.gap = True
            self.gap_reason = gap_reason or "upstream gap"
        if len(samples) > self.limit:
            samples = samples[-self.limit:].copy()
            self.gap = True
            self.gap_reason = "oversized capture chunk"
        if self.samples + len(samples) > self.limit:
            self.chunks.clear()
            self.samples = 0
            self.gap = True
            self.gap_reason = "capture buffer overflow"
        self.chunks.append(_CapturedAudio(samples, discard))
        self.samples += len(samples)
        self.available.set()

    def fail(self, error: Exception) -> None:
        self.error = error
        self.available.set()

    async def get(self) -> _CapturedAudio | None:
        if not self.chunks and self.error is None:
            self.available.clear()
            try:
                await asyncio.wait_for(self.available.wait(), .01)
            except asyncio.TimeoutError:
                return None
        if self.error is not None:
            raise self.error
        chunk = self.chunks.popleft()
        self.samples -= len(chunk.samples)
        # Drain available chunks together without waiting for a batch to fill.
        # A discard boundary must never become authorized microphone input.
        parts = [chunk.samples]
        size = len(chunk.samples)
        while (self.chunks and size < self.batch_limit
               and self.chunks[0].discard == chunk.discard):
            following = self.chunks.popleft().samples
            parts.append(following)
            size += len(following)
            self.samples -= len(following)
        if len(parts) > 1:
            import numpy as np
            samples = np.concatenate(parts)
        else:
            samples = chunk.samples
        result = _CapturedAudio(samples, chunk.discard, self.gap, self.gap_reason)
        self.gap = False
        self.gap_reason = None
        return result


@dataclass
class _ReplyStream:
    """Reachy's reading of one Muse reply: its latest text and, for expressive JSON, the sentences decoded so far."""

    reply: Reply
    decoder: object = None
    fed: int = 0
    closed: bool = False
    index: int = 0
    spoken: int = 0
    revised: bool = False
    skip_before: int = 0
    queued: bool = False


class ReplyReader:
    """Read one Muse turn's events in Reachy's reply style.

    Status becomes a ``BackendStatusSegment``. A marker or plain reply becomes its
    message ID once it is done; an expressive JSON reply becomes its sentence and
    progress frames as they stream, one message at a time.
    """

    def __init__(self, turn: MuseTurn, style: ReplyStyle):
        self.turn = turn
        self.style = style
        self.streams: dict[str, _ReplyStream] = {}

    def read(self, events: list[TurnEvent]) -> list[
            str | SpeechSegment | ProgressSegment | BackendStatusSegment | TaskFinished]:
        queued = []
        for event in events:
            if isinstance(event, AgentStatus):
                queued.append(BackendStatusSegment(
                    backend_status_from_event(event.activity_code, event.activity_text)))
            elif isinstance(event, TaskFinished):
                queued.append(event)
            else:
                queued.extend(self._reply(event))
        return queued

    def has_reply_text(self) -> bool:
        """Muse has started its answer: words of a reply, not a progress frame."""
        return any(stream.queued or (self.style is not ReplyStyle.EXPRESSIVE_JSON and stream.reply.text.strip())
                   for stream in self.streams.values())

    def _reply(self, reply: Reply) -> list[str | SpeechSegment | ProgressSegment]:
        stream = self.streams.get(reply.message_id)
        if stream is None:
            stream = self.streams[reply.message_id] = _ReplyStream(reply)
            if self.style is ReplyStyle.EXPRESSIVE_JSON:
                stream.decoder = SentenceStream()
        stream.reply = reply
        if self.style is ReplyStyle.EXPRESSIVE_JSON:
            return self._stream_segments()
        if reply.done and reply.text.strip() and not stream.queued:
            stream.queued = True
            return [reply.message_id]
        return []

    def _stream_segments(self) -> list[SpeechSegment | ProgressSegment]:
        queued = []
        for message_id, stream in self.streams.items():
            if stream.closed:
                continue
            try:
                if stream.reply.done:
                    try:
                        sentences = stream.decoder.finish(stream.reply.text)
                    except ReplyRevisionError:
                        if stream.spoken:
                            log.warning("Muse revised a spoken reply; remaining speech for that message skipped")
                            stream.revised = True
                            sentences = []
                        else:
                            stream.skip_before = stream.index
                            stream.decoder = SentenceStream()
                            sentences = stream.decoder.finish(stream.reply.text)
                    stream.closed = True
                else:
                    sentences = stream.decoder.feed(stream.reply.streamed[stream.fed:])
                    stream.fed = len(stream.reply.streamed)
            except ReplyProtocolError:
                log.warning("Muse returned an invalid sentence frame; speech for that message stopped")
                self.turn.abandon_reply(message_id)
                stream.reply = self.turn.replies[message_id]
                stream.closed = stream.revised = True
                stream.queued = bool(stream.spoken)
                sentences = []
            for sentence in sentences:
                if sentence.kind == "progress":
                    if stream.reply.bound and not stream.reply.done:
                        queued.append(ProgressSegment(sentence.text))
                    continue
                index = stream.index
                stream.index += 1
                if sentence.text.strip() or sentence.expression:
                    queued.append(SpeechSegment(message_id, index, sentence.text, sentence.expression))
                    stream.queued = True
            if len(queued) > 32:
                raise ValueError("Muse returned too many queued speech segments")
            if not stream.reply.done:
                break
        return queued

@dataclass(eq=False)
class MuseJob:
    """One request Muse works on, from dispatch until its answer is delivered or dropped."""

    request: str | None            # the words Muse was sent; "" for audio only, None for Reachy's setup message
    style: ReplyStyle              # the reply style it was sent with, which picks its parser
    turn: MuseTurn
    started: float                 # when the voice turn that asked it began
    defer_playback: bool           # answers go through the speaker queue instead of being spoken inline
    dispatched_at: float = field(default_factory=time.monotonic)
    replies: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=32))
    intent: Intent = Intent.QUESTION
    progress: ProgressPlan | None = None
    aside: asyncio.Task | None = None  # Reachy's own line about the job: its acknowledgement or a progress update
    aside_audible: bool = False        # an acknowledgement has started playing, so it is not cut off mid-word
    answered: bool = False
    dropped: bool = False          # the user cancelled it; nothing more of it is spoken
    turned_away: bool = False      # Reachy already told the user a further request must wait
    last_expression: Expression | None = None  # picks the closing motion
    timeline: TurnTimeline = field(default_factory=TurnTimeline)
    reader: ReplyReader = field(init=False)

    def __post_init__(self) -> None:
        self.reader = ReplyReader(self.turn, self.style)


@dataclass(frozen=True)
class _HeldRequest:
    """A request heard while Muse worked on another; it is sent once that one finishes."""

    audio: bytes
    text: str | None
    wake_epoch: int
    timeline: TurnTimeline


@dataclass(eq=False)
class _BriefingSlot:
    request: str
    attended: Callable[[], bool]
    abandoned: bool = False
    result: asyncio.Future[tuple[SpokenLine, ...] | None] = field(default_factory=lambda: asyncio.get_running_loop().create_future())


class VoiceConversation:
    def __init__(self, session: LinkSession, hardware, *, backends: Backends,
                 session_id: str | None = None,
                 silence_s: float = 2.0, reply_timeout_s: float = REPLY_TIMEOUT_S,
                 wake_detector=None, wake_timeout_s: float = 10.0,
                 owns_chat: bool = False, speech_gate=None, local_chat: bool = True,
                 audio_diagnostics: bool = False, short_utterance_silence_s: float | None = None,
                 morning: MorningRoutine | None = None) -> None:
        if not math.isfinite(wake_timeout_s) or wake_timeout_s < 0:
            raise ValueError("wake timeout must be a finite, nonnegative number")
        self.session = session
        self.hardware = hardware
        self.session_id = session_id
        self.owns_chat = owns_chat
        self.silence_s = silence_s
        self.short_utterance_silence_s = short_utterance_silence_s
        self.reply_timeout_s = reply_timeout_s
        self.backends = backends
        self.local_chat = local_chat
        self._recent = Recent()
        self._prepared_speech = {}
        self._closing_speech = set()
        self.wake_detector = wake_detector
        self.speech_gate = speech_gate
        self.audio_diagnostics = audio_diagnostics
        self.wake_timeout_s = wake_timeout_s
        self._wake_deadline = None
        self._wake_at: float | None = None
        self._wake_strip_required = False
        self._wake_first_request_pending = False
        self._wake_window = WakeWindow.AWAKE
        self._hush_floor = 0
        # The style Muse was last told to answer in. A dedicated chat was set up with the
        # local style; a shared wake chat learns it from the first request.
        self._muse_style: ReplyStyle | None = (
            None if wake_detector is not None and not owns_chat else backends.local_style)
        self.job: MuseJob | None = None
        self._stop_sent_at: float | None = None
        # True while the user is mid-utterance or Reachy is still acting on what they said.
        self._hearing_open: Callable[[], bool] = lambda: False
        self._seen_sequences: set[int] = set()
        self._sequence_order = deque()
        self._muted = False
        self._echo_until = 0.0
        # Each line Reachy played recently, and when its echo has died away.
        self._audible: dict[str, float] = {}
        self._input_overflow = False
        self._turn_started = None
        self._logged_activity = None
        self._activity_log_count = 0
        self._logged_progress_status = None
        self._replay_scope = ReplayScope()
        self._playback = SpeechPlayback()
        self._speaker_lock = asyncio.Lock()
        self._speech_pending = 0
        self._audio_clear_attempted = False
        self._output_queue: asyncio.Queue[_SpeechJob] = asyncio.Queue(maxsize=32)
        self._output_busy = False
        self._output_prefetched = False
        self._planner = plan.ExpressionPlanner()
        self._muse_status: BackendStatus | None = None
        self._moments = MomentDirector(backends.motion, hardware)
        self._morning = morning
        self._briefing_slots: deque[_BriefingSlot] = deque()

    async def _plan(self, event: plan.Event) -> None:
        for cue in self._planner.on(event, time.monotonic()):
            await self._apply_cue(cue)

    async def _apply_cue(self, cue: plan.Cue) -> None:
        if isinstance(cue, plan.Gesture):
            await asyncio.to_thread(self.hardware.gesture, cue.name)
        else:
            await asyncio.to_thread(self.hardware.set_state, cue.state.value,
                                    expression=cue.expression and cue.expression.value)

    def _without_wake(self, text: str) -> str:
        phrase = self.wake_detector.phrase
        if self._wake_strip_required:
            return question_after_wake(text, phrase) or ""
        return strip_wake_prefix(text, phrase)

    def _pose_expression(self, state: str, expression: str | None) -> str | None:
        if state != "thinking" or expression is not None:
            return expression
        lean = self._planner.resting(plan.State.THINKING).expression
        return lean and lean.value

    def _has_output(self) -> bool:
        return (self._speech_pending > 0 or self._speaker_lock.locked() or self._output_busy
                or self._output_prefetched or not self._output_queue.empty())

    async def _speech_with_timeout(self, speaking) -> None:
        task = asyncio.create_task(speaking)
        started = time.monotonic()
        held = self._playback.paused_s
        try:
            while not task.done():
                elapsed = time.monotonic() - started - (self._playback.paused_s - held)
                if elapsed >= SPEECH_TIMEOUT_S:
                    raise TimeoutError("Reachy's speech playback did not complete in time")
                await asyncio.wait({task}, timeout=min(.1, SPEECH_TIMEOUT_S - elapsed))
            task.result()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _play_output(self) -> None:
        """One speaker owner and one sentence of synthesis ahead of it."""
        async def next_job():
            job = await self._output_queue.get()
            self._output_prefetched = True
            try:
                if job.prepared_stream is None and job.text:
                    job = replace(job, prepared_stream=self.backends.voice.prepare(
                        _line(job.message_id, job.text, job.expression), self.hardware.output_sample_rate))
            except BaseException:
                self._output_prefetched = False
                self._output_queue.task_done()
                raise
            return job
        async def close_job(job):
            try:
                if job.prepared_stream is not None:
                    await job.prepared_stream.aclose()
            finally:
                self._output_queue.task_done()
        following = asyncio.create_task(next_job())
        try:
            while True:
                job = await asyncio.shield(following)
                self._output_prefetched = False
                following = asyncio.create_task(next_job())
                self._output_busy = True
                try:
                    if job.serial <= self._hush_floor:
                        raise _SpeechSuperseded
                    await self._speech_with_timeout(self._speak(
                        job.message_id, text=job.text, expression=job.expression,
                        state=job.state, stream_segment=True, prepared_stream=job.prepared_stream,
                        is_current=job.is_current, on_start=job.on_start, role=job.role,
                        turn_started=job.started, speech_serial=job.serial))
                except _SpeechSuperseded:
                    pass
                finally:
                    await close_job(job)
                    self._output_busy = False
                if self._output_queue.empty() and not following.done():
                    if self._wake_deadline is not None:
                        self._wake_deadline = time.monotonic() + (self.wake_timeout_s or 10.0)
                    await self._plan(plan.OutputIdle(user_speaking=self._playback.user_speaking,
                                                     turn_open=self.job is not None,
                                                     wake_open=self._wake_deadline is not None))
        finally:
            if not following.done():
                following.cancel()
            result, = await asyncio.gather(following, return_exceptions=True)
            if isinstance(result, _SpeechJob):
                await close_job(result)
            while not self._output_queue.empty():
                await close_job(self._output_queue.get_nowait())
            self._output_prefetched = False

    def _input_blocked(self) -> bool:
        """Unknown audio routes suppress input only while speaker audio can echo."""
        if getattr(self.hardware, "echo_cancelled_input", False) is True:
            return False
        return self._muted or time.monotonic() < self._echo_until

    async def run(self) -> None:
        await self.session.registered.wait()
        subscriber = asyncio.create_task(self._subscribe())
        tasks = {subscriber}
        try:
            await self.warm_up()
            microphone = asyncio.create_task(self._microphone())
            moments = asyncio.create_task(self._moments.keep_ready())
            morning = asyncio.create_task(self._morning.run(self)) if self._morning is not None else None
            watched = {subscriber, microphone} | ({morning} if morning is not None else set())
            tasks.update(watched, {moments})
            done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            raise ConnectionError("Muse voice stream ended")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._moments.close()
            await asyncio.to_thread(self.hardware.clear_audio)

    async def warm_up(self) -> None:
        rate = self.hardware.output_sample_rate
        warming = [asyncio.create_task(self.backends.voice.warm(rate))]
        if self.backends.progress_voice is not None:
            warming.append(asyncio.create_task(self.backends.progress_voice.warm(rate)))
        group = asyncio.gather(*warming)
        try:
            await asyncio.shield(group)
        finally:
            for task in warming:
                task.cancel()
            cleanup = asyncio.gather(group, *warming, return_exceptions=True)
            cancelled = False
            while True:
                try:
                    await asyncio.shield(cleanup)
                    break
                except asyncio.CancelledError:
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError

    async def _subscribe(self) -> None:
        subscription = self.session.subscribe_chat(self.session_id)
        try:
            async for event in subscription:
                seq = event.get("seq")
                if isinstance(seq, int) and seq > 0:
                    if seq in self._seen_sequences:
                        continue
                    self._seen_sequences.add(seq)
                    self._sequence_order.append(seq)
                    if len(self._sequence_order) > 2048:
                        self._seen_sequences.discard(self._sequence_order.popleft())
                job = self.job
                if job is not None:
                    replies = job.turn.replies
                    had_message = bool(replies)
                    had_text = any(reply.text.strip() for reply in replies.values())
                    completed_before = {message_id for message_id, reply in replies.items() if reply.done}
                    await self._queue_replies(job.reader.read(job.turn.feed(event)), job)
                    self._log_backend_activity()
                    if self._turn_started is not None:
                        now = time.monotonic()
                        elapsed = now - self._turn_started
                        for message_id, reply in replies.items():
                            if reply.done and message_id not in completed_before:
                                job.timeline.mark("reply_done", now)
                                log.info("Muse completed a reply message %.1fs after the voice turn", elapsed)
                        if not had_message and replies:
                            log.info("Muse first assistant event arrived %.1fs after the voice turn", elapsed)
                        if not had_text and any(reply.text.strip() for reply in replies.values()):
                            job.timeline.mark("first_text", now)
                            log.info("Muse first reply text arrived %.1fs after the voice turn", elapsed)
                elif self.owns_chat:
                    self._replay_scope.observe_idle(event, self.session_id)
        finally:
            await subscription.aclose()

    def _log_backend_activity(self) -> None:
        code = self.job.turn.activity_code if self.job is not None else None
        if code is not None and code != self._logged_activity:
            self._logged_activity = code
            if self._activity_log_count < 32:
                elapsed = time.monotonic() - self._turn_started if self._turn_started is not None else 0
                log.info("Muse backend activity %s %.1fs after the voice turn", code, elapsed)
                self._activity_log_count += 1

    async def _vary_thinking_pose_if_cooled_down(self) -> None:
        status = self._muse_status
        if status is not None:
            await self._plan(plan.MuseStatus(
                status.phase, status.activity.current_text if status.activity else None,
                output=self._has_output()))

    async def _queue_replies(self, replies, job: MuseJob) -> None:
        pending = []
        while not job.replies.empty():
            pending.append(job.replies.get_nowait())
        status_arrived = False
        for reply in (*pending, *replies):
            if isinstance(reply, BackendStatusSegment):
                self._muse_status = reply.status
                status_arrived = True
                job.timeline.mark("first_status")
                if job.progress is not None:
                    accepted = job.progress.set_status(reply.status, time.monotonic())
                    if accepted and reply.status != self._logged_progress_status:
                        self._logged_progress_status = reply.status
                        if self._activity_log_count < 32:
                            elapsed = (time.monotonic() - self._turn_started
                                       if self._turn_started is not None else 0)
                            log.info("Muse public progress phase=%s stage=%s %.1fs after the voice turn",
                                     reply.status.phase,
                                     reply.status.activity.current_text if reply.status.activity else "none", elapsed)
                            self._activity_log_count += 1
                continue
            if isinstance(reply, ProgressSegment):
                if job.progress is not None:
                    job.progress.offer(reply.text, time.monotonic(), source=reply.source)
                continue
            if isinstance(reply, SpeechSegment):
                stream = job.reader.streams[reply.message_id]
                if stream.revised or reply.index < stream.skip_before:
                    self._discard_prepared((reply.message_id, reply.index))
                    continue
            if job.progress is not None:
                job.progress.stop()
            try:
                job.replies.put_nowait(reply)
            except asyncio.QueueFull:
                raise ValueError("Muse returned too many queued speech segments") from None
            key = (reply.message_id, reply.index) if isinstance(reply, SpeechSegment) else None
            if (not job.defer_playback and key is not None and key not in self._prepared_speech
                    and reply.text.strip() and len(self._prepared_speech) < 2):
                prepared = self.backends.voice.prepare(
                    _line(reply.message_id, reply.text, reply.expression), self.hardware.output_sample_rate)
                if prepared is not None:
                    self._prepared_speech[key] = prepared
        if status_arrived:
            await self._vary_thinking_pose_if_cooled_down()

    def _discard_prepared(self, key) -> None:
        prepared = self._prepared_speech.pop(key, None)
        if prepared is not None:
            prepared.cancel()
            closing = asyncio.create_task(prepared.aclose())
            self._closing_speech.add(closing)
            closing.add_done_callback(self._closing_speech.discard)

    def _notice_fits(self, recorder) -> bool:
        """Whether a notice can be spoken now: the speaker queue has room and the user is not talking."""
        return not self._output_queue.full() and not recorder.active and not self._playback.user_speaking

    async def _microphone(self) -> None:
        hearing = self.backends.hearing
        await self.session.chat_subscribed.wait()
        await hearing.start()
        drain_until = time.monotonic() + 0.25
        for _ in range(1000):
            if time.monotonic() >= drain_until or await asyncio.to_thread(self.hardware.read_audio) is None:
                break
        gap_quiet_samples = 0
        recorder = None
        retry_notice_pending = False

        def transcription_failed():
            nonlocal retry_notice_pending
            retry_notice_pending = True
            log.warning("Reachy discarded an incomplete transcription; retry notice pending")

        turn_open = False

        heard_over: set[str] = set()

        def make_recorder(**options):
            nonlocal gap_quiet_samples, turn_open
            turn_open = False
            heard_over.clear()
            if recorder is not None:
                recorder.abort()
            if self._playback.user_speaking:
                gap_quiet_samples = math.ceil(self.silence_s * self.hardware.sample_rate)
            if self.short_utterance_silence_s is not None:
                options["short_utterance_silence_s"] = self.short_utterance_silence_s
            return hearing.open_turn(self.hardware.sample_rate, silence_s=self.silence_s,
                                     vad=self.speech_gate, **options)

        recorder = make_recorder()
        recorder_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reachy-recorder")
        last_slow_feed = 0.0
        async def record(sample):
            nonlocal last_slow_feed, gap_quiet_samples, turn_open
            # Keep VAD ahead of transcription/TTS work in the shared executor.
            started = time.monotonic()
            feed = asyncio.get_running_loop().run_in_executor(recorder_worker, recorder.feed, sample)
            try:
                await asyncio.shield(feed)
            except asyncio.CancelledError:
                await asyncio.gather(feed, return_exceptions=True)
                raise
            endpoint_at = time.monotonic()
            quiet_s = recorder.quiet_s
            if recorder.active or recorder.speech_active:
                heard_over.update(text for text, until in self._audible.items() if until > endpoint_at)
            heard = recorder.take()
            for _ in range(heard.failures):
                transcription_failed()
            if heard.transcript_lost:
                self._wake_strip_required = False
                self._wake_first_request_pending = False
                recorder.finish_initial_capture()
            if heard.ended is not None:
                turns_awaiting_transcript.add(heard.ended)
            # Updated with turns_awaiting_transcript, so no instant shows the user done before their turn is queued.
            turn_open = recorder.active
            speaking = recorder.speech_active
            if speaking:
                gap_quiet_samples = 0
            else:
                gap_quiet_samples = max(0, gap_quiet_samples - len(sample))
            was_speaking = self._playback.user_speaking
            self._playback.set_user_speaking(speaking or gap_quiet_samples > 0)
            if self._playback.user_speaking != was_speaking:
                await self._plan(plan.UserSpeech(speaking=self._playback.user_speaking,
                                                 output=self._has_output(),
                                                 turn_running=turn_task is not None,
                                                 turns_waiting=bool(pending_turns)))
            partial = recorder.partial
            if partial is not None:
                if self.wake_detector is not None:
                    partial = replace(partial, text=self._without_wake(partial.text))
                await self._plan(plan.Heard(partial, speech_active=speaking, output=self._has_output()))
            elapsed = time.monotonic() - started
            if elapsed > max(.1, len(sample) / self.hardware.sample_rate * 2) and started - last_slow_feed > 10:
                log.warning("Reachy microphone processing took %.3fs for %.3fs of audio",
                            elapsed, len(sample) / self.hardware.sample_rate)
                last_slow_feed = started
            if heard.ended is None:
                return None
            over = tuple(sorted(heard_over))
            heard_over.clear()
            return _RecordedTurn(heard.ended, endpoint_at, endpoint_at - quiet_s, over)

        wake = self.wake_detector
        if wake is not None:
            await asyncio.to_thread(wake.reset)
            self._wake_deadline = None
            self._wake_strip_required = False
            self._wake_first_request_pending = False
        pre_roll = deque()
        pre_roll_samples = 0
        pre_roll_limit = 3 * self.hardware.sample_rate
        def trim_pre_roll(limit: int) -> None:
            nonlocal pre_roll_samples
            while pre_roll_samples > limit:
                excess = pre_roll_samples - limit
                oldest = pre_roll.popleft()
                if len(oldest) > excess:
                    pre_roll.appendleft(oldest[excess:])
                    pre_roll_samples -= excess
                else:
                    pre_roll_samples -= len(oldest)
        wake_resampler = None
        if wake is not None and self.hardware.sample_rate != 16000:
            import av
            wake_resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)

        async def detects_wake(sample) -> tuple[bool, int | None]:
            if wake_resampler is None:
                detected = await asyncio.to_thread(wake.feed, sample)
                event = getattr(wake, "last_detection", None) if detected else None
                tail = getattr(event, "post_wake_samples", None)
                if (type(tail) is int and 0 <= tail <= pre_roll_samples
                        and getattr(event, "epoch", None) == getattr(wake, "epoch", None)
                        and getattr(event, "feed_id", None) == getattr(wake, "feed_id", None)
                        and type(getattr(event, "epoch", None)) is int
                        and type(getattr(event, "feed_id", None)) is int):
                    return detected, tail
                return detected, None
            import av
            import numpy as np
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(sample[None, :]),
                                              format="flt", layout="mono")
            frame.sample_rate = self.hardware.sample_rate
            for chunk in wake_resampler.resample(frame):
                if await asyncio.to_thread(wake.feed, chunk.to_ndarray().reshape(-1)):
                    return True, None
            return False, None

        if self._wake_window is WakeWindow.ASLEEP:
            await self._plan(plan.Sleep(True, output=self._has_output()))
        else:
            await self._plan(plan.Started())
        turn_task = (asyncio.create_task(self.turn(None))
                     if self.backends.local_style is not ReplyStyle.MUSE_VOICE
                     and wake is None and not self.owns_chat else None)
        pending_turns = deque()
        turns_awaiting_transcript = set()
        wake_epoch = 0
        active_wake_epoch = 0
        held = None
        follow_up_task = None
        self._hearing_open = lambda: turn_open or bool(turns_awaiting_transcript) or follow_up_task is not None

        async def follow_up(request, held):
            try:
                endpoint = await request.ended.endpoint()
                if endpoint is None:
                    transcription_failed()
                    return held
                request.timeline.facts["forced"] = "yes" if endpoint.forced else "no"
                if endpoint.text is not None:
                    request.timeline.mark("transcript")
                return await self._follow_up(endpoint, request, held)
            finally:
                turns_awaiting_transcript.discard(request.ended)
        async def queue_turn(recorded: _RecordedTurn) -> None:
            nonlocal recorder
            if recorded.ended.truncated:
                log.info("voice turn reached the 60-second recording limit; continuing capture")
            if len(pending_turns) < PENDING_TURN_LIMIT:
                recorded.ended.start()
                timeline = TurnTimeline()
                if self._wake_first_request_pending and self._wake_at is not None:
                    timeline.mark("wake", self._wake_at)
                timeline.mark("speech_end", recorded.speech_end_at)
                timeline.mark("endpoint", recorded.endpoint_at)
                pending_turns.append(_QueuedTurn(recorded.ended, self._wake_strip_required, wake_epoch,
                                                timeline, recorded.heard_over))
                if self._wake_strip_required:
                    recorder.finish_initial_capture()
                self._wake_strip_required = False
                self._wake_first_request_pending = False
                log.info("Reachy queued a voice turn; %d pending", len(pending_turns))
            else:
                recorded.ended.cancel()
                turns_awaiting_transcript.discard(recorded.ended)
                self._input_overflow = True
                log.warning("Reachy question queue full; saved turns retained, retry notice pending")
            if wake is not None and self.wake_timeout_s == 0:
                self._wake_deadline = None
                self._wake_strip_required = False
                self._wake_first_request_pending = False
                recorder = make_recorder()
                await asyncio.to_thread(wake.reset)
        capture = _CaptureBuffer(self.hardware.sample_rate, capacity_s=3.0)
        producer = asyncio.create_task(self._capture_microphone(capture), name="reachy-microphone-capture")
        speaker = asyncio.create_task(self._play_output(), name="reachy-speaker")
        ready = False
        blocked_episode = False
        try:
            while True:
                if speaker.done():
                    speaker.result()
                    raise ConnectionError("Reachy speaker ended")
                notice = self.backends.take_notice() if self._notice_fits(recorder) else None
                if notice is not None:
                    log.warning("Reachy's companion stopped answering; using on-robot speech")
                    self._output_queue.put_nowait(_SpeechJob(None, notice, state="listening"))
                if retry_notice_pending and not self._output_queue.full():
                    self._output_queue.put_nowait(_SpeechJob(
                        None, TRANSCRIPTION_RETRY_CUE, expression="curious", state="listening"))
                    retry_notice_pending = False
                if turn_task is not None and turn_task.done():
                    outcome = turn_task.result()
                    turn_task = None
                    if wake is not None and active_wake_epoch == wake_epoch:
                        if outcome in (TurnOutcome.WAKE_CUE, TurnOutcome.ACCEPTED):
                            self._wake_first_request_pending = False
                            # A confirmed cue also opens time for a question in wake-every-turn mode.
                            window = self.wake_timeout_s or (10.0 if outcome is TurnOutcome.WAKE_CUE else 0.0)
                            self._wake_deadline = time.monotonic() + window
                        elif (self._wake_deadline is not None
                              and time.monotonic() >= self._wake_deadline and not recorder.active):
                            # False speech must not keep authorizing new audio forever.
                            # Already admitted WAVs retain their own wake boundary.
                            self._wake_deadline = None
                            self._wake_strip_required = False
                            self._wake_first_request_pending = False
                            recorder = make_recorder()
                            pre_roll.clear()
                            pre_roll_samples = 0
                            await asyncio.to_thread(wake.reset)
                            if wake_resampler is not None:
                                import av
                                wake_resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
                            log.info("Reachy's wake window closed after empty recognition and inactivity")
                    await self._plan(plan.TurnDone(
                        turns_waiting=bool(pending_turns) or held is not None, output=self._has_output(),
                        recording=recorder.active, wake_open=wake is not None and self._wake_deadline is not None))
                if follow_up_task is not None and follow_up_task.done():
                    held = follow_up_task.result()
                    follow_up_task = None
                if turn_task is None and follow_up_task is None and held is not None:
                    request, held = held, None
                    active_wake_epoch = request.wake_epoch
                    await self._plan(plan.Working(output=self._has_output(),
                                                  user_speaking=self._playback.user_speaking))
                    turn_task = asyncio.create_task(self.turn(
                        request.audio, wake_strip_required=False, recognized_text=request.text,
                        defer_playback=request.text is not None, timeline=request.timeline))
                if turn_task is None and follow_up_task is None and pending_turns:
                    request = pending_turns.popleft()
                    active_wake_epoch = request.wake_epoch
                    await self._plan(plan.Working(output=self._has_output(),
                                                  user_speaking=self._playback.user_speaking))
                    async def process(request=request):
                        try:
                            endpoint = await request.ended.endpoint()
                            if endpoint is None:
                                transcription_failed()
                                return TurnOutcome.EMPTY
                            turns_awaiting_transcript.discard(request.ended)
                            timeline = request.timeline
                            timeline.facts["forced"] = "yes" if endpoint.forced else "no"
                            if endpoint.text is None:
                                return await self.turn(endpoint.audio,
                                    wake_strip_required=request.wake_strip_required, timeline=timeline,
                                    heard_over=request.heard_over)
                            timeline.mark("transcript")
                            log.info("Reachy transcript ready for dispatch %.3fs after endpoint",
                                     timeline.marks["transcript"] - timeline.marks["endpoint"])
                            return await self.turn(endpoint.audio,
                                wake_strip_required=request.wake_strip_required,
                                recognized_text=endpoint.text, defer_playback=True, timeline=timeline,
                                heard_over=request.heard_over)
                        finally:
                            turns_awaiting_transcript.discard(request.ended)
                    turn_task = asyncio.create_task(process())
                if (turn_task is None and follow_up_task is None and held is None and not pending_turns
                        and self._briefing_slots):
                    turn_task = asyncio.create_task(self._collect_briefing(self._briefing_slots.popleft()))
                if (turn_task is not None and follow_up_task is None and pending_turns
                        and self.job is not None and self.job.turn.acknowledged):
                    follow_up_task = asyncio.create_task(follow_up(pending_turns.popleft(), held))
                if not ready and turn_task is None:
                    log.info("Reachy speech yield ready: microphone priority, ordered background replies")
                    log.info("Reachy continuous capture ready: silence %.1fs, pending limit %d, %s",
                             self.silence_s, PENDING_TURN_LIMIT,
                             "echo-cancelled duplex" if getattr(self.hardware, "echo_cancelled_input", False) is True
                             else "playback echo gate")
                    log.info("Reachy is ready. %s", "Say Hey Muse to open the microphone." if wake is not None
                             else "Speak naturally; pause to send your turn to Muse.")
                    ready = True
                if (wake is not None and not self._input_blocked() and turn_task is None
                        and follow_up_task is None and held is None
                        and not pending_turns and not self._has_output()):
                    if (self._wake_window is WakeWindow.WOKEN and not self._wake_strip_required
                            and self._wake_deadline is not None and not recorder.active
                            and time.monotonic() >= self._wake_at + self.silence_s):
                        # The acoustic handoff yields no turn for the wake phrase alone.
                        self._wake_window = WakeWindow.AWAKE
                        self._wake_first_request_pending = False
                        self._wake_deadline = time.monotonic() + (self.wake_timeout_s or 10.0)
                        await self._announce(WAKE_GREETING, started=None, defer_playback=True)
                    if self._wake_deadline is not None and (
                            self._wake_window in (WakeWindow.DISMISSED, WakeWindow.ASLEEP) or (
                            time.monotonic() >= self._wake_deadline and not recorder.active)):
                        if self._wake_window is not WakeWindow.ASLEEP:
                            self._wake_window = WakeWindow.AWAKE
                        self._wake_deadline = None
                        self._wake_strip_required = False
                        self._wake_first_request_pending = False
                        recorder = make_recorder()
                        await asyncio.to_thread(wake.reset)
                        if wake_resampler is not None:
                            import av
                            wake_resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
                        await self._plan(plan.WakeClosed())
                        log.info("Reachy's wake window closed after inactivity")
                captured = await capture.get()
                if captured is None:
                    continue
                sample = captured.samples
                if captured.gap:
                    recorder = make_recorder()
                    pre_roll.clear()
                    pre_roll_samples = 0
                    if self._wake_deadline is not None:
                        # The acoustic invocation still authorizes fresh audio.
                        # All audio requiring a pre-wake text boundary was discarded.
                        self._wake_strip_required = False
                    if wake is not None:
                        await asyncio.to_thread(wake.reset)
                    if wake_resampler is not None:
                        import av
                        wake_resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
                    await self._plan(plan.CaptureGap(wake_open=self._wake_deadline is not None,
                                                     muted=self._muted, turn_running=turn_task is not None))
                    log.warning("Reachy microphone capture lost continuity (%s); speech detectors reset; %s",
                                captured.gap_reason or "upstream gap",
                                "microphone remains open" if self._wake_deadline is not None else "waiting for wake")
                if captured.discard or self._input_blocked():
                    # Unqualified speaker paths cannot stitch user speech across echo.
                    if not blocked_episode:
                        recorder = make_recorder()
                        pre_roll.clear()
                        pre_roll_samples = 0
                        if self._wake_first_request_pending:
                            self._wake_deadline = None
                            self._wake_strip_required = False
                            self._wake_first_request_pending = False
                        if wake is not None:
                            await asyncio.to_thread(wake.reset)
                        if wake_resampler is not None:
                            import av
                            wake_resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
                        blocked_episode = True
                    continue
                blocked_episode = False
                if wake is not None:
                    now = time.monotonic()
                    if self._wake_deadline is None:
                        pre_roll.append(sample.copy())
                        pre_roll_samples += len(sample)
                        trim_pre_roll(pre_roll_limit)
                        detected, wake_tail = await detects_wake(sample)
                        if not detected:
                            continue
                        wake_epoch += 1
                        self._wake_at = now
                        self._wake_deadline = now + (self.wake_timeout_s or 10.0)
                        self._wake_strip_required = wake_tail is None
                        self._wake_first_request_pending = True
                        if self._wake_window is WakeWindow.ASLEEP:
                            await self._wake_up(WakeWindow.WOKEN)
                        if self._has_output() or self._muted:
                            self._playback.set_user_speaking(True)
                        await self._plan(plan.WakeHeard())
                        await asyncio.to_thread(wake.reset)
                        log.info("Reachy heard its wake phrase; microphone open")
                        if wake_tail is None:
                            # Untimed detectors retain the exact text boundary guard.
                            recorder = make_recorder(initial_min_s=.06)
                        else:
                            # The acoustic detector supplies a fresh cut after the
                            # keyword. Discard all earlier audio and capture only
                            # the request; do not transcribe the invocation again.
                            trim_pre_roll(wake_tail)
                            recorder = make_recorder()
                            log.info("Wake handoff retained %.3fs of post-keyword audio; request capture open",
                                     wake_tail / self.hardware.sample_rate)
                        wav = None
                        for captured in pre_roll:
                            completed = await record(captured)
                            while completed is not None:
                                if wake_tail is not None:
                                    await queue_turn(completed)
                                    if self.wake_timeout_s == 0:
                                        break
                                else:
                                    if wav is not None:
                                        wav.ended.cancel()
                                        turns_awaiting_transcript.discard(wav.ended)
                                    wav = completed
                                # Drain any same-buffer speech after an older pre-roll endpoint.
                                completed = await record(captured[:0])
                            if wake_tail is not None and self.wake_timeout_s == 0 and self._wake_deadline is None:
                                break
                        if recorder.active:
                            if wav is not None:
                                wav.ended.cancel()
                                turns_awaiting_transcript.discard(wav.ended)
                            wav = None
                        pre_roll.clear()
                        pre_roll_samples = 0
                        if wake_tail is not None or wav is None:
                            continue
                    else:
                        wav = await record(sample)
                    if wav is None:
                        continue
                else:
                    active_before = recorder.active
                    wav = await record(sample)
                    if recorder.active != active_before:
                        await self._plan(plan.Recording(active=recorder.active,
                                                        turn_running=turn_task is not None))
                if wav is None:
                    continue
                while wav is not None:
                    await queue_turn(wav)
                    if wake is not None and self.wake_timeout_s == 0:
                        break
                    # Retain resampler/VAD state and drain endpoints in this SDK chunk.
                    wav = await record(sample[:0])
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
            recorder.abort()
            self._hearing_open = lambda: False
            for task in (turn_task, follow_up_task):
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            for ended in turns_awaiting_transcript:
                ended.cancel()
            await asyncio.gather(*(ended.endpoint() for ended in turns_awaiting_transcript), return_exceptions=True)
            speaker.cancel()
            await asyncio.gather(speaker, return_exceptions=True)
            self._playback.set_user_speaking(False)
            recorder_worker.shutdown(wait=True)

    async def _capture_microphone(self, capture: _CaptureBuffer) -> None:
        """Drain the SDK independently of keyword inference and conversation work."""
        from reachy_muse.hardware import ReachyHardwareError
        last_sample = time.monotonic()
        previous_duration = None
        try:
            while True:
                muted_at_start = self._input_blocked()
                read = asyncio.create_task(asyncio.to_thread(self.hardware.read_audio))
                try:
                    sample = await asyncio.shield(read)
                except asyncio.CancelledError:
                    await asyncio.gather(read, return_exceptions=True)
                    raise
                now = time.monotonic()
                if sample is None or not len(sample):
                    if now - last_sample > 15:
                        raise ReachyHardwareError("Reachy microphone supplied no audio for 15 seconds")
                    await asyncio.sleep(.002)
                    continue
                duration = len(sample) / self.hardware.sample_rate
                # SDK audio has no PTS. A long wall-time stall is a conservative gap signal.
                gap = previous_duration is not None and now - last_sample > max(duration, previous_duration) + .25
                previous_duration = duration
                last_sample = now
                capture.push(sample, discard=muted_at_start or self._input_blocked(), gap=gap,
                             gap_reason="SDK read or capture scheduling stall" if gap else None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            capture.fail(exc)

    async def turn(self, wav: bytes | None, *, wake_strip_required: bool | None = None,
                   recognized_text: str | None = None, defer_playback: bool = False,
                   timeline: TurnTimeline | None = None, heard_over: tuple[str, ...] = ()) -> TurnOutcome:
        if wav is None and self.wake_detector is not None:
            return TurnOutcome.EMPTY
        timeline = timeline or TurnTimeline()
        woken = self._wake_window is WakeWindow.WOKEN
        if woken:
            self._wake_window = WakeWindow.AWAKE
        started = self._turn_started = time.monotonic()
        log.info("Reachy voice turn started")
        turn_cancelled = False
        completed = False
        opening = None
        local_deferred = False
        try:
            await self._plan(plan.TurnStarted(output=self._has_output(),
                                              user_speaking=self._playback.user_speaking))
            text = None
            if wav is not None:
                text = await self._hear(wav, recognized_text, wake_strip_required, started=started,
                                        defer_playback=defer_playback, heard_over=heard_over, woken=woken)
                if isinstance(text, TurnOutcome):
                    completed = True
                    return text
                if text is not None and await self._obey(text, started, timeline) is not None:
                    completed = True
                    return TurnOutcome.ACCEPTED
            style = self.backends.reply_style()
            if text is not None and not self._has_output() and not self._playback.user_speaking:
                heard = self._recent.heard(text, may_chat=self.local_chat and maybe_chat(text))
                opening = asyncio.create_task(self.backends.narrator.acknowledge(heard))
                if heard.may_chat:
                    opener = await opening
                    if isinstance(opener, LocalReply):
                        local_deferred = defer_playback
                        await self._chat_locally(heard, opener, started=started, defer_playback=defer_playback,
                                                 timeline=timeline)
                        self._moments.finish_turn(lambda: not self._has_output())
                        completed = True
                        return TurnOutcome.ACCEPTED
            job = await self._dispatch(wav, text, started=started, defer_playback=defer_playback,
                                       timeline=timeline, opening=opening, style=style)
            outcome = await self._deliver(job)
            if job.dropped:
                self._moments.stop()
            else:
                self._moments.finish_turn(lambda: not self._has_output())
            completed = True
            return outcome
        except asyncio.CancelledError:
            turn_cancelled = True
            raise
        finally:
            if opening is not None and not opening.done():
                opening.cancel()
                await asyncio.gather(opening, return_exceptions=True)
            if not completed:
                self._moments.stop()
            job = self.job
            try:
                if job is not None:
                    try:
                        await self._end_asides(job, let_playing_finish=False, turn_cancelled=turn_cancelled)
                    finally:
                        job.progress = None
            finally:
                prepared, self._prepared_speech = self._prepared_speech, {}
                await asyncio.gather(*(audio.aclose() for audio in prepared.values()), return_exceptions=True)
                await asyncio.gather(*self._closing_speech, return_exceptions=True)
                self._closing_speech.clear()
                if job is not None:
                    job.turn.retire()
                if not local_deferred and (job is None or not job.answered):
                    timeline.report()
                self.job = None
                try:
                    # Deferred output owns its cleanup; backend failures cannot
                    # pause an earlier reply's shared GStreamer pipeline.
                    if (not defer_playback and (not completed
                            or getattr(self.hardware, "echo_cancelled_input", False) is not True)):
                        await self._clear_idle_audio()
                except Exception:
                    if not turn_cancelled:
                        raise
                finally:
                    log.info("Reachy voice turn ended")

    async def _chat_locally(self, heard: Heard, reply: LocalReply, *, started: float,
                            defer_playback: bool, timeline: TurnTimeline) -> None:
        def audible() -> None:
            timeline.mark("chat_audio")
            timeline.report()
        for index, line in enumerate(reply.lines):
            expression = line.expression.value if line.expression else None
            on_start = audible if index == 0 else None
            if defer_playback:
                await self._output_queue.put(_SpeechJob(None, line.text, expression=expression, on_start=on_start,
                                                        started=started, state="listening", role="answer"))
            else:
                await self._speak(None, text=line.text, expression=expression, role="answer", state="listening",
                                  stream_segment=index < len(reply.lines) - 1, on_start=on_start)
        self._recent.chatted(heard.text, " ".join(line.text for line in reply.lines))
        self._recent.asked(heard.text)

    async def _announce(self, text: str, *, started: float | None, defer_playback: bool) -> None:
        if defer_playback:
            await self._output_queue.put(_SpeechJob(None, text, started=started, state="listening"))
        else:
            await self._speak(None, text=text, state="listening", stream_segment=True)

    async def _hear(self, wav: bytes, recognized_text: str | None, wake_strip_required: bool | None, *,
                    started: float, defer_playback: bool,
                    heard_over: tuple[str, ...] = (), woken: bool = False) -> str | None | TurnOutcome:
        """The request's words with the wake phrase removed; None when Muse hears only the audio.

        The wake phrase alone gets "Yes?", or the wake-up greeting when it just woke Reachy.
        """
        cue = WAKE_GREETING if woken else WAKE_CUE
        if not (self.backends.hearing.transcribes or recognized_text is not None):
            return None
        text = (recognized_text if recognized_text is not None
                else await self.backends.hearing.transcribe(wav))
        strip_required = (self._wake_strip_required if wake_strip_required is None
                          else wake_strip_required)
        if wake_strip_required is None:
            self._wake_strip_required = False
        if not text.strip():
            log.info("Speech recognition outcome empty")
            if self.wake_detector is not None and strip_required:
                self._moments.cue(Moment.GREETING)
                await self._announce(cue, started=started, defer_playback=defer_playback)
                return TurnOutcome.WAKE_CUE
            log.info("No words recognized in microphone input; listening again")
            return TurnOutcome.EMPTY
        if self.wake_detector is not None:
            if strip_required:
                question = question_after_wake(text, self.wake_detector.phrase)
                # The wake detector confirms the acoustic trigger, but ASR must identify its boundary.
                log.info("Wake transcription outcome %s", "missing_boundary" if question is None
                         else "request" if question else "wake_only")
                text = question or ""
            else:
                text = strip_wake_prefix(text, self.wake_detector.phrase)
            if not text:
                self._moments.cue(Moment.GREETING)
                await self._announce(cue, started=started, defer_playback=defer_playback)
                return TurnOutcome.WAKE_CUE
        # "Stop" or "be quiet" said over Reachy's own line is a barge-in even when the line used those words.
        if is_echo(text, heard_over) and classify_follow_up(text) not in (FollowUp.SILENCE, FollowUp.CANCEL):
            log.info("Reachy ignored %d characters heard over its own speech that only repeat it", len(text))
            return TurnOutcome.EMPTY
        log.info("Recognized %d characters from Reachy's microphone", len(text))
        await self._plan(plan.Working(output=self._has_output(),
                                      user_speaking=self._playback.user_speaking))
        return text

    async def _dispatch(self, wav: bytes | None, text: str | None, *, started: float,
                        defer_playback: bool, timeline: TurnTimeline, opening=None,
                        style: ReplyStyle | None = None) -> MuseJob:
        """Send one request to Muse; the returned job is ``self.job`` until its answer is delivered."""
        style = style or self.backends.reply_style()
        job = self.job = MuseJob(
            None if wav is None else text or "", style,
            MuseTurn(self.session_id, owns_chat=self.owns_chat, replay_scope=self._replay_scope,
                     ignore_parentless_until=(0.0 if self._stop_sent_at is None
                                              else self._stop_sent_at + STOP_REPLY_WINDOW_S)),
            started, defer_playback, timeline=timeline)
        self._logged_activity = None
        self._activity_log_count = 0
        self._logged_progress_status = None
        self._muse_status = None
        options = {}
        if style is ReplyStyle.MUSE_VOICE:
            options["output_modality"] = "voice"
        if (text is not None and opening is not None and not self._has_output()
                and not self._playback.user_speaking):
            job.aside = asyncio.create_task(self._acknowledge(job, opening))
        elif opening is not None:
            opening.cancel()
        note, count = self._recent.chat_note() if text is not None and wav is not None else ("", 0)
        if wav is not None:
            self._moments.cue(Moment.THINKING)
        if wav is None:
            message = self._voice_context(style) + SETUP_REQUEST[style]
            timeline.mark("dispatch")
            ack = await self.session.send_chat(message, self.session_id)
        elif text is not None:
            message = self._ensure_muse_style(style, text, note=note)
            timeline.mark("dispatch")
            ack = await self.session.send_chat(message, self.session_id, **options)
        else:
            timeline.mark("dispatch")
            ack = await self.session.send_voice(wav, self.session_id, **options)
        response = acknowledgement(ack)
        self._recent.forget_chat(count)
        if text is not None and wav is not None:
            self._recent.asked(text)
        timeline.mark("muse_ack")
        if text is not None or wav is None:
            self._muse_style = style
        if wav is not None and self.backends.progress_voice is not None:
            job.progress = self.backends.narrator.progress(job.request, time.monotonic(), job.intent)
        await self._queue_replies(job.reader.read(job.turn.acknowledge(response)), job)
        self._log_backend_activity()
        if wav is None:
            log.info("Muse accepted Reachy's expression setup")
        else:
            log.info("Muse accepted a %.1f-second voice turn", max(0, len(wav) - 44) / 32000)
        return job

    async def _end_asides(self, job: MuseJob, *, let_playing_finish: bool, keep_progress: bool = False,
                          turn_cancelled: bool = False) -> None:
        """Wait out Reachy's current line about ``job``, cancelling it unless ``let_playing_finish``
        and it is an acknowledgement already playing.

        Further progress updates stop too, unless ``keep_progress``.
        """
        if not keep_progress and job.progress is not None:
            job.progress.stop()
        if job.aside is not None:
            task, job.aside = job.aside, None
            if not (let_playing_finish and job.aside_audible):
                task.cancel()
            job.aside_audible = False
            result, = await asyncio.gather(task, return_exceptions=True)
            if isinstance(result, Exception):
                from reachy_muse.hardware import ReachyHardwareError
                if isinstance(result, ReachyHardwareError) and not turn_cancelled:
                    raise result
                log.warning("Reachy's own spoken line failed: %s", type(result).__name__)

    async def _deliver(self, job: MuseJob) -> TurnOutcome:
        """Relay Muse's progress and answer for ``job`` until Muse is done with it."""
        turn = job.turn
        started = job.started
        request_text = job.request or ""
        response_deadline = job.dispatched_at + self.reply_timeout_s
        waiting_for_segment = False
        progress_said = []

        def answer_audible(then: Callable[[], None] | None = None) -> Callable[[], None]:
            def started() -> None:
                job.timeline.mark("first_audio")
                job.timeline.report()
                if then is not None:
                    then()
            return started
        while True:
            if job.dropped:
                log.info("Reachy stopped relaying a cancelled Muse request")
                return TurnOutcome.ACCEPTED
            now = time.monotonic()
            if turn.task_finished:
                self._moments.prepare_closing(job.last_expression)
                await self._end_asides(job, let_playing_finish=True)
            if job.answered and turn.complete(now) and job.replies.empty():
                return TurnOutcome.ACCEPTED
            if (not job.answered and job.replies.empty() and turn.settled(now)
                    and not job.reader.has_reply_text()):
                await self._end_asides(job, let_playing_finish=True)
                log.warning("Muse finished without reply text; returning to listening")
                if self.backends.voice.speaks_text:
                    from reachy_muse.hardware import ReachyHardwareError
                    try:
                        await self._announce("Muse returned an empty reply. Please try again.",
                                             started=started, defer_playback=job.defer_playback)
                    except ReachyHardwareError:
                        raise
                    except Exception as exc:
                        log.warning("Empty-reply announcement failed: %s", type(exc).__name__)
                return TurnOutcome.ACCEPTED
            if now >= response_deadline:
                raise TimeoutError("Muse's voice reply did not complete in time")
            try:
                reply = await asyncio.wait_for(job.replies.get(), 0.1)
            except asyncio.TimeoutError:
                if job.aside is not None and job.aside.done():
                    await self._end_asides(job, let_playing_finish=True, keep_progress=True)
                await self._vary_thinking_pose_if_cooled_down()
                if (self._input_overflow and job.aside is None and self.backends.voice.speaks_text
                        and not self._has_output() and not self._playback.user_speaking):
                    self._input_overflow = False
                    job.aside = asyncio.create_task(self._speak(
                        None, text=INPUT_OVERFLOW_CUE, state="thinking", stream_segment=True))
                if (job.progress is not None and job.aside is None and not job.answered
                        and not self._has_output() and not self._playback.user_speaking):
                    progress = job.progress.take(time.monotonic())
                    if progress is not None:
                        log.info("Reachy selected %s progress %.1fs after the voice turn",
                                 progress.source, time.monotonic() - started)
                        job.aside = asyncio.create_task(
                            self._say_progress(job, progress, progress_said))
                if (job.style is ReplyStyle.EXPRESSIVE_JSON and not waiting_for_segment
                        and not self._has_output() and not self._playback.user_speaking):
                    await self._plan(plan.Working(output=False, user_speaking=False))
                    waiting_for_segment = True
                continue
            if isinstance(reply, TaskFinished):
                continue
            if not job.answered:
                response_deadline += await self._hold_answer(job)
                if job.dropped:
                    continue
                now = time.monotonic()
            log.info("%s next Muse reply %.1fs after the voice turn",
                     "Queueing" if job.defer_playback else "Playing", now - started)
            if isinstance(reply, SpeechSegment):
                stream = job.reader.streams[reply.message_id]
                if stream.revised or reply.index < stream.skip_before:
                    self._discard_prepared((reply.message_id, reply.index))
                    continue
                await self._end_asides(job, let_playing_finish=True)
                def is_current(stream=stream, reply=reply):
                    return (not job.dropped and not stream.revised
                            and reply.index >= stream.skip_before)
                if not is_current():
                    self._discard_prepared((reply.message_id, reply.index))
                    continue
                if not job.answered and job.request is not None:
                    await self._answer_begins(job)
                job.last_expression = Expression.parse(reply.expression) or job.last_expression
                def mark_spoken(stream=stream):
                    stream.spoken += 1
                prepared = self._prepared_speech.get((reply.message_id, reply.index))
                if job.defer_playback:
                    self._prepared_speech.pop((reply.message_id, reply.index), None)
                    queued_at = time.monotonic()
                    await self._output_queue.put(_SpeechJob(
                        reply.message_id, reply.text, reply.expression, prepared,
                        is_current, answer_audible(mark_spoken), started))
                    response_deadline += time.monotonic() - queued_at
                    await self._queue_replies([], job)
                    job.answered = True
                    continue
                speaking = self._speak(reply.message_id, text=reply.text, expression=reply.expression,
                                       stream_segment=True,
                                       prepared_stream=prepared,
                                       is_current=is_current, on_start=answer_audible(mark_spoken))
            else:
                await self._end_asides(job, let_playing_finish=True)
                if not job.answered and job.request is not None:
                    await self._answer_begins(job)
                lines = await self.backends.narrator.lines(
                    request_text, turn.replies[reply].text, job.style, message_id=reply)
                job.timeline.mark("narration_done")
                job.last_expression = next((line.expression for line in reversed(lines) if line.expression),
                                           job.last_expression)
                self._moments.prepare_closing(job.last_expression)
                if job.defer_playback:
                    queued_at = time.monotonic()
                    for line in lines:
                        await self._output_queue.put(_SpeechJob(
                            reply, line.text, line.expression.value if line.expression else None,
                            is_current=lambda: not job.dropped, on_start=answer_audible(), started=started))
                    response_deadline += time.monotonic() - queued_at
                    job.answered = True
                    continue
                prepared = None
                speaking = self._speak_lines(reply, lines, on_start=answer_audible())
            waiting_for_segment = False
            speech_started = time.monotonic()
            try:
                await self._speech_with_timeout(speaking)
            except _SpeechSuperseded:
                if isinstance(reply, SpeechSegment):
                    self._discard_prepared((reply.message_id, reply.index))
                    continue
            finally:
                # Response waiting and draining accepted audio have separate budgets.
                response_deadline += time.monotonic() - speech_started
                if prepared is not None:
                    self._prepared_speech.pop((reply.message_id, reply.index), None)
                    await prepared.aclose()
            if isinstance(reply, SpeechSegment):
                # Refill the single sentence of lookahead after the current one finishes.
                await self._queue_replies([], job)
            job.answered = True

    async def _hold_answer(self, job: MuseJob) -> float:
        """Wait until the user is not mid-utterance, at most ``ANSWER_HOLD_LIMIT_S``; returns how long it waited.

        Talk that never pauses, such as a conversation in the room, must not keep the answer forever.
        """
        since = time.monotonic()
        while self._hearing_open() and not job.dropped and time.monotonic() - since < ANSWER_HOLD_LIMIT_S:
            await asyncio.sleep(.05)
        held = time.monotonic() - since
        if held >= ANSWER_HOLD_LIMIT_S:
            log.info("Reachy released Muse's answer after holding it %.1fs while the user kept talking", held)
        elif held >= .05:
            log.info("Reachy held Muse's answer %.1fs until the user finished speaking", held)
        return held

    async def _answer_begins(self, job: MuseJob) -> None:
        self._moments.stop_cue()
        await self._plan(plan.AnswerArrived(output=self._has_output()))
        topic = request_topic(job.request or "")
        if not topic or not self.backends.voice.speaks_text or time.monotonic() - job.dispatched_at <= LATE_ANSWER_S:
            return
        # A late answer can follow other talk; name what it answers, in the user's own words.
        intro = f"About {topic}:"
        log.info("Reachy introduced an answer that arrived %.1fs after its request",
                 time.monotonic() - job.dispatched_at)
        if job.defer_playback:
            await self._output_queue.put(_SpeechJob(None, intro, is_current=lambda: not job.dropped,
                                                    started=job.started))
        else:
            await self._speak(None, text=intro, stream_segment=True)

    async def _follow_up(self, endpoint: Endpoint, request: _QueuedTurn,
                         held: _HeldRequest | None) -> _HeldRequest | None:
        """Act on an utterance heard while Muse works; returns the request now waiting behind the job.

        Cancel and added detail apply to the newest request: the held one if any, else the running job.
        """
        text = await self._hear(endpoint.audio, endpoint.text, request.wake_strip_required,
                                started=time.monotonic(), defer_playback=True, heard_over=request.heard_over)
        if isinstance(text, TurnOutcome):
            return held
        kind = FollowUp.NEW if text is None else classify_follow_up(text)
        # A one-word command ("goodnight", "louder") is obeyed before one-word asides are dropped.
        command = (await self._obey(text, request.timeline.marks["endpoint"], request.timeline)
                   if kind is FollowUp.NEW and text is not None else None)
        if command is not None:
            return None if COMMANDS[command].wake_window in (WakeWindow.DISMISSED, WakeWindow.ASLEEP) else held
        if kind in (FollowUp.NEW, FollowUp.ADD_DETAIL) and text is not None and too_short_for_follow_up(text):
            log.info("Reachy ignored a one-word follow-up while Muse worked")
            return held
        job = self.job
        running = job if job is not None and job.request is not None and not job.dropped else None
        log.info("Reachy heard a %s follow-up while Muse worked", kind.value)
        if kind is FollowUp.SILENCE:
            self._hush()
        if kind is FollowUp.SILENCE and running is not None and running.answered:
            await self._drop(running)
            return held
        if kind in (FollowUp.CANCEL, FollowUp.SILENCE):
            if kind is FollowUp.CANCEL or not self._has_output():
                await self._announce(CANCEL_ACKNOWLEDGEMENT, started=None, defer_playback=True)
            if held is None and running is not None:
                await self._drop(running)
            return None
        if kind is FollowUp.ADD_DETAIL:
            if held is not None:
                return replace(held, text=f"{held.text} {text}")
            if running is not None and not running.answered:
                await self._add_detail(running, text)
                return None
        if held is not None:
            if job is None or not job.turned_away:
                await self._announce(BUSY_CUE, started=None, defer_playback=True)
            if job is not None:
                job.turned_away = True
            return held
        await self._plan(plan.RequestHeld())
        return _HeldRequest(endpoint.audio, text, request.wake_epoch, request.timeline)

    async def _obey(self, text: str, started: float, timeline: TurnTimeline) -> DeviceCommand | None:
        allowed = executable(wake=self.wake_detector is not None, output=self._has_output(),
                             face=getattr(self.hardware, "face_tracking_enabled", False) is True,
                             volume=callable(getattr(self.hardware, "change_volume", None)))
        command = await self.backends.commands.recognize(text, allowed)
        if command is None:
            return None
        spec = COMMANDS[command]
        if spec.silence_speech:
            self._hush()
        elif spec.wake_window in (WakeWindow.DISMISSED, WakeWindow.ASLEEP):
            self._moments.stop()
        if spec.wake_window in (WakeWindow.DISMISSED, WakeWindow.ASLEEP):
            # "Stop listening" said before a sleeping Reachy's window closes leaves it asleep.
            if self._wake_window is not WakeWindow.ASLEEP:
                self._wake_window = spec.wake_window
            job = self.job
            if job is not None and job.request is not None and not job.dropped:
                await self._drop(job)
        reply = spec.reply
        if spec.volume_step:
            if await asyncio.to_thread(self.hardware.change_volume, spec.volume_step) is None:
                reply = VOLUME_FAILED
            else:
                self._echo_until = max(self._echo_until, time.monotonic() + VOLUME_SOUND_S)
        if spec.face_tracking is not None:
            await asyncio.to_thread(self.hardware.allow_face_tracking, spec.face_tracking)
        if reply is not None:
            await self._announce(reply, started=None, defer_playback=True)
        else:
            await self._apply_cue(plan.Gesture(plan.NOD))
        if spec.wake_window is WakeWindow.ASLEEP:
            await self._plan(plan.Sleep(True, output=self._has_output()))
        elif spec.wake_window is WakeWindow.AWAKE:
            await self._wake_up(WakeWindow.AWAKE)
            self._moments.cue(Moment.GREETING)
        log.info("Reachy handled a device command: %s in %.2f s", command.name, time.monotonic() - started)
        timeline.facts["command"] = command.value
        timeline.report()
        return command

    def _hush(self) -> None:
        self._moments.stop()
        self._hush_floor = next(_SPEECH_SERIALS)

    async def _wake_up(self, window: WakeWindow) -> None:
        """Leave the sleep pose; the wake phrase and "wake up" both come here.

        The motion loop lifts the head and unfolds the antennas when the state leaves sleeping.
        """
        self._wake_window = window
        await self._plan(plan.Sleep(False, output=self._has_output()))

    async def _drop(self, job: MuseJob, *, stop_motion: bool = True) -> None:
        """Stop relaying ``job``. Muse has no cancel call, so Reachy asks it in words on the same chat.

        Once the answer has started, Muse's work is done: Reachy only skips the lines not yet spoken.
        """
        job.dropped = True
        if stop_motion:
            self._moments.stop()
        if job.progress is not None:
            job.progress.stop()
        if job.answered:
            log.info("Reachy skipped the rest of an answer the user cancelled")
            return
        response = acknowledgement(await self.session.send_chat(STOP_REQUEST, self.session_id))
        self._stop_sent_at = time.monotonic()
        self._replay_scope.retire_request(response)
        log.info("Reachy dropped a Muse request and asked Muse to stop working on it")

    async def _add_detail(self, job: MuseJob, text: str) -> None:
        job.turn.expect_follow_up()
        response = acknowledgement(await self.session.send_chat(text, self.session_id))
        await self._queue_replies(job.reader.read(job.turn.acknowledge(response, follow_up=True)), job)
        log.info("Reachy sent Muse more detail for the running request")

    async def _acknowledge(self, job: MuseJob, opening) -> None:
        """Prepare an acknowledgement at once; speak it only if Muse is still silent at the grace deadline."""
        hold = await opening
        if isinstance(hold, LocalReply):
            return
        job.intent = hold.intent
        if job.progress is not None:
            job.progress.intent = hold.intent
        line = hold.line
        if line is None:
            return
        prepared = self.backends.voice.prepare(line, self.hardware.output_sample_rate)
        def audible() -> None:
            job.aside_audible = True
            job.timeline.mark("ack_audio")
        try:
            await asyncio.sleep(job.dispatched_at + ACK_GRACE_S - time.monotonic())
            if job.reader.has_reply_text() or not job.replies.empty():
                log.info("Reachy skipped its acknowledgement; Muse's answer began within %.1fs", ACK_GRACE_S)
                return
            log.info("Reachy selected a question acknowledgement %.1fs after the voice turn",
                     time.monotonic() - job.started)
            speech, prepared = prepared, None
            await self._speak(None, text=line.text, state="thinking", stream_segment=True, role=line.role,
                              prepared_stream=speech, on_start=audible)
        finally:
            if prepared is not None:
                prepared.cancel()
                await prepared.aclose()

    async def _say_progress(self, job: MuseJob, progress: ProgressUpdate, already_said: list) -> None:
        if progress.source == "waiting":
            text = progress.text
        else:
            line = await self.backends.narrator.say_progress(job.request or "", progress.text, tuple(already_said))
            if line is None:
                return
            text = line.text
            already_said.append(text)
        await self._speak(None, text=text, state="thinking", stream_segment=True,
                          voice=self.backends.progress_voice, progress=True)

    async def _speak_lines(self, message_id: str, lines, *, on_start: Callable[[], None] | None = None) -> None:
        for index, line in enumerate(lines):
            await self._speak(message_id, text=line.text,
                              expression=line.expression.value if line.expression else None,
                              stream_segment=index < len(lines) - 1, on_start=on_start)

    def _ensure_muse_style(self, style: ReplyStyle, text: str, *, note: str = "") -> str:
        """The request, prefixed with the voice setup when Muse last heard another style."""
        if style is self._muse_style:
            return muse_request(text, note=note)
        log.info("Reachy switched Muse's reply style to %s", style.value)
        return muse_request(text, note=note, context=self._voice_context(style))

    def _voice_context(self, style: ReplyStyle | None = None) -> str:
        return voice_context(
            style or self.backends.local_style,
            motion_enabled=getattr(self.hardware, "motion_enabled", True),
            antenna_mode=getattr(self.hardware, "antenna_mode", "both"),
            face_tracking_enabled=getattr(self.hardware, "face_tracking_enabled", False),
        )

    async def _speak(self, message_id: str | None, *, text: str | None = None,
                     state: str = "speaking",
                     expression: str | None = None, stream_segment: bool = False,
                     voice=None, progress: bool = False, prepared_stream=None,
                     is_current=None, on_start=None, turn_started: float | None = None,
                     role: str | None = None, speech_serial: int | None = None) -> None:
        # Direct acknowledgements/progress and deferred replies share this owner.
        # Interrupted audio is flushed before a waiting speaker acquires it.
        speech_serial = next(_SPEECH_SERIALS) if speech_serial is None else speech_serial
        self._speech_pending += 1
        try:
            async with self._speaker_lock:
                if speech_serial <= self._hush_floor:
                    raise _SpeechSuperseded
                try:
                    await self._speak_owned(message_id, text=text, state=state,
                        expression=expression,
                        stream_segment=stream_segment, voice=voice,
                        progress=progress, prepared_stream=prepared_stream,
                        is_current=is_current, on_start=on_start, turn_started=turn_started,
                        role=role)
                except BaseException as exc:
                    try:
                        await self._clear_owned_audio(propagate_cancel=False)
                    except Exception:
                        if not isinstance(exc, asyncio.CancelledError):
                            raise
                    raise
        finally:
            self._speech_pending -= 1

    async def _clear_owned_audio(self, *, propagate_cancel: bool = True) -> None:
        """Flush under speaker ownership, joining any physical clear on cancellation."""
        if self._audio_clear_attempted:
            return
        self._audio_clear_attempted = True
        task = asyncio.create_task(asyncio.to_thread(self.hardware.clear_audio))
        cancelled = False
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                cancelled = True
        self._echo_until = min(self._echo_until, time.monotonic() + ECHO_TAIL_S)
        if cancelled and propagate_cancel:
            raise asyncio.CancelledError

    async def _clear_idle_audio(self) -> None:
        # A backend turn has no authority over another turn's active playback.
        # Count awakened lock waiters too: release can precede their acquisition.
        if not self._speech_pending and not self._speaker_lock.locked():
            async with self._speaker_lock:
                await self._clear_owned_audio()

    async def _speak_owned(self, message_id: str | None, *, text: str | None = None,
                           state: str = "speaking",
                           expression: str | None = None, stream_segment: bool = False,
                           voice=None, progress: bool = False, prepared_stream=None,
                           is_current=None, on_start=None, turn_started: float | None = None,
                           role: str | None = None) -> None:
        voice = voice if voice is not None else self.backends.voice
        samples_played = 0
        hush_floor = self._hush_floor
        write_timing = {"dispatch_ms_max": 0.0, "write_ms_max": 0.0, "return_ms_max": 0.0}
        started = self._turn_started if turn_started is None else turn_started
        async def hardware_call(function, *args, **kwargs):
            task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
            cancelled = False
            while True:
                try:
                    result = await asyncio.shield(task)
                    break
                except asyncio.CancelledError:
                    if task.cancelled():
                        raise
                    # Repeated cancellation must still wait for the pending hardware command.
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError
            return result
        async def begin() -> None:
            if is_current is not None and not is_current():
                raise _SpeechSuperseded
            if progress:
                await hardware_call(self.hardware.cue_progress)
            await hardware_call(self.hardware.set_state, state, expression=self._pose_expression(state, expression))
            if is_current is not None and not is_current():
                raise _SpeechSuperseded
            self._recent.spoke(line.text)
            if on_start is not None:
                on_start()
            if started is not None:
                log.info("%s started %.1fs after the voice turn",
                         _SPEECH_LABELS[line.role][0],
                         time.monotonic() - started)
            if expression and expression not in ("nod", "shake"):
                log.info("Muse expressed %s while Reachy spoke", expression)
        async def pause() -> None:
            self._muted = False
            await hardware_call(self.hardware.set_state, "listening")
            log.info("Reachy speech paused for microphone input")
        async def resume() -> None:
            await hardware_call(self.hardware.set_state, state, expression=self._pose_expression(state, expression))
            log.info("Reachy speech resumed after microphone silence")
        async def push(samples) -> None:
            nonlocal samples_played
            if self._hush_floor != hush_floor:
                raise _SpeechSuperseded
            self._audio_clear_attempted = False
            self._muted = True
            self._echo_until = time.monotonic() + .13 + ECHO_TAIL_S
            self._audible[line.text] = self._echo_until
            if len(self._audible) > 8:
                del self._audible[min(self._audible, key=self._audible.get)]
            if self.audio_diagnostics:
                dispatched = time.monotonic()
                entered = finished = dispatched
                def measured_write():
                    nonlocal entered, finished
                    entered = time.monotonic()
                    try:
                        self.hardware.play_audio(samples)
                    finally:
                        finished = time.monotonic()
                try:
                    await hardware_call(measured_write)
                finally:
                    resumed = time.monotonic()
                    for key, elapsed in (("dispatch_ms_max", entered - dispatched),
                                         ("write_ms_max", finished - entered),
                                         ("return_ms_max", resumed - finished)):
                        write_timing[key] = max(write_timing[key], elapsed * 1000)
            else:
                await hardware_call(self.hardware.play_audio, samples)
            samples_played += len(samples)
        player = PCMPlayer(self.hardware.output_sample_rate, push, self._playback,
                           on_start=begin, on_pause=pause, on_resume=resume,
                           diagnostics=self.audio_diagnostics)
        text = text or ""
        line = _line(message_id, text, expression, progress=progress, role=role)
        if voice.speaks_text and not text:
            if expression:
                await self._express(expression)
                return
            raise ValueError("Muse returned an empty spoken response")
        if prepared_stream is not None:
            speech = prepared_stream
        else:
            speech = voice.stream(line, self.hardware.output_sample_rate)
        try:
            try:
                async for samples in speech:
                    await player.play(samples)
            finally:
                await speech.aclose()
            if samples_played == 0:
                if is_current is not None and not is_current():
                    raise _SpeechSuperseded
                raise ValueError("Muse returned no playable speech")
            await player.finish()
            if not stream_segment:
                await asyncio.sleep(.25)
                await hardware_call(self.hardware.set_state, "thinking",
                                    expression=self._pose_expression("thinking", None))
            log.info("Reachy spoke %.1f seconds of %s",
                     samples_played / self.hardware.output_sample_rate,
                     _SPEECH_LABELS[line.role][1])
        except BaseException:
            if self._muted:
                # Hold the unknown echo path closed until its owner flushes output.
                self._echo_until = math.inf
            raise
        finally:
            self._muted = False
            if self.audio_diagnostics:
                log.info("Reachy playback timing %s", {**player.metrics, **write_timing})

    async def _express(self, expression: str) -> None:
        result = await asyncio.to_thread(self.hardware.run_command, "reachy.expression",
                                        {"name": expression}, 20000)
        if result.get("ok"):
            log.info("Muse expressed %s through Reachy", expression)
        else:
            log.warning("Reachy rejected Muse expression %s: %s", expression, result.get("error"))

    async def greet(self, *, waiting: bool) -> None:
        if self._wake_window is WakeWindow.ASLEEP:
            await self._wake_up(WakeWindow.AWAKE)
        else:
            self._wake_window = WakeWindow.AWAKE
        self._open_follow_up_window()
        self._moments.cue(Moment.GREETING)
        await self._output_queue.put(_SpeechJob(None, "Good morning!", "happy", state="listening"))
        if waiting:
            await self._output_queue.put(_SpeechJob(None, "I'm getting your briefing ready.", state="listening"))

    async def fetch(self, request: str, *, attended: Callable[[], bool]) -> tuple[SpokenLine, ...] | None:
        slot = _BriefingSlot(request, attended)
        self._briefing_slots.append(slot)
        try:
            return await asyncio.shield(slot.result)
        finally:
            if slot in self._briefing_slots:
                self._briefing_slots.remove(slot)
            elif not slot.result.done():
                slot.abandoned = True

    async def hold(self, line: str) -> None:
        await self._output_queue.put(_SpeechJob(None, line, state="thinking"))

    async def brief(self, lines: tuple[SpokenLine, ...]) -> float:
        self._open_follow_up_window()
        started: list[float] = []
        hush_floor = self._hush_floor
        for line in lines:
            if self._hush_floor != hush_floor:
                break
            await self._output_queue.put(_SpeechJob(
                "morning-briefing", line.text, line.expression.value if line.expression else None,
                on_start=lambda: started.append(time.monotonic())))
        deadline = time.monotonic() + SPEECH_TIMEOUT_S * len(lines)
        while self._has_output() and time.monotonic() < deadline:
            await asyncio.sleep(.05)
        return time.monotonic() - started[0] if started else 0.0

    def _open_follow_up_window(self) -> None:
        if self.wake_detector is not None:
            self._wake_deadline = time.monotonic() + (self.wake_timeout_s or 10.0)

    async def _collect_briefing(self, slot: _BriefingSlot) -> TurnOutcome:
        lines = None
        try:
            lines = await self._briefing_reply(slot)
        except Exception as exc:
            log.warning("Reachy morning briefing: Muse request failed: %s", type(exc).__name__)
        finally:
            if not slot.result.done():
                slot.result.set_result(lines)
        return TurnOutcome.EMPTY

    async def _briefing_reply(self, slot: _BriefingSlot) -> tuple[SpokenLine, ...] | None:
        request = slot.request
        timeline = TurnTimeline()
        started = self._turn_started = time.monotonic()
        style = self.backends.reply_style()
        job = self.job = MuseJob(
            request, style,
            MuseTurn(self.session_id, owns_chat=self.owns_chat, replay_scope=self._replay_scope,
                     ignore_parentless_until=(0.0 if self._stop_sent_at is None
                                              else self._stop_sent_at + STOP_REPLY_WINDOW_S)),
            started, True, timeline=timeline)
        self._logged_activity = None
        self._activity_log_count = 0
        self._logged_progress_status = None
        self._muse_status = None
        shown = False
        try:
            timeline.mark("dispatch")
            message = request
            if style is not self._muse_style:
                message = self._voice_context(style) + "\n\n" + request
            response = acknowledgement(await self.session.send_chat(message, self.session_id))
            timeline.mark("muse_ack")
            self._muse_style = style
            await self._queue_replies(job.reader.read(job.turn.acknowledge(response)), job)
            log.info("Muse accepted Reachy's morning briefing request")
            heard: list[tuple[str, int, SpokenLine]] = []
            deadline = job.dispatched_at + self.reply_timeout_s
            while not job.dropped:
                if slot.abandoned:
                    await self._drop(job, stop_motion=False)
                    break
                if not shown and slot.attended():
                    shown = True
                    await self._plan(plan.Working(output=self._has_output(),
                                                  user_speaking=self._playback.user_speaking))
                    if not self._has_output():
                        self._moments.cue(Moment.THINKING)
                now = time.monotonic()
                if heard and job.turn.complete(now) and job.replies.empty():
                    streams = job.reader.streams
                    return tuple(line for message_id, index, line in heard
                                 if not streams[message_id].revised and index >= streams[message_id].skip_before)
                if job.turn.settled(now) and job.replies.empty() and not job.reader.has_reply_text():
                    log.warning("Muse finished the morning briefing without reply text")
                    return None
                if now >= deadline:
                    log.warning("Muse's morning briefing did not complete in time")
                    return None
                try:
                    reply = await asyncio.wait_for(job.replies.get(), 0.1)
                except asyncio.TimeoutError:
                    continue
                if isinstance(reply, SpeechSegment):
                    heard.append((reply.message_id, reply.index,
                                  _line(reply.message_id, reply.text, reply.expression)))
                elif isinstance(reply, str):
                    heard.extend((reply, 0, line) for line in briefing_lines(job.turn.replies[reply].text))
            return ()
        finally:
            if shown:
                self._moments.stop_cue()
            job.turn.retire()
            timeline.report()
            self.job = None


def _backends_required():
    raise TypeError("ReachyService requires backends")


@dataclass
class ReachyService(Service):
    session_id: str | None = None
    prepare_chat: Callable[[LinkSession, str], Awaitable[str]] | None = None
    owns_chat: bool = False
    silence_s: float = 2.0
    # Required: Python 3.9 dataclasses cannot follow inherited defaults with a field that has none.
    backends: Callable[[LinkSession], Backends] = field(default_factory=_backends_required)
    local_chat: bool = True
    audio_diagnostics: bool = False
    wake_detector: object | None = None
    wake_timeout_s: float = 10.0
    speech_gate: object | None = None
    short_utterance_silence_s: float | None = None
    morning: MorningRoutine | None = None
    _asleep: bool = field(default=False, init=False, repr=False)
    _hardware_error: Exception | None = field(default=None, init=False, repr=False)

    async def run(self) -> None:
        await super().run()
        if self._hardware_error is not None:
            raise self._hardware_error

    def describe(self) -> DeviceDescription:
        return DeviceDescription(self.identity.node_id, "Reachy Mini", __version__, COMMAND_SPECS)

    def run_command(self, name: str, params: dict, timeout_ms: int | None = None) -> dict:
        return self.executor.run_command(name, params, timeout_ms)

    async def serve(self, session: LinkSession, vm: dict, stop: asyncio.Event) -> Outcome:
        from reachy_muse.hardware import ReachyHardwareError

        hardware = self.executor
        stop_link = asyncio.Event()
        async def converse():
            await session.registered.wait()
            session_id = self.session_id
            if self.prepare_chat is not None:
                session_id = await self.prepare_chat(session, vm["vm_id"] or vm["vm_name"])
            if self.owns_chat and (not isinstance(session_id, str) or not session_id):
                raise ValueError("Reachy's dedicated side chat was not selected")
            if self.morning is not None:
                self.morning.select_chat(vm["vm_id"] or vm["vm_name"], session_id)
            voice = VoiceConversation(session, hardware, backends=self.backends(session),
                                      session_id=session_id, silence_s=self.silence_s,
                                      wake_detector=self.wake_detector, wake_timeout_s=self.wake_timeout_s,
                                      owns_chat=self.owns_chat, speech_gate=self.speech_gate,
                                      local_chat=self.local_chat, audio_diagnostics=self.audio_diagnostics,
                                      short_utterance_silence_s=self.short_utterance_silence_s,
                                      morning=self.morning)
            voice._wake_window = WakeWindow.ASLEEP if self._asleep else WakeWindow.AWAKE
            try:
                await voice.run()
            finally:
                self._asleep = voice._wake_window is WakeWindow.ASLEEP
        link_task = asyncio.create_task(session.run(stop_link))
        voice_task = asyncio.create_task(converse())
        stop_task = asyncio.create_task(stop.wait())
        outcome = Outcome.CLOSED
        try:
            done, _ = await asyncio.wait({link_task, voice_task, stop_task},
                                         return_when=asyncio.FIRST_COMPLETED)
            if stop_task in done:
                outcome = Outcome.STOPPED
            elif link_task in done:
                outcome = link_task.result()
            else:
                voice_task.result()
        except ReachyHardwareError as exc:
            self._hardware_error = exc
            self.stop()
            raise
        except Exception as exc:
            log.warning("Reachy conversation ended: %s: %s", type(exc).__name__, exc)
            await asyncio.to_thread(hardware.set_state, "error")
        finally:
            stop_link.set()
            voice_task.cancel()
            stop_task.cancel()
            await asyncio.gather(voice_task, stop_task, return_exceptions=True)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(link_task, 5)
        return outcome
