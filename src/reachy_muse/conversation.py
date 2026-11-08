# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Hands-free Muse conversations through Reachy Mini's own audio hardware."""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import CancelledError as CommandCancelled
from concurrent.futures import TimeoutError as CommandTimeout
import inspect
import itertools
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import partial
from typing import Awaitable, Callable

from reachy_muse import expression_plan as plan
from musegadget.link_client import DeviceDescription, LinkSession, Outcome, acknowledgement
from musegadget.muse_turn import AgentStatus, MuseTurn, ReplayScope, Reply, TaskFinished, TurnEvent
from reachy_muse.contract.commands import parse_command
from reachy_muse.contract.expressions import Expression
from reachy_muse.contract.prompts import STOP_REQUEST, muse_request
from reachy_muse.contract.replies import (ReplyProtocolError, ReplyRevisionError, ReplyStyle, SentenceStream,
                                          first_plain_sentence, spoken_reply)
from reachy_muse.capabilities import EMOJI, MARKUP, WAKE_CUE, Backends, CapabilityUnavailable, EndedTurn, Endpoint, Heard, LocalReply, SpokenLine
from reachy_muse.protocol import MAX_LINE_BYTES
from reachy_muse.device_command import COMMANDS, VOLUME_FAILED, DeviceCommand, WakeWindow, executable
from reachy_muse.follow_up import (FollowUp, classify_follow_up, is_echo, interrupts_playback, request_topic,
                                   too_short_for_follow_up, repeats_playback)
from reachy_muse.contract.intent import Intent, maybe_chat
from reachy_muse.morning_briefing import MorningRoutine, briefing_lines
from reachy_muse.recent import Recent
from reachy_muse.motion.moments import CLOSINGS, Moment, MomentDirector
from reachy_muse.motion.sentence import classify_sentence
from reachy_muse.progress import BackendStatus, ProgressPlan, ProgressUpdate, backend_status_from_event
from reachy_muse.pairing import PairedService
from reachy_muse.registration import reachy_description
from reachy_muse.speech_playback import PCMPlayer, PlaybackCursor, SpeechPlayback
from reachy_muse.local_speech import PreparedSpeech, SpeechRestart
from reachy_muse.turn_timeline import TurnTimeline
from reachy_muse.startup import StartupTimeline
from reachy_muse.speaker_direction import WakeDirection
from reachy_muse.presence import PresenceFusion
from reachy_muse.inbox import INBOX_GENERATION, Inbox, outcome
from reachy_muse.notifications import Notifications
from reachy_muse.wake_policy import WakeContext

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
WAITING_LEAD_S = 3.0               # synthesis of the waiting cue starts this long before it is due
CANCEL_ACKNOWLEDGEMENT = "Okay, I'll drop that."
# Muse may answer Reachy's stop request without naming it as the parent; such messages
# this soon after a stop are taken as that answer, not the next request's.
STOP_REPLY_WINDOW_S = 5.0
BUSY_CUE = "One at a time, please."
CALL_LOST_CUE = "I lost my line to Muse. Please say that again."
THINKING_AFTER_S = 0.3
CALL_COMMAND_DEADLINE_S = 0.25
CLOSING_CLIPS = {Moment.CELEBRATION: ("success1", "happy"), Moment.APOLOGY: ("sad",)}
CLOSING_QUIET_S = 0.3
# The daemon plays a sound at a new volume, outside Reachy's own echo gate.
VOLUME_SOUND_S = 1.0
_SPEECH_SERIALS = itertools.count(1)
_MICROPHONE_READER = threading.Lock()
# What Reachy says when the wake phrase alone wakes it, the same words as "wake up".
WAKE_GREETING = COMMANDS[DeviceCommand.WAKE_UP].reply


@contextlib.contextmanager
def _microphone_reader():
    from reachy_muse.hardware import ReachyHardwareError
    if not _MICROPHONE_READER.acquire(blocking=False):
        message = "Reachy microphone reader is already running; refusing a second reader"
        log.warning(message)
        raise ReachyHardwareError(message)
    try:
        yield
    finally:
        _MICROPHONE_READER.release()


async def _finish_cleanup(work):
    task = asyncio.ensure_future(work)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


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
class NarrationSegment:
    message_id: str
    version: int


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


class _CommandParams(dict):
    def __init__(self, name, params, timeout_ms):
        received_at = time.monotonic()
        command = parse_command(name, params, timeout_ms=timeout_ms)
        super().__init__(command.params)
        self.deadline = received_at + command.timeout_s
        if name == "reachy.say":
            self.serial = next(_SPEECH_SERIALS)


@dataclass(frozen=True)
class _CapturedAudio:
    samples: object
    discard: bool = False
    gap: bool = False
    gap_reason: str | None = None
    captured_at: float = field(default_factory=time.monotonic)
    wake_context: WakeContext | None = None


@dataclass(frozen=True)
class _RecordedTurn:
    ended: EndedTurn
    endpoint_at: float
    speech_end_at: float
    heard_over: tuple[str, ...]    # Reachy's own lines that were audible while the user spoke
    call_input: object = None


@dataclass(frozen=True)
class _QueuedTurn:
    ended: EndedTurn
    wake_strip_required: bool
    wake_epoch: int
    timeline: TurnTimeline
    heard_over: tuple[str, ...]
    call: object = None          # the realtime call that streamed this turn's audio to Muse


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
    on_accepting: Callable[[bool], Awaitable[None]] | None = None


_SPEECH_LABELS = {
    "ack": ("Reachy acknowledgement", "acknowledgement audio"),
    "progress": ("Reachy progress", "progress audio"),
    "answer": ("Muse speech", "Muse audio"),
    "notice": ("Reachy notice", "notice audio"),
    "cue": ("Reachy cue", "cue audio"),
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
        self.generation = 0
        self.discarded = asyncio.Event()

    async def discard_pending(self) -> None:
        self.discarded = asyncio.Event()
        self.generation += 1
        self.chunks.clear()
        self.samples = 0
        self.gap = False
        self.gap_reason = None
        if self.error is None:
            await self.discarded.wait()
        if self.error is not None:
            raise self.error

    def push(self, samples, *, discard: bool = False, gap: bool = False,
             gap_reason: str | None = None, captured_at: float | None = None,
             wake_context: WakeContext | None = None) -> None:
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
        self.chunks.append(_CapturedAudio(samples, discard, captured_at=(
            time.monotonic() if captured_at is None else captured_at), wake_context=wake_context))
        self.samples += len(samples)
        self.available.set()

    def fail(self, error: Exception) -> None:
        self.error = error
        self.available.set()
        self.discarded.set()

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
        contexts = [chunk.wake_context] if chunk.wake_context is not None else []
        size = len(chunk.samples)
        captured_at = chunk.captured_at
        while (self.chunks and size < self.batch_limit
               and self.chunks[0].discard == chunk.discard):
            following = self.chunks.popleft()
            parts.append(following.samples)
            size += len(following.samples)
            self.samples -= len(following.samples)
            captured_at = following.captured_at
            if following.wake_context is not None:
                contexts.append(following.wake_context)
        if len(parts) > 1:
            import numpy as np
            samples = np.concatenate(parts)
        else:
            samples = chunk.samples
        context = None
        if contexts:
            playback = ()
            for item in contexts:
                overlap = next((size for size in range(min(len(playback), len(item.playback)), 0, -1)
                                if playback[-size:] == item.playback[:size]), 0)
                playback += item.playback[overlap:]
            context = WakeContext(blocked=any(item.blocked for item in contexts),
                                  speaking=any(item.speaking for item in contexts),
                                  playback=playback)
        result = _CapturedAudio(samples, chunk.discard, self.gap, self.gap_reason,
                                captured_at=captured_at, wake_context=context)
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
    plain_prefix: str = ""
    remainder: str | None = None
    narration_version: int = 0
    narration_spoken: bool = False
    accepting: bool = False


class ReplyReader:
    """Read one Muse turn's events in Reachy's reply style.

    Plain replies stream a safe opening and narrate the final remainder. Marker
    replies wait for completion; expressive JSON streams sentence and progress
    frames. Messages retain their arrival order.
    """

    def __init__(self, turn: MuseTurn, style: ReplyStyle):
        self.turn = turn
        self.style = style
        self.streams: dict[str, _ReplyStream] = {}

    def read(self, events: list[TurnEvent]) -> list[
            str | SpeechSegment | NarrationSegment | ProgressSegment | BackendStatusSegment | TaskFinished]:
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

    def has_spoken_reply(self) -> bool:
        return any(stream.spoken or stream.narration_spoken for stream in self.streams.values())

    def awaiting_final_text(self) -> bool:
        return (self.style is ReplyStyle.PLAIN_SHORT
                and any(not stream.closed for stream in self.streams.values()))

    def awaiting_audio(self) -> bool:
        return any(stream.accepting for stream in self.streams.values())

    def set_audio_accepting(self, message_id: str, accepting: bool) -> list[
            SpeechSegment | NarrationSegment | ProgressSegment]:
        self.streams[message_id].accepting = accepting
        if accepting:
            return []
        if self.style is ReplyStyle.EXPRESSIVE_JSON:
            return self._stream_segments()
        return self._plain_segments() if self.style is ReplyStyle.PLAIN_SHORT else []

    def _reply(self, reply: Reply) -> list[str | SpeechSegment | NarrationSegment | ProgressSegment]:
        stream = self.streams.get(reply.message_id)
        if stream is None:
            stream = self.streams[reply.message_id] = _ReplyStream(reply)
            if self.style is ReplyStyle.EXPRESSIVE_JSON:
                stream.decoder = SentenceStream()
        stream.reply = reply
        if self.style is ReplyStyle.EXPRESSIVE_JSON:
            return self._stream_segments()
        if self.style is ReplyStyle.PLAIN_SHORT:
            return self._plain_segments()
        if reply.done and reply.text.strip() and not stream.queued:
            stream.queued = True
            return [reply.message_id]
        return []

    def narration_text(self, message_id: str) -> str:
        stream = self.streams[message_id]
        return stream.reply.text if stream.remainder is None else stream.remainder

    def _plain_segments(self) -> list[SpeechSegment | NarrationSegment]:
        queued = []
        for message_id, stream in self.streams.items():
            if stream.revised:
                continue
            if stream.accepting:
                break
            if stream.reply.done:
                remainder = stream.reply.text
                if not stream.reply.text.strip():
                    stream.closed = False
                    stream.remainder = None
                    stream.narration_version += 1
                    if not stream.spoken:
                        stream.skip_before = stream.index
                    stream.queued = bool(stream.spoken or stream.narration_spoken)
                    continue
                stream.closed = True
                if stream.plain_prefix and not stream.skip_before:
                    final, _ = spoken_reply(stream.reply.text)
                    if final.startswith(stream.plain_prefix):
                        remainder = final[len(stream.plain_prefix):].strip()
                    elif stream.spoken:
                        log.warning("Muse revised a spoken reply; remaining speech for that message skipped")
                        stream.revised = True
                        continue
                    else:
                        stream.skip_before = stream.index
                if remainder != stream.remainder:
                    if stream.narration_spoken:
                        log.warning("Muse revised a spoken reply; remaining speech for that message skipped")
                        stream.revised = True
                        continue
                    stream.remainder = remainder
                    stream.narration_version += 1
                    if remainder.strip():
                        stream.queued = True
                        queued.append(NarrationSegment(message_id, stream.narration_version))
            elif not stream.queued and stream.reply.bound:
                try:
                    prefix = first_plain_sentence(stream.reply.streamed)
                except ReplyProtocolError:
                    prefix = None  # structured replies keep the existing final narration path
                if (prefix and len(prefix.encode("utf-8")) <= MAX_LINE_BYTES
                        and not MARKUP.search(prefix) and not EMOJI.search(prefix)):
                    stream.plain_prefix = prefix
                    queued.append(SpeechSegment(message_id, stream.index, prefix.strip()))
                    stream.index += 1
                    stream.queued = True
            if not stream.reply.done:
                break
        return queued

    def _stream_segments(self) -> list[SpeechSegment | ProgressSegment]:
        queued = []
        for message_id, stream in self.streams.items():
            if stream.closed:
                continue
            if stream.accepting:
                break
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

    request: str | None            # the words Muse was sent; "" for audio only
    style: ReplyStyle              # the local parser and narration for this reply
    turn: MuseTurn
    started: float                 # when the voice turn that asked it began
    defer_playback: bool           # answers go through the speaker queue instead of being spoken inline
    dispatched_at: float = field(default_factory=time.monotonic)
    replies: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=32))
    intent: Intent = Intent.QUESTION
    progress: ProgressPlan | None = None
    waiting_speech: PreparedSpeech | None = None  # the waiting cue's audio, ready before the cue is due
    aside: asyncio.Task | None = None  # Reachy's own line about the job: its acknowledgement or a progress update
    aside_audible: bool = False        # an acknowledgement has started playing, so it is not cut off mid-word
    answered: bool = False
    dropped: bool = False          # the user cancelled it; nothing more of it is spoken
    turned_away: bool = False      # Reachy already told the user a further request must wait
    context_changed: bool = False  # another utterance arrived while this request was pending
    last_expression: Expression | None = None  # picks the closing motion
    timeline: TurnTimeline = field(default_factory=TurnTimeline)
    reader: ReplyReader = field(init=False)

    def __post_init__(self) -> None:
        self.reader = ReplyReader(self.turn, self.style)

    @property
    def backend_pending(self) -> bool:
        return not self.answered or not self.turn.complete(time.monotonic())


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
                 wake_detector=None, wake_timeout_s: float = 10.0, wave_wake: bool = False,
                 wake_policy=None, voice_verifier=None,
                 owns_chat: bool = False, speech_gate=None, local_chat: bool = True,
                 arbitration_shadow: bool = False,
                 audio_diagnostics: bool = False, short_utterance_silence_s: float | None = None,
                 morning: MorningRoutine | None = None, startup: StartupTimeline | None = None,
                 realtime=None, presence: PresenceFusion | None = None) -> None:
        if not math.isfinite(wake_timeout_s) or wake_timeout_s < 0:
            raise ValueError("wake timeout must be a finite, nonnegative number")
        self.session = session
        self._startup = startup or StartupTimeline()
        self._ready = asyncio.Event()
        self.hardware = hardware
        self.session_id = session_id
        self.owns_chat = owns_chat
        self.silence_s = silence_s
        self.short_utterance_silence_s = short_utterance_silence_s
        self.reply_timeout_s = reply_timeout_s
        self.backends = backends
        self.local_chat = local_chat
        self.arbitration_shadow = arbitration_shadow
        self._shadow_turn = 0
        self._recent = Recent()
        self._prepared_speech = {}
        self._closing_speech = set()
        self.wake_detector = wake_detector
        self.wave_wake = wave_wake
        self.wake_enabled = wake_detector is not None or wave_wake
        self._wake_direction: WakeDirection | None = None
        self._direction_audio_until = 0.0
        self._wake_policy = wake_policy
        self._wake_unknown_output_until = 0.0
        self.voice_verifier = voice_verifier
        self._voice_rejection_at = -math.inf
        self.speech_gate = speech_gate
        self.audio_diagnostics = audio_diagnostics
        self.wake_timeout_s = wake_timeout_s
        self._wake_deadline = None
        self._wake_at: float | None = None
        self._wake_strip_required = False
        self._wake_first_request_pending = False
        self._wake_window = WakeWindow.AWAKE
        self._hush_floor = 0
        self._say_hushed = False
        self._turn_active = False
        self._command_tasks: set[asyncio.Task] = set()
        self._closed = False
        self.job: MuseJob | None = None
        self._stop_sent_at: float | None = None
        # True while the user is mid-utterance or Reachy is still acting on what they said.
        self._hearing_open: Callable[[], bool] = lambda: False
        self._seen_sequences: set[int] = set()
        self._sequence_order = deque()
        self._muted = False
        self._echo_until = 0.0
        # Each line Reachy played recently, and when its echo has died away.
        self._audible: deque[tuple[str, float]] = deque(maxlen=8)
        self._input_overflow = False
        self._turn_started = None
        self._logged_activity = None
        self._activity_log_count = 0
        self._logged_progress_status = None
        self._replay_scope = ReplayScope()
        self._playback = SpeechPlayback()
        self._speaker_lock = asyncio.Lock()
        self._speaker_cursor: PlaybackCursor | None = None
        self._speech_pending = 0
        self._speech_progress_at = 0.0
        self._waiting_for_speech = False
        self._speech_source = None
        self._audio_clear_attempted = False
        self._output_queue: asyncio.Queue[_SpeechJob] = asyncio.Queue(maxsize=32)
        self._output_busy = False
        self._output_prefetched = False
        self._planner = plan.ExpressionPlanner()
        self._muse_status: BackendStatus | None = None
        self._moments = MomentDirector(backends.motion, hardware)
        self._morning = morning
        self._presence = presence or PresenceFusion()
        self._briefing_slots: deque[_BriefingSlot] = deque()
        self.realtime = realtime
        self.call = None
        self._lost_call = None
        self._call_tasks: set[asyncio.Task] = set()
        self._barging: asyncio.Task | None = None
        self._user_quiet_at: float | None = None
        self.muse_moving = 0
        self.notifications: Notifications | None = None

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
                or self._output_prefetched or not self._output_queue.empty()
                or (self.call is not None and self.call.audible))

    async def _speech_with_timeout(self, speaking) -> None:
        task = asyncio.create_task(speaking)
        progressed = time.monotonic()
        held = self._playback.paused_s
        try:
            while not task.done():
                now = time.monotonic()
                paused = self._playback.paused_s
                progressed += paused - held
                held = paused
                progressed = max(progressed, self._speech_progress_at)
                if self._waiting_for_speech:
                    progressed = max(progressed, getattr(self._speech_source, "progress_at", 0.0))
                elapsed = now - progressed
                if elapsed >= SPEECH_TIMEOUT_S:
                    raise TimeoutError("Reachy's speech playback stopped making progress")
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
                if job.text and job.state == "speaking" and job.expression is None:
                    job = replace(job, expression=classify_sentence(job.text).expression.value)
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
        playback_cursor = PlaybackCursor()
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
                        on_accepting=job.on_accepting,
                        turn_started=job.started, speech_serial=job.serial,
                        playback_cursor=playback_cursor, drain=False))
                except _SpeechSuperseded:
                    pass
                finally:
                    await close_job(job)
                if self._output_queue.empty() and not following.done():
                    await self._drain_speech(playback_cursor)
                    if not self._output_queue.empty() or following.done():
                        continue
                    self._output_busy = False
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
            self._output_busy = False

    def _input_blocked(self) -> bool:
        """Unknown audio routes suppress input only while speaker audio can echo."""
        if getattr(self.hardware, "echo_cancelled_input", False) is True:
            return False
        return self._muted or time.monotonic() < self._echo_until

    def _direction_eligible(self) -> bool:
        now = time.monotonic()
        return (self.wake_enabled
                and (self._wake_deadline is None or now >= self._wake_deadline)
                and not self._muted and now >= self._echo_until and not self._has_output())

    def _direction_poll_eligible(self) -> bool:
        now = time.monotonic()
        return (self.wake_enabled and not self._muted and now >= self._echo_until
                and not self._has_output()
                and (now < self._direction_audio_until
                     or self._wake_deadline is not None and now < self._wake_deadline))

    def _attend_wake(self, phrase_end: float | None, *, angle: float | None = None) -> None:
        """Prefer the waver's position, or freeze DoA at the invocation boundary."""
        direction = self._wake_direction
        if direction is None and angle is None:
            return
        if not self._direction_eligible():
            angle = None
        elif angle is None and direction is not None and phrase_end is not None:
            angle = direction.estimate(phrase_end)
        if direction is not None:
            direction.clear()
        if angle is None:
            log.info("Wake direction decision=no_confidence")
        else:
            self.hardware.look_at_speaker(angle)

    def _wake_capture_context(self, now):
        return WakeContext(blocked=self._input_blocked() or now < self._wake_unknown_output_until,
                           speaking=self._muted or now < self._echo_until,
                           playback=tuple(text for text, until in tuple(self._audible) if until > now - 2.5))

    async def run(self) -> None:
        await self.session.registered.wait()
        subscriber = asyncio.create_task(self._subscribe())
        tasks = {subscriber}
        try:
            microphone = asyncio.create_task(self._microphone())
            warming = asyncio.create_task(self._warm_after_ready())
            moments = asyncio.create_task(self._moments.keep_ready())
            morning = asyncio.create_task(self._morning.run(self)) if self._morning is not None else None
            notifications = asyncio.create_task(self.notifications.run()) if self.notifications is not None else None
            watched = {task for task in (subscriber, microphone, notifications, morning) if task is not None}
            tasks.update(watched, {moments, warming})
            if self.realtime is not None:
                tasks.add(asyncio.create_task(self._warm_realtime()))
            done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            raise ConnectionError("Muse voice stream ended")
        finally:
            self._closed = True
            tasks.update(self._command_tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._close_call()
            await asyncio.gather(*self._call_tasks, return_exceptions=True)
            if self.realtime is not None:
                await self.realtime.close()
            await self._moments.close()
            await asyncio.to_thread(self.hardware.clear_audio)

    async def _warm_realtime(self) -> None:
        await self._ready.wait()
        await self.realtime.warm()

    async def _warm_after_ready(self) -> None:
        await self._ready.wait()
        try:
            await self.warm_up()
        except Exception as exc:
            log.warning("Reachy fixed speech warm-up failed (%s); live synthesis remains available",
                        type(exc).__name__)

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
                if reply.expression is None:
                    reply = replace(reply, expression=classify_sentence(reply.text).expression.value)
            if isinstance(reply, NarrationSegment):
                stream = job.reader.streams[reply.message_id]
                if stream.revised or reply.version != stream.narration_version:
                    continue
            if job.progress is not None:
                job.progress.stop()
                self._discard_waiting(job)
            try:
                job.replies.put_nowait(reply)
            except asyncio.QueueFull:
                raise ValueError("Muse returned too many queued speech segments") from None
            key = (reply.message_id, reply.index) if isinstance(reply, SpeechSegment) else None
            if (not job.defer_playback and (job.answered or not job.context_changed)
                    and key is not None and key not in self._prepared_speech
                    and reply.text.strip() and len(self._prepared_speech) < 2):
                prepared = self.backends.voice.prepare(
                    _line(reply.message_id, reply.text, reply.expression), self.hardware.output_sample_rate)
                if prepared is not None:
                    self._prepared_speech[key] = prepared
        if status_arrived:
            await self._vary_thinking_pose_if_cooled_down()

    def _discard_prepared(self, key) -> None:
        self._close_prepared(self._prepared_speech.pop(key, None))

    def _discard_waiting(self, job: MuseJob) -> None:
        prepared, job.waiting_speech = job.waiting_speech, None
        self._close_prepared(prepared)

    def _close_prepared(self, prepared) -> None:
        if prepared is not None:
            prepared.cancel()
            closing = asyncio.create_task(prepared.aclose())
            self._closing_speech.add(closing)
            closing.add_done_callback(self._closing_speech.discard)

    async def _voice_rejected(self) -> None:
        if (self.voice_verifier.mode != "strict" or time.monotonic() - self._voice_rejection_at < 5
                or self._output_queue.full()):
            return
        from reachy_muse.voice_verify import rejection_cue
        self._voice_rejection_at = time.monotonic()
        async def audio():
            yield rejection_cue(self.hardware.output_sample_rate)
        self._output_queue.put_nowait(_SpeechJob(None, "", prepared_stream=audio(), state="listening", role="cue"))

    async def _microphone(self) -> None:
        hearing = self.backends.hearing
        await self.session.chat_subscribed.wait()
        self._startup.mark("chat_subscription")
        await hearing.start()
        self._startup.mark("hearing")
        with _microphone_reader():
            drain_until = time.monotonic() + 0.25
            for _ in range(1000):
                if time.monotonic() >= drain_until:
                    break
                read = asyncio.get_running_loop().run_in_executor(None, self.hardware.read_audio)
                if await _finish_cleanup(read) is None:
                    break
        gap_quiet_samples = 0
        recorder = None
        recorder_for_call = False
        retry_notice_pending = False
        voice_gate = None
        if self.voice_verifier is not None:
            from reachy_muse.voice_verify import RequestGate
            voice_gate = RequestGate(self.voice_verifier, self.hardware.sample_rate)

        def transcription_failed(call=None):
            nonlocal retry_notice_pending
            if call is not None and not call.failed:
                log.info("Reachy could not transcribe a turn Muse heard in its call; Muse answers it")
                return
            retry_notice_pending = True
            log.warning("Reachy discarded an incomplete transcription; retry notice pending")

        turn_open = False

        heard_over: set[str] = set()
        admitted_turns = deque()
        turns_awaiting_transcript = set()

        def make_recorder(*, reset_admission=True, **options):
            nonlocal gap_quiet_samples, turn_open, recorder_for_call
            turn_open = False
            heard_over.clear()
            if reset_admission:
                if voice_gate is not None:
                    voice_gate.reset()
                while admitted_turns:
                    ended = admitted_turns.popleft().ended
                    ended.cancel()
                    turns_awaiting_transcript.discard(ended)
            recorder_for_call = self.call is not None
            if recorder_for_call:
                options["commands_only"] = True
                self.call.input.reset()
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
        async def record_admitted(sample, *, send_to_call=True):
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
                if self.call is not None:
                    heard_over.update(self.call.audible_transcripts(ECHO_TAIL_S))
                heard_over.update(text for text, until in self._audible if until > endpoint_at)
            heard = recorder.take()
            for _ in range(heard.failures):
                transcription_failed(self.call)
            if heard.transcript_lost:
                self._wake_strip_required = False
                self._wake_first_request_pending = False
                recorder.finish_initial_capture()
            if heard.ended is not None:
                turns_awaiting_transcript.add(heard.ended)
            # Updated with turns_awaiting_transcript, so no instant shows the user done before their turn is queued.
            turn_open = recorder.active
            partial = recorder.partial
            if partial is not None and self.wake_detector is not None:
                partial = replace(partial, text=self._without_wake(partial.text))
            speaking = recorder.speech_active
            call_input = None
            if self.call is not None:
                call = self.call
                playback = call.audible or endpoint_at < self._echo_until or bool(heard_over)
                if (send_to_call and self._wake_window not in (WakeWindow.DISMISSED, WakeWindow.ASLEEP)
                        and self._wake_deadline is not None and (
                        endpoint_at < self._wake_deadline or recorder.active or speaking or heard.ended is not None)):
                    confirmed_before = call.input.confirmed
                    call.input.push(sample, speaking=speaking and quiet_s == 0,
                                    text=partial.text if partial else "", playback=playback,
                                    heard_over=tuple(heard_over), quiet_s=quiet_s)
                    speaking = speaking and (not playback or call.input.confirmed)
                    if call.audible and call.input.confirmed and not confirmed_before:
                        self._barge_in_call()
                if heard.ended is not None or heard.transcript_lost:
                    from reachy_muse.realtime_call import CallInput
                    call_input = call.input if playback else None
                    call.input = CallInput(call.feed, self.hardware.sample_rate)
            elif (speaking and heard_over and recorder.supports_partials
                    and not self._playback.user_speaking):
                # Hardware AEC can leave residual speech. Keep capturing it, but
                # require words distinct from our output before pausing playback.
                speaking = partial is not None and interrupts_playback(partial.text, tuple(heard_over))
            if wave_greeting is not None and (speaking or heard.ended is not None):
                wave_greeting.cancel()
            if speaking:
                gap_quiet_samples = 0
            else:
                gap_quiet_samples = max(0, gap_quiet_samples - len(sample))
            was_speaking = self._playback.user_speaking
            self._playback.set_user_speaking(speaking or gap_quiet_samples > 0)
            if was_speaking and not self._playback.user_speaking:
                self._user_quiet_at = time.monotonic()
            if self._playback.user_speaking != was_speaking:
                await self._plan(plan.UserSpeech(speaking=self._playback.user_speaking,
                                                 output=self._has_output(),
                                                 turn_running=turn_task is not None,
                                                 turns_waiting=bool(pending_turns)))
            if partial is not None and self.call is None:
                await self._plan(plan.Heard(partial, speech_active=speaking, output=self._has_output()))
            if recorder.active and self.call is None:
                await self._plan(plan.Pause(recorder.quiet_s, output=self._has_output()))
            elapsed = time.monotonic() - started
            if elapsed > max(.1, len(sample) / self.hardware.sample_rate * 2) and started - last_slow_feed > 10:
                log.warning("Reachy microphone processing took %.3fs for %.3fs of audio",
                            elapsed, len(sample) / self.hardware.sample_rate)
                last_slow_feed = started
            if heard.ended is None:
                if heard.transcript_lost:
                    heard_over.clear()
                return None
            over = tuple(sorted(heard_over))
            heard_over.clear()
            return _RecordedTurn(heard.ended, endpoint_at, endpoint_at - quiet_s, over, call_input)

        def recording_active():
            return recorder.active or (voice_gate is not None and voice_gate.active)

        async def record(sample, *, send_to_call=True):
            if voice_gate is None:
                return await record_admitted(sample, send_to_call=send_to_call)
            self.voice_verifier.realtime = self.call is not None
            frame = max(1, self.hardware.sample_rate // 50)
            for offset in range(0, max(1, len(sample)), frame):
                admitted = await asyncio.to_thread(voice_gate.feed, sample[offset:offset + frame])
                if voice_gate.rejected:
                    await self._voice_rejected()
                for start in range(0, max(1, len(admitted)), frame):
                    if len(admitted):
                        await recorder.drain()
                    ended = await record_admitted(admitted[start:start + frame], send_to_call=send_to_call)
                    if ended is not None:
                        if self.wake_enabled and self.wake_timeout_s == 0:
                            return ended
                        admitted_turns.append(ended)
            return admitted_turns.popleft() if admitted_turns else None

        wake = self.wake_detector
        if wake is not None:
            await asyncio.to_thread(wake.reset)
            if self._wake_policy is not None:
                self._wake_policy.reset()
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

        async def detects_wake(sample, context=None) -> tuple[bool, int | None]:
            async def feed(audio):
                budget = getattr(self.hardware, "vision_budget", None)
                if budget is not None:
                    budget.observe("wake", len(audio) / 16000)
                if context is not None:
                    return await self._wake_policy.feed(wake, audio, context=context,
                                                        valid=lambda: not capture.gap)
                return await asyncio.to_thread(wake.feed, audio)
            if wake_resampler is None:
                detected = await feed(sample)
                event = getattr(wake, "last_detection", None) if detected else None
                tail = getattr(event, "post_wake_samples", None)
                available = pre_roll_samples + (len(sample) if context is not None else 0)
                if (type(tail) is int and 0 <= tail <= available
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
                if await feed(chunk.to_ndarray().reshape(-1)):
                    return True, None
            return False, None

        if self._wake_window is WakeWindow.ASLEEP:
            await self._plan(plan.Sleep(True, output=self._has_output()))
        else:
            await self._plan(plan.Started())
        turn_task = None
        pending_turns = deque()
        call_recognitions = {}
        wake_epoch = 0
        active_wake_epoch = 0
        held = None
        follow_up_task = None
        self._hearing_open = lambda: (turn_open or bool(turns_awaiting_transcript) or follow_up_task is not None
                                     or (voice_gate is not None and voice_gate.active))

        async def recognize_call(recorded, call):
            from reachy_muse.streaming_transcription import StreamingTranscriptionError

            recognition = asyncio.create_task(recorded.ended.endpoint())
            try:
                remaining = recorded.endpoint_at + CALL_COMMAND_DEADLINE_S - time.monotonic()
                done, _ = await asyncio.wait({recognition}, timeout=max(0.0, remaining))
                if not done and (call.audio_sent or not call.failed):
                    raise asyncio.TimeoutError
                endpoint = await recognition
            except (asyncio.TimeoutError, CapabilityUnavailable, StreamingTranscriptionError):
                recorded.ended.cancel()
                endpoint = Endpoint(b"", "")
            finally:
                if not recognition.done():
                    recognition.cancel()
                await asyncio.gather(recognition, return_exceptions=True)
            gate = recorded.call_input
            if gate is not None and not gate.sent_speech:
                text = (endpoint.text or "") if endpoint is not None else ""
                if self.wake_detector is not None:
                    text = self._without_wake(text)
                if self.call is call and gate.finish(text, recorded.heard_over):
                    if call.audible:
                        await call.barge_in()
                else:
                    return None
            return endpoint

        async def follow_up(request, held):
            try:
                if request.ended in call_recognitions:
                    endpoint = await call_recognitions[request.ended]
                else:
                    endpoint = await request.ended.endpoint()
                if endpoint is None:
                    transcription_failed(request.call)
                    return held
                request.timeline.facts["forced"] = "yes" if endpoint.forced else "no"
                if endpoint.text is not None:
                    request.timeline.mark("transcript")
                if request.call is not None and (request.call.audio_sent or not request.call.failed):
                    await self._realtime_turn(endpoint, request)
                    return held
                return await self._follow_up(endpoint, request, held)
            finally:
                call_recognitions.pop(request.ended, None)
                turns_awaiting_transcript.discard(request.ended)
        async def queue_turn(recorded: _RecordedTurn) -> None:
            nonlocal recorder
            if recorded.ended.truncated:
                log.info("voice turn reached the 60-second recording limit; continuing capture")
            if len(pending_turns) < PENDING_TURN_LIMIT:
                if self.call is None and self._lost_call is None:
                    recorded.ended.start()
                else:
                    call_recognitions[recorded.ended] = asyncio.create_task(
                        recognize_call(recorded, self.call or self._lost_call))
                timeline = TurnTimeline()
                if self._wake_first_request_pending and self._wake_at is not None:
                    timeline.mark("wake", self._wake_at)
                timeline.mark("speech_end", recorded.speech_end_at)
                timeline.mark("endpoint", recorded.endpoint_at)
                pending_turns.append(_QueuedTurn(recorded.ended, self._wake_strip_required, wake_epoch,
                                                timeline, recorded.heard_over, self.call or self._lost_call))
                if self.realtime is not None and pending_turns[-1].call is None:
                    self.realtime.chat_turn()
                self._lost_call = None
                if self._wake_strip_required:
                    recorder.finish_initial_capture()
                self._wake_strip_required = False
                self._wake_first_request_pending = False
                if pending_turns[-1].call is None:
                    log.info("Reachy queued a voice turn; %d pending", len(pending_turns))
            else:
                recorded.ended.cancel()
                turns_awaiting_transcript.discard(recorded.ended)
                if self.call is None and self._lost_call is None:
                    self._input_overflow = True
                    log.warning("Reachy question queue full; saved turns retained, retry notice pending")
            if self.wake_enabled and self.wake_timeout_s == 0:
                self._wake_deadline = None
                self._wake_strip_required = False
                self._wake_first_request_pending = False
                recorder = make_recorder()
                if wake is not None:
                    await asyncio.to_thread(wake.reset)
        capture = _CaptureBuffer(self.hardware.sample_rate, capacity_s=3.0)
        capture_stop = threading.Event()
        producer = asyncio.create_task(self._capture_microphone(capture, stop=capture_stop),
                                       name="reachy-microphone-capture")
        speaker = asyncio.create_task(self._play_output(), name="reachy-speaker")
        direction_task = None
        if (self.wake_enabled and callable(getattr(self.hardware, "read_doa", None))
                and callable(getattr(self.hardware, "look_at_speaker", None))
                and getattr(self.hardware, "motion_enabled", True)):
            self._wake_direction = WakeDirection(self.hardware.read_doa)
            direction_task = asyncio.create_task(self._wake_direction.sample_while(self._direction_poll_eligible))
        ready = False
        blocked_episode = False
        wave_active = False
        wave_poll_at = -math.inf
        wave_operation = None
        wave_greeting = None

        async def watch_wave(active):
            nonlocal wave_active, wave_poll_at, wave_operation
            if self.wave_wake and (active != wave_active or active and time.monotonic() - wave_poll_at >= .25):
                wave_operation = asyncio.create_task(asyncio.to_thread(self.hardware.set_wave_watching, active))
                await asyncio.shield(wave_operation)
                wave_active, wave_poll_at = active, time.monotonic()
                wave_operation = None

        try:
            while True:
                if speaker.done():
                    speaker.result()
                    raise ConnectionError("Reachy speaker ended")
                if self.backends.companion is not None and self.backends.companion.take_outage():
                    log.warning("Reachy's companion stopped answering; using on-robot speech")
                if self.call is not None and self.call.done:
                    await self._call_ended(capturing=recorder.active or recorder.speech_active)
                if recorder_for_call != (self.call is not None) and not recorder.active:
                    recorder = make_recorder(reset_admission=False)
                if retry_notice_pending and not self._output_queue.full():
                    self._output_queue.put_nowait(_SpeechJob(
                        None, TRANSCRIPTION_RETRY_CUE, expression="curious", state="listening"))
                    retry_notice_pending = False
                if turn_task is not None and turn_task.done():
                    outcome = turn_task.result()
                    turn_task = None
                    if self.wake_enabled and active_wake_epoch == wake_epoch:
                        if (outcome in (TurnOutcome.WAKE_CUE, TurnOutcome.ACCEPTED)
                                and self._wake_window not in (WakeWindow.DISMISSED, WakeWindow.ASLEEP)):
                            self._wake_first_request_pending = False
                            # A confirmed cue also opens time for a question in wake-every-turn mode.
                            window = self.wake_timeout_s or (10.0 if outcome is TurnOutcome.WAKE_CUE else 0.0)
                            self._wake_deadline = time.monotonic() + window
                        elif (self._wake_deadline is not None
                              and time.monotonic() >= self._wake_deadline and not recording_active()):
                            # False speech must not keep authorizing new audio forever.
                            # Already admitted WAVs retain their own wake boundary.
                            self._wake_deadline = None
                            self._wake_strip_required = False
                            self._wake_first_request_pending = False
                            recorder = make_recorder()
                            pre_roll.clear()
                            pre_roll_samples = 0
                            self._wake_window_closed()
                            if wake is not None:
                                await asyncio.to_thread(wake.reset)
                            if wake_resampler is not None:
                                import av
                                wake_resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
                            log.info("Reachy's wake window closed after empty recognition and inactivity")
                    await self._plan(plan.TurnDone(
                        turns_waiting=bool(pending_turns) or held is not None, output=self._has_output(),
                        recording=recorder.active, wake_open=self.wake_enabled and self._wake_deadline is not None))
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
                    if request.call is None:
                        await self._plan(plan.Working(output=self._has_output(),
                                                      user_speaking=self._playback.user_speaking))
                    async def process(request=request):
                        try:
                            call = request.call
                            if request.ended in call_recognitions:
                                endpoint = await call_recognitions[request.ended]
                            else:
                                endpoint = await request.ended.endpoint()
                            if endpoint is None:
                                transcription_failed(request.call)
                                return TurnOutcome.EMPTY
                            turns_awaiting_transcript.discard(request.ended)
                            timeline = request.timeline
                            timeline.facts["forced"] = "yes" if endpoint.forced else "no"
                            if call is not None and (call.audio_sent or not call.failed):
                                return await self._realtime_turn(endpoint, request)
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
                            call_recognitions.pop(request.ended, None)
                            turns_awaiting_transcript.discard(request.ended)
                    turn_task = asyncio.create_task(process())
                if (turn_task is None and follow_up_task is None and held is None and not pending_turns
                        and self._briefing_slots):
                    turn_task = asyncio.create_task(self._collect_briefing(self._briefing_slots.popleft()))
                if (turn_task is not None and follow_up_task is None and pending_turns
                        and (pending_turns[0].call is not None
                             or self.job is not None and self.job.turn.acknowledged)):
                    follow_up_task = asyncio.create_task(follow_up(pending_turns.popleft(), held))
                if not ready and turn_task is None:
                    log.info("Reachy speech yield ready: microphone priority, ordered background replies")
                    log.info("Reachy continuous capture ready: silence %.1fs, pending limit %d, %s",
                             self.silence_s, PENDING_TURN_LIMIT,
                             "echo-cancelled duplex" if getattr(self.hardware, "echo_cancelled_input", False) is True
                             else "playback echo gate")
                    log.info("Reachy is ready. %s", "Wave to open the microphone." if self.wave_wake and wake is None
                             else "Say Hey Muse to open the microphone." if wake is not None
                             else "Speak naturally; pause to send your turn to Muse.")
                    self._startup.mark("microphone_ready")
                    self._ready.set()
                    ready = True
                if (self.wake_enabled and not self._input_blocked() and turn_task is None
                        and follow_up_task is None and held is None
                        and not pending_turns and not self._has_output()):
                    if (self._wake_window is WakeWindow.WOKEN and not self._wake_strip_required
                            and self.call is None
                            and self._wake_deadline is not None and not recording_active()
                            and time.monotonic() >= self._wake_at + self.silence_s):
                        # The acoustic handoff yields no turn for the wake phrase alone.
                        self._wake_window = WakeWindow.AWAKE
                        self._wake_first_request_pending = False
                        self._wake_deadline = time.monotonic() + (self.wake_timeout_s or 10.0)
                        await self._announce(WAKE_GREETING, started=None, defer_playback=True)
                    if self._wake_deadline is not None and (
                            self._wake_window in (WakeWindow.DISMISSED, WakeWindow.ASLEEP) or (
                            time.monotonic() >= self._wake_deadline and not recording_active()
                            and not (self.call is not None and self.call.working))):
                        if self._wake_window is not WakeWindow.ASLEEP:
                            self._wake_window = WakeWindow.AWAKE
                        self._wake_deadline = None
                        self._wake_strip_required = False
                        self._wake_first_request_pending = False
                        self._wake_window_closed()
                        recorder = make_recorder()
                        if wake is not None:
                            await asyncio.to_thread(wake.reset)
                        if wake_resampler is not None:
                            import av
                            wake_resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
                        await self._plan(plan.WakeClosed())
                        log.info("Reachy's wake window closed after inactivity")
                wave_eligible = (self.wave_wake and not (self.call is not None and self.call.working)
                                 and not self._call_tasks
                                 and not self._input_blocked() and not self._has_output()
                                 and turn_task is None and follow_up_task is None and held is None
                                 and not pending_turns and not self._briefing_slots and not recording_active())
                await watch_wave(wave_eligible)
                if self.wave_wake and wave_eligible and self.hardware.take_wave():
                    await watch_wave(False)
                    if not self.hardware.claim_wave():
                        continue
                    self._observe_presence("wave")
                    already_awake = self._wake_deadline is not None or self.call is not None
                    if already_awake:
                        await asyncio.to_thread(self.hardware.gesture, "wave")
                        log.info("Wave back cues=1")
                        continue
                    await capture.discard_pending()
                    self._attend_wake(time.monotonic(), angle=getattr(self.hardware, "wave_direction", None))
                    asleep = self._wake_window is WakeWindow.ASLEEP
                    wake_epoch += 1
                    await self._admit_wake(time.monotonic(), strip_required=False, source="wave")
                    from reachy_muse.motion.wave_back import DURATION_S
                    if self._wake_deadline is not None:
                        self._wake_deadline += DURATION_S
                    self._wake_window = WakeWindow.AWAKE
                    self._wake_first_request_pending = False
                    self._open_call()
                    recorder.abort()
                    recorder = make_recorder()
                    pre_roll.clear()
                    pre_roll_samples = 0
                    if wake is not None:
                        await asyncio.to_thread(wake.reset)
                    await asyncio.to_thread(self.hardware.gesture, "wave")

                    async def greet_after_wave(was_asleep=asleep, turn_started=self._turn_started):
                        from reachy_muse.motion.wave_back import DURATION_S
                        await asyncio.sleep(DURATION_S)
                        if self._wake_deadline is None:
                            return
                        if (self._turn_started == turn_started
                                and not (self._has_output() or self._hearing_open() or self._turn_active)):
                            await self._announce(WAKE_GREETING if was_asleep else WAKE_CUE,
                                                 started=None, defer_playback=True)
                    wave_greeting = self._background(greet_after_wave())
                    log.info("Wave cues=1")
                    continue
                captured = await capture.get()
                if captured is None:
                    continue
                sample = captured.samples
                if captured.gap:
                    if self._wake_direction is not None:
                        self._wake_direction.clear()
                    if self._wake_policy is not None:
                        self._wake_policy.reset()
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
                input_blocked = captured.discard or self._input_blocked()
                observed_wake = None
                now = time.monotonic()
                wake_required = self._wake_deadline is None or (now >= self._wake_deadline and not recording_active())
                if wake is not None and self._wake_policy is not None:
                    if blocked_episode and not input_blocked:
                        await asyncio.to_thread(wake.reset)
                        self._wake_policy.reset()
                    observed_wake = await detects_wake(sample, replace(
                        captured.wake_context or self._wake_capture_context(now),
                        armed=wake_required,
                    ))
                if input_blocked:
                    if self._wake_direction is not None:
                        self._wake_direction.clear()
                    # Unqualified speaker paths cannot stitch user speech across echo.
                    if not blocked_episode:
                        recorder = make_recorder()
                        pre_roll.clear()
                        pre_roll_samples = 0
                        if self._wake_first_request_pending:
                            # Muse can answer in a call before Reachy's endpoint; the window stays open for it.
                            if self.call is None:
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
                if self._wake_direction is not None and len(sample):
                    import numpy as np
                    if float(np.mean(np.square(sample))) >= .0001:
                        self._direction_audio_until = time.monotonic() + .5
                if self.wake_enabled:
                    now = time.monotonic()
                    if wake_required:
                        if wake is None:
                            continue
                        pre_roll.append(sample.copy())
                        pre_roll_samples += len(sample)
                        trim_pre_roll(pre_roll_limit)
                        detected, wake_tail = observed_wake if observed_wake is not None else await detects_wake(sample)
                        if not detected:
                            continue
                        await watch_wave(False)
                        self._attend_wake(captured.captured_at - wake_tail / self.hardware.sample_rate
                                          if wake_tail is not None else None)
                        if self.voice_verifier is not None:
                            import numpy as np
                            self.voice_verifier.realtime = (self.call is not None or
                                                            (self.realtime is not None and self.realtime.usable))
                            candidate = np.concatenate(tuple(pre_roll))
                            if wake_tail:
                                candidate = candidate[:-wake_tail]
                            candidate = candidate[-int(1.5 * self.hardware.sample_rate):]
                            self._voice_verifying = True
                            try:
                                decision = await asyncio.to_thread(
                                    self.voice_verifier.verify, candidate, self.hardware.sample_rate)
                            finally:
                                self._voice_verifying = False
                            if not decision.allowed:
                                pre_roll.clear()
                                pre_roll_samples = 0
                                await asyncio.to_thread(wake.reset)
                                if wake_resampler is not None:
                                    import av
                                    wake_resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
                                await self._voice_rejected()
                                continue
                        wake_epoch += 1
                        await self._admit_wake(time.monotonic(), strip_required=wake_tail is None)
                        await asyncio.to_thread(wake.reset)
                        log.info("Reachy heard its wake phrase; microphone open")
                        if wake_tail is None:
                            # Untimed detectors retain the exact text boundary guard.
                            await self._close_call()
                            recorder = make_recorder(initial_min_s=.06)
                        else:
                            self._open_call()
                            # The acoustic detector supplies a fresh cut after the
                            # keyword. Discard all earlier audio and capture only
                            # the request; do not transcribe the invocation again.
                            trim_pre_roll(wake_tail)
                            recorder = make_recorder()
                            log.info("Wake handoff retained %.3fs of post-keyword audio; request capture open",
                                     wake_tail / self.hardware.sample_rate)
                        wav = None
                        for captured in pre_roll:
                            completed = await record(captured, send_to_call=wake_tail is not None)
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
                                completed = await record(captured[:0], send_to_call=wake_tail is not None)
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
                                                        turn_running=turn_task is not None or wav is not None))
                if wav is None:
                    continue
                while wav is not None:
                    await queue_turn(wav)
                    if self.wake_enabled and self.wake_timeout_s == 0:
                        break
                    # Retain resampler/VAD state and drain endpoints in this SDK chunk.
                    wav = await record(sample[:0])
        finally:
            capture_stop.set()
            tasks = [task for task in (producer, speaker, direction_task, wave_greeting,
                                      turn_task, follow_up_task) if task is not None]
            tasks.extend(call_recognitions.values())
            for task in tasks:
                task.cancel()
            for ended in turns_awaiting_transcript:
                ended.cancel()
            self._wake_direction = None
            self._hearing_open = lambda: False

            async def cleanup():
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    if self.wave_wake:
                        if wave_operation is not None:
                            await asyncio.gather(wave_operation, return_exceptions=True)
                        await asyncio.to_thread(self.hardware.set_wave_watching, False)
                    await asyncio.gather(*(ended.endpoint() for ended in turns_awaiting_transcript),
                                         return_exceptions=True)
                finally:
                    recorder.abort()
                    self._playback.set_user_speaking(False)
                    recorder_worker.shutdown(wait=True)

            await _finish_cleanup(cleanup())

    async def _capture_microphone(self, capture: _CaptureBuffer, *, stop=None) -> None:
        """Drain the SDK on one dedicated reader, independent of asyncio progress."""
        from reachy_muse.hardware import ReachyHardwareError
        loop = asyncio.get_running_loop()
        stop = stop if stop is not None else threading.Event()
        lock = threading.Lock()
        pending = deque()
        pending_samples = 0
        scheduled = False
        overflow = False
        budget = getattr(self.hardware, "vision_budget", None)

        def deliver():
            nonlocal pending_samples, scheduled, overflow
            if budget is not None:
                budget.audio_busy(self._turn_active or self._has_output() or self._hearing_open()
                                  or getattr(self._wake_policy, "verifying", False)
                                  or getattr(self, "_voice_verifying", False))
            with lock:
                chunks = list(pending)
                pending.clear()
                lost, overflow = overflow, False
                pending_samples = 0
                scheduled = False
            for generation, sample, options in chunks:
                if generation == capture.generation:
                    if lost:
                        options.update(gap=True, gap_reason="capture handoff overflow")
                        lost = False
                    capture.push(sample, **options)

        def read_continuously():
            nonlocal pending_samples, scheduled, overflow
            last_sample = time.monotonic()
            previous_duration = None
            generation = 0
            try:
                while not stop.is_set():
                    muted_at_start = self._input_blocked()
                    reader = self.hardware.read_audio
                    discarded = None
                    if generation != capture.generation:
                        generation = capture.generation
                        discarded = capture.discarded
                        previous_duration = None
                        reader = self.hardware.discard_microphone
                    sample = reader()
                    now = time.monotonic()
                    duration = len(sample) / self.hardware.sample_rate if sample is not None else 0
                    if budget is not None:
                        budget.observe("capture", duration, now=now)
                    if generation != capture.generation:
                        continue
                    if discarded is not None:
                        loop.call_soon_threadsafe(discarded.set)
                    if not duration:
                        if now - last_sample > 15:
                            raise ReachyHardwareError("Reachy microphone supplied no audio for 15 seconds")
                        stop.wait(.002)
                        continue
                    gap = previous_duration is not None and now - last_sample > max(duration, previous_duration) + .25
                    previous_duration, last_sample = duration, now
                    options = dict(discard=muted_at_start or self._input_blocked(), gap=gap,
                                   captured_at=now, gap_reason="SDK read or capture scheduling stall" if gap else None)
                    if self._wake_policy is not None:
                        options["wake_context"] = self._wake_capture_context(now)
                    with lock:
                        if pending_samples + len(sample) > capture.limit:
                            pending.clear()
                            pending_samples = 0
                            overflow = True
                        pending.append((generation, sample, options))
                        pending_samples += len(sample)
                        if not scheduled:
                            scheduled = True
                            loop.call_soon_threadsafe(deliver)
            except Exception as exc:
                loop.call_soon_threadsafe(capture.fail, exc)

        try:
            with _microphone_reader():
                worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reachy-microphone")
                if budget is not None:
                    budget.capture_active(True)
                running = loop.run_in_executor(worker, read_continuously)
                try:
                    await asyncio.shield(running)
                finally:
                    stop.set()
                    try:
                        await _finish_cleanup(running)
                    finally:
                        worker.shutdown(wait=True)
                        if budget is not None:
                            budget.capture_active(False)
        except ReachyHardwareError as exc:
            capture.fail(exc)

    async def turn(self, wav: bytes | None, *, wake_strip_required: bool | None = None,
                   recognized_text: str | None = None, defer_playback: bool = False,
                   timeline: TurnTimeline | None = None, heard_over: tuple[str, ...] = ()) -> TurnOutcome:
        if wav is None:
            return TurnOutcome.EMPTY
        self._turn_active = True
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
                if text is not None:
                    self._presence.mark("voice", time.monotonic())
                    timeline.mark("transcript")
                self._say_hushed = False
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
                self._moments.finish_turn(lambda: not self._has_output(), expression=lambda: job.last_expression)
            completed = True
            return outcome
        except asyncio.CancelledError:
            turn_cancelled = True
            raise
        finally:
            self._turn_active = False
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
                        self._discard_waiting(job)
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
                    if not defer_playback and not completed:
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
                    heard_over: tuple[str, ...] = (), woken: bool = False,
                    held: _HeldRequest | None = None, quiet: bool = False) -> str | None | TurnOutcome:
        """The request's words with the wake phrase removed; None when Muse hears only the audio.

        The wake phrase alone gets "Yes?", or the wake-up greeting when it just woke Reachy, unless
        ``quiet``: Muse heard the request in a call and answers it in its own voice.
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
            self._observe_arbitration(text, held)
            log.info("Speech recognition outcome empty")
            if self.wake_detector is not None and strip_required:
                if not quiet:
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
                self._observe_arbitration(text, held)
                if not quiet:
                    self._moments.cue(Moment.GREETING)
                    await self._announce(cue, started=started, defer_playback=defer_playback)
                return TurnOutcome.WAKE_CUE
        self._observe_arbitration(text, held)
        # "Stop" or "be quiet" said over Reachy's own line is a barge-in even when the line used those words.
        if is_echo(text, heard_over) and classify_follow_up(text) not in (FollowUp.SILENCE, FollowUp.CANCEL):
            log.info("Reachy ignored %d characters heard over its own speech that only repeat it", len(text))
            return TurnOutcome.EMPTY
        log.info("Recognized %d characters from Reachy's microphone", len(text))
        await self._plan(plan.Working(output=self._has_output(),
                                      user_speaking=self._playback.user_speaking))
        return text

    def _observe_arbitration(self, text: str, held: _HeldRequest | None) -> None:
        if not self.arbitration_shadow:
            return
        from reachy_muse.arbitration import Snapshot, Target, TextRef, bounded_target
        from reachy_muse.arbitration_shadow import observe

        self._shadow_turn += 1
        try:
            job = self.job
            active = None
            if job is not None and job.request is not None and not job.dropped:
                ref = f"job-{id(job)}"
                active = bounded_target(Target(ref, TextRef(ref + ":request", "user", job.request),
                                               state="completed" if job.turn.task_finished else "running",
                                               omitted=True))
            waiting = None
            if held is not None and held.text is not None:
                ref = f"held-{id(held)}"
                waiting = bounded_target(Target(ref, TextRef(ref + ":request", "user", held.text),
                                                state="held", omitted=True))
            output = self._has_output()
            observe(Snapshot(str(self._shadow_turn), text, context_epoch=self._shadow_turn,
                             admission="follow_up" if job is not None else "wake",
                             capabilities=executable(
                                 wake=self.wake_enabled, output=output,
                                 face=getattr(self.hardware, "face_tracking_enabled", False) is True,
                                 volume=callable(getattr(self.hardware, "change_volume", None))),
                             local_chat=self.local_chat, output=output, owns_chat=self.owns_chat,
                             active=active, held=waiting))
        except Exception:
            log.info("Reachy arbitration shadow status=invalid_snapshot")

    async def _dispatch(self, wav: bytes, text: str | None, *, started: float,
                        defer_playback: bool, timeline: TurnTimeline, opening=None,
                        style: ReplyStyle | None = None) -> MuseJob:
        """Send one request to Muse; the returned job is ``self.job`` until its answer is delivered."""
        style = style or self.backends.reply_style()
        job = self.job = MuseJob(
            text or "", style,
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
        note, count = self._recent.chat_note() if text is not None else ("", 0)
        self._moments.cue(Moment.THINKING)
        if text is not None:
            message = muse_request(text, note=note)
            timeline.mark("dispatch")
            ack = await self.session.send_chat(message, self.session_id, **options)
        else:
            timeline.mark("dispatch")
            ack = await self.session.send_voice(wav, self.session_id, **options)
        response = acknowledgement(ack)
        self._recent.forget_chat(count)
        if text is not None:
            self._recent.asked(text)
        timeline.mark("muse_ack")
        if self.backends.progress_voice is not None:
            job.progress = self.backends.narrator.progress(job.request, time.monotonic(), job.intent)
        await self._queue_replies(job.reader.read(job.turn.acknowledge(response)), job)
        self._log_backend_activity()
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
        playback_cursor = PlaybackCursor()

        def answer_audible(then: Callable[[], None] | None = None,
                           expression: Expression | None = None) -> Callable[[], None]:
            def started() -> None:
                self._moments.stop_cue()
                job.last_expression = expression
                self._moments.prepare_closing(expression)
                job.answered = True
                job.timeline.mark("first_audio")
                job.timeline.report()
                if then is not None:
                    then()
            return started

        async def accepting(message_id: str, value: bool) -> None:
            await self._queue_replies(job.reader.set_audio_accepting(message_id, value), job)

        while True:
            if job.dropped:
                log.info("Reachy stopped relaying a cancelled Muse request")
                return TurnOutcome.ACCEPTED
            now = time.monotonic()
            if turn.task_finished:
                self._moments.prepare_closing(job.last_expression)
                await self._end_asides(job, let_playing_finish=True)
            if ((job.reader.has_spoken_reply() or job.reader.has_reply_text())
                    and turn.complete(now) and job.replies.empty()
                    and not job.reader.awaiting_audio()
                    and (not turn.task_finished or not job.reader.awaiting_final_text() or turn.settled(now))):
                await self._drain_speech(playback_cursor)
                return TurnOutcome.ACCEPTED
            if (not job.reader.has_spoken_reply() and job.replies.empty() and turn.settled(now)
                    and not job.reader.awaiting_audio()
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
                waiting = (job.progress.upcoming_waiting(time.monotonic(), WAITING_LEAD_S)
                           if job.progress is not None and not job.answered else None)
                if waiting is not None and len(waiting.encode("utf-8")) <= MAX_LINE_BYTES:
                    job.waiting_speech = self._prepare_line(self.backends.progress_voice,
                                                            _line(None, waiting, None, progress=True))
                if (job.progress is not None and job.aside is None and not job.answered
                        and not self._has_output() and not self._playback.user_speaking):
                    progress = job.progress.take(time.monotonic())
                    if progress is not None:
                        log.info("Reachy selected %s progress %.1fs after the voice turn",
                                 progress.source, time.monotonic() - started)
                        job.aside = asyncio.create_task(self._say_progress(progress, job))
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
                expression = classify_sentence(reply.text, Expression.parse(reply.expression)).expression
                def mark_spoken(stream=stream):
                    stream.spoken += 1
                on_accepting = partial(accepting, reply.message_id)
                prepared = self._prepared_speech.get((reply.message_id, reply.index))
                if job.defer_playback:
                    self._prepared_speech.pop((reply.message_id, reply.index), None)
                    queued_at = time.monotonic()
                    await self._output_queue.put(_SpeechJob(
                        reply.message_id, reply.text, expression.value, prepared,
                        is_current, answer_audible(mark_spoken, expression), started,
                        on_accepting=on_accepting))
                    response_deadline += time.monotonic() - queued_at
                    await self._queue_replies([], job)
                    job.answered = True
                    continue
                speaking = self._speak(reply.message_id, text=reply.text, expression=expression.value,
                                       stream_segment=True,
                                       prepared_stream=prepared,
                                       is_current=is_current, on_start=answer_audible(mark_spoken, expression),
                                       on_accepting=on_accepting, playback_cursor=playback_cursor,
                                       drain=job.replies.empty())
            else:
                message_id = reply.message_id if isinstance(reply, NarrationSegment) else reply
                stream = job.reader.streams[message_id]
                def is_current(stream=stream, reply=reply):
                    return (not job.dropped and not stream.revised
                            and (not isinstance(reply, NarrationSegment)
                                 or reply.version == stream.narration_version))
                if not is_current():
                    continue
                await self._end_asides(job, let_playing_finish=True)
                if not job.answered and job.request is not None:
                    await self._answer_begins(job)
                if not is_current():
                    continue
                lines = await self.backends.narrator.lines(
                    request_text, job.reader.narration_text(message_id), job.style, message_id=message_id)
                if not is_current():
                    continue
                job.timeline.mark("narration_done")
                lines = tuple(replace(line, expression=classify_sentence(line.text, line.expression).expression)
                              if line.text else line
                              for line in lines)
                def mark_narrated(stream=stream):
                    stream.narration_spoken = True
                on_accepting = partial(accepting, message_id)
                if job.defer_playback:
                    queued_at = time.monotonic()
                    for line in lines:
                        await self._output_queue.put(_SpeechJob(
                            message_id, line.text, line.expression.value if line.expression else None,
                            is_current=is_current, on_start=answer_audible(mark_narrated, line.expression),
                            started=started, on_accepting=on_accepting))
                    response_deadline += time.monotonic() - queued_at
                    job.answered = True
                    continue
                prepared = None
                speaking = self._speak_lines(message_id, lines, is_current=is_current,
                    on_line_start=lambda line: answer_audible(mark_narrated, line.expression),
                    on_accepting=on_accepting, playback_cursor=playback_cursor)
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
        await self._plan(plan.AnswerArrived(output=self._has_output()))
        topic = request_topic(job.request or "")
        if (not job.context_changed or not topic or not self.backends.voice.speaks_text
                or time.monotonic() - job.dispatched_at <= LATE_ANSWER_S):
            return
        # A late answer can follow other talk; name what it answers, in the user's own words.
        intro = f"About {topic}:"
        log.info("Reachy introduced an answer that arrived %.1fs after its request",
                 time.monotonic() - job.dispatched_at)
        if job.defer_playback:
            await self._output_queue.put(_SpeechJob(None, intro, is_current=lambda: not job.dropped,
                                                    started=job.started))
        else:
            for key in tuple(self._prepared_speech):
                self._discard_prepared(key)
            await asyncio.gather(*self._closing_speech)
            await self._speech_with_timeout(self._speak(None, text=intro, stream_segment=True))

    async def _follow_up(self, endpoint: Endpoint, request: _QueuedTurn,
                         held: _HeldRequest | None) -> _HeldRequest | None:
        """Act on an utterance heard while Muse works; returns the request now waiting behind the job.

        Cancel and added detail apply to the newest request: the held one if any, else the running job.
        """
        text = await self._hear(endpoint.audio, endpoint.text, request.wake_strip_required,
                                started=time.monotonic(), defer_playback=True, heard_over=request.heard_over,
                                held=held)
        if isinstance(text, TurnOutcome):
            return held
        kind = FollowUp.NEW if text is None else classify_follow_up(text)
        # A one-word command ("goodnight", "louder") is obeyed before one-word asides are dropped.
        command = (await self._obey(text, request.timeline.marks["endpoint"], request.timeline)
                   if kind is FollowUp.NEW and text is not None else None)
        if command is not None:
            if not COMMANDS[command].silence_speech:
                self._say_hushed = False
            return None if COMMANDS[command].wake_window in (WakeWindow.DISMISSED, WakeWindow.ASLEEP) else held
        if kind in (FollowUp.NEW, FollowUp.ADD_DETAIL) and text is not None and too_short_for_follow_up(text):
            log.info("Reachy ignored a one-word follow-up while Muse worked")
            return held
        job = self.job
        running = job if job is not None and job.request is not None and not job.dropped else None
        if running is not None:
            running.context_changed = True
        log.info("Reachy heard a %s follow-up while Muse worked", kind.value)
        if kind is FollowUp.SILENCE:
            self._hush()
        elif kind in (FollowUp.CANCEL, FollowUp.ADD_DETAIL):
            self._say_hushed = False
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
            if running is not None and running.backend_pending:
                await self._add_detail(running, text)
                return None
        if held is not None:
            if job is None or not job.turned_away:
                await self._announce(BUSY_CUE, started=None, defer_playback=True)
            if job is not None:
                job.turned_away = True
            return held
        self._say_hushed = False
        await self._plan(plan.RequestHeld())
        return _HeldRequest(endpoint.audio, text, request.wake_epoch, request.timeline)

    async def _obey(self, text: str, started: float, timeline: TurnTimeline) -> DeviceCommand | None:
        allowed = executable(wake=self.wake_enabled, output=self._has_output(),
                             face=getattr(self.hardware, "face_tracking_enabled", False) is True,
                             volume=callable(getattr(self.hardware, "change_volume", None)))
        command = await self.backends.commands.recognize(text, allowed)
        if command is None:
            return None
        spec = COMMANDS[command]
        if spec.wake_window in (WakeWindow.DISMISSED, WakeWindow.ASLEEP):
            # "Stop listening" said before a sleeping Reachy's window closes leaves it asleep.
            if self._wake_window is not WakeWindow.ASLEEP:
                self._wake_window = spec.wake_window
            self._wake_deadline = None
            await self._close_call()
        elif self.call is not None:
            # Muse heard the command too; Reachy has handled it, so Muse's answer to it is dropped.
            await self.call.barge_in()
        if spec.silence_speech:
            self._hush()
        elif spec.wake_window in (WakeWindow.DISMISSED, WakeWindow.ASLEEP):
            self._moments.stop()
        if spec.wake_window in (WakeWindow.DISMISSED, WakeWindow.ASLEEP):
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

    async def _realtime_turn(self, endpoint: Endpoint, request: _QueuedTurn) -> TurnOutcome:
        """A turn Muse already heard in a realtime call: Reachy acts only on its own device commands."""
        started = time.monotonic()
        if self._wake_window is WakeWindow.WOKEN:
            self._wake_window = WakeWindow.AWAKE
        text = endpoint.text or ""
        if self.wake_detector is not None:
            text = (question_after_wake(text, self.wake_detector.phrase) or "" if request.wake_strip_required
                    else strip_wake_prefix(text, self.wake_detector.phrase))
        if request.heard_over and (not text.strip() or repeats_playback(text, request.heard_over)):
            return TurnOutcome.EMPTY
        self._presence.mark("voice", started)
        if text.strip() and await self._obey(text, started, request.timeline) is not None:
            return TurnOutcome.ACCEPTED
        call = request.call
        if call.failed:
            await self._call_lost()
            return TurnOutcome.EMPTY
        replied_at = call.speaker.reply_started
        if not call.audible and (replied_at is None or replied_at < request.timeline.marks["speech_end"]):
            call.awaiting_answer = True
        request.timeline.facts["path"] = "realtime"
        request.timeline.report()
        return TurnOutcome.ACCEPTED

    def _open_call(self) -> None:
        if self.realtime is None or self.call is not None or not self.realtime.usable:
            return
        from reachy_muse.motion.reply_motion import ReplyMotion
        library = self.backends.motion if getattr(self.hardware, "library_expressions", False) else None
        motion = ReplyMotion(library=library, muse_moving=lambda: self.muse_moving > 0)
        self.call = self.realtime.open(self.hardware, motion, on_audible=self._muse_audible,
                                       on_start=self._muse_speaks, on_idle=self._muse_quiet,
                                       on_working=self._muse_working, on_reply_done=self._muse_replied,
                                       own_speaker=self._own_speaker)

    async def _call_ended(self, *, capturing: bool) -> None:
        call, self.call = self.call, None
        await call.close()
        if capturing and call.audio_sent:
            # Muse may have heard the request in progress, so it never goes to chat as well.
            self._lost_call = call
        if call.failed:
            log.warning("Reachy's realtime call to Muse failed; its next request goes by chat")
            await self.realtime.call_failed()
            if call.awaiting_answer:
                await self._call_lost()
        elif call.reason == "remote_stop" and self._wake_deadline is not None:
            log.info("Muse ended the realtime call; Reachy's wake window closes")
            self._wake_deadline = time.monotonic()

    async def _call_lost(self) -> None:
        log.info("Reachy's realtime call ended before Muse answered; asking the user to repeat")
        await self._announce(CALL_LOST_CUE, started=None, defer_playback=True)

    def _wake_window_closed(self) -> None:
        self._lost_call = None
        call, self.call = self.call, None
        if call is not None:
            self._background(call.close())

    async def _close_call(self) -> None:
        call, self.call = self.call, None
        if call is not None:
            await call.close()

    def _background(self, work) -> asyncio.Task:
        task = asyncio.create_task(work)
        self._call_tasks.add(task)
        task.add_done_callback(self._call_tasks.discard)
        return task

    def _barge_in_call(self) -> None:
        if self._barging is None or self._barging.done():
            log.info("Reachy stopped Muse's realtime reply for the user's speech")
            self._barging = self._background(self.call.barge_in())

    def _muse_audible(self, text: str | None, until: float, *, speech: bool = True) -> None:
        self._echo_until = max(self._echo_until, until + ECHO_TAIL_S)
        if speech and not text:
            self._wake_unknown_output_until = max(self._wake_unknown_output_until, until + ECHO_TAIL_S + 1.5)
        if text:
            if self._audible and self._audible[-1][0] == text:
                self._audible[-1] = (text, until + ECHO_TAIL_S)
            else:
                self._audible.append((text, until + ECHO_TAIL_S))

    async def _muse_speaks(self) -> None:
        if self.call is not None:
            self.call.awaiting_answer = False
        self._moments.stop_cue()
        if self._user_quiet_at is not None:
            log.info("Muse's realtime reply started %.2f s after the user stopped speaking",
                     time.monotonic() - self._user_quiet_at)
            self._user_quiet_at = None
        await asyncio.to_thread(self.hardware.set_state, "speaking")

    async def _muse_quiet(self) -> None:
        working = self.call is not None and self.call.working
        if self._wake_deadline is not None:
            self._wake_deadline = time.monotonic() + (self.wake_timeout_s or 10.0)
        await self._plan(plan.OutputIdle(user_speaking=self._playback.user_speaking, turn_open=working,
                                         wake_open=self._wake_deadline is not None))
        if working and not self._playback.user_speaking:
            self._moments.cue(Moment.THINKING)

    def _muse_working(self, working: bool) -> None:
        if working:
            self._background(self._think_unless_answered())
        elif self.call is not None and not self.call.audible:
            self._background(self._muse_quiet())

    async def _think_unless_answered(self) -> None:
        await asyncio.sleep(THINKING_AFTER_S)
        call = self.call
        if call is None or not call.working or call.audible or self._playback.user_speaking:
            return
        self._moments.cue(Moment.THINKING)
        await self._plan(plan.Working(output=False, user_speaking=False))

    def _muse_replied(self, response_id: str | None) -> None:
        call = self.call
        if call is None:
            return
        closing = partial(self._reply_closing, call, response_id)
        if self._moments.active:
            self._moments.finish_turn(lambda: not call.audible and not call.working, expression=closing)
        else:
            self._background(self._close_with_clip(call, closing))

    def _reply_closing(self, call, response_id: str | None) -> Expression | None:
        """The closing a reply earns, unless Muse still works, has started another reply, or moved Reachy itself."""
        if (self.call is not call or call.working or call.latest_response != response_id
                or call.muse_moved or self.muse_moving > 0):
            return None
        return call.motion.closing(response_id)

    async def _close_with_clip(self, call, closing: Callable[[], Expression | None]) -> None:
        """Without companion motion, a reply's closing is a recorded clip, else a built-in expression.

        Muse reports a reply done while its audio still streams, so the closing waits until the
        speaker has been quiet for a moment, every sentence has had its expression, and Muse's
        work on the turn has settled.
        """
        deadline = time.monotonic() + 30
        quiet_since = None
        while time.monotonic() < deadline and self.call is call:
            if call.audible or call.working:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = time.monotonic()
            elif time.monotonic() - quiet_since >= CLOSING_QUIET_S:
                break
            await asyncio.sleep(.05)
        moment = CLOSINGS.get(closing())
        if moment is None or call.audible or self._playback.user_speaking:
            return
        for name in CLOSING_CLIPS[moment]:
            result = await asyncio.to_thread(self.hardware.run_command, "reachy.expression",
                                             {"name": name, "intensity": 0.6}, 5000)
            if result.get("ok"):
                log.info("Reachy closed Muse's realtime reply with %s", name)
                return

    def muse_moved(self, change: int) -> None:
        """Muse started (+1) or finished (-1) moving Reachy itself; local reply motion waits meanwhile."""
        self.muse_moving = max(0, self.muse_moving + change)
        if change > 0 and self.call is not None:
            self.call.muse_moved = True
            if self.call.audible:
                log.info("Muse moved Reachy during its realtime reply; Reachy's own expressions wait")

    def _hush(self) -> None:
        self._moments.stop()
        self._hush_floor = next(_SPEECH_SERIALS)
        self._say_hushed = True
        for task in self._command_tasks:
            task.cancel()

    def _observe_presence(self, source: str) -> None:
        self._presence.mark(source, time.monotonic())
        if self._morning is not None:
            self._morning.observe_presence()

    async def _admit_wake(self, now: float, *, strip_required: bool, source: str = "voice") -> None:
        """Shared acoustic/visual wake state and greeting motion."""
        self._say_hushed = False
        self._observe_presence(source)
        self._wake_at = now
        self._wake_deadline = now + (self.wake_timeout_s or 10.0)
        self._wake_strip_required = strip_required
        self._wake_first_request_pending = True
        if self._wake_window is WakeWindow.ASLEEP:
            await self._wake_up(WakeWindow.WOKEN)
        elif self._wake_window is WakeWindow.DISMISSED:
            self._wake_window = WakeWindow.AWAKE
        if self._has_output() or self._muted:
            self._playback.set_user_speaking(True)
        await self._plan(plan.WakeHeard())

    async def _wake_up(self, window: WakeWindow) -> None:
        """Leave the sleep pose; the wake phrase and "wake up" both come here.

        The motion loop lifts the head and unfolds the antennas when the state leaves sleeping.
        """
        self._wake_window = window
        self._say_hushed = False
        await self._plan(plan.Sleep(False, output=self._has_output()))

    async def _drop(self, job: MuseJob, *, stop_motion: bool = True) -> None:
        """Stop relaying ``job``. Muse has no cancel call, so Reachy asks it in words on the same chat.

        After Muse finishes, Reachy only skips the lines not yet spoken.
        """
        job.dropped = True
        if stop_motion:
            self._moments.stop()
        if job.progress is not None:
            job.progress.stop()
        if not job.backend_pending:
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

    async def _say_progress(self, progress: ProgressUpdate, job: MuseJob | None = None) -> None:
        prepared = None
        if job is not None and progress.source == "waiting":
            prepared, job.waiting_speech = job.waiting_speech, None
        lines = [""]
        for word in progress.text.split():
            text = f"{lines[-1]} {word}".strip()
            if len(text.encode("utf-8")) > MAX_LINE_BYTES:
                lines.append(word)
            else:
                lines[-1] = text
        for text in lines:
            speech, prepared = prepared, None
            try:
                await self._speak(None, text=text, state="thinking", stream_segment=True,
                                  voice=self.backends.progress_voice, progress=True, prepared_stream=speech)
            finally:
                if speech is not None:
                    speech.cancel()
                    await speech.aclose()

    async def _drain_speech(self, cursor: PlaybackCursor) -> None:
        player = PCMPlayer(self.hardware.output_sample_rate, None, self._playback, cursor=cursor)
        await player.finish()

    @contextlib.asynccontextmanager
    async def _own_speaker(self, cursor: PlaybackCursor):
        async with self._speaker_lock:
            if self._wake_direction is not None:
                self._wake_direction.clear()
            if self._speaker_cursor is not None and self._speaker_cursor is not cursor:
                await self._drain_speech(self._speaker_cursor)
            self._speaker_cursor = cursor
            yield

    def _prepare_line(self, voice, line: SpokenLine):
        prepared = voice.prepare(line, self.hardware.output_sample_rate)
        return prepared if prepared is not None else PreparedSpeech(
            voice.stream(line, self.hardware.output_sample_rate), self.hardware.output_sample_rate)

    async def _speak_lines(self, message_id: str, lines, *, is_current: Callable[[], bool] | None = None,
                           on_start: Callable[[], None] | None = None, on_line_start=None,
                           on_accepting=None, playback_cursor=None) -> None:
        lines = tuple(replace(line, muse_message_id=message_id) for line in lines)
        if not lines:
            return
        cursor = playback_cursor if playback_cursor is not None else PlaybackCursor()
        prepared = self._prepare_line(self.backends.voice, lines[0]) if lines[0].text else None
        following = None
        try:
            for index, line in enumerate(lines):
                if is_current is not None and not is_current():
                    raise _SpeechSuperseded
                if index + 1 < len(lines):
                    next_line = lines[index + 1]
                    following = self._prepare_line(self.backends.voice, next_line) if next_line.text else None
                await self._speak(message_id, text=line.text,
                    expression=line.expression.value if line.expression else None,
                    stream_segment=index < len(lines) - 1, is_current=is_current,
                    on_start=on_line_start(line) if on_line_start else on_start,
                    prepared_stream=prepared, on_accepting=on_accepting,
                    playback_cursor=cursor, drain=index == len(lines) - 1)
                if prepared is not None:
                    await prepared.aclose()
                prepared, following = following, None
        finally:
            await asyncio.gather(*(item.aclose() for item in (prepared, following) if item is not None))

    async def _speak(self, message_id: str | None, *, text: str | None = None,
                     state: str = "speaking",
                     expression: str | None = None, stream_segment: bool = False,
                     voice=None, progress: bool = False, prepared_stream=None,
                     is_current=None, on_start=None, turn_started: float | None = None,
                     role: str | None = None, speech_serial: int | None = None,
                     on_accepting=None, playback_cursor=None, drain: bool = True, allowed=None) -> None:
        # Direct acknowledgements/progress and deferred replies share this owner.
        # Interrupted audio is flushed before a waiting speaker acquires it.
        speech_serial = next(_SPEECH_SERIALS) if speech_serial is None else speech_serial
        self._speech_pending += 1
        try:
            playback_cursor = playback_cursor if playback_cursor is not None else PlaybackCursor()
            async with self._own_speaker(playback_cursor):
                if speech_serial <= self._hush_floor or (is_current is not None and not is_current()):
                    raise _SpeechSuperseded
                accepted_audio = False
                def accepted():
                    nonlocal accepted_audio
                    accepted_audio = True
                try:
                    await self._speak_owned(message_id, text=text, state=state,
                        expression=expression,
                        stream_segment=stream_segment, voice=voice,
                        progress=progress, prepared_stream=prepared_stream,
                        is_current=is_current, on_start=on_start, turn_started=turn_started,
                        role=role, on_accepting=on_accepting,
                        playback_cursor=playback_cursor, drain=drain, on_audio=accepted, allowed=allowed)
                except BaseException as exc:
                    if isinstance(exc, _SpeechSuperseded) and not accepted_audio:
                        raise
                    try:
                        await self._clear_owned_audio(propagate_cancel=False)
                    except Exception:
                        if not isinstance(exc, asyncio.CancelledError):
                            raise
                    raise
        finally:
            self._speech_pending -= 1

    def _say_blocked(self) -> str | None:
        if self._closed:
            return "disconnected"
        if not self._ready.is_set():
            return "not_ready"
        if self._wake_window is WakeWindow.ASLEEP:
            return "asleep"
        if self._say_hushed:
            return "hushed"
        if (self._turn_active or self.job is not None or self._hearing_open() or self._playback.user_speaking
                or (self.call is not None and (self.call.audible or getattr(self.call, "working", False)))):
            return "busy"
        if not (self.backends.command_voice or self.backends.voice).speaks_text:
            return "speech_unavailable"
        return None

    async def say(self, text: str, timeout_s: float, *, speech_serial: int | None = None,
                  allowed=None, before_audio=None) -> dict:
        """Queue one bounded line with the same speaker ownership as replies."""
        speech_serial = next(_SPEECH_SERIALS) if speech_serial is None else speech_serial
        started = False
        def allow_audio():
            nonlocal started
            if allowed is not None and not allowed():
                return False
            if not started:
                if self._say_blocked() is not None or (before_audio is not None and not before_audio()):
                    return False
                started = True
            return True
        task = asyncio.current_task()
        self._command_tasks.add(task)
        try:
            reason = self._say_blocked() or ("unavailable" if allowed is not None and not allowed() else None)
            if reason is None:
                try:
                    await asyncio.wait_for(self._speak(
                        None, text=text, stream_segment=True, voice=self.backends.command_voice,
                        speech_serial=speech_serial, allowed=allow_audio,
                        is_current=lambda: self._say_blocked() is None and (allowed is None or allowed())), timeout_s)
                except _SpeechSuperseded:
                    reason = self._say_blocked() or "interrupted"
                except asyncio.TimeoutError:
                    reason = "timeout"
                except asyncio.CancelledError:
                    reason = ((self._say_blocked() or "interrupted")
                              if speech_serial <= self._hush_floor else "disconnected")
                except Exception:
                    reason = "speech_failed"
                finally:
                    if not self._has_output():
                        await self._plan(plan.OutputIdle(user_speaking=self._playback.user_speaking,
                                                         turn_open=self.job is not None or self._turn_active,
                                                         wake_open=self._wake_deadline is not None))
            return {"ok": True, "spoken": False, "reason": reason} if reason else {"ok": True, "spoken": True}
        finally:
            self._command_tasks.discard(task)

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
        if self._speaker_cursor is not None:
            self._speaker_cursor.end = 0.0
        self._echo_until = min(self._echo_until, time.monotonic() + ECHO_TAIL_S)
        if cancelled and propagate_cancel:
            raise asyncio.CancelledError

    async def _clear_idle_audio(self) -> None:
        # A backend turn has no authority over another turn's active playback.
        # Count awakened lock waiters too: release can precede their acquisition.
        if not self._speech_pending and not self._speaker_lock.locked():
            async with self._speaker_lock:
                if self._speaker_cursor is not None:
                    await self._drain_speech(self._speaker_cursor)
                await self._clear_owned_audio()

    async def _speak_owned(self, message_id: str | None, *, text: str | None = None,
                           state: str = "speaking",
                           expression: str | None = None, stream_segment: bool = False,
                           voice=None, progress: bool = False, prepared_stream=None,
                           is_current=None, on_start=None, turn_started: float | None = None,
                           role: str | None = None, on_accepting=None,
                           playback_cursor=None, drain: bool = True, on_audio=None, allowed=None) -> None:
        voice = voice if voice is not None else self.backends.voice
        sentence = classify_sentence(text, Expression.parse(expression)) if text and state == "speaking" else None
        if sentence is not None:
            expression = sentence.expression.value
        timed_motion = sentence is not None and callable(getattr(self.hardware, "finish_sentence", None))
        samples_played = 0
        expression_task = None
        announced = False
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
            if not timed_motion:
                if state == "speaking":
                    self._moments.stop_cue()
                await hardware_call(self.hardware.set_state, state, expression=self._pose_expression(state, expression))
            if is_current is not None and not is_current():
                raise _SpeechSuperseded
            if not timed_motion:
                audible()
        def audible() -> None:
            nonlocal announced
            if announced:
                return
            announced = True
            self._recent.spoke(line.text)
            if on_start is not None:
                on_start()
            if started is not None:
                log.info("%s started %.1fs after the voice turn",
                         _SPEECH_LABELS[line.role][0],
                         time.monotonic() - started)
            if expression and expression not in ("nod", "shake"):
                log.info("Reachy spoke with expression %s", expression)
        async def pause() -> None:
            self._muted = False
            await hardware_call(self.hardware.set_state, "listening")
            log.info("Reachy speech paused for microphone input")
        async def resume() -> None:
            await hardware_call(self.hardware.set_state, state,
                                expression=None if timed_motion else self._pose_expression(state, expression))
            log.info("Reachy speech resumed after microphone silence")
        async def push(samples) -> None:
            nonlocal samples_played
            if self._hush_floor != hush_floor or (allowed is not None and not allowed()):
                raise _SpeechSuperseded
            self._audio_clear_attempted = False
            self._muted = True
            self._muse_audible(line.text, time.monotonic() + .13, speech=line.role != "cue")
            current_sentence = sentence
            if timed_motion and samples_played == 0 and expression_task is not None:
                if expression_task.done() and not expression_task.cancelled():
                    result = expression_task.result()
                    if result is not None:
                        current_sentence = replace(sentence, motion=result)
                else:
                    expression_task.cancel()
            motion = {"sentence": current_sentence} if timed_motion and samples_played == 0 else {}
            accepting = timed_motion and samples_played == 0
            if accepting and on_accepting is not None:
                await on_accepting(True)
            try:
                if self.audio_diagnostics:
                    dispatched = time.monotonic()
                    entered = finished = dispatched
                    def measured_write():
                        nonlocal entered, finished
                        entered = time.monotonic()
                        try:
                            self.hardware.play_audio(samples, **motion)
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
                    await hardware_call(self.hardware.play_audio, samples, **motion)
                if accepting:
                    audible()
                samples_played += len(samples)
                if on_audio is not None:
                    on_audio()
                self._speech_progress_at = time.monotonic()
            finally:
                if accepting and on_accepting is not None:
                    await on_accepting(False)
        player = PCMPlayer(self.hardware.output_sample_rate, push, self._playback,
                           on_start=begin, on_pause=pause, on_resume=resume,
                           diagnostics=self.audio_diagnostics, cursor=playback_cursor)
        text = text or ""
        line = _line(message_id, text, expression, progress=progress, role=role)
        if voice.speaks_text and not text and prepared_stream is None:
            if expression:
                await self._express(expression)
                return
            raise ValueError("Muse returned an empty spoken response")
        if prepared_stream is not None:
            speech = prepared_stream
        else:
            speech = self._prepare_line(voice, line)
        source = self.backends.motion
        if (timed_motion and getattr(self.hardware, "library_expressions", False)
                and source is not None and getattr(source, "library_available", False)):
            async def expression_motion():
                try:
                    return await asyncio.wait_for(source.generate(sentence.expression.value, 2.5), .05)
                except (asyncio.TimeoutError, CapabilityUnavailable):
                    return None
                except Exception:
                    log.exception("Sentence expression motion failed")
                    return None
            expression_task = asyncio.create_task(expression_motion())
        try:
            try:
                while True:
                    self._speech_source = speech
                    self._waiting_for_speech = True
                    try:
                        samples = await speech.__anext__()
                    except StopAsyncIteration:
                        break
                    finally:
                        self._waiting_for_speech = False
                        self._speech_source = None
                    self._speech_progress_at = time.monotonic()
                    if isinstance(samples, SpeechRestart):
                        await player.finish()
                        if timed_motion:
                            self.hardware.finish_sentence()
                        samples_played = 0
                        player = PCMPlayer(self.hardware.output_sample_rate, push, self._playback,
                            on_start=begin, on_pause=pause, on_resume=resume,
                            diagnostics=self.audio_diagnostics, cursor=playback_cursor)
                        continue
                    await player.play(samples)
            finally:
                await speech.aclose()
            if samples_played == 0:
                if is_current is not None and not is_current():
                    raise _SpeechSuperseded
                raise ValueError("Muse returned no playable speech")
            if timed_motion:
                self.hardware.finish_sentence()
            if drain:
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
            if expression_task is not None:
                expression_task.cancel()
            self._muted = False
            if self.audio_diagnostics:
                log.info("Reachy playback timing %s", {**player.metrics, **write_timing,
                    **(speech.metrics if isinstance(speech, PreparedSpeech) else {})})

    async def _express(self, expression: str) -> None:
        result = await asyncio.to_thread(self.hardware.run_command, "reachy.expression",
                                        {"name": expression}, 20000)
        if result.get("ok"):
            log.info("Muse expressed %s through Reachy", expression)
        else:
            log.warning("Reachy rejected Muse expression %s: %s", expression, result.get("error"))

    async def greet(self, *, waiting: bool, wait_for_turn: bool = False) -> None:
        while wait_for_turn and (self._wake_first_request_pending or self._hearing_open()
                                 or self._turn_active or self._has_output()
                                 or (self.call is not None and getattr(self.call, "working", False))):
            await asyncio.sleep(.05)
        if self._say_hushed and (wait_for_turn or self._wake_window is not WakeWindow.ASLEEP):
            return
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
        if self._say_hushed:
            return
        await self._output_queue.put(_SpeechJob(None, line, state="thinking"))

    async def brief(self, lines: tuple[SpokenLine, ...]) -> float:
        if self._say_hushed:
            return 0.0
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
        if self.wake_enabled:
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
            response = acknowledgement(await self.session.send_chat(request, self.session_id))
            timeline.mark("muse_ack")
            await self._queue_replies(job.reader.read(job.turn.acknowledge(response)), job)
            log.info("Muse accepted Reachy's morning briefing request")
            heard: list[tuple[str, int, int | None, SpokenLine]] = []
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
                if (heard and job.turn.complete(now) and job.replies.empty()
                        and (not job.turn.task_finished or not job.reader.awaiting_final_text()
                             or job.turn.settled(now))):
                    streams = job.reader.streams
                    return tuple(line for message_id, index, version, line in heard
                                 if not streams[message_id].revised and index >= streams[message_id].skip_before
                                 and (version is None or version == streams[message_id].narration_version))
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
                    heard.append((reply.message_id, reply.index, None,
                                  _line(reply.message_id, reply.text, reply.expression)))
                elif isinstance(reply, NarrationSegment):
                    heard.extend((reply.message_id, 1, reply.version, line)
                                 for line in briefing_lines(job.reader.narration_text(reply.message_id)))
                elif isinstance(reply, str):
                    heard.extend((reply, 1, None, line) for line in briefing_lines(job.reader.narration_text(reply)))
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
class ReachyService(PairedService):
    session_id: str | None = None
    prepare_chat: Callable[[LinkSession, str], Awaitable[str]] | None = None
    owns_chat: bool = False
    introduce_to_muse: bool = False
    silence_s: float = 2.0
    # Required: Python 3.9 dataclasses cannot follow inherited defaults with a field that has none.
    backends: Callable[[LinkSession], Backends] = field(default_factory=_backends_required)
    local_chat: bool = True
    arbitration_shadow: bool = False
    audio_diagnostics: bool = False
    wake_detector: object | None = None
    wave_wake: bool = False
    wake_policy: object | None = None
    voice_verifier: object | None = None
    wake_timeout_s: float = 10.0
    speech_gate: object | None = None
    short_utterance_silence_s: float | None = None
    morning: MorningRoutine | None = None
    place: str | None = None
    inbox: Inbox = field(default_factory=Inbox)
    _presence: PresenceFusion = field(default_factory=PresenceFusion, init=False, repr=False)
    startup: StartupTimeline | None = None
    muse_path: str = "chat"
    # Tests replace the voice-only connection and the SDK call; both default to the real ones.
    voice_link: Callable[[dict], Awaitable[object]] | None = None
    voice_call: Callable | None = None
    _asleep: bool = field(default=False, init=False, repr=False)
    _say_hushed: bool = field(default=False, init=False, repr=False)
    _voice: tuple[asyncio.AbstractEventLoop, VoiceConversation] | None = field(default=None, init=False, repr=False)
    _hardware_error: Exception | None = field(default=None, init=False, repr=False)

    async def run(self) -> None:
        camera = asyncio.create_task(self.morning.watch_camera()) if self.morning is not None else None
        service = asyncio.create_task(super().run())
        tasks = {service} | ({camera} if camera is not None else set())
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._hardware_error is not None:
            raise self._hardware_error

    def unpaired(self) -> None:
        self.inbox.clear()

    async def _session(self, vm: dict, pairing: dict) -> tuple[Outcome, float]:
        if self.inbox.path is not None:
            generation = pairing.get(INBOX_GENERATION, "")
            if not self.inbox.current(generation=generation):
                self.inbox = Inbox(self.inbox.path, clock=self.inbox.clock, generation=generation)
        elif not self.inbox.current():
            self.inbox = Inbox(clock=self.inbox.clock)
        return await super()._session(vm, pairing)

    def presence_result(self) -> dict:
        observe = getattr(self.executor, "presence_observations", None)
        if observe is not None:
            for source, observation in observe().items():
                self._presence.observe(source, observation)
        voice = self._voice[1] if self._voice is not None else None
        available = (voice is not None and voice._say_blocked() is None
                     and not voice._has_output() and not voice._wake_first_request_pending)
        return {**self._presence.result(time.monotonic()), "available": available,
                **({"place": self.place} if self.place is not None else {})}

    def describe(self) -> DeviceDescription:
        return reachy_description(self.identity.node_id, place=self.place)

    def run_command(self, name: str, params: dict, timeout_ms: int | None = None, *,
                    session: LinkSession | None = None) -> dict:
        if name == "reachy.notify":
            return self._notify(params, timeout_ms, session=session)
        if name in ("reachy.expression", "reachy.move"):
            return self._muse_motion(name, params, timeout_ms)
        if name not in ("reachy.say", "reachy.presence"):
            return self.executor.run_command(name, params, timeout_ms)
        try:
            if not isinstance(params, _CommandParams):
                params = _CommandParams(name, params, timeout_ms)
            command = parse_command(name, params, timeout_ms=timeout_ms)
        except ValueError:
            result = {"ok": False, "error": "invalid_parameters"}
        else:
            active_voice = self._voice
            if name == "reachy.presence":
                self.executor.presence(max(0, params.deadline - time.monotonic()))
                result = {"ok": True, **self.presence_result()}
            elif active_voice is None or (session is not None and active_voice[1].session is not session):
                result = {"ok": True, "spoken": False, "reason": "disconnected"}
            else:
                loop, voice = active_voice
                deadline = params.deadline

                async def speak():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return {"ok": True, "spoken": False, "reason": "timeout"}
                    return await voice.say(command.params["text"], remaining, speech_serial=params.serial)

                speech = speak()
                try:
                    future = asyncio.run_coroutine_threadsafe(speech, loop)
                except RuntimeError:
                    speech.close()
                    result = {"ok": True, "spoken": False, "reason": "disconnected"}
                else:
                    try:
                        result = future.result(max(0, deadline - time.monotonic()) + 1)
                    except CommandCancelled:
                        result = {"ok": True, "spoken": False, "reason": "disconnected"}
                    except CommandTimeout:
                        future.cancel()
                        result = {"ok": True, "spoken": False, "reason": "timeout"}
        outcome = result.get("reason") or result.get("error")
        if outcome is None:
            outcome = ("spoken" if name == "reachy.say" else
                       {True: "present", False: "absent", None: "unknown"}[result["present"]])
        log.info("Reachy command name=%s outcome=%s", name, outcome)
        if result["ok"]:
            return {"ok": True, "payload": {key: value for key, value in result.items() if key != "ok"}}
        return result

    def _notify(self, params: dict, timeout_ms: int | None, *, session: LinkSession | None) -> dict:
        try:
            if not isinstance(params, _CommandParams):
                params = _CommandParams("reachy.notify", params, timeout_ms)
            active = self._voice
            if session is not None and (active is None or active[1].session is not session):
                result = outcome("refused", "disconnected")
            elif active is None:
                result = self.inbox.add(params)
            else:
                loop, voice = active
                async def notify():
                    return await voice.notifications.notify(params, max(0, params.deadline - time.monotonic()))
                pending = notify()
                try:
                    future = asyncio.run_coroutine_threadsafe(pending, loop)
                except RuntimeError:
                    pending.close()
                    result = outcome("refused", "disconnected")
                else:
                    try:
                        result = future.result(max(0, params.deadline - time.monotonic()) + 1)
                    except (CommandTimeout, CommandCancelled):
                        future.cancel()
                        result = self.inbox.result(params["id"])
        except ValueError:
            result = outcome("refused", "invalid_parameters")
        except OSError:
            result = outcome("refused", "storage_unavailable")
        log.info("Reachy command name=reachy.notify outcome=%s reason=%s", result["status"], result["reason"])
        return {"ok": True, "payload": {key: value for key, value in result.items() if key != "ok"}}

    def _muse_motion(self, name: str, params: dict, timeout_ms: int | None) -> dict:
        """Run Muse's expression or move; while it runs, Reachy's own reply motion holds back."""
        active_voice = self._voice
        if active_voice is not None:
            try:
                active_voice[0].call_soon_threadsafe(active_voice[1].muse_moved, 1)
            except RuntimeError:
                active_voice = None
        try:
            return self.executor.run_command(name, params, timeout_ms)
        finally:
            if active_voice is not None:
                with contextlib.suppress(RuntimeError):
                    active_voice[0].call_soon_threadsafe(active_voice[1].muse_moved, -1)

    def _realtime(self, vm: dict, session_id: str | None) -> object | None:
        if self.muse_path != "realtime":
            return None
        from reachy_muse.realtime_call import RealtimeVoice
        connect = self.voice_link or self._voice_link
        return RealtimeVoice(lambda: connect(vm), device_id=self.identity.node_id,
                             thread_id=session_id if self.owns_chat else None,
                             half_duplex=getattr(self.executor, "echo_cancelled_input", False) is not True,
                             voice_call=self.voice_call)

    async def _voice_link(self, vm: dict):
        """A connection for voice calls only. It never registers, so this service keeps Muse's commands."""
        from musegadget import config
        from musegadget.service import DEFAULT_NOISE_HOST
        from reachy_muse.realtime_call import LINK_TIMEOUT_S, VoiceLink
        pairing = config.load_json(config.PAIRING_FILE) or {}
        options = {}
        if "app_id" in inspect.signature(LinkSession).parameters:
            options["app_id"] = "hatch-web"
        link = LinkSession(noise_host=pairing.get("noise_host") or DEFAULT_NOISE_HOST,
                           vm_id=vm["vm_id"] or vm["vm_name"], vm_auth_token=vm["vm_auth_token"],
                           device=self.describe(), register_device=False,
                           run_command=lambda *unused: {"ok": False, "error": "unsupported command"},
                           **options)
        stop = asyncio.Event()
        running = asyncio.create_task(link.run(stop))

        async def close() -> None:
            stop.set()
            try:
                await asyncio.wait_for(asyncio.shield(running), 3)
            except (Exception, asyncio.CancelledError):
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
        try:
            await link.wait_transport_ready(LINK_TIMEOUT_S)
        except BaseException:
            await close()
            raise
        return VoiceLink(link, lambda: not running.done(), close)

    async def serve(self, session: LinkSession, vm: dict, stop: asyncio.Event) -> Outcome:
        from reachy_muse.hardware import ReachyHardwareError

        def invoke(message):
            name = message.get("command")
            if name in ("reachy.say", "reachy.presence", "reachy.notify"):
                params = message.get("params")
                try:
                    params = _CommandParams(name, params if isinstance(params, dict) else {},
                                            message.get("timeout_ms") or None)
                except ValueError:
                    pass
                else:
                    message = {**message, "params": params}
            return LinkSession._invoke(session, message)

        session._invoke = invoke
        session._run_command = partial(self.run_command, session=session)
        hardware = self.executor
        startup, self.startup = self.startup or StartupTimeline(), None
        stop_link = asyncio.Event()
        async def converse():
            await session.registered.wait()
            startup.mark("registered")
            if self.introduce_to_muse:
                from reachy_muse.introduction import _UncertainIntroduction, introduce_once
                try:
                    await introduce_once(session, vm_id=vm["vm_id"] or vm["vm_name"],
                                         node_id=self.identity.node_id, stop=stop_link)
                except _UncertainIntroduction as exc:
                    log.warning("Skipping introduction: %s", exc)
            session_id = self.session_id
            if self.prepare_chat is not None:
                session_id = await self.prepare_chat(session, vm["vm_id"] or vm["vm_name"])
            if self.owns_chat and (not isinstance(session_id, str) or not session_id):
                raise ValueError("Reachy's dedicated side chat was not selected")
            startup.mark("chat_setup")
            if self.morning is not None:
                self.morning.select_chat(vm["vm_id"] or vm["vm_name"], session_id)
            voice = VoiceConversation(session, hardware, backends=self.backends(session),
                                      session_id=session_id, silence_s=self.silence_s,
                                      wake_detector=self.wake_detector, wake_timeout_s=self.wake_timeout_s,
                                      wave_wake=self.wave_wake,
                                      wake_policy=self.wake_policy,
                                      voice_verifier=self.voice_verifier,
                                      owns_chat=self.owns_chat, speech_gate=self.speech_gate,
                                      local_chat=self.local_chat, audio_diagnostics=self.audio_diagnostics,
                                      arbitration_shadow=self.arbitration_shadow,
                                      short_utterance_silence_s=self.short_utterance_silence_s,
                                      morning=self.morning, startup=startup,
                                      realtime=self._realtime(vm, session_id), presence=self._presence)
            voice.notifications = Notifications(self.inbox, self.presence_result, voice)
            voice._wake_window = WakeWindow.ASLEEP if self._asleep else WakeWindow.AWAKE
            voice._say_hushed = self._say_hushed
            self._voice = (asyncio.get_running_loop(), voice)
            try:
                await voice.run()
            finally:
                self._voice = None
                self._asleep = voice._wake_window is WakeWindow.ASLEEP
                self._say_hushed = voice._say_hushed
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
                if outcome is Outcome.STOPPED and not stop.is_set():
                    outcome = Outcome.CLOSED
            else:
                voice_task.result()
        except ReachyHardwareError as exc:
            self._hardware_error = exc
            self.stop()
            raise
        except Exception as exc:
            log.warning("Reachy conversation ended: %s: %s", type(exc).__name__, exc)
        finally:
            stop_link.set()
            voice_task.cancel()
            stop_task.cancel()
            await asyncio.gather(voice_task, stop_task, return_exceptions=True)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(link_task, 5)
        if outcome is Outcome.UNPAIRED:
            self.unpaired()
        if outcome is not Outcome.STOPPED:
            await asyncio.to_thread(hardware.set_state, "sleeping" if self._asleep else "error")
        return outcome
