# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Muse realtime voice calls on Reachy: its microphone in, Muse's voice out, its body in step.

``RealtimeVoice`` opens each call on a voice-only connection beside Reachy's registered session,
so Muse can still invoke Reachy's commands during the call. A ``RealtimeCall`` streams the
echo-cancelled microphone to Muse at 24 kHz and plays each reply chunk through Reachy's own
speaker path once it has playback ownership. The SDK's paced ``write`` is not used for sound:
its sleep-per-frame pacing drifts behind real time, and the accepted chunks carry the sample
offsets that sentence alignment needs. Motion never holds audio back; marks only split a chunk.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import numpy as np

from musegadget.streaming import HTTPStatusError
from reachy_muse.motion.reply_alignment import SAMPLE_RATE
from reachy_muse.motion.reply_motion import Mark, ReplyMotion
from reachy_muse.speech_playback import PCMPlayer, PlaybackCursor, SpeechPlayback
from reachy_muse.follow_up import FollowUp, classify_follow_up, repeats_playback, word_count
from reachy_muse.person_gaze import MuseActivity

log = logging.getLogger(__name__)

FRAME_SAMPLES = 960
CATCH_UP_FRAMES = 10
SILENT_FRAME = bytes(FRAME_SAMPLES * 2)
LINK_TIMEOUT_S = 10.0
READY_TIMEOUT_S = 8.0
BACKLOG_S = LINK_TIMEOUT_S + READY_TIMEOUT_S + 2.0
SPEAKER_LEAD_S = 0.1
IDLE_AFTER_S = 0.3
SPEAKER_DRAIN_S = 5.0
CALL_LIMIT_S = 900.0
NATURAL_ENDS = frozenset({"remote_stop", "idle_timeout", "overall_timeout"})
BARGE_IN_SPEECH_S = 0.3
DISTINCT_WORDS_S = 0.12
INTERRUPTION_BUFFER_S = 2.0
INTERRUPTION_RECOGNITION_S = 0.75
WAITING_CUE_S = 4.0
NO_EVENT_S = 15.0
WORKING_STALE_S = 30.0
RESPONSE_AUDIO_S = 8.0
# A tool call's silent response answers its turn once Muse stays quiet this long: Working can come back on just after.
QUIET_SETTLE_S = 1.0


class CallInput:
    """Hold playback-time audio until sustained speech has words distinct from Muse's.

    Silence keeps the remote VAD clock moving while local recognition catches up. Only
    a confirmed interruption releases the bounded pre-roll, preserving its first words.
    Echo and unrecognized room sounds never enter Muse's microphone stream. One gate serves
    one utterance, which can run on past the endpoint of a turn within it.
    """

    def __init__(self, feed, sample_rate: int):
        self.feed = feed
        self.rate = sample_rate
        self.reset()

    def reset(self) -> None:
        self._pending = deque()
        self._samples = 0
        self._speech_samples = 0
        self._distinct_samples = 0
        self._quiet_samples = 0
        self.confirmed = False
        self.sent_speech = False
        self.heard_speech = False

    def push(self, samples, *, speaking: bool, text: str, playback: bool,
             heard_over: tuple[str, ...], quiet_s: float = 0.0) -> bool:
        self.heard_speech = self.heard_speech or speaking
        if not playback or self.confirmed:
            self.feed.push(samples, silent=quiet_s >= len(samples) / self.rate + FRAME_SAMPLES / SAMPLE_RATE)
            self.sent_speech = self.sent_speech or speaking
            return speaking
        if repeats_playback(text, heard_over):
            self.reset()
            self.feed.push(np.zeros_like(samples))
            return False
        if speaking and self._quiet_samples:
            self._speech_samples = 0
            self._distinct_samples = 0
        self._quiet_samples = 0 if speaking else self._quiet_samples + len(samples)
        if self._quiet_samples > self.rate * INTERRUPTION_RECOGNITION_S:
            self.reset()
        self._pending.append(samples.copy())
        self._samples += len(samples)
        while self._pending and self._samples > self.rate * INTERRUPTION_BUFFER_S:
            self._samples -= len(self._pending.popleft())
        if speaking:
            self._speech_samples = min(self._speech_samples + len(samples), self._samples)
        distinct = self._distinct(text, heard_over)
        self._distinct_samples = self._distinct_samples + len(samples) if distinct else 0
        if (self._speech_samples >= BARGE_IN_SPEECH_S * self.rate
                and self._distinct_samples >= DISTINCT_WORDS_S * self.rate):
            self._confirm()
            return speaking
        self.feed.push(np.zeros_like(samples))
        return False

    def finish(self, text: str, heard_over: tuple[str, ...]) -> bool:
        """Confirm on a finished turn's words. Held audio stays: the utterance can go on past the turn."""
        if (not self.confirmed and self._speech_samples >= BARGE_IN_SPEECH_S * self.rate
                and self._distinct(text, heard_over)):
            self._confirm()
        return self.confirmed

    def _distinct(self, text: str, heard_over: tuple[str, ...]) -> bool:
        enough_words = word_count(text) >= 2 or classify_follow_up(text) in (
            FollowUp.CANCEL, FollowUp.SILENCE)
        return bool(enough_words and heard_over and not repeats_playback(text, heard_over))

    def _confirm(self) -> None:
        self.confirmed = True
        self.sent_speech = True
        for pending in self._pending:
            self.feed.push(pending)
        self._pending.clear()
        self._samples = 0


def realtime_supported() -> bool:
    """Whether the installed Muse SDK reports a voice call's reply transcript, lifecycle and audio."""
    try:
        from musegadget import voice_call
    except ImportError:
        return False
    names = ("VoiceCall", "AudioChunk", "TranscriptDelta", "TranscriptDone", "ResponseStarted",
             "ResponseDone", "Working", "PlaybackFlushed")
    if not all(hasattr(voice_call, name) for name in names):
        return False
    return "on_event" in inspect.signature(voice_call.VoiceCall).parameters


def _failure_details(exc: Exception) -> str:
    """Limit HTTP diagnostics to the status and a short error code."""
    kind = type(exc).__name__
    if not isinstance(exc, HTTPStatusError):
        return kind
    code = "unknown"
    try:
        body = json.loads(exc.body)
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        value = error.get("code")
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value):
            code = value
    return f"{kind} status={exc.status} code={code}"


class _Resampler:
    """Streaming mono resampling with PyAV: float32 in, int16 (``s16``) or float32 (``flt``) out."""

    def __init__(self, source_rate: int, target_rate: int, *, output: str) -> None:
        self._source_rate = source_rate
        self._dtype = np.int16 if output == "s16" else np.float32
        self._av = None
        if source_rate != target_rate:
            import av
            self._av = av
            self._resampler = av.AudioResampler(format=output, layout="mono", rate=target_rate)

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if self._av is None:
            if self._dtype is np.int16:
                return np.clip(np.round(samples * 32767), -32768, 32767).astype(np.int16)
            return samples
        frame = self._av.AudioFrame.from_ndarray(samples[None, :], format="flt", layout="mono")
        frame.sample_rate = self._source_rate
        parts = [chunk.to_ndarray().reshape(-1) for chunk in self._resampler.resample(frame)]
        return np.concatenate(parts).astype(self._dtype) if parts else np.zeros(0, dtype=self._dtype)


class MicrophoneFeed:
    """Reachy's microphone as a call's input: 24 kHz PCM16 frames.

    Frames wait here until the call is ready, for as long as connecting and admission can take,
    so Muse hears a request from its first word. Live catch-up skips confirmed silence,
    keeping the last 400 ms of each pause. Closing drains the remaining audio.
    ``sent`` counts frames the call has taken.
    """

    def __init__(self, input_rate: int) -> None:
        self.input_rate = input_rate
        self._resample = _Resampler(input_rate, SAMPLE_RATE, output="s16")
        self._pending = np.zeros(0, dtype=np.int16)
        self._pending_silent = True
        self._frames: deque[tuple[bytes, bool, float]] = deque(maxlen=int(BACKLOG_S * SAMPLE_RATE / FRAME_SAMPLES))
        self._arrived = asyncio.Event()
        self._closed = False
        self.sent = 0
        self.evictions = 0
        self._max_queue_age_s = 0.0

    @property
    def backlog_bytes(self) -> int:
        return len(self._frames) * FRAME_SAMPLES * 2

    @property
    def max_queue_age_s(self) -> float:
        if self._frames:
            self._max_queue_age_s = max(self._max_queue_age_s, time.monotonic() - self._frames[0][2])
        return self._max_queue_age_s

    def _pop_frame(self) -> bytes:
        frame, _, queued_at = self._frames.popleft()
        self._max_queue_age_s = max(self._max_queue_age_s, time.monotonic() - queued_at)
        return frame

    def push(self, samples: np.ndarray, *, silent: bool = False) -> None:
        if self._closed or not len(samples):
            return
        data = np.concatenate([self._pending, self._resample(samples)])
        whole = len(data) // FRAME_SAMPLES * FRAME_SAMPLES
        for start in range(0, whole, FRAME_SAMPLES):
            pcm = data[start:start + FRAME_SAMPLES].astype("<i2").tobytes()
            quiet = pcm == SILENT_FRAME or silent and (start > 0 or self._pending_silent)
            if len(self._frames) == self._frames.maxlen:
                self._pop_frame()
                self.evictions += 1
            self._frames.append((pcm, quiet, time.monotonic()))
        self._pending = data[whole:]
        self._pending_silent = not len(self._pending) or silent and (whole > 0 or self._pending_silent)
        if whole:
            self._arrived.set()

    def close(self) -> None:
        self._closed = True
        self._arrived.set()

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        while True:
            if not self._closed:
                quiet = 0
                for _, silent, _ in self._frames:
                    if not silent:
                        break
                    quiet += 1
                for _ in range(quiet - CATCH_UP_FRAMES):
                    self._pop_frame()
            if self._frames:
                self.sent += 1
                return self._pop_frame()
            if self._closed:
                raise StopAsyncIteration
            self._arrived.clear()
            await self._arrived.wait()


@dataclass(frozen=True)
class _Chunk:
    response_id: Optional[str]
    offset: int
    pcm: bytes


class RealtimeSpeaker:
    """Plays accepted reply PCM through ``hardware.play_audio`` with the chat path's pacing.

    Each chunk is split at the motion marks inside it, so a sentence's expression or a prosody
    accent rides on the first sample it belongs to. ``on_start`` runs before the first push of a
    reply, ``on_audible`` hears every push with the sentence it belongs to and when
    its sound ends, and ``on_idle`` runs once the queue has drained and the last push has played.
    ``reply_started`` is when the latest reply's first chunk began to play.
    """

    def __init__(self, hardware, motion: ReplyMotion, *, on_audible: Callable[[Optional[str], float], None],
                 on_start: Callable[[], Awaitable[None]], on_idle: Callable[[], Awaitable[None]],
                 own_speaker=None) -> None:
        self._hardware = hardware
        self._motion = motion
        self._on_audible = on_audible
        self._on_start = on_start
        self._on_idle = on_idle
        self._own_speaker = own_speaker or (lambda cursor: contextlib.nullcontext())
        self._owner = None
        self._timed = callable(getattr(hardware, "finish_sentence", None))
        self._rate = hardware.output_sample_rate
        self._resample = _Resampler(SAMPLE_RATE, self._rate, output="flt")
        self._queue: asyncio.Queue[_Chunk] = asyncio.Queue()
        self._cursor = PlaybackCursor()
        self._task: Optional[asyncio.Task] = None
        self._flushing: Optional[asyncio.Future] = None
        self.failed = asyncio.get_running_loop().create_future()
        self._closed = False
        self._speaking = False
        self._sentence_open = False
        self._extras: dict = {}
        self._audible_text: Optional[str] = None
        self._response_id: Optional[str] = None
        self._audible_responses: dict[Optional[str], float] = {}
        self._replies: list = []
        self._pending_replies: set[Optional[str]] = set()
        self.reply_started: Optional[float] = None

    @property
    def busy(self) -> bool:
        return self._speaking or self._owner is not None or not self._queue.empty()

    def audible_responses(self, tail_s: float) -> tuple[Optional[str], ...]:
        now = time.monotonic()
        self._audible_responses = {reply: until for reply, until in self._audible_responses.items()
                                   if until + tail_s > now}
        return tuple(self._audible_responses)

    def chunk(self, response_id: Optional[str], offset: int, pcm: bytes) -> None:
        if self._closed or self.failed.done():
            return
        self._pending_replies.add(response_id)
        self._queue.put_nowait(_Chunk(response_id, offset, pcm))
        self._start()

    def flush_now(self) -> asyncio.Future:
        """Drop queued and playing reply audio; chunks that arrive afterwards play once it is gone."""
        flushing = self._flushing is not None and not self._flushing.done()
        if not self.busy and not flushing and not self.failed.done():
            done = asyncio.get_running_loop().create_future()
            done.set_result(None)
            return done
        self._motion.flushed(self._pending_replies)
        self._pending_replies.clear()
        while not self._queue.empty():
            self._queue.get_nowait()
        if flushing:
            return self._flushing
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
        self._flushing = asyncio.create_task(self._cleared(task))
        self._flushing.add_done_callback(self._flushed)
        return self._flushing

    async def close(self) -> None:
        self._closed = True
        await asyncio.gather(self.flush_now(), return_exceptions=True)
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
        await asyncio.gather(*(item for item in (task, self._flushing) if item is not None),
                             return_exceptions=True)

    async def drain(self) -> None:
        while self.busy:
            if self.failed.done():
                raise self.failed.result()
            await asyncio.sleep(.01)

    def _start(self) -> None:
        if (not self._closed and not self.failed.done()
                and (self._task is None or self._task.done())
                and (self._flushing is None or self._flushing.done())):
            self._task = asyncio.create_task(self._pace(), name="reachy-realtime-speaker")
            self._task.add_done_callback(self._ended)

    def _ended(self, task) -> None:
        if not task.cancelled():
            error = task.exception()
            if error is not None and not self.failed.done():
                self.failed.set_result(error)

    def _flushed(self, task) -> None:
        self._ended(task)
        if not self._queue.empty():
            self._start()

    async def _cleared(self, task) -> None:
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        self._cursor.end = 0.0
        self._sentence_open = False
        self._extras = {}
        self._audible_text = None
        self._replies = []
        try:
            if self._owner is not None:
                await asyncio.to_thread(self._hardware.clear_audio)
        finally:
            now = time.monotonic()
            self._audible_responses = {reply: min(until, now) for reply, until in self._audible_responses.items()}
            speaking, self._speaking = self._speaking, False
            try:
                if speaking and self._owner is not None:
                    await self._on_idle()
            finally:
                await self._release()

    async def _release(self) -> None:
        owner, self._owner = self._owner, None
        if owner is not None:
            await owner.__aexit__(None, None, None)

    async def _pace(self) -> None:
        player = PCMPlayer(self._rate, self._push, SpeechPlayback(), lead_s=SPEAKER_LEAD_S, cursor=self._cursor)
        while True:
            try:
                item = await asyncio.wait_for(self._queue.get(), IDLE_AFTER_S)
            except asyncio.TimeoutError:
                if self._speaking:
                    await self._finish(player)
                continue
            if not self._speaking:
                self._speaking = True
                owner = self._own_speaker(self._cursor)
                await owner.__aenter__()
                self._owner = owner
            if item.response_id not in self._replies:
                self._replies.append(item.response_id)
                self.reply_started = time.monotonic()
                await self._on_start()
            self._response_id = item.response_id
            count = len(item.pcm) // 2
            marks = self._motion.marks(item.response_id, item.offset, count)
            cuts = sorted({0, count, *(mark.sample for mark in marks if 0 < mark.sample < count)})
            for start, stop in zip(cuts, cuts[1:]):
                for mark in marks:
                    if mark.sample == start:
                        self._apply(mark)
                pcm = np.frombuffer(item.pcm[start * 2:stop * 2], dtype="<i2").astype(np.float32) / 32768
                samples = self._resample(pcm)
                if len(samples):
                    await player.play(samples)

    def _apply(self, mark: Mark) -> None:
        if mark.ends:
            if self._timed and self._sentence_open:
                self._hardware.finish_sentence()
            self._sentence_open = False
            return
        if mark.revised:
            if self._timed:
                self._extras["sentence"] = mark.sentence
            return
        if mark.text is not None:
            if self._timed and self._sentence_open:
                self._hardware.finish_sentence()
            self._sentence_open = mark.sentence is not None
            self._audible_text = mark.text
            if mark.sentence is not None and self._timed:
                self._extras["sentence"] = mark.sentence
        if mark.accent is not None and self._timed:
            self._extras.setdefault("accents", []).append((0, mark.accent))

    async def _push(self, samples) -> None:
        extras, self._extras = self._extras, {}
        task = asyncio.create_task(asyncio.to_thread(self._hardware.play_audio, samples, **extras))
        cancelled = False
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                cancelled = True
        now = time.monotonic()
        until = max(self._cursor.end, now) + len(samples) / self._rate
        self._audible_responses[self._response_id] = until
        if len(self._audible_responses) > 8:
            del self._audible_responses[min(self._audible_responses, key=self._audible_responses.get)]
        self._on_audible(self._audible_text, until)
        if cancelled:
            raise asyncio.CancelledError

    async def _finish(self, player: PCMPlayer) -> None:
        await player.finish()
        if not self._queue.empty():
            return
        if self._timed and self._sentence_open:
            self._hardware.finish_sentence()
        self._sentence_open = False
        self._audible_text = None
        replies, self._replies = self._replies, []
        for response_id in replies:
            self._pending_replies.discard(response_id)
            self._motion.audio_done(response_id)
            report = self._motion.report(response_id)
            if report is not None:
                log.info("Reachy's motion for Muse's realtime reply: %s", report)
        if self._queue.empty():
            self._speaking = False
            await self._on_idle()
            await self._release()


@dataclass
class PendingTurn:
    ordinal: int
    endpoint_at: float
    speech_end_at: float
    words: int
    feed_sent: int
    checked_at: float
    waiting_s: float = 0.0
    event_silence_s: float = 0.0
    cued_at: Optional[float] = None


@dataclass
class _Response:
    ordinal: int
    started_at: float
    during_work: bool
    audio_bytes: int = 0
    done: bool = False
    completed: bool = False


class RealtimeCall:
    """One call: the SDK ``VoiceCall`` plus Reachy's microphone feed, speaker and reply motion.

    ``latest_response`` is the reply Muse started last, and ``muse_moved`` whether Muse has moved
    Reachy itself since then. ``on_working`` hears each Working event, and ``on_working(True)`` each
    response Muse starts while Working is on, and each that completes then without audio, as a tool
    call's does. A spoken reply's ResponseDone shows no further work: it can come after the reply's
    audio has played. ``worked_at`` is when the last event that shows Muse at work came: Working
    coming on, or one of those responses. ``face_follow_seconds`` gives the robot's running totals
    of paused and held face following, for the call's summary.
    """

    def __init__(self, feed: MicrophoneFeed, speaker: RealtimeSpeaker, motion: ReplyMotion, *,
                 on_working: Callable[[bool], None], on_reply_done: Callable[[Optional[str]], None],
                 face_follow_seconds: Callable[[], tuple[float, float]] = lambda: (0.0, 0.0)) -> None:
        self.voice_call = None
        self.feed = feed
        self.input = CallInput(feed, feed.input_rate)
        self.speaker = speaker
        self.motion = motion
        self._on_working = on_working
        self._on_reply_done = on_reply_done
        self.working = False
        self.pending: list[PendingTurn] = []
        self.turns = 0
        self.outcomes = {"answered": 0, "silent": 0, "unanswered": 0, "recovered": 0, "not_awaited": 0}
        self.latest_response: Optional[str] = None
        self.muse_moved = False
        self.reason: Optional[str] = None
        self.failure: Optional[BaseException] = None
        self.loss_announced = False
        self.opened_at = time.monotonic()
        self.ready_at: Optional[float] = None
        self.link_reused = False
        self.last_event_at: Optional[float] = None
        self.worked_at: Optional[float] = None
        self._quiet_since: Optional[float] = None
        self._moved_at: Optional[float] = None
        self._working_since: Optional[float] = None
        self._worked_s = 0.0
        self._face_follow_seconds = face_follow_seconds
        self._face_follow_at_open = face_follow_seconds()
        self._responses: dict[Optional[str], _Response] = {}
        self._response_count = 0
        self._audio_bytes = 0
        self._working_seen = False
        self._task: Optional[asyncio.Task] = None
        self._closing = False
        self._transcripts: dict[Optional[str], str] = {}

    def accept_turn(self, *, endpoint_at: float, speech_end_at: float, words: int, awaits_answer: bool = True) -> bool:
        self.turns += 1
        turn = PendingTurn(self.turns, endpoint_at, speech_end_at, words, self.feed.sent, endpoint_at)
        replied_at = self.speaker.reply_started
        if replied_at is not None and replied_at >= speech_end_at:
            self._outcome(turn, "answered", replied_at)
            return False
        if self._answered_silently(turn, time.monotonic()):
            self._outcome(turn, "silent")
            return False
        if self.done:
            self._outcome(turn, "unanswered")
            return False
        if not awaits_answer:
            self._outcome(turn, "not_awaited")
            return False
        self.pending.append(turn)
        return True

    def finish_pending(self, outcome: str, replied_at: Optional[float] = None) -> None:
        pending, self.pending = self.pending, []
        for turn in pending:
            self._outcome(turn, outcome, replied_at)

    def finish_silent_answers(self) -> bool:
        """Finish the pending turns Muse answered without speaking, such as by moving Reachy; whether any were."""
        now = time.monotonic()
        answered = [turn for turn in self.pending if self._answered_silently(turn, now)]
        self.pending = [turn for turn in self.pending if turn not in answered]
        for turn in answered:
            self._outcome(turn, "silent")
        return bool(answered)

    def moved_by_muse(self) -> None:
        """Muse began moving Reachy itself: its reply's closing gives way, and the move shows Muse acted."""
        self.muse_moved = True
        self._moved_at = time.monotonic()

    def _answered_silently(self, turn: PendingTurn, now: float) -> bool:
        """Whether Muse answered the turn by acting: a response begun after its last voiced sample completed
        without audio, and either Muse moved Reachy since that sample, or the latest such response began while
        Muse worked and Muse has since been quiet, neither working nor in a response, for ``QUIET_SETTLE_S``.

        Without either sign, the response looks just like a spoken reply the SDK held or dropped. Muse can
        gesture while a tool call hangs, so a move made while Muse still works on the turn doesn't count;
        Working stuck on from an earlier turn doesn't stop it.
        """
        silent = [response for response in self._responses.values()
                  if response.completed and not response.audio_bytes and response.started_at >= turn.speech_end_at]
        if not silent:
            return False
        if (self._moved_at is not None and self._moved_at >= turn.speech_end_at
                and (not self.working or self._working_since < turn.speech_end_at)):
            return True
        latest = max(silent, key=lambda response: response.started_at)
        return latest.during_work and self._quiet_since is not None and now - self._quiet_since >= QUIET_SETTLE_S

    def _outcome(self, turn: PendingTurn, outcome: str, replied_at: Optional[float] = None) -> None:
        self.outcomes[outcome] += 1
        now = time.monotonic()
        held = getattr(self.voice_call, "held", None)
        log.info("Realtime turn ordinal=%d outcome=%s words=%d voice_to_reply_ms=%s endpoint_to_reply_ms=%s "
                 "held=%s feed_sent_delta=%d last_event_age_ms=%s",
                 turn.ordinal, outcome, turn.words,
                 round((replied_at - turn.speech_end_at) * 1000) if replied_at is not None else "none",
                 round((replied_at - turn.endpoint_at) * 1000) if replied_at is not None else "none",
                 "unknown" if held is None else "yes" if held else "no", self.feed.sent - turn.feed_sent,
                 round((now - self.last_event_at) * 1000) if self.last_event_at is not None else "none")
        if replied_at is not None:
            log.info("Muse's realtime reply started %.2f s after turn %d's last voiced sample",
                     replied_at - turn.speech_end_at, turn.ordinal)

    def _events_fresh(self, now: float) -> bool:
        return self.last_event_at is not None and now - self.last_event_at < WORKING_STALE_S

    @property
    def at_work(self) -> bool:
        """Muse is working on a request: Working is on and its work has not gone stale.

        The robot's SDK can leave Working on after Muse stops, so Working no longer counts once no
        event has shown Muse at work for ``WORKING_STALE_S``.
        """
        return self.working and self.worked_at is not None and time.monotonic() - self.worked_at < WORKING_STALE_S

    def _progress_suspended(self, now: float) -> bool:
        if not self._events_fresh(now):
            return False
        opened = [response for response in self._responses.values() if not response.done]
        latest = self._responses.get(self.latest_response)
        if (latest is not None and not latest.done and latest.audio_bytes == 0
                and now - latest.started_at >= RESPONSE_AUDIO_S
                and any(latest.started_at >= turn.speech_end_at for turn in self.pending)):
            return False
        settling = self._quiet_since is not None and now - self._quiet_since < QUIET_SETTLE_S
        return self.working or bool(opened) or settling

    def check_pending(self, *, paused: bool, window_expired: bool) -> tuple[list[int], Optional[str]]:
        """Advance the pending turns' clocks: the turns now due a waiting cue, and why to recover the call, if so.

        A turn's clock runs from its endpoint, but not before the call is ready, while ``paused``, or
        while Muse shows progress. A later turn is a retry only if it ended after the earlier one's
        cue. ``window_expired`` recovers the call only while the clock runs, once a turn had its cue.
        """
        now = time.monotonic()
        running = self.ready_at is not None and not paused and not self._progress_suspended(now)
        cues = []
        for turn in self.pending:
            start = max(turn.checked_at, self.ready_at or now)
            elapsed = max(0.0, now - start)
            turn.checked_at = now
            if self.last_event_at is not None and self.last_event_at >= start:
                turn.event_silence_s = 0.0
                event_elapsed = now - self.last_event_at
            else:
                event_elapsed = elapsed
            if paused:
                continue
            turn.event_silence_s += event_elapsed
            if running:
                turn.waiting_s += elapsed
                if turn.waiting_s >= WAITING_CUE_S and turn.cued_at is None:
                    turn.cued_at = now
                    cues.append(turn.ordinal)
        if paused:
            return cues, None
        if any(turn.event_silence_s >= (WORKING_STALE_S if self.working else NO_EVENT_S) for turn in self.pending):
            return cues, "working_stale" if self.working else "no_events"
        if not running:
            return cues, None
        if any(first.cued_at is not None and second.cued_at is not None and second.speech_end_at >= first.cued_at
               for first, second in zip(self.pending, self.pending[1:])):
            return cues, "retry"
        if window_expired and any(turn.cued_at is not None for turn in self.pending):
            return cues, "window_expired"
        return cues, None

    def audible_transcripts(self, tail_s: float) -> tuple[str, ...]:
        return tuple(self._transcripts[reply] for reply in self.speaker.audible_responses(tail_s)
                     if reply in self._transcripts)

    def activity(self) -> MuseActivity:
        """When Muse's responses began, whether the latest is open or finished without audio, and when the work
        it is busy with began. Muse that has sent nothing for WORKING_STALE_S is neither in a response nor at
        work."""
        recent = self.last_event_at is not None and time.monotonic() - self.last_event_at < WORKING_STALE_S
        latest = self._responses.get(self.latest_response) if recent else None
        return MuseActivity(tuple(sorted(response.started_at for response in self._responses.values())),
                            latest is not None and not latest.done,
                            latest is not None and latest.completed and not latest.audio_bytes,
                            self._working_since if recent and self.working else None)

    def on_event(self, event) -> None:
        """The SDK's event callback: it runs on the event loop and an exception here would end the call."""
        kind = type(event).__name__
        self.last_event_at = time.monotonic()
        if self._closing or self.done:
            return
        if kind == "Working":
            if not self._working_seen or self.working != event.working:
                log.info("Realtime event=Working working=%s", "on" if event.working else "off")
            self._working_seen = True
        elif kind == "ResponseStarted" and event.response_id not in self._responses:
            self._response_count += 1
            self._responses[event.response_id] = _Response(self._response_count, self.last_event_at, self.working)
            log.info("Realtime event=ResponseStarted ordinal=%d", self._response_count)
        elif kind == "AudioChunk":
            self._audio_bytes += len(event.pcm)
            response = self._responses.get(event.response_id)
            if response is not None:
                response.audio_bytes += len(event.pcm)
        elif kind == "ResponseDone":
            response = self._responses.get(event.response_id)
            if response is not None and not response.done:
                response.done = True
                response.completed = event.status in (None, "completed")
                status = event.status if event.status in ("completed", "cancelled", "failed", "incomplete") else "unknown"
                log.info("Realtime event=ResponseDone ordinal=%d status=%s audio_bytes=%d",
                         response.ordinal, status, response.audio_bytes)
        elif kind == "PlaybackFlushed":
            log.info("Realtime event=PlaybackFlushed")
        if kind == "PlaybackFlushed" and self.draining:
            return
        if kind in ("TranscriptDelta", "TranscriptDone"):
            previous = self._transcripts.get(event.response_id, "") if kind == "TranscriptDelta" else ""
            self._transcripts[event.response_id] = (previous + event.text)[-8000:]
            if len(self._transcripts) > 8:
                del self._transcripts[next(iter(self._transcripts))]
        if kind == "AudioChunk":
            self.speaker.chunk(event.response_id, event.offset, event.pcm)
        elif kind == "PlaybackFlushed":
            self.speaker.flush_now()
        elif kind == "Working":
            if event.working and not self.working:
                self._working_since = self.last_event_at
            elif self.working and not event.working:
                self._worked_s += self.last_event_at - self._working_since
            self.working = event.working
        elif kind == "ResponseStarted":
            self.latest_response = event.response_id
            self.muse_moved = False
        if kind in ("Working", "ResponseStarted", "ResponseDone"):
            if self.working or any(not response.done for response in self._responses.values()):
                self._quiet_since = None
            elif self._quiet_since is None:
                self._quiet_since = self.last_event_at
        done = self._responses.get(event.response_id) if kind == "ResponseDone" else None
        new_work = self.working and (kind == "ResponseStarted"
                                     or done is not None and done.completed and not done.audio_bytes)
        if new_work or kind == "Working" and event.working:
            self.worked_at = self.last_event_at
        try:
            if kind == "AudioChunk":
                self.motion.audio(event.response_id, event.offset, event.pcm)
            elif kind == "TranscriptDelta":
                self.motion.words(event.response_id, event.text)
            elif kind == "TranscriptDone":
                self.motion.transcript(event.response_id, event.text)
            elif kind == "ResponseStarted":
                self.motion.started(event.response_id)
            elif kind == "ResponseDone":
                if event.status in (None, "completed"):
                    self._on_reply_done(event.response_id)
            elif kind == "Working":
                self._on_working(event.working)
            elif kind == "PlaybackFlushed":
                self.motion.flushed(event.response_ids)
            if new_work:
                self._on_working(True)
        except Exception as exc:
            log.warning("Reachy's realtime motion failed on a %s event (%s)", kind, type(exc).__name__)

    def start(self, build: Callable[["RealtimeCall"], Awaitable[object]]) -> None:
        """Run the call that ``build`` makes; the microphone feed buffers until it is ready."""
        self._task = asyncio.create_task(self._run(build), name="reachy-realtime-call")

    async def _run(self, build) -> None:
        running = None
        readiness = None
        try:
            self.voice_call = await build(self)
            running = asyncio.create_task(self.voice_call.run(self.feed, _SilentOutput(self), realtime_input=True))
            if getattr(self.voice_call, "ready", None) is not None:
                readiness = asyncio.create_task(self._ready())
            await asyncio.wait((running, self.speaker.failed), return_when=asyncio.FIRST_COMPLETED)
            if self.speaker.failed.done():
                raise self.speaker.failed.result()
            self.reason = running.result()
            if self.reason == "remote_stop":
                try:
                    await asyncio.wait_for(self.speaker.drain(), SPEAKER_DRAIN_S)
                except asyncio.TimeoutError:
                    await self.speaker.flush_now()
            log.info("Muse realtime call ended: %s after %.1f s", self.reason, time.monotonic() - self.opened_at)
        except asyncio.CancelledError:
            if not self._closing:
                raise
        except Exception as exc:
            self.failure = exc
            log.warning("Muse realtime call failed (%s) after %.1f s", _failure_details(exc),
                        time.monotonic() - self.opened_at)
        finally:
            tasks = [task for task in (running, readiness) if task is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.finish_pending("unanswered")
            now = time.monotonic()
            worked = self._worked_s + (now - self._working_since if self.working else 0.0)
            totals = zip(self._face_follow_seconds(), self._face_follow_at_open)
            paused, held = (total - at_open for total, at_open in totals)
            log.info("Realtime call_summary duration_ms=%d turns=%d answered=%d silent=%d unanswered=%d recovered=%d "
                     "not_awaited=%d responses=%d audio_bytes=%d feed_sent=%d feed_evictions=%d max_queue_age_ms=%d "
                     "working_s=%.1f face_paused_s=%.1f face_held_s=%.1f",
                     round((now - self.opened_at) * 1000), self.turns,
                     self.outcomes["answered"], self.outcomes["silent"], self.outcomes["unanswered"],
                     self.outcomes["recovered"], self.outcomes["not_awaited"],
                     self._response_count, self._audio_bytes, self.feed.sent, self.feed.evictions,
                     round(self.feed.max_queue_age_s * 1000), worked, paused, held)

    async def _ready(self) -> None:
        await self.voice_call.ready.wait()
        self.ready_at = time.monotonic()
        log.info("Realtime call_ready elapsed_ms=%d link=%s queued_feed_frames=%d",
                 round((self.ready_at - self.opened_at) * 1000), "reused" if self.link_reused else "new",
                 self.feed.backlog_bytes // (FRAME_SAMPLES * 2))

    @property
    def done(self) -> bool:
        return self._task is not None and self._task.done()

    @property
    def draining(self) -> bool:
        return not self._closing and getattr(self.voice_call, "reason", None) == "remote_stop"

    @property
    def audio_sent(self) -> bool:
        return self.feed.sent > 0

    @property
    def failed(self) -> bool:
        return self.failure is not None or (self.done and not self._closing and self.reason not in NATURAL_ENDS)

    @property
    def audible(self) -> bool:
        return self.speaker.busy

    async def barge_in(self) -> None:
        """Stop Muse's reply now: fence its audio in the SDK, ask Muse to cancel it, and flush the speaker."""
        flushed = self.speaker.flush_now()
        if self.voice_call is not None and not self.done:
            with contextlib.suppress(Exception):
                await self.voice_call.barge_in()
        await flushed

    def detach(self) -> None:
        """From now on, ignore the call's events and stop feeding it the microphone; ``close`` ends it."""
        if not self.done:
            self._closing = True
        self.feed.close()

    async def close(self) -> None:
        """End the call. The SDK's cleanup still hangs up with Muse when this task is cancelled again."""
        self.detach()
        speaker = asyncio.create_task(self.speaker.close())
        if self._task is not None and not self._task.done():
            self._task.cancel()
        cleanup = asyncio.gather(*(task for task in (speaker, self._task) if task is not None),
                                 return_exceptions=True)
        cancelled = False
        while True:
            try:
                await asyncio.shield(cleanup)
                break
            except asyncio.CancelledError:
                cancelled = True
        self.motion.close()
        if cancelled:
            raise asyncio.CancelledError


class _SilentOutput:
    """The SDK's playback sink. Reachy plays accepted chunks itself; a flush still clears them."""

    def __init__(self, call: RealtimeCall) -> None:
        self._call = call

    def write(self, pcm: bytes) -> None:
        pass

    def flush(self) -> None:
        if not self._call.draining:
            self._call.speaker.flush_now()


@dataclass(frozen=True)
class VoiceLink:
    """A ready voice-only transport, whether it is still connected, and how to close it."""

    session: object
    alive: Callable[[], bool]
    close: Callable[[], Awaitable[None]]


class RealtimeVoice:
    """Opens Muse realtime calls with Reachy's device identity and dedicated thread.

    ``connect`` returns a ``VoiceLink`` that does not register the device; it is reused for later
    calls while it stays connected and no call on it has failed. ``voice_call`` builds the SDK
    call and defaults to ``musegadget.voice_call.VoiceCall``.
    """

    def __init__(self, connect: Callable[[], Awaitable[VoiceLink]], *, device_id: str,
                 thread_id: Optional[str] = None, half_duplex: bool = False, voice_call=None) -> None:
        self._connect = connect
        self.device_id = device_id
        self.thread_id = thread_id
        self.half_duplex = half_duplex
        self._voice_call = voice_call
        self._link: Optional[VoiceLink] = None
        self._connecting: Optional[asyncio.Task] = None
        self._warming: Optional[asyncio.Task] = None
        self._failed = False
        self._link_used = False

    @property
    def usable(self) -> bool:
        """False from a failed call until a request has gone to Muse by chat."""
        return not self._failed

    async def warm(self) -> None:
        """Connect ahead of the first wake word so a call opens without a handshake."""
        try:
            await self._ready_link()
        except Exception as exc:
            log.info("Muse realtime voice link is not ready yet (%s)", type(exc).__name__)

    def open(self, hardware, motion: ReplyMotion, *, on_audible, on_start, on_idle, on_working,
             on_reply_done, own_speaker=None) -> RealtimeCall:
        """Start a call at once; it connects, if needed, and is admitted while the microphone buffers."""
        call = RealtimeCall(MicrophoneFeed(hardware.sample_rate),
                            RealtimeSpeaker(hardware, motion, on_audible=on_audible, on_start=on_start,
                                            on_idle=on_idle, own_speaker=own_speaker),
                            motion, on_working=on_working, on_reply_done=on_reply_done,
                            face_follow_seconds=lambda: getattr(hardware, "face_follow_seconds", (0.0, 0.0)))
        call.start(self._build)
        log.info("Muse realtime call opening, %s duplex", "half" if self.half_duplex else "full")
        return call

    async def _build(self, call: RealtimeCall):
        factory = self._voice_call
        if factory is None:
            from musegadget.voice_call import VoiceCall as factory
        link = await self._ready_link()
        call.link_reused = self._link_used
        self._link_used = True
        return factory(link.session, device_id=self.device_id, thread_id=self.thread_id,
                       half_duplex=self.half_duplex, ready_timeout=READY_TIMEOUT_S,
                       idle_timeout=CALL_LIMIT_S, overall_timeout=CALL_LIMIT_S, on_event=call.on_event)

    async def _ready_link(self) -> VoiceLink:
        if self._link is not None and not self._link.alive():
            await self._drop_link()
        if self._link is None:
            if self._connecting is None or self._connecting.done():
                self._connecting = asyncio.ensure_future(self._connect_link())
            self._link = await asyncio.shield(self._connecting)
        return self._link

    async def _connect_link(self) -> VoiceLink:
        self._link = await self._connect()
        return self._link

    async def call_failed(self) -> None:
        """Drop the connection; the next request goes to Muse by chat."""
        self._failed = True
        await self._drop_link()

    def chat_turn(self) -> None:
        """A request went to Muse by chat: calls open again, on a link reconnected ahead of the next one."""
        if self._failed:
            self._failed = False
            self._warming = asyncio.ensure_future(self.warm())

    async def close(self) -> None:
        for task in (self._warming, self._connecting):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await self._drop_link()

    def detach_link(self) -> Optional[VoiceLink]:
        """Stop offering the current link, so the next call connects a new one; the caller closes this one."""
        self._link_used = False
        link, self._link = self._link, None
        return link

    async def _drop_link(self) -> None:
        link = self.detach_link()
        if link is not None:
            with contextlib.suppress(Exception):
                await link.close()
