# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Muse realtime voice calls on Reachy: its microphone in, Muse's voice out, its body in step.

``RealtimeVoice`` opens each call on a voice-only connection beside Reachy's registered session,
so Muse can still invoke Reachy's commands during the call. A ``RealtimeCall`` streams the
echo-cancelled microphone to Muse at 24 kHz and plays each reply chunk through Reachy's own
speaker path as soon as the SDK accepts it. The SDK's paced ``write`` is not used for sound:
its sleep-per-frame pacing drifts behind real time, and the accepted chunks carry the sample
offsets that sentence alignment needs. Motion never holds audio back; marks only split a chunk.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import numpy as np

from reachy_muse.motion.reply_alignment import SAMPLE_RATE
from reachy_muse.motion.reply_motion import Mark, ReplyMotion
from reachy_muse.speech_playback import PCMPlayer, PlaybackCursor, SpeechPlayback

log = logging.getLogger(__name__)

FRAME_SAMPLES = 960
LINK_TIMEOUT_S = 10.0
READY_TIMEOUT_S = 8.0
BACKLOG_S = LINK_TIMEOUT_S + READY_TIMEOUT_S + 2.0
CATCH_UP_FRAMES = 10
QUIET_RMS = 300.0
SPEAKER_LEAD_S = 0.1
IDLE_AFTER_S = 0.3
CALL_LIMIT_S = 900.0
NATURAL_ENDS = frozenset({"remote_stop", "idle_timeout", "overall_timeout"})


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


def _rms(pcm: bytes) -> float:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    return float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0


class MicrophoneFeed:
    """Reachy's microphone as a call's input: 24 kHz PCM16 frames.

    Frames wait here until the call is ready, for as long as connecting and admission can take,
    so Muse hears a request from its first word. Then they go out at the call's real-time pace. While
    more than ``CATCH_UP_FRAMES`` wait, quiet frames are skipped so the backlog shrinks without
    losing speech. ``sent`` counts frames the call has taken.
    """

    def __init__(self, input_rate: int) -> None:
        self._resample = _Resampler(input_rate, SAMPLE_RATE, output="s16")
        self._pending = np.zeros(0, dtype=np.int16)
        self._frames: deque[bytes] = deque(maxlen=int(BACKLOG_S * SAMPLE_RATE / FRAME_SAMPLES))
        self._arrived = asyncio.Event()
        self._closed = False
        self.sent = 0

    def push(self, samples: np.ndarray) -> None:
        if self._closed:
            return
        data = np.concatenate([self._pending, self._resample(samples)])
        whole = len(data) // FRAME_SAMPLES * FRAME_SAMPLES
        for start in range(0, whole, FRAME_SAMPLES):
            self._frames.append(data[start:start + FRAME_SAMPLES].astype("<i2").tobytes())
        self._pending = data[whole:]
        if whole:
            self._arrived.set()

    def close(self) -> None:
        self._closed = True
        self._arrived.set()

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        while True:
            while len(self._frames) > CATCH_UP_FRAMES and _rms(self._frames[0]) < QUIET_RMS:
                self._frames.popleft()
            if self._frames:
                self.sent += 1
                return self._frames.popleft()
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
    stretch of speech, ``on_audible`` hears every push with the sentence it belongs to and when
    its sound ends, and ``on_idle`` runs once the queue has drained and the last push has played.
    ``reply_started`` is when the latest reply's first chunk began to play.
    """

    def __init__(self, hardware, motion: ReplyMotion, *, on_audible: Callable[[Optional[str], float], None],
                 on_start: Callable[[], Awaitable[None]], on_idle: Callable[[], Awaitable[None]]) -> None:
        self._hardware = hardware
        self._motion = motion
        self._on_audible = on_audible
        self._on_start = on_start
        self._on_idle = on_idle
        self._timed = callable(getattr(hardware, "finish_sentence", None))
        self._rate = hardware.output_sample_rate
        self._resample = _Resampler(SAMPLE_RATE, self._rate, output="flt")
        self._queue: asyncio.Queue[_Chunk] = asyncio.Queue()
        self._cursor = PlaybackCursor()
        self._task: Optional[asyncio.Task] = None
        self._flushing: Optional[asyncio.Future] = None
        self._speaking = False
        self._sentence_open = False
        self._extras: dict = {}
        self._audible_text: Optional[str] = None
        self._replies: list = []
        self.reply_started: Optional[float] = None

    @property
    def busy(self) -> bool:
        return self._speaking or not self._queue.empty()

    def chunk(self, response_id: Optional[str], offset: int, pcm: bytes) -> None:
        self._queue.put_nowait(_Chunk(response_id, offset, pcm))
        self._start()

    def flush_now(self) -> asyncio.Future:
        """Drop queued and playing reply audio; chunks that arrive afterwards play once it is gone."""
        flushing = self._flushing is not None and not self._flushing.done()
        if not self.busy and not flushing:
            done = asyncio.get_running_loop().create_future()
            done.set_result(None)
            return done
        had_audio = self.busy
        while not self._queue.empty():
            self._queue.get_nowait()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
        self._flushing = asyncio.ensure_future(self._cleared(self._flushing, task, had_audio))
        return self._flushing

    async def close(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
        await asyncio.gather(*(item for item in (task, self._flushing) if item is not None),
                             return_exceptions=True)

    def _start(self) -> None:
        if (self._task is None or self._task.done()) and (self._flushing is None or self._flushing.done()):
            self._task = asyncio.create_task(self._pace(), name="reachy-realtime-speaker")

    async def _cleared(self, previous, task, had_audio: bool) -> None:
        await asyncio.gather(*(item for item in (previous, task) if item is not None), return_exceptions=True)
        self._cursor = PlaybackCursor()
        self._sentence_open = False
        self._extras = {}
        self._audible_text = None
        try:
            if had_audio:
                await asyncio.to_thread(self._hardware.clear_audio)
        finally:
            if self._speaking:
                self._speaking = False
                await self._on_idle()
            if not self._queue.empty():
                self._task = asyncio.create_task(self._pace(), name="reachy-realtime-speaker")

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
                await self._on_start()
            if item.response_id not in self._replies:
                self._replies.append(item.response_id)
                self.reply_started = time.monotonic()
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
        self._on_audible(self._audible_text, max(self._cursor.end, now) + len(samples) / self._rate)
        if cancelled:
            raise asyncio.CancelledError

    async def _finish(self, player: PCMPlayer) -> None:
        if self._timed and self._sentence_open:
            self._hardware.finish_sentence()
        self._sentence_open = False
        await player.finish()
        self._audible_text = None
        replies, self._replies = self._replies, []
        for response_id in replies:
            report = self._motion.report(response_id)
            if report is not None:
                log.info("Reachy's motion for Muse's realtime reply: %s", report)
        if self._queue.empty():
            self._speaking = False
            await self._on_idle()


class RealtimeCall:
    """One call: the SDK ``VoiceCall`` plus Reachy's microphone feed, speaker and reply motion.

    ``latest_response`` is the reply Muse started last, and ``muse_moved`` whether Muse has moved
    Reachy itself since then.
    """

    def __init__(self, feed: MicrophoneFeed, speaker: RealtimeSpeaker, motion: ReplyMotion, *,
                 on_working: Callable[[bool], None], on_reply_done: Callable[[Optional[str]], None]) -> None:
        self.voice_call = None
        self.feed = feed
        self.speaker = speaker
        self.motion = motion
        self._on_working = on_working
        self._on_reply_done = on_reply_done
        self.working = False
        self.awaiting_answer = False
        self.latest_response: Optional[str] = None
        self.muse_moved = False
        self.reason: Optional[str] = None
        self.failure: Optional[BaseException] = None
        self.opened_at = time.monotonic()
        self._task: Optional[asyncio.Task] = None
        self._closing = False

    def on_event(self, event) -> None:
        """The SDK's event callback: it runs on the event loop and an exception here would end the call."""
        kind = type(event).__name__
        if kind == "AudioChunk":
            self.speaker.chunk(event.response_id, event.offset, event.pcm)
        elif kind == "PlaybackFlushed":
            self.speaker.flush_now()
        elif kind == "Working":
            self.working = event.working
        elif kind == "ResponseStarted":
            self.latest_response = event.response_id
            self.muse_moved = False
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
                self.motion.done(event.response_id)
                if event.status in (None, "completed"):
                    self._on_reply_done(event.response_id)
            elif kind == "Working":
                self._on_working(event.working)
            elif kind == "PlaybackFlushed":
                self.motion.flushed(event.response_ids)
        except Exception:
            log.exception("Reachy's realtime motion failed on a %s event", kind)

    def start(self, build: Callable[["RealtimeCall"], Awaitable[object]]) -> None:
        """Run the call that ``build`` makes; the microphone feed buffers until it is ready."""
        self._task = asyncio.create_task(self._run(build), name="reachy-realtime-call")

    async def _run(self, build) -> None:
        try:
            self.voice_call = await build(self)
            self.reason = await self.voice_call.run(self.feed, _SilentOutput(self), realtime_input=True)
            log.info("Muse realtime call ended: %s after %.1f s", self.reason, time.monotonic() - self.opened_at)
        except asyncio.CancelledError:
            if not self._closing:
                raise
        except Exception as exc:
            self.failure = exc
            log.warning("Muse realtime call failed (%s) after %.1f s", type(exc).__name__,
                        time.monotonic() - self.opened_at)

    @property
    def done(self) -> bool:
        return self._task is not None and self._task.done()

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

    async def close(self) -> None:
        """End the call. The SDK's cleanup still hangs up with Muse when this task is cancelled again."""
        self._closing = True
        self.feed.close()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self.speaker.close()
        self.motion.close()


class _SilentOutput:
    """The SDK's playback sink. Reachy plays accepted chunks itself; a flush still clears them."""

    def __init__(self, call: RealtimeCall) -> None:
        self._call = call

    def write(self, pcm: bytes) -> None:
        pass

    def flush(self) -> None:
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
             on_reply_done) -> RealtimeCall:
        """Start a call at once; it connects, if needed, and is admitted while the microphone buffers."""
        call = RealtimeCall(MicrophoneFeed(hardware.sample_rate),
                            RealtimeSpeaker(hardware, motion, on_audible=on_audible, on_start=on_start,
                                            on_idle=on_idle),
                            motion, on_working=on_working, on_reply_done=on_reply_done)
        call.start(self._build)
        log.info("Muse realtime call opening, %s duplex", "half" if self.half_duplex else "full")
        return call

    async def _build(self, call: RealtimeCall):
        factory = self._voice_call
        if factory is None:
            from musegadget.voice_call import VoiceCall as factory
        link = await self._ready_link()
        return factory(link.session, device_id=self.device_id, thread_id=self.thread_id,
                       half_duplex=self.half_duplex, ready_timeout=READY_TIMEOUT_S,
                       idle_timeout=CALL_LIMIT_S, overall_timeout=CALL_LIMIT_S, on_event=call.on_event)

    async def _ready_link(self) -> VoiceLink:
        if self._link is not None and not self._link.alive():
            await self._drop_link()
        if self._link is None:
            if self._connecting is None or self._connecting.done():
                self._connecting = asyncio.ensure_future(self._connect())
            self._link = await asyncio.shield(self._connecting)
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

    async def _drop_link(self) -> None:
        link, self._link = self._link, None
        if link is not None:
            with contextlib.suppress(Exception):
                await link.close()
