# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Reachy's side of the companion link, and the capabilities a companion answers.

Wire types stay in this module. Callers see Endpoint, Partial and SpokenLine, and every
failure is a CapabilityUnavailable that the failover wrappers turn into on-robot work.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import io
import logging
import random
import re
import ssl
import time
import wave
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Callable, Dict, FrozenSet, List, Optional, Tuple
from urllib.parse import urlsplit

from reachy_muse.contract.replies import ReplyStyle
from reachy_muse.contract.intent import Intent
from reachy_muse.capabilities import CapabilityUnavailable, Endpoint, Heard, HeardAudio, Motion, Partial, SpokenLine
from reachy_muse.device_command import DeviceCommand
from reachy_muse.protocol import (
    MAX_ALREADY_SAID, MAX_ANIMATION_MS, MAX_COMMAND_TEXT_BYTES, MAX_INTENT_BYTES, MAX_LINE_BYTES, MAX_MIC_PCM_BYTES,
    MAX_OUTSTANDING_ANIMATIONS, MAX_OUTSTANDING_NARRATIONS, MAX_REPLY_BYTES, MAX_REQUEST_BYTES, MAX_STATUS_BYTES,
    MAX_TEXT_FRAME_BYTES, MIN_ANIMATION_MS, VERSION, Animate, AnimateResult, AudioKind, Classify,
    ClassifyResult, CloseHow, Error,
    ErrorCode, HearClose, HearEndpoint, HearOpen, HearPartial, Hello, Narrate, NarratedLine, NarrateOp,
    NarrateResult, Op, ProtocolError, Reason, Sender, SequenceCheck, SpeakCancel, SpeakDone, SpeakStart,
    VoiceRole, Welcome, decode, decode_audio, encode, encode_audio,
)

log = logging.getLogger(__name__)

PING_INTERVAL_S = 5.0
# Recorded motion-load p99 is 5.105 s (ten samples); round 3x up to 20 s.
# This is a conservative stall allowance, not a measured network RTT p99.
PING_TIMEOUT_S = 20.0
CLOSE_TIMEOUT_S = 0.1
HANDSHAKE_TIMEOUT_S = 10.0
BACKOFF_S = (0.2, 5.0)
HEALTHY_S = 10.0                   # a session shorter than this is a failed attempt, not a recovery
READY_S = 0.1                     # reject immediate post-welcome failures before routing new turns
# From the robot's silence endpoint to the companion's forced endpoint. Healthy finals take
# 0.66 s alone and 1.12 s at the median with motion; a later one is transcribed on the robot.
ENDPOINT_WAIT_S = 1.5
# Pongs prove the socket, not the voice. Measured loaded first audio peaks at 1.47 s and the
# 06:12 incident's slowest line took 3.6 s; a voice silent for longer restarts on the robot.
SPEECH_STALL_S = 6.0
_PIN = re.compile(r"sha256:([0-9a-f]{64})")
_FATAL = (ErrorCode.AUTH, ErrorCode.VERSION, ErrorCode.INTERNAL, ErrorCode.MODEL)


def parse_pin(pin: str) -> str:
    """The certificate digest from a pairing pin: "sha256:" and 64 lowercase hex digits."""
    match = _PIN.fullmatch(pin)
    if match is None:
        raise ValueError("the companion pin must be sha256: followed by 64 lowercase hex digits")
    return match[1]


def parse_url(url: str, *, insecure: bool = False) -> str:
    """The companion's base URL as `companion pair` prints it: wss://HOST:PORT, no path."""
    parts = urlsplit(url)
    if parts.scheme not in ("wss", "ws") or not parts.hostname or parts.path not in ("", "/") \
            or parts.query or parts.fragment or parts.username or parts.password:
        raise ValueError("the companion URL must look like wss://HOST:PORT")
    if parts.scheme == "ws" and not insecure:
        raise ValueError("the companion URL must use wss://; plain ws:// is for tests only")
    return f"{parts.scheme}://{parts.netloc}"


class LinkState(Enum):
    CONNECTING = "connecting"      # no attempt has finished yet
    UP = "up"                      # the companion answered welcome
    DOWN = "down"


class _Done:
    pass


_DONE = _Done()


class _Connection:
    """One WebSocket session. Nothing survives it: its streams and narrations fail with it."""

    def __init__(self, ws, ops: Tuple[Op, ...]):
        self.ws = ws
        self.ops = frozenset(ops)
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.hearing: Dict[int, _HearStream] = {}
        self.speaking: Dict[int, asyncio.Queue] = {}
        self.narrations: Dict[int, asyncio.Future] = {}
        self.animations: Dict[int, asyncio.Future] = {}
        self.sequence = SequenceCheck()
        self.closed = False
        self.last_progress_at = time.monotonic()
        self._next_hear = -1
        self._next_speak = 0

    def stream_id(self, op: Op) -> int:
        if op is Op.HEAR:
            self._next_hear = self._next_hear + 2 if self._next_hear < 65533 else 1
            return self._next_hear
        self._next_speak = self._next_speak + 2 if self._next_speak < 65534 else 2
        return self._next_speak

    def send(self, frame) -> None:
        if self.closed:
            raise CapabilityUnavailable("the companion link is down")
        self.outbox.put_nowait(frame if isinstance(frame, bytes) else encode(frame))

    def fail(self, reason: str) -> None:
        if self.closed:
            return
        self.closed = True
        for queue in self.speaking.values():
            queue.put_nowait(CapabilityUnavailable(reason))
        for future in (*self.narrations.values(), *self.animations.values()):
            if not future.done():
                future.set_exception(CapabilityUnavailable(reason))
        for stream in self.hearing.values():
            stream.fail(CapabilityUnavailable(reason))


class _HearStream:
    def __init__(self, connection: _Connection, stream: int):
        self.connection = connection
        self.id = stream
        self.audio = bytearray()
        self.inbox: List[object] = []
        self.failure: Optional[CapabilityUnavailable] = None
        self.finished_at: Optional[float] = None   # when the robot's VAD ended the utterance
        self.endpoint: Optional[HearEndpoint] = None
        self.settled = asyncio.Event()             # an endpoint or a failure arrived
        self._seq = 0

    def send(self, pcm: bytes) -> None:
        self.audio.extend(pcm)
        for offset in range(0, len(pcm), MAX_MIC_PCM_BYTES):
            self.connection.send(encode_audio(AudioKind.MIC, self.id, self._seq, pcm[offset:offset + MAX_MIC_PCM_BYTES]))
            self._seq += 1

    def close(self, how: CloseHow) -> None:
        if how is CloseHow.ABORT:
            self.connection.hearing.pop(self.id, None)
        if not self.connection.closed:
            self.connection.send(HearClose(self.id, how))

    def finish(self) -> None:
        self.finished_at = time.monotonic()
        self.close(CloseHow.FINISH)

    def fail(self, failure: CapabilityUnavailable) -> None:
        self.failure = failure
        self.settled.set()


def _clip(text: str, limit: int) -> str:
    return text.encode("utf-8")[:limit].decode("utf-8", "ignore")


def _fit(text: str, limit: int, boundary: str) -> str:
    """``text`` cut to ``limit`` UTF-8 bytes at the last ``boundary`` that fits, else at the limit."""
    head = _clip(text, limit)
    if head == text:
        return text
    # One character past the cut, so a boundary is not mistaken for one at the cut itself.
    ends = [m.end() for m in re.finditer(boundary, text[:len(head) + 1]) if m.end() <= len(head)]
    return head[:ends[-1]].rstrip() if ends else head


class CompanionLink:
    """The robot's one WebSocket to its companion: pinned TLS, handshake, liveness and reconnect.

    ``transitions`` records every state change. The link is up only after ``welcome``.
    """

    def __init__(self, url: str, *, token: str, out_rate: int, pin: Optional[str] = None,
                 insecure: bool = False, robot: str = "Reachy Mini",
                 ops: Tuple[Op, ...] = (Op.HEAR, Op.SPEAK, Op.NARRATE, Op.CLASSIFY, Op.ANIMATE),
                 backoff_s: Tuple[float, float] = BACKOFF_S, handshake_timeout_s: float = HANDSHAKE_TIMEOUT_S,
                 ping_interval_s: float = PING_INTERVAL_S, ping_timeout_s: float = PING_TIMEOUT_S,
                 healthy_s: float = HEALTHY_S):
        self.url = parse_url(url, insecure=insecure) + "/v1"
        if self.url.startswith("wss:") and pin is None:
            raise ValueError("a wss:// companion needs its certificate pin")
        if self.url.startswith("ws:") and pin is not None:
            raise ValueError("a pinned companion needs wss://; plain ws:// has no certificate to pin")
        self._digest = parse_pin(pin) if pin is not None else None
        self._token = token
        self.out_rate = out_rate
        self._robot = robot
        self._ops = ops
        self._backoff_s = backoff_s
        self._handshake_timeout_s = handshake_timeout_s
        self._ping = (ping_interval_s, ping_timeout_s)
        self._healthy_s = healthy_s
        self.transitions: List[LinkState] = [LinkState.CONNECTING]
        self.models: Tuple[Tuple[str, str], ...] = ()
        self._connection: Optional[_Connection] = None
        self._session: Optional[asyncio.Future] = None
        self._settled: Optional[asyncio.Event] = None
        self._outages = 0
        self._reported_outages = 0
        self._narrate_id = 0
        self._up_since: Optional[float] = None
        self._last_session_healthy = False

    @classmethod
    def from_config(cls, saved: dict, *, out_rate: int) -> CompanionLink:
        return cls(saved["url"], pin=saved["pin"], token=saved["token"], out_rate=out_rate)

    @property
    def state(self) -> LinkState:
        return self.transitions[-1]

    @property
    def up(self) -> bool:
        """Ready for new turns after welcome; outage accounting has a longer health window."""
        ready = self._up_since is not None and time.monotonic() - self._up_since >= min(READY_S, self._healthy_s)
        return self.can(Op.NARRATE) and (self._outages == 0 or ready)

    def _healthy(self) -> bool:
        return self._up_since is not None and time.monotonic() - self._up_since >= self._healthy_s

    def can(self, op: Op) -> bool:
        return (self.state is LinkState.UP and self._connection is not None
                and not self._connection.closed and op in self._connection.ops)

    @property
    def last_progress_at(self) -> float:
        return self._connection.last_progress_at if self._connection is not None else 0.0

    @property
    def responsive(self) -> bool:
        """A received frame or pong proves life, regardless of inference speed."""
        return (self._connection is not None and not self._connection.closed
                and time.monotonic() - self.last_progress_at < sum(self._ping))

    def take_outage(self) -> bool:
        if self._outages == self._reported_outages:
            return False
        self._reported_outages = self._outages
        return True

    async def settled(self) -> None:
        """Wait until the first connection attempt has an outcome."""
        if self._settled is None:
            self._settled = asyncio.Event()
        if self.state is LinkState.CONNECTING:
            await self._settled.wait()

    def _set(self, state: LinkState) -> None:
        """Record a state change. Only the first outage, or one after a healthy session, is counted."""
        if state is self.state:
            return
        if state is LinkState.UP:
            self._up_since = time.monotonic()
        else:
            self._last_session_healthy = self._healthy()
            if self._last_session_healthy or self._outages == 0:
                self._outages += 1
            self._up_since = None
        self.transitions.append(state)
        log.info("Reachy companion link %s", state.value)

    async def run(self) -> None:
        if self._settled is None:
            self._settled = asyncio.Event()
        failures = 0
        low, high = self._backoff_s
        delay = low
        while True:
            self._last_session_healthy = False
            self._session = asyncio.ensure_future(self._connect())
            try:
                await asyncio.wait({self._session})
            except asyncio.CancelledError:
                self._session.cancel()
                await asyncio.gather(self._session, return_exceptions=True)
                raise
            finally:
                if self._connection is not None:
                    self._connection.fail("the companion link closed")
                self._connection = None
                self._set(LinkState.DOWN)
                self._settled.set()
            failures = 0 if self._last_session_healthy else failures + 1
            if not self._session.cancelled() and self._session.exception() is not None:
                error = self._session.exception()
                (log.warning if failures <= 1 else log.debug)(
                    "Reachy companion link failed: %s: %s", type(error).__name__, error)
            delay = low if failures <= 1 else min(high, delay * 2)
            await asyncio.sleep(max(low, random.uniform(delay / 2, delay)))

    def _ssl(self) -> Optional[ssl.SSLContext]:
        if not self.url.startswith("wss:"):
            return None
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    async def _open(self):
        from websockets.asyncio.client import connect

        options = dict(open_timeout=self._handshake_timeout_s, ping_interval=None,
                       close_timeout=CLOSE_TIMEOUT_S,
                       max_size=2 * MAX_TEXT_FRAME_BYTES)
        if "proxy" in inspect.signature(connect).parameters:
            options["proxy"] = None
        context = self._ssl()
        if context is not None:
            options["ssl"] = context
        return await connect(self.url, **options)

    async def _connect(self) -> None:
        ws = await self._open()
        try:
            if self._digest is not None:
                der = ws.transport.get_extra_info("ssl_object").getpeercert(binary_form=True)
                if not hmac.compare_digest(hashlib.sha256(der or b"").hexdigest(), self._digest):
                    raise ConnectionError("the companion's certificate does not match its pin")
            await ws.send(encode(Hello((VERSION,), self._token, self._ops, self.out_rate, self._robot)))
            first = await asyncio.wait_for(ws.recv(), self._handshake_timeout_s)
            if not isinstance(first, str):
                raise ProtocolError(ErrorCode.BAD_REQUEST, Reason.WRONG_TYPE, "binary frame before welcome")
            welcome = decode(first, sender=Sender.COMPANION)
            if isinstance(welcome, Error):
                raise ConnectionError(f"the companion refused hello: {welcome.code.value}: {welcome.message}")
            if not isinstance(welcome, Welcome) or welcome.version != VERSION:
                raise ProtocolError(ErrorCode.BAD_REQUEST, Reason.WRONG_TYPE, "expected welcome")
            connection = _Connection(ws, welcome.ops)
            self.models = welcome.models
            self._connection = connection
            self._set(LinkState.UP)
            self._settled.set()
            writer = asyncio.ensure_future(self._write(connection))
            heartbeat = asyncio.create_task(self._heartbeat(connection))
            try:
                async for frame in ws:
                    self._receive(connection, frame)
            finally:
                connection.fail("the companion link closed")
                self._set(LinkState.DOWN)
                writer.cancel()
                heartbeat.cancel()
                await asyncio.gather(writer, heartbeat, return_exceptions=True)
        finally:
            try:
                await asyncio.wait_for(ws.close(), .5)
            except Exception:
                ws.transport.abort()

    async def _write(self, connection: _Connection) -> None:
        while True:
            await connection.ws.send(await connection.outbox.get())

    async def _heartbeat(self, connection: _Connection) -> None:
        """Inference can be slow while its socket is healthy. Only silence kills it."""
        interval, inactivity = self._ping

        async def while_alive(waiting):
            task = asyncio.ensure_future(waiting)
            try:
                while True:
                    if task.done():
                        return task.result()
                    remaining = inactivity - (time.monotonic() - connection.last_progress_at)
                    if remaining <= 0:
                        connection.fail("the companion sent no frames or pong")
                        raise TimeoutError
                    try:
                        return await asyncio.wait_for(asyncio.shield(task), remaining)
                    except asyncio.TimeoutError:
                        if task.done():
                            raise
                        if time.monotonic() - connection.last_progress_at >= inactivity:
                            connection.fail("the companion sent no frames or pong")
                            raise
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        try:
            while True:
                await asyncio.sleep(interval)
                pong = await while_alive(connection.ws.ping())
                await while_alive(pong)
                connection.last_progress_at = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception:
            connection.fail("the companion heartbeat failed")
            connection.ws.transport.abort()

    def _receive(self, connection: _Connection, frame) -> None:
        if isinstance(frame, bytes):
            audio = decode_audio(frame)
            queue = connection.speaking.get(audio.stream)
            if audio.kind is not AudioKind.SPEECH:
                raise ProtocolError(ErrorCode.BAD_REQUEST, Reason.WRONG_DIRECTION, "microphone audio from the companion")
            connection.last_progress_at = time.monotonic()
            if queue is not None:
                connection.sequence.accept(audio)
                queue.put_nowait(audio.pcm)
            return
        message = decode(frame, sender=Sender.COMPANION)
        connection.last_progress_at = time.monotonic()
        if isinstance(message, (HearPartial, HearEndpoint)):
            stream = connection.hearing.get(message.stream)
            if stream is not None:
                if isinstance(message, HearEndpoint):
                    del connection.hearing[message.stream]
                if stream.finished_at is None:
                    stream.inbox.append(message)
                elif isinstance(message, HearEndpoint):
                    stream.endpoint = message
                    stream.settled.set()
        elif isinstance(message, SpeakDone):
            queue = connection.speaking.pop(message.stream, None)
            if queue is not None:
                queue.put_nowait(_DONE)
        elif isinstance(message, (NarrateResult, ClassifyResult)):
            future = connection.narrations.pop(message.id, None)
            if future is not None and not future.done():
                future.set_result(message if isinstance(message, NarrateResult) else message.command)
        elif isinstance(message, AnimateResult):
            future = connection.animations.pop(message.id, None)
            if future is not None and not future.done():
                future.set_result(message)
        elif isinstance(message, Error):
            self._error(connection, message)

    def _error(self, connection: _Connection, error: Error) -> None:
        if error.code in _FATAL:
            raise ConnectionError(f"the companion failed: {error.code.value}: {error.message}")
        failure = CapabilityUnavailable(f"companion {error.code.value}: {error.message}", link_down=False)
        if error.stream is not None and error.stream in connection.hearing:
            connection.hearing.pop(error.stream).fail(failure)
        elif error.stream is not None and error.stream in connection.speaking:
            connection.speaking.pop(error.stream).put_nowait(failure)
            connection.sequence.forget(error.stream)
        elif error.id is not None and (error.id in connection.narrations or error.id in connection.animations):
            future = connection.narrations.pop(error.id, None) or connection.animations.pop(error.id)
            if not future.done():
                future.set_exception(failure)
        else:
            log.warning("Reachy's companion reported %s: %s", error.code.value, error.message)

    def _live(self, op: Op) -> _Connection:
        if not self.can(op):
            raise CapabilityUnavailable(f"the companion cannot {op.value} now", link_down=self.state is not LinkState.UP)
        return self._connection

    def open_hear(self) -> _HearStream:
        connection = self._live(Op.HEAR)
        stream = _HearStream(connection, connection.stream_id(Op.HEAR))
        connection.send(HearOpen(stream.id, 0))
        connection.hearing[stream.id] = stream
        return stream

    async def speak(self, text: str, voice: VoiceRole) -> AsyncIterator[bytes]:
        """The companion's speech for one line, as s16le PCM at ``out_rate``."""
        connection = self._live(Op.SPEAK)
        if not text or len(text.encode("utf-8")) > MAX_LINE_BYTES:
            raise CapabilityUnavailable("the line does not fit one companion speech request", link_down=False)
        stream = connection.stream_id(Op.SPEAK)
        queue: asyncio.Queue = asyncio.Queue()
        connection.speaking[stream] = queue
        connection.send(SpeakStart(stream, text, voice))
        finished = False
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), SPEECH_STALL_S)
                except asyncio.TimeoutError:
                    raise CapabilityUnavailable("the companion voice stalled", link_down=False) from None
                if item is _DONE:
                    finished = True
                    return
                if isinstance(item, CapabilityUnavailable):
                    finished = True
                    raise item
                yield item
        finally:
            connection.sequence.forget(stream)
            if not finished and connection.speaking.pop(stream, None) is not None and not connection.closed:
                connection.send(SpeakCancel(stream))

    async def narrate(self, op: NarrateOp, request: str, *, status: Optional[str] = None,
                      already_said: Tuple[str, ...] = (), reply: Optional[str] = None,
                      previous: Optional[str] = None, may_chat: bool = False) -> NarrateResult:
        return await self._ask(Op.NARRATE, lambda identity: Narrate(
            identity, op, _clip(request, MAX_REQUEST_BYTES), status,
            tuple(_clip(line, MAX_LINE_BYTES) for line in already_said[-MAX_ALREADY_SAID:]), reply, previous, may_chat))

    async def classify(self, text: str, allowed: FrozenSet[DeviceCommand]) -> Optional[DeviceCommand]:
        return await self._ask(Op.CLASSIFY, lambda identity: Classify(
            identity, _clip(text, MAX_COMMAND_TEXT_BYTES), tuple(command for command in DeviceCommand if command in allowed)))

    async def animate(self, intent: str, duration_s: float) -> AnimateResult:
        duration_ms = min(max(round(duration_s * 1000), MIN_ANIMATION_MS), MAX_ANIMATION_MS)
        return await self._ask(Op.ANIMATE, lambda identity: Animate(identity, _clip(intent, MAX_INTENT_BYTES), duration_ms))

    async def _ask(self, op: Op, build: Callable[[int], object]):
        connection = self._live(op)
        waiting, limit = ((connection.animations, MAX_OUTSTANDING_ANIMATIONS) if op is Op.ANIMATE
                          else (connection.narrations, MAX_OUTSTANDING_NARRATIONS))
        if len(waiting) >= limit:
            raise CapabilityUnavailable(f"too many {op.value} requests outstanding", link_down=False)
        self._narrate_id = (self._narrate_id + 1) % 2 ** 32
        future = asyncio.get_running_loop().create_future()
        waiting[self._narrate_id] = future
        identity = self._narrate_id
        sent = False
        cancelled_animation = False
        try:
            connection.send(build(identity))
            sent = True
            return await future
        except asyncio.CancelledError:
            # The protocol has no animate cancellation. Keep the slot until the
            # companion answers so an abandoned request still counts against its limit.
            cancelled_animation = sent and op is Op.ANIMATE
            raise
        finally:
            if not cancelled_animation:
                waiting.pop(identity, None)


def _wav(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm)
    return buffer.getvalue()


class _HeardTurn:
    """A turn the companion already ended and transcribed; its audio ends at ``end_sample``."""

    def __init__(self, endpoint: Endpoint, truncated: bool, end_sample: int):
        self._endpoint = endpoint
        self.truncated = truncated
        self.end_sample = end_sample

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        pass

    async def endpoint(self) -> Optional[Endpoint]:
        return self._endpoint


class _FinishedTurn:
    """A turn the robot's VAD ended first; the companion's forced endpoint brings its words.

    The words must arrive within ``ENDPOINT_WAIT_S`` of the robot's endpoint. A companion that
    is alive but busy fails only this turn, so the robot can transcribe the same audio.
    """

    def __init__(self, stream: _HearStream, truncated: bool, end_sample: int):
        self._stream = stream
        self._cancelled = False
        self.truncated = truncated
        self.end_sample = end_sample

    @property
    def wav(self) -> bytes:
        return _wav(bytes(self._stream.audio))

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        self._cancelled = True
        self._stream.close(CloseHow.ABORT)
        self._stream.settled.set()

    async def endpoint(self) -> Optional[Endpoint]:
        stream = self._stream
        remaining = stream.finished_at + ENDPOINT_WAIT_S - time.monotonic()
        try:
            if not stream.settled.is_set():
                await asyncio.wait_for(stream.settled.wait(), max(0.0, remaining))
        except asyncio.TimeoutError:
            stream.close(CloseHow.ABORT)
            raise CapabilityUnavailable("the companion did not end the turn in time", link_down=False) from None
        if self._cancelled:
            raise asyncio.CancelledError
        if stream.endpoint is None:
            raise stream.failure
        return Endpoint(self.wav, stream.endpoint.text, stream.endpoint.forced)


class CompanionHearingTurn:
    """Robot VAD opens each hear stream; the earlier of the companion's endpoint and the robot's ends it.

    When the robot's VAD ends an utterance first, the stream closes with ``finish`` and the
    turn ends at once; its words follow with the companion's forced endpoint. Liveness of
    the link never holds a turn open.
    """

    supports_partials = True

    def __init__(self, link: CompanionLink, recorder):
        self._link = link
        self._recorder = recorder
        self._open: Optional[_HearStream] = None
        self._ended = deque()
        self._fed = 0
        self._vad_ends = deque()
        self.partial: Optional[Partial] = None

    @property
    def active(self) -> bool:
        return self._recorder.active or self._open is not None

    @property
    def speech_active(self) -> bool:
        return self._recorder.speech_active

    @property
    def quiet_s(self) -> float:
        return self._recorder.quiet_s

    def feed(self, samples) -> None:
        self._fed += len(samples)
        if self._recorder.feed(samples) is not None:
            self._vad_ends.append(self._fed)

    def finish_initial_capture(self) -> None:
        self._recorder.finish_initial_capture()

    def take(self) -> HeardAudio:
        from reachy_muse.voice_audio import SpeechAudio, SpeechEnd

        for event in self._recorder.take_audio_events():
            if isinstance(event, SpeechAudio):
                if self._open is None:
                    self._open = self._link.open_hear()
                self._open.send(event.pcm)
            elif isinstance(event, SpeechEnd) and self._open is not None:
                stream, self._open = self._open, None
                if event.accepted:
                    stream.finish()
                    self._ended.append(_FinishedTurn(stream, self._recorder.last_truncated,
                                                     self._vad_ends.popleft() if self._vad_ends else self._fed))
                else:
                    stream.close(CloseHow.ABORT)
                self.partial = None
        stream = self._open
        if stream is not None:
            messages, stream.inbox = stream.inbox, []
            for message in messages:
                if isinstance(message, HearPartial):
                    self.partial = Partial(message.text, message.rev, stream.id)
                    continue
                self._open = None
                self._recorder.reset()
                self.partial = None
                self._ended.append(_HeardTurn(Endpoint(_wav(bytes(stream.audio)), message.text, message.forced),
                                              False, self._fed))
                break
        if self._ended:
            return HeardAudio(self._ended.popleft())
        if stream is not None and stream.failure is not None:
            raise stream.failure
        return HeardAudio()

    def abort(self) -> None:
        if self._open is not None:
            self._open.close(CloseHow.ABORT)
        self._open = None
        self.partial = None


class CompanionHearing:
    transcribes = True

    def __init__(self, link: CompanionLink):
        self.link = link

    @property
    def available(self) -> bool:
        return self.link.can(Op.HEAR)

    async def start(self) -> None:
        # The link connects independently; local hearing works while it connects.
        pass

    def open_turn(self, sample_rate: int, *, silence_s: float, vad=None, recorder=None,
                  **options) -> CompanionHearingTurn:
        from reachy_muse.voice_audio import TurnRecorder

        if recorder is None:
            recorder = TurnRecorder(sample_rate, silence_s=silence_s, max_s=60.0, stream_audio=True,
                                    **({"vad": vad} if vad is not None else {}), **options)
        else:
            recorder.enable_streaming()
        return CompanionHearingTurn(self.link, recorder)

    async def transcribe(self, wav: bytes) -> str:
        raise CapabilityUnavailable("the companion transcribes live turns only", link_down=False)


class CompanionVoice:
    """Speech synthesized by the companion at the robot's output rate."""

    speaks_text = True

    def __init__(self, link: CompanionLink, role: VoiceRole = VoiceRole.MAIN):
        self.link = link
        self.role = role

    @property
    def available(self) -> bool:
        return self.link.can(Op.SPEAK)

    async def warm(self, output_rate: int) -> None:
        pass

    async def stream(self, line: SpokenLine, output_rate: int):
        import numpy as np

        if output_rate != self.link.out_rate:
            raise CapabilityUnavailable("the companion speaks at another rate", link_down=False)
        remaining = line.text.strip()
        while remaining:
            text = _fit(remaining, MAX_LINE_BYTES, r"\s+")
            remaining = remaining[len(text):].lstrip()
            speech = self.link.speak(text, self.role)
            try:
                async for pcm in speech:
                    yield np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
            finally:
                await speech.aclose()

    def prepare(self, line: SpokenLine, output_rate: int):
        return None


@dataclass(frozen=True)
class CompanionOpening:
    intent: Optional[Intent]
    lines: Tuple[SpokenLine, ...]


class CompanionNarrator:
    """The companion's narrator. Ids and request text only; never Muse identifiers."""

    def __init__(self, link: CompanionLink):
        self.link = link

    @property
    def available(self) -> bool:
        return self.link.can(Op.NARRATE)

    async def acknowledge(self, heard: Heard) -> CompanionOpening:
        result = await self.link.narrate(NarrateOp.ACKNOWLEDGE, heard.text,
                                         previous=heard.previous or None, already_said=heard.recently_said,
                                         may_chat=heard.may_chat)
        return CompanionOpening(result.intent, tuple(_spoken(line, "ack") for line in result.lines))

    async def say_progress(self, request: str, status: str,
                           already_said: Tuple[str, ...]) -> Optional[SpokenLine]:
        lines = await self.link.narrate(NarrateOp.PROGRESS, request, status=_fit(status, MAX_STATUS_BYTES, r"\S+"),
                                        already_said=already_said)
        return _spoken(lines.lines[0], "progress") if lines.lines else None

    async def lines(self, request: str, reply: str, style: ReplyStyle, *,
                    message_id: Optional[str] = None) -> Tuple[SpokenLine, ...]:
        if style is not ReplyStyle.PLAIN_SHORT or not reply:
            raise CapabilityUnavailable("the companion condenses only plain replies", link_down=False)
        if len(reply.encode("utf-8")) > MAX_REPLY_BYTES:
            raise CapabilityUnavailable("the reply is too long for the companion to condense", link_down=False)
        lines = await self.link.narrate(NarrateOp.LINES, request, reply=reply)
        return tuple(_spoken(line, "answer") for line in lines.lines)


class CompanionCommandModel:
    def __init__(self, link: CompanionLink):
        self.link = link

    @property
    def available(self) -> bool:
        return self.link.can(Op.CLASSIFY)

    async def classify(self, text: str, allowed: FrozenSet[DeviceCommand]) -> Optional[DeviceCommand]:
        return await self.link.classify(text, allowed)


class CompanionMotion:
    def __init__(self, link: CompanionLink):
        self.link = link

    @property
    def available(self) -> bool:
        return self.link.can(Op.ANIMATE)

    @property
    def library_available(self) -> bool:
        return self.available and dict(self.link.models).get(Op.ANIMATE.value) == "reachy-motion-library"

    async def generate(self, intent: str, duration_s: float) -> Optional[Motion]:
        result = await self.link.animate(intent, duration_s)
        return Motion(result.fps, result.frames) if result.frames else None


def _spoken(line: NarratedLine, role) -> SpokenLine:
    return SpokenLine(line.text, line.expression, role)
