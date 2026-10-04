"""The companion's WebSocket server, speaking the SDK's wire protocol v1.

One `_Session` per robot connection. Each hear stream, speak stream and narration
runs as its own task, so a slow engine never stalls the receive loop.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import ssl
from dataclasses import dataclass
from http import HTTPStatus
from typing import Awaitable, Callable, Dict, Optional, Sequence, Tuple, Union

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from musegadget.reachy_companion_protocol import (
    MAX_ERROR_MESSAGE_BYTES,
    MAX_OUTSTANDING_NARRATIONS,
    MAX_TEXT_FRAME_BYTES,
    VERSION,
    AudioFrame,
    AudioKind,
    CloseHow,
    Error,
    ErrorCode,
    HearClose,
    HearEndpoint,
    HearOpen,
    HearPartial,
    Hello,
    Message,
    Narrate,
    NarratedLine,
    NarrateOp,
    NarrateResult,
    Op,
    ProtocolError,
    Sender,
    SequenceCheck,
    SpeakCancel,
    SpeakDone,
    SpeakStart,
    Welcome,
    decode,
    decode_audio,
    encode,
    encode_audio,
)

from reachy_companion.engines import Endpoint, Hearing, HearingSession, Narrator, Partial, Voice

log = logging.getLogger("reachy_companion")

SUPPORTED_VERSIONS = (VERSION,)
PATH = "/v1"
HELLO_TIMEOUT = 5.0
CLOSE_POLICY_VIOLATION = 1008


@dataclass(frozen=True)
class Engines:
    """An op is offered exactly when its engine is present."""

    hearing: Optional[Hearing] = None
    voice: Optional[Voice] = None
    narrator: Optional[Narrator] = None

    def offered(self) -> Dict[Op, str]:
        engines = {Op.HEAR: self.hearing, Op.SPEAK: self.voice, Op.NARRATE: self.narrator}
        return {op: engine.name for op, engine in engines.items() if engine is not None}


class CompanionServer:
    """`async with CompanionServer(engines, token) as server:` serves until the block exits."""

    def __init__(self, engines: Engines, token: str, *, ssl_context: Optional[ssl.SSLContext] = None, host: str = "127.0.0.1", port: int = 0) -> None:
        self.engines = engines
        self.token = token.encode()
        self.ssl_context = ssl_context
        self.host = host
        self.port = port
        self._server: Optional[Server] = None

    @property
    def url(self) -> str:
        return f"{'wss' if self.ssl_context else 'ws'}://{self.host}:{self.port}{PATH}"

    async def __aenter__(self) -> CompanionServer:
        self._server = await serve(
            self._connection,
            self.host,
            self.port,
            ssl=self.ssl_context,
            process_request=_only_v1,
            ping_interval=2,
            ping_timeout=3,
            # Leave room above the codec's limit so an oversize frame gets error{bad_request}, not a bare close.
            max_size=4 * MAX_TEXT_FRAME_BYTES,
        )
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _connection(self, ws: ServerConnection) -> None:
        try:
            await _Session(self, ws).run()
        except ConnectionClosed:
            pass


def _only_v1(connection: ServerConnection, request):
    if request.path != PATH:
        return connection.respond(HTTPStatus.NOT_FOUND, f"the companion serves {PATH}\n")
    return None


@dataclass
class _Hear:
    session: HearingSession
    queue: "asyncio.Queue[Union[bytes, CloseHow]]"
    task: Optional[asyncio.Task] = None


@dataclass
class _Speech:
    task: Optional[asyncio.Task] = None
    frames: int = 0


_NARRATE: Dict[NarrateOp, Callable[[Narrator, Narrate], Awaitable[Sequence[NarratedLine]]]] = {
    NarrateOp.ACKNOWLEDGE: lambda narrator, m: narrator.acknowledge(m.request),
    NarrateOp.PROGRESS: lambda narrator, m: narrator.progress(m.request, m.status or "", m.already_said),
    NarrateOp.LINES: lambda narrator, m: narrator.lines(m.request, m.reply or ""),
}


class _Session:
    def __init__(self, server: CompanionServer, ws: ServerConnection) -> None:
        self.engines = server.engines
        self.token = server.token
        self.ws = ws
        self.ops: Tuple[Op, ...] = ()
        self.out_rate = 0
        self.sequence = SequenceCheck()
        self.hearing: Dict[int, _Hear] = {}
        self.speaking: Dict[int, _Speech] = {}
        self.narrating: Dict[int, asyncio.Task] = {}

    async def run(self) -> None:
        if not await self._handshake():
            return
        try:
            async for frame in self.ws:
                try:
                    if isinstance(frame, bytes):
                        await self._audio(decode_audio(frame))
                    else:
                        await self._message(decode(frame, sender=Sender.ROBOT))
                except ProtocolError as error:
                    await self._send(_error(error.code, str(error)))
                except Exception as error:
                    log.exception("handling a frame failed")
                    await self._send(_error(ErrorCode.INTERNAL, f"{type(error).__name__}: {error}"))
        finally:
            await self._stop_everything()

    async def _handshake(self) -> bool:
        try:
            first = await asyncio.wait_for(self.ws.recv(), HELLO_TIMEOUT)
        except asyncio.TimeoutError:
            await self.ws.close(CLOSE_POLICY_VIOLATION, "no hello")
            return False
        try:
            hello = decode(first, sender=Sender.ROBOT) if isinstance(first, str) else None
        except ProtocolError:
            hello = None
        if not isinstance(hello, Hello) or not hmac.compare_digest(hello.token.encode(), self.token):
            log.warning("refused %s: no hello with the paired token", self.ws.remote_address)
            return await self._refuse(ErrorCode.AUTH, "the first frame must be hello with the paired token")
        versions = [v for v in hello.versions if v in SUPPORTED_VERSIONS]
        if not versions:
            return await self._refuse(ErrorCode.VERSION, f"this companion speaks versions {list(SUPPORTED_VERSIONS)}")
        offered = self.engines.offered()
        self.ops = tuple(op for op in dict.fromkeys(hello.ops) if op in offered)
        self.out_rate = hello.out_rate
        await self._send(Welcome(max(versions), self.ops, tuple((op.value, offered[op]) for op in self.ops)))
        log.info("robot %r connected from %s with ops %s", hello.robot, self.ws.remote_address, [op.value for op in self.ops])
        return True

    async def _refuse(self, code: ErrorCode, message: str) -> bool:
        await self._send(_error(code, message))
        await self.ws.close(CLOSE_POLICY_VIOLATION, code.value)
        return False

    async def _send(self, message: Message) -> None:
        await self.ws.send(encode(message))

    async def _message(self, message: Message) -> None:
        op, handle = _ROUTES[type(message)]
        if op is not None and op not in self.ops:
            await self._send(_error(ErrorCode.BAD_REQUEST, f"op {op.value} was not granted in welcome", **_target(message)))
            return
        await handle(self, message)

    async def _second_hello(self, message: Hello) -> None:
        await self._send(_error(ErrorCode.BAD_REQUEST, "hello was already received"))

    # ---- hear

    async def _hear_open(self, message: HearOpen) -> None:
        if message.stream in self.hearing:
            await self._send(_error(ErrorCode.BAD_REQUEST, f"hear stream {message.stream} is already open", stream=message.stream))
            return
        assert self.engines.hearing is not None
        hear = _Hear(self.engines.hearing.open(message.pre_roll_frames), asyncio.Queue())
        self.hearing[message.stream] = hear
        hear.task = asyncio.create_task(self._guard(lambda: self._hear(message.stream, hear), stream=message.stream))

    async def _hear_close(self, message: HearClose) -> None:
        hear = self.hearing.get(message.stream)
        if hear is not None:
            hear.queue.put_nowait(message.how)

    async def _audio(self, frame: AudioFrame) -> None:
        if frame.kind is not AudioKind.MIC:
            await self._send(_error(ErrorCode.BAD_REQUEST, "the robot sends only microphone audio", stream=frame.stream))
            return
        hear = self.hearing.get(frame.stream)
        if hear is None:
            return  # frames still in flight after an endpoint, a close or a gap
        try:
            self.sequence.accept(frame)
        except ProtocolError as error:
            self._end_hearing(frame.stream, hear)
            hear.queue.put_nowait(CloseHow.ABORT)
            await self._send(_error(ErrorCode.GAP, str(error), stream=frame.stream))
            return
        hear.queue.put_nowait(frame.pcm)

    async def _hear(self, stream: int, hear: _Hear) -> None:
        rev = 0
        try:
            while True:
                item = await hear.queue.get()
                if item is CloseHow.ABORT:
                    return
                if item is CloseHow.FINISH:
                    text = await hear.session.finish()
                    self._end_hearing(stream, hear)
                    await self._send(HearEndpoint(stream, text, True))
                    return
                for event in await hear.session.feed(item):
                    if isinstance(event, Partial):
                        await self._send(HearPartial(stream, rev, event.text))
                        rev += 1
                    elif isinstance(event, Endpoint):
                        self._end_hearing(stream, hear)
                        await self._send(HearEndpoint(stream, event.text, False))
                        return
        finally:
            self._end_hearing(stream, hear)
            await hear.session.close()

    def _end_hearing(self, stream: int, hear: _Hear) -> None:
        if self.hearing.get(stream) is hear:
            del self.hearing[stream]
            self.sequence.forget(stream)

    # ---- speak

    async def _speak_start(self, message: SpeakStart) -> None:
        if message.stream in self.speaking:
            await self._send(_error(ErrorCode.BAD_REQUEST, f"speak stream {message.stream} is already speaking", stream=message.stream))
            return
        speech = self.speaking[message.stream] = _Speech()
        speech.task = asyncio.create_task(self._guard(lambda: self._speak(message, speech), stream=message.stream))

    async def _speak_cancel(self, message: SpeakCancel) -> None:
        speech = self.speaking.pop(message.stream, None)
        if speech is None:
            return  # idempotent: already done or cancelled
        assert speech.task is not None
        speech.task.cancel()
        await asyncio.gather(speech.task, return_exceptions=True)
        await self._send(SpeakDone(message.stream, speech.frames))

    async def _speak(self, message: SpeakStart, speech: _Speech) -> None:
        assert self.engines.voice is not None
        frame_bytes = self.out_rate // 10 * 2
        pending = b""
        try:
            async for chunk in self.engines.voice.speak(message.text, message.voice, self.out_rate):
                pending += chunk
                while len(pending) >= frame_bytes:
                    await self._speech_frame(message.stream, speech, pending[:frame_bytes])
                    pending = pending[frame_bytes:]
            if pending:
                await self._speech_frame(message.stream, speech, pending)
        finally:
            ours = self.speaking.get(message.stream) is speech
            if ours:
                del self.speaking[message.stream]
        if ours:
            await self._send(SpeakDone(message.stream, speech.frames))

    async def _speech_frame(self, stream: int, speech: _Speech, pcm: bytes) -> None:
        frame = encode_audio(AudioKind.SPEECH, stream, speech.frames, pcm)
        # websockets writes the frame before its first await, so a cancel can't land between the write and the count.
        speech.frames += 1
        await self.ws.send(frame)

    # ---- narrate

    async def _narrate(self, message: Narrate) -> None:
        if message.id in self.narrating:
            await self._send(_error(ErrorCode.BAD_REQUEST, f"narration {message.id} is already outstanding", id=message.id))
            return
        if len(self.narrating) >= MAX_OUTSTANDING_NARRATIONS:
            await self._send(_error(ErrorCode.OVERLOADED, f"{MAX_OUTSTANDING_NARRATIONS} narrations are outstanding", id=message.id))
            return
        self.narrating[message.id] = asyncio.create_task(self._guard(lambda: self._narration(message), id=message.id))

    async def _narration(self, message: Narrate) -> None:
        assert self.engines.narrator is not None
        try:
            lines = await _NARRATE[message.op](self.engines.narrator, message)
            frame = encode(NarrateResult(message.id, tuple(lines)))
        finally:
            self.narrating.pop(message.id, None)
        await self.ws.send(frame)

    # ---- lifecycle

    async def _guard(self, work: Callable[[], Awaitable[None]], *, stream: Optional[int] = None, id: Optional[int] = None) -> None:
        """Run one op. An engine failure, including output the codec refuses, is error{model}."""
        try:
            await work()
        except ConnectionClosed:
            pass
        except Exception as error:
            log.exception("op failed (stream=%s id=%s)", stream, id)
            try:
                await self._send(_error(ErrorCode.MODEL, f"{type(error).__name__}: {error}", stream=stream, id=id))
            except ConnectionClosed:
                pass

    async def _stop_everything(self) -> None:
        """Abort hearing through its queue, so each session is closed even if its task has not started yet."""
        hearing = [h.task for h in self.hearing.values()]
        for hear in self.hearing.values():
            hear.queue.put_nowait(CloseHow.ABORT)
        cancelled = [s.task for s in self.speaking.values()] + list(self.narrating.values())
        for task in cancelled:
            task.cancel()
        await asyncio.gather(*hearing, *cancelled, return_exceptions=True)


_ROUTES: Dict[type, Tuple[Optional[Op], Callable[[_Session, Message], Awaitable[None]]]] = {
    Hello: (None, _Session._second_hello),
    HearOpen: (Op.HEAR, _Session._hear_open),
    HearClose: (Op.HEAR, _Session._hear_close),
    SpeakStart: (Op.SPEAK, _Session._speak_start),
    SpeakCancel: (Op.SPEAK, _Session._speak_cancel),
    Narrate: (Op.NARRATE, _Session._narrate),
}


def _target(message: Message) -> Dict[str, int]:
    if isinstance(message, Narrate):
        return {"id": message.id}
    return {"stream": message.stream}


def _error(code: ErrorCode, message: str, *, stream: Optional[int] = None, id: Optional[int] = None) -> Error:
    clipped = message.encode("utf-8")[:MAX_ERROR_MESSAGE_BYTES].decode("utf-8", "ignore")
    return Error(code, clipped, stream, id)
