"""The companion's WebSocket server, speaking the SDK's wire protocol v1.

One `_Session` per robot connection. Each hear stream, speak stream, narration,
classification and animation runs as its own task, so a slow engine never stalls the receive loop.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import ssl
from collections import OrderedDict
from contextlib import aclosing
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Awaitable, Callable, Dict, Optional, Sequence, Tuple, Union

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from reachy_muse.contract.expressions import Expression
from reachy_muse.protocol import (
    MAX_ERROR_MESSAGE_BYTES,
    MAX_OUTSTANDING_ANIMATIONS,
    MAX_OUTSTANDING_NARRATIONS,
    MAX_TEXT_FRAME_BYTES,
    VERSION,
    Animate,
    AnimateResult,
    AudioFrame,
    AudioKind,
    Classify,
    ClassifyResult,
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
    VoiceRole,
    Welcome,
    decode,
    decode_audio,
    encode,
    encode_audio,
)

from reachy_muse.companion_server.engines import (
    Narrated,
    Animator, Classifier, Endpoint, Hearing, Narrator, Partial, SpeechFirst, Voice,
)

log = logging.getLogger("reachy_muse.companion_server")

SUPPORTED_VERSIONS = (VERSION,)
PATH = "/v1"
HELLO_TIMEOUT = 5.0
CLOSE_POLICY_VIOLATION = 1008
CLOSE_INTERNAL_ERROR = 1011
REMEMBERED_EXPRESSIONS = 64


@dataclass(frozen=True)
class Engines:
    """An op is offered exactly when its engine is present."""

    hearing: Optional[Hearing] = None
    voice: Optional[Voice] = None
    narrator: Optional[Narrator] = None
    motion: Optional[Animator] = None
    classifier: Optional[Classifier] = None
    speech_first: SpeechFirst = field(default_factory=SpeechFirst)

    def offered(self) -> Dict[Op, str]:
        engines = {Op.HEAR: self.hearing, Op.SPEAK: self.voice, Op.NARRATE: self.narrator, Op.CLASSIFY: self.classifier,
                   Op.ANIMATE: self.motion}
        return {op: engine.name for op, engine in engines.items() if engine is not None}


class CompanionServer:
    """`async with CompanionServer(engines, token) as server:` serves until the block exits."""

    def __init__(self, engines: Engines, token: str, *, ssl_context: Optional[ssl.SSLContext] = None, host: str = "127.0.0.1", port: int = 0) -> None:
        offered = engines.offered()
        encode(Welcome(VERSION, tuple(offered), tuple((op.value, name) for op, name in offered.items())))  # refuse names the codec would refuse
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
    """Microphone frames and the finishing close, in arrival order. Abort and gap cancel the task instead."""

    queue: "asyncio.Queue[Union[bytes, CloseHow]]"
    task: Optional[asyncio.Task] = None


@dataclass
class _Speech:
    task: Optional[asyncio.Task] = None
    frames: int = 0


WARM_TTL_S = 15.0
WARM_ENTRIES = 2


@dataclass
class _Warm:
    chunks: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: Optional[asyncio.Task] = None
    expiry: Optional[asyncio.TimerHandle] = None

    def cancel(self) -> None:
        if self.expiry is not None:
            self.expiry.cancel()
        if self.task is not None:
            self.task.cancel()


_NARRATE: Dict[NarrateOp, Callable[[Narrator, Narrate], Awaitable[Union[Narrated, Sequence[NarratedLine]]]]] = {
    NarrateOp.ACKNOWLEDGE: lambda narrator, m: narrator.acknowledge(
        m.request, previous=m.previous or "", recently_said=m.already_said, may_chat=m.may_chat),
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
        self.animating: Dict[int, asyncio.Task] = {}
        self.warm: Dict[str, _Warm] = {}
        self.warming: set[asyncio.Task] = set()
        self.expressions: OrderedDict[str, Optional[Expression]] = OrderedDict()

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
                except ConnectionClosed:
                    raise
                except Exception as error:
                    log.exception("handling a frame failed")
                    await self._fatal(ErrorCode.INTERNAL, error)
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
        hear = self.hearing[message.stream] = _Hear(asyncio.Queue())
        hear.task = asyncio.create_task(self._guard(self._speech_first(lambda: self._hear(message, hear)), stream=message.stream))

    async def _hear_close(self, message: HearClose) -> None:
        hear = self.hearing.get(message.stream)
        if hear is None:
            return
        if message.how is CloseHow.FINISH:
            hear.queue.put_nowait(CloseHow.FINISH)
        else:
            self._end_hearing(message.stream, hear)

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
            await self._send(_error(ErrorCode.GAP, str(error), stream=frame.stream))
            return
        hear.queue.put_nowait(frame.pcm)

    async def _hear(self, message: HearOpen, hear: _Hear) -> None:
        assert self.engines.hearing is not None
        stream, rev = message.stream, 0
        session = self.engines.hearing.open(message.pre_roll_frames)
        try:
            while True:
                item = await hear.queue.get()
                if item is CloseHow.FINISH:
                    text = await session.finish()
                    self._end_hearing(stream, hear)
                    await self._send(HearEndpoint(stream, text, True))
                    return
                for event in await session.feed(item):
                    if isinstance(event, Partial):
                        await self._send(HearPartial(stream, rev, event.text))
                        rev += 1
                    elif isinstance(event, Endpoint):
                        self._end_hearing(stream, hear)
                        await self._send(HearEndpoint(stream, event.text, False))
                        return
        finally:
            self._end_hearing(stream, hear)
            await session.close()

    def _end_hearing(self, stream: int, hear: _Hear) -> None:
        """Free the stream id at once; a cancelled task sends nothing more under it."""
        if self.hearing.get(stream) is hear:
            del self.hearing[stream]
            self.sequence.forget(stream)
            assert hear.task is not None
            if hear.task is not asyncio.current_task():
                hear.task.cancel()

    # ---- speak

    async def _speak_start(self, message: SpeakStart) -> None:
        if message.stream in self.speaking:
            await self._send(_error(ErrorCode.BAD_REQUEST, f"speak stream {message.stream} is already speaking", stream=message.stream))
            return
        speech = self.speaking[message.stream] = _Speech()
        speech.task = asyncio.create_task(self._guard(self._speech_first(lambda: self._speak(message, speech)), stream=message.stream))

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
            warm = self.warm.pop(message.text, None) if message.voice is VoiceRole.MAIN else None
            if warm is not None and warm.expiry is not None:
                warm.expiry.cancel()
            source = (self._warm_chunks(warm) if warm is not None
                      else self.engines.voice.speak(message.text, message.voice, self.out_rate,
                                                    self.expressions.get(message.text)))
            async with aclosing(source) as chunks:
                async for chunk in chunks:
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
        await self._start(message.id, self._speech_first(lambda: self._narration(message)), self.narrating, MAX_OUTSTANDING_NARRATIONS, "narration")

    async def _classify(self, message: Classify) -> None:
        await self._start(message.id, self._speech_first(lambda: self._classification(message)), self.narrating, MAX_OUTSTANDING_NARRATIONS, "narration")

    async def _animate(self, message: Animate) -> None:
        await self._start(message.id, lambda: self._animation(message), self.animating, MAX_OUTSTANDING_ANIMATIONS, "animation")

    async def _start(self, id: int, work: Callable[[], Awaitable[None]], tasks: Dict[int, asyncio.Task], limit: int,
                     kind: str) -> None:
        if id in self.narrating or id in self.animating:
            await self._send(_error(ErrorCode.BAD_REQUEST, f"{kind} {id} is already outstanding", id=id))
            return
        if len(tasks) >= limit:
            await self._send(_error(ErrorCode.OVERLOADED, f"{limit} {kind}s are outstanding", id=id))
            return
        tasks[id] = asyncio.create_task(self._guard(work, id=id))

    async def _narration(self, message: Narrate) -> None:
        assert self.engines.narrator is not None
        try:
            result = await _NARRATE[message.op](self.engines.narrator, message)
            narrated = result if isinstance(result, Narrated) else Narrated(None, tuple(result))
            frame = encode(NarrateResult(message.id, narrated.lines, narrated.intent))
            self._remember(narrated.lines)
            if message.op is NarrateOp.ACKNOWLEDGE and self.engines.voice is not None:
                for line in narrated.lines[:WARM_ENTRIES]:
                    self._warm_voice(line.text)
        finally:
            self.narrating.pop(message.id, None)
        await self.ws.send(frame)

    def _warm_voice(self, text: str) -> None:
        old = self.warm.pop(text, None)
        if old is not None:
            old.cancel()
        while len(self.warm) >= WARM_ENTRIES:
            self.warm.pop(next(iter(self.warm))).cancel()
        warm = self.warm[text] = _Warm()

        async def produce() -> None:
            assert self.engines.voice is not None
            try:
                async with aclosing(self.engines.voice.speak(text, VoiceRole.MAIN, self.out_rate,
                                                             self.expressions.get(text))) as chunks:
                    async for chunk in chunks:
                        warm.chunks.put_nowait(chunk)
            except Exception as error:
                warm.chunks.put_nowait(error)
            finally:
                warm.chunks.put_nowait(None)

        def expire() -> None:
            if self.warm.get(text) is warm:
                self.warm.pop(text).cancel()

        warm.task = asyncio.create_task(self._speech_first(produce)())
        self.warming.add(warm.task)
        warm.task.add_done_callback(self.warming.discard)
        warm.expiry = asyncio.get_running_loop().call_later(WARM_TTL_S, expire)

    async def _warm_chunks(self, warm: _Warm):
        try:
            while True:
                chunk = await warm.chunks.get()
                if chunk is None:
                    return
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
        finally:
            warm.cancel()
            if warm.task is not None:
                await asyncio.gather(warm.task, return_exceptions=True)

    def _remember(self, lines: Sequence[NarratedLine]) -> None:
        for line in lines:
            self.expressions[line.text] = line.expression
            self.expressions.move_to_end(line.text)
        while len(self.expressions) > REMEMBERED_EXPRESSIONS:
            self.expressions.popitem(last=False)

    async def _classification(self, message: Classify) -> None:
        assert self.engines.classifier is not None
        try:
            command = await self.engines.classifier.classify(message.text, message.allowed)
            frame = encode(ClassifyResult(message.id, command if command in message.allowed else None))
        finally:
            self.narrating.pop(message.id, None)
        await self.ws.send(frame)

    async def _animation(self, message: Animate) -> None:
        """Motion is decoration: an engine that fails answers with no frames and keeps the link up."""
        assert self.engines.motion is not None
        try:
            try:
                animation = await self.engines.motion.animate(message.intent, message.duration_ms / 1000)
                frame = encode(AnimateResult(message.id, animation.fps, animation.frames))
            except Exception:
                log.exception("motion for request %s failed; answering with no frames", message.id)
                frame = encode(AnimateResult(message.id, 25))
        finally:
            self.animating.pop(message.id, None)
        await self.ws.send(frame)

    # ---- lifecycle

    def _speech_first(self, work: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
        async def first() -> None:
            with self.engines.speech_first.busy():
                await work()
        return first

    async def _guard(self, work: Callable[[], Awaitable[None]], *, stream: Optional[int] = None, id: Optional[int] = None) -> None:
        """Run one op. An engine failure, including output the codec refuses, is error{model}."""
        try:
            await work()
        except ConnectionClosed:
            pass
        except Exception as error:
            log.exception("op failed (stream=%s id=%s)", stream, id)
            await self._fatal(ErrorCode.MODEL, error, stream=stream, id=id)

    async def _fatal(self, code: ErrorCode, error: Exception, *, stream: Optional[int] = None, id: Optional[int] = None) -> None:
        """Spec rule 5: error{model|internal} takes the link down on the robot, so close it here too."""
        try:
            await self._send(_error(code, f"{type(error).__name__}: {error}", stream=stream, id=id))
            await self.ws.close(CLOSE_INTERNAL_ERROR, code.value)
        except ConnectionClosed:
            pass

    async def _stop_everything(self) -> None:
        tasks = ([h.task for h in self.hearing.values()] + [s.task for s in self.speaking.values()]
                 + list(self.narrating.values()) + list(self.animating.values()) + list(self.warming))
        for warm in self.warm.values():
            warm.cancel()
        self.warm.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


_ROUTES: Dict[type, Tuple[Optional[Op], Callable[[_Session, Message], Awaitable[None]]]] = {
    Hello: (None, _Session._second_hello),
    HearOpen: (Op.HEAR, _Session._hear_open),
    HearClose: (Op.HEAR, _Session._hear_close),
    SpeakStart: (Op.SPEAK, _Session._speak_start),
    SpeakCancel: (Op.SPEAK, _Session._speak_cancel),
    Narrate: (Op.NARRATE, _Session._narrate),
    Classify: (Op.CLASSIFY, _Session._classify),
    Animate: (Op.ANIMATE, _Session._animate),
}


def _target(message: Message) -> Dict[str, int]:
    if isinstance(message, (Narrate, Classify, Animate)):
        return {"id": message.id}
    return {"stream": message.stream}


def _error(code: ErrorCode, message: str, *, stream: Optional[int] = None, id: Optional[int] = None) -> Error:
    clipped = message.encode("utf-8")[:MAX_ERROR_MESSAGE_BYTES].decode("utf-8", "ignore")
    return Error(code, clipped, stream, id)
