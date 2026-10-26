# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

import asyncio
import logging
import re
import time
from itertools import chain
from typing import Awaitable, Callable, Optional, Tuple

from reachy_muse.contract.expressions import Expression
from reachy_muse.contract.intent import FALLBACK_OPENERS, Intent, guess_intent, pick
from reachy_muse.contract.replies import ReplyStyle, spoken_reply
from reachy_muse.capabilities import WAKE_CUE, Backends, Endpoint, Heard, HeardAudio, Hold, Mode, Opener, Partial, SpokenLine
from reachy_muse.progress import PUBLIC_PROGRESS_PHRASES, ProgressPlan

log = logging.getLogger(__name__)
ACKNOWLEDGEMENTS = tuple(dict.fromkeys(chain.from_iterable(FALLBACK_OPENERS.values())))

class _LocalEndedTurn:
    """A turn Silero or WebRTC ended, recognized by the streaming or the batch transcriber."""

    def __init__(self, wav: bytes, recognition: Optional[asyncio.Future],
                 transcribe: Optional[Callable[[bytes], Awaitable[str]]], truncated: bool):
        self.wav = wav
        self.truncated = truncated
        self._recognition = recognition
        self._transcribe = transcribe
        self._cancelled = False

    def start(self) -> None:
        """Start whole-turn recognition at the endpoint, while earlier turns still play."""
        if self._recognition is None and self._transcribe is not None and not self._cancelled:
            self._recognition = asyncio.create_task(self._transcribe(self.wav))

    def cancel(self) -> None:
        self._cancelled = True
        if self._recognition is not None:
            self._recognition.cancel()

    async def endpoint(self) -> Optional[Endpoint]:
        """The turn as Muse receives it; None when its transcription was lost."""
        from reachy_muse.streaming_transcription import StreamingTranscriptionError, StreamingWorkerError

        self.start()
        if self._recognition is None:
            return Endpoint(self.wav, None)
        try:
            text = await self._recognition
        except StreamingWorkerError:
            raise
        except StreamingTranscriptionError:
            return None
        return Endpoint(self.wav, text)


class LocalHearingTurn:
    """One TurnRecorder, fed to a streaming recognizer while the user speaks."""

    def __init__(self, recorder, recognizer=None, transcribe=None):
        self._recorder = recorder
        self._recognizer = recognizer
        self._transcribe = transcribe
        self._turn = None
        self._failed = False
        self._wav = None

    @property
    def active(self) -> bool:
        return self._recorder.active

    @property
    def speech_active(self) -> bool:
        return getattr(self._recorder, "speech_active", self._recorder.active)

    @property
    def quiet_s(self) -> float:
        return getattr(self._recorder, "quiet_s", 0.0)

    @property
    def partial(self) -> Optional[Partial]:
        if self._turn is None or not self._turn.partial.text:
            return None
        return self._turn.partial

    def feed(self, samples) -> None:
        self._wav = self._recorder.feed(samples)

    def finish_initial_capture(self) -> None:
        self._recorder.finish_initial_capture()

    def take(self) -> HeardAudio:
        from reachy_muse.voice_audio import SpeechAudio, SpeechEnd

        wav, self._wav = self._wav, None
        events = self._recorder.take_audio_events() if self._recognizer is not None else ()
        recognition = None
        failures = 0
        transcript_lost = accepted_turn_lost = False
        audio = bytearray()
        for event in events:
            if isinstance(event, SpeechAudio):
                audio.extend(event.pcm)
            elif isinstance(event, SpeechEnd):
                failures += self._listen(audio)
                if self._failed:
                    transcript_lost = True
                    accepted_turn_lost = accepted_turn_lost or event.accepted
                elif self._turn is not None:
                    if event.accepted:
                        recognition = self._turn.finish()
                    else:
                        self._turn.abort()
                self._turn = None
                self._failed = False
        failures += self._listen(audio)
        ended = None
        if wav is not None and not accepted_turn_lost:
            ended = _LocalEndedTurn(wav, recognition, self._transcribe, self._recorder.last_truncated)
        return HeardAudio(ended, failures, transcript_lost)

    def _listen(self, audio: bytearray) -> int:
        from reachy_muse.streaming_transcription import StreamingTranscriptionError, StreamingWorkerError

        if not audio or self._failed:
            audio.clear()
            return 0
        try:
            if self._turn is None:
                self._turn = self._recognizer.open_turn()
            self._turn.feed(bytes(audio))
        except StreamingWorkerError:
            raise
        except StreamingTranscriptionError:
            self.abort()
            self._failed = True
            return 1
        finally:
            audio.clear()
        return 0

    def abort(self) -> None:
        if self._turn is not None:
            self._turn.abort()
            self._turn = None


class LocalHearing:
    def __init__(self, transcriber=None):
        self.transcriber = transcriber
        self._streaming = callable(getattr(transcriber, "open_turn", None))
        self._batch = callable(getattr(transcriber, "transcribe", None))
        self._lock = None

    @property
    def transcribes(self) -> bool:
        return self.transcriber is not None

    async def start(self) -> None:
        if self._streaming:
            await self.transcriber.start()
            log.info("Reachy streaming recognition ready: audio processed before endpoint")

    def open_turn(self, sample_rate: int, *, silence_s: float, vad=None, **options) -> LocalHearingTurn:
        from reachy_muse.voice_audio import TurnRecorder

        recorder = TurnRecorder(sample_rate, silence_s=silence_s, max_s=60.0,
                                **({"stream_audio": True} if self._streaming else {}),
                                **({"vad": vad} if vad is not None else {}), **options)
        return LocalHearingTurn(recorder, self.transcriber if self._streaming else None,
                                self.transcribe if self._batch else None)

    async def transcribe(self, wav: bytes) -> str:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            started = time.monotonic()
            text = await asyncio.wait_for(self.transcriber.transcribe(wav), 60)
            log.info("Speech recognition took %.1fs", time.monotonic() - started)
            return text


class _Replay:
    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    def cancel(self) -> None:
        self._chunks = iter(())

    async def aclose(self) -> None:
        self.cancel()


class LocalVoice:
    speaks_text = True

    def __init__(self, speech, fixed_phrases: Tuple[str, ...] = ()):
        self.speech = speech
        self.fixed_phrases = fixed_phrases
        self.phrases = {}

    async def warm(self, output_rate: int) -> None:
        for phrase in self.fixed_phrases:
            stream = self.speech.stream(phrase, output_rate)
            audio = []
            try:
                async for chunk in stream:
                    audio.append(chunk)
            finally:
                await stream.aclose()
            self.phrases[phrase] = tuple(audio)

    def is_presynthesized(self, text: str) -> bool:
        return bool(self.phrases.get(text))

    def stream(self, line: SpokenLine, output_rate: int):
        if self.is_presynthesized(line.text):
            return _Replay(self.phrases[line.text])
        return self.speech.stream(line.text, output_rate)

    def prepare(self, line: SpokenLine, output_rate: int):
        if self.is_presynthesized(line.text):
            return _Replay(self.phrases[line.text])
        if not callable(getattr(self.speech, "prepare", None)):
            return None
        return self.speech.prepare(line.text, output_rate)


class MuseVoice:
    speaks_text = False

    def __init__(self, session):
        self.session = session

    def stream(self, line: SpokenLine, output_rate: int):
        from reachy_muse.voice_audio import Mp3Decoder

        decoder = Mp3Decoder(output_rate)
        mp3 = self.session.stream_tts(line.muse_message_id)

        async def decoded():
            try:
                async for chunk in mp3:
                    for samples in decoder.feed(chunk):
                        yield samples
            finally:
                await mp3.aclose()
            for samples in decoder.finish():
                yield samples
        return decoded()

    async def warm(self, output_rate: int) -> None:
        pass

    def prepare(self, line: SpokenLine, output_rate: int):
        return None


class RuleNarrator:
    def __init__(self, *, presynthesized: Optional[Callable[[str], bool]] = None):
        self._presynthesized = presynthesized
        self._turns = dict.fromkeys(Intent, 0)

    def hold(self, intent: Intent, heard: Heard) -> Hold:
        turn = self._turns[intent]
        self._turns[intent] += 1
        phrase = pick(FALLBACK_OPENERS[intent], turn, heard.recently_said)
        if phrase is None or self._presynthesized is None or not self._presynthesized(phrase):
            return Hold(intent, None)
        return Hold(intent, SpokenLine(phrase, None, "ack"))

    async def acknowledge(self, heard: Heard) -> Opener:
        return self.hold(guess_intent(heard.text, heard.previous), heard)

    def progress(self, request: str, started: float, intent: Intent = Intent.QUESTION) -> ProgressPlan:
        return ProgressPlan(request, started, intent=intent)

    async def say_progress(self, request: str, status: str,
                           already_said: Tuple[str, ...]) -> Optional[SpokenLine]:
        return SpokenLine(status, None, "progress")

    async def lines(self, request: str, reply: str, style: ReplyStyle, *,
                    message_id: Optional[str] = None) -> Tuple[SpokenLine, ...]:
        if style is ReplyStyle.EXPRESSIVE_JSON:
            raise ValueError("sentence frames are parsed as they stream, by the reply tracker")
        text, expression = spoken_reply(reply)
        if style is ReplyStyle.PLAIN_SHORT:
            return tuple(SpokenLine(sentence, None, "answer") for sentence in split_sentences(text))
        return (SpokenLine(text, Expression.parse(expression), "answer", message_id),)


def split_sentences(text: str) -> Tuple[str, ...]:
    return tuple(part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part)


def backends_for(mode: Mode, session, *, speech=None, progress_speech=None, transcriber=None,
                 stream_replies: bool = False, wake: bool = False, companion=None) -> Backends:
    if mode is Mode.COMPANION:
        from reachy_muse.failover import companion_backends
        local = backends_for(Mode.ON_ROBOT, session, speech=speech, progress_speech=progress_speech,
                             transcriber=transcriber, stream_replies=stream_replies, wake=wake)
        return companion_backends(local, companion)
    if mode is Mode.MUSE_VOICE:
        return Backends(hearing=LocalHearing(), voice=MuseVoice(session), narrator=RuleNarrator(),
                        local_style=ReplyStyle.MUSE_VOICE)
    voice = LocalVoice(speech, (*ACKNOWLEDGEMENTS, *((WAKE_CUE,) if wake else ())))
    return Backends(
        hearing=LocalHearing(transcriber),
        voice=voice,
        narrator=RuleNarrator(presynthesized=voice.is_presynthesized),
        local_style=ReplyStyle.EXPRESSIVE_JSON if stream_replies else ReplyStyle.MARKER,
        progress_voice=(None if progress_speech is None
                        else LocalVoice(progress_speech, PUBLIC_PROGRESS_PHRASES)),
    )
