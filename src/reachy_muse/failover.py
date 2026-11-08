# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Companion capabilities that fall back to Reachy's own, so a turn never waits on a dead link."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Tuple, Union

from reachy_muse.contract.replies import ReplyStyle
from reachy_muse.contract.intent import Intent, guess_intent
from reachy_muse.companion_client import CompanionOpening
from reachy_muse.local_backends import RuleNarrator
from reachy_muse.capabilities import (EMOJI, MARKUP, Backends, CapabilityUnavailable, Endpoint, Heard, HeardAudio, Hold,
                                      LocalReply, Opener, SpokenLine, grounded)
from reachy_muse.device_command import Commands
from reachy_muse.protocol import MAX_LINE_BYTES, MAX_REPLY_BYTES
from reachy_muse.progress import ProgressPlan

log = logging.getLogger(__name__)

REPLAY_LIMIT_S = 60.0              # audio kept for a local replay of an unfinished turn
IDLE_KEEP_S = 1.0                  # between turns, keep only this much
ACK_DEADLINE_S = 0.9
PROGRESS_DEADLINE_S = 1.5
LINES_DEADLINE_S = 4.0
SPOKEN_AS_WRITTEN_LINES = 2       # a plain reply this short is spoken as Muse wrote it, not condensed
STRIKES = 3                        # consecutive failures of one companion op that rest it
COOLDOWN_S = 60.0                  # how long a rested op stays on the robot before the companion is tried again


class _Rest:
    """One companion op's failures in a row; ``STRIKES`` of them rest the op for ``COOLDOWN_S``."""

    def __init__(self, op: str):
        self._op = op
        self._strikes = 0
        self._until = 0.0

    @property
    def resting(self) -> bool:
        return time.monotonic() < self._until

    def fail(self, reason: str) -> None:
        self._strikes += 1
        log.warning("Reachy used its own %s: %s (%d in a row)", self._op, reason, self._strikes)
        if self._strikes >= STRIKES:
            self._strikes = 0
            self._until = time.monotonic() + COOLDOWN_S
            log.warning("Reachy rests its companion's %s for %.0f s", self._op, COOLDOWN_S)

    def succeed(self) -> None:
        self._strikes = 0


@dataclass
class _VoiceReply:
    companion: bool


class CompanionRoute:
    """Whether the companion voices and narrates the current turn, decided once at its start.

    A link that comes back mid-turn waits for the next turn, so one answer never switches
    voices, and an op that keeps failing rests on its own while the link stays up.
    """

    def __init__(self, link):
        self.link = link
        self.this_turn = False
        self.hearing = _Rest("hearing")
        self.speech_lock = asyncio.Lock()
        self.narration = _Rest("narration")
        self._voice_reply: Optional[_VoiceReply] = None
        self._voice_replies: dict[str, _VoiceReply] = {}

    def begin_turn(self) -> bool:
        self.this_turn = self.link.up
        self.begin_reply()
        return self.this_turn

    def begin_reply(self) -> None:
        self._voice_reply = _VoiceReply(self.link.up)

    def voice_reply(self, line: SpokenLine) -> _VoiceReply:
        if line.muse_message_id is not None:
            reply = self._voice_replies.get(line.muse_message_id)
            if reply is None:
                reply = self._voice_replies[line.muse_message_id] = _VoiceReply(self.link.up)
                if len(self._voice_replies) > 128:
                    del self._voice_replies[next(iter(self._voice_replies))]
            return reply
        if self._voice_reply is None:
            self._voice_reply = _VoiceReply(self.this_turn)
        return self._voice_reply

    def take_outage(self) -> bool:
        return self.link.take_outage()


class _CompanionEnded:
    """A companion turn whose words fall back to Reachy's own transcription of its audio."""

    def __init__(self, ended, rest: _Rest):
        self._ended = ended
        self._rest = rest
        self.truncated = ended.truncated
        self.end_sample = ended.end_sample

    def start(self) -> None:
        self._ended.start()

    def cancel(self) -> None:
        self._ended.cancel()

    async def endpoint(self) -> Optional[Endpoint]:
        try:
            endpoint = await self._ended.endpoint()
        except CapabilityUnavailable as error:
            log.warning("Reachy transcribes the turn itself: %s", type(error).__name__)
            if not error.link_down:
                self._rest.fail(type(error).__name__)
            return Endpoint(self._ended.wav, None)
        self._rest.succeed()
        return endpoint


class FailoverHearingTurn:
    """One listening window on the companion, finished locally if the companion fails.

    While the companion hears, fed audio is kept from the end of its last ended turn. If it
    fails mid-turn, a local turn opens and the kept audio replays into it at the next feed,
    which runs off the event loop. The local turn hands back to the companion between user
    turns once the link is up.
    """

    def __init__(self, hearing: FailoverHearing, sample_rate: int, options: dict):
        self._hearing = hearing
        self._rate = sample_rate
        self._options = options
        self._kept = deque()
        self._kept_samples = 0
        self._fed = 0
        self._turn_start_sample = 0
        self._replay = None
        self._on_companion = hearing.companion_ready
        self._turn = (hearing.companion if self._on_companion else hearing.local).open_turn(sample_rate, **options)

    @property
    def active(self) -> bool:
        return self._replay is not None or self._turn.active

    @property
    def speech_active(self) -> bool:
        return self._turn.speech_active

    @property
    def quiet_s(self) -> float:
        return self._turn.quiet_s

    @property
    def supports_partials(self) -> bool:
        return self._turn.supports_partials

    @property
    def partial(self):
        return self._turn.partial

    def feed(self, samples) -> None:
        self._fed += len(samples)
        self._keep(samples)
        if self._replay is not None:
            import numpy as np
            samples, self._replay = np.concatenate([*self._replay, samples]), None
        self._turn.feed(samples)

    async def drain(self) -> None:
        await self._turn.drain()

    def _keep(self, samples) -> None:
        self._kept.append(samples.copy())
        self._kept_samples += len(samples)
        self._trim(REPLAY_LIMIT_S)

    def _trim(self, seconds: float) -> None:
        while len(self._kept) > 1 and self._kept_samples - len(self._kept[0]) >= seconds * self._rate:
            self._kept_samples -= len(self._kept.popleft())

    def _forget_until(self, sample: int) -> None:
        """Drop kept audio before ``sample``, counted in samples fed to this window."""
        excess = sample - (self._fed - self._kept_samples)
        while excess > 0 and self._kept:
            oldest = self._kept[0]
            if len(oldest) <= excess:
                self._kept.popleft()
            else:
                self._kept[0] = oldest[excess:]
            dropped = min(len(oldest), excess)
            self._kept_samples -= dropped
            excess -= dropped

    def take(self) -> HeardAudio:
        if not self._on_companion:
            heard = self._turn.take()
            if heard.ended is not None:
                self._forget_until(self._fed)
            elif not self._turn.active:
                self._trim(IDLE_KEEP_S)
            if (self._hearing.companion_ready and heard.ended is None and self._replay is None
                    and not self._turn.active and self._turn.quiet_s >= self._options["silence_s"]):
                self._switch(on_companion=True)
            return heard
        try:
            heard = self._turn.take()
        except CapabilityUnavailable as error:
            log.warning("Reachy's companion stopped hearing (%s); finishing the turn on the robot", error)
            if not error.link_down:
                self._hearing.rest.fail(str(error))
            self._replay = tuple(self._kept)
            self._switch(on_companion=False)
            return HeardAudio()
        if heard.ended is not None:
            self._forget_until(self._turn_start_sample + heard.ended.end_sample)
            return HeardAudio(_CompanionEnded(heard.ended, self._hearing.rest), heard.failures, heard.transcript_lost)
        elif not self._turn.active:
            self._trim(IDLE_KEEP_S)
        return heard

    def _switch(self, *, on_companion: bool) -> None:
        options = dict(self._options, recorder=self._turn.recorder) if on_companion else self._options
        self._turn.abort()
        if not on_companion:
            self._kept.clear()
            self._kept_samples = 0
        self._on_companion = on_companion
        self._turn_start_sample = self._fed
        hearing = self._hearing.companion if on_companion else self._hearing.local
        self._turn = hearing.open_turn(self._rate, **options)

    def finish_initial_capture(self) -> None:
        self._turn.finish_initial_capture()

    def abort(self) -> None:
        self._turn.abort()


class FailoverHearing:
    def __init__(self, companion, local, rest: _Rest):
        self.companion = companion
        self.local = local
        self.rest = rest

    @property
    def companion_ready(self) -> bool:
        return self.companion.available and not self.rest.resting

    @property
    def transcribes(self) -> bool:
        return self.local.transcribes

    async def start(self) -> None:
        await self.local.start()
        await self.companion.start()

    def open_turn(self, sample_rate: int, **options) -> FailoverHearingTurn:
        return FailoverHearingTurn(self, sample_rate, options)

    async def transcribe(self, wav: bytes) -> str:
        return await self.local.transcribe(wav)


class FailoverVoice:
    """Keep one voice per reply; a dead link restarts its sentence locally."""

    speaks_text = True

    def __init__(self, companion, local, route: CompanionRoute, *, standalone: bool = False):
        self.companion = companion
        self.local = local
        self.route = route
        self.standalone = standalone

    async def warm(self, output_rate: int) -> None:
        await self.local.warm(output_rate)

    def is_presynthesized(self, text: str) -> bool:
        return self.local.is_presynthesized(text)

    def prepare(self, line: SpokenLine, output_rate: int):
        from reachy_muse.local_speech import PreparedSpeech
        reply = self._reply_for(line)
        return PreparedSpeech(self._stream(line, output_rate, reply), sample_rate=output_rate,
                              progress_at=lambda: (getattr(self.route.link, "last_progress_at", 0.0)
                                                   if reply.companion else 0.0))

    def _reply_for(self, line: SpokenLine) -> _VoiceReply:
        if line.on_robot and self.local.is_presynthesized(line.text):
            return _VoiceReply(False)
        return _VoiceReply(self.route.link.up) if self.standalone else self.route.voice_reply(line)

    def stream(self, line: SpokenLine, output_rate: int):
        return self._stream(line, output_rate, self._reply_for(line))

    async def _stream(self, line: SpokenLine, output_rate: int, reply: _VoiceReply):
        from reachy_muse.local_speech import SpeechRestart

        if reply.companion:
            async with self.route.speech_lock:
                # A preceding prepared sentence may have lost the link while we waited.
                if reply.companion and self.companion.available:
                    speech = self.companion.stream(line, output_rate)
                    spoke = False
                    try:
                        async for chunk in speech:
                            spoke = spoke or bool(len(chunk))
                            yield chunk
                        return
                    except CapabilityUnavailable as error:
                        reply.companion = False
                        log.warning("Reachy companion voice unavailable (%s); restarting the sentence locally",
                                    type(error).__name__)
                        if spoke:
                            yield SpeechRestart()
                    finally:
                        await speech.aclose()
                else:
                    reply.companion = False
        speech = self.local.stream(line, output_rate)
        try:
            async for chunk in speech:
                yield chunk
        finally:
            await speech.aclose()


class _Late:
    """The companion's narration missed its deadline: it is busy, not broken."""


LATE = _Late()


def settle(heard: Heard, guess: Intent, got: Union[CompanionOpening, _Late, None], local: RuleNarrator) -> Opener:
    if got is LATE:
        # A companion this slow to narrate is slow to speak too; the recording plays at once.
        opening = local.hold(guess, heard)
        if isinstance(opening, Hold) and opening.line is not None:
            return Hold(opening.intent, replace(opening.line, on_robot=True))
        return opening
    if got is None:
        return local.hold(guess, heard)
    intent = got.intent or guess
    if intent is Intent.CHAT:
        if got.intent is Intent.CHAT and heard.may_chat and got.lines:
            return LocalReply(tuple(SpokenLine(line.text, line.expression, "answer") for line in got.lines))
        return Hold(Intent.CHAT, None)
    return Hold(intent, got.lines[0] if got.lines else None)


class FailoverNarrator:
    """The companion narrates within deadlines; Reachy's rules answer whenever it can't."""

    def __init__(self, companion, local, route: CompanionRoute):
        self.companion = companion
        self.local = local
        self.route = route

    @property
    def _on_companion(self) -> bool:
        return self.route.this_turn and self.companion.available and not self.route.narration.resting

    async def _ask(self, call, deadline_s: float, sources: Tuple[str, ...],
                   lines: Callable[[Any], Tuple[SpokenLine, ...]] = lambda result: result) -> Any:
        try:
            result = await asyncio.wait_for(call, deadline_s)
        except CapabilityUnavailable as error:
            if not error.link_down:
                self.route.narration.fail(str(error))
            return None
        except asyncio.TimeoutError:
            self.route.narration.fail("narration timed out")
            return LATE
        if not all(grounded(line.text, *sources) for line in lines(result)):
            self.route.narration.fail("narration added facts that are not in the request or reply")
            return None
        self.route.narration.succeed()
        return result

    async def acknowledge(self, heard: Heard) -> Opener:
        guess = guess_intent(heard.text, heard.previous)
        got = None
        if self._on_companion:
            got = await self._ask(self.companion.acknowledge(heard), ACK_DEADLINE_S, (heard.text, heard.previous),
                                  lines=lambda opening: opening.lines)
        if got is LATE:
            log.info("Reachy acknowledges in its own words; its companion's narration was late")
        opening = settle(heard, guess, got, self.local)
        if isinstance(opening, LocalReply) and not isinstance(got, CompanionOpening):
            self.route.this_turn = False
        return opening

    def progress(self, request: str, started: float, intent: Intent = Intent.QUESTION) -> ProgressPlan:
        return self.local.progress(request, started, intent)

    async def say_progress(self, request: str, status: str,
                           already_said: Tuple[str, ...]) -> Optional[SpokenLine]:
        if self._on_companion:
            line = await self._ask(self.companion.say_progress(request, status, already_said),
                                   PROGRESS_DEADLINE_S, (status, request),
                                   lines=lambda line: () if line is None else (line,))
            if isinstance(line, SpokenLine):
                return line
        return await self.local.say_progress(request, status, already_said)

    async def lines(self, request: str, reply: str, style: ReplyStyle, *,
                    message_id: Optional[str] = None) -> Tuple[SpokenLine, ...]:
        local = await self.local.lines(request, reply, style, message_id=message_id)
        if style is ReplyStyle.PLAIN_SHORT and local and self._on_companion:
            spoken = " ".join(line.text for line in local)
            if len(spoken.encode("utf-8")) > MAX_REPLY_BYTES:
                log.info("Reachy speaks a %d-byte reply itself; the companion narrates at most %d",
                         len(spoken.encode("utf-8")), MAX_REPLY_BYTES)
                self.route.this_turn = False
                return local
            if _speakable_as_written(local):
                log.info("Reachy speaks a %d-sentence reply as written, without condensing it", len(local))
                return local
            result = await self._ask(self.companion.lines(request, spoken, style), LINES_DEADLINE_S,
                                     (request, spoken))
            if result and result is not LATE:
                return result
        return local


def _speakable_as_written(lines: Tuple[SpokenLine, ...]) -> bool:
    """Already what the companion's condense would produce: few short sentences with no markup."""
    return len(lines) <= SPOKEN_AS_WRITTEN_LINES and all(
        len(line.text.encode("utf-8")) <= MAX_LINE_BYTES and not MARKUP.search(line.text)
        and not EMOJI.search(line.text) for line in lines)


def companion_backends(local: Backends, link) -> Backends:
    """Mode 3: the companion's parts, each backed by the on-robot part of mode 2."""
    from reachy_muse.companion_client import (
        CompanionCommandModel, CompanionHearing, CompanionMotion, CompanionNarrator, CompanionVoice, VoiceRole,
    )

    route = CompanionRoute(link)
    progress = (None if local.progress_voice is None
                else FailoverVoice(CompanionVoice(link, VoiceRole.PROGRESS), local.progress_voice, route))
    return Backends(
        hearing=FailoverHearing(CompanionHearing(link), local.hearing, route.hearing),
        voice=FailoverVoice(CompanionVoice(link), local.voice, route),
        command_voice=FailoverVoice(CompanionVoice(link), local.voice, route, standalone=True),
        narrator=FailoverNarrator(CompanionNarrator(link), local.narrator, route),
        local_style=local.local_style,
        progress_voice=progress,
        companion=route,
        commands=Commands(CompanionCommandModel(link)),
        motion=CompanionMotion(link),
    )
