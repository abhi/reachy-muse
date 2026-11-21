# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from itertools import groupby
from types import MappingProxyType
from typing import FrozenSet, Mapping, Optional, Protocol

from reachy_muse.follow_up import FollowUp, classify_follow_up

log = logging.getLogger(__name__)

CLASSIFY_DEADLINE_S = 0.3
MODEL_MAX_WORDS = 10
VOLUME_STEP = 10
VOLUME_FAILED = "I couldn't change my volume."


class DeviceCommand(Enum):
    SLEEP = "sleep"
    WAKE_UP = "wake_up"
    STOP_LISTENING = "stop_listening"
    STOP_TALKING = "stop_talking"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    FACE_TRACKING_ON = "face_tracking_on"
    FACE_TRACKING_OFF = "face_tracking_off"


class Need(Enum):
    WAKE_WORD = "wake_word"
    SPEECH_QUEUED = "speech_queued"
    FACE = "face"
    VOLUME = "volume"


class WakeWindow(Enum):
    """What governs Reachy's wake window; a device command can move it to AWAKE, DISMISSED or ASLEEP."""

    AWAKE = "awake"          # the wake phrase opens the window and its timeout closes it
    DISMISSED = "dismissed"  # the window closes at the next quiet moment
    ASLEEP = "asleep"        # as DISMISSED, in the sleep pose; the wake phrase wakes Reachy first
    WOKEN = "woken"          # the wake phrase just woke Reachy; it greets the user unless a request follows


@dataclass(frozen=True)
class Spec:
    needs: Need
    phrases: str
    evidence: FrozenSet[str]
    reply: Optional[str]
    silence_speech: bool = False
    wake_window: Optional[WakeWindow] = None
    volume_step: int = 0
    face_tracking: Optional[bool] = None


_BIT = r"(?:a (?:little )?bit |a little )?"

COMMANDS: Mapping[DeviceCommand, Spec] = MappingProxyType({
    DeviceCommand.SLEEP: Spec(
        Need.WAKE_WORD,
        r"go (?:back )?to sleep|go to bed|sleep|take a nap|nap time|have a rest|get some rest|good ?night|night night",
        frozenset({"sleep", "asleep", "nap", "bed", "bedtime", "rest", "snooze", "doze", "night", "hay"}),
        "Good night.", silence_speech=True, wake_window=WakeWindow.ASLEEP),
    DeviceCommand.WAKE_UP: Spec(
        Need.WAKE_WORD,
        r"wake up|rise and shine",
        frozenset({"wake", "awake", "rise"}),
        "I'm awake.", wake_window=WakeWindow.AWAKE),
    DeviceCommand.STOP_LISTENING: Spec(
        Need.WAKE_WORD,
        r"stop listening(?: to me)?|that's all|that is all",
        frozenset({"listen", "listening", "hearing", "mic", "microphone", "ears", "done", "finished"}),
        "Okay, I'll stop listening.", silence_speech=True, wake_window=WakeWindow.DISMISSED),
    DeviceCommand.STOP_TALKING: Spec(Need.SPEECH_QUEUED, "", frozenset(), None, silence_speech=True),
    DeviceCommand.VOLUME_UP: Spec(
        Need.VOLUME,
        rf"{_BIT}louder|(?:speak|talk) (?:up|{_BIT}louder)|be {_BIT}louder|volume up"
        r"|turn (?:up (?:the|your) volume|(?:the|your) volume up|yourself up)"
        r"|(?:increase|raise) (?:the|your) volume|i (?:can't|cannot) hear you",
        frozenset({"volume", "louder", "loud", "loudly", "hear", "voice"}),
        None, volume_step=VOLUME_STEP),
    DeviceCommand.VOLUME_DOWN: Spec(
        Need.VOLUME,
        rf"{_BIT}(?:quieter|softer)|(?:speak|talk) (?:{_BIT}(?:quieter|softer)|more (?:quietly|softly))"
        rf"|be {_BIT}quieter|volume down|turn (?:down (?:the|your) volume|(?:the|your) volume down|yourself down)"
        rf"|(?:lower|decrease|reduce) (?:the|your) volume|lower your voice|(?:you're |you are )?{_BIT}too loud",
        frozenset({"volume", "quieter", "quiet", "quietly", "softer", "soft", "softly", "loud", "lower", "voice", "pipe"}),
        None, volume_step=-VOLUME_STEP),
    DeviceCommand.FACE_TRACKING_ON: Spec(
        Need.FACE,
        r"(?:keep looking|look) at me|look over here|follow my face|track my face|watch me"
        r"|(?:turn on|start) face (?:tracking|following)|face tracking on",
        frozenset({"look", "looking", "face", "follow", "track", "tracking", "watch", "eyes"}),
        "Okay, I'll look at you.", face_tracking=True),
    DeviceCommand.FACE_TRACKING_OFF: Spec(
        Need.FACE,
        r"stop (?:looking at|following|tracking|watching|staring at) (?:me|my face)|don't (?:look at|follow|watch|stare at) me"
        r"|(?:turn off|stop) face (?:tracking|following)|face tracking off|look away",
        frozenset({"look", "looking", "face", "follow", "following", "track", "tracking", "watch", "watching",
                   "stare", "staring"}),
        "Okay, I'll stop following you.", face_tracking=False),
})

_LEAD = r"(?:oh|okay|ok|alright|hey|um|uh|so|now|well|muse|reachy|robot)"
_POLITE = r"(?:please|(?:can|could|would|will) you(?: please)?|i (?:want|need) you to|you (?:can|should|may)|(?:it's )?time to|go ahead and)"
_TAIL = r"(?:please|now|right now|then|thanks|thank you|for now|for a (?:bit|while)|for me|a (?:little )?bit|a little|muse|reachy)"


def _request_grammar(phrases: str) -> re.Pattern:
    """A spoken request: lead words, a polite form, one of ``phrases``, then tail words."""
    return re.compile(rf"(?:{_LEAD} )*(?:{_POLITE} )?(?:{phrases})(?: {_TAIL})*")


_GRAMMAR = _request_grammar("|".join(f"(?P<{command.value}>{spec.phrases})"
                                     for command, spec in COMMANDS.items() if spec.phrases))
_OTHER_PERSON = r"(?:him|her|them|my friend|someone(?: else)?|somebody(?: else)?|the other person|my guest)"
_GAZE = _request_grammar(r"(?:say (?:hi|hello) to|greet) (?P<greeted>[a-z0-9']+(?: [a-z0-9']+){0,3})"
                         rf"|look at {_OTHER_PERSON}")
_NOT_ANOTHER_PERSON = frozenset((
    "me", "myself", "us", "ourselves", "you", "yourself", "yourselves", "it", "this", "that",
    "muse", "reachy", "robot", "nobody", "none"))
# Quotation marks, not apostrophes inside a word.
_QUOTED = re.compile(r"[\"“”]|(?<![a-z])['‘’]|['‘’](?![a-z])", re.IGNORECASE)
# "Say hi to her for me", "when you call her" or "later": the greeting is for someone who is not here.
_ABSENT = re.compile(r"\b(?:for (?:me|us)|from (?:me|us)|when(?:ever)?|if|next time|later|tomorrow|tonight)\b")
_QUESTION_OPENERS = frozenset((
    "what", "what's", "whats", "when", "where", "who", "why", "how", "which", "is", "are", "was", "were",
    "does", "did", "do", "should"))
_NEGATIONS = frozenset(("don't", "dont", "not", "never", "no"))
_ELSEWHERE = frozenset((
    "tv", "television", "music", "song", "songs", "playlist", "spotify", "radio", "podcast", "video", "movie",
    "light", "lights", "lamp", "thermostat", "heater", "fan", "alarm", "timer", "phone", "laptop", "computer",
    "app", "zoom", "iphone", "car", "door", "kitchen", "bedroom", "baby", "kids", "dog", "cat"))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.replace("’", "'").casefold())


def _spoken(text: str) -> str:
    return " ".join(_words(text))


def parse_command(text: str) -> Optional[DeviceCommand]:
    command = whole_command(text)
    if command is not None:
        return command
    if classify_follow_up(text) is FollowUp.CANCEL:
        return DeviceCommand.STOP_TALKING
    return None


def whole_command(text: str) -> Optional[DeviceCommand]:
    """A hardware command, without converting work cancellation into playback control."""
    match = _GRAMMAR.fullmatch(_spoken(text))
    if match is not None:
        return DeviceCommand(match.lastgroup)
    if classify_follow_up(text) is FollowUp.SILENCE:
        return DeviceCommand.STOP_TALKING
    return None


_SOUND_CODES = str.maketrans({letter: str(code) for code, letters in enumerate(
    ("aeiouhwy", "bfpv", "cgjkqsxz", "dt", "l", "mn", "r")) for letter in letters})


def _sound_key(word: str) -> str:
    return "".join(code for code, _ in groupby(word.translate(_SOUND_CODES)) if code != "0")


def command_after_wake(text: str, phrase: str) -> Optional[str]:
    """Match an acoustic wake prefix without consuming a recognized device command."""
    words, wake = _words(text), _words(phrase)
    if not wake:
        return None
    boundary_guards = (_NEGATIONS | _ELSEWHERE | _QUESTION_OPENERS) - set(wake)
    expected = "".join(wake)
    leading = "".join(wake[:-1])
    best_score = 0.0
    command = None
    for end in range(max(1, len(wake) - 1), min(len(words), len(wake) + 2)):
        prefix = words[:end]
        if not "".join(prefix).startswith(leading) or boundary_guards.intersection(prefix):
            continue
        if len(prefix) > len(wake) and "".join(prefix) != expected:
            continue
        score = SequenceMatcher(None, "".join(prefix), expected, autojunk=False).ratio()
        sound = (len(prefix) == len(wake) and prefix[:-1] == wake[:-1]
                 and len(_sound_key(wake[-1])) >= 2 and _sound_key(prefix[-1]) == _sound_key(wake[-1]))
        if sound:
            score = 1.0
        if score > best_score:
            best_score = score
            remainder = " ".join(words[end:])
            if score >= .75 and whole_command(remainder) is not None:
                command = remainder
    return command


def gaze_requested(text: str) -> bool:
    """A request to face someone other than the speaker. It is never a device command.

    A greeting may name the person in up to four words; "look at" takes only the short list.
    """
    spoken = _spoken(text)
    if _QUOTED.search(text) or _ABSENT.search(spoken) or not _NEGATIONS.isdisjoint(spoken.split()):
        return False
    match = _GAZE.fullmatch(spoken)
    return match is not None and (match["greeted"] is None
                                  or match["greeted"].split()[0] not in _NOT_ANOTHER_PERSON)


def worth_asking(text: str, asked: FrozenSet[DeviceCommand]) -> bool:
    if gaze_requested(text):
        return False
    words = _words(text)
    if not words or len(words) > MODEL_MAX_WORDS or words[0] in _QUESTION_OPENERS:
        return False
    if any(word in _NEGATIONS or word in _ELSEWHERE for word in words):
        return False
    return any(supported_by(command, text) for command in asked)


def supported_by(command: DeviceCommand, text: str) -> bool:
    return not COMMANDS[command].evidence.isdisjoint(_words(text))


def executable(*, wake: bool, output: bool, face: bool, volume: bool) -> FrozenSet[DeviceCommand]:
    have = {Need.WAKE_WORD: wake, Need.SPEECH_QUEUED: output, Need.FACE: face, Need.VOLUME: volume}
    return frozenset(command for command, spec in COMMANDS.items() if have[spec.needs])


class CommandModel(Protocol):
    @property
    def available(self) -> bool: ...

    async def classify(self, text: str, allowed: FrozenSet[DeviceCommand]) -> Optional[DeviceCommand]: ...


@dataclass(frozen=True)
class Commands:
    model: Optional[CommandModel] = None

    async def recognize(self, text: str, allowed: FrozenSet[DeviceCommand]) -> Optional[DeviceCommand]:
        command = parse_command(text)
        if command is not None:
            return command if command in allowed else None
        asked = frozenset(command for command in allowed if COMMANDS[command].evidence)
        if self.model is None or not asked or not worth_asking(text, asked) or not self.model.available:
            return None
        try:
            command = await asyncio.wait_for(self.model.classify(text, asked), CLASSIFY_DEADLINE_S)
        except Exception as error:
            log.info("Reachy's command model gave no answer: %s", type(error).__name__)
            return None
        return command if command in asked and supported_by(command, text) else None
