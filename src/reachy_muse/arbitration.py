# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Shared, effect-free arbitration contract and conservative final rules.

The live conversation can observe these decisions, but does not execute them yet.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Literal

from reachy_muse.contract.intent import ACTION_VERBS, DENY_PHRASES, DENY_WORDS
from reachy_muse.device_command import DeviceCommand, whole_command
from reachy_muse.follow_up import FollowUp, classify_follow_up

MAX_TRANSCRIPT_BYTES = 4096
MAX_CONTEXT_BYTES = 2048
MAX_TARGET_BYTES = 768
MAX_LINE_BYTES = 240


class Code(str, Enum):
    EMPTY = "empty"
    NOISE = "noise"
    NOT_ADDRESSED = "not_addressed"
    AUDIBLE_ECHO = "audible_echo"
    WHOLE_DEVICE = "whole_device_command"
    UNSUPPORTED_DEVICE = "unsupported_device"
    WHOLE_SOCIAL = "whole_social"
    EXPLICIT_CANCEL = "explicit_cancel"
    CLARIFICATION = "clarification_answer"
    EXPLICIT_DETAIL = "explicit_detail"
    UNRESOLVED = "unresolved"
    MODEL = "model_proposal"
    MODEL_DEADLINE = "model_deadline"
    MODEL_INVALID = "model_invalid"


@dataclass(frozen=True)
class Evidence:
    source: Literal["rule", "recognizer", "robot_state", "model", "deadline"]
    code: Code
    ref: str | None = None
    span: tuple[int, int] | None = None


@dataclass(frozen=True)
class TextRef:
    id: str
    author: Literal["user", "muse", "local"]
    text: str = field(repr=False)


@dataclass(frozen=True)
class Target:
    id: str
    request: TextRef
    amendments: tuple[TextRef, ...] = ()
    clarification: TextRef | None = None
    state: Literal["running", "held", "completed"] = "running"
    omitted: bool = False

    @property
    def excerpts(self) -> tuple[TextRef, ...]:
        return (self.request, *self.amendments, *((self.clarification,) if self.clarification else ()))


def bounded_target(target: Target) -> Target:
    """Keep both targets representable; omitted context cannot authorize a semantic target."""
    from dataclasses import replace

    remaining = MAX_TARGET_BYTES
    omitted = target.omitted or len(target.amendments) > 3

    def excerpt(item: TextRef) -> TextRef:
        nonlocal remaining, omitted
        raw = item.text.encode("utf-8")
        kept = raw[:remaining].decode("utf-8", errors="ignore")
        remaining -= len(kept.encode("utf-8"))
        omitted |= kept != item.text
        return replace(item, text=kept)

    request = excerpt(target.request)
    amendments = tuple(excerpt(item) for item in target.amendments[:3])
    clarification = excerpt(target.clarification) if target.clarification else None
    return replace(target, request=request, amendments=amendments, clarification=clarification, omitted=omitted)


@dataclass(frozen=True)
class Snapshot:
    turn_id: str
    text: str = field(repr=False)
    revision: int = 0
    context_epoch: int = 0
    generation: int = 0
    phase: Literal["provisional", "final"] = "final"
    budget_ms: int = 50
    admission: Literal["wake", "follow_up", "closed", "other_person"] = "wake"
    user_speech: bool = True
    noise: bool = False
    echo_ref: str | None = None
    truncated: bool = False
    recognizer: Literal["local", "companion", "fixture"] = "local"
    capabilities: frozenset[DeviceCommand] = frozenset()
    local_chat: bool = True
    output: bool = False
    owns_chat: bool = False
    muse_ready: bool = True
    active: Target | None = None
    held: Target | None = None
    recent: tuple[TextRef, ...] = ()

    @property
    def targets(self) -> tuple[Target, ...]:
        return tuple(target for target in (self.active, self.held) if target is not None)

    @property
    def refs(self) -> tuple[TextRef, ...]:
        return tuple(item for target in self.targets for item in target.excerpts) + self.recent

    @property
    def model_eligible(self) -> bool:
        return (self.admission in ("wake", "follow_up") and not self.truncated
                and len(self.text.encode("utf-8")) <= MAX_TRANSCRIPT_BYTES and self.budget_ms > 0)


@dataclass(frozen=True)
class Speech:
    text: str = field(repr=False)
    role: Literal["local_reply", "opener"] = "local_reply"
    template_id: str | None = None
    source_refs: tuple[str, ...] = ()
    audio_stream: int | None = None


@dataclass(frozen=True)
class WaitPlan:
    latency_class: Literal["ordinary", "tools", "long_reply", "unknown"] = "unknown"
    first_text_p50_ms: int | None = None
    first_text_p95_ms: int | None = None
    opener_at_ms: int | None = None
    opener: Speech | None = None
    progress_first_ms: int = 5000
    progress_interval_ms: int = 15000
    progress_max: int = 3
    motion: Literal["none", "cached_thinking"] = "cached_thinking"


@dataclass(frozen=True)
class MuseRequest:
    length: Literal["brief", "detailed", "user_requested"] = "brief"
    format: Literal["plain_speech", "spoken_steps"] = "plain_speech"
    context_refs: tuple[str, ...] = ()
    wait: WaitPlan = WaitPlan()


@dataclass(frozen=True)
class LocalDevice:
    command: DeviceCommand
    kind: Literal["LOCAL_DEVICE"] = field(default="LOCAL_DEVICE", init=False)


@dataclass(frozen=True)
class LocalReply:
    speech: Speech
    kind: Literal["LOCAL_REPLY"] = field(default="LOCAL_REPLY", init=False)


@dataclass(frozen=True)
class Muse:
    request: MuseRequest = MuseRequest()
    kind: Literal["MUSE"] = field(default="MUSE", init=False)


@dataclass(frozen=True)
class MuseFollowUp:
    target_id: str
    request: MuseRequest = MuseRequest()
    kind: Literal["MUSE_FOLLOW_UP"] = field(default="MUSE_FOLLOW_UP", init=False)


@dataclass(frozen=True)
class MuseCancel:
    target_id: str
    kind: Literal["MUSE_CANCEL"] = field(default="MUSE_CANCEL", init=False)


@dataclass(frozen=True)
class Ignore:
    reason: Literal["empty", "noise", "echo", "not_addressed"]
    kind: Literal["IGNORE"] = field(default="IGNORE", init=False)


Action = LocalDevice | LocalReply | Muse | MuseFollowUp | MuseCancel | Ignore


@dataclass(frozen=True)
class Decision:
    turn_id: str
    revision: int
    context_epoch: int
    generation: int
    phase: Literal["provisional", "final"]
    confidence: float
    evidence: tuple[Evidence, ...]
    action: Action


class InvalidArbitration(ValueError):
    """Contains only a fixed error code, never caller or model content."""


def validate_snapshot(snapshot: Snapshot) -> None:
    if (not isinstance(snapshot.turn_id, str) or not snapshot.turn_id
            or not isinstance(snapshot.text, str) or any(type(n) is not int or n < 0 for n in
                                   (snapshot.revision, snapshot.context_epoch, snapshot.generation, snapshot.budget_ms))
            or snapshot.phase not in ("provisional", "final")
            or snapshot.admission not in ("wake", "follow_up", "closed", "other_person")):
        raise InvalidArbitration("identity")
    if (any(type(flag) is not bool for flag in (snapshot.user_speech, snapshot.noise, snapshot.truncated,
                                              snapshot.local_chat, snapshot.output, snapshot.owns_chat, snapshot.muse_ready))
            or snapshot.recognizer not in ("local", "companion", "fixture")):
        raise InvalidArbitration("input_evidence")
    if any(not isinstance(command, DeviceCommand) for command in snapshot.capabilities):
        raise InvalidArbitration("capability")
    ids = [target.id for target in snapshot.targets]
    refs = [item.id for item in snapshot.refs]
    if (any(not isinstance(id, str) or not id for id in (*ids, *refs))
            or len(ids) != len(set(ids)) or len(refs) != len(set(refs))):
        raise InvalidArbitration("references")
    for target in snapshot.targets:
        if (len(target.amendments) > 3 or target.request.author != "user"
                or type(target.omitted) is not bool
                or any(item.author != "user" for item in target.amendments)
                or (target.clarification is not None and target.clarification.author != "muse")
                or sum(len(item.text.encode("utf-8")) for item in target.excerpts) > MAX_TARGET_BYTES):
            raise InvalidArbitration("target_context")
    if ((snapshot.active is not None and snapshot.active.state not in ("running", "completed"))
            or (snapshot.held is not None and snapshot.held.state != "held")):
        raise InvalidArbitration("target_state")
    if (len(snapshot.recent) > 10 or any(item.author not in ("user", "muse", "local") for item in snapshot.recent)
            or sum(len(item.text.encode("utf-8")) for item in snapshot.refs) > MAX_CONTEXT_BYTES):
        raise InvalidArbitration("context_limit")
    if snapshot.echo_ref is not None and not any(
            item.id == snapshot.echo_ref and item.author == "local" for item in snapshot.recent):
        raise InvalidArbitration("echo_reference")


TEMPLATES = MappingProxyType({
    "hello": "Hello!",
    "morning": "Good morning!",
    "afternoon": "Good afternoon!",
    "evening": "Good evening!",
    "thanks": "You're welcome.",
    "ready": "Ready for a chat.",
    "pleased": "I'm glad you're here.",
    "empathy": "I'm here for a chat.",
    "bye": "Bye!",
})

_SOCIAL = (
    (r"(?:hi|hello|hey|hiya|yo)(?: reachy| there)?", "hello"),
    (r"good morning(?: reachy)?", "morning"),
    (r"good afternoon(?: reachy)?", "afternoon"),
    (r"good evening(?: reachy)?", "evening"),
    (r"(?:okay |ok )?(?:thanks?|thank you|cheers)(?: (?:so|very) much)?"
     r"(?: (?:that was|that is|that's) (?:really |so )?(?:great|helpful|nice|perfect))?", "thanks"),
    (r"how are you(?: doing)?(?: today)?|how(?:'s| is) it going|how are things|what(?:'s| is) up"
     r"|how(?:'s| is| was) your day(?: been| going)?", "ready"),
    (r"(?:you(?:'re| are)|that (?:was|is)|that's) (?:a )?(?:really |very |so )?"
     r"(?:funny|great|the best|helpful|nice|cute)(?: little robot| robot)?"
     r"|haha|ha|wow|yay|cool|nice|awesome", "pleased"),
    (r"i(?:'m| am)(?: feeling)? (?:(?:so|really|a bit|kind of) )?"
     r"(?:tired|sleepy|happy|sad|stressed|nervous|exhausted)(?: today)?|i had a rough day", "empathy"),
    (r"(?:bye|goodbye|see you(?: later)?)(?: reachy)?|it was nice chatting with you", "bye"),
)


def words(text: str) -> str:
    return " ".join(re.findall(r"[\w']+", text.casefold().replace("’", "'")))


def social_template(text: str) -> str | None:
    normalized = words(text)
    tokens = normalized.split()
    if (len(tokens) > 12 or re.search(r"\d", text)
            or set(tokens).intersection((*DENY_WORDS, *ACTION_VERBS, "and", "but", "then"))
            or any(re.search(r"\b" + re.escape(phrase) + r"\b", normalized) for phrase in DENY_PHRASES)):
        return None
    return next((name for pattern, name in _SOCIAL if re.fullmatch(pattern, normalized)), None)


def template_speech(name: str, text: str) -> Speech | None:
    from reachy_muse.capabilities import grounded
    from reachy_muse.companion_server.grounding import ungrounded

    line = TEMPLATES[name]
    # The closed bank is reviewed persona policy, not generated text citing itself.
    if ungrounded(line, text, TEMPLATES[name]) or not grounded(line, text, TEMPLATES[name]):
        return None
    return Speech(line, template_id=name, source_refs=("template:" + name,))


def _single_target(snapshot: Snapshot) -> Target | None:
    targets = snapshot.targets
    if len(targets) == 1 and targets[0].request.text.strip() and not targets[0].omitted:
        return targets[0]
    return None


def _cancel_target(snapshot: Snapshot) -> Target | None:
    if classify_follow_up(snapshot.text) is FollowUp.CANCEL:
        return _single_target(snapshot)
    names = r"recipe|weather|search|draft|explanation|comparison"
    match = re.fullmatch(rf"(?:please )?cancel (?:the |my )?({names})"
                         r"(?: request)?", words(snapshot.text))
    if match is None or any(target.omitted or not target.request.text.strip() for target in snapshot.targets):
        return None
    matches = []
    for target in snapshot.targets:
        topics = set()
        affirmative = False
        for item in (target.request, *target.amendments):
            text = words(item.text)
            if re.search(r"\b(?:no|not|never|neither|nor|without|except|exclude|avoid|skip|forget|cancel|stop|"
                         r"drop|omit|remove|instead|rather|replace|switch|change|actually|scratch)\b|n't\b", text):
                return None
            mentioned = set(re.findall(rf"\b(?:{names})\b", text))
            if mentioned:
                if not re.fullmatch(
                        r"(?:please )?(?:(?:can|could|would|will) you )?"
                        r"(?:find|get|give|show|prepare|write|make|create|draft|search|explain|compare|"
                        r"tell|what(?:'s| is)) "
                        r"(?:me )?(?:a |an |the |my )?"
                        r"(?:(?:dinner|vegetarian|vegan|email|short|detailed) )?"
                        rf"(?:{names})(?: forecast)?(?: in [a-z]+(?: [a-z]+){{0,2}})?"
                        r"(?: (?:for )?(?:today|tomorrow|tonight))?(?: please)?", text):
                    return None
                affirmative = True
            elif item is target.request:
                if not re.fullmatch(r"(?:please )?help me (?:choose|plan) (?:breakfast|lunch|dinner)", text):
                    return None
                topics.add("recipe")
            elif not re.fullmatch(
                    r"(?:(?:for )?(?:today|tomorrow|tonight)|"
                    r"for (?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve) (?:people|servings))"
                    r"(?: please)?", text):
                return None
            topics.update(mentioned)
            if len(topics) > 1:
                return None
        if not affirmative:
            return None
        if match[1] in topics:
            matches.append(target)
    return matches[0] if len(matches) == 1 else None


def _detail(snapshot: Snapshot) -> tuple[Target, Code] | None:
    target = _single_target(snapshot)
    if target is None or target.state == "completed":
        return None
    text = words(snapshot.text)
    if target.clarification:
        question = target.clarification.text.strip().casefold()
        if (re.fullmatch(r"(?:(?:should|shall|can) i (?:use (?:today|tomorrow)|include (?:weekends|an example))"
                         r"|(?:do you want|would you like) (?:the short version|more detail|an example))\?", question)
                and text in ("yes", "no", "yes please", "no thanks")):
            return target, Code.CLARIFICATION
    if re.fullmatch(r"(?:please |(?:can|could|would|will) you (?:please )?)?"
                    r"(?:give me the short version|explain that in more detail|use simpler words|"
                    r"say it as steps|include an example)(?: please)?", text):
        return target, Code.EXPLICIT_DETAIL
    return None


def decide(snapshot: Snapshot) -> Decision:
    """Normalize only for matching. The decision never replaces the retained transcript."""
    validate_snapshot(snapshot)
    if snapshot.truncated:
        raise InvalidArbitration("retry_transcript")

    def result(action: Action, code: Code, ref: str | None = None) -> Decision:
        return Decision(snapshot.turn_id, snapshot.revision, snapshot.context_epoch, snapshot.generation,
                        snapshot.phase, 1.0, (Evidence("rule", code, ref),), action)

    if snapshot.admission in ("closed", "other_person"):
        return result(Ignore("not_addressed"), Code.NOT_ADDRESSED)
    if not snapshot.text.strip():
        return result(Ignore("empty"), Code.EMPTY)
    if snapshot.noise and not snapshot.user_speech:
        return result(Ignore("noise"), Code.NOISE)
    if (snapshot.echo_ref is not None and not snapshot.user_speech and snapshot.admission != "wake"
            and any(item.id == snapshot.echo_ref and f" {words(snapshot.text)} " in f" {words(item.text)} "
                    for item in snapshot.recent)):
        return result(Ignore("echo"), Code.AUDIBLE_ECHO, snapshot.echo_ref)
    quotation_text = re.sub(r"(?<=\w)['’](?=\w)", "", snapshot.text)
    if any(character in "'`＇" or any(mark in unicodedata.name(character, "")
                                    for mark in ("QUOTATION MARK", "CORNER BRACKET"))
           for character in quotation_text):
        return result(Muse(), Code.UNRESOLVED)
    normalized = words(snapshot.text)
    command = whole_command(snapshot.text) if normalized.isascii() else None
    stop = re.fullmatch(
            r"(?:(?:oh|okay|ok|no|wait|actually|um|uh|hmm|please) )*stop"
            r"(?: stop)*(?: (?:please|thanks|thank you|now))*", normalized) is not None
    if stop and snapshot.output:
        command = DeviceCommand.STOP_TALKING
    if command is not None:
        if command in snapshot.capabilities and (command is not DeviceCommand.STOP_TALKING or snapshot.output):
            return result(LocalDevice(command), Code.WHOLE_DEVICE)
        return result(Muse(), Code.UNSUPPORTED_DEVICE)
    target = (_single_target(snapshot) if stop else _cancel_target(snapshot)) if normalized.isascii() else None
    if target is not None:
        return result(MuseCancel(target.id), Code.EXPLICIT_CANCEL, target.id)
    detail = _detail(snapshot)
    if detail is not None:
        target, code = detail
        return result(MuseFollowUp(target.id), code, target.id)
    template = social_template(snapshot.text) if snapshot.local_chat else None
    speech = template_speech(template, snapshot.text) if template else None
    if speech is not None:
        return result(LocalReply(speech), Code.WHOLE_SOCIAL)
    return result(Muse(), Code.UNRESOLVED)


def validate_decision(snapshot: Snapshot, decision: Decision) -> None:
    validate_snapshot(snapshot)
    if (any(type(n) is not int for n in (decision.revision, decision.context_epoch, decision.generation))
            or (decision.turn_id, decision.revision, decision.context_epoch, decision.generation, decision.phase)
            != (snapshot.turn_id, snapshot.revision, snapshot.context_epoch, snapshot.generation, snapshot.phase)):
        raise InvalidArbitration("stale_decision")
    if (type(decision.confidence) not in (int, float) or not math.isfinite(decision.confidence)
            or not 0 <= decision.confidence <= 1 or not 1 <= len(decision.evidence) <= 8):
        raise InvalidArbitration("confidence_or_evidence")
    refs = {item.id for item in snapshot.refs} | {target.id for target in snapshot.targets}
    boundaries = {0}
    offset = 0
    for character in snapshot.text:
        offset += len(character.encode("utf-8"))
        boundaries.add(offset)
    for evidence in decision.evidence:
        if (not isinstance(evidence.code, Code)
                or evidence.source not in ("rule", "recognizer", "robot_state", "model", "deadline")
                or (evidence.ref is not None and evidence.ref not in refs)):
            raise InvalidArbitration("evidence")
        if evidence.span is not None:
            if len(evidence.span) != 2 or any(type(n) is not int for n in evidence.span):
                raise InvalidArbitration("evidence_span")
            start, end = evidence.span
            if start not in boundaries or end not in boundaries or start >= end:
                raise InvalidArbitration("evidence_span")
    action = decision.action
    if not isinstance(action, (LocalDevice, LocalReply, Muse, MuseFollowUp, MuseCancel, Ignore)):
        raise InvalidArbitration("action")
    if isinstance(action, (MuseFollowUp, MuseCancel)):
        if not any(target.id == action.target_id and not target.omitted and target.request.text.strip()
                   for target in snapshot.targets):
            raise InvalidArbitration("target")
    if isinstance(action, (Muse, MuseFollowUp)):
        request = action.request
        if (request.length not in ("brief", "detailed", "user_requested")
                or request.format not in ("plain_speech", "spoken_steps")
                or len(request.context_refs) > 3 or not set(request.context_refs).issubset(refs)
                or request.wait != WaitPlan()):
            raise InvalidArbitration("request_plan")
    # Phase 1 admits only rule-supported effects. Model proposals remain an offline measurement.
    baseline = decide(snapshot).action
    if action != baseline and not (isinstance(action, Muse) and isinstance(baseline, Muse)):
        raise InvalidArbitration("unsupported_effect")


def muse_text(snapshot: Snapshot, decision: Decision) -> str:
    validate_decision(snapshot, decision)
    if snapshot.phase != "final" or not isinstance(decision.action, (Muse, MuseFollowUp)):
        raise InvalidArbitration("not_a_final_muse_request")
    return snapshot.text
