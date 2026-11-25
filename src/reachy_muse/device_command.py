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


class PhotoSubject(Enum):
    """Whom or what a granted photo request asks Reachy to frame."""

    ME = "me"
    HELD = "held"
    US = "us"
    OTHER = "other"
    THIS = "this"
    SCENE = "scene"
    UNPARSED = "unparsed"


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
    "muse", "reachy", "robot", "alexa", "siri", "google", "nobody", "none"))
_PEOPLE = frozenset(("kids", "children", "guests", "visitors", "family", "boys", "girls"))
_TRAILING_TAIL = re.compile(rf"(?: (?:{_TAIL}))+$")
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
# A photo's subject ends at these words: what follows is a condition, a time, a recipient or another request.
_PHOTO_STOP = r"(?:and|then|when|whenever|if|later|tomorrow|tonight|every|after|before|until|once|while|to|for)"
_PHOTO_WORD = rf"(?!{_PHOTO_STOP}\b)[a-z0-9']+"
_SUBJECT = rf"{_PHOTO_WORD}(?: {_PHOTO_WORD}){{0,4}}"
_ONE_PHOTO = rf"(?:a|an|one|another|a quick|a new) (?:photo|picture|pic|snapshot|selfie)(?: of {_SUBJECT})?"
_PHOTO_QUESTION = (r"(?:tell me|describe|what|what's|whats|who|who's|which|how|how's|is|are|does|do"
                   r"|can you|could you|read|identify|explain|count|check|guess"
                   r"|ask(?: muse| it)? (?:what|who|which|how|if|whether|about))\b.*")
# After "look at this", a question rather than an exclamation such as "what a mess".
_LOOK_QUESTION = (r"(?:tell me|describe|read|identify|explain|what(?! an? )|what's|whats|who|who's|which"
                  r"|is (?:it|this|that)|are (?:these|those)|does (?:it|this|that)|can you|could you"
                  r"|how (?:many|much|old|big|long|do|does|is|are))\b.*")
# Places a thing is shown. A screen, page, picture or photo of the user's is as often on a phone.
_IN_VIEW = (r"(?:table|desk|counter|shelf|floor|wall|window|whiteboard|board|plate|bowl|box|bag|hand|hands"
            r"|book|room|kitchen|fridge)")
# "What's on my plate" or "on the board" is often a to-do list; these are places a thing is shown.
_ON_VIEW = r"(?:table|desk|counter|shelf|floor|wall|window|whiteboard)"
_LOOKS = r"(?:outfit|hair|haircut|shirt|dress|look|shoes|glasses|jacket|tie|makeup|sweater|hat)"
# "Can you see my calendar?" asks about the user's data, so only these things are seen; a board, window, table,
# book or box also names data.
_SEEN_THING = (rf"(?:{_LOOKS}|face|eyes|hand|hands|cat|dog|puppy|kitten|plant|plants|desk|counter|shelf|floor|wall"
               r"|whiteboard|plate|bowl|bag|room|kitchen|fridge)")
# Kinds of things seen rather than read or heard: animals, plants, food and objects. "What kind of email is
# this?" is not one, and neither is a bird, a bug, a shell, a tool, a coin or a rock, which also name sounds or data.
_SEEN_KIND = (r"(?:(?:animal|dog|cat|puppy|kitten|fish|insect|spider|snake|lizard|frog|butterfly|beetle|plant"
              r"|flower|tree|mushroom|weed|cactus|fruit|vegetable|nut|food|dish|cheese|bread|cake|stone|crystal"
              r"|car|bike|shoe|hat|shirt|dress|jacket|fabric|wood|toy|cup|mug|bottle|chair|lamp)(?:s|es)?"
              r"|leaf|leaves|berry|berries)")
# Things read off what Reachy is shown; a note, list, letter or page is as often in the user's apps.
_READABLE = r"(?:label|sign|menu|receipt|bottle|box|package|ingredients|poster|tag|price|book|paper|whiteboard)"
_KIND = r"(?:color|colour|kind|type|sort|breed|brand|make|species|size|shape)"
_ASK_TO_SEE = (
    r"what (?:do|can) you see(?: (?:here|there|in front of you|from here|from there|from where you are"
    rf"|(?:on|in|under|behind|near|next to|over) (?:the|my|this|your) {_IN_VIEW}))?"
    r"|what are you seeing|what(?:'s| is) in front of you"
    r"|what am i (?:holding(?: up)?|showing you|wearing|pointing at)"
    r"|what(?:'s| is) this (?:i'm|i am) holding|what(?:'s| is) in my hands?"
    r"|what (?:do i have|have i got|i've got) in my hands?"
    rf"|what(?:'s| is) (?:on|under|behind|next to) (?:the|my|this|your) {_ON_VIEW}"
    r"|what(?:'s| is) this"
    rf"|what {_KIND}(?: of (?:{_PHOTO_WORD} )?{_SEEN_KIND})? (?:is this|are these)(?: {_SEEN_KIND})?"
    rf"|what does this(?: {_READABLE})? say|read this(?: {_READABLE})?(?: out loud| aloud| to me)?"
    r"|tell me what this (?:is|says)|identify this"
    rf"|(?:can|do) you see (?:this|me|us|what i'm holding|what i am holding|(?:my|the) {_SEEN_THING})"
    r"|(?:tell me|describe) what you (?:can )?see"
    r"|describe (?:this|me|us|the room|what's in front of you|what is in front of you)"
    rf"|how do i look|do i look (?:okay|ok|good|alright|all right|fine|nice|presentable|ready)"
    rf"|what do you think (?:of|about) (?:my|this) {_LOOKS}"
    rf"|how many {_PHOTO_WORD}(?: {_PHOTO_WORD})? am i holding(?: up)?|how many (?:fingers|people) (?:do|can) you see")
_TAKE_PHOTO = (rf"(?:take|snap|grab) {_ONE_PHOTO}|photograph (?!of\b){_SUBJECT}|send muse {_ONE_PHOTO}"
               rf"|send {_ONE_PHOTO} to muse")
_SHOW_MUSE = r"show this to muse|show muse this|send this to muse|send muse this"
_THEN = r"(?:(?:and|then|and then) )?"
# Narrower politeness than device commands: "you can take a photo later" describes rather than asks. Muse is the
# one place a photo may be sent: "take a photo for Muse", "take a picture and send it to Muse".
_PHOTO = re.compile(
    rf"(?:{_LEAD} )*(?:(?:please|(?:can|could|would|will) you(?: please)?|i (?:want|need) you to"
    r"|i'd like you to|i would like you to|go ahead and) )?(?:"
    rf"(?:{_TAKE_PHOTO})(?: for muse)?(?: (?:and|then|and then) send it to muse)?"
    rf"(?: {_THEN}(?P<question>{_PHOTO_QUESTION}))?"
    rf"|(?:{_SHOW_MUSE})(?: {_THEN}(?P<shown>{_PHOTO_QUESTION}))?"
    rf"|{_ASK_TO_SEE}"
    rf"|(?:look|have a look|take a look) at this(?: and)? (?P<look>{_LOOK_QUESTION})"
    r")(?: (?:please|now|right now|for me|thanks|thank you|reachy|muse))*")
# Anywhere in the request, these put the photo off until later.
_PHOTO_LATER = frozenset((
    "later", "tomorrow", "tonight", "soon", "every", "minute", "minutes", "min", "mins", "hour", "hours", "hr", "hrs",
    "seconds", "secs", "sec", "o'clock", "pm", "noon", "midnight", "week", "weeks", "month", "months", "year",
    "years", "timer", "countdown", "delay", "delayed", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday"))
_COUNT = (r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty"
          r"|forty|fifty|sixty|a few|a couple|a couple of|few|several)")
_PHOTO_AT_A_TIME = re.compile(
    r"\b(?:at (?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|night|dawn|dusk|sunrise|sunset)"
    rf"|in (?:{_COUNT} )?(?:sec|secs|second|seconds|min|mins|minute|minutes|hour|hours|moment|moments|bit|while"
    r"|jiffy|day|days)"
    r"|on (?:three|3|go|cue|my (?:count|signal|mark|cue|go)|the count of [a-z0-9]+)|count(?:ing)? (?:to|down|from))\b"
    # "this morning" or "in the evening" is a time; "my morning coffee" or "the evening sky" is not.
    r"|\b(?:this|next|every|each|tomorrow|yesterday|later|until|till|by|during|in|on|over)(?: the)?"
    r" (?:morning|afternoon|evening|night|weekend)s?\b|\b(?:mornings|afternoons|evenings|nights|weekends)\b")
# A question after the photo may not share, keep or delete it, or take the request back.
_PHOTO_SHARED = frozenset((
    "send", "sends", "sending", "sent", "share", "shares", "sharing", "shared", "post", "posts", "posting", "posted",
    "upload", "uploads", "uploading", "uploaded", "save", "saves", "saving", "saved", "store", "stores", "storing",
    "stored", "keep", "keeps", "keeping", "kept", "delete", "deletes", "deleting", "deleted", "mail", "mails",
    "mailing", "mailed", "email", "emails", "emailing", "emailed", "text", "texts", "texting", "texted", "message",
    "messages", "messaging", "messaged", "forward", "forwards", "forwarding", "forwarded", "whatsapp", "tweet",
    "tweets", "tweeting", "tweeted", "instagram", "facebook", "dm", "airdrop", "print", "prints", "printing",
    "printed", "publish", "publishes", "publishing", "published", "album", "remember", "remembers", "remembering",
    "don't", "dont", "never"))
# Quotation marks, not apostrophes inside a word or closing a plural possessive, as in "the kids' drawing".
_PHOTO_QUOTED = re.compile(r"[\"“”]|(?<![a-z])['‘’]|(?<!s)['‘’](?![a-z])|(?<=s)['‘’](?! [a-z])", re.IGNORECASE)


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
    return match is not None and (match["greeted"] is None or _greets_someone_here(match["greeted"]))


def _greets_someone_here(greeted: str) -> bool:
    """Names and "my mom" are people in the room; "the camera" or "Bob in the email" are not."""
    greeted = _TRAILING_TAIL.sub("", greeted)
    if re.fullmatch(_OTHER_PERSON, greeted):
        return True
    words = greeted.split()
    if words[0] in _NOT_ANOTHER_PERSON or "in" in words or "on" in words:
        return False
    return words[0] != "the" or (len(words) == 2 and words[1] in _PEOPLE)


def photo_requested(text: str) -> bool:
    """A spoken request for one photo now, or to say what Reachy sees. It is never a device command.

    The request must open the utterance, so negated, quoted, reported and conditional forms never match.
    One that puts the photo off, or asks to share or keep it, is refused.
    """
    spoken = _spoken(text)
    if _PHOTO_QUOTED.search(text) or not _PHOTO_LATER.isdisjoint(spoken.split()) or _PHOTO_AT_A_TIME.search(spoken):
        return False
    match = _PHOTO.fullmatch(spoken)
    return match is not None and _PHOTO_SHARED.isdisjoint(
        (match["question"] or match["shown"] or match["look"] or "").split())


_WORN = r"(?:face|smile|eyes|hair|haircut|glasses|makeup|hat)"
_LOOKS_FACE = rf"(?:{_WORN}|look)"
_LOOKS_BODY = r"(?:outfit|shirt|dress|shoes|jacket|tie|sweater|hands?)"
# Words before a look that ask about the speaker's own, as in "my hair", "think of this hat" or "does this shirt".
_ASKED_LOOK = r"(?:my|(?:think (?:of|about)|does|do|if|whether) (?:this|these))"
_RELATIVE = (r"(?:friend|guest|mom|mum|mother|dad|father|wife|husband|partner|son|daughter|brother|sister|kid|boy|girl"
             r"|baby|grandma|grandpa|grandmother|grandfather|boyfriend|girlfriend|roommate|colleague|neighbou?r"
             r"|uncle|aunt|auntie|cousin|nephew|niece)")
_PHOTO_SLOT = re.compile(rf"\b(?:photo|picture|pic|snapshot|selfie) of ({_SUBJECT})|\bphotograph (?!of\b)({_SUBJECT})")
# Words after a photo's subject that leave whom it frames as it is, as in "of me again" or "of us together".
_SUBJECT_TAIL = re.compile(rf"(?: (?:{_TAIL}|again|here|there|today|together|too|also|as well"
                           r"|(?:right|over) (?:here|there)|this time|one more time|once more))+$")
# The subject a request names, matched whole and in this order.
_SLOT_SUBJECTS = tuple((re.compile(pattern), subject) for pattern, subject in (
    (rf"(?:me|myself) (?:holding|wearing|showing|with|in) .+|my (?:[a-z]+ )?{_LOOKS_BODY}"
     r"|what i(?:'m| am) (?:holding|wearing)",
     PhotoSubject.HELD),
    (rf"(?:me|myself)(?: [a-z']+)*|my (?:[a-z]+ )?{_LOOKS_FACE}", PhotoSubject.ME),
    (r"(?!.* on (?:my|the) (?:screen|tv|phone|laptop)\b)"
     r"(?:us|everyone|everybody|(?:both|all|the (?:two|three|four|five)) of us)(?: [a-z']+)*|ourselves"
     r"|the (?:whole )?(?:group|family|kids|children|guests)|these (?:people|guys|folks|kids|children)"
     r"|(?:my|our) (?:whole )?(?:family|friends|kids|children|guests|parents|grandparents)", PhotoSubject.US),
    (r"him|her|them|someone else|somebody else|the other person"
     rf"|(?:the|this|that) (?:person|guy|man|woman|lady|boy|girl|kid|baby)|my (?:[a-z']+ )?{_RELATIVE}(?: [a-z']+)?",
     PhotoSubject.OTHER),
    (r"(?:this|these)(?: [a-z']+){0,2}", PhotoSubject.THIS),
    (r"(?:the|this|my) (?:room|view|table|desk|kitchen|counter|shelf|window|wall|floor)", PhotoSubject.SCENE),
))
# Without a named subject, what the request asks about, searched for in this order.
_ASKED_SUBJECTS = tuple((re.compile(pattern), subject) for pattern, subject in (
    (r"\bus\b|\bourselves\b|\beveryone\b|\beverybody\b|\bhow many people\b|\bselfie\b", PhotoSubject.US),
    (rf"\bwhat (?:am i|i(?:'m| am)) (?:holding|showing|wearing)|\bi(?:'m| am) holding\b|\bin my hands?\b"
     rf"|\bam i holding\b|\bhow many fingers\b|\b{_ASKED_LOOK} {_LOOKS_BODY}\b", PhotoSubject.HELD),
    (rf"\b(?:see|describe) me\b|\b(?:how (?:[a-z]+ )?|(?:do|if|whether) )i look\b"
     rf"(?! (?:after|up|into|through)\b)|\bwhat i look like\b|\b(?:i'm|i am|am i) smiling\b"
     rf"|\b(?:my|think (?:of|about) (?:this|these)) {_LOOKS_FACE}\b"
     rf"|\b(?:does|do|if|whether) (?:this|these) {_WORN}\b"
     rf"|\b(?:does|do|if|whether) (?:this|these) look(?: [a-z']+)* on me(?: {_TAIL})*$", PhotoSubject.ME),
    (r"\bthis\b|\bthese\b", PhotoSubject.THIS),
))


def photo_subject(text: str) -> PhotoSubject:
    """Whom or what a request that ``photo_requested`` granted asks Reachy to frame: the subject it names, as in "a
    photo of my friend", or else what it asks about, as in "what am I holding?". It never grants a photo.

    A named subject it does not know, such as "my coffee" or "Sarah", is UNPARSED; a request about nobody and
    nothing in particular is the SCENE.
    """
    spoken = _spoken(text)
    slot = _PHOTO_SLOT.search(spoken)
    if slot is not None:
        words = _SUBJECT_TAIL.sub("", next(group for group in slot.groups() if group))
        return next((subject for pattern, subject in _SLOT_SUBJECTS if pattern.fullmatch(words)),
                    PhotoSubject.UNPARSED)
    return next((subject for pattern, subject in _ASKED_SUBJECTS if pattern.search(spoken)), PhotoSubject.SCENE)


_NAMES = frozenset(("muse", "reachy"))
_NAME_OPENERS = frozenset(("oh", "okay", "ok", "alright", "hey", "hi", "um", "uh", "so", "now", "well"))
_NAME_CLOSERS = frozenset(("please", "now", "right", "thanks", "thank", "you"))
# A closing name after these is who the words are about or for, as in "send it to Muse", not who they are said to.
_NAME_OBJECT = frozenset(("to", "for", "of", "at", "about", "with", "from", "like", "send", "show", "ask", "tell", "call"))


def names_reachy(text: str) -> bool:
    """Whether the words open or close by naming Muse or Reachy, as in "Muse, take a photo" or "what's this, Reachy?"."""
    words = _words(text)
    while words and words[0] in _NAME_OPENERS:
        words = words[1:]
    if words and words[0] in _NAMES:
        return True
    while words and words[-1] in _NAME_CLOSERS:
        words = words[:-1]
    return len(words) > 1 and words[-1] in _NAMES and words[-2] not in _NAME_OBJECT


def worth_asking(text: str, asked: FrozenSet[DeviceCommand]) -> bool:
    if gaze_requested(text) or photo_requested(text):
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
