# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

import re
from enum import Enum
from typing import Optional, Tuple


class Intent(str, Enum):
    CHAT = "chat"
    QUESTION = "question"
    ACTION = "action"
    FOLLOW_UP = "follow_up"

    @classmethod
    def parse(cls, value: object) -> Optional[Intent]:
        try:
            return cls(value)
        except (ValueError, TypeError):
            return None


DENY_WORDS = tuple("weather rain forecast temperature time date today's tomorrow tonight calendar schedule "
                   "remind reminder timer alarm email emails message text call send book order buy price cost "
                   "news search find look check play who where when which why explain recommend suggest idea "
                   "ideas please remember forget".split())
DENY_PHRASES = ("what time", "how much", "how many", "tell me", "should i", "can you", "could you", "would you",
                "my list", "my notes", "my tasks", "my contacts", "my files", "my account")
SOCIAL_PATTERNS = (
    r"(?:hi|hello|hey|hiya|yo)(?: reachy| there)?",
    r"(?:good (?:morning|afternoon|evening|night)|goodnight|bye|goodbye|see you(?: later)?)(?: reachy)?",
    r"(?:thanks?|thank you|cheers)(?: (?:so|very) much)?(?: (?:that was|that is|that's) (?:really |so )?(?:great|helpful|nice|perfect))?",
    r"(?:haha|ha|lol|wow|yay|cool|nice|awesome|okay|ok|great)(?: (?:that's|that is|that was) (?:really |so )?(?:funny|great|nice))?",
    r"(?:ugh|argh|phew)(?: what a (?:day|week|morning|night))?",
    r"how are you(?: doing)?(?: today)?",
    r"how(?:'s| is) it going|how are things|what(?:'s| is) up",
    r"how(?:'s| is| was) your day(?: been| going)?",
    r"(?:you(?:'re| are)|that (?:was|is)|that's) (?:a )?(?:really |very |so |pretty )?(?:funny|awesome|great|the best|helpful|"
    r"perfect|nice|sweet|cool|cute|smart)(?: of you)?(?: little robot| robot)?",
    r"do you ever (?:get bored|feel lonely|dream|get tired)(?: sitting there| up there)?",
    r"i(?:'m| am) (?:(?:so|really|a bit|kind of) )?(?:tired|sleepy|bored|happy|sad|stressed|nervous|excited|exhausted)(?: today)?",
)
FIRST_PERSON_PATTERNS = (r"i (?:just |really )?(?:love|like|enjoy|feel|got back|had|went|am feeling)\b.*",)
FOLLOW_UP_PREFIXES = ("and", "what about", "how about", "also", "then", "so", "wait", "why is that",
                      "say that again", "what did you just say", "tell me more", "really", "are you sure",
                      "make it", "that", "it", "which one")
ACTION_VERBS = tuple("remind set add book order send text email call schedule put cancel buy write save find play "
                     "turn create make delete move".split())
POLITENESS = ("can you", "could you", "would you", "please")


def _words(text: str) -> str:
    return " ".join(re.findall(r"[\w']+", text.casefold().replace("’", "'")))


def _leads(text: str, prefixes) -> bool:
    return any(text == prefix or text.startswith(prefix + " ") for prefix in prefixes)


def maybe_chat(text: str) -> bool:
    words = _words(text)
    if not words or len(words.split()) > 12 or re.search(r"\d", text):
        return False
    if set(words.split()).intersection(DENY_WORDS):
        return False
    if any(re.search(r"\b" + re.escape(phrase) + r"\b", words) for phrase in DENY_PHRASES):
        return False
    if any(re.fullmatch(pattern, words) for pattern in SOCIAL_PATTERNS):
        return True
    return "?" not in text and any(re.fullmatch(pattern, words) for pattern in FIRST_PERSON_PATTERNS)


def guess_intent(text: str, previous: str = "") -> Intent:
    words = _words(text)
    if maybe_chat(text):
        return Intent.CHAT
    if previous and _leads(words, FOLLOW_UP_PREFIXES):
        return Intent.FOLLOW_UP
    while _leads(words, POLITENESS):
        prefix = next(prefix for prefix in POLITENESS if words == prefix or words.startswith(prefix + " "))
        words = words[len(prefix):].strip()
    return Intent.ACTION if _leads(words, ACTION_VERBS) else Intent.QUESTION


FALLBACK_OPENERS = {
    Intent.QUESTION: ("Hmm, good question.", "Ooh, let me think.", "Give me a second on that.", "Hmm, just a moment.", "Oh, interesting question."),
    Intent.ACTION: ("On it.", "Sure, I'll get that going.", "Okay, I'll ask Muse to handle that.", "Sure thing, just a moment."),
    Intent.FOLLOW_UP: ("Right, about that.", "Mm, okay.", "Sure, give me a sec.", "Good point, just a moment."),
    Intent.CHAT: (),
}

# A narrow grammar can answer these without inventing facts or dropping a request.
FALLBACK_CHAT = (
    (r"(?:hi|hello|hey|hiya|yo)(?: there)?", "Hello!"),
    (r"good morning", "Good morning!"),
    (r"good afternoon", "Good afternoon!"),
    (r"good evening", "Good evening!"),
    (r"good ?night|bye|goodbye|see you(?: later)?", "See you later!"),
    (r"(?:thanks?|thank you|cheers)(?: (?:so|very) much)?", "You're welcome!"),
    (r"how are you(?: doing)?(?: today)?|how(?:'s| is) it going|how are things|what(?:'s| is) up",
     "I'm here and ready to chat."),
)
CHAT_PHRASES = tuple(reply for _, reply in FALLBACK_CHAT)


def chat_reply(text: str) -> Optional[str]:
    if maybe_chat(text):
        words = _words(text).removesuffix(" reachy")
        return next((reply for pattern, reply in FALLBACK_CHAT if re.fullmatch(pattern, words)), None)
    return None


def pick(phrases: Tuple[str, ...], turn: int, recently_said: Tuple[str, ...]) -> Optional[str]:
    for offset in range(len(phrases)):
        phrase = phrases[(turn + offset) % len(phrases)]
        if phrase not in recently_said:
            return phrase
    return None
