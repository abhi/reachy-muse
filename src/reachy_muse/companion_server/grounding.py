"""The narrator's own check that a spoken line adds no facts.

The robot runs the SDK's `grounded()` on every line and strikes the companion when one fails.
This check runs the installed `grounded()` too, and adds stricter rules of its own, so it does not
depend on how a later SDK normalizes text. It reads these facts from a line:

- numbers, with decimals, number words ("twenty-five"), ordinals ("12th", "third") and thousands commas;
- times ("7:45 pm", "8 a.m.", "noon"); a line may not add a.m. or p.m. that its source lacks;
- amounts ("$4.50", "12 dollars", "15%", "15 percent"); a line may not add a currency;
- date words: today, tonight, tomorrow, yesterday, morning, afternoon, evening, weekend, weekdays and months;
- names: capitalized words anywhere. A sentence may start with a common word or a verb in -ing;
  any other capitalized first word counts as a name.

Every fact must appear in one of the sources. Like `grounded()`, this catches invented figures and
names. It does not prove that the line is faithful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import FrozenSet, List, Optional, Tuple

from reachy_muse.capabilities import grounded

_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
          "sixteen seventeen eighteen nineteen").split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
_ORDINAL_UNITS = ("zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
                  "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth").split()
_ORDINAL_TENS = {"twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50, "sixtieth": 60, "seventieth": 70,
                 "eightieth": 80, "ninetieth": 90, "hundredth": 100, "thousandth": 1000}
_WORD_VALUES = {**{w: n for n, w in enumerate(_UNITS)}, **_TENS, **{w: n for n, w in enumerate(_ORDINAL_UNITS)}, **_ORDINAL_TENS}
_NUMBER_WORD = "|".join(sorted([*_WORD_VALUES, *_SCALES], key=len, reverse=True))
_NUMBER_PHRASE = re.compile(rf"\b(?:{_NUMBER_WORD})(?:(?:[\s-]+|\s+and\s+)(?:{_NUMBER_WORD}))*\b", re.IGNORECASE)

_MERIDIEM = r"(?:\s?([ap])\.?\s?m\b\.?)"
_TIME = re.compile(rf"\b(\d{{1,2}}):(\d{{2}}){_MERIDIEM}?|\b(\d{{1,2}}){_MERIDIEM}|\b(noon|midnight)\b", re.IGNORECASE)
_CURRENCY_SYMBOLS = {"$": "usd", "€": "eur", "£": "gbp", "¥": "jpy"}
_CURRENCY_WORDS = {"dollar": "usd", "dollars": "usd", "bucks": "usd", "euro": "eur", "euros": "eur",
                   "pound": "gbp", "pounds": "gbp", "yen": "jpy", "cent": "cent", "cents": "cent",
                   "percent": "%", "per cent": "%", "%": "%"}
_NUMBER = r"\d+(?:,\d{3})*(?:\.\d+)?"
_AMOUNT = re.compile(rf"([$€£¥])\s?({_NUMBER})|({_NUMBER})\s?(%|per cent\b|percent\b|dollars?\b|bucks\b|euros?\b|pounds?\b|yen\b|cents?\b)",
                     re.IGNORECASE)
_DIGITS = re.compile(rf"{_NUMBER}(?:st|nd|rd|th)?", re.IGNORECASE)
_DATE_WORDS = frozenset(
    "today tonight tomorrow yesterday morning afternoon evening weekend "
    "monday tuesday wednesday thursday friday saturday sunday "
    "january february march april june july august september october november december".split())
_WORD = re.compile(r"[^\W\d_][\w'’-]*")
_SENTENCE_END = re.compile(r"[.!?:;]['\")\]]*$")

# Words that may start a sentence without being a name. Closed-class words and conversational openers only:
# any other capitalized first word must appear in a source.
COMMON = frozenset("""
a an the this that these those there here it its it's i i'm i'll i've i'd you you're you'll you've your yours
we we're we'll our he she they them their his her my me us what which who whom whose when where why how
and but or so yet nor if then than because although though while as also just still only even not no yes
yeah yep nope ok okay sure alright right well oh ah hmm sorry thanks thank please hey hi hello great good
nice got let let's looks look seems sounds checking looking working searching getting finding pulling
is are was were be been being am do does did don't doesn't didn't can can't could couldn't would wouldn't
should shouldn't will won't shall may might must have has had haven't hasn't isn't aren't wasn't weren't
to of in on at by for with from about into over after before during until up down out off again once
all any each every both few more most some such one none nothing something anything everything
unfortunately however also actually currently right now still almost around about
ooh oh aw aww mm mmm hmm haha ha wow yay ugh whoa ah aha nice fair cool totally absolutely sounds
glad happy need want sweet anytime same yeah yep mhm oof interesting love lucky got hang huh
""".split())


@dataclass(frozen=True)
class Time:
    hour: int
    minute: int
    meridiem: Optional[str]  # "a", "p" or None when the text gave neither


@dataclass(frozen=True)
class Facts:
    numbers: FrozenSet[Decimal]
    times: FrozenSet[Time]
    amounts: FrozenSet[Tuple[str, Decimal]]
    date_words: FrozenSet[str]
    words: FrozenSet[str]


def facts(text: str) -> Facts:
    times, rest = _times(text)
    amounts = frozenset((unit, value) for unit, value in _amounts(rest))
    numbers = {_decimal(m[0]) for m in _DIGITS.finditer(rest)}
    numbers |= {value for phrase in _NUMBER_PHRASE.findall(rest) for value in _phrase_values(phrase)}
    numbers |= {Decimal(t.hour) for t in times} | {Decimal(t.minute) for t in times if t.minute}
    words = {_bare(w) for w in _WORD.findall(text)}
    return Facts(frozenset(numbers), frozenset(times), amounts, frozenset(words & _DATE_WORDS), frozenset(words))


ROBOT_REJECTS = "the robot's grounded() rejects it"


def ungrounded(line: str, *sources: str) -> List[str]:
    """Why the robot could reject `line`, as short descriptions. Empty means grounded.

    The robot's `grounded()` compares figures as written, so "40 percent" does not match "40%" there.
    """
    problems = missing_facts(line, *sources)
    if not problems and not grounded(line, *sources):
        problems.append(ROBOT_REJECTS)
    return problems


def missing_facts(line: str, *sources: str) -> List[str]:
    """The facts in `line` that no source states, by this module's rules alone."""
    said = facts(line)
    known = _merge(facts(s) for s in sources)
    problems: List[str] = []
    for time in sorted(said.times, key=lambda t: (t.hour, t.minute)):
        if not _time_known(time, known):
            problems.append(f"the time {_show(time)}")
    for unit, value in sorted(said.amounts):
        if (unit, value) not in known.amounts:
            problems.append(f"the amount {value} {unit}")
    for value in sorted(said.numbers - known.numbers):
        problems.append(f"the number {value}")
    for word in sorted(said.date_words - known.words):
        problems.append(f"the date word {word!r}")
    for name in _names(_TIME.sub(" ", line)):
        if _bare(name) not in known.words:
            problems.append(f"the name {name!r}")
    return problems


def _times(text: str) -> Tuple[FrozenSet[Time], str]:
    found = set()
    for m in _TIME.finditer(text):
        if m[1]:
            found.add(Time(int(m[1]), int(m[2]), (m[3] or "").lower() or None))
        elif m[4]:
            found.add(Time(int(m[4]), 0, m[5].lower()))
        else:
            found.add(Time(12, 0, "p" if m[6].lower() == "noon" else "a"))
    return frozenset(found), _TIME.sub(" ", text)


def _time_known(time: Time, known: Facts) -> bool:
    if any(t.hour == time.hour and t.minute == time.minute and (time.meridiem is None or t.meridiem == time.meridiem)
           for t in known.times):
        return True
    # "8:00" or "8:30" with no a.m./p.m. restates a bare "eight" or "eight thirty".
    return time.meridiem is None and Decimal(time.hour) in known.numbers and (not time.minute or Decimal(time.minute) in known.numbers)


def _amounts(text: str):
    for m in _AMOUNT.finditer(text):
        if m[1]:
            yield _CURRENCY_SYMBOLS[m[1]], _decimal(m[2])
        else:
            yield _CURRENCY_WORDS[m[4].lower()], _decimal(m[3])
    for m in re.finditer(rf"\b((?:{_NUMBER_WORD})(?:[\s-]+(?:{_NUMBER_WORD}))*)\s+(percent|dollars?|euros?|pounds?|cents?)\b", text, re.IGNORECASE):
        for value in _phrase_values(m[1]):
            yield _CURRENCY_WORDS[m[2].lower()], value


def _decimal(token: str) -> Decimal:
    return Decimal(re.sub(r"(?:st|nd|rd|th)$", "", token.replace(",", ""), flags=re.IGNORECASE))


def _phrase_values(phrase: str) -> List[Decimal]:
    """ "twenty-five" -> [25]; "two hundred and five" -> [205]; "three four" -> [3, 4]. """
    values: List[Decimal] = []
    total, current = 0, None
    for word in re.split(r"[\s-]+", phrase.lower()):
        if word == "and":
            continue
        if word in _SCALES:
            current = (current or 1) * _SCALES[word]
            if _SCALES[word] >= 1000:
                total, current = total + current, None
            continue
        value = _WORD_VALUES[word]
        if current is not None and not _joins(current, value):
            values.append(Decimal(total + current))
            total, current = 0, None
        current = (current or 0) + value
    if current is not None or total:
        values.append(Decimal(total + (current or 0)))
    return values


def _joins(current: int, value: int) -> bool:
    """Whether `value` continues the number spoken so far: "twenty" + "five", "two hundred" + "ten"."""
    last = current % 100
    if last == 0:
        return current > 0 and value < 100
    return last >= 20 and last % 10 == 0 and value < 10


def _merge(all_facts) -> Facts:
    merged = [frozenset()] * 5
    for f in all_facts:
        merged = [a | b for a, b in zip(merged, (f.numbers, f.times, f.amounts, f.date_words, f.words))]
    return Facts(*merged)


def _names(line: str) -> List[str]:
    names = []
    initial = True
    for token in line.split():
        word = _WORD.search(token)
        if word is not None:
            name = word[0]
            bare = _bare(name)
            if name[0].isupper() and bare not in _SELF and not (initial and _starter(bare)):
                names.append(name)
        initial = bool(_SENTENCE_END.search(token)) or (initial and word is None)
    return names


_SELF = frozenset({"i", "i'm", "i'll", "i've", "i'd"})


def _starter(word: str) -> bool:
    """Whether a capitalized first word is plain English: a common word, or a verb in -ing like "Checking"."""
    return word in COMMON or (len(word) >= 6 and word.endswith("ing"))


def _bare(word: str) -> str:
    return re.sub(r"['’]s$", "", word.casefold().replace("’", "'")).strip("'-")


def _show(time: Time) -> str:
    return f"{time.hour}:{time.minute:02d}{' ' + time.meridiem + '.m.' if time.meridiem else ''}"
