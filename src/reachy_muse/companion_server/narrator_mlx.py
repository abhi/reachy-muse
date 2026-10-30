"""Narration for C4: a small instruction model on MLX writes the spoken lines.

The model answers in JSON. Every answer is parsed and checked here before it leaves: the shape,
the line count and length, no markup, an expression from the SDK's vocabulary, no claimed result
in an acknowledgement, no repeated progress line, and no fact that the request, status or reply
does not state. A rejected answer gets one retry that names the problems, if the robot's deadline
leaves time for it. Then the narrator falls back: nothing for acknowledge and progress, so the robot
says its own line, and the reply's own first sentences for lines.

One model serves every narration and every device command classification. Its own thread runs
each call in turn, apart from the hearing and voice threads.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from reachy_muse.companion_server.watchdog import EngineExecutor, watchdog
from reachy_muse.contract.expressions import Expression
from reachy_muse.contract.intent import FALLBACK_OPENERS, Intent, guess_intent, maybe_chat, pick
from reachy_muse.contract.persona import REACHY_PERSONA
from reachy_muse.companion_server.engines import Narrated
from reachy_muse.capabilities import EMOJI, MARKUP
from reachy_muse.device_command import DeviceCommand
from reachy_muse.protocol import MAX_LINE_BYTES, NarratedLine, NarrateOp

from reachy_muse.companion_server.grounding import ROBOT_REJECTS, ungrounded

log = logging.getLogger("reachy_muse.companion_server")

DEFAULT_REPO = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
PREFIX = '{"lines": ['
EXPRESSIONS = tuple(e.value for e in Expression)

_RULES = f"""You write what a small home robot says out loud. Answer with JSON only, in this shape:
{{"lines": [{{"text": "<one spoken sentence>", "expression": "<expression>"}}]}}
The expression is one of: {", ".join(EXPRESSIONS)}.
Rules for every text:
- One plain spoken sentence of at most 25 words. No markdown, lists, emoji, links or quotation marks.
- Never add a fact. Every number, time, date, amount and name you say must be in the input.
- Copy figures exactly as the input writes them. Keep "7:45 pm" as "7:45 pm" and "$25.50" as "$25.50"; never turn words into digits or digits into words.
- Speak to the user: call them "you" and their things "your", never "me" or "my".
- Start each sentence with a plain word such as "Sure", "Okay", "Your", "It", "The" or "There", or with a name from the input."""


CLASSIFY_RULES = """You sort what a user said to Reachy, a small tabletop robot, into one label.
A label other than NONE only when the words are a direct order to the robot itself:
SLEEP: go to sleep, power down for the night.
WAKE_UP: wake up.
STOP_LISTENING: stop listening, end the conversation, the user is done talking to the robot.
STOP_TALKING: stop speaking right now, be quiet, hush.
VOLUME_UP: speak louder, the user cannot hear the robot.
VOLUME_DOWN: speak more softly.
FACE_TRACKING_ON: look at or follow the user's face.
FACE_TRACKING_OFF: stop looking at or following the user.
NONE: everything else. Every question, every request to cancel, stop or change something other than the robot, every request about the world, music, other devices, apps, timers, people, health or sleep habits is NONE. When unsure, answer NONE.
The robot itself is the only thing these labels control. Orders about anything else are NONE, for example:
"pause the song" NONE
"stop the alarm" NONE
"cancel my order" NONE
"turn down the TV" NONE
"wake me up at six" NONE
"how do I fall asleep faster" NONE
Answer with the label only."""
CLASSIFY_MAX_TOKENS = 8
_SENTENCE = "\x00"


@dataclass(frozen=True)
class _Op:
    task: str
    max_lines: int
    max_tokens: int
    budget_s: float  # a little under the robot's deadline for this op


OPS: Dict[NarrateOp, _Op] = {
    NarrateOp.ACKNOWLEDGE: _Op("Choose the intent and write Reachy's opening.",
                               max_lines=2, max_tokens=90, budget_s=0.8),
    NarrateOp.PROGRESS: _Op(
        "The user is waiting while their request is worked on. Write exactly one line, at most 12 words, "
        "that tells them what is happening now, based only on the status. "
        'For example, "Looking at flights now." becomes "Still looking at flights for you." '
        'and "Comparing the options." becomes "Okay, comparing the options now." '
        "Do not guess results, websites, names, prices or counts. "
        "Do not repeat a line already said; say something new or say it differently. Use the expression thinking.",
        max_lines=1, max_tokens=48, budget_s=1.35),
    NarrateOp.LINES: _Op(
        "Turn the reply into one to three short sentences for the robot to say, leading with the answer to the request. "
        "Never write more than three, even for a long reply: keep the facts that answer the request and drop the rest. "
        'Keep the facts that matter word for word: numbers, times, amounts, names, and every "not", "no" or "couldn\'t". '
        "If the reply says it could not find or verify something, say so. If the reply does not answer the request, "
        "say what the reply does say and never fill the gap. Leave out markdown, lists, links and filler. "
        "Give each sentence the expression that fits it: happy for good news, sad for bad news, shake for a no, "
        "nod to confirm, surprised for something unexpected, neutral otherwise.",
        max_lines=3, max_tokens=220, budget_s=3.6),
}


@dataclass(frozen=True)
class Ask:
    """One narration request, as the robot sent it."""

    op: NarrateOp
    request: str
    status: str = ""
    already_said: Tuple[str, ...] = ()
    reply: str = ""
    previous: str = ""
    may_chat: bool = False

    @property
    def chat_allowed(self) -> bool:
        return self.may_chat and maybe_chat(self.request)

    def sources(self) -> Tuple[str, ...]:
        """The texts a line may take facts from: the same ones the robot checks it against."""
        if self.op is NarrateOp.ACKNOWLEDGE:
            return (self.request, self.previous)
        if self.op is NarrateOp.PROGRESS:
            return (self.status, self.request)
        return (self.request, self.reply)


@dataclass(frozen=True)
class Attempt:
    raw: str
    problems: Tuple[str, ...]
    seconds: float


@dataclass(frozen=True)
class Narration:
    lines: Tuple[NarratedLine, ...]
    attempts: Tuple[Attempt, ...]
    fell_back: bool
    intent: Optional[Intent] = None


@dataclass(frozen=True)
class _Opening:
    when: str
    say: str
    max_words: int
    max_lines: int


OPENINGS = {
    Intent.CHAT: _Opening(
        "social talk only: greetings, thanks, feelings, banter, comments about Reachy. Nothing to look up or do.",
        'when "Reachy may answer: yes", reply in one or two short sentences, warm and natural, maybe asking '
        'something back. No facts, numbers, names, dates, advice or promises. When "no", return no lines.', 20, 2),
    Intent.QUESTION: _Opening(
        "the user wants to know something: facts, advice, ideas, recommendations, their own data, time, weather "
        "or news. A request that makes sense on its own is a question, even right after another one.",
        "one short reaction that shows interest in the topic without answering it. Never give an answer, a fact, "
        "a suggestion, a yes or a no.", 8, 1),
    Intent.ACTION: _Opening(
        "the user wants something done: a message, reminder, timer, calendar entry, order, booking, list item, "
        "note, or something played.",
        "one short line saying you'll get it done. Never say it is done, set, sent, booked, added or saved.", 8, 1),
    Intent.FOLLOW_UP: _Opening(
        "the request only makes sense with the previous exchange: it points back with words like that, it, again, "
        "instead, which one or the day after, or asks what Reachy just said.",
        "one short connective that picks the thread back up without answering.", 8, 1),
}
_EXAMPLES = (
    ("why do cats purr", "no", "question", "Ooh, I love a cat mystery."),
    ("what's a good way to learn Spanish", "no", "question", "Nice, let me dig into that."),
    ("is coffee bad for you", "no", "question", "Hmm, fair thing to wonder."),
    ("text my sister that I'm on my way", "no", "action", "You got it, give me a sec."),
    ("add bread to my list", "no", "action", "Sure thing, I'll take care of it."),
    ("play some jazz", "no", "action", "Ooh, good pick, getting it going."),
    ("and which one was faster", "no", "follow_up", "Right, that part, hang on."),
    ("say that again", "no", "follow_up", "Of course, here it comes."),
    ("I had the best pizza today", "yes", "chat", "Oh yum, lucky you! What kind?"),
    ("you're hilarious", "yes", "chat", "Ha, I try my best!"),
    ("hey Reachy", "no", "chat", ""),
)
_LOOK = {"question": "curious", "action": "nod", "follow_up": "listening", "chat": "happy"}
OPENING_PREFIX = '{"intent": "'
OPENING_RULES = (
    "You speak as Reachy. " + REACHY_PERSONA + "\n"
    "The user just said something to Reachy, and Muse, Reachy's assistant, will handle it. Answer with JSON only:\n"
    '{"intent": "<intent>", "lines": [{"text": "<what Reachy says now>", "expression": "<expression>"}]}\n'
    f"The expression is one of: {', '.join(EXPRESSIONS)}.\n"
    "First choose the intent:\n"
    + "".join(f"- {intent.value}: {opening.when}\n" for intent, opening in OPENINGS.items())
    + "Then write what Reachy says right away, while Muse works:\n"
    + "".join(f"- {intent.value}: {opening.say} At most {opening.max_words} words.\n" for intent, opening in OPENINGS.items())
    + "Every line:\n"
    "- React to what the user said; never repeat or paraphrase their words.\n"
    '- Sound different from every line under "Reachy said lately": a different first word and different wording.\n'
    "- Every number, time, date and name you say must be in the request.\n"
    '- Plain spoken words: no markdown, lists, emoji or quotation marks. Call the user "you".\n'
    "Examples; vary the wording and never copy them:\n"
    + "".join(f'Request: {request} / Reachy may answer: {may} -> {{"intent": "{intent}", "lines": '
              + (f'[{{"text": "{line}", "expression": "{_LOOK[intent]}"}}]}}' if line else "[]}") + "\n"
              for request, may, intent, line in _EXAMPLES)
).rstrip("\n")


def messages(ask: Ask) -> List[Dict[str, str]]:
    op = OPS[ask.op]
    if ask.op is NarrateOp.ACKNOWLEDGE:
        said = "\n".join(f"- {line}" for line in ask.already_said) or "(nothing yet)"
        allowed = ask.chat_allowed
        given = (f"Request: {ask.request}\nPrevious request: {ask.previous}\nReachy said lately:\n{said}"
                 f"\nReachy may answer: {'yes' if allowed else 'no'}")
    elif ask.op is NarrateOp.PROGRESS:
        said = "\n".join(f"- {line}" for line in ask.already_said) or "(nothing yet)"
        given = f"Request: {ask.request}\nStatus: {ask.status}\nAlready said:\n{said}"
    else:
        given = f"Request: {ask.request}\nReply:\n{ask.reply}"
    rules = OPENING_RULES if ask.op is NarrateOp.ACKNOWLEDGE else f"{_RULES}\n\n{op.task}"
    return [{"role": "system", "content": rules}, {"role": "user", "content": given}]


def parse(ask: Ask, raw: str) -> Tuple[Narrated, Tuple[str, ...]]:
    """The lines in a model answer, and every reason to reject them. No reasons means they may be spoken."""
    try:
        value, _ = json.JSONDecoder().raw_decode(raw.strip())
    except json.JSONDecodeError as error:
        return Narrated(None, ()), (f"the answer is not valid JSON ({error.msg})",)
    items = value.get("lines") if isinstance(value, dict) else None
    if not isinstance(items, list) or not all(isinstance(i, dict) and isinstance(i.get("text"), str) for i in items):
        return Narrated(None, ()), ('the answer must be {"lines": [{"text": ..., "expression": ...}]}',)
    lines = tuple(NarratedLine(" ".join(i["text"].split()), Expression.parse(i.get("expression"))) for i in items)
    intent = Intent.parse(value.get("intent")) if ask.op is NarrateOp.ACKNOWLEDGE else None
    if intent is Intent.CHAT and not (ask.chat_allowed):
        items, lines = [], ()
    problems = [*_count_problems(ask, lines, intent)]
    if ask.op is NarrateOp.ACKNOWLEDGE:
        if intent is None:
            problems.append("the intent is missing or unknown")
        elif sum(len(line.text.split()) for line in lines) > OPENINGS[intent].max_words:
            problems.append(f"the opening exceeds {OPENINGS[intent].max_words} words")
        problems += [f"the line {line.text[:60]!r} is a fragment" for line in lines if len(line.text.split()) < 2]
        if intent is not Intent.CHAT:
            problems += [f"the line {line.text[:60]!r} answers or reports instead of only reacting"
                         for line in lines if _STATES_A_RESULT.search(line.text)]
    for item, line in zip(items, lines):
        problems += _line_problems(ask, line.text)
        if line.expression is None:
            problems.append(f"the expression {item.get('expression')!r} is not one of {', '.join(EXPRESSIONS)}")
    return Narrated(intent, lines), tuple(problems)


def _count_problems(ask: Ask, lines: Sequence[NarratedLine], intent: Optional[Intent]) -> List[str]:
    most = OPENINGS[intent].max_lines if intent is not None else OPS[ask.op].max_lines
    if len(lines) > most:
        return [f"there are {len(lines)} lines; at most {most} are allowed"]
    if ask.op is NarrateOp.LINES and not lines:
        return ["there are no lines"]
    return []


_STATES_A_RESULT = re.compile(
    r"\b(?:is|are|was|were|isn't|aren't|wasn't|weren't|has|have|had|can't|cannot|said|means|yes|no|not|set|sent|booked|"
    r"added|saved|ordered|scheduled|cancell?ed|updated|created|confirmed|completed|finished|started|ready|cheaper|safe|"
    r"how about|maybe)\b", re.IGNORECASE)
_CLAIM = re.compile(r"\b(?:done|all set|i've|i have|has been|have been|here's|here is|found|it's set|is set)\b", re.IGNORECASE)


def _line_problems(ask: Ask, text: str) -> List[str]:
    problems = []
    quoted = repr(text[:60])
    if not text:
        problems.append("a line is empty")
    if len(text.encode("utf-8")) > MAX_LINE_BYTES:
        problems.append(f"the line {quoted} is longer than {MAX_LINE_BYTES} bytes")
    if MARKUP.search(text) or EMOJI.search(text):
        problems.append(f"the line {quoted} has markup, a link or an emoji")
    if ask.op is NarrateOp.ACKNOWLEDGE and _CLAIM.search(text):
        problems.append(f"the line {quoted} claims a result before the work has started")
    if ask.op in (NarrateOp.PROGRESS, NarrateOp.ACKNOWLEDGE) and _plain(text) in {_plain(s) for s in ask.already_said}:
        problems.append(f"the line {quoted} was already said")
    if ask.op is NarrateOp.ACKNOWLEDGE:
        request_words = _plain(ask.request).split()
        line_words = _plain(text).split()
        runs = {tuple(request_words[i:i + 3]) for i in range(len(request_words) - 2)}
        if any(tuple(line_words[i:i + 3]) in runs for i in range(len(line_words) - 2)):
            problems.append(f"the line {quoted} restates the request")
    for fact in ungrounded(text, *ask.sources()):
        if fact == ROBOT_REJECTS:
            problems.append(f"the robot's own check rejects the line {quoted}: start it with a plain word and copy every figure as written")
        else:
            problems.append(f"the line {quoted} says {fact}, which the input does not")
    return problems


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())


def fallback(ask: Ask) -> Tuple[NarratedLine, ...]:
    """What the narrator says when the model's answer can't be used."""
    if ask.op is not NarrateOp.LINES:
        return ()
    text = re.sub(r"[*_`#|>\[\]{}]+", " ", ask.reply)
    text = re.sub(r"(?m)^\s*(?:[-•]|\d+[.)])\s+", "", text)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()][:3]
    return tuple(NarratedLine(_clip(" ".join(s.split())), None) for s in sentences)


def _clip(text: str) -> str:
    """At most MAX_LINE_BYTES, cut between words so no figure is cut in half."""
    if len(text.encode("utf-8")) <= MAX_LINE_BYTES:
        return text
    words = text.split()
    while len(" ".join(words).encode("utf-8")) > MAX_LINE_BYTES:
        words.pop()
    return " ".join(words)


class MlxNarrator:
    def __init__(self, repo: str = DEFAULT_REPO) -> None:
        self.repo = repo
        self.name = repo.split("/")[-1]
        self._thread = EngineExecutor("narrator")
        self._model: Any = None
        self._tokenizer: Any = None
        self._rules_cache: Any = None
        self._opening_cache: Any = None
        self._opening_tokens: Tuple[int, ...] = ()
        self._fallback_turns = dict.fromkeys(Intent, 0)
        self._turn_end = ""

    async def load(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._thread, self.load_now)

    def load_now(self) -> None:
        from mlx_lm import load

        started = time.perf_counter()
        self._model, self._tokenizer = load(self.repo)
        self.narrate(Ask(NarrateOp.LINES, "is it raining", reply="It is not raining in Oslo right now."))
        self._cache_classify_rules()
        self._cache_opening_rules()
        self.classify_now("go to sleep", tuple(DeviceCommand))
        log.info("narrator %s ready in %.1f s", self.repo, time.perf_counter() - started)

    async def acknowledge(self, request: str, *, previous: str = "", recently_said: Sequence[str] = (),
                          may_chat: bool = False) -> Narrated:
        ask = Ask(NarrateOp.ACKNOWLEDGE, request, already_said=tuple(recently_said), previous=previous,
                  may_chat=may_chat)
        narration = await asyncio.get_running_loop().run_in_executor(self._thread, self.narrate, ask)
        return Narrated(narration.intent, narration.lines)

    async def progress(self, request: str, status: str, already_said: Sequence[str]) -> Sequence[NarratedLine]:
        return await self._run(Ask(NarrateOp.PROGRESS, request, status=status, already_said=tuple(already_said)))

    async def lines(self, request: str, reply: str) -> Sequence[NarratedLine]:
        return await self._run(Ask(NarrateOp.LINES, request, reply=reply))

    async def classify(self, text: str, allowed: Sequence[DeviceCommand]) -> Optional[DeviceCommand]:
        return await asyncio.get_running_loop().run_in_executor(self._thread, self.classify_now, text, tuple(allowed))

    def classify_now(self, text: str, allowed: Tuple[DeviceCommand, ...]) -> Optional[DeviceCommand]:
        from mlx_lm import stream_generate

        labels = ", ".join([command.name for command in allowed] + ["NONE"])
        tokens = self._tokenizer.encode(f"Labels: {labels}\nSentence: {text}{self._turn_end}", add_special_tokens=False)
        raw = ""
        for response in stream_generate(self._model, self._tokenizer, tokens, max_tokens=CLASSIFY_MAX_TOKENS,
                                        prompt_cache=copy.deepcopy(self._rules_cache)):
            watchdog.progress()
            raw += response.text
            if "\n" in raw:
                break
        label = raw.strip()
        return next((command for command in allowed if command.name == label), None)

    def _cache_classify_rules(self) -> None:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        template = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": CLASSIFY_RULES}, {"role": "user", "content": _SENTENCE}],
            add_generation_prompt=True, tokenize=False)
        head, self._turn_end = template.split(_SENTENCE)
        self._rules_cache = make_prompt_cache(self._model)
        self._model(mx.array(self._tokenizer.encode(head, add_special_tokens=False))[None], cache=self._rules_cache)
        mx.eval([entry.state for entry in self._rules_cache])

    def _cache_opening_rules(self) -> None:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        template = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": OPENING_RULES}, {"role": "user", "content": _SENTENCE}],
            add_generation_prompt=True, tokenize=False)
        head, _ = template.split(_SENTENCE)
        self._opening_tokens = tuple(self._tokenizer.encode(head, add_special_tokens=False))
        self._opening_cache = make_prompt_cache(self._model)
        self._model(mx.array(self._opening_tokens)[None], cache=self._opening_cache)
        mx.eval([entry.state for entry in self._opening_cache])

    async def _run(self, ask: Ask) -> Tuple[NarratedLine, ...]:
        narration = await asyncio.get_running_loop().run_in_executor(self._thread, self.narrate, ask)
        return narration.lines

    def narrate(self, ask: Ask) -> Narration:
        """Ask the model, check its answer, retry once if there is time, else fall back. Runs on the narrator thread."""
        op = OPS[ask.op]
        started = time.perf_counter()
        conversation = messages(ask)
        attempts: List[Attempt] = []
        intent: Optional[Intent] = None
        while True:
            began = time.perf_counter()
            raw = self._generate(conversation, op.max_tokens, OPENING_PREFIX if ask.op is NarrateOp.ACKNOWLEDGE else PREFIX)
            result, problems = parse(ask, raw)
            intent = result.intent or intent
            attempts.append(Attempt(raw, problems, time.perf_counter() - began))
            if not problems:
                return Narration(result.lines, tuple(attempts), fell_back=False, intent=result.intent)
            spent = time.perf_counter() - started
            if len(attempts) == 2 or spent + attempts[-1].seconds > op.budget_s:
                log.warning("narrator fell back on %s: %d rejected attempts, %d problems",
                            ask.op.value, len(attempts), len(problems))
                if ask.op is NarrateOp.ACKNOWLEDGE:
                    return self._fallback_opening(ask, intent, tuple(attempts))
                return Narration(fallback(ask), tuple(attempts), fell_back=True)
            conversation = conversation + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "That answer was rejected: " + "; ".join(problems) + ". Write it again and follow every rule."},
            ]

    def _fallback_opening(self, ask: Ask, intent: Optional[Intent], attempts: Tuple[Attempt, ...]) -> Narration:
        intent = intent or guess_intent(ask.request, ask.previous)
        phrase = pick(FALLBACK_OPENERS[intent], self._fallback_turns[intent], ask.already_said)
        self._fallback_turns[intent] += 1
        return Narration(() if phrase is None else (NarratedLine(phrase, None),), attempts, fell_back=True, intent=intent)

    def _generate(self, conversation: List[Dict[str, str]], max_tokens: int, prefix: str) -> str:
        from mlx_lm import stream_generate

        prompt = self._tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False) + prefix
        tokens = self._tokenizer.encode(prompt, add_special_tokens=False)
        cache = None
        if prefix == OPENING_PREFIX and self._opening_cache is not None:
            head = self._opening_tokens
            if tuple(tokens[:len(head)]) == head:
                tokens = tokens[len(head):]
                cache = copy.deepcopy(self._opening_cache)
        text = prefix
        for response in stream_generate(self._model, self._tokenizer, tokens, max_tokens=max_tokens, prompt_cache=cache):
            watchdog.progress()
            text += response.text
            if "}" in response.text and _closed(text):
                break
        return text


def _closed(text: str) -> bool:
    try:
        json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return False
    return True
