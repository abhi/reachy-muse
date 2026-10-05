"""Narration for C4: a small instruction model on MLX writes the spoken lines.

The model answers in JSON. Every answer is parsed and checked here before it leaves: the shape,
the line count and length, no markup, an expression from the SDK's vocabulary, no claimed result
in an acknowledgement, no repeated progress line, and no fact that the request, status or reply
does not state. A rejected answer gets one retry that names the problems, if the robot's deadline
leaves time for it. Then the narrator falls back: nothing for acknowledge and progress, so the robot
says its own line, and the reply's own first sentences for lines.

One model serves every narration. Its own thread runs each call in turn, apart from the hearing
and voice threads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from musegadget.reachy_capabilities import Expression
from musegadget.reachy_companion_protocol import MAX_LINE_BYTES, NarratedLine, NarrateOp

from reachy_companion.grounding import ROBOT_REJECTS, ungrounded

log = logging.getLogger("reachy_companion")

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


@dataclass(frozen=True)
class _Op:
    task: str
    max_lines: int
    max_tokens: int
    budget_s: float  # a little under the robot's deadline for this op


OPS: Dict[NarrateOp, _Op] = {
    NarrateOp.ACKNOWLEDGE: _Op(
        "The user just finished speaking. Write exactly one line, at most 12 words, that restates what they asked "
        "in the request's own key words, so they know you heard it right. "
        'For example, "set a timer for ten minutes" becomes "Sure, a timer for ten minutes." and '
        '"what\'s the weather in Oslo" becomes "Okay, checking the weather in Oslo." '
        "Do not answer the request, and do not say that anything is done, set, found, added or booked: the work has not started. "
        'If the request is a greeting, a thank-you or small talk rather than a task, such as "how are you" or "good morning", '
        'answer {"lines": []} so nothing is said before the reply. '
        "Use the expression nod or listening.",
        max_lines=1, max_tokens=48, budget_s=0.8),
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

    def sources(self) -> Tuple[str, ...]:
        """The texts a line may take facts from: the same ones the robot checks it against."""
        if self.op is NarrateOp.ACKNOWLEDGE:
            return (self.request,)
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


def messages(ask: Ask) -> List[Dict[str, str]]:
    op = OPS[ask.op]
    if ask.op is NarrateOp.ACKNOWLEDGE:
        given = f"Request: {ask.request}"
    elif ask.op is NarrateOp.PROGRESS:
        said = "\n".join(f"- {line}" for line in ask.already_said) or "(nothing yet)"
        given = f"Request: {ask.request}\nStatus: {ask.status}\nAlready said:\n{said}"
    else:
        given = f"Request: {ask.request}\nReply:\n{ask.reply}"
    return [{"role": "system", "content": f"{_RULES}\n\n{op.task}"}, {"role": "user", "content": given}]


def parse(ask: Ask, raw: str) -> Tuple[Tuple[NarratedLine, ...], Tuple[str, ...]]:
    """The lines in a model answer, and every reason to reject them. No reasons means they may be spoken."""
    try:
        value, _ = json.JSONDecoder().raw_decode(raw.strip())
    except json.JSONDecodeError as error:
        return (), (f"the answer is not valid JSON ({error.msg})",)
    items = value.get("lines") if isinstance(value, dict) else None
    if not isinstance(items, list) or not all(isinstance(i, dict) and isinstance(i.get("text"), str) for i in items):
        return (), ('the answer must be {"lines": [{"text": ..., "expression": ...}]}',)
    lines = tuple(NarratedLine(" ".join(i["text"].split()), Expression.parse(i.get("expression"))) for i in items)
    problems = [*_count_problems(ask, lines)]
    for item, line in zip(items, lines):
        problems += _line_problems(ask, line.text)
        if line.expression is None:
            problems.append(f"the expression {item.get('expression')!r} is not one of {', '.join(EXPRESSIONS)}")
    return lines, tuple(problems)


def _count_problems(ask: Ask, lines: Sequence[NarratedLine]) -> List[str]:
    most = OPS[ask.op].max_lines
    if len(lines) > most:
        return [f"there are {len(lines)} lines; at most {most} are allowed"]
    if ask.op is NarrateOp.LINES and not lines:
        return ["there are no lines"]
    return []


MARKUP = re.compile(r"[*#`_~|<>\[\]{}]|https?://|www\.|^\s*(?:[-•]|\d+[.)])\s")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿]")
_CLAIM = re.compile(r"\b(?:done|all set|i've|i have|has been|have been|here's|here is|found|it's set|is set)\b", re.IGNORECASE)


def _line_problems(ask: Ask, text: str) -> List[str]:
    problems = []
    quoted = repr(text[:60])
    if not text:
        problems.append("a line is empty")
    if len(text.encode("utf-8")) > MAX_LINE_BYTES:
        problems.append(f"the line {quoted} is longer than {MAX_LINE_BYTES} bytes")
    if MARKUP.search(text) or _EMOJI.search(text):
        problems.append(f"the line {quoted} has markup, a link or an emoji")
    if ask.op is NarrateOp.ACKNOWLEDGE and _CLAIM.search(text):
        problems.append(f"the line {quoted} claims a result before the work has started")
    if ask.op is NarrateOp.PROGRESS and _plain(text) in {_plain(s) for s in ask.already_said}:
        problems.append(f"the line {quoted} was already said")
    for fact in ungrounded(text, *ask.sources()):
        if fact == ROBOT_REJECTS:
            problems.append(f"the robot's own check rejects the line {quoted}: start it with a plain word and copy every figure as written")
        else:
            problems.append(f"the line {quoted} says {fact}, which the input does not")
    return problems


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())


_CHAT = re.compile(
    r"(?:\s*(?:hi|hello|hey|hiya|yo|reachy|there|good (?:morning|afternoon|evening|night)|morning|"
    r"thanks?(?: you)?(?: (?:so|very) much)?|thank you|cheers|bye|goodbye|see you(?: later)?|good job|well done|"
    r"how are you(?: doing)?(?: today)?|how(?:'s| is) it going|how are things|what'?s up|what is up|sup|"
    r"how(?:'s| is| was) your day(?: been| going)?|sleep well|you(?:'re| are) (?:funny|awesome|great|the best)|"
    r"that (?:was|is|'s) (?:really |very |so )?(?:helpful|great|useful|perfect|nice)|"
    r"nice to meet you|okay|ok|cool|great|nice|awesome|got it)\b)+")


def small_talk(request: str) -> bool:
    """A greeting, thanks or other chat, or a request too short to restate. It gets no acknowledgement.

    Restating "how are you" gave lines like "Let me show how are you", which is worse than saying nothing.
    """
    words = re.sub(r"[^\w\s']", " ", request.casefold()).split()
    return len(words) < 3 or _CHAT.fullmatch(" ".join(words)) is not None


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
        self._thread = ThreadPoolExecutor(1, thread_name_prefix="narrator")
        self._model: Any = None
        self._tokenizer: Any = None

    async def load(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._thread, self.load_now)

    def load_now(self) -> None:
        from mlx_lm import load

        started = time.perf_counter()
        self._model, self._tokenizer = load(self.repo)
        self.narrate(Ask(NarrateOp.LINES, "is it raining", reply="It is not raining in Oslo right now."))
        log.info("narrator %s ready in %.1f s", self.repo, time.perf_counter() - started)

    async def acknowledge(self, request: str) -> Sequence[NarratedLine]:
        return await self._run(Ask(NarrateOp.ACKNOWLEDGE, request))

    async def progress(self, request: str, status: str, already_said: Sequence[str]) -> Sequence[NarratedLine]:
        return await self._run(Ask(NarrateOp.PROGRESS, request, status=status, already_said=tuple(already_said)))

    async def lines(self, request: str, reply: str) -> Sequence[NarratedLine]:
        return await self._run(Ask(NarrateOp.LINES, request, reply=reply))

    async def _run(self, ask: Ask) -> Tuple[NarratedLine, ...]:
        narration = await asyncio.get_running_loop().run_in_executor(self._thread, self.narrate, ask)
        return narration.lines

    def narrate(self, ask: Ask) -> Narration:
        """Ask the model, check its answer, retry once if there is time, else fall back. Runs on the narrator thread."""
        if ask.op is NarrateOp.ACKNOWLEDGE and small_talk(ask.request):
            return Narration((), (), fell_back=False)
        op = OPS[ask.op]
        started = time.perf_counter()
        conversation = messages(ask)
        attempts: List[Attempt] = []
        while True:
            began = time.perf_counter()
            raw = self._generate(conversation, op.max_tokens)
            lines, problems = parse(ask, raw)
            attempts.append(Attempt(raw, problems, time.perf_counter() - began))
            if not problems:
                return Narration(lines, tuple(attempts), fell_back=False)
            spent = time.perf_counter() - started
            if len(attempts) == 2 or spent + attempts[-1].seconds > op.budget_s:
                log.warning("narrator fell back on %s: %s", ask.op.value, "; ".join(problems))
                return Narration(fallback(ask), tuple(attempts), fell_back=True)
            conversation = conversation + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "That answer was rejected: " + "; ".join(problems) + ". Write it again and follow every rule."},
            ]

    def _generate(self, conversation: List[Dict[str, str]], max_tokens: int) -> str:
        from mlx_lm import stream_generate

        prompt = self._tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False) + PREFIX
        tokens = self._tokenizer.encode(prompt, add_special_tokens=False)
        text = PREFIX
        for response in stream_generate(self._model, self._tokenizer, tokens, max_tokens=max_tokens):
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
