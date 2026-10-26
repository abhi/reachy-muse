from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from websockets.asyncio.client import connect

from reachy_muse.capabilities import grounded
from reachy_muse.contract.intent import guess_intent, maybe_chat

HERE = Path(__file__).resolve().parent
ACK_GRACE_S = 1.0
_STOP = frozenset("""a an the i i'm i'll you your it it's is are be to of in on for and or but so that this
me my we our at with what how do does can could would just let let's okay ok sure oh hmm well right now
that's there here up some any about""".split())


def words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.casefold().replace("’", "'"))


def parroted(opener: str, request: str) -> bool:
    said, asked = words(opener), words(request)
    trigrams = {tuple(asked[i:i + 3]) for i in range(len(asked) - 2)}
    if any(tuple(said[i:i + 3]) in trigrams for i in range(len(said) - 2)):
        return True
    content = [w for w in said if w not in _STOP]
    shared = [w for w in content if w in set(asked)]
    return len(shared) >= 3 and len(shared) / len(content) >= 0.6


def repeated(opener: str, before: Sequence[str]) -> bool:
    head = words(opener)[:3]
    return any(words(opener) == words(b) or (len(head) == 3 and words(b)[:3] == head) for b in before[-3:])


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def spread(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    return {"p50": round(statistics.median(values), 3), "p95": round(percentile(values, 0.95), 3), "n": len(values)}


def start_server(src: Path, port: int) -> tuple:
    state = Path(tempfile.mkdtemp(prefix="reachy-openers-"))
    env = {**os.environ, "PYTHONPATH": str(src), "PYTHONDONTWRITEBYTECODE": "1"}
    base = [sys.executable, "-m", "reachy_muse.companion_server", "--state-dir", str(state)]
    subprocess.run([*base, "pair", "--host", "127.0.0.1"], check=True, capture_output=True, env=env)
    server = subprocess.Popen([*base, "serve", "--insecure", "--port", str(port)], stderr=subprocess.PIPE, text=True, env=env)
    assert server.stderr is not None
    for line in server.stderr:
        sys.stderr.write(line)
        found = re.search(r"serving (ws://\S+)", line)
        if found:
            # Keep draining the log, or a full pipe stalls the server mid-run.
            threading.Thread(target=lambda: [sys.stderr.write(rest) for rest in server.stderr], daemon=True).start()
            return server, found.group(1), (state / "token").read_text()
    raise SystemExit(f"serve exited with {server.wait()}")


class Robot:
    def __init__(self, ws) -> None:
        self.ws = ws
        self.ids = 0

    async def narrate(self, request: str, already_said: Sequence[str], extra: Dict[str, Any], *,
                      previous: str = "", may_chat: bool = False) -> Dict[str, Any]:
        self.ids += 1
        frame = {"t": "narrate", "id": self.ids, "op": "acknowledge", "request": request, **extra}
        if already_said:
            frame["already_said"] = list(already_said[-4:])
        if previous:
            frame["previous"] = previous
        if may_chat:
            frame["may_chat"] = True
        await self.ws.send(json.dumps(frame))
        while True:
            message = json.loads(await self.ws.recv())
            if message.get("id") == self.ids:
                return message

    async def first_audio(self, text: str) -> float:
        self.ids += 1
        stream = self.ids * 2
        started = time.perf_counter()
        await self.ws.send(json.dumps({"t": "speak_start", "stream": stream, "text": text, "voice": "main"}))
        first = None
        while True:
            frame = await self.ws.recv()
            if isinstance(frame, bytes):
                first = first if first is not None else time.perf_counter() - started
                continue
            message = json.loads(frame)
            if message.get("t") == "speak_done" and message.get("stream") == stream:
                return first if first is not None else float("nan")


async def run(url: str, token: str, cases: List[Dict[str, Any]], rounds: int, extra: Dict[str, Any],
              local_chat: bool) -> List[Dict[str, Any]]:
    async with connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"t": "hello", "versions": [1], "token": token, "ops": ["speak", "narrate"],
                                  "out_rate": 24000, "robot": "bench"}))
        welcome = json.loads(await ws.recv())
        assert welcome.get("t") == "welcome", welcome
        robot = Robot(ws)
        await robot.narrate("is it going to rain", (), extra)
        results = []
        for round_ in range(rounds):
            said: List[str] = []
            previous = ""
            for case in cases:
                context = said + ([case["prev"]] if case.get("prev") else [])
                may_chat = local_chat and maybe_chat(case["text"])
                started = time.perf_counter()
                answer = await robot.narrate(case["text"], context, extra, previous=previous, may_chat=may_chat)
                narrate_s = time.perf_counter() - started
                if answer.get("t") == "error":
                    results.append({**case, "round": round_, "error": answer.get("message")})
                    continue
                lines = [line[0] for line in answer.get("lines", [])]
                opener = " ".join(lines)
                local = answer.get("intent") == "chat" and may_chat and bool(lines)
                audio_s = await robot.first_audio(lines[0]) if lines else None
                results.append({
                    "text": case["text"], "want_intent": case["intent"], "want_local": case["local"], "round": round_,
                    "opener": opener, "intent": answer.get("intent"), "reply": local, "gated": may_chat,
                    "grammar_intent": guess_intent(case["text"], previous).value,
                    "grounded": bool(opener) and all(grounded(line, case["text"], previous) for line in lines),
                    "parroted": bool(opener) and parroted(opener, case["text"]),
                    "repeated": bool(opener) and repeated(opener, said),
                    "same_start": bool(opener) and bool(said) and words(opener)[:1] == words(said[-1])[:1],
                    "narrate_s": round(narrate_s, 3), "first_audio_s": None if audio_s is None else round(audio_s, 3),
                })
                if opener:
                    said.extend(lines)
                previous = case["text"]
        return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    done = [r for r in results if "error" not in r]
    spoken = [r for r in done if r["opener"]]
    has_intent = [r for r in done if r["intent"]]
    local = [r for r in done if r["reply"]]
    queued = [r for r in spoken if not r["reply"]]
    rate = lambda part, whole: round(len(part) / len(whole), 3) if whole else None
    return {
        "utterances": len(done), "errors": len(results) - len(done),
        "opener_rate": rate(spoken, done),
        "intent_accuracy": rate([r for r in has_intent if r["intent"] == r["want_intent"]], done) if has_intent else None,
        "grammar_intent_accuracy": rate([r for r in done if r.get("grammar_intent") == r["want_intent"]], done),
        "gated": len([r for r in done if r.get("gated")]),
        "grounded_rate": rate([r for r in spoken if r["grounded"]], spoken),
        "parroting_rate": rate([r for r in spoken if r["parroted"]], spoken),
        "repeat_rate": rate([r for r in spoken if r["repeated"]], spoken),
        "same_first_word_rate": rate([r for r in spoken if r.get("same_start")], spoken),
        "local_chat": {"answered": len(local), "should": sum(r["want_local"] for r in done),
                       "false_positives": [r["text"] for r in local if not r["want_local"]],
                       "missed": len([r for r in done if r["want_local"] and not r["reply"]])},
        "narrate_s": spread([r["narrate_s"] for r in done]),
        "first_audio_s": spread([r["first_audio_s"] for r in spoken if r["first_audio_s"] is not None]),
        "opener_audio_after_dispatch_s": spread([max(ACK_GRACE_S, r["narrate_s"]) + r["first_audio_s"] for r in queued]),
        "chat_reply_audio_s": spread([r["narrate_s"] + r["first_audio_s"] for r in local]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--src", type=Path, default=HERE.parent / "src")
    parser.add_argument("--port", type=int, default=8798)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--extra", default="{}", help="JSON fields added to every narrate frame")
    parser.add_argument("--local-chat", choices=("on", "off"), default="on")
    args = parser.parse_args()
    if args.port in (8765, 8766):
        raise SystemExit("8765 and 8766 belong to the live companion")
    cases = json.loads((HERE / "contextual_openers.json").read_text())["cases"]
    server, url, token = start_server(args.src.resolve(), args.port)
    try:
        results = asyncio.run(run(url, token, cases, args.rounds, json.loads(args.extra), args.local_chat == "on"))
    finally:
        server.terminate()
        server.wait()
    out = HERE / "contextual_openers_runs" / f"{args.label}.json"
    out.parent.mkdir(exist_ok=True)
    summary = summarize(results)
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
