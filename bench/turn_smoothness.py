"""Replay the November 5 06:11 turn through the production conversation, with fakes and its timings.

A companion over loopback WebSocket keeps answering pings while its hearing lags: it sends no
partials, and its endpoint arrives 37.28 s after the user stops. While that backlog lasts,
its narration takes 3 s and its speech 3.6 s to start. Muse acknowledges 1.66 s after
dispatch, reports "working" 0.3 s later, and answers 93.1 s after dispatch. The narrator
labels every line neutral. Reachy's own recognizer takes 2.232 s, as on November 4 at 06:43.
No model, device, service, or Muse is used.

Run with another checkout's src first on PYTHONPATH to replay that revision:

    PYTHONPATH=src python -m bench.turn_smoothness --output result.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import fake_companion  # noqa: E402
from fake_companion import FakeCompanion  # noqa: E402
from reachy_muse.contract.expressions import Expression  # noqa: E402
from reachy_muse.protocol import HearEndpoint, NarrateOp, encode  # noqa: E402
from test_companion import Conversation  # noqa: E402

REQUEST = "can you find me a vegetarian dinner spot near the office"
REPLY = ("Green Table is the closest vegetarian spot, with great reviews. Unfortunately it closes at eight "
         "tonight. Would you like me to book a table?")
SPOKEN_S = 5.8                 # the WAV Muse accepted
SILENCE_S = 2.0                # the robot's companion-mode silence endpoint
COMPANION_ENDPOINT_S = 37.28   # speech end to the companion's own endpoint
BUSY_NARRATION_S = 3.0         # past the 0.9 s acknowledgement deadline
BUSY_SPEECH_S = 3.6            # 06:12:30.866 selected, 06:12:34.486 audible
ROBOT_ASR_S = 2.232            # November 4 06:43:51, Whisper on the robot after a companion failure
MUSE_ACK_S = 1.66
MUSE_STATUS_S = .296
MUSE_ANSWER_S = 93.1           # dispatch to first reply text


class RobotRecognizer:
    async def transcribe(self, wav):
        await asyncio.sleep(ROBOT_ASR_S)
        return REQUEST


class Clock(logging.Handler):
    """Conversation log lines, stamped on the event loop's clock."""

    def __init__(self, loop):
        super().__init__(logging.INFO)
        self.loop = loop
        self.lines = []

    def emit(self, record):
        self.lines.append((self.loop.time(), record.getMessage()))


def busy_companion(busy_until):
    """Speech that starts late while the companion's hearing backlog lasts."""
    session = fake_companion._Session
    speak = session._speak

    async def slow_speak(self, stream):
        await asyncio.sleep(BUSY_SPEECH_S if asyncio.get_running_loop().time() < busy_until[0] else .2)
        await speak(self, stream)

    session._speak = slow_speak
    return lambda: setattr(session, "_speak", speak)


async def replay(muse_answer_s: float = MUSE_ANSWER_S) -> dict:
    loop = asyncio.get_running_loop()
    busy_until = [float("inf")]
    restore = busy_companion(busy_until)
    clock = Clock(loop)
    for name in ("reachy_muse.conversation", "reachy_muse.turn_timeline", "reachy_muse.failover"):
        logging.getLogger(name).addHandler(clock)
        logging.getLogger(name).setLevel(logging.INFO)
    marks = {}

    async def muse(session, request):
        await asyncio.sleep(MUSE_STATUS_S)
        await session.emit("agent.status", request=request, parent=None, activity_code="working")
        await asyncio.sleep(muse_answer_s - MUSE_ACK_S - MUSE_STATUS_S)
        await session.answer(request, REPLY)

    try:
        async with FakeCompanion(partials=(REQUEST,), frames_per_partial=10 ** 6, no_endpoint=True,
                                 narrate_delay_s={NarrateOp.ACKNOWLEDGE: BUSY_NARRATION_S},
                                 expression=Expression.NEUTRAL) as companion:
            robot = Conversation(None, companion.url[:-3], muse, transcriber=RobotRecognizer(), silence_s=SILENCE_S)
            send_chat = robot.session.send_chat

            async def acknowledged_late(*args, **options):
                await asyncio.sleep(MUSE_ACK_S)
                return await send_chat(*args, **options)
            robot.session.send_chat = acknowledged_late
            stamp = robot.recording.add
            robot.recording.add = lambda event: stamp({**event, "t": loop.time()})

            async def user():
                await robot.until(lambda: robot.link.up)
                await asyncio.sleep(SILENCE_S + 1)
                await robot.speak(SPOKEN_S)
                marks["speech_end"] = loop.time()
                busy_until[0] = marks["speech_end"] + COMPANION_ENDPOINT_S

                async def late_endpoint():
                    await asyncio.sleep(COMPANION_ENDPOINT_S)
                    for ws in companion._server.connections:
                        await ws.send(encode(HearEndpoint(1, REQUEST, False)))
                endpoint = asyncio.create_task(late_endpoint())
                await robot.until(lambda: robot.session.scripts and robot.session.scripts[-1].done(),
                                  COMPANION_ENDPOINT_S + muse_answer_s + 30)
                await robot.quiet(span=1.0)
                endpoint.cancel()
            await robot.run(user)
    finally:
        restore()
        for name in ("reachy_muse.conversation", "reachy_muse.turn_timeline", "reachy_muse.failover"):
            logging.getLogger(name).removeHandler(clock)
    return summarize(marks["speech_end"], robot.recording.events, clock.lines)


def summarize(speech_end: float, events: list, lines: list) -> dict:
    def after(t):
        return round(t - speech_end, 3)

    timeline = next(text for _, text in lines if text.startswith("Reachy turn timeline "))
    stages = dict(re.findall(r"(\w+)=([0-9.]+)", timeline))
    origin = float(stages["speech_end"])
    result = {stage: round(float(value) - origin, 3) for stage, value in stages.items()
              if stage != "speech_end" and re.fullmatch(r"[0-9.]+", value)}
    moves = [(event["t"], event.get("name") or event["state"]) for event in events
             if event.get("robot") in ("gesture", "set_state") and event["t"] > speech_end]
    result["first_motion"] = [after(moves[0][0]), moves[0][1]] if moves else None
    result["thinking_pose"] = next((after(t) for t, move in moves if move == "thinking"), None)
    spoken = [(t, text) for t, text in lines if re.match(r"Reachy (acknowledgement|progress) started", text)]
    result["first_reachy_speech"] = [after(spoken[0][0]), spoken[0][1].split()[1]] if spoken else None
    result["acknowledged"] = any(text.startswith("Reachy acknowledgement started") for _, text in lines)
    result["waiting_cue"] = next((after(t) for t, text in lines if text.startswith("Reachy progress started")), None)
    result["expressions"] = [event["expression"] for event in events if event.get("state") == "speaking"]
    result["requests"] = [event["text"] for event in events if event.get("muse") == "send_chat"]
    result["log"] = [[after(t), text] for t, text in lines if t >= speech_end and not text.startswith("Reachy turn")]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--muse-answer-s", type=float, default=MUSE_ANSWER_S)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(replay(args.muse_answer_s))
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
