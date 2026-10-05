import asyncio
import json
import sys
import time
from pathlib import Path

import pytest
from websockets.asyncio.client import connect

from musegadget.reachy_companion_protocol import Hello, Narrate, NarrateOp, NarrateResult, Op, Sender, Welcome, decode, encode
from reachy_companion.server import CompanionServer, Engines

pytestmark = pytest.mark.real

EVALS = Path(__file__).resolve().parents[1] / "evals" / "narrator"
sys.path.insert(0, str(EVALS))
from run import ask_for, score  # noqa: E402

CASES = {case["id"]: case for case in json.loads((EVALS / "cases.json").read_text())["cases"]}
CHOSEN = ("ack-reminder-sister-745", "progress-fares-after-flights", "adversarial-dentist-no-appointment")
DEADLINES_S = {NarrateOp.ACKNOWLEDGE: 0.9, NarrateOp.PROGRESS: 1.5, NarrateOp.LINES: 4.0}
TOKEN = "rc1_test_token_0123456789abcdef0123456789"


def test_the_real_narrator_answers_three_eval_cases_over_the_websocket():
    from reachy_companion.narrator_mlx import MlxNarrator

    async def run():
        narrator = MlxNarrator()
        await narrator.load()
        async with CompanionServer(Engines(narrator=narrator), TOKEN) as companion:
            async with connect(companion.url) as ws:
                await ws.send(encode(Hello((1,), TOKEN, (Op.NARRATE,), 24000, "test")))
                assert isinstance(decode(await ws.recv(), sender=Sender.COMPANION), Welcome)
                results = {}
                for id, name in enumerate(CHOSEN):
                    ask = ask_for(CASES[name])
                    started = time.perf_counter()
                    await ws.send(encode(Narrate(id, ask.op, ask.request, ask.status or None, ask.already_said, ask.reply or None)))
                    result = decode(await ws.recv(), sender=Sender.COMPANION)
                    assert isinstance(result, NarrateResult) and result.id == id, result
                    results[name] = (result.lines, time.perf_counter() - started)
                return results

    for name, (lines, seconds) in asyncio.run(run()).items():
        case = CASES[name]
        print(f"{name}: {[(line.text, line.expression) for line in lines]} in {seconds:.2f} s")
        verdict = score(case, lines, fell_back=False)
        assert lines and verdict["pass"], verdict
        assert seconds < DEADLINES_S[NarrateOp(case["op"])]
