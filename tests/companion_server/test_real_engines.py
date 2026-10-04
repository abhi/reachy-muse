import asyncio
import json
from pathlib import Path

import pytest

from bench.latency import hear, missing, read_wav, robot, speak, speech_end_s
from reachy_companion.engines import EchoNarrator
from reachy_companion.server import CompanionServer, Engines

pytestmark = pytest.mark.real

DATA = Path(__file__).parent / "data"
CASES = json.loads((DATA / "manifest.json").read_text())["cases"]
EXTRA = {"reminder": ["at eight"]}
TOKEN = "rc1_test_token_0123456789abcdef0123456789"


def test_real_engines_hear_the_critical_words_and_speak_over_the_websocket():
    from reachy_companion.hearing_voxtral import VoxtralHearing
    from reachy_companion.voice_kokoro import KokoroVoice

    async def run():
        hearing, voice = VoxtralHearing(), KokoroVoice()
        await asyncio.gather(hearing.load(), voice.load())
        async with CompanionServer(Engines(hearing=hearing, voice=voice, narrator=EchoNarrator()), TOKEN) as companion:
            ws = await robot(companion.url, TOKEN, out_rate=16000)
            heard = {}
            for stream, case in zip(range(1, 99, 2), CASES):
                pcm = read_wav(DATA / case["file"])
                heard[case["name"]] = (await hear(ws, stream, pcm, finish=True), speech_end_s(pcm))
            spoken = await speak(ws, 2, "Your reminder is set for eight tonight. I will call Maya at nine.")
            await ws.close()
            return heard, spoken

    heard, spoken = asyncio.run(run())
    for case in CASES:
        result, speech_end = heard[case["name"]]
        print(f"{case['name']}: {result.text!r} forced={result.forced} endpoint {result.endpoint_s - speech_end:.3f} s after speech")
        assert missing(result.text, case["critical"] + EXTRA.get(case["name"], [])) == [], result.text
    assert {name: result.forced for name, (result, _) in heard.items()} == {"reminder": False, "correction": False, "negation": False}
    print(f"speak: {spoken.frames} frames, first after {spoken.first_frame_s:.3f} s")
    assert spoken.frames > 0
    assert spoken.done_frames == spoken.frames
