import asyncio
import logging
from types import SimpleNamespace

import pytest

from reachy_muse.companion_server import hearing_voxtral as hearing_module
from reachy_muse.companion_server.engines import Partial


@pytest.mark.parametrize("decode_s, expected", [(.2, []), (.4, []), (.6, [
    "hearing call=feed total=1.200s queue=0.000s stages=voxtral_open:0.200s voxtral_feed:0.400s voxtral_step:0.600s",
])])
def test_hearing_calls_log_only_slow_stage_timings(monkeypatch, caplog, decode_s, expected):
    now, progress = [0.0], []
    monkeypatch.setattr(hearing_module, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    monkeypatch.setattr(hearing_module, "watchdog", SimpleNamespace(progress=lambda: progress.append(round(now[0], 3))))

    class Stream:
        pending = False

        def feed(self, samples):
            now[0] += .4
            self.pending = True

        def step(self, **kwargs):
            if not self.pending:
                return []
            self.pending = False
            now[0] += decode_s
            return ["private-hearing-words"]

    def create_stream(**kwargs):
        now[0] += .2
        return Stream()

    hearing = hearing_module.VoxtralHearing()
    hearing._models = SimpleNamespace(voxtral=SimpleNamespace(create_streaming_session=create_stream))
    caplog.set_level(logging.INFO, logger="reachy_muse.companion_server")

    async def scenario():
        session = hearing.open(0)
        assert await session.feed(b"\x11\x22" * 160) == [Partial("private-hearing-words")]
        await session.close()

    try:
        asyncio.run(scenario())
    finally:
        hearing._thread.shutdown()
    assert caplog.messages == expected
    assert "private-hearing-words" not in caplog.text
    assert progress[:2] == [.2, .6]
