import asyncio

from bench.turn2 import engine_replay


def test_slow_hearing_open_and_feed_leave_websocket_pings_responsive():
    result = asyncio.run(engine_replay())
    assert result["partial"] == "Should I work out today?"
    assert result["endpoint"] == "Should I work out today?"
    assert result["pongs_during_open"] >= 5
    assert result["pongs_during_feed"] >= 5
    assert result["max_ping_gap_ms"] < 150
