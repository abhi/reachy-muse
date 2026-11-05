"""The 06:11 replay must keep reproducing a turn whose companion is alive but backlogged."""

import asyncio

import pytest

pytest.importorskip("av")


def test_the_incident_replay_ends_the_turn_and_speaks_before_muse_answers():
    from bench.turn_smoothness import replay

    result = asyncio.run(replay(muse_answer_s=3.0))
    assert (result["endpoint"], result["first_motion"][1], result["acknowledged"]) == (2.0, "nod", True)
    assert result["expressions"] == ["happy", "sad", "curious"]
    assert result["requests"] == ["can you find me a vegetarian dinner spot near the office"]
    assert result["first_reachy_speech"][0] < result["first_audio"]
