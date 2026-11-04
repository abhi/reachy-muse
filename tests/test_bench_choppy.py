"""The CPU incident replay must preserve recorded speech through real playback."""

import asyncio

from bench.choppy import replay


def test_recorded_fallback_pcm_survives_sentence_preparation_and_playback():
    result = asyncio.run(replay("cpu_fallback_pipeline", short=True))
    assert result["pcm_frames"] == 4800
    assert result["pcm_sha256"] == "17744bb6a4ed1d9d51d85d7eb645cde30352530d494233c434027f9e3193bae4"
    assert result["voices"] == ["piper"]
    assert result["voice_switches"] == 0
    assert result["queue_clears"] == 0
