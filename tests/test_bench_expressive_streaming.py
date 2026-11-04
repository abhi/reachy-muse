"""Exercise the offline replay's public result against literal PCM expectations."""
import asyncio

from bench.expressive_streaming import once


def test_prepared_replay_preserves_pcm_and_starts_before_reply_completion():
    result = asyncio.run(once(prepared=True))
    assert result["pcm_frames"] == 360
    assert result["chunk_frames"] == [80, 40, 80, 40, 80, 40]
    assert result["sentence_count"] == 3
    assert result["classified_count"] == 3
    assert result["expressions"] == {"neutral": 1, "curious": 1, "happy": 1}
    assert result["finish_notifications"] == 3
    assert result["first_push_before_completion"] is True
    assert result["first_push_ms"] >= 0
    assert result["on_start_ms"] >= 0


def test_streamed_and_prepared_replay_emit_identical_audio():
    streamed = asyncio.run(once(prepared=False))
    prepared = asyncio.run(once(prepared=True))
    assert streamed["pcm_sha256"] == prepared["pcm_sha256"]
    assert streamed["chunk_frames"] == prepared["chunk_frames"]
