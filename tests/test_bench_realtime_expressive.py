import pytest

from bench.realtime_expressive import PROFILES, REPLIES, alignment, replay, synthesize
from reachy_muse.motion.reply_motion import ReplyMotion

np = pytest.importorskip("numpy")


@pytest.mark.parametrize("regime", ["ahead", "with_audio", "late"])
def test_the_benchmark_finishes_the_transcript_only_after_all_its_deltas(monkeypatch, regime):
    events = []

    class Motion(ReplyMotion):
        def words(self, response_id, delta):
            events.append(("words", delta))
            super().words(response_id, delta)

        def transcript(self, response_id, text):
            events.append(("done", text))
            super().transcript(response_id, text)

    monkeypatch.setattr("bench.realtime_expressive.ReplyMotion", Motion)
    text = REPLIES[0]
    pcm, sentences, starts = synthesize(text, np.random.default_rng(7))
    replay(text, pcm, sentences, starts, regime)
    assert events[-1] == ("done", text)
    assert "".join(value for kind, value in events[:-1]) == text


def test_replayed_replies_start_each_sentence_s_expression_where_its_speech_starts():
    report = alignment(8, 20261105, PROFILES["clear"])
    assert {regime: (item["sentences"], item["within_150ms"], item["shown"], item["non_neutral_expressions"])
            for regime, item in report.items()} == {
        "ahead": (21, 1.0, 1.0, 10), "with_audio": (21, .952, 1.0, 10), "late": (21, .905, 1.0, 9)}
