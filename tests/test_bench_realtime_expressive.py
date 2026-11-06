import pytest

from bench.realtime_expressive import PROFILES, alignment

pytest.importorskip("numpy")


def test_replayed_replies_start_each_sentence_s_expression_where_its_speech_starts():
    report = alignment(8, 20261105, PROFILES["clear"])
    assert {regime: (item["sentences"], item["within_150ms"], item["shown"], item["non_neutral_expressions"])
            for regime, item in report.items()} == {
        "ahead": (21, 1.0, 1.0, 10), "with_audio": (21, .952, 1.0, 10), "late": (21, 1.0, 1.0, 9)}
