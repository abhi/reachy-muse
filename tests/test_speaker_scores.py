"""tools/speaker_scores.py turns the companion's speaker lines into score distributions and error rates."""

import pytest

from tools import speaker_scores as tool

LOG = """\
2026-11-21 09:00:05,100 INFO speaker stream=1 voiced_s=2.10 score=0.8100 label=owner
2026-11-21 09:01:00,200 INFO robot 'Reachy Mini' connected from ('192.0.2.1', 1) with ops ['hear', 'speaker']
2026-11-21 09:02:05,100 INFO speaker stream=3 voiced_s=0.60 score=none label=unknown
2026-11-21 09:03:05,100 INFO speaker stream=5 voiced_s=3.00 score=0.7000 label=other
2026-11-21 18:00:05,100 INFO speaker stream=1 voiced_s=2.50 score=0.3000 label=other
2026-11-21 18:01:05,100 INFO speaker stream=3 voiced_s=2.00 score=0.7600 label=owner
2026-11-21 18:02:05,100 INFO speaker stream=5 voiced_s=1.20 score=-0.0500 label=other
"""


def test_the_report_needs_no_spans_to_show_the_score_distribution(tmp_path, capsys):
    log = tmp_path / "companion.log"
    log.write_text(LOG)
    assert tool.main([str(log)]) == 0
    assert capsys.readouterr().out == (
        "6 requests: 2 owner, 3 other, 1 unknown (no print, or under 1 s of speech)\n"
        "\n"
        "       scored\n"
        "count       5\n"
        "  min  -0.050\n"
        "  p10  -0.050\n"
        "  p50   0.700\n"
        "  p90   0.810\n"
        "  max   0.810\n")


def test_owner_and_other_spans_give_each_thresholds_false_others_and_false_owners(tmp_path, capsys):
    log = tmp_path / "companion.log"
    log.write_text(LOG)
    assert tool.main([str(log), "--owner", "2026-11-21T09:00", "2026-11-21T10:00",
                      "--other", "2026-11-21 18:00", "2026-11-21 19:00"]) == 0
    rows = capsys.readouterr().out.splitlines()
    assert rows[2:9] == [
        "       scored  owner   other",
        "count       5      2       3",
        "  min  -0.050  0.700  -0.050",
        "  p10  -0.050  0.700  -0.050",
        "  p50   0.700  0.700   0.300",
        "  p90   0.810  0.810   0.760",
        "  max   0.810  0.810   0.760",
    ]
    table = {row.split()[0]: row.split(None, 1)[1] for row in rows[10:]}
    assert (table["threshold"], table["0.30"], table["0.70"], table["0.75"], table["0.80"], table["0.90"]) == (
        "false other  false owner", "0/2 0%      2/3 67%", "0/2 0%      1/3 33%", "1/2 50%      1/3 33%",
        "1/2 50%       0/3 0%", "2/2 100%       0/3 0%")


def test_lines_without_a_score_and_other_log_lines_are_read_but_not_scored():
    requests = tool.parse(LOG.splitlines())
    assert [(request.voiced_s, request.score, request.label) for request in requests[:2]] == [
        (2.1, .81, "owner"), (.6, None, "unknown")]
    assert len(requests) == 6


@pytest.mark.parametrize("argv, error", [
    (["--owner", "2026-11-21T10:00", "2026-11-21T09:00"], "a span ends before it starts"),
    (["--other", "yesterday", "today"], "Invalid isoformat string: 'yesterday'"),
])
def test_a_span_that_cannot_be_read_is_refused(argv, error, capsys):
    with pytest.raises(SystemExit):
        tool.main(argv)
    assert error in capsys.readouterr().err


def test_a_log_without_speaker_lines_says_so(tmp_path, capsys):
    log = tmp_path / "companion.log"
    log.write_text("2026-11-21 09:01:00,200 INFO serving wss://0.0.0.0:8765/v1 (real engines)\n")
    assert tool.main([str(log)]) == 1
    assert capsys.readouterr().err == "no speaker lines found\n"
