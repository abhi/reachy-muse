"""tools/speaker_scores.py turns the companion's speaker lines into score distributions and error rates."""

import pytest

from reachy_muse.voice_verify import DEFAULT_THRESHOLD
from tools import speaker_scores as tool

BEFORE = """\
2026-11-21 08:50:00,000 INFO speaker stream=1 voiced_s=2.00 score=0.9900 label=owner
2026-11-21 08:55:00,000 INFO voice print enrolled from 5 utterances with 12.0 s of speech; agreement 0.9100 0.9000 \
0.8800 0.9300 0.9200
"""
LOG = BEFORE + """\
2026-11-21 09:00:05,100 INFO speaker stream=1 voiced_s=2.10 score=0.8100 label=owner
2026-11-21 09:01:00,200 INFO robot 'Reachy Mini' connected from ('192.0.2.1', 1) with ops ['hear', 'speaker']
2026-11-21 09:02:05,100 INFO speaker stream=3 voiced_s=0.60 score=none label=unknown
2026-11-21 09:03:05,100 INFO speaker stream=5 voiced_s=3.00 score=0.7000 label=other
2026-11-21 18:00:05,100 INFO speaker stream=1 voiced_s=2.50 score=0.3000 label=other
2026-11-21 18:01:05,100 INFO speaker stream=3 voiced_s=2.00 score=0.7600 label=owner
2026-11-21 18:02:05,100 INFO speaker stream=5 voiced_s=1.20 score=-0.0500 label=other
"""


def run(tmp_path, capsys, text, *argv):
    log = tmp_path / "companion.log"
    log.write_text(text)
    status = tool.main([str(log), *argv])
    return status, capsys.readouterr()


def test_the_report_covers_the_requests_since_the_last_enrollment(tmp_path, capsys):
    status, said = run(tmp_path, capsys, LOG)
    assert (status, said.out) == (0, (
        "6 requests since the enrollment at 2026-11-21 08:55:00: 2 owner, 3 other, 1 unknown\n"
        "\n"
        "       scored\n"
        "count       5\n"
        "  min  -0.050\n"
        "  p10  -0.050\n"
        "  p50   0.700\n"
        "  p90   0.810\n"
        "  max   0.810\n"))


def test_owner_and_other_spans_give_each_thresholds_false_others_and_false_owners(tmp_path, capsys):
    status, said = run(tmp_path, capsys, LOG, "--owner", "2026-11-21T08:00", "2026-11-21T10:00",
                       "--other", "2026-11-21 18:00", "2026-11-21 19:00")
    rows = said.out.splitlines()
    assert rows[2:9] == [
        "       scored  owner   other",
        "count       5      2       3",
        "  min  -0.050  0.700  -0.050",
        "  p10  -0.050  0.700  -0.050",
        "  p50   0.700  0.700   0.300",
        "  p90   0.810  0.810   0.760",
        "  max   0.810  0.810   0.760",
    ]
    table = {row.split()[0]: " ".join(row.split()[1:]) for row in rows[10:]}
    assert (table["threshold"], table["0.30"], table["0.70"], table["0.74"], table["0.75"], table["0.80"],
            table["0.90"]) == (
        "false other false owner", "0/2 0% 2/3 67%", "0/2 0% 1/3 33%", "(default) 1/2 50% 1/3 33%", "1/2 50% 1/3 33%",
        "1/2 50% 0/3 0%", "2/2 100% 0/3 0%")


def test_the_table_has_a_row_for_the_threshold_the_companion_started_with(tmp_path, capsys):
    started = ("2026-11-21 08:00:00,000 INFO speaker recognition ready: model wespeaker_en_voxceleb_resnet34, "
               "threshold 0.62, a voice print is enrolled\n")
    status, said = run(tmp_path, capsys, started + LOG, "--owner", "2026-11-21T08:00", "2026-11-21T10:00",
                       "--other", "2026-11-21 18:00", "2026-11-21 19:00")
    rows = [row.split() for row in said.out.splitlines()[11:]]
    assert [row for row in rows if "(in" in row or "(default)" in row] == [
        ["0.62", "(in", "use)", "0/2", "0%", "1/3", "33%"]]


def test_the_default_threshold_here_is_the_packages():
    assert tool.DEFAULT_THRESHOLD == DEFAULT_THRESHOLD


def test_min_speech_leaves_out_shorter_requests(tmp_path, capsys):
    status, said = run(tmp_path, capsys, LOG, "--min-speech", "2")
    assert said.out.splitlines()[0] == "4 requests since the enrollment at 2026-11-21 08:55:00: 2 owner, 2 other, 0 unknown"
    assert said.out.splitlines()[4] == "  min   0.300"


def test_a_log_without_an_enrollment_is_reported_whole(tmp_path, capsys):
    status, said = run(tmp_path, capsys, LOG.replace(BEFORE, ""))
    assert said.out.splitlines()[0] == "6 requests in the whole log, which has no enrollment: 2 owner, 3 other, 1 unknown"


@pytest.mark.parametrize("argv, error", [
    (["--owner", "2026-11-21T10:00", "2026-11-21T09:00"], "a span ends before it starts"),
    (["--other", "yesterday", "today"], "Invalid isoformat string: 'yesterday'"),
])
def test_a_span_that_cannot_be_read_is_refused(argv, error, capsys):
    with pytest.raises(SystemExit):
        tool.main(argv)
    assert error in capsys.readouterr().err


def test_a_log_without_speaker_lines_says_so(tmp_path, capsys):
    status, said = run(tmp_path, capsys, "2026-11-21 09:01:00,200 INFO serving wss://0.0.0.0:8765/v1 (real engines)\n")
    assert (status, said.err) == (1, "no speaker lines found\n")
