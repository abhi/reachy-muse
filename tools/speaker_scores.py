#!/usr/bin/env python3
"""Measure the companion's owner-voice labels from its log. No audio is needed.

    python3 tools/speaker_scores.py companion.log
    python3 tools/speaker_scores.py companion.log --owner 2026-11-21T09:00 2026-11-21T12:00 \\
        --other 2026-11-22T18:00 2026-11-22T19:30

The companion logs one line for each request it labels: the seconds of speech it heard, the
request's score against the owner's voice print, and the label it sent the robot. Each --owner
span is a time when only the owner spoke to Reachy, and each --other span a time when only other
people did. With both, the report shows how often each threshold would call the owner someone
else (false other) and call someone else the owner (false owner). Standard library only.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple

_LINE = re.compile(r"(?P<when>\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d)\S*\s.*?\bspeaker stream=\d+ "
                   r"voiced_s=(?P<voiced>\d+(?:\.\d+)?) score=(?P<score>-?\d+(?:\.\d+)?|none) label=(?P<label>[a-z]+)")
THRESHOLDS = tuple(round(.3 + .05 * step, 2) for step in range(13))
Span = Tuple[datetime, datetime]


@dataclass(frozen=True)
class Request:
    when: datetime
    voiced_s: float
    score: Optional[float]
    label: str


def parse(lines: Iterable[str]) -> List[Request]:
    requests = []
    for line in lines:
        match = _LINE.search(line)
        if match is not None:
            score = None if match["score"] == "none" else float(match["score"])
            requests.append(Request(datetime.fromisoformat(match["when"]), float(match["voiced"]), score, match["label"]))
    return requests


def scores(requests: Iterable[Request], spans: Sequence[Span] = ()) -> List[float]:
    """The sorted scores of scored requests, only those inside ``spans`` when any are given."""
    return sorted(request.score for request in requests if request.score is not None
                  and (not spans or any(start <= request.when <= end for start, end in spans)))


def _rank(values: Sequence[float], fraction: float) -> float:
    return values[max(0, math.ceil(fraction * len(values)) - 1)]


def report(requests: Sequence[Request], owner: Sequence[Span] = (), other: Sequence[Span] = ()) -> str:
    labels = Counter(request.label for request in requests)
    lines = [f"{len(requests)} requests: {labels['owner']} owner, {labels['other']} other, "
             f"{labels['unknown']} unknown (no print, or under 1 s of speech)"]
    groups = [("scored", scores(requests))] + [(name, scores(requests, spans))
                                               for name, spans in (("owner", owner), ("other", other)) if spans]
    rows = [("", *(name for name, _ in groups)), ("count", *(str(len(values)) for _, values in groups))]
    for name, fraction in (("min", 0.0), ("p10", .1), ("p50", .5), ("p90", .9), ("max", 1.0)):
        rows.append((name, *(f"{_rank(values, fraction):.3f}" if values else "-" for _, values in groups)))
    lines += ["", *_table(rows)]
    if owner and other:
        genuine, impostor = scores(requests, owner), scores(requests, other)
        rows = [("threshold", "false other", "false owner")]
        for threshold in THRESHOLDS:
            rows.append((f"{threshold:.2f}", _rate(sum(score < threshold for score in genuine), len(genuine)),
                         _rate(sum(score >= threshold for score in impostor), len(impostor))))
        lines += ["", *_table(rows)]
    return "\n".join(lines) + "\n"


def _rate(count: int, total: int) -> str:
    return f"{count}/{total} {100 * count / total:.0f}%" if total else "-"


def _table(rows: Sequence[Sequence[str]]) -> List[str]:
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return ["  ".join(cell.rjust(width) for cell, width in zip(row, widths)).rstrip() for row in rows]


def _span(values: Sequence[str]) -> Span:
    start, end = (datetime.fromisoformat(value) for value in values)
    if end < start:
        raise argparse.ArgumentTypeError("a span ends before it starts")
    return start, end


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("log", nargs="?", type=argparse.FileType("r"), default=sys.stdin,
                        help="the companion's log; standard input when omitted")
    for name in ("owner", "other"):
        parser.add_argument(f"--{name}", nargs=2, action="append", default=[], metavar=("FROM", "TO"),
                            help=f"a span, in local ISO time, when only {'the owner' if name == 'owner' else 'others'} "
                                 "spoke to Reachy; repeat for more")
    args = parser.parse_args(argv)
    try:
        owner, other = [_span(span) for span in args.owner], [_span(span) for span in args.other]
    except (ValueError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    requests = parse(args.log)
    if not requests:
        print("no speaker lines found", file=sys.stderr)
        return 1
    sys.stdout.write(report(requests, owner, other))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
