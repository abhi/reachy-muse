#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tabulate Reachy's per-turn latency from its journal.

    ssh pollen@reachy-mini.local 'journalctl -u muse-reachy --since "-2h" --no-pager -o cat' \\
        | python3 tools/turn_timeline.py

Each "Reachy turn timeline" line becomes one row. By default a cell is the seconds since the
turn's first stage; with --steps it is the seconds since the previous stage the turn reached.
Standard library only, so it runs anywhere the journal can be read.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Iterable, List, Tuple

STAGES = ("wake", "speech_end", "endpoint", "transcript", "dispatch", "muse_ack", "chat_audio", "ack_audio",
          "first_status", "first_text", "reply_done", "narration_done", "first_audio")
_LINE = re.compile(r"^(?:(?P<when>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\S*\s.*?)?Reachy turn timeline (?P<fields>.*)$")


def parse(lines: Iterable[str]) -> List[Tuple[str, dict, dict]]:
    """(log time, stage seconds, other facts) for each timeline line."""
    turns = []
    for line in lines:
        match = _LINE.search(line.rstrip("\n"))
        if match is None:
            continue
        stages, facts = {}, {}
        for field in match["fields"].split():
            key, _, value = field.partition("=")
            if key in STAGES:
                stages[key] = float(value)
            else:
                facts[key] = value
        turns.append((match["when"] or "", stages, facts))
    return turns


def table(turns: List[Tuple[str, dict, dict]], *, steps: bool = False) -> str:
    stages = [stage for stage in STAGES if any(stage in turn[1] for turn in turns)]
    facts = sorted({key for _, _, turn_facts in turns for key in turn_facts})
    header = ["time", *stages, "total", *facts]
    rows = [header]
    for when, times, turn_facts in turns:
        row, previous = [when[11:] or "-"], 0.0
        for stage in stages:
            if stage not in times:
                row.append("-")
                continue
            row.append(f"{times[stage] - previous if steps else times[stage]:.2f}")
            if steps:
                previous = times[stage]
        row.append(f"{max(times.values()):.2f}" if times else "-")
        row += [turn_facts.get(key, "-") for key in facts]
        rows.append(row)
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    return "\n".join("  ".join(cell.rjust(width) for cell, width in zip(row, widths)) for row in rows) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("journal", nargs="?", type=argparse.FileType("r"), default=sys.stdin,
                        help="journal text; standard input when omitted")
    parser.add_argument("--steps", action="store_true", help="show each stage's time since the previous one")
    args = parser.parse_args(argv)
    turns = parse(args.journal)
    if not turns:
        print("no turn timeline lines found", file=sys.stderr)
        return 1
    sys.stdout.write(table(turns, steps=args.steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
