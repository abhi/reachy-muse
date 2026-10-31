# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Final-rule observations only. This module has no executor or retained transcripts."""

import logging
import time

from reachy_muse.arbitration import Snapshot, decide, validate_decision

log = logging.getLogger(__name__)


def observe(snapshot: Snapshot) -> None:
    started = time.perf_counter()
    decision = decide(snapshot)
    validate_decision(snapshot, decision)
    elapsed_ms = (time.perf_counter() - started) * 1000
    log.info("Reachy arbitration shadow route=%s evidence=%s rules_ms=%.3f",
             decision.action.kind, decision.evidence[0].code.value, elapsed_ms)
