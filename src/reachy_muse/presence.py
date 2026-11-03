# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Coarse local observations, without images or identity."""

from dataclasses import dataclass

PRESENCE_FRESH_S = 2.0


@dataclass(frozen=True)
class Presence:
    present: bool | None = None
    observed_at: float | None = None
    last_seen_at: float | None = None

    def observe(self, present: bool | None, now: float) -> "Presence":
        return Presence(present, now, now if present else self.last_seen_at)

    def result(self, now: float) -> dict:
        fresh = self.observed_at is not None and now - self.observed_at <= PRESENCE_FRESH_S
        return {"ok": True, "present": self.present if fresh else None,
                "last_seen_seconds": (None if self.last_seen_at is None
                                      else round(max(0.0, now - self.last_seen_at)))}
