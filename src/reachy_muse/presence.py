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


class PresenceFusion:
    """Positive cues outlive brief camera gaps; absence requires a fresh camera observation."""

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._observations: dict[str, Presence] = {}

    def observe(self, source: str, observation: Presence) -> None:
        with self._lock:
            previous = self._observations.get(source, Presence())
            last_seen = max((at for at in (previous.last_seen_at, observation.last_seen_at)
                             if at is not None), default=None)
            self._observations[source] = Presence(observation.present, observation.observed_at, last_seen)

    def mark(self, source: str, now: float) -> None:
        self.observe(source, Presence(True, now, now))

    def result(self, now: float) -> dict:
        with self._lock:
            observations = tuple(self._observations.items())
        seen = [(state.last_seen_at, source) for source, state in observations if state.last_seen_at is not None]
        last_seen, last_source = max(seen, default=(None, None))
        fresh = [(state.observed_at, source, state.present) for source, state in observations
                 if state.observed_at is not None
                 and 0 <= now - state.observed_at <= (30 if source in ("voice", "wave") else PRESENCE_FRESH_S)]
        positive = [(at, source) for at, source, present in fresh if present is True]
        present = True if positive else False if any(value is False for _, _, value in fresh) else None
        source = max(positive)[1] if positive else last_source
        if source is None and fresh:
            source = max(fresh)[1]
        return {"present": present,
                "last_seen_seconds": None if last_seen is None else round(max(0, now - last_seen)),
                "source": source}
