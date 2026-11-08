# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Deliver the local inbox between conversation turns through Reachy's speech owner."""

import asyncio
import logging
import time

from reachy_muse.inbox import Inbox
from reachy_muse.motion.wave_back import DURATION_S as ATTENTION_S

log = logging.getLogger(__name__)


class Notifications:
    def __init__(self, inbox: Inbox, presence, voice):
        self.inbox = inbox
        self.presence = presence
        self.voice = voice
        self._lock = asyncio.Lock()

    def available(self) -> bool:
        return (self.voice._say_blocked() is None
                and not self.voice._has_output() and not self.voice._wake_first_request_pending)

    async def notify(self, params: dict, timeout_s: float) -> dict:
        result = self.inbox.add(params)
        if result["status"] == "queued" and params["mode"] == "now":
            await self.deliver(timeout_s)
            result = self.inbox.result(params["id"])
        if result["status"] == "queued" and self.presence()["present"] is True:
            reason = self.voice._say_blocked() or "awaiting_availability"
            result = {**result, "reason": reason}
        return result

    async def run(self) -> None:
        while not self.voice._closed:
            try:
                await self.deliver(30)
            except OSError:
                log.warning("Reachy inbox storage unavailable")
            await asyncio.sleep(.25)

    async def deliver(self, timeout_s: float) -> None:
        item = self.inbox.pending()
        if (item is None or self._lock.locked() or self.presence()["present"] is not True
                or not self.available() or timeout_s <= ATTENTION_S):
            return
        async with self._lock:
            item = self.inbox.pending()
            if item is None:
                return
            deadline = time.monotonic() + timeout_s
            await asyncio.to_thread(self.voice.hardware.gesture, "wave")
            await asyncio.sleep(ATTENTION_S)
            if (not self.available() or self.presence()["present"] is False
                    or time.monotonic() >= deadline):
                return
            if not self.inbox.claim(item["id"]):
                return
            result = await self.voice.say(
                item["text"], max(0, deadline - time.monotonic()),
                allowed=lambda: (self.presence()["present"] is not False
                                 and self.inbox.clock() < item["expires_at"]))
            self.inbox.finish(item["id"], spoken=result["spoken"], reason=result.get("reason", "spoken"))
