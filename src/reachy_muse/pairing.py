# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Coordinate token rotation between Reachy's service and standalone commands."""

from __future__ import annotations

import asyncio
import fcntl
import logging
import time

from musegadget import config, muse_api
from musegadget.service import Service, TOKEN_REFRESH_AGE_S, TOKEN_RETRY_S


log = logging.getLogger(__name__)


class PairedService(Service):
    async def _maybe_refresh(self, pairing: dict, force: bool = False) -> dict | None:
        return await asyncio.to_thread(self._refresh_pairing, pairing, force)

    def _refresh_pairing(self, pairing: dict, force: bool) -> dict | None:
        directory = config.state_dir()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        with (directory / "pairing.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = config.load_json(config.PAIRING_FILE, directory)
            if current != pairing:
                return current
            age = time.time() - pairing.get("access_token_saved_at", 0)
            if not force and age < TOKEN_REFRESH_AGE_S:
                return pairing
            if not force and time.monotonic() - self._last_refresh_attempt < TOKEN_RETRY_S:
                return pairing
            self._last_refresh_attempt = time.monotonic()
            tokens, status = muse_api.refresh_device_token(
                pairing["refresh_token"], self.identity.node_id,
                muse_api.api_root(pairing.get("api_url_v2", "")), self.sdk_token)
            current = config.load_json(config.PAIRING_FILE, directory)
            if current != pairing:
                return current
            if tokens:
                pairing = {**pairing, "access_token": tokens["access_token"],
                           "refresh_token": tokens["refresh_token"],
                           "access_token_saved_at": int(time.time())}
                config.save_json(config.PAIRING_FILE, pairing, directory)
                log.info("device token rotated")
                return pairing
            if status == 401:
                config.delete_json(config.PAIRING_FILE, directory)
                log.error("pairing revoked; run `muse-reachy pair` to set up again")
                return None
            return None if force else pairing
