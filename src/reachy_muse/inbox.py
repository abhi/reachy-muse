# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""A bounded local delivery ledger. Completed IDs remain until their original expiry."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

INBOX_FILE = "reachy-inbox.json"
MAX_ITEMS = 20


def clear_inbox(path: Path) -> None:
    path.unlink(missing_ok=True)
    path.with_suffix(".tmp").unlink(missing_ok=True)


def outcome(status: str, reason: str) -> dict:
    return {"ok": True, "status": status, "reason": reason}


class Inbox:
    def __init__(self, path: Path | None = None, *, clock=time.time):
        self.path = path
        self.clock = clock
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}
        if path is not None and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data["version"] != 1 or len(data["items"]) > MAX_ITEMS:
                    raise ValueError()
                for item in data["items"]:
                    if (not isinstance(item["id"], str) or type(item["expires_at"]) not in (int, float)
                            or not math.isfinite(item["expires_at"])
                            or item["status"] not in ("queued", "delivered", "refused")
                            or not isinstance(item["fingerprint"], str)
                            or not isinstance(item["reason"], str)
                            or item["status"] == "queued" and not isinstance(item["text"], str)):
                        raise ValueError()
                    self._items[item["id"]] = item
                self._prune()
            except (OSError, ValueError, KeyError, TypeError):
                raise ValueError("Reachy's inbox could not be read") from None

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        try:
            with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump({"version": 1, "items": list(self._items.values())}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _prune(self) -> None:
        expired = [key for key, item in self._items.items() if item["expires_at"] <= self.clock()]
        for key in expired:
            del self._items[key]
        if expired:
            self._save()

    def add(self, params: dict) -> dict:
        with self._lock:
            previous = self._items.get(params["id"])
            self._prune()
            if previous is not None and previous["expires_at"] <= self.clock():
                return outcome("expired", "ttl_elapsed")
            fingerprint = hashlib.sha256(json.dumps([params["text"], params["kind"]]).encode()).hexdigest()
            if previous is not None:
                if previous["fingerprint"] != fingerprint:
                    return outcome("refused", "id_conflict")
                return outcome(previous["status"], previous["reason"])
            if params["expires_in_s"] == 0:
                return outcome("expired", "ttl_elapsed")
            if len(self._items) >= MAX_ITEMS:
                return outcome("refused", "inbox_full")
            item = {"id": params["id"], "text": params["text"], "kind": params["kind"],
                    "fingerprint": fingerprint, "expires_at": self.clock() + params["expires_in_s"],
                    "status": "queued", "reason": "awaiting_presence", "delivered_at": None}
            self._items[item["id"]] = item
            try:
                self._save()
            except OSError:
                del self._items[item["id"]]
                return outcome("refused", "storage_unavailable")
            return outcome("queued", "awaiting_presence")

    def pending(self) -> dict | None:
        with self._lock:
            self._prune()
            return next((dict(item) for item in self._items.values() if item["status"] == "queued"), None)

    def claim(self, id: str) -> bool:
        with self._lock:
            self._prune()
            item = self._items.get(id)
            if item is None or item["status"] != "queued":
                return False
            previous = dict(item)
            item.update(status="refused", reason="delivery_uncertain", text=None)
            try:
                self._save()
            except OSError:
                self._items[id] = previous
                return False
            return True

    def finish(self, id: str, *, spoken: bool, reason: str = "spoken") -> None:
        with self._lock:
            item = self._items.get(id)
            if item is not None:
                item.update(status="delivered" if spoken else "refused", reason=reason,
                            delivered_at=self.clock() if spoken else None)
                self._save()

    def result(self, id: str) -> dict:
        with self._lock:
            self._prune()
            item = self._items.get(id)
            return outcome(item["status"], item["reason"]) if item else outcome("expired", "ttl_elapsed")

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            if self.path is not None:
                clear_inbox(self.path)
