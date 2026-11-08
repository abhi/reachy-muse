# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""A bounded local delivery ledger. Completed IDs remain until their original expiry."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

from musegadget import config, muse_api

INBOX_FILE = "reachy-inbox.json"
MAX_ITEMS = 20


def owner_binding(pairing: dict | None) -> str:
    if not pairing or not pairing.get("access_token") or not pairing.get("username"):
        return ""
    account = [muse_api.api_root(pairing.get("api_url_v2", "")), pairing["username"]]
    return hashlib.sha256(json.dumps(account).encode()).hexdigest()


@contextmanager
def _storage(path: Path | None):
    if path is None:
        yield
        return
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def outcome(status: str, reason: str) -> dict:
    return {"ok": True, "status": status, "reason": reason}


class Inbox:
    def __init__(self, path: Path | None = None, *, clock=time.time, owner: str | None = None):
        self.path = path
        self.clock = clock
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}
        self._owner = self._disk_owner() if owner is None else owner
        with self._locked():
            self._prune()

    def _disk_owner(self) -> str:
        directory = self.path.parent if self.path is not None else None
        return owner_binding(config.load_json(config.PAIRING_FILE, directory))

    def _load(self, owner: str) -> None:
        if self.path is None:
            return
        self._items.clear()
        if not owner:
            self.path.unlink(missing_ok=True)
            self.path.with_suffix(".tmp").unlink(missing_ok=True)
            return
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError()
            if data.get("owner") != owner:
                self.path.unlink()
                self.path.with_suffix(".tmp").unlink(missing_ok=True)
                return
            if self._owner != owner:
                return
            if data["version"] != 1 or len(data["items"]) > MAX_ITEMS:
                raise ValueError()
            for item in data["items"]:
                if (not isinstance(item["id"], str) or type(item["expires_at"]) not in (int, float)
                        or not math.isfinite(item["expires_at"])
                        or item["status"] not in ("queued", "delivered", "refused")
                        or not isinstance(item["receipt"], str)
                        or not isinstance(item["fingerprint"], str)
                        or not isinstance(item["reason"], str)
                        or item["status"] == "queued" and not isinstance(item["text"], str)):
                    raise ValueError()
                self._items[item["id"]] = item
        except (OSError, ValueError, KeyError, TypeError):
            raise ValueError("Reachy's inbox could not be read") from None

    @contextmanager
    def _locked(self):
        with self._lock, _storage(self.path):
            owner = self._disk_owner()
            self._load(owner)
            matches = bool(owner and owner == self._owner)
            if not matches:
                self._items.clear()
            yield matches

    def current(self, pending: dict) -> bool:
        with self._locked():
            item = self._items.get(pending["id"])
            return (item is not None and item["receipt"] == pending["receipt"]
                    and self.clock() < item["expires_at"])

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        try:
            with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump({"version": 1, "owner": self._owner,
                           "items": list(self._items.values())}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    def _prune(self) -> None:
        expired = [key for key, item in self._items.items() if item["expires_at"] <= self.clock()]
        for key in expired:
            del self._items[key]
        if expired:
            self._save()

    def add(self, params: dict) -> dict:
        with self._locked() as matches:
            if not matches:
                return outcome("refused", "disconnected")
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
                    "receipt": uuid.uuid4().hex,
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
        with self._locked():
            self._prune()
            return next((dict(item) for item in self._items.values() if item["status"] == "queued"), None)

    def claim(self, pending: dict) -> bool:
        with self._locked():
            self._prune()
            item = self._items.get(pending["id"])
            if (item is None or item["receipt"] != pending["receipt"]
                    or item["status"] != "queued"):
                return False
            previous = dict(item)
            item.update(status="refused", reason="delivery_uncertain", text=None)
            try:
                self._save()
            except OSError:
                self._items[item["id"]] = previous
                return False
            return True

    def finish(self, pending: dict, *, spoken: bool, reason: str = "spoken") -> None:
        with self._locked():
            item = self._items.get(pending["id"])
            if item is not None and item["receipt"] == pending["receipt"]:
                item.update(status="delivered" if spoken else "refused", reason=reason,
                            delivered_at=self.clock() if spoken else None, text=None)
                self._save()

    def result(self, id: str) -> dict:
        with self._locked():
            self._prune()
            item = self._items.get(id)
            return outcome(item["status"], item["reason"]) if item else outcome("expired", "ttl_elapsed")
