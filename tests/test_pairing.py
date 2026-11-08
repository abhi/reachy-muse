# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
import threading
import time

import pytest

from musegadget import config, muse_api
from musegadget.identity import Identity

from reachy_muse import introduction
from reachy_muse.conversation import ReachyService
from test_introduction import Session


PAIRING = {"access_token": "expired", "refresh_token": "valid-refresh", "username": "paired-user"}
TOKENS = {"access_token": "fresh-access", "refresh_token": "fresh-refresh"}
VM = {"vm_id": "vm-1", "is_default": True, "vm_auth_token": "fake-vm-token"}


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv(config.STATE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(config, "sdk_token", lambda: "fake-sdk-token")
    ident = Identity("02:00:00:ab:cd:ef")
    monkeypatch.setattr(introduction.identity, "load_or_create", lambda: ident)
    monkeypatch.setattr(introduction, "LinkSession", lambda **kwargs: Session())
    monkeypatch.setattr(muse_api, "fetch_vms_with_status", lambda token, api:
                        ([], 401) if token == "expired" else ([VM], 200))
    monkeypatch.setattr(muse_api, "refresh_device_token", lambda *args: pytest.fail("unexpected refresh"))
    config.save_json(config.PAIRING_FILE, PAIRING, tmp_path)
    return ReachyService(identity=ident, executor=None, sdk_token="fake-sdk-token", backends=lambda _: None)


def test_standalone_uses_rotation_completed_during_discovery(tmp_path, monkeypatch, service):
    async def check():
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        fetched, refreshed = [], []

        def fetch(token, api):
            fetched.append(token)
            if token == "expired":
                loop.call_soon_threadsafe(started.set)
                assert release.wait(2)
                return [], 401
            return [VM], 200

        def refresh(token, *args):
            refreshed.append(token)
            return (TOKENS, 200) if len(refreshed) == 1 else (None, 401)

        monkeypatch.setattr(muse_api, "fetch_vms_with_status", fetch)
        monkeypatch.setattr(muse_api, "refresh_device_token", refresh)
        task = asyncio.create_task(introduction.introduce_paired())
        try:
            await started.wait()
            rotated = await service._maybe_refresh(PAIRING, force=True)
            release.set()
            result, = await asyncio.gather(task, return_exceptions=True)
            assert config.load_json(config.PAIRING_FILE, tmp_path) == rotated
            assert refreshed == ["valid-refresh"]
            assert fetched == ["expired", "fresh-access"]
            assert result is None
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(asyncio.wait_for(check(), 3))


@pytest.mark.parametrize("first", ["service", "standalone"])
def test_service_and_standalone_share_one_token_rotation(tmp_path, monkeypatch, service, first):
    async def check():
        started, discovered = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        refreshed = []

        def fetch(token, api):
            loop.call_soon_threadsafe(discovered.set)
            return ([], 401) if token == "expired" else ([VM], 200)

        def refresh(token, *args):
            refreshed.append(token)
            if len(refreshed) != 1:
                return None, 401
            loop.call_soon_threadsafe(started.set)
            assert release.wait(2)
            return TOKENS, 200

        monkeypatch.setattr(muse_api, "fetch_vms_with_status", fetch)
        monkeypatch.setattr(muse_api, "refresh_device_token", refresh)
        calls = {"service": lambda: service._maybe_refresh(PAIRING, force=True),
                 "standalone": introduction.introduce_paired}
        tasks = [asyncio.create_task(calls[first]())]
        try:
            await started.wait()
            tasks.append(asyncio.create_task(calls["standalone" if first == "service" else "service"]()))
            await discovered.wait()
            await asyncio.wait({tasks[1]}, timeout=0.05)
            release.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert refreshed == ["valid-refresh"]
            saved = config.load_json(config.PAIRING_FILE, tmp_path)
            assert saved["access_token"] == "fresh-access"
            assert saved["refresh_token"] == "fresh-refresh"
            assert saved["username"] == "paired-user"
            assert saved["access_token_saved_at"] > 0
            assert results == ([saved, None] if first == "service" else [None, saved])
        finally:
            release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(asyncio.wait_for(check(), 3))


@pytest.mark.parametrize("status", [200, 401])
@pytest.mark.parametrize("entrypoint", ["service", "standalone"])
def test_refresh_preserves_pairing_replaced_during_request(tmp_path, monkeypatch, service, status, entrypoint):
    replacement = {"access_token": "replacement-access", "refresh_token": "replacement-refresh",
                   "username": "replacement-user"}

    def refresh(*args):
        config.save_json(config.PAIRING_FILE, replacement, tmp_path)
        return (TOKENS, 200) if status == 200 else (None, 401)

    monkeypatch.setattr(muse_api, "refresh_device_token", refresh)

    async def check():
        call = (service._maybe_refresh(PAIRING, force=True) if entrypoint == "service"
                else introduction.introduce_paired())
        result, = await asyncio.gather(call, return_exceptions=True)
        assert config.load_json(config.PAIRING_FILE, tmp_path) == replacement
        assert result == (replacement if entrypoint == "service" else None)

    asyncio.run(asyncio.wait_for(check(), 3))


def test_revoked_pairing_clears_the_notification_ledger(tmp_path, monkeypatch, service):
    from reachy_muse.inbox import Inbox
    from test_inbox import notice
    service.inbox = Inbox(tmp_path / "inbox.json")
    service.inbox.add(notice())
    monkeypatch.setattr(muse_api, "refresh_device_token", lambda *args: (None, 401))
    assert asyncio.run(service._maybe_refresh(PAIRING, force=True)) is None
    assert service.inbox.pending() is None
    assert (tmp_path / "inbox.json").exists() is False


@pytest.mark.parametrize("outcome", ["WINDOW_CLOSED", "INVALID_SDK_TOKEN", "cancelled"])
@pytest.mark.parametrize("reconnect", [False, True])
def test_failed_forced_pairing_keeps_queued_items_deliverable(tmp_path, monkeypatch, outcome, reconnect):
    from musegadget.pair import PairOutcome, PairResult
    from reachy_muse import cli
    from reachy_muse.inbox import INBOX_FILE, Inbox
    from reachy_muse.pairing import PairedService
    from test_inbox import notice
    from test_registered_commands import command_device, registered_device

    monkeypatch.setenv(config.STATE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr("reachy_muse.notifications.ATTENTION_S", .01)
    config.save_json(config.PAIRING_FILE, PAIRING)
    path = tmp_path / INBOX_FILE

    def pair(**kwargs):
        if outcome == "cancelled":
            raise KeyboardInterrupt()
        return PairResult(PairOutcome[outcome], sdk_token_problem="invalid SDK token")

    monkeypatch.setattr("musegadget.pair.pair", pair)

    async def connected(self, vm, pairing):
        async with registered_device(self, monkeypatch) as (_, _, voice):
            assert self.inbox.pending()["text"] == "Call Mom."
            self._presence.mark("voice", time.monotonic())
            await voice.notifications.deliver(30)
            assert self.inbox.result("old") == {"ok": True, "status": "delivered", "reason": "spoken"}

    monkeypatch.setattr(PairedService, "_session", connected)

    async def check():
        service, _, speech = command_device()
        service.inbox = Inbox(path)
        assert service.inbox.add(notice("old"))["status"] == "queued"
        before = path.read_bytes()
        args = cli.parser().parse_args(["pair", "--force", "--no-introduce"])
        if outcome == "cancelled":
            with pytest.raises(KeyboardInterrupt):
                cli.pair_reachy(args)
        else:
            assert cli.pair_reachy(args) == 1
        assert config.load_json(config.PAIRING_FILE) == PAIRING
        assert path.read_bytes() == before
        assert service.inbox.pending()["text"] == "Call Mom."
        assert Inbox(path).pending()["text"] == "Call Mom."
        if reconnect:
            await service._session({}, PAIRING)
        else:
            await connected(service, {}, PAIRING)
        assert speech.words == ["Call Mom."]

    asyncio.run(check())


def test_startup_without_credentials_discards_the_queue(tmp_path, monkeypatch):
    from reachy_muse import cli
    from reachy_muse.inbox import INBOX_FILE, Inbox
    from test_inbox import notice

    monkeypatch.setenv(config.STATE_DIR_ENV, str(tmp_path))
    config.save_json(config.PAIRING_FILE, PAIRING)
    path = tmp_path / INBOX_FILE
    inbox = Inbox(path)
    assert inbox.add(notice())["status"] == "queued"
    config.delete_json(config.PAIRING_FILE)
    monkeypatch.setattr(cli, "_hardware", lambda args: pytest.fail("hardware started without credentials"))
    assert asyncio.run(cli.run(cli.parser().parse_args(["run"]))) == 1
    assert inbox.pending() is None
    assert path.exists() is False
    assert Inbox(path).add(notice()) == {"ok": True, "status": "refused", "reason": "disconnected"}


def test_startup_missing_credentials_read_cannot_delete_a_new_owners_queue(tmp_path, monkeypatch):
    from reachy_muse import cli
    from reachy_muse.inbox import INBOX_FILE, Inbox
    from test_inbox import notice

    monkeypatch.setenv(config.STATE_DIR_ENV, str(tmp_path))
    path = tmp_path / INBOX_FILE
    load = config.load_json

    def pair_after_read(name, *args):
        saved = load(name, *args)
        if name == config.PAIRING_FILE and saved is None:
            config.save_json(config.PAIRING_FILE, PAIRING)
            assert Inbox(path).add(notice())["status"] == "queued"
        return saved

    monkeypatch.setattr(config, "load_json", pair_after_read)
    assert asyncio.run(cli.run(cli.parser().parse_args(["run"]))) == 1
    assert config.load_json(config.PAIRING_FILE) == PAIRING
    assert Inbox(path).pending()["text"] == "Call Mom."
