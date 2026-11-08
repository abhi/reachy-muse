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


@pytest.mark.parametrize("force", [False, True])
def test_connection_during_pair_completion_uses_published_inbox_generation(tmp_path, monkeypatch, force):
    from musegadget.pair import PairOutcome, PairResult
    from reachy_muse import cli
    from reachy_muse.inbox import INBOX_FILE, INBOX_GENERATION, Inbox
    from reachy_muse.pairing import PairedService
    from test_inbox import notice
    from test_registered_commands import command_device, registered_device

    monkeypatch.setenv(config.STATE_DIR_ENV, str(tmp_path))

    async def check():
        service, _, _ = command_device()
        service.inbox = Inbox(tmp_path / INBOX_FILE)
        if force:
            config.save_json(config.PAIRING_FILE, PAIRING)
        saved, refreshing = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        results = []

        def pair(**kwargs):
            config.save_json(config.PAIRING_FILE, {
                "username": "new", "access_token": "new-token", "access_token_saved_at": time.time()})
            loop.call_soon_threadsafe(saved.set)
            assert release.wait(2)
            return PairResult(PairOutcome.PAIRED)

        refresh = service._refresh_pairing

        def refreshing_pairing(*args):
            loop.call_soon_threadsafe(refreshing.set)
            return refresh(*args)

        async def connected(self, vm, pairing):
            async with registered_device(self, monkeypatch) as (session, _, voice):
                await asyncio.to_thread(release.wait, 2)
                await pairing_task
                results.append(await asyncio.to_thread(session._run_command, "reachy.notify", notice()))

        async def connect():
            pairing = await service._maybe_refresh(config.load_json(config.PAIRING_FILE))
            await service._session({}, pairing)

        monkeypatch.setattr("musegadget.pair.pair", pair)
        monkeypatch.setattr(service, "_refresh_pairing", refreshing_pairing)
        monkeypatch.setattr(PairedService, "_session", connected)
        flags = ["--force"] if force else []
        pairing_task = asyncio.create_task(asyncio.to_thread(
            cli.pair_reachy, cli.parser().parse_args(["pair", "--no-introduce", *flags])))
        connection = None
        try:
            await asyncio.wait_for(saved.wait(), 2)
            connection = asyncio.create_task(connect())
            await asyncio.wait_for(refreshing.wait(), 2)
            await asyncio.wait({connection}, timeout=.05)
            release.set()
            await asyncio.wait_for(connection, 2)
            assert await pairing_task == 0
            assert INBOX_GENERATION in config.load_json(config.PAIRING_FILE)
            assert results == [{"ok": True, "payload": {"status": "queued", "reason": "awaiting_presence"}}]
        finally:
            release.set()
            await asyncio.gather(pairing_task, *([connection] if connection else []), return_exceptions=True)

    asyncio.run(check())
