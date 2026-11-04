# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
import threading

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
