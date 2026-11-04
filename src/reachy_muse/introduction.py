# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Introduce Reachy in the user's main Muse chat, once per destination and text version."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
from pathlib import Path
import time

from musegadget import config, identity, muse_api
from musegadget.link_client import DeviceDescription, LinkSession, acknowledgement
from musegadget.muse_turn import MuseTurn
from musegadget.service import DEFAULT_NOISE_HOST, Service

from reachy_muse import __version__
from reachy_muse.contract.commands import COMMAND_SPECS
from reachy_muse.contract.prompts import REACHY_INTRODUCTION


INTRODUCTION_VERSION = hashlib.sha256(REACHY_INTRODUCTION.encode("utf-8")).hexdigest()
INTRODUCTION_STATE_FILE = "reachy-introduction.json"
INTRODUCTION_TIMEOUT_S = 180
INTRODUCTION_CLEANUP_S = 2
log = logging.getLogger(__name__)


class _UncertainIntroduction(RuntimeError):
    pass


class _IntroductionTurn(MuseTurn):
    def __init__(self):
        super().__init__()
        self._after_user = False

    def feed(self, event):
        if not self.acknowledged:
            return super().feed(event)
        payload = event.get("payload")
        if event.get("type") != "event" or not isinstance(payload, dict):
            return []
        chat = payload.get("session_id") or event.get("session_id")
        if chat and self.session_id and chat != self.session_id:
            return []
        message_id = payload.get("message_id") or event.get("message_id") or payload.get("id")
        message_id = message_id if isinstance(message_id, str) else None
        if event.get("event") == "message.user":
            self._after_user = message_id in self._user_ids
            return []
        parent = payload.get("reply_to_message_id") or payload.get("parent_message_id")
        parent = parent if isinstance(parent, str) else None
        if (self._after_user or parent in self._user_ids or message_id in self.replies
                or parent in self.replies):
            return super().feed(event)
        return []


def _receipts(directory: Path) -> list[dict]:
    try:
        state = json.loads((directory / INTRODUCTION_STATE_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        raise ValueError("Reachy's saved introduction could not be read") from exc
    if isinstance(state, dict) and isinstance(state.get("version"), str):
        raise _UncertainIntroduction(
            "Reachy's saved introduction has no destination. Check your main Muse chat "
            "before replacing the legacy receipt in reachy-introduction.json.")
    receipts = state.get("receipts") if isinstance(state, dict) else None
    if not isinstance(receipts, list) or any(
            not isinstance(receipt, dict)
            or any(not isinstance(receipt.get(key), str)
                   for key in ("version", "vm_id", "node_id", "username", "api"))
            or receipt.get("status") not in ("pending", "sent") for receipt in receipts):
        raise ValueError("Reachy's saved introduction is invalid")
    return receipts


def _receipt(directory: Path, vm_id: str, node_id: str) -> dict:
    pairing = config.load_json(config.PAIRING_FILE, directory) or {}
    return {"version": INTRODUCTION_VERSION, "vm_id": vm_id, "node_id": node_id,
            "username": pairing.get("username", ""),
            "api": muse_api.api_root(pairing.get("api_url_v2", ""))}


def introduction_sent(directory: Path, *, vm_id: str, node_id: str) -> bool:
    expected = _receipt(directory, vm_id, node_id)
    for receipt in _receipts(directory):
        if all(receipt.get(key) == value for key, value in expected.items()):
            if receipt["status"] == "pending":
                raise _UncertainIntroduction(
                    "Reachy's introduction outcome is unknown. Check your main Muse chat "
                    "before resolving its pending receipt in reachy-introduction.json.")
            return True
    return False


async def introduce_once(session: LinkSession, directory: Path | None = None, *,
                         vm_id: str, node_id: str, stop: asyncio.Event) -> None:
    directory = directory or config.state_dir()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    # The CLI and a running service may opt in at the same time.
    with (directory / "reachy-introduction.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Reachy's introduction is already in progress") from exc
        if introduction_sent(directory, vm_id=vm_id, node_id=node_id):
            return
        receipt = _receipt(directory, vm_id, node_id)
        sending = asyncio.create_task(_send_and_consume(session, directory, receipt))
        try:
            done, _ = await asyncio.wait({sending}, timeout=INTRODUCTION_TIMEOUT_S)
            if sending not in done:
                raise asyncio.TimeoutError
            sending.result()
        except Exception as exc:
            # Network errors can contain reply text or credentials.
            raise RuntimeError("Reachy's introduction did not finish") from exc
        finally:
            if not sending.done():
                stop.set()
            await _cancel_tasks(sending)


def _retrieve(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()


async def _cancel_tasks(*tasks: asyncio.Task) -> None:
    for task in tasks:
        task.cancel()
        task.add_done_callback(_retrieve)
    await asyncio.wait(tasks, timeout=INTRODUCTION_CLEANUP_S)


async def _send_and_consume(session: LinkSession, directory: Path, receipt: dict) -> None:
    await session.wait_transport_ready()
    turn = _IntroductionTurn()
    armed = asyncio.Event()
    drain = asyncio.create_task(_consume_reply(session.subscribe_chat(), turn, armed))
    subscribed = asyncio.create_task(session.chat_subscribed.wait())
    try:
        done, _ = await asyncio.wait({drain, subscribed}, return_when=asyncio.FIRST_COMPLETED)
        if drain in done:
            drain.result()
        receipts = _receipts(directory)
        receipt["status"] = "pending"
        receipts.append(receipt)
        config.save_json(INTRODUCTION_STATE_FILE, {"receipts": receipts}, directory)
        armed.set()
        result = await session.send_chat(REACHY_INTRODUCTION)
        if result.get("ok") is False:
            receipts.remove(receipt)
            config.save_json(INTRODUCTION_STATE_FILE, {"receipts": receipts}, directory)
        turn.acknowledge(acknowledgement(result))
        receipt["status"] = "sent"
        config.save_json(INTRODUCTION_STATE_FILE, {"receipts": receipts}, directory)
        log.info("introduction sent version=%s", INTRODUCTION_VERSION)
        await drain
    finally:
        drain.cancel()
        subscribed.cancel()
        await asyncio.gather(drain, subscribed, return_exceptions=True)


async def _consume_reply(subscription, turn: MuseTurn, armed: asyncio.Event) -> None:
    pending = asyncio.create_task(subscription.__anext__())
    try:
        while True:
            now = time.monotonic()
            if turn.settled(now) or (any(reply.text.strip() for reply in turn.replies.values())
                                     and turn.complete(now)):
                break
            done, _ = await asyncio.wait({pending}, timeout=0.1)
            if pending not in done:
                continue
            try:
                event = pending.result()
            except StopAsyncIteration as exc:
                raise RuntimeError("Muse's introduction reply ended early") from exc
            if armed.is_set():
                turn.feed(event)
            pending = asyncio.create_task(subscription.__anext__())
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await subscription.aclose()


async def introduce_paired() -> None:
    """Use the saved pairing without starting robot hardware or voice playback."""
    pairing = config.load_json(config.PAIRING_FILE)
    if not pairing:
        raise ValueError("Reachy is not paired with Muse. Run `muse-reachy pair` on the robot first.")
    ident = identity.load_or_create()
    try:
        api = muse_api.api_root(pairing.get("api_url_v2", ""))
        vms, status = await asyncio.to_thread(
            muse_api.fetch_vms_with_status, pairing["access_token"], api)
        if status == 401:
            service = Service(identity=ident, executor=None, sdk_token=config.sdk_token())
            pairing = await service._maybe_refresh(pairing, force=True)
            if pairing is not None:
                vms, status = await asyncio.to_thread(
                    muse_api.fetch_vms_with_status, pairing["access_token"], api)
        if status != 200 or not vms:
            raise RuntimeError("Muse is unavailable")
        vm = next((vm for vm in vms if vm["is_default"]), vms[0])
        vm_id = vm["vm_id"] or vm["vm_name"]
        if introduction_sent(config.state_dir(), vm_id=vm_id, node_id=ident.node_id):
            return
        session = LinkSession(
            noise_host=pairing.get("noise_host") or DEFAULT_NOISE_HOST,
            vm_id=vm_id, vm_auth_token=vm["vm_auth_token"],
            device=DeviceDescription(ident.node_id, "Reachy Mini", __version__, COMMAND_SPECS),
            register_device=False,
            run_command=lambda *args: {"ok": False, "error": "Reachy is not running"})
        stop = asyncio.Event()
        connection = asyncio.create_task(session.run(stop))
        introduction = asyncio.create_task(introduce_once(
            session, vm_id=vm_id, node_id=ident.node_id, stop=stop))
        try:
            done, _ = await asyncio.wait({connection, introduction}, return_when=asyncio.FIRST_COMPLETED)
            if introduction not in done:
                raise RuntimeError("Muse disconnected before the introduction finished")
            introduction.result()
        finally:
            stop.set()
            await _cancel_tasks(introduction, connection)
    except _UncertainIntroduction:
        raise
    except Exception as exc:
        raise RuntimeError("Reachy's introduction did not finish") from exc
