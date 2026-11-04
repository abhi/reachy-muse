# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Introduce Reachy in the user's main Muse chat, once per text version."""

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
log = logging.getLogger(__name__)


def introduction_sent(directory: Path) -> bool:
    try:
        state = json.loads((directory / INTRODUCTION_STATE_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as exc:
        raise ValueError("Reachy's saved introduction could not be read") from exc
    if not isinstance(state, dict) or not isinstance(state.get("version"), str):
        raise ValueError("Reachy's saved introduction is invalid")
    return state["version"] == INTRODUCTION_VERSION


async def introduce_once(session: LinkSession, directory: Path | None = None) -> None:
    directory = directory or config.state_dir()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    # The CLI and a running service may opt in at the same time.
    with (directory / "reachy-introduction.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Reachy's introduction is already in progress") from exc
        if introduction_sent(directory):
            return
        try:
            await asyncio.wait_for(_send_and_consume(session, directory), INTRODUCTION_TIMEOUT_S)
        except Exception as exc:
            # Network errors can contain reply text or credentials.
            raise RuntimeError("Reachy's introduction did not finish") from exc


async def _send_and_consume(session: LinkSession, directory: Path) -> None:
    await session.wait_transport_ready()
    turn = MuseTurn()
    armed = asyncio.Event()
    drain = asyncio.create_task(_consume_reply(session.subscribe_chat(), turn, armed))
    subscribed = asyncio.create_task(session.chat_subscribed.wait())
    try:
        done, _ = await asyncio.wait({drain, subscribed}, return_when=asyncio.FIRST_COMPLETED)
        if drain in done:
            drain.result()
        armed.set()
        turn.acknowledge(acknowledgement(await session.send_chat(REACHY_INTRODUCTION)))
        # A lost reply must not cause another post on reconnect or restart.
        config.save_json(INTRODUCTION_STATE_FILE, {"version": INTRODUCTION_VERSION}, directory)
        log.info("introduction sent version=%s", INTRODUCTION_VERSION)
        await drain
    finally:
        drain.cancel()
        subscribed.cancel()
        await asyncio.gather(drain, subscribed, return_exceptions=True)


async def _consume_reply(subscription, turn: MuseTurn, armed: asyncio.Event) -> None:
    pending = asyncio.create_task(subscription.__anext__())
    try:
        while not (turn.replies and turn.complete(time.monotonic())):
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
    if introduction_sent(config.state_dir()):
        return
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
        session = LinkSession(
            noise_host=pairing.get("noise_host") or DEFAULT_NOISE_HOST,
            vm_id=vm["vm_id"] or vm["vm_name"], vm_auth_token=vm["vm_auth_token"],
            device=DeviceDescription(ident.node_id, "Reachy Mini", __version__, COMMAND_SPECS),
            register_device=False,
            run_command=lambda *args: {"ok": False, "error": "Reachy is not running"})
        stop = asyncio.Event()
        connection = asyncio.create_task(session.run(stop))
        introduction = asyncio.create_task(introduce_once(session))
        try:
            done, _ = await asyncio.wait({connection, introduction}, return_when=asyncio.FIRST_COMPLETED)
            if introduction not in done:
                raise RuntimeError("Muse disconnected before the introduction finished")
            introduction.result()
        finally:
            stop.set()
            introduction.cancel()
            connection.cancel()
            await asyncio.gather(introduction, connection, return_exceptions=True)
    except Exception as exc:
        raise RuntimeError("Reachy's introduction did not finish") from exc
