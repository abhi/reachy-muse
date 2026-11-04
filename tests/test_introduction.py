# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
import hashlib
import json
import logging

import pytest

from musegadget import config, muse_turn
from musegadget.identity import Identity
from musegadget.link_client import LinkSession, Outcome
from musegadget.noise import ApplicationResponse, BodyChunk, ServiceFrame

from reachy_muse import cli, introduction
from reachy_muse.contract.prompts import REACHY_INTRODUCTION
from reachy_muse.conversation import ReachyService
from fake_muse_vm import make_session, registered_session


def event(name, **payload):
    payload = {"session_id": "main-chat", "reply_to_message_id": "intro-user", **payload}
    if payload["reply_to_message_id"] is None:
        del payload["reply_to_message_id"]
    return {"type": "event", "event": name, "payload": payload}


class Session:
    def __init__(self, *, ack=None, reply=True):
        self.registered = asyncio.Event()
        self.registered.set()
        self.chat_subscribed = asyncio.Event()
        self.posted = asyncio.Event()
        self.events = asyncio.Queue()
        self.posts = []
        self.closed = False
        self.connection_closed = False
        self.reply = reply
        self.ack = ack if ack is not None else {
            "ok": True, "response": {"session_id": "main-chat", "message_id": "intro-user"}}

    async def run(self, stop):
        try:
            await stop.wait()
            return Outcome.STOPPED
        finally:
            self.connection_closed = True

    async def wait_transport_ready(self):
        await self.registered.wait()

    async def send_chat(self, message):
        assert self.chat_subscribed.is_set()
        self.posts.append(message)
        self.posted.set()
        if self.reply:
            self.finish_reply()
        # Exercise replies arriving before the POST acknowledgement.
        await asyncio.sleep(0)
        return self.ack

    def finish_reply(self):
        self.events.put_nowait(event("task.status", task_id="intro-task", status="running"))
        self.events.put_nowait(event("delta.message_done", message_id="intro-reply",
                                     display_text="Private reply that must never be spoken or logged."))
        self.events.put_nowait(event("task.status", task_id="intro-task", status="completed"))

    async def subscribe_chat(self):
        self.chat_subscribed.set()
        try:
            while True:
                item = await self.events.get()
                if item is None:
                    return
                yield item
        finally:
            self.chat_subscribed.clear()
            self.closed = True


def test_exact_introduction_and_character_budget():
    assert REACHY_INTRODUCTION == (
        "I have a Reachy Mini robot at home that talks with you by voice. "
        "Reachy is one of my persistent surfaces, like the Muse app. "
        "When I talk through Reachy, your replies are spoken aloud, so keep them short. "
        "You can speak to me in the room with reachy.say and check whether someone is there "
        "with reachy.presence, for example to deliver a reminder when I'm home. Please remember this."
    )
    assert len(REACHY_INTRODUCTION) == 379
    assert len(REACHY_INTRODUCTION) < 600
    assert introduction.INTRODUCTION_VERSION == hashlib.sha256(REACHY_INTRODUCTION.encode()).hexdigest()


def test_preview_needs_no_pairing_token_state_or_hardware(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("preview must not read credentials, connect, or start hardware")

    monkeypatch.setattr(config, "load_json", forbidden)
    monkeypatch.setattr(cli.identity, "load_or_create", forbidden)
    monkeypatch.setattr(cli, "_hardware", forbidden)
    monkeypatch.setattr(introduction, "introduce_paired", forbidden)
    assert cli.main(["--state-dir", str(tmp_path), "--sdk-token-file", str(tmp_path / "absent"),
                     "introduce"]) == 0
    assert capsys.readouterr().out == (
        REACHY_INTRODUCTION + "\n\nRun `muse-reachy introduce --yes` to send this to your main Muse chat.\n")
    assert list(tmp_path.iterdir()) == []
    assert cli.parser().parse_args(["run"]).introduce_to_muse is False


def test_sent_once_across_restarts_and_again_for_new_text(tmp_path, monkeypatch, caplog):
    async def check():
        first = Session()
        await introduction.introduce_once(first, tmp_path)
        assert first.posts == [REACHY_INTRODUCTION]
        assert first.closed and first.events.empty()
        state_file = tmp_path / introduction.INTRODUCTION_STATE_FILE
        assert json.loads(state_file.read_text()) == {"version": introduction.INTRODUCTION_VERSION}
        assert state_file.stat().st_mode & 0o777 == 0o600
        reconnect = Session()
        await introduction.introduce_once(reconnect, tmp_path)
        assert reconnect.posts == []
        changed = REACHY_INTRODUCTION + " Thank you."
        monkeypatch.setattr(introduction, "REACHY_INTRODUCTION", changed)
        monkeypatch.setattr(introduction, "INTRODUCTION_VERSION", hashlib.sha256(changed.encode()).hexdigest())
        updated = Session()
        await introduction.introduce_once(updated, tmp_path)
        assert updated.posts == [changed]
        await introduction.introduce_once(updated, tmp_path)
        assert updated.posts == [changed]

    original_version = introduction.INTRODUCTION_VERSION
    with caplog.at_level(logging.INFO, logger=introduction.__name__):
        asyncio.run(check())
    assert caplog.messages == [f"introduction sent version={original_version}",
                               f"introduction sent version={introduction.INTRODUCTION_VERSION}"]


def test_waits_for_registration_and_consumes_whole_reply(tmp_path):
    async def check():
        session = Session(reply=False)
        session.registered.clear()
        task = asyncio.create_task(introduction.introduce_once(session, tmp_path))
        await asyncio.sleep(0)
        assert session.posts == []
        session.registered.set()
        await session.posted.wait()
        session.events.put_nowait(event("task.status", task_id="intro-task", status="running"))
        session.events.put_nowait(event("delta.text_append", message_id="intro-reply", text="Private partial"))
        await asyncio.sleep(0.02)
        assert not task.done()
        assert introduction.introduction_sent(tmp_path)
        session.finish_reply()
        await task
        assert session.closed and session.events.empty()

    asyncio.run(asyncio.wait_for(check(), 2))


@pytest.mark.parametrize("ack", [
    {"ok": False, "status": 500, "response": {"secret": "private"}},
    {"ok": True, "response": None},
    {"ok": True, "response": {"session_id": "main-chat"}},
])
def test_rejected_or_invalid_ack_does_not_record_success(tmp_path, ack):
    session = Session(ack=ack)
    with pytest.raises(RuntimeError, match="^Reachy's introduction did not finish$"):
        asyncio.run(introduction.introduce_once(session, tmp_path))
    assert not (tmp_path / introduction.INTRODUCTION_STATE_FILE).exists()
    assert session.closed
    retry = Session()
    asyncio.run(introduction.introduce_once(retry, tmp_path))
    assert retry.posts == [REACHY_INTRODUCTION]


@pytest.mark.parametrize("failure", ["disconnect", "timeout", "cancel"])
def test_accepted_post_is_not_repeated_if_reply_is_lost(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(introduction, "INTRODUCTION_TIMEOUT_S", 0.1)

    async def check():
        session = Session(reply=False)
        task = asyncio.create_task(introduction.introduce_once(session, tmp_path))
        await session.posted.wait()
        await asyncio.sleep(0.02)
        assert introduction.introduction_sent(tmp_path)
        if failure == "disconnect":
            session.events.put_nowait(None)
        elif failure == "cancel":
            task.cancel()
        with pytest.raises(asyncio.CancelledError if failure == "cancel" else RuntimeError):
            await task
        assert session.closed
        retry = Session()
        await introduction.introduce_once(retry, tmp_path)
        assert retry.posts == []

    asyncio.run(asyncio.wait_for(check(), 2))


def test_concurrent_introductions_cannot_both_post(tmp_path):
    async def check():
        first = Session(reply=False)
        task = asyncio.create_task(introduction.introduce_once(first, tmp_path))
        await first.posted.wait()
        second = Session()
        with pytest.raises(RuntimeError, match="already in progress"):
            await introduction.introduce_once(second, tmp_path)
        assert second.posts == []
        first.finish_reply()
        await task
        await introduction.introduce_once(second, tmp_path)
        assert second.posts == []

    asyncio.run(asyncio.wait_for(check(), 2))


@pytest.mark.parametrize("saved", ["invalid json", "[]", '{"version": 42}'])
def test_corrupt_state_fails_without_posting(tmp_path, saved):
    (tmp_path / introduction.INTRODUCTION_STATE_FILE).write_text(saved)
    session = Session()
    with pytest.raises(ValueError, match="saved introduction"):
        asyncio.run(introduction.introduce_once(session, tmp_path))
    assert session.posts == []


def test_cli_yes_uses_pairing_without_robot_and_deduplicates(tmp_path, monkeypatch, capsys):
    sessions = []

    def connect(**kwargs):
        assert kwargs["vm_id"] == "default-vm"
        assert kwargs["run_command"]("reachy.say", {"text": "Do not speak."}) == {
            "ok": False, "error": "Reachy is not running"}
        session = Session()
        sessions.append(session)
        return session

    monkeypatch.setenv(config.STATE_DIR_ENV, str(tmp_path))
    config.save_json(config.PAIRING_FILE, {"access_token": "fake-token"}, tmp_path)
    monkeypatch.setattr(cli.identity, "load_or_create", lambda: Identity("02:00:00:ab:cd:ef"))
    monkeypatch.setattr(cli, "_hardware", lambda *args: pytest.fail("introduction started hardware"))
    monkeypatch.setattr(introduction.muse_api, "fetch_vms_with_status", lambda *args: ([
        {"vm_id": "other-vm", "is_default": False},
        {"vm_id": "default-vm", "is_default": True, "vm_auth_token": "fake-vm-token"}], 200))
    monkeypatch.setattr(introduction, "LinkSession", connect)
    for _ in range(2):
        assert cli.main(["--state-dir", str(tmp_path), "introduce", "--yes"]) == 0
    assert len(sessions) == 1
    assert sessions[0].posts == [REACHY_INTRODUCTION]
    assert sessions[0].closed and sessions[0].connection_closed
    assert capsys.readouterr().out == (REACHY_INTRODUCTION + "\n") * 2


def test_unpaired_yes_does_not_connect(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(introduction, "LinkSession", lambda **kwargs: pytest.fail("unpaired connection"))
    assert cli.main(["--state-dir", str(tmp_path), "introduce", "--yes"]) == 1
    assert "not paired with Muse" in capsys.readouterr().err


def test_standalone_introduction_preserves_running_device_registration(tmp_path, monkeypatch):
    monkeypatch.setenv(config.STATE_DIR_ENV, str(tmp_path))
    config.save_json(config.PAIRING_FILE, {"access_token": "fake-token"}, tmp_path)
    monkeypatch.setattr(introduction.identity, "load_or_create", lambda: Identity("02:00:00:ab:cd:ef"))
    monkeypatch.setattr(introduction.muse_api, "fetch_vms_with_status", lambda *args: ([
        {"vm_id": "vm-1", "is_default": True, "vm_auth_token": "fake-vm-token"}], 200))

    async def check():
        running, running_vm, stop, connection = await registered_session()
        link, vm = make_session(lambda *args: pytest.fail("standalone command"), [])
        monkeypatch.setattr(introduction, "LinkSession", lambda **kwargs: LinkSession(
            **kwargs, connect=link._connect))
        task = asyncio.create_task(introduction.introduce_paired())
        try:
            await vm.handshake()
            subscribe = await vm.next_frame()
            assert subscribe.value.path == "/chat/subscribe"
            await vm.send_frame(ServiceFrame.response(subscribe.stream_id, ApplicationResponse(
                status=200, end_body=False)))
            post = await vm.next_frame()
            assert json.loads(post.value.body)["message"] == REACHY_INTRODUCTION
            await vm.send_frame(ServiceFrame.response(post.stream_id, ApplicationResponse(
                status=200, body=b'{"session_id":"main-chat","message_id":"intro-user"}', end_body=True)))
            await vm.send_frame(ServiceFrame.body_chunk(subscribe.stream_id, BodyChunk(data=b"".join(
                json.dumps(reply).encode() + b"\n" for reply in [
                    event("delta.message_done", message_id="intro-reply", display_text="Remembered."),
                    event("task.status", status="completed")]))))
            await task
            assert running.registered.is_set()
            await running_vm.send_message({"method": "link.invoke", "id": "still-running",
                                           "command": "system.run", "params": {}})
            result = await running_vm.next_message()
            assert result["id"] == "still-running" and result["ok"] is True
        finally:
            stop.set()
            task.cancel()
            await asyncio.gather(task, connection, return_exceptions=True)

    asyncio.run(asyncio.wait_for(check(), 3))


def test_standalone_refreshes_rejected_access_token(tmp_path, monkeypatch):
    monkeypatch.setenv(config.STATE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(config, "sdk_token", lambda: "fake-sdk-token")
    config.save_json(config.PAIRING_FILE, {
        "access_token": "expired", "refresh_token": "valid-refresh", "username": "paired-user"}, tmp_path)
    monkeypatch.setattr(introduction.identity, "load_or_create", lambda: Identity("02:00:00:ab:cd:ef"))
    fetched, refreshed, sessions = [], [], []

    def fetch(token, api):
        fetched.append(token)
        return ([], 401) if token == "expired" else ([
            {"vm_id": "vm-1", "is_default": True, "vm_auth_token": "fresh-vm-token"}], 200)

    def refresh(token, node_id, api, sdk_token):
        refreshed.append((token, node_id, sdk_token))
        return {"access_token": "fresh-access", "refresh_token": "fresh-refresh"}, 200

    def connect(**kwargs):
        assert kwargs["vm_auth_token"] == "fresh-vm-token"
        session = Session()
        sessions.append(session)
        return session

    monkeypatch.setattr(introduction.muse_api, "fetch_vms_with_status", fetch)
    monkeypatch.setattr(introduction.muse_api, "refresh_device_token", refresh)
    monkeypatch.setattr(introduction, "LinkSession", connect)
    asyncio.run(introduction.introduce_paired())
    assert fetched == ["expired", "fresh-access"]
    assert refreshed == [("valid-refresh", "homelink-abcdef", "fake-sdk-token")]
    saved = config.load_json(config.PAIRING_FILE, tmp_path)
    assert saved["access_token"] == "fresh-access"
    assert saved["refresh_token"] == "fresh-refresh"
    assert saved["username"] == "paired-user"
    assert saved["access_token_saved_at"] > 0
    assert sessions[0].posts == [REACHY_INTRODUCTION]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("selected", [None, "explicit-chat", "dedicated-chat"])
def test_service_consumes_introduction_before_voice_on_every_route(tmp_path, monkeypatch, enabled, selected):
    import reachy_muse.conversation as conversation

    monkeypatch.setenv(config.STATE_DIR_ENV, str(tmp_path))

    async def check():
        session = Session()
        stop = asyncio.Event()
        calls = []

        class Voice:
            def __init__(self, link, hardware, **kwargs):
                assert link is session
                assert session.posts == ([REACHY_INTRODUCTION] if enabled else [])
                assert session.closed is enabled
                assert session.events.empty()
                calls.append(kwargs["session_id"])

            async def run(self):
                stop.set()

        async def prepare(link, vm_id):
            assert session.closed is enabled
            return "dedicated-chat"

        monkeypatch.setattr(conversation, "VoiceConversation", Voice)
        service = ReachyService(
            identity=Identity("02:00:00:ab:cd:ef"), executor=object(), backends=lambda _: None,
            session_id=selected, owns_chat=selected == "dedicated-chat",
            prepare_chat=prepare if selected == "dedicated-chat" else None,
            introduce_to_muse=enabled)
        assert await service.serve(session, {"vm_id": "vm-1"}, stop) is Outcome.STOPPED
        assert calls == [selected]
        assert (tmp_path / introduction.INTRODUCTION_STATE_FILE).exists() is enabled

    asyncio.run(asyncio.wait_for(check(), 2))


def test_real_sdk_posts_text_to_main_chat_and_closes_reply_subscription(tmp_path):
    async def check():
        session, vm, stop, connection = await registered_session()
        task = asyncio.create_task(introduction.introduce_once(session, tmp_path))
        try:
            subscribe = await vm.next_frame()
            assert subscribe.value.path == "/chat/subscribe"
            assert json.loads(subscribe.value.body) == {}
            await vm.send_frame(ServiceFrame.response(subscribe.stream_id, ApplicationResponse(
                status=200, end_body=False)))
            post = await vm.next_frame()
            assert json.loads(post.value.body) == {
                "message": REACHY_INTRODUCTION, "output_modality": "text", "device_id": "homelink-abcdef"}
            replies = [event("message.user", message_id="intro-user", reply_to_message_id=None),
                       event("task.status", task_id="intro-task", status="running", reply_to_message_id=None),
                       event("delta.message_done", message_id="intro-reply", display_text="I'll remember.",
                             reply_to_message_id=None),
                       event("task.status", task_id="intro-task", status="completed", reply_to_message_id=None)]
            await vm.send_frame(ServiceFrame.body_chunk(subscribe.stream_id, BodyChunk(
                data=b"".join(json.dumps(reply).encode() + b"\n" for reply in replies))))
            await vm.send_frame(ServiceFrame.response(post.stream_id, ApplicationResponse(
                status=200, body=json.dumps({"session_id": "main-chat", "message_id": "intro-user"}).encode(),
                end_body=True)))
            await task
            reset = await vm.next_frame()
            assert reset.kind == "reset" and reset.stream_id == subscribe.stream_id
            assert not session.chat_subscribed.is_set()
            assert introduction.introduction_sent(tmp_path)
        finally:
            stop.set()
            task.cancel()
            await asyncio.gather(task, connection, return_exceptions=True)

    asyncio.run(asyncio.wait_for(check(), 3))


@pytest.mark.parametrize("foreign_turn", ["earlier", "later"])
def test_real_sdk_parentless_replies_follow_the_acknowledged_user(tmp_path, foreign_turn):
    async def check():
        session, vm, stop, connection = await registered_session()
        task = asyncio.create_task(introduction.introduce_once(session, tmp_path))

        async def publish(*events):
            await vm.send_frame(ServiceFrame.body_chunk(subscribe.stream_id, BodyChunk(
                data=b"".join(json.dumps(item).encode() + b"\n" for item in events))))

        try:
            subscribe = await vm.next_frame()
            await vm.send_frame(ServiceFrame.response(subscribe.stream_id, ApplicationResponse(
                status=200, end_body=False)))
            post = await vm.next_frame()
            if foreign_turn == "later":
                await publish(event("message.user", message_id="intro-user", reply_to_message_id=None))
            await publish(
                event("message.user", message_id="phone-user", reply_to_message_id=None),
                event("message.assistant", message_id="phone-reply", display_text="A phone answer.",
                      reply_to_message_id=None),
                event("task.status", task_id="phone-task", status="completed", reply_to_message_id=None))
            await vm.send_frame(ServiceFrame.response(post.stream_id, ApplicationResponse(
                status=200, body=b'{"session_id":"main-chat","message_id":"intro-user"}', end_body=True)))
            await asyncio.sleep(0.15)
            assert not task.done(), "another user's reply closed the introduction subscription"
            if foreign_turn == "earlier":
                await publish(event("message.user", message_id="intro-user", reply_to_message_id=None))
            parent = None if foreign_turn == "earlier" else "intro-user"
            await publish(
                event("task.status", task_id="intro-task", status="running", reply_to_message_id=parent),
                event("message.assistant", message_id="intro-reply", display_text="Remembered.",
                      reply_to_message_id=parent),
                event("task.status", task_id="intro-task", status="completed", reply_to_message_id=parent))
            await task
            reset = await vm.next_frame()
            assert reset.kind == "reset" and reset.stream_id == subscribe.stream_id
        finally:
            stop.set()
            task.cancel()
            await asyncio.gather(task, connection, return_exceptions=True)

    asyncio.run(asyncio.wait_for(check(), 3))


def test_terminal_task_without_assistant_message_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(muse_turn, "REPLY_QUIET_S", 0.02)

    async def check():
        session = Session(reply=False)
        task = asyncio.create_task(introduction.introduce_once(session, tmp_path))
        try:
            await session.posted.wait()
            session.events.put_nowait(event("task.status", task_id="intro-task", status="completed"))
            await asyncio.wait({task}, timeout=0.5)
            assert ("finished" if task.done() else "waiting") == "finished"
            await task
            assert session.closed and session.events.empty()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(check())


def test_blank_completion_waits_for_reordered_assistant_text(tmp_path):
    async def check():
        session = Session(reply=False)
        task = asyncio.create_task(introduction.introduce_once(session, tmp_path))
        try:
            await session.posted.wait()
            session.events.put_nowait(event("delta.message_done", message_id="intro-reply", display_text=""))
            session.events.put_nowait(event("task.status", task_id="intro-task", status="completed"))
            await asyncio.sleep(0.15)
            session.events.put_nowait(event("message.assistant", message_id="intro-reply",
                                           display_text="The final text arrived after task completion."))
            await task
            assert session.events.qsize() == 0
            assert session.closed
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(asyncio.wait_for(check(), 2))
