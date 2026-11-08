"""Notification admission, ordering and speech use local fakes, never a Muse connection."""

import asyncio
import logging
import os
import stat
import time

import pytest

from musegadget import config

from reachy_muse.device_command import WakeWindow
from reachy_muse.inbox import Inbox
from reachy_muse.notifications import Notifications
from reachy_muse.presence import Presence
from test_inbox import notice
from test_registered_commands import command_device, wait_until, registered_device

pytestmark = pytest.mark.usefixtures("paired_inbox")


@pytest.fixture(autouse=True)
def short_gesture(monkeypatch):
    monkeypatch.setattr("reachy_muse.notifications.ATTENTION_S", .01)


def device(tmp_path=None):
    service, voice, speech = command_device()
    if tmp_path is not None:
        service.inbox = Inbox(tmp_path / "inbox.json")
    voice._presence = service._presence
    voice.notifications = Notifications(service.inbox, service.presence_result, voice)
    return service, voice, speech


async def notify(service, **params):
    return await asyncio.to_thread(service.run_command, "reachy.notify", params)


def test_now_gestures_then_speaks_once_and_never_logs_text(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)

    async def check():
        service, voice, speech = device(tmp_path)
        service._presence.mark("voice", time.monotonic())
        params = {**notice(text="A private reminder."), "mode": "now"}
        expected = {"ok": True, "payload": {"status": "delivered", "reason": "spoken"}}
        assert await notify(service, **params) == expected
        assert voice.hardware.gestures == ["wave"]
        assert speech.words == ["A private reminder."]
        assert sum(len(samples) for samples in voice.hardware.played) == 320
        assert await notify(service, **params) == expected
        assert voice.hardware.gestures == ["wave"]
        assert speech.words == ["A private reminder."]
        assert voice.session.sent == voice.session.setup_messages == voice.session.tts_requests == []
        assert Inbox(tmp_path / "inbox.json").result("n1") == {
            "ok": True, "status": "delivered", "reason": "spoken"}
    asyncio.run(check())
    assert "A private reminder" not in caplog.text


def test_failed_claim_directory_sync_prevents_playback(tmp_path, monkeypatch):
    async def check():
        service, voice, _ = device(tmp_path)
        service._presence.mark("voice", time.monotonic())
        service.inbox.add(notice())
        fsync = os.fsync

        def sync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("directory sync failed")
            fsync(fd)

        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", sync)
            await voice.notifications.deliver(30)
        assert voice.hardware.played == []
        assert service.inbox.result("n1") == {
            "ok": True, "status": "refused", "reason": "delivery_uncertain"}
        assert service.inbox.add(notice("n2"))["status"] == "queued"
        await voice.notifications.deliver(30)
        assert sum(len(samples) for samples in voice.hardware.played) == 320
        assert service.inbox.result("n2") == {"ok": True, "status": "delivered", "reason": "spoken"}

    asyncio.run(check())


@pytest.mark.parametrize("blocked", ["absent", "asleep", "hushed", "turn", "hearing", "output", "wake", "realtime"])
def test_now_queues_until_presence_and_availability_allow_fifo_delivery(blocked):
    async def check():
        service, voice, speech = device()
        if blocked != "absent":
            service._presence.mark("voice", time.monotonic())
        if blocked == "asleep":
            voice._wake_window = WakeWindow.ASLEEP
        elif blocked == "hushed":
            voice._say_hushed = True
        elif blocked == "turn":
            voice._turn_active = True
        elif blocked == "hearing":
            voice._hearing_open = lambda: True
        elif blocked == "output":
            voice._output_busy = True
        elif blocked == "wake":
            voice._wake_first_request_pending = True
        elif blocked == "realtime":
            from types import SimpleNamespace
            voice.call = SimpleNamespace(audible=False, working=True)
        assert (await notify(service, **{**notice("a", "First."), "mode": "now"}))["payload"]["status"] == "queued"
        assert (await notify(service, **notice("b", "Second.", kind="suggestion")))["payload"]["status"] == "queued"
        await voice.notifications.deliver(30)
        assert speech.words == voice.hardware.gestures == []
        service._presence.mark("wave", time.monotonic())
        voice._wake_window = WakeWindow.AWAKE
        voice._say_hushed = voice._turn_active = voice._output_busy = voice._wake_first_request_pending = False
        voice._hearing_open = lambda: False
        voice.call = None
        await voice.notifications.deliver(30)
        assert speech.words == ["First."]
        assert service.inbox.result("b")["status"] == "queued"
        await voice.notifications.deliver(30)
        assert speech.words == ["First.", "Second."]
        assert voice.hardware.gestures == ["wave", "wave"]
    asyncio.run(check())


@pytest.mark.parametrize("source", ["camera", "face", "voice", "wave"])
def test_background_delivery_responds_to_each_presence_source(source):
    async def check():
        service, voice, speech = device()
        await notify(service, **notice())
        worker = asyncio.create_task(voice.notifications.run())
        try:
            await asyncio.sleep(.02)
            assert speech.words == []
            if source in ("face", "camera"):
                now = time.monotonic()
                voice.hardware.presence_observations = lambda: {source: Presence(True, now, now)}
            else:
                voice._observe_presence(source)
            await wait_until(lambda: service.inbox.result("n1")["status"] == "delivered")
            assert speech.words == ["Call Mom."]
        finally:
            voice._closed = True
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
    asyncio.run(check())


def test_turn_starting_during_the_attention_gesture_keeps_the_item_queued():
    async def check():
        service, voice, speech = device()
        service._presence.mark("voice", time.monotonic())
        voice.hardware.gesture = lambda _: setattr(voice, "_turn_active", True)
        assert await notify(service, **{**notice(), "mode": "now"}) == {
            "ok": True, "payload": {"status": "queued", "reason": "busy"}}
        assert speech.words == []
        assert service.inbox.pending()["id"] == "n1"
    asyncio.run(check())


def test_turn_starting_during_synthesis_defers_notification_until_available(tmp_path):
    from test_registered_commands import np

    async def check():
        entered, release = asyncio.Event(), asyncio.Event()

        class DelayedSpeech:
            async def stream(self, text, rate):
                entered.set()
                await release.wait()
                yield np.full(320, .1, dtype=np.float32)

        service, voice, _ = command_device(speech=DelayedSpeech())
        service.inbox = Inbox(tmp_path / "inbox.json")
        voice.notifications = Notifications(service.inbox, service.presence_result, voice)
        service._presence.mark("voice", time.monotonic())
        delivery = asyncio.create_task(notify(service, **{**notice(), "mode": "now"}))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            voice._turn_active = True
        finally:
            release.set()
        assert await delivery == {"ok": True, "payload": {"status": "queued", "reason": "busy"}}
        assert voice.hardware.played == []
        assert Inbox(tmp_path / "inbox.json").pending()["text"] == "Call Mom."
        await voice.notifications.deliver(30)
        assert voice.hardware.played == []
        voice._turn_active = False
        await voice.notifications.deliver(30)
        assert [samples.tolist() for samples in voice.hardware.played] == [
            np.full(320, .1, dtype=np.float32).tolist()]
        assert service.inbox.pending() is None
        assert service.inbox.result("n1") == {"ok": True, "status": "delivered", "reason": "spoken"}
        await voice.notifications.deliver(30)
        assert len(voice.hardware.played) == 1
    asyncio.run(check())


def test_expiry_during_gesture_drops_the_item_without_speech():
    async def check():
        service, voice, speech = device()
        service._presence.mark("voice", time.monotonic())
        assert await notify(service, **{**notice(ttl=.001), "mode": "now"}) == {
            "ok": True, "payload": {"status": "expired", "reason": "ttl_elapsed"}}
        assert speech.words == []
    asyncio.run(check())


@pytest.mark.parametrize("replace_during", ["gesture", "speech"])
def test_expired_delivery_cannot_consume_or_finish_a_reused_id(replace_during):
    async def check():
        service, voice, speech = device()
        now = [100]
        service.inbox.clock = lambda: now[0]
        service._presence.mark("voice", time.monotonic())
        service.inbox.add(notice(ttl=1))

        def replace():
            now[0] = 102
            assert service.inbox.pending() is None
            assert service.inbox.add(notice(text="Replacement."))["status"] == "queued"

        if replace_during == "gesture":
            voice.hardware.gesture = lambda _: replace()
        else:
            async def say(*args, **kwargs):
                replace()
                return {"spoken": True}
            voice.say = say
        await voice.notifications.deliver(30)
        assert service.inbox.result("n1") == {
            "ok": True, "status": "queued", "reason": "awaiting_presence"}
        assert service.inbox.pending()["text"] == "Replacement."
        assert speech.words == []
    asyncio.run(check())


@pytest.mark.parametrize("text", ["What time is it?", "", "volume up"])
def test_realtime_follow_up_refreshes_expired_wake_presence_and_releases_notifications(text):
    from types import SimpleNamespace
    from reachy_muse.capabilities import Endpoint
    from reachy_muse.conversation import _QueuedTurn, TurnOutcome
    from reachy_muse.turn_timeline import TurnTimeline

    async def check():
        service, voice, speech = device()
        service._presence.mark("voice", time.monotonic() - 31)
        service.inbox.add(notice())
        assert service.presence_result()["present"] is None
        call = SimpleNamespace(failed=False, audible=False, working=False,
                               speaker=SimpleNamespace(reply_started=None))
        timeline = TurnTimeline()
        timeline.mark("speech_end")
        request = _QueuedTurn(None, False, 0, timeline, (), call)
        assert await voice._realtime_turn(Endpoint(b"", text), request) is TurnOutcome.ACCEPTED
        assert service.presence_result()["present"] is True
        await voice.notifications.deliver(30)
        assert speech.words == ["Call Mom."]
    asyncio.run(check())


def test_concurrent_retries_and_worker_deliver_an_id_once():
    async def check():
        service, voice, speech = device()
        service._presence.mark("voice", time.monotonic())
        params = {**notice(), "mode": "now"}
        results = await asyncio.gather(*(notify(service, **params) for _ in range(8)))
        assert {result["payload"]["status"] for result in results} <= {"queued", "delivered"}
        await voice.notifications.deliver(30)
        assert speech.words == ["Call Mom."]
        assert service.inbox.result("n1") == {"ok": True, "status": "delivered", "reason": "spoken"}
    asyncio.run(check())


@pytest.mark.parametrize("change", [{"expires_in_s": 86401}, {"expires_in_s": True}, {"expires_in_s": float("nan")},
                                    {"text": ""}, {"text": "é" * 121}, {"id": ""}, {"id": "x" * 129},
                                    {"mode": "later"}, {"kind": "alarm"}, {"extra": "private"}])
def test_invalid_notification_returns_refused_without_synthesis(change):
    async def check():
        service, voice, speech = device()
        assert await notify(service, **{**notice(), **change}) == {
            "ok": True, "payload": {"status": "refused", "reason": "invalid_parameters"}}
        assert service.inbox.pending() is None
        assert speech.words == []
    asyncio.run(check())


def test_notification_is_registered_and_returns_the_sdk_wire_payload(monkeypatch):
    async def check():
        service, _, speech = device()
        async with registered_device(service, monkeypatch) as (session, vm, voice):
            service._presence.mark("voice", time.monotonic())
            await vm.send_message({"method": "link.invoke", "id": "command-1", "command": "reachy.notify",
                                   "params": {**notice(), "mode": "now"}})
            assert await asyncio.wait_for(vm.next_message(), 2) == {
                "method": "link.result", "id": "command-1", "ok": True,
                "payload": {"status": "delivered", "reason": "spoken"}}
            assert speech.words == ["Call Mom."]
    asyncio.run(check())


@pytest.mark.parametrize("during_gesture", [False, True])
def test_changed_credentials_discard_the_running_inbox(tmp_path, during_gesture):
    from reachy_muse.inbox import INBOX_FILE

    async def check():
        service, voice, speech = device()
        service.inbox = Inbox(tmp_path / INBOX_FILE)
        voice.notifications = Notifications(service.inbox, service.presence_result, voice)
        service._presence.mark("voice", time.monotonic())
        service.inbox.add(notice())

        def repair():
            config.save_json(config.PAIRING_FILE, {"username": "new", "access_token": "new-token"})

        if during_gesture:
            voice.hardware.gesture = lambda _: repair()
        else:
            repair()
        await voice.notifications.deliver(30)
        assert speech.words == []
        assert service.inbox.pending() is None
        assert not (tmp_path / INBOX_FILE).exists()
        replacement = Inbox(tmp_path / INBOX_FILE)
        assert replacement.add(notice(text="New owner."))["status"] == "queued"
        assert service.inbox.add(notice("old", "Old owner."))["status"] == "refused"
        assert Inbox(tmp_path / INBOX_FILE).pending()["text"] == "New owner."
    asyncio.run(check())


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("unpaired", [False, True])
def test_changed_credentials_stop_notification_pcm_already_playing(tmp_path, streaming, unpaired):
    from reachy_muse.inbox import INBOX_FILE
    from test_registered_commands import np

    closed = []

    class Speech:
        async def stream(self, text, rate):
            try:
                for _ in range(20 if streaming else 1):
                    yield np.full(320 if streaming else 6400, .1, dtype=np.float32)
            finally:
                closed.append(True)

    async def check():
        service, voice, _ = command_device(speech=Speech())
        service.inbox = Inbox(tmp_path / INBOX_FILE)
        voice.notifications = Notifications(service.inbox, service.presence_result, voice)
        service._presence.mark("voice", time.monotonic())
        play = voice.hardware.play_audio

        def repair_after_write(samples):
            play(samples)
            if len(voice.hardware.played) == 1:
                if unpaired:
                    config.delete_json(config.PAIRING_FILE)
                else:
                    config.save_json(config.PAIRING_FILE, {"username": "new", "access_token": "new-token"})

        voice.hardware.play_audio = repair_after_write
        await notify(service, **{**notice(), "mode": "now"})
        assert sum(len(samples) for samples in voice.hardware.played) == 320
        assert voice.hardware.cleared == 1
        assert closed == [True]
        assert not (tmp_path / INBOX_FILE).exists()
    asyncio.run(check())


def test_reconnect_with_same_credentials_keeps_the_queue(tmp_path, monkeypatch):
    from reachy_muse.pairing import PairedService

    async def check():
        service, _, speech = device(tmp_path)
        assert service.inbox.add(notice())["status"] == "queued"
        pairing = config.load_json(config.PAIRING_FILE)

        async def connected(self, vm, credentials):
            async with registered_device(self, monkeypatch) as (_, _, voice):
                assert self.inbox.pending()["text"] == "Call Mom."
                await voice.notifications.deliver(30)
                if vm["present"]:
                    assert self.inbox.result("n1") == {"ok": True, "status": "delivered", "reason": "spoken"}
                else:
                    assert self.inbox.result("n1") == {"ok": True, "status": "queued", "reason": "awaiting_presence"}

        monkeypatch.setattr(PairedService, "_session", connected)
        await service._session({"present": False}, pairing)
        service._presence.mark("voice", time.monotonic())
        await service._session({"present": True}, pairing)
        assert speech.words == ["Call Mom."]

    asyncio.run(check())


def test_reconnect_preserves_new_items_when_an_old_notification_expires(tmp_path, monkeypatch):
    import threading
    from reachy_muse.pairing import PairedService

    async def check():
        service, _, speech = device(tmp_path)
        now = [100]
        service.inbox.clock = lambda: now[0]
        service._presence.mark("voice", time.monotonic())
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        old_notifications = []

        def gesture(_):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(2)

        service.executor.gesture = gesture

        async def connected(self, vm, pairing):
            async with registered_device(self, monkeypatch) as (_, wire, voice):
                if not old_notifications:
                    old_notifications.append(voice.notifications)
                    await wire.send_message({"method": "link.invoke", "id": "old", "command": "reachy.notify",
                                             "params": {**notice("a", ttl=1), "mode": "now"}})
                    await asyncio.wait_for(entered.wait(), 2)
                else:
                    await wire.send_message({"method": "link.invoke", "id": "new", "command": "reachy.notify",
                                             "params": notice("b", "New note.")})
                    result = await asyncio.wait_for(wire.next_message(), 2)
                    assert result["payload"]["status"] == "queued"
                    now[0] = 102
                    release.set()
                    await wait_until(lambda: not old_notifications[0]._lock.locked())
                    assert Inbox(tmp_path / "inbox.json", clock=lambda: now[0]).result("b") == {
                        "ok": True, "status": "queued", "reason": "awaiting_presence"}
                    assert speech.words == []

        monkeypatch.setattr(PairedService, "_session", connected)
        try:
            await service._session({}, config.load_json(config.PAIRING_FILE))
            await service._session({}, config.load_json(config.PAIRING_FILE))
        finally:
            release.set()
    asyncio.run(check())


def test_obsolete_sdk_invocation_cannot_repopulate_an_unpaired_inbox(tmp_path, monkeypatch):
    async def check():
        service, _, speech = device(tmp_path)
        async with registered_device(service, monkeypatch) as (session, vm, voice):
            old_command = session._run_command
            config.delete_json(config.PAIRING_FILE)
        service.inbox = Inbox(tmp_path / "inbox.json")
        async with registered_device(service, monkeypatch):
            result = await asyncio.to_thread(old_command, "reachy.notify", notice())
            assert result == {"ok": True, "payload": {"status": "refused", "reason": "disconnected"}}
            assert service.inbox.pending() is None
            assert speech.words == []
            assert not (tmp_path / "inbox.json").exists()
    asyncio.run(check())


def test_repair_reconnect_binds_the_inbox_to_replacement_credentials(tmp_path, monkeypatch):
    from musegadget.pair import PairOutcome, PairResult
    from reachy_muse import cli
    from reachy_muse.inbox import INBOX_FILE
    from reachy_muse.pairing import PairedService

    old_pairing = {"username": "old", "access_token": "old-token"}
    monkeypatch.setenv(config.STATE_DIR_ENV, str(tmp_path))
    config.save_json(config.PAIRING_FILE, old_pairing)

    def pair(**kwargs):
        config.save_json(config.PAIRING_FILE, {"username": "new", "access_token": "new-token"})
        return PairResult(PairOutcome.PAIRED)

    async def connected(service, vm, pairing):
        async with registered_device(service, monkeypatch) as (session, _, voice):
            return await asyncio.to_thread(session._run_command, "reachy.notify", notice())

    monkeypatch.setattr("musegadget.pair.pair", pair)
    monkeypatch.setattr(PairedService, "_session", connected)

    async def check():
        service, _, _ = device()
        service.inbox = Inbox(tmp_path / INBOX_FILE)
        assert cli.pair_reachy(cli.parser().parse_args(["pair", "--force", "--no-introduce"])) == 0
        assert await service._session({}, old_pairing) == {
            "ok": True, "payload": {"status": "refused", "reason": "disconnected"}}
        assert await service._session({}, config.load_json(config.PAIRING_FILE)) == {
            "ok": True, "payload": {"status": "queued", "reason": "awaiting_presence"}}
    asyncio.run(check())


def test_unpair_during_notification_admission_cannot_restore_the_old_message(tmp_path, monkeypatch):
    import threading

    async def check():
        service, voice, speech = device(tmp_path)
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        add = service.inbox.add

        def delayed_add(*args, **kwargs):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(2)
            return add(*args, **kwargs)

        monkeypatch.setattr(service.inbox, "add", delayed_add)
        service._voice = None
        pending = asyncio.create_task(notify(service, **notice()))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            config.delete_json(config.PAIRING_FILE)
            release.set()
            assert await pending == {
                "ok": True, "payload": {"status": "refused", "reason": "disconnected"}}
            assert service.inbox.pending() is None
            assert not (tmp_path / "inbox.json").exists()
        finally:
            release.set()
            await pending
    asyncio.run(check())


def test_service_delivers_a_queued_note_after_a_voice_turn_finishes(monkeypatch):
    from test_device_command_turns import rig_with, service, run, played
    from test_pins import Transcripts, Wake
    from test_wave_conversation import continuous_microphone
    rig = rig_with(monkeypatch)
    continuous_microphone(rig.hardware)

    async def muse(session, request):
        assert "tts:Call Mom." not in played(rig.recording.events)
        await session.answer(request, "It is noon.")

    robot = service(rig, muse, Transcripts("Hey Muse, what time is it?"), wake_detector=Wake())
    robot.inbox.add(notice())

    async def user():
        assert played(rig.recording.events) == []
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: robot.inbox.result("n1")["status"] == "delivered", 5)

    events = run(rig, robot, user, rest=("idle", "listening"))
    assert played(events) == ["tts:It is noon.", "tts:Call Mom."]
    assert rig.hardware.gestures == ["wave"]


@pytest.mark.parametrize("after_gesture, status", [(None, "delivered"), (False, "queued")])
def test_gesture_may_age_a_camera_observation_but_a_fresh_absence_defers(after_gesture, status):
    async def check():
        service, voice, speech = device()
        observed = [True]
        voice.notifications.presence = lambda: {"present": observed[0]}
        voice.hardware.gesture = lambda _: observed.__setitem__(0, after_gesture)
        result = await notify(service, **{**notice(), "mode": "now"})
        assert result["payload"]["status"] == status
        assert speech.words == (["Call Mom."] if status == "delivered" else [])
    asyncio.run(check())


def test_failed_speech_is_not_replayed_and_does_not_log_its_text(caplog):
    class FailingSpeech:
        async def stream(self, text, rate):
            raise RuntimeError(text)
            yield

    async def check():
        service, voice, speech = command_device(speech=FailingSpeech())
        voice.notifications = Notifications(service.inbox, service.presence_result, voice)
        service._presence.mark("voice", time.monotonic())
        params = {**notice(text="A private failed reminder."), "mode": "now"}
        expected = {"ok": True, "payload": {"status": "refused", "reason": "speech_failed"}}
        assert await notify(service, **params) == expected
        assert await notify(service, **params) == expected
        assert voice.hardware.gestures == ["wave"]
        assert voice.hardware.played == []
    asyncio.run(check())
    assert "A private failed reminder." not in caplog.text


def test_shutdown_during_synthesis_stops_the_background_delivery_worker():
    async def check():
        entered = asyncio.Event()

        class HangingSpeech:
            async def stream(self, text, rate):
                entered.set()
                await asyncio.Event().wait()
                yield

        service, voice, _ = command_device(speech=HangingSpeech())
        voice.notifications = Notifications(service.inbox, service.presence_result, voice)
        service._presence.mark("voice", time.monotonic())
        service.inbox.add(notice())
        worker = asyncio.create_task(voice.notifications.run())
        await asyncio.wait_for(entered.wait(), 2)
        voice._closed = True
        worker.cancel()
        await asyncio.wait_for(worker, .5)
        assert voice.hardware.played == []
        assert service.inbox.result("n1") == {"ok": True, "status": "refused", "reason": "disconnected"}
    asyncio.run(check())
