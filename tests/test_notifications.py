"""Notification admission, ordering and speech use local fakes, never a Muse connection."""

import asyncio
import logging
import time

import pytest

from reachy_muse.device_command import WakeWindow
from reachy_muse.inbox import Inbox
from reachy_muse.notifications import Notifications
from reachy_muse.presence import Presence
from test_inbox import notice
from test_registered_commands import command_device, wait_until, registered_device


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


@pytest.mark.parametrize("blocked", ["absent", "asleep", "hushed", "turn", "hearing", "output", "wake"])
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
        assert (await notify(service, **{**notice("a", "First."), "mode": "now"}))["payload"]["status"] == "queued"
        assert (await notify(service, **notice("b", "Second.", kind="suggestion")))["payload"]["status"] == "queued"
        await voice.notifications.deliver(30)
        assert speech.words == voice.hardware.gestures == []
        service._presence.mark("wave", time.monotonic())
        voice._wake_window = WakeWindow.AWAKE
        voice._say_hushed = voice._turn_active = voice._output_busy = voice._wake_first_request_pending = False
        voice._hearing_open = lambda: False
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


def test_expiry_during_gesture_drops_the_item_without_speech():
    async def check():
        service, voice, speech = device()
        service._presence.mark("voice", time.monotonic())
        assert await notify(service, **{**notice(ttl=.001), "mode": "now"}) == {
            "ok": True, "payload": {"status": "expired", "reason": "ttl_elapsed"}}
        assert speech.words == []
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


def test_unpair_event_clears_queued_and_delivered_ids(tmp_path, monkeypatch):
    async def check():
        service, _, _ = device(tmp_path)
        async with registered_device(service, monkeypatch) as (session, vm, voice):
            await notify(service, **notice())
            await vm.send_message({"type": "event", "event": "link.unpaired"})
            await wait_until(lambda: not (tmp_path / "inbox.json").exists())
            assert service.inbox.pending() is None
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
