# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
import logging

import pytest

from musegadget.identity import Identity
from reachy_muse.capabilities import Mode
from reachy_muse.device_command import WakeWindow
from reachy_muse.conversation import ReachyService, VoiceConversation
from reachy_muse.local_backends import backends_for
from fake_muse_vm import make_session
from test_conversation import FakeHardware, FakeSession

np = pytest.importorskip("numpy")


class Speech:
    def __init__(self):
        self.words = []

    async def stream(self, text, rate):
        self.words.append(text)
        yield np.full(320, .1, dtype=np.float32)


def command_device(*, mode=Mode.ON_ROBOT, speech=None):
    session, hardware = FakeSession(), FakeHardware()
    speech = speech or Speech()
    bundle = backends_for(mode, session, speech=speech)
    voice = VoiceConversation(session, hardware, backends=bundle)
    voice._ready.set()
    service = ReachyService(identity=Identity("02:00:00:ab:cd:ef"), executor=hardware,
                           backends=lambda _: bundle)
    service._voice = (asyncio.get_running_loop(), voice)
    return service, voice, speech


async def invoke(service, text="Lunch is ready.", timeout_ms=None):
    return await asyncio.to_thread(service.run_command, "reachy.say", {"text": text}, timeout_ms)


async def wait_until(predicate):
    async def waiting():
        while not predicate():
            await asyncio.sleep(.001)
    await asyncio.wait_for(waiting(), 2)


@pytest.mark.parametrize("command, hushed, expected", [
    ("reachy.say", False, {"method": "link.result", "id": "command-1", "ok": True,
                           "payload": {"spoken": True}}),
    ("reachy.say", True, {"method": "link.result", "id": "command-1", "ok": True,
                          "payload": {"spoken": False, "reason": "hushed"}}),
    ("reachy.presence", False, {"method": "link.result", "id": "command-1", "ok": True,
                                "payload": {"present": True, "last_seen_seconds": 0}}),
])
def test_command_results_reach_the_sdk_wire_payload(command, hushed, expected):
    async def check():
        service, voice, _ = command_device()
        voice.hardware.presence = lambda: {"ok": True, "present": True, "last_seen_seconds": 0}
        if hushed:
            voice._hush()
        session, vm = make_session(service.run_command, [])
        stop = asyncio.Event()
        running = asyncio.create_task(session.run(stop))
        try:
            await vm.handshake()
            await vm.accept_control_stream()
            register = await vm.next_message()
            await vm.send_message({"type": "res", "id": register["id"], "ok": True})
            await session.registered.wait()
            await vm.send_message({"method": "link.invoke", "id": "command-1", "command": command,
                                   "params": {"text": "Lunch is ready."} if command == "reachy.say" else {}})
            assert await asyncio.wait_for(vm.next_message(), 2) == expected
        finally:
            stop.set()
            await running

    asyncio.run(check())


def test_registered_say_uses_normal_playback_without_muse_or_text_logs(caplog):
    caplog.set_level(logging.INFO)

    async def check():
        service, voice, speech = command_device()
        result = await invoke(service, "A private reminder.")
        assert result == {"ok": True, "payload": {"spoken": True}}
        assert speech.words == ["A private reminder."]
        assert sum(len(samples) for samples in voice.hardware.played) == 320
        assert voice.hardware.play_expressions == [("speaking", None)]
        assert voice.session.sent == voice.session.setup_messages == voice.session.tts_requests == []
        assert voice.hardware.states[-1] == "idle"
        assert not voice._command_tasks

    asyncio.run(check())
    assert "A private reminder" not in caplog.text
    assert [record.message for record in caplog.records if "Reachy command" in record.message] == [
        "Reachy command name=reachy.say outcome=spoken"]


@pytest.mark.parametrize("blocked", ["asleep", "hushed", "busy", "hearing", "user_speaking", "not_ready", "disconnected"])
def test_say_refuses_before_synthesis(blocked):
    async def check():
        service, voice, speech = command_device()
        if blocked == "asleep":
            voice._wake_window = WakeWindow.ASLEEP
        elif blocked == "hushed":
            voice._hush()
        elif blocked == "busy":
            voice._turn_active = True
        elif blocked == "hearing":
            voice._hearing_open = lambda: True
        elif blocked == "user_speaking":
            voice._playback.set_user_speaking(True)
        elif blocked == "not_ready":
            voice._ready.clear()
        else:
            service._voice = None
        reason = "busy" if blocked in ("hearing", "user_speaking") else blocked
        assert await invoke(service) == {"ok": True, "payload": {"spoken": False, "reason": reason}}
        assert speech.words == voice.hardware.played == []

    asyncio.run(check())


def test_say_from_a_muse_turn_refuses_instead_of_waiting_for_its_caller():
    async def check():
        service, voice, speech = command_device()
        voice.job = object()
        assert await invoke(service) == {"ok": True, "payload": {"spoken": False, "reason": "busy"}}
        assert speech.words == []
    asyncio.run(check())


@pytest.mark.parametrize("change, expected", [(None, {"ok": True, "payload": {"spoken": True}}),
    ("sleep", {"ok": True, "payload": {"spoken": False, "reason": "asleep"}}),
    ("hush", {"ok": True, "payload": {"spoken": False, "reason": "hushed"}}),
    ("turn", {"ok": True, "payload": {"spoken": False, "reason": "busy"}})])
def test_say_waits_for_current_line_and_rechecks_admission(change, expected):
    async def check():
        service, voice, speech = command_device()
        await voice._speaker_lock.acquire()
        pending = asyncio.create_task(invoke(service))
        await wait_until(lambda: voice._speech_pending == 1)
        assert not pending.done() and speech.words == []
        if change == "sleep":
            voice._wake_window = WakeWindow.ASLEEP
        elif change == "hush":
            voice._hush()
        elif change == "turn":
            voice._turn_active = True
        voice._speaker_lock.release()
        assert await pending == expected
        assert speech.words == (["Lunch is ready."] if change is None else [])
    asyncio.run(check())


def test_waiting_say_times_out_and_never_plays_later():
    async def check():
        service, voice, speech = command_device()
        async with voice._speaker_lock:
            assert await invoke(service, timeout_ms=20) == {"ok": True, "payload": {"spoken": False, "reason": "timeout"}}
        assert voice._speech_pending == 0
        assert speech.words == voice.hardware.played == []
    asyncio.run(check())


def test_hush_then_wake_cannot_revive_a_command_waiting_for_its_speaker_task():
    async def check():
        service, voice, speech = command_device()
        queued, resume = asyncio.Event(), asyncio.Event()
        speak = voice._speak

        async def scheduled(*args, **kwargs):
            queued.set()
            await resume.wait()
            await speak(*args, **kwargs)

        voice._speak = scheduled
        pending = asyncio.create_task(invoke(service))
        await queued.wait()
        voice._hush()
        await voice._wake_up(WakeWindow.AWAKE)
        resume.set()
        assert await pending == {"ok": True, "payload": {"spoken": False, "reason": "interrupted"}}
        assert speech.words == voice.hardware.played == []
    asyncio.run(check())


def test_hush_stops_playback_and_refuses_new_say_until_wake():
    async def check():
        resume = asyncio.Event()

        class LongSpeech(Speech):
            async def stream(self, text, rate):
                self.words.append(text)
                yield np.full(3200, .1, dtype=np.float32)
                await resume.wait()
                yield np.full(320, .2, dtype=np.float32)

        service, voice, speech = command_device(speech=LongSpeech())
        pending = asyncio.create_task(invoke(service))
        await wait_until(lambda: bool(voice.hardware.played))
        voice._hush()
        resume.set()
        assert await pending == {"ok": True, "payload": {"spoken": False, "reason": "hushed"}}
        assert voice.hardware.cleared == 1
        assert all(float(samples.max()) < .15 for samples in voice.hardware.played)
        assert await invoke(service) == {"ok": True, "payload": {"spoken": False, "reason": "hushed"}}
        await voice._wake_up(WakeWindow.AWAKE)
        assert await invoke(service) == {"ok": True, "payload": {"spoken": True}}
    asyncio.run(check())


def test_cancelled_command_flushes_output_and_reports_disconnection():
    async def check():
        closed = asyncio.Event()

        class StalledSpeech(Speech):
            async def stream(self, text, rate):
                try:
                    yield np.full(3200, .1, dtype=np.float32)
                    await asyncio.Event().wait()
                finally:
                    closed.set()

        service, voice, _ = command_device(speech=StalledSpeech())
        pending = asyncio.create_task(invoke(service))
        await wait_until(lambda: bool(voice.hardware.played))
        for task in tuple(voice._command_tasks):
            task.cancel()
        assert await pending == {"ok": True, "payload": {"spoken": False, "reason": "disconnected"}}
        assert closed.is_set() and voice.hardware.cleared == 1
        assert not voice._command_tasks
    asyncio.run(check())


def test_muse_voice_cannot_synthesize_command_text_and_never_contacts_muse():
    async def check():
        service, voice, _ = command_device(mode=Mode.MUSE_VOICE)
        assert await invoke(service) == {"ok": True, "payload": {"spoken": False, "reason": "speech_unavailable"}}
        assert voice.session.sent == voice.session.setup_messages == voice.session.tts_requests == []
    asyncio.run(check())


def test_sleep_and_hush_survive_service_reconnect(monkeypatch):
    from musegadget.link_client import Outcome

    async def check():
        voices = asyncio.Queue()

        async def running(voice):
            voice._ready.set()
            voices.put_nowait(voice)
            await asyncio.Event().wait()

        class Session(FakeSession):
            async def run(self, stop):
                await stop.wait()
                return Outcome.CLOSED

        monkeypatch.setattr(VoiceConversation, "run", running)
        service, _, speech = command_device()
        for reconnect in (False, True):
            stop = asyncio.Event()
            session = Session()
            serving = asyncio.create_task(service.serve(session, {"vm_id": "vm-1"}, stop))
            voice = await voices.get()
            if not reconnect:
                voice._wake_window = WakeWindow.ASLEEP
                voice._hush()
            assert await invoke(service) == {"ok": True, "payload": {"spoken": False, "reason": "asleep"}}
            assert voice._say_hushed is True
            stop.set()
            assert await serving is Outcome.STOPPED
            assert service._voice is None
            assert session.sent == session.setup_messages == []
        assert speech.words == []
    asyncio.run(check())


def test_invalid_command_text_is_not_logged_or_synthesized(caplog):
    caplog.set_level(logging.INFO)
    async def check():
        service, voice, speech = command_device()
        caplog.clear()
        assert await invoke(service, "secret" * 100) == {"ok": False, "error": "invalid_parameters"}
        assert speech.words == []
    asyncio.run(check())
    assert "secret" not in caplog.text
    assert [record.message for record in caplog.records] == [
        "Reachy command name=reachy.say outcome=invalid_parameters"]


def test_registered_presence_returns_only_the_coarse_result_and_logs_the_outcome(caplog):
    caplog.set_level(logging.INFO)
    class Camera:
        def presence(self):
            return {"ok": True, "present": True, "last_seen_seconds": 0}
    service = ReachyService(identity=Identity("02:00:00:ab:cd:ef"), executor=Camera(), backends=lambda _: None)
    assert service.run_command("reachy.presence", {}) == {
        "ok": True, "payload": {"present": True, "last_seen_seconds": 0}}
    assert [record.message for record in caplog.records] == [
        "Reachy command name=reachy.presence outcome=present"]
