# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
import logging
import threading
import time
from types import SimpleNamespace

import pytest

from musegadget.identity import Identity
from reachy_muse.arrival import PersonBox
from reachy_muse.capabilities import Endpoint, Mode
from reachy_muse.device_command import WakeWindow
from reachy_muse.conversation import ReachyService, VoiceConversation, _HeldRequest, _QueuedTurn, _SpeechJob
from reachy_muse.hardware import ReachyController
from reachy_muse.local_backends import backends_for
from reachy_muse.presence import Presence
from reachy_muse.speech_playback import PCMPlayer
from reachy_muse.turn_timeline import TurnTimeline
from fake_muse_vm import make_session
from test_conversation import FakeHardware, FakeSession, reply_job

np = pytest.importorskip("numpy")


class Speech:
    def __init__(self):
        self.words = []

    async def stream(self, text, rate):
        self.words.append(text)
        yield np.full(320, .1, dtype=np.float32)


def command_device(*, mode=Mode.ON_ROBOT, speech=None, companion=None):
    session, hardware = FakeSession(), FakeHardware()
    speech = speech or Speech()
    bundle = backends_for(mode, session, speech=speech, companion=companion)
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


@asynccontextmanager
async def registered_device(service, monkeypatch):
    async def running(voice):
        voice._ready.set()
        try:
            await asyncio.Event().wait()
        finally:
            voice._closed = True

    monkeypatch.setattr(VoiceConversation, "run", running)
    session, vm = make_session(service.run_command, [])
    stop = asyncio.Event()
    serving = asyncio.create_task(service.serve(session, {"vm_id": "vm-1"}, stop))
    try:
        await asyncio.wait_for(vm.handshake(), 2)
        await vm.accept_control_stream()
        register = await vm.next_message()
        await vm.send_message({"type": "res", "id": register["id"], "ok": True})
        await wait_until(lambda: service._voice is not None
                         and service._voice[1].session is session and service._voice[1]._ready.is_set())
        yield session, vm, service._voice[1]
    finally:
        stop.set()
        await serving


@pytest.mark.parametrize("command, hushed, expected", [
    ("reachy.say", False, {"method": "link.result", "id": "command-1", "ok": True,
                           "payload": {"spoken": True}}),
    ("reachy.say", True, {"method": "link.result", "id": "command-1", "ok": True,
                          "payload": {"spoken": False, "reason": "hushed"}}),
    ("reachy.presence", False, {"method": "link.result", "id": "command-1", "ok": True,
                                "payload": {"present": True, "last_seen_seconds": 0, "source": "camera", "available": True}}),
])
def test_command_results_reach_the_sdk_wire_payload(command, hushed, expected):
    async def check():
        service, voice, _ = command_device()
        voice.hardware.presence = lambda timeout_s: {"ok": True, "present": True, "last_seen_seconds": 0}
        voice.hardware.presence_observations = lambda: {"camera": Presence(True, time.monotonic(), time.monotonic())}
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
        assert voice.hardware.play_expressions == [("speaking", "neutral")]
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


@pytest.mark.parametrize("failure", ["speech_failed", "timeout"])
def test_failed_say_preserves_a_deferred_reply_tail(monkeypatch, failure):
    async def check():
        now = queued_end = 0.0
        discarded_frames = 0
        draining = asyncio.Queue()
        release = asyncio.Event()

        async def sleep(seconds):
            nonlocal now
            if seconds > 0:
                end = now + seconds
                draining.put_nowait(None)
                await release.wait()
                now = max(now, end)
            await asyncio.sleep(0)

        monkeypatch.setattr("reachy_muse.conversation.PCMPlayer",
                            partial(PCMPlayer, clock=lambda: now, sleep=sleep))

        class FailedSpeech:
            async def stream(self, text, rate):
                if text == "An earlier reply.":
                    yield np.full(320, .25, np.float32)
                elif failure == "timeout":
                    await asyncio.Event().wait()
                else:
                    raise RuntimeError("Synthesis failed")

        service, voice, _ = command_device(speech=FailedSpeech())
        play_audio = voice.hardware.play_audio

        def play(samples):
            nonlocal queued_end
            play_audio(samples)
            queued_end = max(queued_end, now) + len(samples) / voice.hardware.output_sample_rate

        def clear():
            nonlocal discarded_frames
            discarded_frames += round(max(0, queued_end - now) * voice.hardware.output_sample_rate)

        voice.hardware.play_audio = play
        voice.hardware.clear_audio = clear
        voice._output_queue.put_nowait(_SpeechJob(None, "An earlier reply."))
        speaker = asyncio.create_task(voice._play_output())
        handoff = None
        pending = None
        try:
            await voice._output_queue.join()
            await draining.get()
            pending = asyncio.create_task(invoke(service, timeout_ms=200))
            if failure == "speech_failed":
                handoff = asyncio.create_task(draining.get())
                await asyncio.wait({pending, handoff}, return_when=asyncio.FIRST_COMPLETED)
                release.set()
            assert await pending == {"ok": True, "payload": {"spoken": False, "reason": failure}}
            assert discarded_frames == 0
            assert np.concatenate(voice.hardware.played).tolist() == [.25] * 320
        finally:
            release.set()
            tasks = [task for task in (speaker, handoff, pending) if task is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(check())


def test_outer_say_timeout_cancels_speech_before_the_loop_resumes():
    async def check():
        service, voice, speech = command_device()
        assert service.run_command("reachy.say", {"text": "An expired reminder."}, 1) == {
            "ok": True, "payload": {"spoken": False, "reason": "timeout"}}
        assert await invoke(service, "A new reminder.") == {"ok": True, "payload": {"spoken": True}}
        assert speech.words == ["A new reminder."]
        assert sum(len(samples) for samples in voice.hardware.played) == 320
        assert not voice._command_tasks

    asyncio.run(check())


@pytest.mark.parametrize("queue", ["slots", "executor"])
def test_say_deadline_includes_sdk_dispatch_queue(monkeypatch, queue):
    async def check():
        stalled, resume = threading.Event(), threading.Event()

        def frame():
            stalled.set()
            resume.wait()

        controller = ReachyController(enable_motion=False)
        controller._robot = SimpleNamespace(media=SimpleNamespace(get_frame=frame))
        service, _, speech = command_device()
        service.executor.presence = controller.presence
        service.executor.presence_observations = controller.presence_observations
        count = 4 if queue == "slots" else 2
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=count))
        try:
            async with registered_device(service, monkeypatch) as (session, vm, voice):
                for index in range(count):
                    await vm.send_message({"method": "link.invoke", "id": f"presence-{index}",
                                           "command": "reachy.presence", "params": {}, "timeout_ms": 500})
                await wait_until(lambda: stalled.is_set() and session._invokes._value == 4 - count)
                await vm.send_message({"method": "link.invoke", "id": "expired", "command": "reachy.say",
                                       "params": {"text": "An expired reminder."}, "timeout_ms": 150})
                results = [await asyncio.wait_for(vm.next_message(), 2) for _ in range(count + 1)]
                assert next(result for result in results if result["id"] == "expired") == {
                    "method": "link.result", "id": "expired", "ok": True,
                    "payload": {"spoken": False, "reason": "timeout"}}
                assert speech.words == []
                assert voice.hardware.played == []
        finally:
            resume.set()

    asyncio.run(check())


@pytest.mark.parametrize("queue", ["slots", "executor"])
@pytest.mark.parametrize("params", [{}, None])
def test_presence_deadline_includes_sdk_dispatch_queue(monkeypatch, queue, params):
    now = [100.0]
    monkeypatch.setattr("reachy_muse.conversation.time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr("reachy_muse.hardware.time", SimpleNamespace(monotonic=lambda: now[0]))

    async def check():
        stalled, resume = threading.Event(), threading.Event()
        reads = []

        def block():
            stalled.set()
            resume.wait()

        def frame():
            reads.append(True)
            return object()

        controller = ReachyController(enable_motion=False)
        controller._robot = SimpleNamespace(media=SimpleNamespace(get_frame=frame))
        controller._presence_detector = SimpleNamespace(detect=lambda image: [PersonBox(0, 0, .2, .3, .9)])
        controller._presence = Presence(True, 90, 90)
        service, _, _ = command_device()
        service.executor.presence = controller.presence
        service.executor.presence_observations = controller.presence_observations
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_running_loop()
        loop.set_default_executor(executor)
        async with registered_device(service, monkeypatch) as (session, vm, _):
            if queue == "slots":
                for _ in range(4):
                    await session._invokes.acquire()
            else:
                blocker = loop.run_in_executor(None, block)
                await wait_until(stalled.is_set)
            try:
                await vm.send_message({"method": "link.invoke", "id": "expired", "command": "reachy.presence",
                                       "params": params, "timeout_ms": 20})
                await wait_until(lambda: bool(session._invokes._waiters) if queue == "slots"
                                 else executor._work_queue.qsize() == 1)
                now[0] += .08
            finally:
                if queue == "slots":
                    for _ in range(4):
                        session._invokes.release()
                else:
                    resume.set()
                    await blocker
            assert await asyncio.wait_for(vm.next_message(), 2) == {
                "method": "link.result", "id": "expired", "ok": True,
                "payload": {"present": None, "last_seen_seconds": 10, "source": "camera", "available": True}}
            assert reads == []

    asyncio.run(check())


@pytest.mark.parametrize("queue", ["executor", "loop"])
@pytest.mark.parametrize("reopen", ["wake", "follow_up"])
def test_hush_invalidates_say_before_its_loop_callback(monkeypatch, queue, reopen):
    async def check():
        queued, resume = threading.Event(), threading.Event()

        class CommandExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                if queue == "executor" and args and args[0] == "reachy.say":
                    def waiting():
                        queued.set()
                        assert resume.wait(2)
                        return fn(*args, **kwargs)
                    return super().submit(waiting)
                return super().submit(fn, *args, **kwargs)

        schedule = asyncio.run_coroutine_threadsafe

        def waiting(coroutine, loop):
            queued.set()
            assert resume.wait(2)
            return schedule(coroutine, loop)

        if queue == "loop":
            monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", waiting)
        asyncio.get_running_loop().set_default_executor(CommandExecutor(max_workers=2))
        service, _, speech = command_device()
        try:
            async with registered_device(service, monkeypatch) as (_, vm, voice):
                await vm.send_message({"method": "link.invoke", "id": "old", "command": "reachy.say",
                                       "params": {"text": "An old reminder."}})
                await wait_until(queued.is_set)
                voice._hush()
                if reopen == "wake":
                    await voice._wake_up(WakeWindow.AWAKE)
                else:
                    timeline = TurnTimeline()
                    timeline.mark("endpoint")
                    request = _QueuedTurn(None, False, 0, timeline, ())
                    voice.job = reply_job()
                    accepted = await voice._follow_up(Endpoint(b"wav", "Find tomorrow's weather."), request, None)
                    assert accepted.text == "Find tomorrow's weather."
                    voice.job = None
                resume.set()
                assert await asyncio.wait_for(vm.next_message(), 2) == {
                    "method": "link.result", "id": "old", "ok": True,
                    "payload": {"spoken": False, "reason": "interrupted"}}
                assert speech.words == []
                assert voice.hardware.played == []
                await vm.send_message({"method": "link.invoke", "id": "new", "command": "reachy.say",
                                       "params": {"text": "A new reminder."}})
                assert await asyncio.wait_for(vm.next_message(), 2) == {
                    "method": "link.result", "id": "new", "ok": True, "payload": {"spoken": True}}
                assert speech.words == ["A new reminder."]
                assert sum(len(samples) for samples in voice.hardware.played) == 320
        finally:
            resume.set()

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


def test_hush_interrupts_stalled_synthesis_and_flushes_before_the_next_reply():
    async def check():
        stalled, closed = asyncio.Event(), asyncio.Event()
        flushing, finish_flush = threading.Event(), threading.Event()

        class StalledSpeech(Speech):
            async def stream(self, text, rate):
                self.words.append(text)
                if text == "Lunch is ready.":
                    try:
                        yield np.full(3200, .1, dtype=np.float32)
                        stalled.set()
                        await asyncio.Event().wait()
                    finally:
                        closed.set()
                else:
                    yield np.full(320, .2, dtype=np.float32)

        service, voice, speech = command_device(speech=StalledSpeech())
        clear_audio = voice.hardware.clear_audio

        def clear():
            flushing.set()
            assert finish_flush.wait(2)
            clear_audio()

        voice.hardware.clear_audio = clear
        pending = asyncio.create_task(invoke(service, timeout_ms=5000))
        await stalled.wait()
        voice._hush()
        reply = asyncio.create_task(voice._speak(None, text="A later reply.", stream_segment=True))
        try:
            await wait_until(flushing.is_set)
            assert closed.is_set() and voice._speaker_lock.locked()
            assert speech.words == ["Lunch is ready."] and not reply.done()
        finally:
            finish_flush.set()
            await asyncio.gather(pending, reply)
        assert pending.result() == {"ok": True, "payload": {"spoken": False, "reason": "hushed"}}
        assert speech.words == ["Lunch is ready.", "A later reply."]
        assert voice.hardware.cleared == 1 and not voice._speaker_lock.locked()
        assert not voice._command_tasks

    asyncio.run(check())


@pytest.mark.parametrize("text, waiting", [("okay", False), ("thanks", False), ("also", False),
                                           ("Find tomorrow's weather.", True)])
def test_ignored_follow_up_keeps_later_room_commands_hushed(text, waiting):
    async def check():
        service, voice, speech = command_device()
        timeline = TurnTimeline()
        timeline.mark("endpoint")
        request = _QueuedTurn(None, False, 0, timeline, ())
        held = _HeldRequest(b"wav", "Find lunch options.", 0, timeline) if waiting else None
        voice.job = reply_job()
        voice._hush()
        assert await voice._follow_up(Endpoint(b"wav", text), request, held) is held
        voice.job = None
        assert await invoke(service) == {"ok": True, "payload": {"spoken": False, "reason": "hushed"}}
        assert speech.words == voice.session.sent == voice.session.setup_messages == []

    asyncio.run(check())


@pytest.mark.parametrize("text, waiting, expected", [
    ("Find tomorrow's weather.", False, "Find tomorrow's weather."),
    ("Also make it vegetarian.", True, "Find lunch options. Also make it vegetarian."),
])
def test_accepted_follow_up_allows_later_room_commands(text, waiting, expected):
    async def check():
        service, voice, _ = command_device()
        timeline = TurnTimeline()
        timeline.mark("endpoint")
        request = _QueuedTurn(None, False, 0, timeline, ())
        held = _HeldRequest(b"wav", "Find lunch options.", 0, timeline) if waiting else None
        voice.job = reply_job()
        voice._hush()
        accepted = await voice._follow_up(Endpoint(b"wav", text), request, held)
        assert accepted.text == expected
        voice.job = None
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


@pytest.mark.parametrize("recovered", [False, True])
def test_say_selects_companion_without_changing_an_existing_reply_route(recovered):
    class Companion:
        up = True
        out_rate = 8000

        def __init__(self):
            self.words = []

        def can(self, op):
            return self.up

        async def speak(self, text, role):
            self.words.append(text)
            yield np.full(320, 8192, dtype="<i2").tobytes()

    async def check():
        companion = Companion()
        service, voice, speech = command_device(mode=Mode.COMPANION, companion=companion)
        if recovered:
            companion.up = False
            voice.backends.reply_style()
            await voice._speak(None, text="A local reply.", stream_segment=True)
            assert speech.words == ["A local reply."]
            speech.words.clear()
            voice.hardware.played.clear()
            companion.up = True
        assert await invoke(service) == {"ok": True, "payload": {"spoken": True}}
        assert companion.words == ["Lunch is ready."]
        assert speech.words == []
        np.testing.assert_array_equal(np.concatenate(voice.hardware.played), [.25] * 320)
        if recovered:
            await voice._speak(None, text="The rest of the local reply.", stream_segment=True)
            assert speech.words == ["The rest of the local reply."]
            assert companion.words == ["Lunch is ready."]
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


def test_say_uses_one_voice_snapshot_during_disconnect():
    class DisconnectingService(ReachyService):
        def __getattribute__(self, name):
            value = super().__getattribute__(name)
            if name == "_voice" and value is not None:
                loop, voice = value
                disconnected = threading.Event()

                def disconnect():
                    voice._closed = True
                    self._voice = None
                    disconnected.set()

                loop.call_soon_threadsafe(disconnect)
                assert disconnected.wait(2)
            return value

    async def check():
        original, voice, speech = command_device()
        service = DisconnectingService(identity=original.identity, executor=original.executor,
                                       backends=original.backends)
        service._voice = original._voice
        assert await invoke(service) == {"ok": True, "payload": {"spoken": False, "reason": "disconnected"}}
        assert speech.words == voice.hardware.played == []

    asyncio.run(check())


def test_queued_sdk_say_cannot_reach_a_replacement_voice(monkeypatch):
    from musegadget.link_client import Outcome

    async def check():
        workers = []
        resume = threading.Event()

        class CommandExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                if args and args[0] == "reachy.say":
                    def queued():
                        resume.wait()
                        return fn(*args, **kwargs)
                    future = super().submit(queued)
                    workers.append(future)
                    return future
                return super().submit(fn, *args, **kwargs)

        async def running(voice):
            voice._ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                voice._closed = True

        monkeypatch.setattr(VoiceConversation, "run", running)
        service, _, speech = command_device()
        loop = asyncio.get_running_loop()
        loop.set_default_executor(CommandExecutor(max_workers=2))
        try:
            for reconnect in (False, True):
                session, vm = make_session(service.run_command, [])
                stop = asyncio.Event()
                serving = asyncio.create_task(service.serve(session, {"vm_id": "vm-1"}, stop))
                try:
                    await asyncio.wait_for(vm.handshake(), 2)
                    await vm.accept_control_stream()
                    register = await vm.next_message()
                    await vm.send_message({"type": "res", "id": register["id"], "ok": True})
                    await wait_until(lambda: service._voice is not None
                                     and service._voice[1].session is session and service._voice[1]._ready.is_set())
                    if not reconnect:
                        await vm.send_message({"method": "link.invoke", "id": "old", "command": "reachy.say",
                                               "params": {"text": "An old reminder."}})
                        await wait_until(lambda: len(workers) == 1)
                        await vm.ws.close()
                        assert await serving is Outcome.CLOSED
                    else:
                        resume.set()
                        assert await asyncio.wait_for(asyncio.wrap_future(workers[0]), 2) == {
                            "ok": True, "payload": {"spoken": False, "reason": "disconnected"}}
                        assert speech.words == service.executor.played == []
                        await vm.send_message({"method": "link.invoke", "id": "new", "command": "reachy.say",
                                               "params": {"text": "A new reminder."}})
                        assert await asyncio.wait_for(vm.next_message(), 2) == {
                            "method": "link.result", "id": "new", "ok": True, "payload": {"spoken": True}}
                        assert speech.words == ["A new reminder."]
                finally:
                    stop.set()
                    await serving
        finally:
            resume.set()

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
        def presence_observations(self):
            return {"camera": Presence(True, time.monotonic(), time.monotonic())}

        def presence(self, timeout_s):
            return {"ok": True, "present": True, "last_seen_seconds": 0}
    service = ReachyService(identity=Identity("02:00:00:ab:cd:ef"), executor=Camera(), backends=lambda _: None)
    assert service.run_command("reachy.presence", {}) == {
        "ok": True, "payload": {"present": True, "last_seen_seconds": 0, "source": "camera", "available": False}}
    assert [record.message for record in caplog.records] == [
        "Reachy command name=reachy.presence outcome=present"]


def test_stalled_presence_checks_release_sdk_slots_across_reconnects():
    async def check():
        stalled, resume, invoked = threading.Event(), threading.Event(), threading.Event()
        reads, calls = [], []

        def frame():
            reads.append(True)
            stalled.set()
            resume.wait()

        controller = ReachyController(enable_motion=False)
        controller._robot = SimpleNamespace(media=SimpleNamespace(get_frame=frame))
        service = ReachyService(identity=Identity("02:00:00:ab:cd:ef"), executor=controller,
                               backends=lambda _: None)

        def run_command(*args):
            calls.append(args[0])
            if len(calls) == 4:
                invoked.set()
            return service.run_command(*args)

        try:
            for reconnect in (False, True):
                session, vm = make_session(run_command, [])
                session._invokes = service._invoke_slots
                stop = asyncio.Event()
                running = asyncio.create_task(session.run(stop))
                try:
                    await vm.handshake()
                    await vm.accept_control_stream()
                    register = await vm.next_message()
                    await vm.send_message({"type": "res", "id": register["id"], "ok": True})
                    await session.registered.wait()
                    if not reconnect:
                        for index in range(4):
                            await vm.send_message({"method": "link.invoke", "id": f"presence-{index}",
                                                   "command": "reachy.presence", "params": {}, "timeout_ms": 10})
                        assert await asyncio.to_thread(invoked.wait, 1)
                        assert await asyncio.to_thread(stalled.wait, 1)
                    else:
                        await vm.send_message({"method": "link.invoke", "id": "say", "command": "reachy.say",
                                               "params": {"text": "Lunch is ready."}})
                        assert await asyncio.wait_for(vm.next_message(), .2) == {
                            "method": "link.result", "id": "say", "ok": True,
                            "payload": {"spoken": False, "reason": "disconnected"}}
                        assert reads == [True]
                finally:
                    stop.set()
                    await running
        finally:
            resume.set()

    asyncio.run(check())
