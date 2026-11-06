"""The pieces of a Muse realtime call on Reachy: SDK detection, flags, microphone, speaker, fallback."""

import asyncio
import base64
import json
import threading
from types import SimpleNamespace

import pytest

import reachy_muse.cli
from musegadget.identity import Identity
from musegadget.streaming import HTTPStatusError
from reachy_muse.conversation import ReachyService
from reachy_muse.motion.reply_motion import ReplyMotion
from reachy_muse.realtime_call import (MicrophoneFeed, RealtimeCall, RealtimeSpeaker, RealtimeVoice, VoiceLink,
                                       realtime_supported)

np = pytest.importorskip("numpy")
pytest.importorskip("av")


def test_an_sdk_whose_voice_call_reports_no_events_leaves_realtime_unavailable(monkeypatch):
    from musegadget import voice_call

    class OldVoiceCall:
        def __init__(self, session, *, device_id, thread_id=None, half_duplex=True):
            pass

    monkeypatch.setattr(voice_call, "VoiceCall", OldVoiceCall)
    assert realtime_supported() is False


@pytest.mark.parametrize("options, path", [([], "chat"), (["--muse-path", "chat"], "chat")])
def test_chat_stays_the_default_muse_path(options, path):
    args = reachy_muse.cli.parser().parse_args(["run", *options])
    assert reachy_muse.cli.realtime_path(args) == path


@pytest.mark.parametrize("options, error", [
    (["--mode", "on-robot"], "--muse-path realtime needs --wake-model"),
    (["--wake-model", "wake.onnx"], "--muse-path realtime needs --mode on-robot or companion"),
])
def test_realtime_needs_a_wake_word_and_local_recognition(options, error):
    args = reachy_muse.cli.parser().parse_args(["run", "--muse-path", "realtime", *options])
    with pytest.raises(ValueError, match=error):
        reachy_muse.cli.realtime_path(args)


def test_realtime_on_an_sdk_without_voice_events_runs_the_chat_path(monkeypatch):
    monkeypatch.setattr("reachy_muse.realtime_call.realtime_supported", lambda: False)
    args = reachy_muse.cli.parser().parse_args(
        ["run", "--muse-path", "realtime", "--mode", "on-robot", "--wake-model", "wake.onnx"])
    assert reachy_muse.cli.realtime_path(args) == "chat"


@pytest.mark.parametrize("supports_app_id, expected", [
    (True, ("hatch-web", False)),
    (False, (None, False)),
], ids=["current-sdk", "older-sdk"])
def test_voice_link_passes_the_voice_app_id_when_the_sdk_accepts_it(monkeypatch, supports_app_id, expected):
    from musegadget import config

    class OldLink:
        def __init__(self, *, noise_host, vm_id, vm_auth_token, device, register_device, run_command):
            self.app_id = None
            self.register_device = register_device

        async def run(self, stop):
            await stop.wait()

        async def wait_transport_ready(self, timeout):
            pass

    class Link(OldLink):
        def __init__(self, *, app_id=None, **options):
            super().__init__(**options)
            self.app_id = app_id

    monkeypatch.setattr("reachy_muse.conversation.LinkSession", Link if supports_app_id else OldLink)
    monkeypatch.setattr(config, "load_json", lambda *args: {"noise_host": "noise.test"})

    async def scenario():
        service = ReachyService(identity=Identity("02:00:00:ab:cd:ef"), executor=object(), backends=lambda _: None)
        link = await service._voice_link({"vm_id": "vm-1", "vm_name": "test-vm", "vm_auth_token": "test-token"})
        try:
            return link.session.app_id, link.session.register_device
        finally:
            await link.close()

    assert asyncio.run(scenario()) == expected


def test_the_microphone_reaches_muse_as_24_khz_frames_and_a_backlog_sheds_silence_first():
    async def scenario():
        feed = MicrophoneFeed(16000)
        feed.push(np.zeros(16000, np.float32))
        feed.push(np.full(3200, .3, np.float32))
        feed.close()
        frames = [frame async for frame in feed]
        loud = [frame for frame in frames if np.frombuffer(frame, "<i2").max() > 9000]
        return len(frames), len(loud), {len(frame) for frame in frames}, feed.sent

    count, loud, sizes, sent = asyncio.run(scenario())
    assert sizes == {1920}
    assert (count, loud, sent) == (10, 4, 10)


def test_the_microphone_keeps_a_whole_request_while_the_call_is_still_being_admitted():
    async def scenario():
        feed = MicrophoneFeed(24000)
        for _ in range(12):
            feed.push(np.full(24000, .3, np.float32))
        feed.close()
        return len([frame async for frame in feed])

    assert asyncio.run(scenario()) == 300


class Speaker:
    sample_rate = 16000
    output_sample_rate = 8000

    def __init__(self):
        self.played = []
        self.cleared = 0

    def play_audio(self, samples):
        self.played.append(len(samples))

    def clear_audio(self):
        self.cleared += 1


def tone(seconds):
    t = np.arange(int(seconds * 24000)) / 24000
    return (4000 * np.sin(2 * np.pi * 150 * t)).astype("<i2").tobytes()


def test_muse_audio_plays_through_the_speaker_and_a_flush_clears_it():
    async def scenario():
        hardware = Speaker()
        lifecycle = []

        async def started():
            lifecycle.append("start")

        async def idle():
            lifecycle.append("idle")

        speaker = RealtimeSpeaker(hardware, ReplyMotion(), on_audible=lambda text, until: None,
                                  on_start=started, on_idle=idle)
        for offset in range(0, 48000, 4800):
            speaker.chunk("r1", offset // 2, tone(1.0)[offset:offset + 4800])
        while not speaker.busy or sum(hardware.played) < 4000:
            await asyncio.sleep(.01)
        speaker.chunk("r1", 24000, tone(5.0))
        await speaker.flush_now()
        flushed = (speaker.busy, hardware.cleared, list(lifecycle))
        await speaker.close()
        return flushed, sum(hardware.played)

    (busy, cleared, lifecycle), played = asyncio.run(scenario())
    assert (busy, cleared, lifecycle) == (False, 1, ["start", "idle"])
    assert 4000 <= played < 12000


@pytest.mark.parametrize("failure_at", ["play_audio", "on_start", "on_audible", "on_idle"])
def test_a_speaker_failure_ends_the_call_and_clears_its_playback_state(failure_at):
    from musegadget import voice_call as sdk

    async def scenario():
        failed = asyncio.Event()
        closed = asyncio.Event()
        error = OSError("speaker disconnected")
        loop = asyncio.get_running_loop()

        def fail(where):
            if failure_at == where:
                loop.call_soon_threadsafe(failed.set)
                raise error

        class Hardware(Speaker):
            def play_audio(self, samples):
                fail("play_audio")
                super().play_audio(samples)

        class Call:
            async def run(self, source, output, *, realtime_input=False):
                call.on_event(sdk.AudioChunk("r1", 0, tone(.1)))
                try:
                    await asyncio.Event().wait()
                finally:
                    closed.set()

        async def start():
            fail("on_start")

        async def idle():
            fail("on_idle")

        async def build(call):
            return Call()

        motion = ReplyMotion()
        hardware = Hardware()
        speaker = RealtimeSpeaker(hardware, motion, on_audible=lambda *args: fail("on_audible"),
                                  on_start=start, on_idle=idle)
        call = RealtimeCall(MicrophoneFeed(16000), speaker, motion, on_working=lambda working: None,
                            on_reply_done=lambda reply: None)
        call.start(build)
        try:
            await asyncio.wait_for(failed.wait(), 3)
            for _ in range(30):
                await asyncio.sleep(0)
            ended = call.done, call.failed, call.failure is error
        finally:
            await call.close()
        return ended, speaker.busy, hardware.cleared, closed.is_set()

    assert asyncio.run(scenario()) == ((True, True, True), False, 1, True)


@pytest.mark.parametrize("first", ["local", "muse"])
@pytest.mark.parametrize("flush", [False, True])
def test_local_and_realtime_speech_wait_for_each_others_playback_ownership(first, flush):
    from musegadget import voice_call as sdk
    from test_conversation import FakeSession
    from test_turn_pacing import Labels, conversation_for

    async def scenario():
        labels = Labels()
        entered, release = threading.Event(), threading.Event()
        played = []

        class Hardware(Speaker):
            sample_rate = output_sample_rate = 24000

            def set_state(self, state, *, expression=None):
                pass

            def play_audio(self, samples):
                label = "local" if samples[0] == labels.codes.get("Notification.") else "muse"
                if played[-1:] != [label]:
                    played.append(label)
                if len(played) == 1:
                    entered.set()
                    release.wait(3)

            def clear_audio(self):
                played.append("clear")
                super().clear_audio()

        class Call:
            def __init__(self, *args, **kwargs):
                pass

            async def run(self, *args, **kwargs):
                await asyncio.Event().wait()

        async def close():
            pass

        async def link():
            return VoiceLink(object(), lambda: True, close)

        hardware = Hardware()
        voice = conversation_for(labels, hardware, "")(FakeSession())
        voice.realtime = RealtimeVoice(link, device_id="synthetic", voice_call=Call)
        voice._open_call()

        def muse():
            voice.call.on_event(sdk.AudioChunk("r1", 0, tone(.2)))

        local = None
        try:
            if first == "local":
                local = asyncio.create_task(voice._speak(None, text="Notification."))
            else:
                muse()
            assert await asyncio.to_thread(entered.wait, 2)
            if first == "local":
                muse()
            else:
                local = asyncio.create_task(voice._speak(None, text="Notification."))
            await asyncio.sleep(.05)
            before_release = played.copy()
            flushing = voice.call.speaker.flush_now() if flush else None
            release.set()
            if flushing is not None:
                await flushing
            await local
            while voice.call.audible:
                await asyncio.sleep(.01)
            return before_release, played, hardware.cleared
        finally:
            release.set()
            if local is not None:
                await asyncio.gather(local, return_exceptions=True)
            await voice._close_call()
            await voice.realtime.close()

    other = "muse" if first == "local" else "local"
    expected = ([first], [first, other], 0)
    if flush:
        expected = ([first], ["local"], 0) if first == "local" else ([first], ["muse", "clear", "local"], 1)
    assert asyncio.run(asyncio.wait_for(scenario(), 5)) == expected


def test_closing_during_the_idle_callback_releases_playback_ownership():
    async def scenario():
        lock = asyncio.Lock()
        idle = asyncio.Event()
        hardware = Speaker()

        async def start():
            pass

        async def quiet():
            idle.set()
            await asyncio.Event().wait()

        speaker = RealtimeSpeaker(hardware, ReplyMotion(), on_audible=lambda *args: None,
                                  on_start=start, on_idle=quiet, own_speaker=lambda cursor: lock)
        speaker.chunk("r1", 0, tone(.04))
        await idle.wait()
        await speaker.close()
        return lock.locked(), hardware.cleared

    assert asyncio.run(asyncio.wait_for(scenario(), 2)) == (False, 1)


@pytest.mark.parametrize("ending", ["drain", "interrupt", "timeout", "close"])
def test_server_hangup_drains_the_real_speaker_waiting_for_local_speech(monkeypatch, ending):
    from musegadget import voice_call as sdk
    from test_conversation import FakeSession
    from test_turn_pacing import Labels, conversation_for

    if ending == "timeout":
        monkeypatch.setattr("reachy_muse.realtime_call.SPEAKER_DRAIN_S", .05)

    async def scenario():
        labels = Labels()
        entered, release = threading.Event(), threading.Event()
        stopped = asyncio.Event()
        played = []

        class Hardware(Speaker):
            sample_rate = output_sample_rate = 24000

            def set_state(self, state, *, expression=None):
                pass

            def play_audio(self, samples):
                if samples[0] == labels.codes.get("Notification."):
                    entered.set()
                    release.wait(3)
                else:
                    played.append(len(samples))

        class Call(sdk.VoiceCall):
            async def _start(self):
                async def stream():
                    yield (json.dumps({"type": "hatch.voice.session", "call_id": self.call_id,
                                       "session_id": "synthetic"}) + "\n").encode()
                    yield (json.dumps({"type": "response.output_audio.delta",
                                       "delta": base64.b64encode(tone(.08)).decode()}) + "\n").encode()
                    yield (json.dumps({"type": "conversation.item.added", "item": {
                        "type": "session_control", "name": "stop", "call_id": "control"}}) + "\n").encode()
                self._stream = stream()

            async def _event(self, event):
                pass

            async def _cleanup(self):
                stopped.set()

        async def close():
            pass

        async def link():
            return VoiceLink(object(), lambda: True, close)

        hardware = Hardware()
        voice = conversation_for(labels, hardware, "")(FakeSession())
        voice.realtime = RealtimeVoice(link, device_id="synthetic", voice_call=Call)
        local = asyncio.create_task(voice._speak(None, text="Notification."))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            voice._open_call()
            call = voice.call
            await asyncio.wait_for(stopped.wait(), 2)
            while call.reason is None:
                await asyncio.sleep(0)
            if ending == "interrupt":
                await call.barge_in()
            elif ending == "timeout":
                await asyncio.wait_for(call._task, 1)
            elif ending == "close":
                await asyncio.wait_for(call.close(), 1)
            release.set()
            await local
            await call._task
            return sum(played), call.reason, hardware.cleared
        finally:
            release.set()
            await asyncio.gather(local, return_exceptions=True)
            await voice._close_call()
            await voice.realtime.close()

    assert asyncio.run(asyncio.wait_for(scenario(), 5)) == (1920 if ending == "drain" else 0, "remote_stop", 0)


def test_the_sdk_s_two_flush_callbacks_preserve_fresh_audio_that_arrives_during_cleanup():
    from musegadget import voice_call as sdk
    from reachy_muse.realtime_call import _SilentOutput

    class Hardware(Speaker):
        output_sample_rate = 24000

        def __init__(self):
            super().__init__()
            self.first_played = threading.Event()
            self.clearing = threading.Event()
            self.release = threading.Event()
            self.fresh = threading.Event()
            self.buffer = []

        def play_audio(self, samples):
            self.buffer.extend(samples.tolist())
            self.first_played.set()
            if np.max(samples) > .2:
                self.fresh.set()

        def clear_audio(self):
            self.cleared += 1
            if self.cleared == 1:
                self.clearing.set()
                self.release.wait(3)
            else:
                self.fresh.wait(3)
            self.buffer.clear()

    async def scenario():
        hardware = Hardware()

        async def nothing():
            pass

        motion = ReplyMotion()
        speaker = RealtimeSpeaker(hardware, motion, on_audible=lambda *args: None,
                                  on_start=nothing, on_idle=nothing)
        call = RealtimeCall(MicrophoneFeed(16000), speaker, motion, on_working=lambda working: None,
                            on_reply_done=lambda reply: None)
        remote = sdk.VoiceCall(object(), device_id="homelink-synthetic", on_event=call.on_event)
        remote._output = _SilentOutput(call)
        try:
            remote._queue_pcm(tone(.2), "r1")
            assert await asyncio.to_thread(hardware.first_played.wait, 3)
            await remote.barge_in()
            assert await asyncio.to_thread(hardware.clearing.wait, 3)
            remote._queue_pcm(np.full(1920, 10000, "<i2").tobytes(), "r2")
            hardware.release.set()
            await speaker._flushing
            assert await asyncio.to_thread(hardware.fresh.wait, 3)
            while speaker.busy:
                await asyncio.sleep(.01)
            return hardware.cleared, len(hardware.buffer)
        finally:
            hardware.release.set()
            await call.close()

    assert asyncio.run(asyncio.wait_for(scenario(), 5)) == (1, 1920)


def test_a_voice_link_that_will_not_connect_leaves_the_next_call_failed_and_the_chat_path_open():
    async def scenario():
        async def refuse():
            raise ConnectionError("no route to Muse")

        voice = RealtimeVoice(refuse, device_id="homelink-abcdef", thread_id="robot-chat",
                              voice_call=lambda *args, **kwargs: None)

        async def nothing():
            pass

        call = voice.open(Speaker(), ReplyMotion(), on_audible=lambda *args: None, on_start=nothing,
                          on_idle=nothing, on_working=lambda working: None, on_reply_done=lambda reply: None)
        while not call.done:
            await asyncio.sleep(.01)
        failed = (call.failed, call.audio_sent, type(call.failure).__name__)
        await call.close()
        await voice.call_failed()
        refused = voice.usable
        voice.chat_turn()
        return failed, refused, voice.usable

    assert asyncio.run(scenario()) == ((True, False, "ConnectionError"), False, True)


@pytest.mark.parametrize("failure, details", [
    pytest.param(HTTPStatusError(400, b'{"ok":false,"error":{"code":"voice_app_id_required",'
                                     b'"message":"private user data"},"transcript":"private words"}'),
                 "HTTPStatusError status=400 code=voice_app_id_required", id="voice-app-id-required"),
    pytest.param(HTTPStatusError(502, b"private invalid JSON"),
                 "HTTPStatusError status=502 code=unknown", id="invalid-json"),
    pytest.param(HTTPStatusError(502, b"\xff"),
                 "HTTPStatusError status=502 code=unknown", id="invalid-encoding"),
    pytest.param(HTTPStatusError(400, b'["private user data"]'),
                 "HTTPStatusError status=400 code=unknown", id="non-object-body"),
    pytest.param(HTTPStatusError(400, b'{"error":"private user data"}'),
                 "HTTPStatusError status=400 code=unknown", id="non-object-error"),
    pytest.param(HTTPStatusError(400, b'{"error":{"message":"private user data"}}'),
                 "HTTPStatusError status=400 code=unknown", id="missing-code"),
    pytest.param(HTTPStatusError(400, b'{"error":{"code":{"message":"private user data"}}}'),
                 "HTTPStatusError status=400 code=unknown", id="non-string-code"),
    pytest.param(HTTPStatusError(400, b'{"error":{"code":"private user data\\nforged log line"}}'),
                 "HTTPStatusError status=400 code=unknown", id="unsafe-code"),
    pytest.param(HTTPStatusError(400, b'{"error":{"code":"' + b"x" * 65 + b'"}}'),
                 "HTTPStatusError status=400 code=unknown", id="oversized-code"),
    pytest.param(ConnectionError("private connection details"), "ConnectionError", id="non-http-error"),
])
def test_call_failure_warning_reports_only_the_http_status_and_safe_error_code(monkeypatch, caplog, failure, details):
    monkeypatch.setattr("reachy_muse.realtime_call.time", SimpleNamespace(monotonic=lambda: 10.0))

    class RefusedCall:
        async def run(self, source, output, *, realtime_input=False):
            raise failure

    async def scenario():
        async def nothing():
            pass

        async def build(call):
            return RefusedCall()

        motion = ReplyMotion()
        speaker = RealtimeSpeaker(Speaker(), motion, on_audible=lambda *args: None, on_start=nothing, on_idle=nothing)
        call = RealtimeCall(MicrophoneFeed(16000), speaker, motion, on_working=lambda working: None,
                            on_reply_done=lambda reply: None)
        call.start(build)
        while not call.done:
            await asyncio.sleep(0)
        failed = call.failed, call.audio_sent
        await call.close()
        return failed

    assert asyncio.run(scenario()) == (True, False)
    assert caplog.messages == [f"Muse realtime call failed ({details}) after 0.0 s"]


def test_a_muse_called_expression_holds_reachy_s_own_reply_motion_while_it_runs():
    class Executor:
        def __init__(self):
            self.running = threading.Event()
            self.release = threading.Event()

        def run_command(self, name, params, timeout_ms):
            self.running.set()
            self.release.wait(2)
            return {"ok": True, "payload": {"name": params["name"]}}

    class Voice:
        muse_moving = 0

        def muse_moved(self, change):
            self.muse_moving += change

    async def scenario():
        executor = Executor()
        service = ReachyService(identity=Identity("02:00:00:ab:cd:ef"), executor=executor, backends=lambda _: None)
        voice = Voice()
        service._voice = (asyncio.get_running_loop(), voice)
        command = asyncio.create_task(asyncio.to_thread(service.run_command, "reachy.expression", {"name": "nod"}))
        await asyncio.to_thread(executor.running.wait, 2)
        await asyncio.sleep(.01)
        during = voice.muse_moving
        executor.release.set()
        result = await command
        await asyncio.sleep(.01)
        return during, voice.muse_moving, result

    assert asyncio.run(scenario()) == (1, 0, {"ok": True, "payload": {"name": "nod"}})


def test_calls_open_with_reachy_s_identity_thread_and_duplex_and_reuse_one_voice_link():
    if not realtime_supported():
        pytest.skip("the Muse SDK has no voice call events")
    opened = []
    links = []

    class Call:
        def __init__(self, session, **options):
            opened.append((session, options["device_id"], options["thread_id"], options["half_duplex"]))
            self.ready = asyncio.Event()

        async def run(self, source, output, *, realtime_input=False):
            self.ready.set()
            return "remote_stop"

    async def connect():
        links.append(object())

        async def close():
            pass
        return VoiceLink(links[-1], lambda: True, close)

    async def scenario():
        voice = RealtimeVoice(connect, device_id="homelink-abcdef", thread_id="robot-chat", voice_call=Call)

        async def nothing():
            pass

        reasons = []
        for _ in range(2):
            call = voice.open(Speaker(), ReplyMotion(), on_audible=lambda *args: None, on_start=nothing,
                              on_idle=nothing, on_working=lambda working: None, on_reply_done=lambda reply: None)
            while not call.done:
                await asyncio.sleep(.01)
            reasons.append((call.reason, call.failed))
            await call.close()
        await voice.close()
        return reasons

    assert asyncio.run(scenario()) == [("remote_stop", False)] * 2
    assert opened == [(links[0], "homelink-abcdef", "robot-chat", False)] * 2


def test_after_a_failed_call_the_chat_turn_that_follows_reconnects_the_voice_link_for_the_next_call():
    async def scenario():
        links = []

        async def connect():
            links.append(object())

            async def close():
                pass
            return VoiceLink(links[-1], lambda: True, close)

        def broken(*args, **kwargs):
            raise ConnectionError("media reset")

        voice = RealtimeVoice(connect, device_id="homelink-abcdef", voice_call=broken)

        async def nothing():
            pass

        await voice.warm()
        call = voice.open(Speaker(), ReplyMotion(), on_audible=lambda *args: None, on_start=nothing,
                          on_idle=nothing, on_working=lambda working: None, on_reply_done=lambda reply: None)
        while not call.done:
            await asyncio.sleep(.01)
        await call.close()
        await voice.call_failed()
        voice.chat_turn()
        for _ in range(100):
            if len(links) == 2:
                break
            await asyncio.sleep(.01)
        await voice.close()
        return len(links), voice.usable

    assert asyncio.run(scenario()) == (2, True)


@pytest.mark.parametrize("warm_again", [False, True])
@pytest.mark.parametrize("waiter", ["warm", "call"])
def test_a_connection_completed_after_its_waiter_is_cancelled_is_reused_and_closed(warm_again, waiter):
    async def scenario():
        connecting, release = asyncio.Event(), asyncio.Event()
        links, closed = [], []

        async def connect():
            connecting.set()
            await release.wait()
            session = object()
            links.append(session)

            async def close():
                closed.append(session)

            return VoiceLink(session, lambda: True, close)

        voice = RealtimeVoice(connect, device_id="synthetic")

        async def nothing():
            pass

        if waiter == "warm":
            warming = asyncio.create_task(voice.warm())
        else:
            call = voice.open(Speaker(), ReplyMotion(), on_audible=lambda *args: None,
                              on_start=nothing, on_idle=nothing, on_working=lambda working: None,
                              on_reply_done=lambda reply: None)
        await connecting.wait()
        if waiter == "warm":
            warming.cancel()
            await asyncio.gather(warming, return_exceptions=True)
        else:
            await call.close()
        release.set()
        await asyncio.sleep(0)
        if warm_again:
            await voice.warm()
        await voice.close()
        await voice.close()
        return len(links), len(closed), closed == links

    assert asyncio.run(scenario()) == (1, 1, True)


def test_each_sentence_s_closing_gesture_lands_where_its_speech_ends_not_where_the_next_begins():
    if not realtime_supported():
        pytest.skip("the Muse SDK has no voice call events")
    from musegadget import voice_call as sdk

    class Timed(Speaker):
        output_sample_rate = 24000

        def __init__(self):
            super().__init__()
            self.ends = []

        def play_audio(self, samples, *, sentence=None, accents=()):
            self.played.append(len(samples))

        def finish_sentence(self):
            self.ends.append(sum(self.played))

    async def scenario():
        hardware = Timed()
        idle = asyncio.Event()

        async def nothing():
            pass

        async def quiet():
            idle.set()

        motion = ReplyMotion()
        speaker = RealtimeSpeaker(hardware, motion, on_audible=lambda *args: None, on_start=nothing, on_idle=quiet)
        call = RealtimeCall(MicrophoneFeed(16000), speaker, motion, on_working=lambda working: None,
                            on_reply_done=lambda reply: None)
        text = "Your table for two is booked. Shall I add a reminder?"
        pcm = tone(1.6) + bytes(int(.5 * 24000) * 2) + tone(1.4) + bytes(int(.3 * 24000) * 2)
        call.on_event(sdk.ResponseStarted("r1"))
        call.on_event(sdk.TranscriptDelta("r1", text))
        for offset in range(0, len(pcm), 4800):
            call.on_event(sdk.AudioChunk("r1", offset // 2, pcm[offset:offset + 4800]))
        call.on_event(sdk.TranscriptDone("r1", text))
        call.on_event(sdk.ResponseDone("r1", "completed"))
        await asyncio.wait_for(idle.wait(), 10)
        await call.close()
        return [round(sample / 24000, 2) for sample in hardware.ends]

    assert asyncio.run(scenario()) == [1.6, 3.8]


def test_response_done_before_the_remaining_pcm_does_not_end_a_question_at_its_internal_pause():
    from musegadget import voice_call as sdk

    class Hardware(Speaker):
        output_sample_rate = 24000

        def __init__(self):
            super().__init__()
            self.ends = []

        def play_audio(self, samples, *, sentence=None, accents=()):
            self.played.append(len(samples))

        def finish_sentence(self):
            self.ends.append(sum(self.played))

    async def scenario():
        hardware = Hardware()
        idle = asyncio.Event()

        async def nothing():
            pass

        async def quiet():
            idle.set()

        motion = ReplyMotion()
        speaker = RealtimeSpeaker(hardware, motion, on_audible=lambda *args: None,
                                  on_start=nothing, on_idle=quiet)
        call = RealtimeCall(MicrophoneFeed(16000), speaker, motion, on_working=lambda working: None,
                            on_reply_done=lambda reply: None)
        try:
            call.on_event(sdk.TranscriptDone("r1", "Would you like a reminder?"))
            call.on_event(sdk.AudioChunk("r1", 0, tone(.5)))
            call.on_event(sdk.ResponseDone("r1", "completed"))
            call.on_event(sdk.AudioChunk("r1", 12000, bytes(7200 * 2)))
            while sum(hardware.played) < 14400:
                await asyncio.sleep(.01)
            call.on_event(sdk.AudioChunk("r1", 19200, tone(.9) + bytes(7200 * 2)))
            await asyncio.wait_for(idle.wait(), 5)
            return [round(sample / 24000, 2) for sample in hardware.ends]
        finally:
            await call.close()

    assert asyncio.run(asyncio.wait_for(scenario(), 6)) == [2.0]
