"""The pieces of a Muse realtime call on Reachy: SDK detection, flags, microphone, speaker, fallback."""

import asyncio
import threading

import pytest

import reachy_muse.cli
from musegadget.identity import Identity
from reachy_muse.conversation import ReachyService
from reachy_muse.motion.reply_motion import ReplyMotion
from reachy_muse.realtime_call import (MicrophoneFeed, RealtimeSpeaker, RealtimeVoice, VoiceLink,
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
