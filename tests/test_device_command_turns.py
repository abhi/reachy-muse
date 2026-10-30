import asyncio
import dataclasses
import time

import pytest

import reachy_muse.conversation
from musegadget.identity import Identity
from reachy_muse.capabilities import Mode
from reachy_muse.contract.prompts import STOP_REQUEST
from reachy_muse.device_command import Commands, DeviceCommand
from reachy_muse.local_backends import backends_for
from test_conversation import sentence_frame
from test_pins import PinHardware, PinSession, Rig, TextCodedVoice, Transcripts, Vad, Wake

np = pytest.importorskip("numpy")

QUIET_DOWN = "Could you please quiet down a little?"


class SpeakerHardware(PinHardware):
    def __init__(self, recording, chunk_s=0.0):
        super().__init__(recording)
        self.chunk_s = chunk_s
        self.volume_steps = []

    def play_audio(self, samples):
        super().play_audio(samples)
        time.sleep(self.chunk_s)

    def change_volume(self, step):
        self.volume_steps.append(step)
        return 50 + step


class LongVoice(TextCodedVoice):
    def __init__(self, recording, chunks):
        super().__init__(recording, "tts")
        self.chunks = chunks

    def stream(self, text, output_rate):
        first = super().stream(text, output_rate)

        async def audio():
            async for samples in first:
                for _ in range(self.chunks):
                    yield samples
        return audio()


class FakeModel:
    def __init__(self, answer, *, delay_s=0.0):
        self.answer = answer
        self.delay_s = delay_s
        self.asked = []
        self.available = True

    async def classify(self, text, allowed):
        self.asked.append(text)
        await asyncio.sleep(self.delay_s)
        return self.answer


def service(rig, muse, transcripts, *, model=None, voice=None, stream_replies=False, **fields):
    rig.session = PinSession(rig.recording, muse)

    def backends(session):
        bundle = backends_for(Mode.ON_ROBOT, session, speech=voice or rig.voice(), progress_speech=rig.voice("progress_tts"),
                              transcriber=transcripts, stream_replies=stream_replies, wake="wake_detector" in fields)
        return bundle if model is None else dataclasses.replace(bundle, commands=Commands(model))

    async def prepare_chat(session, vm):
        return "robot-chat"
    return reachy_muse.conversation.ReachyService(
        identity=Identity("02:00:00:ab:cd:ef"), executor=rig.hardware, silence_s=.2, speech_gate=Vad(),
        backends=backends, prepare_chat=prepare_chat, owns_chat=True, **fields)


def run(rig, service, user, *, rest=("idle",)):
    async def converse():
        stop = asyncio.Event()
        camera = asyncio.create_task(service.morning.watch_camera()) if service.morning is not None else None
        session = asyncio.create_task(service.serve(rig.session, {"vm_id": "vm-1"}, stop))
        try:
            await rig.quiet()
            await user()
            await rig.quiet(*rest)
        finally:
            stop.set()
            await asyncio.wait_for(session, 5)
            await asyncio.gather(*rig.session.scripts)
            if camera is not None:
                camera.cancel()
                await asyncio.gather(camera, return_exceptions=True)
    asyncio.run(asyncio.wait_for(converse(), 20))
    return rig.recording.events


def sent_to_muse(events):
    return [event["text"] for event in events if event.get("muse") == "send_chat"]


def played(events):
    return [event["audio"] for event in events if event.get("robot") == "play"]


def rig_with(monkeypatch, chunk_s=0.0):
    rig = Rig(monkeypatch)
    rig.hardware = SpeakerHardware(rig.recording, chunk_s)
    return rig


async def never_asked(session, request):
    raise AssertionError(f"Muse received request {request}")


def test_hush_discards_a_briefing_waiting_for_speaker_ownership():
    from reachy_muse.conversation import _SpeechJob
    from reachy_muse.turn_timeline import TurnTimeline
    from test_conversation import FakeSession, cancel_task
    from test_turn_pacing import Labels, PacedHardware, conversation_for

    async def scenario():
        labels = Labels()
        hardware = PacedHardware(labels)
        conversation = conversation_for(labels, hardware, "")(FakeSession())
        await conversation._output_queue.put(_SpeechJob("morning-briefing", "Stale briefing."))
        async with conversation._speaker_lock:
            speaker = asyncio.create_task(conversation._play_output())
            while not conversation._speech_pending:
                await asyncio.sleep(0)
            await conversation._obey("Stop talking.", time.monotonic(), TurnTimeline())
        try:
            await conversation._output_queue.join()
            assert len(hardware.played) == 0
        finally:
            await cancel_task(speaker)

    asyncio.run(asyncio.wait_for(scenario(), 2))


def test_go_to_sleep_never_reaches_muse_and_puts_reachy_to_sleep_until_the_wake_phrase(monkeypatch):
    rig = rig_with(monkeypatch)
    transcripts = Transcripts("Hey Muse, go to sleep.", "What's the weather tomorrow?")
    conversation = service(rig, never_asked, transcripts, wake_detector=Wake(), wake_timeout_s=30)

    async def user():
        rig.say(wake=True)
        await rig.quiet("sleeping")
        rig.say()

    events = run(rig, conversation, user, rest=("sleeping",))
    assert sent_to_muse(events) == []
    assert played(events) == ["tts:Good night."]
    assert transcripts.texts == ["What's the weather tomorrow?"]
    assert rig.states()[-1] == "sleeping"


def test_a_paraphrase_the_model_names_turns_the_volume_down_without_muse(monkeypatch):
    rig = rig_with(monkeypatch)
    model = FakeModel(DeviceCommand.VOLUME_DOWN)
    conversation = service(rig, never_asked, Transcripts(QUIET_DOWN), model=model)

    async def user():
        rig.say()

    events = run(rig, conversation, user)
    assert sent_to_muse(events) == []
    assert (model.asked, rig.hardware.volume_steps, played(events)) == ([QUIET_DOWN], [-10], [])
    assert {"robot": "gesture", "name": "nod"} in events


def test_a_model_past_its_deadline_leaves_the_request_to_muse(monkeypatch):
    rig = rig_with(monkeypatch)

    async def muse(session, request):
        await session.answer(request, "Sure, I'll speak more softly. [reachy:nod]")

    conversation = service(rig, muse, Transcripts(QUIET_DOWN), model=FakeModel(DeviceCommand.VOLUME_DOWN, delay_s=2))

    async def user():
        rig.say()

    events = run(rig, conversation, user)
    assert sent_to_muse(events) == [QUIET_DOWN]
    assert rig.hardware.volume_steps == []
    assert played(events)[-1] == "tts:Sure, I'll speak more softly."


def test_a_bare_stop_with_nothing_playing_still_goes_to_muse(monkeypatch):
    rig = rig_with(monkeypatch)

    async def muse(session, request):
        await session.answer(request, "Okay. [reachy:nod]")

    async def user():
        rig.say()

    events = run(rig, service(rig, muse, Transcripts("Stop.")), user)
    assert sent_to_muse(events) == ["Stop."]


def test_stop_talking_while_an_earlier_answer_plays_cuts_the_rest_of_it(monkeypatch):
    rig = rig_with(monkeypatch, chunk_s=.1)
    rig.hardware.echo_cancelled_input = True
    lines = ["tts:Robots began as toys.", "tts:Then they built cars.", "tts:Now they fold laundry."]

    async def muse(session, request):
        frames = [sentence_frame(line[4:]) for line in lines]
        await session.answer(request, "".join(frames))

    conversation = service(rig, muse, Transcripts("Tell me about robots.", "Stop talking."),
                           voice=LongVoice(rig.recording, chunks=20), stream_replies=True)

    async def user():
        rig.say()
        await rig.wait_until_or_timeout(lambda: {"robot": "play", "audio": lines[0]} in rig.recording.events, 5)
        rig.say()

    events = run(rig, conversation, user)
    assert sent_to_muse(events) == ["Tell me about robots."]
    assert played(events) == lines[:1]
    assert {"robot": "gesture", "name": "nod"} in events


@pytest.mark.parametrize("reply", ["Hello there.", "Hello there. [reachy:happy]"])
def test_hushing_the_shared_chat_startup_reply_keeps_the_session_open(monkeypatch, reply):
    from musegadget.link_client import Outcome

    rig = rig_with(monkeypatch, chunk_s=.1)
    rig.hardware.echo_cancelled_input = True

    async def muse(session, request):
        await session.answer(request, reply if request == 1 else "Sunny tomorrow. [reachy:happy]")

    conversation = rig.service(muse, owned=False, speech=LongVoice(rig.recording, chunks=20),
                               transcriber=Transcripts("Stop talking.", "What's the weather tomorrow?"))

    async def check():
        stop = asyncio.Event()
        session = asyncio.create_task(conversation.serve(rig.session, {"vm_id": "vm-1"}, stop))
        try:
            await rig.wait_until_or_timeout(lambda: played(rig.recording.events) == ["tts:Hello there."], 5)
            assert played(rig.recording.events) == ["tts:Hello there."]
            rig.say()
            await rig.quiet()
            assert rig.states()[-1] == "idle"
            assert not session.done() and not rig.session.subscription_closed
            assert len(rig.hardware.played) < 20
            rig.say()
            await rig.quiet()
            assert played(rig.recording.events) == ["tts:Hello there.", "tts:Sunny tomorrow."]
            assert sent_to_muse(rig.recording.events)[1:] == ["What's the weather tomorrow?"]
        finally:
            stop.set()
            outcome = await asyncio.wait_for(session, 5)
            await asyncio.gather(*rig.session.scripts)
        assert outcome is Outcome.STOPPED

    asyncio.run(asyncio.wait_for(check(), 20))


def test_go_to_sleep_while_muse_works_drops_the_request_and_puts_reachy_to_sleep(monkeypatch):
    rig = rig_with(monkeypatch)
    transcripts = Transcripts("Hey Muse, search for the latest robot news.", "Go to sleep.", "What time is it?")

    async def muse(session, request):
        if request == 1:
            await rig.wait_until_or_timeout(lambda: session.requests == 2, 10)
            await session.answer(request, "Robots learned to fold laundry. [reachy:happy]")
        else:
            await session.answer(request, "Okay, I stopped. [reachy:nod]")

    conversation = service(rig, muse, transcripts, wake_detector=Wake(), wake_timeout_s=30)

    async def user():
        rig.say(wake=True)
        await rig.quiet("thinking")
        rig.say()
        await rig.quiet("sleeping")
        rig.say()

    events = run(rig, conversation, user, rest=("sleeping",))
    assert sent_to_muse(events) == ["search for the latest robot news.", STOP_REQUEST]
    assert "tts:Robots learned to fold laundry." not in played(events)
    assert played(events)[-1] == "tts:Good night."
    assert transcripts.texts == ["What time is it?"]
