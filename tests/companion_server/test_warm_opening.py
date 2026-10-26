import asyncio

from reachy_muse.companion_server.engines import EchoNarrator, SpeechFirst, ToneVoice
from reachy_muse.companion_server.server import Engines, _Session
from reachy_muse.protocol import Narrate, NarrateOp, NarrateResult, NarratedLine, Sender, SpeakStart, VoiceRole, decode


class CountingVoice(ToneVoice):
    def __init__(self):
        super().__init__(chunks=2)
        self.calls = []

    async def speak(self, text, role, out_rate, expression=None):
        self.calls.append((text, role, out_rate))
        async for chunk in super().speak(text, role, out_rate):
            yield chunk


class Socket:
    def __init__(self):
        self.frames = []

    async def send(self, frame):
        self.frames.append(frame)


class Server:
    token = "test"

    def __init__(self, voice):
        self.engines = Engines(voice=voice, narrator=EchoNarrator(), speech_first=SpeechFirst())


def test_speak_start_drains_the_just_narrated_audio_from_one_producer():
    async def scenario():
        voice, socket = CountingVoice(), Socket()
        session = _Session(Server(voice), socket)
        session.out_rate = 16000
        await session._narration(Narrate(1, NarrateOp.ACKNOWLEDGE, "Hmm, good question."))
        assert decode(socket.frames[0], sender=Sender.COMPANION) == NarrateResult(1, (
            NarratedLine("Hmm, good question.", None),))
        await session._speak_start(SpeakStart(2, "Hmm, good question.", VoiceRole.MAIN))
        await asyncio.gather(*(speech.task for speech in session.speaking.values()))
        assert voice.calls == [("Hmm, good question.", VoiceRole.MAIN, 16000)]
        assert session.warm == {}
        assert len([frame for frame in socket.frames if isinstance(frame, bytes)]) == 2
        await session._stop_everything()
    asyncio.run(scenario())


def test_warm_entries_are_bounded_expire_and_close_cancels_producers(monkeypatch):
    import reachy_muse.companion_server.server as server
    monkeypatch.setattr(server, "WARM_TTL_S", .01)

    class WaitingVoice:
        name = "waiting"

        async def speak(self, text, role, out_rate, expression=None):
            await asyncio.Event().wait()
            yield b""

    async def scenario():
        session = _Session(Server(WaitingVoice()), Socket())
        session.out_rate = 16000
        for text in ("First.", "Second.", "Third."):
            session._warm_voice(text)
        assert tuple(session.warm) == ("Second.", "Third.")
        await asyncio.sleep(.03)
        assert session.warm == {}
        session._warm_voice("Fourth.")
        await asyncio.sleep(0)
        tasks = tuple(session.warming)
        await session._stop_everything()
        assert session.warm == {}
        assert all(task.done() for task in tasks)
    asyncio.run(scenario())
