"""Run real capture/admission with deterministic voices and in-memory sinks."""

import asyncio
from types import SimpleNamespace

import pytest
import numpy as np

from test_conversation import bounded
from test_wake_handoff import handoff
from test_voice_verify import Vad, verifier


@pytest.fixture(autouse=True)
def deterministic_vad(monkeypatch):
    monkeypatch.setattr("webrtcvad.Vad", lambda mode: Vad())


def test_strict_rejects_wake_before_hearing_or_call_and_then_accepts_operator(handoff):
    async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
        openings = []
        conversation._open_call = lambda: openings.append("open")
        await feed((.5, 20), (.25, 14))
        assert (conversation._wake_deadline, len(recordings), openings) == (None, 0, [])
        await conversation._output_queue.join()
        while conversation._has_output():
            await asyncio.sleep(.01)
        assert sum(len(audio) for audio in hardware.played) == 1280
        await feed((.25, 14), (.125, 25), (0, 16))
        await accepted.wait()
        assert openings == ["open"]
        assert session.setup_messages[0][0] == "Find flights from Boston to Paris."
        assert set(recordings[0]) == {0, 4096}

    asyncio.run(bounded(handoff(run, verifier=verifier())))


def test_impostor_followup_stays_local_after_verified_wake(handoff):
    async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
        await feed((.25, 14), (0, 16))
        assert conversation._wake_deadline is not None
        await feed((.5, 25), (0, 16))
        assert recordings == []
        await conversation._output_queue.join()
        while conversation._has_output():
            await asyncio.sleep(.01)
        await feed((.125, 25), (0, 16))
        await accepted.wait()
        assert len(recordings) == 1
        assert set(recordings[0]) == {0, 4096}
        assert session.setup_messages[0][0] == "Find flights from Boston to Paris."

    asyncio.run(bounded(handoff(run, verifier=verifier())))


@pytest.mark.parametrize("realtime,expected", [(False, True), (True, False)])
def test_soft_policy_uses_configured_realtime_even_before_call_exists(handoff, realtime, expected):
    async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
        if realtime:
            conversation.realtime = SimpleNamespace(usable=True)
        await feed((.5, 20), (.25, 14), (.5, 25), (0, 16))
        if expected:
            await accepted.wait()
            assert set(recordings[0]) == {0, 16384}
            assert session.setup_messages[0][0] == "Find flights from Boston to Paris."
        else:
            assert (conversation._wake_deadline, recordings, session.setup_messages) == (None, [], [])
        assert hardware.played == []

    asyncio.run(bounded(handoff(run, verifier=verifier("soft", realtime))))


def test_soft_chat_fallback_observes_instead_of_gating(handoff):
    async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
        conversation.realtime = SimpleNamespace(usable=True, chat_turn=lambda: None)
        await feed((.5, 20), (.25, 14), (.5, 25), (0, 16))
        assert (conversation._wake_deadline, recordings) == (None, [])
        conversation.realtime.usable = False
        await feed((.5, 20), (.25, 14), (.5, 25), (0, 16))
        await accepted.wait()
        assert set(recordings[0]) == {0, 16384}
        assert session.setup_messages[0][0] == "Find flights from Boston to Paris."
    asyncio.run(bounded(handoff(run, verifier=verifier("soft", True))))


def test_capture_gap_discards_operator_segment_before_impostor(handoff):
    async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
        await feed((.25, 14), (.125, 10))
        await feed((.5, 25), (0, 16), gap=True)
        assert recordings == []
        await conversation._output_queue.join()
        while conversation._has_output():
            await asyncio.sleep(.01)
        await feed((.125, 25), (0, 16))
        await accepted.wait()
        assert set(recordings[0]) == {0, 4096}

    asyncio.run(bounded(handoff(run, verifier=verifier())))


def test_wake_embedding_uses_recent_audio_not_older_operator_speech(handoff):
    subject = verifier()
    subject.encoder.embed = lambda audio, rate: np.array([1., 0.] if audio[0] < .4 else [0., 1.])
    async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
        await feed((.125, 50))
        await feed((.5, 90))
        await feed((.25, 14))
        assert (conversation._wake_deadline, recordings) == (None, [])
        await feed((.25, 14), (.125, 25), (0, 16))
        await accepted.wait()
        assert set(recordings[0]) == {0, 4096}
    asyncio.run(bounded(handoff(run, verifier=subject)))


@pytest.mark.parametrize("mode", ["strict", "soft"])
def test_call_failure_before_verified_release_preserves_chat_request(handoff, mode):
    from reachy_muse.companion_client import CompanionHearing, CompanionLink, LinkState, _Connection
    from reachy_muse.protocol import CloseHow, HearClose, HearEndpoint, Op, encode
    from reachy_muse.realtime_call import CallInput

    async def scenario():
        link = CompanionLink("ws://127.0.0.1:9000", token="test", out_rate=16000, insecure=True)
        link._connection = connection = _Connection(None, (Op.HEAR,))
        link._set(LinkState.UP)
        sent = connection.send
        closes = []
        companion_recordings = []

        def send(frame):
            sent(frame)
            if isinstance(frame, HearClose):
                closes.append(frame.how)
                if frame.how is CloseHow.FINISH:
                    companion_recordings.append(np.frombuffer(connection.hearing[frame.stream].audio, dtype="<i2").copy())
                    link._receive(connection, encode(HearEndpoint(
                        frame.stream, "Find flights from Boston to Paris.", True)))

        connection.send = send

        async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
            async def close():
                pass

            async def failed():
                conversation.realtime.usable = False

            call = SimpleNamespace(done=False, failed=True, audio_sent=False, awaiting_answer=False,
                                   audible=False, close=close, feed=SimpleNamespace(push=lambda samples: None))
            call.input = CallInput(call.feed, hardware.sample_rate)
            conversation.realtime = SimpleNamespace(usable=True, call_failed=failed, chat_turn=lambda: None)
            conversation._open_call = lambda: setattr(conversation, "call", call)
            await feed((.25, 14), (.125, 15))
            assert connection.hearing == {}
            call.done = True
            await feed((0, 1))
            await feed((0, 16))
            for _ in range(20):
                if session.setup_messages:
                    break
                await asyncio.sleep(.01)
            assert [message[0] for message in session.setup_messages] == ["Find flights from Boston to Paris."]
            assert closes == [CloseHow.FINISH]
            assert np.count_nonzero(companion_recordings[0] == 4096) == 4800

        await handoff(run, verifier=verifier(mode), hearing=CompanionHearing(link))

    asyncio.run(bounded(scenario()))


def test_single_turn_wake_cannot_dispatch_audio_from_previous_window(handoff):
    from reachy_muse.conversation import TurnOutcome

    async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
        dispatched = []

        async def turn(audio, **options):
            dispatched.append(options["recognized_text"])
            return TurnOutcome.ACCEPTED

        conversation.turn = turn
        conversation.wake_timeout_s = 0
        await feed((.25, 14), (.125, 25), (0, 16), (.1875, 25), (0, 16))
        await feed((0, 1))
        assert dispatched == ["Find flights from Boston to Paris."]
        await feed((.25, 14), (0, 16))
        await feed((0, 1))
        assert dispatched == ["Find flights from Boston to Paris."]
        assert len(recordings) == 1
        assert set(recordings[0]) == {0, 4096}

    asyncio.run(bounded(handoff(run, verifier=verifier())))


def test_rejection_waits_for_speaker_without_blocking_capture_and_accounts_for_echo(handoff):
    import time

    from reachy_muse.speech_playback import PlaybackCursor
    from reachy_muse.voice_verify import rejection_cue

    async def run(conversation, session, hardware, wake, recognition, recordings, accepted, feed):
        blocked = []
        play = hardware.play_audio

        def play_audio(samples):
            blocked.append(conversation._input_blocked())
            play(samples)

        hardware.play_audio = play_audio
        cursor = PlaybackCursor()
        async with conversation._own_speaker(cursor):
            await conversation._voice_rejected()
            assert hardware.played == []
            assert cursor.end == 0
            await feed((.25, 14), (0, 20))
            assert conversation._wake_deadline is not None
        started = time.monotonic()
        await conversation._output_queue.join()
        assert np.concatenate(hardware.played).tolist() == rejection_cue(8000).tolist()
        assert blocked and all(blocked)
        assert conversation._speaker_cursor.end >= started + .16
        while conversation._has_output():
            await asyncio.sleep(.01)
        assert not conversation._input_blocked()

    asyncio.run(bounded(handoff(run, verifier=verifier())))


def test_cancelling_rejection_playback_joins_write_before_clearing_audio():
    import threading

    from reachy_muse.capabilities import Mode
    from reachy_muse.conversation import VoiceConversation
    from reachy_muse.local_backends import backends_for
    from test_conversation import FakeHardware, FakeSession, cancel_task

    async def scenario():
        hardware, session = FakeHardware(), FakeSession()
        conversation = VoiceConversation(session, hardware, voice_verifier=verifier(),
                                         backends=backends_for(Mode.ON_ROBOT, session, speech=object()))
        entered, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        writes = []

        def play_audio(samples):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(2)
            writes.append("write")

        hardware.play_audio = play_audio
        hardware.clear_audio = lambda: writes.append("clear")
        speaker = asyncio.create_task(conversation._play_output())
        try:
            await conversation._voice_rejected()
            await entered.wait()
            speaker.cancel()
            await asyncio.sleep(0)
            assert writes == []
            release.set()
            await cancel_task(speaker)
            assert writes == ["write", "clear"]
            assert conversation._speaker_cursor.end == 0
            assert not conversation._muted
        finally:
            release.set()
            if not speaker.done():
                await cancel_task(speaker)

    asyncio.run(bounded(scenario()))
