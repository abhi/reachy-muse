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
        assert [len(audio) for audio in hardware.played] == [1280]
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
