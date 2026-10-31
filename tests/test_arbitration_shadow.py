import asyncio
import logging

import pytest

from reachy_muse.arbitration import Snapshot
from reachy_muse.arbitration_shadow import observe
from reachy_muse.cli import parser
from reachy_muse.contract.intent import Intent
from reachy_muse.conversation import TurnOutcome
from test_conversation import FakeHardware, FakeSession
from test_local_chat import Companion, Speech, robot


def test_shadow_flag_defaults_off():
    assert parser().parse_args(["run"]).arbitration_shadow is False
    assert parser().parse_args(["run", "--arbitration-shadow"]).arbitration_shadow is True


def test_observation_returns_no_decision_to_execute_and_logs_only_codes(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    times = iter((1.0, 1.002))
    monkeypatch.setattr("reachy_muse.arbitration_shadow.time.perf_counter", lambda: next(times))
    assert observe(Snapshot("opaque", "private sentinel", active=None)) is None
    assert [record.getMessage() for record in caplog.records] == [
        "Reachy arbitration shadow route=MUSE evidence=unresolved rules_ms=2.000"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("session_id", [None, "existing-chat"])
def test_shadow_social_decision_does_not_replace_the_legacy_muse_send(enabled, session_id, caplog):
    caplog.set_level(logging.INFO)

    async def scenario():
        session, hardware, speech = FakeSession(), FakeHardware(), Speech()
        companion = Companion(Intent.QUESTION)
        conversation = robot(session, hardware, speech, companion)
        conversation.arbitration_shadow = enabled
        conversation.session_id = session_id
        result = await conversation.turn(b"voice", recognized_text="Good morning.")
        assert result is TurnOutcome.ACCEPTED
        assert session.setup_messages == [("Good morning.", session_id)]
        assert [heard.text for heard in companion.heard] == ["Good morning."]
        assert speech.spoken == []
        assert hardware.played == []

    asyncio.run(scenario())
    observations = [r.getMessage() for r in caplog.records if "arbitration shadow" in r.getMessage()]
    assert len(observations) == int(enabled)
    if enabled:
        assert observations[0].startswith("Reachy arbitration shadow route=LOCAL_REPLY evidence=whole_social rules_ms=")
    assert "Good morning" not in caplog.text


def test_invalid_shadow_snapshot_cannot_interrupt_existing_routing(monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    def fail(_):
        raise ValueError("private model sentinel")

    monkeypatch.setattr("reachy_muse.arbitration_shadow.observe", fail)

    async def scenario():
        session, hardware, speech = FakeSession(), FakeHardware(), Speech()
        conversation = robot(session, hardware, speech, Companion(Intent.QUESTION))
        conversation.arbitration_shadow = True
        await conversation.turn(b"voice", recognized_text="private user sentinel")
        assert session.setup_messages == [("private user sentinel", None)]
        assert hardware.played == []

    asyncio.run(scenario())
    assert "Reachy arbitration shadow status=invalid_snapshot" in caplog.text
    assert "sentinel" not in caplog.text


def test_disabled_shadow_does_not_even_construct_a_snapshot(monkeypatch):
    def forbidden(_):
        pytest.fail("shadow invoked")
    monkeypatch.setattr("reachy_muse.arbitration_shadow.observe", forbidden)

    async def scenario():
        session = FakeSession()
        conversation = robot(session, FakeHardware(), Speech(), Companion(Intent.QUESTION))
        await conversation.turn(b"voice", recognized_text="What's the weather?")
        assert session.setup_messages == [("What's the weather?", None)]

    asyncio.run(scenario())


def test_real_command_executor_with_shadow_keeps_the_same_hardware_and_muse_effects(monkeypatch, caplog):
    from test_device_command_turns import never_asked, run, sent_to_muse, rig_with, service
    from test_pins import Transcripts, Wake

    caplog.set_level(logging.INFO)
    rig = rig_with(monkeypatch)
    conversation = service(rig, never_asked, Transcripts("Hey Muse, go to sleep."),
                           wake_detector=Wake(), wake_timeout_s=30, arbitration_shadow=True)

    async def user():
        rig.say(wake=True)

    events = run(rig, conversation, user, rest=("sleeping",))
    assert sent_to_muse(events) == []
    assert rig.states()[-1] == "sleeping"
    assert [event["audio"] for event in events if event.get("robot") == "play"] == ["tts:Good night."]
    assert "route=LOCAL_DEVICE evidence=whole_device_command" in caplog.text


def test_active_and_held_context_are_bounded_and_mark_legacy_amendments_missing(monkeypatch):
    from reachy_muse.conversation import _HeldRequest
    from reachy_muse.turn_timeline import TurnTimeline
    from test_conversation import owned_job

    captured = []
    monkeypatch.setattr("reachy_muse.arbitration_shadow.observe", captured.append)

    async def scenario():
        conversation = robot(FakeSession(), FakeHardware(), Speech(), Companion(Intent.QUESTION))
        conversation.arbitration_shadow = True
        conversation.job = owned_job()
        conversation.job.request = "Weather " + "é" * 1000
        held = _HeldRequest(b"", "Recipe " + "é" * 1000, 1, TurnTimeline())
        conversation._observe_arbitration("Cancel the recipe.", held)

    asyncio.run(scenario())
    [snapshot] = captured
    assert len(snapshot.active.request.text.encode("utf-8")) == 768
    assert len(snapshot.held.request.text.encode("utf-8")) == 767
    assert snapshot.active.omitted is True
    assert snapshot.held.omitted is True
    assert snapshot.active.id != snapshot.held.id
