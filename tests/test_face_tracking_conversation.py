# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Both conversation transports reach the hardware states that enable local gaze."""

import asyncio

import pytest

from reachy_muse.hardware import ReachyController
from reachy_muse.motion import MotionStyle
from reachy_muse.motion.layers import MotionLayers
from reachy_muse.realtime_call import realtime_supported
from test_device_command_turns import run, service
from test_face_tracking_motion import Follower
from test_hardware import FakeRobot, np, pose_factory
from test_realtime_turns import (
    ScriptedCalls, StreamingTranscripts, reply_events, rig_for, realtime_service, voiced,
)


@pytest.mark.parametrize('path', ['chat', 'realtime'])
def test_conversation_enables_gaze_while_listening_thinking_and_speaking(monkeypatch, path):
    if path == 'realtime' and not realtime_supported():
        pytest.skip('the Muse SDK has no voice call events')
    rig = rig_for(monkeypatch)
    controller = ReachyController(pose_factory=pose_factory)
    controller._np, controller._robot = np, FakeRobot()
    controller._layers = MotionLayers(MotionStyle())
    controller._output_sample_rate = rig.hardware.output_sample_rate
    controller._start_music()
    controller._require_running = lambda: None
    controller._face_follower = follower = Follower()
    states = []
    set_state = rig.hardware.set_state
    play_audio = rig.hardware.play_audio

    def observe():
        controller._update_face_tracking(controller._state)
        states.append((controller._state, follower.active))

    def state_changed(state, **kwargs):
        set_state(state, **kwargs)
        controller.set_state(state, **kwargs)
        observe()

    def audio_played(samples, **kwargs):
        play_audio(samples, **kwargs)
        controller.play_audio(samples, **kwargs)
        observe()

    rig.hardware.set_state = state_changed
    rig.hardware.play_audio = audio_played
    rig.hardware.finish_sentence = controller.finish_sentence
    transcripts = StreamingTranscripts('Hey Muse, tell me about trains.')

    async def realtime_answer(sdk, emit, script):
        emit(sdk.Working(True))
        await asyncio.sleep(.6)
        reply_events(sdk, emit, 'r1', 'Here are some facts about trains.', voiced(.8))
        emit(sdk.Working(False))

    async def chat_answer(session, request):
        await asyncio.sleep(.6)
        await session.answer(request, 'Here are some facts about trains.')

    if path == 'realtime':
        conversation = realtime_service(rig, transcripts, ScriptedCalls(answer=realtime_answer))
    else:
        from test_pins import Wake
        conversation = service(rig, chat_answer, transcripts, wake_detector=Wake(), wake_timeout_s=2)

    async def user():
        rig.say(wake=True)
        await rig.wait_until_or_timeout(lambda: ('speaking', True) in states, 5)

    run(rig, conversation, user)
    for state in ('listening', 'thinking', 'speaking'):
        assert (state, True) in states
        assert (state, False) not in states
    assert states[-1] == ('idle', False)
