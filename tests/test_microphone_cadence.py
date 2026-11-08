# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""SDK draining must continue even when the default executor or event loop is busy."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace

import numpy as np

from reachy_muse.conversation import VoiceConversation, _CaptureBuffer


def test_capture_does_not_wait_for_default_executor_or_event_loop():
    async def scenario():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        release, occupied = threading.Event(), threading.Event()
        reads = []

        def occupy():
            occupied.set()
            release.wait(3)

        def read():
            time.sleep(.01)
            reads.append(threading.current_thread().name)
            return np.zeros(160, np.float32)

        hardware = SimpleNamespace(read_audio=read, sample_rate=16000)
        conversation = SimpleNamespace(hardware=hardware, _input_blocked=lambda: False, _wake_policy=None)
        capture = _CaptureBuffer(16000)
        blocker = loop.run_in_executor(None, occupy)
        task = asyncio.create_task(VoiceConversation._capture_microphone(conversation, capture))
        try:
            await asyncio.sleep(.05)
            assert occupied.is_set()
            before = len(reads)
            time.sleep(.15)
            assert len(reads) >= before + 10
            assert set(reads) == {"reachy-microphone_0"}
            chunk = await capture.get()
            assert chunk is not None
            assert chunk.gap is False
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, blocker, return_exceptions=True)
    asyncio.run(scenario())


def test_event_loop_stall_has_bounded_handoff_and_explicit_audio_loss():
    async def scenario():
        def read():
            time.sleep(.005)
            return np.zeros(320, np.float32)

        hardware = SimpleNamespace(read_audio=read, sample_rate=16000)
        conversation = SimpleNamespace(hardware=hardware, _input_blocked=lambda: False, _wake_policy=None)
        capture = _CaptureBuffer(16000, capacity_s=.1)
        task = asyncio.create_task(VoiceConversation._capture_microphone(conversation, capture))
        try:
            await asyncio.sleep(.02)
            time.sleep(.1)
            await asyncio.sleep(.001)
            chunk = await capture.get()
            assert chunk.gap is True
            assert chunk.gap_reason in {"capture handoff overflow", "capture buffer overflow"}
            assert len(chunk.samples) <= 1600
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(scenario())
