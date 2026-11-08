# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Offline capture cadence with real vision workers and synthetic camera/audio.

No devices or network. --reader-revision compares the old capture loop from git.
--executor-load exercises shared-executor starvation separately from vision CPU.
"""

import argparse
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import threading
import time
from types import SimpleNamespace

import numpy as np

from bench.wave import BOX, clip
from reachy_muse.conversation import VoiceConversation, _CaptureBuffer
from reachy_muse.local_face_tracking import FaceFollower
from reachy_muse.vision_budget import VisionBudget
from reachy_muse.wave_wake import WaveWatcher


def compute(seconds, work=None):
    # Python work exercises GIL competition; native inference releases it.
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        work() if work is not None else sum(i * i for i in range(200))


async def measure(args):
    budget = VisionBudget()
    frames = [frame for _, frame, _ in clip("wave", 1001, seconds=3)]
    counts = dict(camera=0, face=0, person=0)
    camera_lock = threading.Lock()
    started = time.monotonic()

    def camera():
        with camera_lock:
            counts["camera"] += 1
            time.sleep(.02)
            return frames[int((time.monotonic() - started) * 8) % len(frames)].copy()

    face_model = person_model = None
    if args.native_models:
        from reachy_muse.local_face_tracking import _YuNetDetector, default_face_model
        from reachy_muse.arrival import YoloxPersonDetector, default_person_model
        matrix = np.array([[160., 0, 80], [0, 160., 60], [0, 0, 1]])
        face_model = _YuNetDetector(default_face_model(), matrix, np.zeros(5))
        person_model = YoloxPersonDetector(default_person_model())

    class Face:
        def detect(self, frame):
            counts["face"] += 1
            compute(.05, (lambda: face_model.detect(frame)) if face_model else None)
            return []

    class Person:
        def detect(self, frame):
            counts["person"] += 1
            compute(.32, (lambda: person_model.detect(frame)) if person_model else None)
            return [BOX]

    face = FaceFollower(camera, read_head_angles=lambda: (0, 0), camera_matrix=None,
                        detector_factory=lambda *args: Face(), budget=budget)
    wave = WaveWatcher(camera, lambda: True, detector_factory=lambda _: Person(), budget=budget)
    face.start()
    face.set_active(True, idle=True)
    wave.start()
    wave.set_active(True)
    read_at = []
    samples = np.zeros(320, np.float32)
    due = time.monotonic() + .02

    def microphone():
        nonlocal due
        time.sleep(max(0, due - time.monotonic()))
        now = time.monotonic()
        # A one-chunk device queue drops input when the reader cannot keep up.
        due = max(due + .02, now)
        read_at.append(now)
        budget.observe("capture", len(samples) / 16000, now=now)
        return samples

    hardware = SimpleNamespace(read_audio=microphone, sample_rate=16000)
    conversation = SimpleNamespace(hardware=hardware, _input_blocked=lambda: False, _wake_policy=None,
                                   _turn_active=False, _has_output=lambda: False, _hearing_open=lambda: False)
    reader = VoiceConversation._capture_microphone
    if args.reader_revision:
        source = subprocess.check_output(["git", "show", f"{args.reader_revision}:src/reachy_muse/conversation.py"], text=True)
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "VoiceConversation")
        method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_capture_microphone")
        namespace = dict(asyncio=asyncio, time=time, _CaptureBuffer=_CaptureBuffer)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "old_capture", "exec"), namespace)
        reader = namespace["_capture_microphone"]
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1 if args.executor_load else 4))
    capture = _CaptureBuffer(16000)
    budget.capture_active(True)
    producer = asyncio.create_task(reader(conversation, capture))

    async def load():
        while True:
            await asyncio.to_thread(time.sleep, .06)

    loader = asyncio.create_task(load()) if args.executor_load else None
    start = time.monotonic()
    received, gaps = 0, 0
    try:
        while time.monotonic() - start < args.seconds:
            chunk = await capture.get()
            if chunk is not None:
                received += len(chunk.samples)
                gaps += chunk.gap
                budget.observe("wake", len(chunk.samples) / 16000)
        elapsed = time.monotonic() - start
    finally:
        producer.cancel()
        if loader is not None:
            loader.cancel()
        await asyncio.gather(producer, *([loader] if loader else []), return_exceptions=True)
        face.close()
        wave.close()
        budget.capture_active(False)
    intervals = [b - a for a, b in zip(read_at, read_at[1:])]
    result = dict(reader=args.reader_revision or "working tree", seconds=round(elapsed, 3),
                  coverage=round(received / 16000 / elapsed, 4), gaps=gaps,
                  cadence_p99_ms=round(sorted(intervals)[int(.99 * (len(intervals) - 1))] * 1000, 2),
                  cadence_max_ms=round(max(intervals) * 1000, 2), **counts)
    print(json.dumps(result, sort_keys=True))
    if not args.reader_revision:
        assert received / 16000 / elapsed >= .97, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--reader-revision")
    parser.add_argument("--executor-load", action="store_true")
    parser.add_argument("--native-models", action="store_true")
    asyncio.run(measure(parser.parse_args()))
