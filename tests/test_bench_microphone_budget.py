"""Both benchmark readers must share the same audio-driven vision budget."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from bench import microphone_budget as bench


@pytest.mark.parametrize("revision", [None, "old"])
def test_benchmark_observes_capture_and_wake_with_an_active_budget(monkeypatch, capsys, revision):
    budgets = []

    class Budget(bench.VisionBudget):
        def __init__(self):
            super().__init__()
            self.states = []
            self.observations = []
            budgets.append(self)

        def capture_active(self, active):
            self.states.append(active)
            super().capture_active(active)

        def observe(self, source, audio_s, **options):
            self.observations.append((source, audio_s, self._capturing))
            super().observe(source, audio_s, **options)

    class Worker:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def set_active(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(bench, "VisionBudget", Budget)
    monkeypatch.setattr(bench, "FaceFollower", Worker)
    monkeypatch.setattr(bench, "WaveWatcher", Worker)
    monkeypatch.setattr(bench.subprocess, "check_output", lambda *args, **kwargs: """
class VoiceConversation:
    async def _capture_microphone(self, capture):
        while True:
            read = asyncio.create_task(asyncio.to_thread(self.hardware.read_audio))
            try:
                sample = await asyncio.shield(read)
            except asyncio.CancelledError:
                await read
                raise
            capture.push(sample, captured_at=time.monotonic())
""")
    args = SimpleNamespace(reader_revision=revision, seconds=1, executor_load=False, native_models=False)
    asyncio.run(bench.measure(args))
    result = json.loads(capsys.readouterr().out)
    assert result["reader"] == (revision or "working tree")
    budget, = budgets
    assert budget.states == [True, False]
    assert {source for source, _, _ in budget.observations} == {"capture", "wake"}
    assert all(active for _, _, active in budget.observations)
    capture_s = sum(duration for source, duration, _ in budget.observations if source == "capture")
    wake_s = sum(duration for source, duration, _ in budget.observations if source == "wake")
    assert capture_s == pytest.approx(wake_s, abs=.1)
