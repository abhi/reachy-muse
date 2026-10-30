"""Exercise process death, including an engine whose asyncio waiter was cancelled. No models."""

import asyncio
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from reachy_muse.companion_server.watchdog import EngineExecutor, Watchdog


@pytest.mark.parametrize("blocked, reason", [
    ("time.sleep(10)", "event loop"),
    ("""
worker = EngineExecutor('stuck-native-call', monitor=monitor)
entered = threading.Event()
def stuck():
    entered.set()
    time.sleep(10)
future = asyncio.get_running_loop().run_in_executor(worker, stuck)
await asyncio.to_thread(entered.wait)
future.cancel()
await asyncio.sleep(10)
""", "stuck-native-call"),
])
def test_stall_exits_nonzero_even_after_asyncio_cancels_the_native_waiter(blocked, reason):
    script = """
import asyncio, threading, time
from reachy_muse.companion_server.watchdog import EngineExecutor, Watchdog
async def main():
    monitor = Watchdog(loop_s=.4, engine_s=.1, check_s=.01)
    monitor.start()
""" + "\n".join("    " + line for line in blocked.splitlines()) + "\nasyncio.run(main())\n"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=5)
    assert result.returncode == 70, result.stderr
    assert f"companion watchdog: {reason} made no progress" in result.stderr
    assert "exiting 70 for restart" in result.stderr


def test_model_load_idle_time_and_a_long_progressing_briefing_do_not_restart():
    exits = []
    monitor = Watchdog(loop_s=1, engine_s=.12, check_s=.01, exit_process=exits.append)

    async def scenario():
        with EngineExecutor("narrator", monitor=monitor) as engine:
            # Loading is unbounded. The CLI arms the watchdog after all engine.load calls.
            await asyncio.wrap_future(engine.submit(time.sleep, .25))
            monitor.start()
            try:
                await asyncio.sleep(.25)  # idle is healthy

                def briefing():
                    for _ in range(30):
                        time.sleep(.01)
                        monitor.progress()
                    return "The whole briefing."

                long = engine.submit(briefing)
                queued = engine.submit(lambda: "Next line.")
                assert await asyncio.wrap_future(long) == "The whole briefing."
                assert await asyncio.wrap_future(queued) == "Next line."
                await asyncio.sleep(.2)  # completed calls must be forgotten
            finally:
                monitor.stop()
        assert exits == []

    asyncio.run(scenario())


def test_chatterbox_timeout_cannot_hide_its_stuck_worker():
    script = """
import asyncio, threading, time
from reachy_muse.companion_server.voice_chatterbox import ChatterboxVoice
from reachy_muse.companion_server.watchdog import watchdog
from reachy_muse.protocol import VoiceRole
entered = threading.Event()
class Model:
    sample_rate = 24000
    def tokens(self, text):
        entered.set()
        time.sleep(10)
        yield [], True
async def main():
    voice = ChatterboxVoice()
    voice._model = Model()
    watchdog.engine_s, watchdog.loop_s, watchdog.check_s = .15, 2, .01
    watchdog.start()
    speech = voice.speak('A long briefing.', VoiceRole.MAIN, 24000)
    pending = asyncio.create_task(speech.__anext__())
    await asyncio.to_thread(entered.wait)
    pending.cancel()
    await asyncio.gather(pending, return_exceptions=True)
    await speech.aclose()
    await asyncio.sleep(10)
asyncio.run(main())
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=5)
    assert result.returncode == 70, result.stderr
    assert "voice-chatterbox made no progress" in result.stderr


def test_narrator_tokens_keep_a_long_briefing_alive(monkeypatch):
    from reachy_muse.companion_server import narrator_mlx

    exits = []
    monitor = Watchdog(loop_s=1, engine_s=.12, check_s=.01, exit_process=exits.append)
    monkeypatch.setattr(narrator_mlx, "watchdog", monitor)

    def tokens(*args, **kwargs):
        for character in "Here is the whole morning briefing.":
            time.sleep(.01)
            yield SimpleNamespace(text=character)

    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(stream_generate=tokens))
    narrator = narrator_mlx.MlxNarrator()
    narrator._thread.shutdown()
    narrator._thread = EngineExecutor("narrator", monitor=monitor)
    narrator._tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "prompt",
                                          encode=lambda *args, **kwargs: [1])

    async def scenario():
        monitor.start()
        try:
            assert await asyncio.get_running_loop().run_in_executor(
                narrator._thread, narrator._generate, [], 100, "") == "Here is the whole morning briefing."
            assert exits == []
        finally:
            monitor.stop()
            narrator._thread.shutdown()

    asyncio.run(scenario())
